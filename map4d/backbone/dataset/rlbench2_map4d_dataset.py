"""RLBench2 dataset wrapper for the standalone Map4D DiT policy."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from helper.keyframe_targets import (
    build_future_keyframe_table,
    canonicalize_quaternion_np,
    gather_map4d_dit_keyframe_targets,
)
from map4d.backbone.common.replay_buffer import ReplayBuffer
from map4d.backbone.dataset.base_dataset import BaseDataset
from map4d.backbone.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


DEFAULT_PUSH_BOX_SIZE_XYZ = (0.192, 0.384, 0.128)


def _to_tensor_tree(value):
    if isinstance(value, dict):
        return {key: _to_tensor_tree(item) for key, item in value.items()}
    array = np.asarray(value)
    if array.dtype.kind in {"b", "i", "u"}:
        return torch.from_numpy(array)
    return torch.from_numpy(array.astype(np.float32, copy=False))


def _slice_with_pad(array: np.ndarray, start: int, horizon: int) -> np.ndarray:
    indices = np.arange(start, start + horizon)
    indices = np.clip(indices, 0, array.shape[0] - 1)
    return array[indices]


def _fit_streaming_normalizer(
    arrays: Iterable[np.ndarray],
    *,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
) -> SingleFieldLinearNormalizer:
    if mode not in {"limits", "gaussian"}:
        raise ValueError(f"Unsupported normalizer mode {mode!r}")

    count = 0
    input_min = None
    input_max = None
    sum_x = None
    sum_x2 = None
    for array in arrays:
        value = np.asarray(array, dtype=np.float64)
        if value.ndim == 0:
            raise ValueError("Cannot fit normalizer from a scalar array")
        value = value.reshape(-1, value.shape[-1])
        if value.shape[0] == 0:
            continue
        chunk_min = value.min(axis=0)
        chunk_max = value.max(axis=0)
        chunk_sum = value.sum(axis=0)
        chunk_sum2 = np.square(value).sum(axis=0)
        if input_min is None:
            input_min = chunk_min
            input_max = chunk_max
            sum_x = chunk_sum
            sum_x2 = chunk_sum2
        else:
            input_min = np.minimum(input_min, chunk_min)
            input_max = np.maximum(input_max, chunk_max)
            sum_x += chunk_sum
            sum_x2 += chunk_sum2
        count += value.shape[0]

    if count == 0 or input_min is None:
        raise ValueError("Cannot fit normalizer from empty arrays")

    input_mean = sum_x / count
    if count > 1:
        variance = (sum_x2 - count * np.square(input_mean)) / (count - 1)
        input_std = np.sqrt(np.maximum(variance, 0.0))
    else:
        input_std = np.zeros_like(input_mean)

    input_min = torch.as_tensor(input_min, dtype=dtype)
    input_max = torch.as_tensor(input_max, dtype=dtype)
    input_mean = torch.as_tensor(input_mean, dtype=dtype)
    input_std = torch.as_tensor(input_std, dtype=dtype)

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range = input_range.clone()
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            if output_max <= 0 or output_min >= 0:
                raise ValueError("fit_offset=False requires output_min < 0 < output_max")
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs = input_abs.clone()
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    else:
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale
        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict={
            "min": input_min,
            "max": input_max,
            "mean": input_mean,
            "std": input_std,
        },
    )


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    return canonicalize_quaternion_np(
        np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)
    )


def _get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def _downsample_mask(mask: np.ndarray, max_n: Optional[int], seed: int = 0) -> np.ndarray:
    train_mask = np.asarray(mask, dtype=bool).copy()
    if max_n is not None and np.sum(train_mask) > max_n:
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        selected = rng.choice(len(curr_train_idxs), size=int(max_n), replace=False)
        train_idxs = curr_train_idxs[selected]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
    return train_mask


class RLBench2Map4DDataset(BaseDataset):
    """Load RLBench2 push-box batches in the Map4D DiT training format.

    The underlying RLBench2 replay buffer provides robot state/action plus
    point-cloud and DINO file references. The Map4D node pose is loaded from a
    compact pose sidecar generated from `task_low_dim_state[:7]`.
    """

    def __init__(
        self,
        data_path: str,
        pcd_path: str,
        lang_emb_path: str,
        dino_path: str,
        stats_filepath: Optional[str] = None,
        point_flow_path: Optional[str] = None,
        pose_path: Optional[str] = None,
        *,
        task_name: str = "rlbench2_push_box",
        horizon_action: int = 16,
        horizon_keyframe: int = 4,
        n_obs_steps: int = 2,
        horizon_continuous: Optional[int] = None,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        start: int = 0,
        end: int = 99,
        pcd_fps: int = 1024,
        skip_ep: Optional[Sequence[int]] = None,
        kp_num: int = 10,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        pcd_type: str = "2views_point_cloud_rps2048",
        prediction_type: str = "keyframe_continuous",
        point_flow_type: str = "rps200",
        add_openess_sampling: bool = False,
        num_objects: int = 1,
        map4d_dim: int = 7,
        size_parameter_dim: int = 3,
        relation_parameter_dim: int = 0,
        robot_state_dim: Optional[int] = None,
        action_type: str = "bimanual_ee_pose",
        keyframe_tcp_dim: int = 7,
        size_xyz: Sequence[float] = DEFAULT_PUSH_BOX_SIZE_XYZ,
        use_rgb: bool = True,
        allow_missing_pose: bool = False,
    ):
        super().__init__()
        self.data_path = str(data_path)
        self.pcd_path = str(pcd_path)
        self.lang_emb_path = str(lang_emb_path)
        self.dino_path = str(dino_path)
        self.stats_filepath = stats_filepath
        self.point_flow_path = point_flow_path
        self.pose_path = pose_path
        self.task_name = task_name
        self.horizon_action = int(horizon_action if horizon_continuous is None else horizon_continuous)
        self.horizon_keyframe = int(horizon_keyframe)
        self.n_obs_steps = int(n_obs_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.pcd_type = str(pcd_type)
        self.prediction_type = str(prediction_type)
        self.point_flow_type = str(point_flow_type)
        self.num_objects = int(num_objects)
        self.map4d_dim = int(map4d_dim)
        self.size_parameter_dim = int(size_parameter_dim)
        self.relation_parameter_dim = int(relation_parameter_dim)
        self.robot_state_dim = None if robot_state_dim is None else int(robot_state_dim)
        self.action_type = str(action_type)
        self.keyframe_tcp_dim = int(keyframe_tcp_dim)
        self.size_xyz = np.asarray(size_xyz, dtype=np.float32).reshape(-1)
        self.use_rgb = bool(use_rgb)
        self.allow_missing_pose = bool(allow_missing_pose)

        if self.num_objects != 1:
            raise ValueError("RLBench2 push-box Map4D currently expects num_objects=1")
        if self.map4d_dim != 7:
            raise ValueError("RLBench2 push-box Map4D currently expects map4d_dim=7")
        if self.size_xyz.shape[0] != self.size_parameter_dim:
            raise ValueError(
                f"size_xyz has dim {self.size_xyz.shape[0]}, expected {self.size_parameter_dim}"
            )
        if self.relation_parameter_dim != 0:
            raise ValueError("RLBench2 push-box has no relation parameters")
        if self.action_type not in {"single_arm_ee_pose", "single_arm_ee_pos", "bimanual_ee_pose"}:
            raise ValueError(
                "action_type must be 'single_arm_ee_pose', 'single_arm_ee_pos', or 'bimanual_ee_pose', "
                f"got {self.action_type!r}"
            )
        if skip_ep is None:
            skip_ep = []
        start = int(start)
        end = int(end)
        pcd_fps = int(pcd_fps)
        kp_num = int(kp_num)
        val_ratio = float(val_ratio)
        if max_train_episodes is not None and str(max_train_episodes).lower() not in {"", "none", "null"}:
            max_train_episodes = int(max_train_episodes)
        else:
            max_train_episodes = None

        keys = ["state", "action", "point_cloud", "lang", "dino_feature"]
        if self.prediction_type == "keyframe_continuous":
            keys.extend(["object_pose", "point_flow", "initial_point_flow"])
        self.replay_buffer = self._load_replay_buffer(
            start=start,
            end=end,
            pcd_fps=pcd_fps,
            skip_ep=skip_ep,
            kp_num=kp_num,
            keys=keys,
        )

        val_mask = _get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = _downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
        self.train_mask = train_mask
        self.trajectories = self._build_trajectories(train_mask)
        self.val_trajectories = self._build_trajectories(~train_mask)
        self.indices = self._build_indices(self.trajectories)
        if self.use_rgb:
            self._validate_real_dino_features()

    def _load_replay_buffer(self, *, start, end, pcd_fps, skip_ep, kp_num, keys):
        if self.prediction_type == "continuous":
            return ReplayBuffer.getData_continuous(
                self.data_path,
                self.pcd_path,
                self.lang_emb_path,
                self.dino_path,
                start=start,
                end=end,
                pcd_fps=pcd_fps,
                skip_ep=skip_ep,
                keys=keys,
            )
        if self.prediction_type == "keyframe":
            return ReplayBuffer.getData_keyframe(
                self.data_path,
                self.pcd_path,
                self.lang_emb_path,
                self.dino_path,
                start=start,
                end=end,
                pcd_fps=pcd_fps,
                skip_ep=skip_ep,
                kp_num=kp_num,
                keys=keys,
            )
        if self.prediction_type == "keyframe_continuous":
            return ReplayBuffer.getData_keyframe_continuous(
                self.data_path,
                self.pcd_path,
                self.lang_emb_path,
                self.dino_path,
                start=start,
                end=end,
                pcd_fps=pcd_fps,
                skip_ep=skip_ep,
                kp_num=kp_num,
                keys=keys,
            )
        raise ValueError(
            f"Invalid prediction_type={self.prediction_type!r}; expected "
            "'continuous', 'keyframe', or 'keyframe_continuous'."
        )

    def _load_pose_sidecar(self) -> Dict[int, np.ndarray]:
        if self.pose_path is None:
            default = Path(self.data_path).parents[2] / "bimanual_push_box_train_poses.npz"
            self.pose_path = str(default)
        if self.pose_path is None or not os.path.exists(self.pose_path):
            raise FileNotFoundError(
                "pose_path is required for RLBench2 Map4D training. Expected an npz "
                "with episode, frame, and pose_xyzw arrays. Missing-pose fallback is not allowed."
            )

        raw = np.load(self.pose_path)
        if not {"episode", "frame"}.issubset(raw.files):
            raise KeyError(f"{self.pose_path} must contain episode and frame arrays")
        if "pose_xyzw" in raw.files:
            pose_xyzw = raw["pose_xyzw"].astype(np.float32)
        elif {"position", "quaternion_xyzw"}.issubset(raw.files):
            pose_xyzw = np.concatenate(
                [raw["position"].astype(np.float32), raw["quaternion_xyzw"].astype(np.float32)],
                axis=-1,
            )
        else:
            raise KeyError(
                f"{self.pose_path} must contain pose_xyzw or position/quaternion_xyzw arrays"
            )

        by_episode: Dict[int, list] = {}
        for episode, frame, pose in zip(raw["episode"], raw["frame"], pose_xyzw):
            by_episode.setdefault(int(episode), []).append((int(frame), pose))

        result = {}
        for episode, rows in by_episode.items():
            rows = sorted(rows, key=lambda item: item[0])
            frames = [frame for frame, _ in rows]
            if frames != list(range(len(frames))):
                raise ValueError(f"{self.pose_path}: episode {episode} has non-contiguous frames")
            pose = np.stack([value for _, value in rows], axis=0).astype(np.float32)
            quat_wxyz = _xyzw_to_wxyz(pose[:, 3:7])
            result[episode] = np.concatenate([pose[:, 0:3], quat_wxyz], axis=-1)
        return result

    def _build_trajectories(self, episode_mask: np.ndarray):
        pose_by_episode = self._load_pose_sidecar()
        trajectories = []
        episode_ends = np.asarray(self.replay_buffer.episode_ends, dtype=np.int64)
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        keyframe_indices = np.asarray(
            self.replay_buffer.meta.get("keyframe_indices", []),
            dtype=np.int64,
        )
        for replay_ep, (start, end) in enumerate(zip(episode_starts, episode_ends)):
            if replay_ep >= len(episode_mask) or not episode_mask[replay_ep]:
                continue
            ep_slice = slice(int(start), int(end))
            point_refs = np.asarray(self.replay_buffer["point_cloud"][ep_slice])
            if point_refs.ndim != 2 or point_refs.shape[-1] < 2:
                raise ValueError(f"Expected point_cloud refs [T,2], got {point_refs.shape}")
            episode_id = int(point_refs[0, 0])
            length = int(end - start)

            if episode_id in pose_by_episode:
                map4d = pose_by_episode[episode_id]
            elif "object_pose" in self.replay_buffer:
                pose = np.asarray(self.replay_buffer["object_pose"][ep_slice], dtype=np.float32)
                quat = pose[:, 3:7]
                # RLBench2 low_dim sidecar is xyzw; older object_pose data may already be wxyz.
                if np.abs(quat[:, 0]).mean() < np.abs(quat[:, -1]).mean():
                    quat = _xyzw_to_wxyz(quat)
                else:
                    quat = canonicalize_quaternion_np(quat)
                map4d = np.concatenate([pose[:, 0:3], quat], axis=-1)
            else:
                raise KeyError(
                    f"No Map4D pose found for RLBench2 episode {episode_id}; "
                    "missing-pose fallback is not allowed"
                )
            if map4d.shape[0] < length:
                raise ValueError(
                    f"Pose episode {episode_id} has {map4d.shape[0]} frames, replay buffer needs {length}"
                )
            map4d = map4d[:length, None, :].astype(np.float32)

            robot_state = np.asarray(self.replay_buffer["state"][ep_slice], dtype=np.float32)
            if self.robot_state_dim is not None:
                robot_state = self._fit_last_dim(robot_state, self.robot_state_dim)
            actions = self._format_actions(np.asarray(self.replay_buffer["action"][ep_slice], dtype=np.float32))

            local_keyframes = keyframe_indices[(keyframe_indices >= start) & (keyframe_indices < end)] - start
            if local_keyframes.size == 0:
                local_keyframes = np.arange(length, dtype=np.int64)
            future_table = build_future_keyframe_table(
                local_keyframes,
                num_frames=length,
                horizon=self.horizon_keyframe,
            )
            tcp_pose = self._tcp_pose_from_robot_state(robot_state)
            keyframe_object, keyframe_tcp_single = gather_map4d_dit_keyframe_targets(
                map4d,
                tcp_pose[:, 0] if tcp_pose.ndim == 3 else tcp_pose,
                future_table,
            )
            if tcp_pose.ndim == 3:
                keyframe_tcp = np.stack(
                    [
                        gather_map4d_dit_keyframe_targets(map4d, tcp_pose[:, arm_idx], future_table)[1]
                        for arm_idx in range(tcp_pose.shape[1])
                    ],
                    axis=1,
                )
            else:
                keyframe_tcp = keyframe_tcp_single

            trajectories.append(
                {
                    "episode_id": episode_id,
                    "robot_state": robot_state.astype(np.float32),
                    "actions": actions.astype(np.float32),
                    "map4d": map4d.astype(np.float32),
                    "size_parameters": self.size_xyz.astype(np.float32),
                    "relation_parameters": np.zeros((self.relation_parameter_dim,), dtype=np.float32),
                    "keyframe_object": keyframe_object.astype(np.float32),
                    "keyframe_tcp": keyframe_tcp.astype(np.float32),
                    "point_refs": point_refs.astype(np.int64),
                    "dino_refs": np.asarray(self.replay_buffer["dino_feature"][ep_slice]).astype(np.int64),
                }
            )
        return trajectories

    @staticmethod
    def _fit_last_dim(array: np.ndarray, dim: int) -> np.ndarray:
        if array.shape[-1] == dim:
            return array
        if array.shape[-1] < dim:
            pad = np.zeros((*array.shape[:-1], dim - array.shape[-1]), dtype=array.dtype)
            return np.concatenate([array, pad], axis=-1)
        return array[..., :dim]

    def _format_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.action_type == "bimanual_ee_pose":
            if actions.shape[-1] < 16:
                raise ValueError(f"bimanual_ee_pose expects action dim >=16, got {actions.shape[-1]}")
            right = actions[:, 0:8].copy()
            left = actions[:, 8:16].copy()
            right[:, 3:7] = _xyzw_to_wxyz(right[:, 3:7])
            left[:, 3:7] = _xyzw_to_wxyz(left[:, 3:7])
            return np.stack([right, left], axis=1).astype(np.float32)
        if actions.shape[-1] >= 8:
            trajectory = actions[:, :7]
            gripper = actions[:, -1:]
        elif actions.shape[-1] == 7:
            trajectory = actions
            gripper = np.zeros((actions.shape[0], 1), dtype=np.float32)
        elif actions.shape[-1] == 4:
            identity = np.broadcast_to(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (actions.shape[0], 4),
            )
            trajectory = np.concatenate([actions[:, :3], identity], axis=-1)
            gripper = actions[:, 3:4]
        else:
            raise ValueError(f"Expected action dim >=4, got {actions.shape[-1]}")
        trajectory[:, 3:7] = canonicalize_quaternion_np(trajectory[:, 3:7])
        return np.concatenate([trajectory, gripper], axis=-1).astype(np.float32)

    def _tcp_pose_from_robot_state(self, robot_state: np.ndarray) -> np.ndarray:
        if self.action_type == "bimanual_ee_pose":
            if robot_state.shape[-1] < 16:
                raise ValueError(f"bimanual_ee_pose expects robot_state dim >=16, got {robot_state.shape[-1]}")
            right = robot_state[:, 0:7].copy()
            left = robot_state[:, 8:15].copy()
            right[:, 3:7] = _xyzw_to_wxyz(right[:, 3:7])
            left[:, 3:7] = _xyzw_to_wxyz(left[:, 3:7])
            return np.stack([right, left], axis=1).astype(np.float32)
        if robot_state.shape[-1] >= 16:
            tcp_pose = robot_state[:, 8:15].copy()
        elif robot_state.shape[-1] >= 7:
            tcp_pose = robot_state[:, :7].copy()
        else:
            tcp_pose = np.zeros((robot_state.shape[0], 7), dtype=np.float32)
            tcp_pose[:, 3] = 1.0
            return tcp_pose
        tcp_pose[:, 3:7] = canonicalize_quaternion_np(tcp_pose[:, 3:7])
        return tcp_pose.astype(np.float32)

    def _build_indices(self, trajectories):
        indices = []
        for traj_idx, traj in enumerate(trajectories):
            action_len = traj["actions"].shape[0]
            for start in range(action_len):
                indices.append((traj_idx, start))
        return indices

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.trajectories = list(self.val_trajectories)
        val_set.val_trajectories = []
        val_set.indices = val_set._build_indices(val_set.trajectories)
        return val_set

    def _load_point_array(self, root: str, ref: np.ndarray) -> np.ndarray:
        episode, step = int(ref[0]), int(ref[1])
        path = os.path.join(root, f"episode{episode}/{self.pcd_type}/step{step:03d}.npy")
        return np.load(path).astype(np.float32)

    def _load_ref_window(self, root: str, refs: np.ndarray, start: int, horizon: int) -> np.ndarray:
        ref_window = _slice_with_pad(refs, start, horizon)
        return np.stack([self._load_point_array(root, ref) for ref in ref_window], axis=0)

    def _validate_real_dino_features(self) -> None:
        if not self.trajectories:
            raise ValueError("RLBench2Map4DDataset has no training trajectories")
        traj = self.trajectories[0]
        if len(traj["point_refs"]) == 0 or len(traj["dino_refs"]) == 0:
            raise ValueError("RLBench2Map4DDataset requires point_cloud and real DINO references")
        point_cloud = self._load_point_array(self.pcd_path, traj["point_refs"][0])
        dino_feature = self._load_point_array(self.dino_path, traj["dino_refs"][0])
        if dino_feature.ndim != 2:
            raise ValueError(
                f"DINO feature must have shape [P,D], got {dino_feature.shape} from {self.dino_path}"
            )
        if point_cloud.shape[0] != dino_feature.shape[0]:
            raise ValueError(
                "Point cloud and DINO feature must share point count, got "
                f"{point_cloud.shape} and {dino_feature.shape}"
            )
        if dino_feature.shape[-1] <= 3:
            raise ValueError(
                f"DINO feature dim must be >3, got {dino_feature.shape[-1]}"
            )
        if np.allclose(dino_feature, 0.0):
            raise ValueError("DINO feature is all zeros; fake semantic fallback is not allowed")
        if point_cloud.ndim == 2 and point_cloud.shape[1] >= 6:
            rgb = point_cloud[:, 3:6].astype(np.float32)
            reps = int(np.ceil(dino_feature.shape[1] / 3))
            rgb_tile = np.tile(rgb, (1, reps))[:, : dino_feature.shape[1]]
            if np.allclose(dino_feature.astype(np.float32), rgb_tile, atol=1e-6):
                raise ValueError(
                    "DINO feature is RGB tiled, not a real DINO embedding. "
                    "Regenerate RLBench2 Map4D DiT data with real DINO features."
                )

    def _iter_pointcloud_arrays_for_normalizer(self):
        total = sum(len(traj["point_refs"]) for traj in self.trajectories)
        progress = tqdm(
            total=total,
            desc="normalizing point_cloud",
            dynamic_ncols=True,
        )
        try:
            for traj in self.trajectories:
                for ref in traj["point_refs"]:
                    yield self._load_point_array(self.pcd_path, ref)
                    progress.update(1)
        finally:
            progress.close()

    def _iter_dino_arrays_for_normalizer(self):
        total = sum(len(traj["dino_refs"]) for traj in self.trajectories)
        progress = tqdm(
            total=total,
            desc="normalizing dino_feature",
            dynamic_ncols=True,
        )
        try:
            for traj in self.trajectories:
                for ref in traj["dino_refs"]:
                    yield self._load_point_array(self.dino_path, ref)
                    progress.update(1)
        finally:
            progress.close()

    def get_normalizer(self, mode="limits", **kwargs) -> LinearNormalizer:
        print(
            f"[RLBench2Map4DDataset] get_normalizer: "
            f"trajectories={len(self.trajectories)}, samples={len(self)}, use_rgb={self.use_rgb}",
            flush=True,
        )
        normalizer = LinearNormalizer()
        robot_state = np.concatenate([traj["robot_state"] for traj in self.trajectories], axis=0)
        node_position = np.concatenate([traj["map4d"][..., 0:3] for traj in self.trajectories], axis=0)
        size_parameters = np.stack([traj["size_parameters"] for traj in self.trajectories], axis=0)
        trajectory_pos = np.concatenate([traj["actions"][..., 0:3].reshape(-1, 3) for traj in self.trajectories], axis=0)
        gripper = np.concatenate([traj["actions"][..., 7:8].reshape(-1, 1) for traj in self.trajectories], axis=0)
        keyframe_map4d_pos = np.concatenate(
            [traj["keyframe_object"][..., 0:3].reshape(-1, 3) for traj in self.trajectories],
            axis=0,
        )
        keyframe_tcp_pos = np.concatenate(
            [traj["keyframe_tcp"][..., 0:3].reshape(-1, 3) for traj in self.trajectories],
            axis=0,
        )
        fields = {
            "robot_state": robot_state,
            "node_position": node_position,
            "size_parameters": size_parameters,
            "trajectory_pos": trajectory_pos,
            "gripper_openness": gripper,
            "keyframe_map4d_pos": keyframe_map4d_pos,
            "keyframe_tcp_pos": keyframe_tcp_pos,
        }
        normalizer.fit(fields, last_n_dims=1, mode=mode, **kwargs)
        normalizer["node_rotation"] = SingleFieldLinearNormalizer.create_identity()
        if self.use_rgb:
            print("[RLBench2Map4DDataset] normalizing point_cloud", flush=True)
            normalizer["point_cloud"] = _fit_streaming_normalizer(
                self._iter_pointcloud_arrays_for_normalizer(),
                mode=mode,
                **kwargs,
            )
            pc_params = normalizer.params_dict["point_cloud"]
            xyz_norm = SingleFieldLinearNormalizer.create_manual(
                pc_params["scale"][:3].detach().clone(),
                pc_params["offset"][:3].detach().clone(),
                {
                    name: value[:3].detach().clone()
                    for name, value in pc_params["input_stats"].items()
                },
            )
            normalizer["node_position"] = xyz_norm
            normalizer["keyframe_map4d_pos"] = xyz_norm
            print("[RLBench2Map4DDataset] normalizing dino_feature", flush=True)
            normalizer["dino_feature"] = _fit_streaming_normalizer(
                self._iter_dino_arrays_for_normalizer(),
                mode=mode,
                **kwargs,
            )
        print("[RLBench2Map4DDataset] get_normalizer done", flush=True)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([traj["actions"] for traj in self.trajectories], axis=0))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]
        obs_start = start - self.n_obs_steps + 1
        action_seq = _slice_with_pad(traj["actions"], start, self.horizon_action)
        if self.action_type == "bimanual_ee_pose":
            action_trajectory = np.transpose(action_seq[..., 0:7], (1, 0, 2))
            gripper_openness = np.transpose(action_seq[..., 7:8], (1, 0, 2))
        elif self.action_type == "single_arm_ee_pos":
            action_trajectory = np.concatenate([action_seq[:, 0:3], action_seq[:, 7:8]], axis=-1)
            gripper_openness = action_seq[:, 7:8]
        else:
            action_trajectory = action_seq[:, 0:7]
            gripper_openness = action_seq[:, 7:8]

        sample = {
            "obs": {
                "robot_state": _slice_with_pad(traj["robot_state"], obs_start, self.n_obs_steps),
                "node_position": _slice_with_pad(traj["map4d"][..., 0:3], obs_start, self.n_obs_steps),
                "node_rotation": _slice_with_pad(traj["map4d"][..., 3:7], obs_start, self.n_obs_steps),
                "size_parameters": traj["size_parameters"],
                "relation_parameters": traj["relation_parameters"],
            },
            "action": {
                "trajectory": action_trajectory,
                "gripper_openness": gripper_openness,
            },
            "keyframe": {
                "map4d": traj["keyframe_object"][start, : self.horizon_keyframe],
                "tcp": traj["keyframe_tcp"][start, : self.horizon_keyframe],
            },
        }
        if self.use_rgb:
            sample["obs"]["point_cloud"] = self._load_ref_window(
                self.pcd_path,
                traj["point_refs"],
                obs_start,
                self.n_obs_steps,
            )
            sample["obs"]["dino_feature"] = self._load_ref_window(
                self.dino_path,
                traj["dino_refs"],
                obs_start,
                self.n_obs_steps,
            )
        return _to_tensor_tree(sample)
