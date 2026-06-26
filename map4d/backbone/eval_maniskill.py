"""ManiSkill rollout evaluation for the standalone Map4D DiT policy."""

from __future__ import annotations

import json
import gc
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from map4d.backbone.policy.map4d_dit_policy import Map4DDiTPolicy
from helper.keyframe_targets import canonicalize_quaternion_np
from scripts.data_collection.helpers.build_point_semantic_features import (
    _assign_camera_patch_features,
    _encode_patch_grid,
    _parse_image_size,
    _preprocess_rgb,
    _project_world_points,
    _sample_patch_grid_at_uv,
    _target_size,
)
from diffusion_policy.dinov3_encoder import _load_backbone


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TASK_SIZE_PARAMETERS = {
    "StackCube-v1": np.asarray(
        [
            0.04,
            0.04,
            0.04,
            0.04,
            0.04,
            0.04,
        ],
        dtype=np.float32,
    ),
    "PlugCharger-v1": np.asarray(
        [
            0.04,
            0.03,
            0.024,
            0.016,
            0.0015,
            0.0064,
            0.02,
            0.004,
            0.04,
            0.004,
            0.045,
            0.025,
            0.1,
            0.1,
        ],
        dtype=np.float32,
    ),
}

TASK_RELATION_PARAMETERS = {
    "StackCube-v1": np.zeros((0,), dtype=np.float32),
    "PlugCharger-v1": np.asarray([0.007], dtype=np.float32),
}

STACKCUBE_TABLE_POSE = np.asarray(
    [-0.12, 0.0, -0.9196429, 0.70710677, 0.0, 0.0, 0.70710677],
    dtype=np.float32,
)

DEFAULT_POINTCLOUD_BBOX = {
    "StackCube-v1": ((-0.8, -0.6, -0.05), (0.5, 0.6, 0.8)),
    "PlugCharger-v1": ((-0.8, -0.6, -0.05), (0.5, 0.6, 0.8)),
}


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _append_episode_metrics(eval_metrics, episode_info):
    for key, value in episode_info.items():
        if key.startswith("_"):
            continue
        if torch.is_tensor(value):
            value = value.float().detach().cpu().numpy()
        eval_metrics[key].append(value)


def _metric_done(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.any().item())
    return bool(np.asarray(value).any())


def _metric_all(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.all().item())
    return bool(np.asarray(value).all())


def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H,W] or [H,W,1], got {depth.shape}")
    depth_m = depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        depth_m = depth_m / 1024.0
    return depth_m


def _pointcloud_from_depth(depth_m: np.ndarray, intrinsic_cv: np.ndarray, extrinsic_cv: np.ndarray) -> np.ndarray:
    height, width = depth_m.shape
    u = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    v = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    pixels = np.stack((u * depth_m, v * depth_m, depth_m), axis=-1)

    intrinsic_cv = np.asarray(intrinsic_cv, dtype=np.float32)
    extrinsic_cv = np.asarray(extrinsic_cv, dtype=np.float32)
    if intrinsic_cv.shape != (3, 3):
        raise ValueError(f"intrinsic_cv must have shape [3,3], got {intrinsic_cv.shape}")
    if extrinsic_cv.shape == (4, 4):
        extrinsic_cv = extrinsic_cv[:3]
    if extrinsic_cv.shape != (3, 4):
        raise ValueError(f"extrinsic_cv must have shape [3,4] or [4,4], got {extrinsic_cv.shape}")
    projection = intrinsic_cv @ extrinsic_cv
    projection_h = np.concatenate([projection, np.asarray([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
    projection_inv = np.linalg.inv(projection_h)[:3]
    pixels_h = np.concatenate(
        [pixels.reshape(-1, 3), np.ones((height * width, 1), dtype=np.float32)], axis=1
    )
    return (pixels_h @ projection_inv.T).astype(np.float32)


def _points_per_camera(total: int, num_cameras: int) -> Sequence[int]:
    base = total // num_cameras
    rem = total % num_cameras
    return [base + (1 if idx < rem else 0) for idx in range(num_cameras)]


def _parse_bbox(value, task_name: str) -> tuple[np.ndarray, np.ndarray]:
    if value is None or str(value).strip().lower() == "auto":
        mins, maxs = DEFAULT_POINTCLOUD_BBOX.get(task_name, DEFAULT_POINTCLOUD_BBOX["StackCube-v1"])
        return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        mins = np.asarray(value[0], dtype=np.float32)
        maxs = np.asarray(value[1], dtype=np.float32)
    else:
        parts = [float(x) for x in str(value).replace(",", " ").split()]
        if len(parts) != 6:
            raise ValueError("rollout.pointcloud_bbox must be auto or six numbers")
        mins = np.asarray(parts[:3], dtype=np.float32)
        maxs = np.asarray(parts[3:], dtype=np.float32)
    if mins.shape != (3,) or maxs.shape != (3,) or np.any(maxs <= mins):
        raise ValueError(f"Invalid rollout.pointcloud_bbox min/max: {mins} {maxs}")
    return mins, maxs


class Map4DDiTManiSkillEvaluator:
    def __init__(self, cfg: DictConfig, *, device: torch.device, output_dir: str):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.env = None
        self.robot_history = []
        self.node_pose_history = []
        self.point_cloud_history = []
        self.rgb_history = []
        self.point_camera_index_history = []
        self.point_pixel_uv_history = []
        self.depth_history = []
        self.intrinsic_history = []
        self.extrinsic_history = []
        self.dino_backbone = None
        self.dino_image_size = None
        self.dino_batch_size = int(self.cfg.get("dinov3_batch_size", 16))
        self.dino_input_multiple = int(self.cfg.get("dinov3_input_multiple", 16))
        self.dino_amp = bool(self.cfg.get("dinov3_amp", True))
        self.node_center_max_views = int(self.cfg.get("node_center_max_views", 2))

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.robot_history = []
        self.node_pose_history = []
        self.point_cloud_history = []
        self.rgb_history = []
        self.point_camera_index_history = []
        self.point_pixel_uv_history = []
        self.depth_history = []
        self.intrinsic_history = []
        self.extrinsic_history = []
        gc.collect()

    def _ensure_env(self):
        if self.env is not None:
            return
        import mani_skill.envs  # noqa: F401

        env_kwargs = dict(
            control_mode=self.cfg.control_mode,
            reward_mode="sparse",
            obs_mode=self.cfg.obs_mode,
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
            max_episode_steps=int(self.cfg.max_episode_steps),
        )
        if int(self.cfg.get("image_size", 0)) > 0:
            image_size = int(self.cfg.image_size)
            env_kwargs["sensor_configs"] = dict(width=image_size, height=image_size)
        env = gym.make(
            self.cfg.env_id,
            num_envs=int(self.cfg.num_eval_envs),
            sim_backend=self.cfg.sim_backend,
            reconfiguration_freq=1,
            **env_kwargs,
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        self.max_episode_steps = int(max_episode_steps or self.cfg.max_episode_steps)
        self.env = ManiSkillVectorEnv(
            env,
            ignore_terminations=True,
            record_metrics=True,
        )

    def evaluate(
        self,
        policy: Map4DDiTPolicy,
        epoch: int,
        iteration: Optional[int] = None,
    ) -> Dict[str, float]:
        self._ensure_env()
        policy.eval()
        eval_metrics = defaultdict(list)
        progress = tqdm(total=int(self.cfg.num_eval_episodes), desc=f"Rollout epoch {epoch}")
        reset_seed = self.cfg.get("seed", None)
        obs, info = self.env.reset(seed=None if reset_seed is None else int(reset_seed))
        self._reset_history(obs)
        episodes = 0

        try:
            with torch.no_grad():
                while episodes < int(self.cfg.num_eval_episodes):
                    policy_obs = self._policy_obs()
                    result = policy.predict_action(policy_obs)
                    actions = self._env_actions(result["action"])
                    for step_idx in range(actions.shape[1]):
                        obs, reward, terminated, truncated, info = self.env.step(actions[:, step_idx])
                        if _metric_done(truncated):
                            break
                        self._append_history(obs)

                    if _metric_done(truncated):
                        assert _metric_all(truncated), (
                            "all episodes should truncate at the same time for fair evaluation with other algorithms"
                        )
                        self._collect_metrics(eval_metrics, info)
                        episodes += self.env.num_envs
                        progress.update(self.env.num_envs)
                        self._reset_history(obs)
        finally:
            progress.close()
            if bool(self.cfg.get("close_after_eval", True)):
                self.close()

        mean_metrics = {}
        for key, values in eval_metrics.items():
            mean_metrics[key] = float(np.mean(np.stack(values)))
        self._write_metrics(epoch, iteration, mean_metrics)
        return mean_metrics

    def _reset_history(self, obs):
        robot_state = self._robot_state(obs)
        node_position, node_rotation = self._node_pose_parts()
        (
            point_cloud,
            rgb,
            point_camera_index,
            point_pixel_uv,
            depth,
            intrinsic,
            extrinsic,
        ) = self._visual_context(obs)
        horizon = int(self.cfg.obs_horizon)
        self.robot_history = [robot_state.copy() for _ in range(horizon)]
        self.node_position_history = [node_position.copy() for _ in range(horizon)]
        self.node_rotation_history = [node_rotation.copy() for _ in range(horizon)]
        self.point_cloud_history = [point_cloud.copy() for _ in range(horizon)]
        self.rgb_history = [rgb.copy() for _ in range(horizon)]
        self.point_camera_index_history = [point_camera_index.copy() for _ in range(horizon)]
        self.point_pixel_uv_history = [point_pixel_uv.copy() for _ in range(horizon)]
        self.depth_history = [depth.copy() for _ in range(horizon)]
        self.intrinsic_history = [intrinsic.copy() for _ in range(horizon)]
        self.extrinsic_history = [extrinsic.copy() for _ in range(horizon)]

    def _append_history(self, obs):
        self.robot_history.append(self._robot_state(obs))
        node_position, node_rotation = self._node_pose_parts()
        self.node_position_history.append(node_position)
        self.node_rotation_history.append(node_rotation)
        (
            point_cloud,
            rgb,
            point_camera_index,
            point_pixel_uv,
            depth,
            intrinsic,
            extrinsic,
        ) = self._visual_context(obs)
        self.point_cloud_history.append(point_cloud)
        self.rgb_history.append(rgb)
        self.point_camera_index_history.append(point_camera_index)
        self.point_pixel_uv_history.append(point_pixel_uv)
        self.depth_history.append(depth)
        self.intrinsic_history.append(intrinsic)
        self.extrinsic_history.append(extrinsic)
        horizon = int(self.cfg.obs_horizon)
        self.robot_history = self.robot_history[-horizon:]
        self.node_position_history = self.node_position_history[-horizon:]
        self.node_rotation_history = self.node_rotation_history[-horizon:]
        self.point_cloud_history = self.point_cloud_history[-horizon:]
        self.rgb_history = self.rgb_history[-horizon:]
        self.point_camera_index_history = self.point_camera_index_history[-horizon:]
        self.point_pixel_uv_history = self.point_pixel_uv_history[-horizon:]
        self.depth_history = self.depth_history[-horizon:]
        self.intrinsic_history = self.intrinsic_history[-horizon:]
        self.extrinsic_history = self.extrinsic_history[-horizon:]

    def _policy_obs(self) -> Dict[str, torch.Tensor]:
        batch_size = self.env.num_envs
        task_name = self.cfg.env_id
        point_cloud = np.stack(self.point_cloud_history, axis=1)
        rgb = np.stack(self.rgb_history, axis=1)
        point_camera_index = np.stack(self.point_camera_index_history, axis=1)
        point_pixel_uv = np.stack(self.point_pixel_uv_history, axis=1)
        node_position = np.stack(self.node_position_history, axis=1)
        depth = np.stack(self.depth_history, axis=1)
        intrinsic = np.stack(self.intrinsic_history, axis=1)
        extrinsic = np.stack(self.extrinsic_history, axis=1)
        dino_feature, node_rgb = self._dino_feature_history(
            rgb,
            point_camera_index,
            point_pixel_uv,
            node_position,
            depth,
            intrinsic,
            extrinsic,
        )
        node_point_cloud = np.concatenate([node_position, node_rgb], axis=-1)
        point_cloud = np.concatenate([point_cloud, node_point_cloud], axis=2)
        obs = {
            "robot_state": torch.as_tensor(np.stack(self.robot_history, axis=1), device=self.device),
            "node_position": torch.as_tensor(node_position, device=self.device),
            "node_rotation": torch.as_tensor(np.stack(self.node_rotation_history, axis=1), device=self.device),
            "size_parameters": torch.as_tensor(
                np.broadcast_to(
                    TASK_SIZE_PARAMETERS[task_name],
                    (batch_size, TASK_SIZE_PARAMETERS[task_name].shape[0]),
                ).copy(),
                device=self.device,
            ),
            "relation_parameters": torch.as_tensor(
                np.broadcast_to(
                    TASK_RELATION_PARAMETERS[task_name],
                    (batch_size, TASK_RELATION_PARAMETERS[task_name].shape[0]),
                ).copy(),
                device=self.device,
            ),
            "point_cloud": torch.as_tensor(point_cloud, device=self.device),
            "dino_feature": torch.as_tensor(dino_feature, device=self.device),
            "rgb": torch.as_tensor(rgb, device=self.device),
            "point_camera_index": torch.as_tensor(
                point_camera_index,
                device=self.device,
                dtype=torch.long,
            ),
            "point_pixel_uv": torch.as_tensor(
                point_pixel_uv,
                device=self.device,
                dtype=torch.long,
            ),
        }
        return obs

    def _ensure_dino_backbone(self):
        if self.dino_backbone is not None:
            return self.dino_backbone
        model = str(self.cfg.get("dinov3_model", "dinov3_vits16"))
        weights_path = str(self.cfg.get("dinov3_weights_path", ""))
        third_party_dir = str(self.cfg.get("dinov3_third_party_dir", ""))
        if not weights_path:
            raise ValueError("rollout.dinov3_weights_path is required for online Map4DDiT eval")
        if not third_party_dir:
            raise ValueError("rollout.dinov3_third_party_dir is required for online Map4DDiT eval")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        backbone = _load_backbone(third_party_dir, weights_path, model)
        backbone.to(self.device).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.dino_backbone = backbone
        image_size = self.cfg.get("image_size", None)
        self.dino_image_size = _parse_image_size(None if image_size is None else str(image_size))
        return self.dino_backbone

    def _dino_feature_history(
        self,
        rgb: np.ndarray,
        point_camera_index: np.ndarray,
        point_pixel_uv: np.ndarray,
        node_xyz: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rgb.ndim != 6 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb history must be [B,T,C,H,W,3], got {rgb.shape}")
        if point_camera_index.shape[:3] != rgb.shape[:2] + (point_camera_index.shape[2],):
            raise ValueError(
                f"point_camera_index must be [B,T,P], got {point_camera_index.shape} for rgb {rgb.shape}"
            )
        batch_size, steps, num_cameras, height, width, _ = rgb.shape
        point_count = int(point_camera_index.shape[2])
        if point_pixel_uv.shape != (batch_size, steps, point_count, 2):
            raise ValueError(
                f"point_pixel_uv must be [B,T,P,2], got {point_pixel_uv.shape}; "
                f"expected {(batch_size, steps, point_count, 2)}"
            )
        if num_cameras != len(tuple(self.cfg.camera_names)):
            raise ValueError(f"rgb has {num_cameras} cameras but cfg.camera_names={list(self.cfg.camera_names)}")
        if node_xyz.ndim != 4 or node_xyz.shape[:2] != rgb.shape[:2] or node_xyz.shape[-1] != 3:
            raise ValueError(f"node_xyz must be [B,T,N,3], got {node_xyz.shape} for rgb {rgb.shape}")
        if depth.shape[:3] != (batch_size, steps, num_cameras):
            raise ValueError(f"depth must be [B,T,C,H,W], got {depth.shape}")
        if intrinsic.shape[:3] != (batch_size, steps, num_cameras):
            raise ValueError(f"intrinsic must be [B,T,C,3,3], got {intrinsic.shape}")
        if extrinsic.shape[:3] != (batch_size, steps, num_cameras):
            raise ValueError(f"extrinsic must be [B,T,C,3,4], got {extrinsic.shape}")

        backbone = self._ensure_dino_backbone()
        flat_camera_index = point_camera_index.reshape(batch_size * steps, point_count)
        flat_pixel_uv = point_pixel_uv.reshape(batch_size * steps, point_count, 2)
        flat_node_xyz = node_xyz.reshape(batch_size * steps, node_xyz.shape[2], 3)
        flat_depth = depth.reshape(batch_size * steps, num_cameras, depth.shape[-2], depth.shape[-1])
        flat_intrinsic = intrinsic.reshape(batch_size * steps, num_cameras, 3, 3)
        flat_extrinsic = extrinsic.reshape(batch_size * steps, num_cameras, 3, 4)
        flat_rgb = rgb.reshape(batch_size * steps, num_cameras, height, width, 3)
        semantic_feature = None
        filled = np.zeros((batch_size * steps, point_count), dtype=bool)
        camera_patch_grids = []
        for camera_idx in range(num_cameras):
            camera_rgb = rgb[:, :, camera_idx].reshape(batch_size * steps, height, width, 3)
            patch_grid, _ = _encode_patch_grid(
                camera_rgb,
                backbone=backbone,
                device=self.device,
                batch_size=self.dino_batch_size,
                image_size=self.dino_image_size,
                multiple=self.dino_input_multiple,
                amp=self.dino_amp,
                preprocess_rgb=_preprocess_rgb,
                target_size_fn=_target_size,
            )
            if semantic_feature is None:
                semantic_feature = np.empty(
                    (batch_size * steps, point_count, int(patch_grid.shape[-1])),
                    dtype=np.float32,
                )
            _assign_camera_patch_features(
                semantic_feature,
                filled,
                patch_grid=patch_grid,
                camera_index=flat_camera_index,
                pixel_uv=flat_pixel_uv,
                camera_idx=camera_idx,
                image_hw=(height, width),
            )
            camera_patch_grids.append(patch_grid)
        if semantic_feature is None:
            raise RuntimeError("No DINO feature was generated")
        if not filled.all():
            missing = int((~filled).sum())
            raise ValueError(f"DINO feature assignment missed {missing} point tokens")
        node_feature, node_rgb = self._node_center_semantic_features_online(
            flat_node_xyz,
            camera_patch_grids=camera_patch_grids,
            rgb=flat_rgb,
            depth=flat_depth,
            intrinsic=flat_intrinsic,
            extrinsic=flat_extrinsic,
            image_hw=(height, width),
        )
        unified_feature = np.concatenate([semantic_feature, node_feature], axis=1)
        return (
            unified_feature.reshape(batch_size, steps, unified_feature.shape[1], unified_feature.shape[-1]),
            node_rgb.reshape(batch_size, steps, node_rgb.shape[1], 3),
        )

    def _node_center_semantic_features_online(
        self,
        node_xyz: np.ndarray,
        *,
        camera_patch_grids: Sequence[np.ndarray],
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        image_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.node_center_max_views <= 0:
            raise ValueError("rollout.node_center_max_views must be positive")
        if node_xyz.ndim != 3 or node_xyz.shape[-1] != 3:
            raise ValueError(f"node_xyz must be [B*T,N,3], got {node_xyz.shape}")
        flat_steps, num_nodes, _ = node_xyz.shape
        num_cameras = len(camera_patch_grids)
        feature_dim = int(camera_patch_grids[0].shape[-1])
        node_feature = np.zeros((flat_steps, num_nodes, feature_dim), dtype=np.float32)
        node_rgb = np.zeros((flat_steps, num_nodes, 3), dtype=np.float32)
        if rgb.shape[:2] != (flat_steps, num_cameras) or rgb.shape[-1] != 3:
            raise ValueError(f"rgb must be [B*T,C,H,W,3], got {rgb.shape}")

        for frame_idx in range(flat_steps):
            candidates = [[] for _ in range(num_nodes)]
            for camera_idx in range(num_cameras):
                camera_depth = depth[frame_idx, camera_idx]
                camera_intrinsic = intrinsic[frame_idx, camera_idx]
                camera_extrinsic = extrinsic[frame_idx, camera_idx]
                uv, z, valid = _project_world_points(node_xyz[frame_idx], camera_intrinsic, camera_extrinsic)
                in_bounds = (
                    valid
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < image_hw[1])
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < image_hw[0])
                )
                visible_indices = np.flatnonzero(in_bounds)
                if visible_indices.size == 0:
                    continue
                uv_int = np.rint(uv[visible_indices]).astype(np.int32)
                uv_int[:, 0] = np.clip(uv_int[:, 0], 0, image_hw[1] - 1)
                uv_int[:, 1] = np.clip(uv_int[:, 1], 0, image_hw[0] - 1)
                depth_at_uv = camera_depth[uv_int[:, 1], uv_int[:, 0]]
                depth_delta = np.abs(depth_at_uv.astype(np.float32) - z[visible_indices])
                distance = np.linalg.norm(
                    node_xyz[frame_idx, visible_indices] - camera_extrinsic[:3, 3][None],
                    axis=1,
                )
                weight = 1.0 / ((depth_delta + 1e-3) * (distance + 1e-3))
                patch_feature = _sample_patch_grid_at_uv(
                    camera_patch_grids[camera_idx][frame_idx],
                    uv[visible_indices],
                    image_hw=image_hw,
                )
                rgb_feature = rgb[frame_idx, camera_idx, uv_int[:, 1], uv_int[:, 0]].astype(np.float32)
                for local_idx, node_idx in enumerate(visible_indices):
                    candidates[int(node_idx)].append(
                        (
                            float(weight[local_idx]),
                            patch_feature[local_idx].astype(np.float32),
                            rgb_feature[local_idx],
                        )
                    )

            for node_idx, node_candidates in enumerate(candidates):
                if not node_candidates:
                    raise ValueError(
                        f"rollout node center frame={frame_idx}, node={node_idx} "
                        "does not project into any configured camera"
                    )
                node_candidates.sort(key=lambda item: item[0], reverse=True)
                selected = node_candidates[: self.node_center_max_views]
                weights = np.asarray([item[0] for item in selected], dtype=np.float32)
                weights = weights / np.clip(weights.sum(), 1e-8, None)
                features = np.stack([item[1] for item in selected], axis=0)
                rgbs = np.stack([item[2] for item in selected], axis=0)
                node_feature[frame_idx, node_idx] = np.sum(features * weights[:, None], axis=0)
                node_rgb[frame_idx, node_idx] = np.sum(rgbs * weights[:, None], axis=0)

        return node_feature, node_rgb

    def _robot_state(self, obs) -> np.ndarray:
        agent = obs["agent"]
        extra = obs.get("extra", {})
        fields = []
        for value in (agent.get("qpos"), agent.get("qvel"), extra.get("tcp_pose"), extra.get("tcp_pos")):
            if value is None:
                continue
            arr = _to_numpy(value).reshape(self.env.num_envs, -1).astype(np.float32)
            fields.append(arr)
        if not fields:
            state = np.zeros((self.env.num_envs, int(self.cfg.robot_state_dim)), dtype=np.float32)
        else:
            state = np.concatenate(fields, axis=-1).astype(np.float32)
        target_dim = int(self.cfg.robot_state_dim)
        if state.shape[-1] < target_dim:
            pad = np.zeros((state.shape[0], target_dim - state.shape[-1]), dtype=np.float32)
            state = np.concatenate([state, pad], axis=-1)
        elif state.shape[-1] > target_dim:
            state = state[:, :target_dim]
        return state.astype(np.float32)

    def _node_pose_parts(self) -> tuple[np.ndarray, np.ndarray]:
        base_env = self.env.base_env
        if self.cfg.env_id == "StackCube-v1":
            poses = [
                self._actor_pose(base_env.cubeA),
                self._actor_pose(base_env.cubeB),
            ]
        else:
            poses = [
                self._actor_pose(base_env.charger),
                self._actor_pose(base_env.receptacle),
            ]
        pos = np.stack([pose[:, 0:3] for pose in poses], axis=1)
        rot = np.stack([canonicalize_quaternion_np(pose[:, 3:7]) for pose in poses], axis=1)
        return pos.astype(np.float32), rot.astype(np.float32)

    def _actor_pose(self, actor) -> np.ndarray:
        raw_pose = _to_numpy(actor.pose.raw_pose).astype(np.float32)
        if raw_pose.ndim == 1:
            raw_pose = raw_pose[None]
        return raw_pose[:, :7]

    def _table_pose(self, env) -> np.ndarray:
        table_scene = getattr(env, "table_scene", None)
        table_actor = getattr(table_scene, "table", None)
        if table_actor is not None:
            return self._actor_pose(table_actor)
        return np.broadcast_to(STACKCUBE_TABLE_POSE, (self.env.num_envs, 7)).astype(np.float32)

    def _visual_context(
        self,
        obs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if "sensor_data" not in obs or "sensor_param" not in obs:
            raise KeyError(
                "Map4D DiT online rollout requires rgb+depth observations with "
                "obs['sensor_data'] and obs['sensor_param']."
            )
        camera_names = tuple(self.cfg.camera_names)
        rgb_images = []
        per_camera_counts = _points_per_camera(int(self.cfg.pointcloud_num_points), len(camera_names))
        per_camera_xyzrgb = []
        per_camera_pixel_uv = []
        per_camera_index = []
        camera_depths = []
        camera_intrinsics = []
        camera_extrinsics = []
        bbox_min, bbox_max = _parse_bbox(self.cfg.get("pointcloud_bbox", "auto"), self.cfg.env_id)
        rng = np.random.default_rng(int(self.cfg.get("pointcloud_seed", 0)) + len(self.point_cloud_history))

        for camera_idx, (camera_name, camera_points) in enumerate(zip(camera_names, per_camera_counts)):
            sensor_data = obs["sensor_data"]
            sensor_param = obs["sensor_param"]
            if camera_name not in sensor_data or camera_name not in sensor_param:
                raise KeyError(f"Observation missing camera {camera_name!r}")
            camera_data = sensor_data[camera_name]
            camera_param = sensor_param[camera_name]
            for key in ("rgb", "depth"):
                if key not in camera_data:
                    raise KeyError(f"Observation camera {camera_name!r} missing sensor_data/{key}")
            for key in ("intrinsic_cv", "extrinsic_cv"):
                if key not in camera_param:
                    raise KeyError(f"Observation camera {camera_name!r} missing sensor_param/{key}")

            rgb = _to_numpy(camera_data["rgb"])
            depth = _to_numpy(camera_data["depth"])
            intrinsic = _to_numpy(camera_param["intrinsic_cv"])
            extrinsic = _to_numpy(camera_param["extrinsic_cv"])
            if rgb.ndim == 3:
                rgb = rgb[None]
            if depth.ndim == 2 or (depth.ndim == 3 and depth.shape[-1] == 1):
                depth = depth[None]
            if intrinsic.ndim == 2:
                intrinsic = intrinsic[None]
            if extrinsic.ndim == 2:
                extrinsic = extrinsic[None]
            if rgb.shape[0] != self.env.num_envs or depth.shape[0] != self.env.num_envs:
                raise ValueError(
                    f"Camera {camera_name!r} batch mismatch: rgb={rgb.shape}, depth={depth.shape}, "
                    f"num_envs={self.env.num_envs}"
                )
            if rgb.shape[-1] != 3:
                raise ValueError(f"Camera {camera_name!r} rgb must have shape [B,H,W,3], got {rgb.shape}")

            rgb_u8 = rgb.astype(np.uint8, copy=False)
            rgb_images.append(rgb_u8)
            depth_m_batch = np.stack([_depth_to_meters(depth[env_idx]) for env_idx in range(self.env.num_envs)], axis=0)
            extrinsic_3x4 = extrinsic[:, :3] if extrinsic.shape[-2:] == (4, 4) else extrinsic
            if extrinsic_3x4.shape[-2:] != (3, 4):
                raise ValueError(f"Camera {camera_name!r} extrinsic_cv must be [B,3,4] or [B,4,4], got {extrinsic.shape}")
            camera_depths.append(depth_m_batch.astype(np.float32))
            camera_intrinsics.append(intrinsic.astype(np.float32))
            camera_extrinsics.append(extrinsic_3x4.astype(np.float32))
            camera_xyzrgb = []
            camera_pixel_uv = []
            camera_index = []
            for env_idx in range(self.env.num_envs):
                depth_m = depth_m_batch[env_idx]
                xyz = _pointcloud_from_depth(depth_m, intrinsic[env_idx], extrinsic[env_idx])
                height, width = depth_m.shape
                u = np.tile(np.arange(width, dtype=np.int32), (height, 1))
                v = np.tile(np.arange(height, dtype=np.int32)[:, None], (1, width))
                pixel_uv = np.stack((u, v), axis=-1).reshape(-1, 2)
                xyzrgb = np.concatenate([xyz, rgb_u8[env_idx].reshape(-1, 3).astype(np.float32)], axis=1)
                sampled, sampled_pixel_uv, sampled_camera_index = self._sample_points_strict(
                    xyzrgb,
                    pixel_uv,
                    np.full((xyzrgb.shape[0],), camera_idx, dtype=np.int64),
                    camera_points,
                    bbox_min,
                    bbox_max,
                    rng,
                )
                camera_xyzrgb.append(sampled)
                camera_pixel_uv.append(sampled_pixel_uv)
                camera_index.append(sampled_camera_index)
            per_camera_xyzrgb.append(camera_xyzrgb)
            per_camera_pixel_uv.append(camera_pixel_uv)
            per_camera_index.append(camera_index)

        env_clouds = []
        env_camera_indices = []
        env_pixel_uvs = []
        for env_idx in range(self.env.num_envs):
            xyzrgb = np.concatenate(
                [per_camera_xyzrgb[camera_idx][env_idx] for camera_idx in range(len(camera_names))],
                axis=0,
            )
            pixel_uv = np.concatenate(
                [per_camera_pixel_uv[camera_idx][env_idx] for camera_idx in range(len(camera_names))],
                axis=0,
            )
            camera_index = np.concatenate(
                [per_camera_index[camera_idx][env_idx] for camera_idx in range(len(camera_names))],
                axis=0,
            )
            if xyzrgb.shape[0] != int(self.cfg.pointcloud_num_points):
                raise ValueError(
                    "Rollout point cloud sampling produced "
                    f"{xyzrgb.shape[0]} points, expected {int(self.cfg.pointcloud_num_points)}"
                )
            env_clouds.append(xyzrgb)
            env_pixel_uvs.append(pixel_uv)
            env_camera_indices.append(camera_index)

        return (
            np.stack(env_clouds, axis=0).astype(np.float32),
            np.stack(rgb_images, axis=1).astype(np.uint8),
            np.stack(env_camera_indices, axis=0).astype(np.int64),
            np.stack(env_pixel_uvs, axis=0).astype(np.int64),
            np.stack(camera_depths, axis=1).astype(np.float32),
            np.stack(camera_intrinsics, axis=1).astype(np.float32),
            np.stack(camera_extrinsics, axis=1).astype(np.float32),
        )

    def _sample_points_strict(
        self,
        xyzrgb: np.ndarray,
        pixel_uv: np.ndarray,
        camera_index: np.ndarray,
        num_points: int,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyz = xyzrgb[:, :3]
        valid = np.isfinite(xyz).all(axis=1)
        valid &= xyz[:, 2] > bbox_min[2]
        inside = valid & np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)
        candidate_indices = np.flatnonzero(inside)
        if len(candidate_indices) == 0:
            raise ValueError(
                "No valid RGB-D points inside rollout.pointcloud_bbox; "
                "check obs_mode, camera params, depth scale, and workspace bbox."
            )
        replace = len(candidate_indices) < num_points
        sampled_indices = rng.choice(candidate_indices, size=num_points, replace=replace)
        return (
            xyzrgb[sampled_indices].astype(np.float32),
            pixel_uv[sampled_indices].astype(np.int64),
            camera_index[sampled_indices].astype(np.int64),
        )

    def _env_actions(self, action: torch.Tensor):
        action = action.detach()
        if self.cfg.action_mode == "pos_gripper":
            if action.shape[-1] == 4:
                action = action
            elif action.shape[-1] == 8:
                action = torch.cat([action[..., 0:3], action[..., 7:8]], dim=-1)
            else:
                raise ValueError(
                    f"pos_gripper rollout expects action dim 4 or 8, got {action.shape[-1]}"
                )
        elif self.cfg.action_mode == "pose":
            action = action[..., :7]
        else:
            raise ValueError(f"Unsupported rollout action_mode={self.cfg.action_mode!r}")
        action = self._clip_action(action)
        if self.cfg.sim_backend == "physx_cpu":
            return action.cpu().numpy()
        return action

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        space = getattr(self.env, "single_action_space", self.env.action_space)
        if not isinstance(space, gym.spaces.Box):
            return action
        low = torch.as_tensor(space.low, device=action.device, dtype=action.dtype)
        high = torch.as_tensor(space.high, device=action.device, dtype=action.dtype)
        while low.ndim < action.ndim:
            low = low.unsqueeze(0)
            high = high.unsqueeze(0)
        return torch.max(torch.min(action, high), low)

    def _collect_metrics(self, eval_metrics, info):
        if "final_info" in info and isinstance(info["final_info"], dict):
            _append_episode_metrics(eval_metrics, info["final_info"]["episode"])
        elif "final_info" in info:
            for final_info in info["final_info"]:
                _append_episode_metrics(eval_metrics, final_info["episode"])
        elif "episode" in info:
            _append_episode_metrics(eval_metrics, info["episode"])
        else:
            raise KeyError(f"Expected episode metrics in info, got keys: {list(info.keys())}")

    def _metrics_record(self, epoch: int, iteration: Optional[int], metrics: Dict[str, float]):
        return {
            "epoch": int(epoch),
            "iteration": int(epoch if iteration is None else iteration),
            "num_eval_episodes": int(self.cfg.num_eval_episodes),
            "run_name": str(self.cfg.run_name),
            **metrics,
        }

    def _write_metrics(self, epoch: int, iteration: Optional[int], metrics: Dict[str, float]):
        record = self._metrics_record(epoch, iteration, metrics)
        run_dir_record = {
            "time": time.time(),
            "num_eval_envs": int(self.cfg.num_eval_envs),
            **record,
        }

        path = Path(self.output_dir) / "rollout_metrics.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run_dir_record, sort_keys=True) + "\n")

        metrics_dir = Path(str(self.cfg.metrics_dir))
        if not metrics_dir.is_absolute():
            metrics_dir = PROJECT_ROOT / metrics_dir
        metrics_dir.mkdir(parents=True, exist_ok=True)
        eval_metrics_path = metrics_dir / f"{self.cfg.run_name}.jsonl"
        with eval_metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"Eval metrics written to {eval_metrics_path}")


def build_rollout_evaluator(cfg: DictConfig, *, device: torch.device, output_dir: str) -> Optional[Map4DDiTManiSkillEvaluator]:
    rollout_cfg = cfg.get("rollout")
    if rollout_cfg is None or not bool(rollout_cfg.enabled):
        return None
    resolved = OmegaConf.to_container(rollout_cfg, resolve=True)
    resolved["env_id"] = cfg.task.name
    resolved["obs_horizon"] = int(cfg.n_obs_steps)
    resolved["robot_state_dim"] = int(cfg.robot_state_dim)
    resolved["use_rgb"] = bool(cfg.policy.model_cfg.use_rgb)
    return Map4DDiTManiSkillEvaluator(OmegaConf.create(resolved), device=device, output_dir=output_dir)
