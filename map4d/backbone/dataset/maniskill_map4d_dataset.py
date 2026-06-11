"""ManiSkill HDF5 dataset for the standalone Map4D DiT policy."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import h5py
except ModuleNotFoundError:  # synthetic smoke tests do not require HDF5 support.
    h5py = None

from helper.keyframe_targets import (
    MAP4D_DIT_TARGET_FORMAT,
    MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT,
    build_future_keyframe_table,
    canonicalize_quaternion_np,
    gather_map4d_dit_keyframe_targets,
    matrix_to_rot6d_np,
)
from map4d.backbone.dataset.base_dataset import BaseDataset
from map4d.backbone.model.common.normalizer import LinearNormalizer
from map4d.representation.maps4d.metadata import (
    get_task_actor_names,
    get_task_parameter_defaults,
)


TASK_ACTOR_NAMES = {
    "StackCube-v1": ("cubeA", "cubeB", "table-workspace"),
    "PlugCharger-v1": ("charger", "receptacle"),
}

TASK_GT_SIZES = {
    "StackCube-v1": (
        0.04,
        0.04,
        0.04,
        0.04,
        0.04,
        0.04,
        1.2090764,
        2.4178784,
        0.91964292762787,
    ),
    "PlugCharger-v1": (0.04, 0.03, 0.024, 0.02, 0.1, 0.1),
}

TASK_SIZE_PARAMETER_DEFAULTS = {
    "StackCube-v1": TASK_GT_SIZES["StackCube-v1"],
    "PlugCharger-v1": (
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
    ),
}

TASK_RELATION_PARAMETER_DEFAULTS = {
    "StackCube-v1": tuple(),
    "PlugCharger-v1": (0.007,),
}


def _traj_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def _quat_wxyz_to_rotation_6d_np(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = canonicalize_quaternion_np(quat_wxyz)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    matrix = np.empty((*quat.shape[:-1], 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix_to_rot6d_np(matrix)


def _axis_angle_to_quaternion_np(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to WXYZ quaternion. Zero vectors map to identity."""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True).clip(min=1e-10)
    half_angle = angle / 2.0
    axis = axis_angle / angle
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    quat = np.concatenate([w, xyz], axis=-1).astype(np.float32)
    # For near-zero rotations, return identity
    small = (angle < 1e-7).squeeze(-1)
    quat[small] = [1.0, 0.0, 0.0, 0.0]
    return quat


def _actor_states_to_map4d_tensor(
    actor_states: Sequence[np.ndarray],
    *,
    sizes: Optional[Iterable[float]],
) -> np.ndarray:
    states = [np.asarray(state, dtype=np.float32) for state in actor_states]
    num_objects = len(states)
    frame_count = states[0].shape[0]
    if any(state.shape[0] != frame_count for state in states):
        raise ValueError("All actor state arrays must have the same frame count.")
    if sizes is None:
        sizes_np = np.zeros((num_objects, 3), dtype=np.float32)
    else:
        sizes_np = np.asarray(tuple(sizes), dtype=np.float32).reshape(num_objects, 3)
    sizes_seq = np.broadcast_to(sizes_np, (frame_count, num_objects, 3))
    positions = np.stack([state[:, 0:3] for state in states], axis=1)
    rotations = np.stack([_quat_wxyz_to_rotation_6d_np(state[:, 3:7]) for state in states], axis=1)
    return np.concatenate([sizes_seq, positions, rotations], axis=-1).astype(np.float32)


def _actor_states_to_pose_map4d_tensor(actor_states: Sequence[np.ndarray]) -> np.ndarray:
    states = [np.asarray(state, dtype=np.float32) for state in actor_states]
    frame_count = states[0].shape[0]
    if any(state.shape[0] != frame_count for state in states):
        raise ValueError("All actor state arrays must have the same frame count.")
    positions = np.stack([state[:, 0:3] for state in states], axis=1)
    rotations = np.stack([_quat_wxyz_to_rotation_6d_np(state[:, 3:7]) for state in states], axis=1)
    return np.concatenate([positions, rotations], axis=-1).astype(np.float32)


def _task_parameters(task_name: str):
    try:
        size_defaults, relation_defaults = get_task_parameter_defaults(task_name)
        return (
            np.asarray(size_defaults, dtype=np.float32),
            np.asarray(relation_defaults, dtype=np.float32),
        )
    except KeyError:
        pass
    return (
        np.asarray(TASK_SIZE_PARAMETER_DEFAULTS.get(task_name, tuple()), dtype=np.float32),
        np.asarray(TASK_RELATION_PARAMETER_DEFAULTS.get(task_name, tuple()), dtype=np.float32),
    )


def _default_actor_names(task_name: str):
    try:
        actor_names = get_task_actor_names(task_name)
        if actor_names:
            return actor_names
    except KeyError:
        pass
    return TASK_ACTOR_NAMES.get(task_name)


def _read_dataset(group: h5py.Group, path: str) -> Optional[np.ndarray]:
    node = group
    parts = path.split("/")
    for part in parts:
        if not isinstance(node, h5py.Group) or part not in node:
            return None
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node[()]
    return None


def _get_dataset_node(group: h5py.Group, path: str) -> Optional[h5py.Dataset]:
    node = group
    for part in path.split("/"):
        if not isinstance(node, h5py.Group) or part not in node:
            return None
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node
    return None


def _decode_h5_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _require_dino_provenance(traj_name: str, feature: h5py.Dataset) -> None:
    flattened = {}
    for node in (feature.file, feature.parent, feature, feature.parent.parent):
        if node is None:
            continue
        for key, value in node.attrs.items():
            flattened[str(key)] = _decode_h5_attr(value)

    for key, value in flattened.items():
        if key in {"semantic_feature_mode", "semantic_feature_source"} and value.lower() in {
            "rgb",
            "pointcloud_rgb",
            "rgb_placeholder",
        }:
            raise ValueError(
                f"{traj_name}: {feature.name} was produced by {key}={value!r}, not a DINO model"
            )

    provenance_text = " ".join(f"{key}={value}" for key, value in flattened.items()).lower()
    if "dino" not in provenance_text:
        raise ValueError(
            f"{traj_name}: {feature.name} is missing DINO provenance metadata. "
            "Expected attrs such as semantic_feature_model=dinov3_* or feature_type=dinov3."
        )


def _read_feature_array(group: h5py.Group, path: str, pool: str = "patch_mean") -> Optional[np.ndarray]:
    node = group
    for part in path.split("/"):
        if not isinstance(node, h5py.Group) or part not in node:
            return None
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node[()]
    if not isinstance(node, h5py.Group):
        return None

    features = []
    for key in sorted(node.keys()):
        child = node[key]
        if isinstance(child, h5py.Dataset):
            features.append(child[()])
        elif isinstance(child, h5py.Group) and pool in child and isinstance(child[pool], h5py.Dataset):
            features.append(child[pool][()])
    if not features:
        return None
    return np.stack(features, axis=1)


def _flatten_time_array(value: np.ndarray, length: int) -> Optional[np.ndarray]:
    value = np.asarray(value)
    if value.shape[0] != length:
        return None
    if not np.issubdtype(value.dtype, np.number):
        return None
    if value.ndim == 1:
        return value[:, None].astype(np.float32)
    if value.ndim <= 3:
        return value.reshape(length, -1).astype(np.float32)
    return None


class ManiSkillMap4DDataset(BaseDataset):
    """Load Map4D DiT batches from ManiSkill demos plus keyframe sidecars."""

    def __init__(
        self,
        demo_path: Optional[str] = None,
        keyframe_sidecar_path: Optional[str] = None,
        *,
        task_name: str = "StackCube-v1",
        actor_names: Optional[Sequence[str]] = None,
        horizon_action: int = 16,
        horizon_keyframe: int = 4,
        n_obs_steps: int = 2,
        num_objects: int = 3,
        robot_state_dim: int = 32,
        map4d_dim: int = 9,
        size_parameter_dim: int = 0,
        relation_parameter_dim: int = 0,
        use_rgb: bool = True,
        use_depth: bool = False,
        rgb_feature_dim: int = 288,
        semantic_feature_dim: Optional[int] = None,
        map_feature_dim: int = 240,
        pointcloud_path: str = "obs/point_cloud/fused",
        dino_feature_path: str = "obs/dino_feature",
        keyframe_tcp_dim: Optional[int] = None,
        allow_missing_rgb_feature: bool = False,
        allow_raw_rgb_stats_feature: bool = False,
        num_traj: Optional[int] = None,
        val_ratio: float = 0.0,
        seed: int = 0,
        synthetic: bool = False,
        synthetic_episodes: int = 8,
        synthetic_length: int = 32,
        strict_target_format: bool = True,
    ):
        super().__init__()
        self.demo_path = demo_path
        self.keyframe_sidecar_path = keyframe_sidecar_path
        self.task_name = task_name
        self.actor_names = tuple(actor_names) if actor_names is not None else _default_actor_names(task_name)
        self.horizon_action = int(horizon_action)
        self.horizon_keyframe = int(horizon_keyframe)
        self.n_obs_steps = int(n_obs_steps)
        self.num_objects = int(num_objects)
        self.robot_state_dim = int(robot_state_dim)
        self.map4d_dim = int(map4d_dim)
        self.size_parameter_dim = int(size_parameter_dim)
        self.relation_parameter_dim = int(relation_parameter_dim)
        self.use_rgb = bool(use_rgb)
        self.use_depth = bool(use_depth)
        self.rgb_feature_dim = int(rgb_feature_dim)
        self.semantic_feature_dim = (
            None if semantic_feature_dim is None else int(semantic_feature_dim)
        )
        self.map_feature_dim = int(map_feature_dim)
        self.pointcloud_path = str(pointcloud_path)
        self.dino_feature_path = str(dino_feature_path)
        self.keyframe_tcp_dim = None if keyframe_tcp_dim is None else int(keyframe_tcp_dim)
        self.allow_missing_rgb_feature = bool(allow_missing_rgb_feature)
        self.allow_raw_rgb_stats_feature = bool(allow_raw_rgb_stats_feature)
        self.strict_target_format = bool(strict_target_format)
        if isinstance(num_traj, str):
            num_traj = None if num_traj.lower() in {"", "none", "null"} else int(num_traj)

        if synthetic:
            self.trajectories = self._build_synthetic(synthetic_episodes, synthetic_length, seed)
        else:
            if demo_path is None:
                raise ValueError("demo_path is required unless synthetic=True")
            if h5py is None:
                raise ModuleNotFoundError("h5py is required to load ManiSkill HDF5 demos")
            self.trajectories = self._load_real_trajectories(demo_path, keyframe_sidecar_path, num_traj)

        self.indices = self._build_indices()
        rng = np.random.default_rng(seed)
        if val_ratio > 0.0 and len(self.indices) > 1:
            mask = rng.random(len(self.indices)) < val_ratio
            if mask.all():
                mask[0] = False
            self.val_indices = [idx for idx, is_val in zip(self.indices, mask) if is_val]
            self.indices = [idx for idx, is_val in zip(self.indices, mask) if not is_val]
        else:
            self.val_indices = []

    def _build_synthetic(self, episodes: int, length: int, seed: int):
        rng = np.random.default_rng(seed)
        trajectories = []
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(episodes):
            robot_state = rng.normal(size=(length + 1, self.robot_state_dim)).astype(np.float32)
            actions = rng.normal(scale=0.05, size=(length, 8)).astype(np.float32)
            actions[:, 3:7] = identity_quat
            actions[:, 7:8] = rng.uniform(-1.0, 1.0, size=(length, 1)).astype(np.float32)
            map4d = rng.normal(scale=0.05, size=(length + 1, self.num_objects, self.map4d_dim)).astype(np.float32)
            if self.map4d_dim == 9:
                map4d[..., 3:9] = identity_rot6d
            else:
                map4d[..., 0:3] = np.abs(map4d[..., 0:3]) + 0.02
                map4d[..., 6:12] = identity_rot6d
            size_parameters = np.abs(
                rng.normal(scale=0.05, size=(self.size_parameter_dim,))
            ).astype(np.float32)
            relation_parameters = np.abs(
                rng.normal(scale=0.01, size=(self.relation_parameter_dim,))
            ).astype(np.float32)
            object_targets = rng.normal(
                scale=0.05, size=(length + 1, self.horizon_keyframe, self.num_objects, 9)
            ).astype(np.float32)
            object_targets[..., 3:9] = identity_rot6d
            tcp_dim = self.keyframe_tcp_dim or 7
            tcp_targets = rng.normal(
                scale=0.05, size=(length + 1, self.horizon_keyframe, tcp_dim)
            ).astype(np.float32)
            if tcp_dim == 7:
                tcp_targets[..., 3:7] = identity_quat
            record = {
                "robot_state": robot_state,
                "actions": actions,
                "map4d": map4d,
                "size_parameters": size_parameters,
                "relation_parameters": relation_parameters,
                "keyframe_object": object_targets,
                "keyframe_tcp": tcp_targets,
            }
            if self.use_rgb:
                record["rgb_feature"] = rng.normal(
                    scale=0.1, size=(length + 1, self.rgb_feature_dim)
                ).astype(np.float32)
                point_count = 128
                record["point_cloud"] = rng.normal(
                    scale=0.1, size=(length + 1, point_count, 6)
                ).astype(np.float32)
                semantic_dim = self.semantic_feature_dim or self.rgb_feature_dim
                record["dino_feature"] = rng.normal(
                    scale=0.1, size=(length + 1, point_count, semantic_dim)
                ).astype(np.float32)
            trajectories.append(record)
        return trajectories

    def _load_point_cloud(self, traj: h5py.Group, length: int) -> np.ndarray:
        point_cloud = _read_dataset(traj, self.pointcloud_path)
        if point_cloud is None:
            raise KeyError(f"{traj.name} missing {self.pointcloud_path}")
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        if point_cloud.ndim != 3 or point_cloud.shape[0] != length or point_cloud.shape[-1] < 3:
            raise ValueError(
                f"{traj.name}/{self.pointcloud_path} must be [T,P,>=3] with T={length}, "
                f"got {point_cloud.shape}"
            )
        return point_cloud

    def _load_dino_feature(self, traj: h5py.Group, length: int, point_count: int) -> np.ndarray:
        dino_node = _get_dataset_node(traj, self.dino_feature_path)
        if dino_node is None:
            raise KeyError(f"{traj.name} missing {self.dino_feature_path}")
        _require_dino_provenance(traj.name, dino_node)
        dino_feature = dino_node[()]
        dino_feature = np.asarray(dino_feature, dtype=np.float32)
        if dino_feature.ndim != 3:
            raise ValueError(f"{traj.name}/{self.dino_feature_path} must be [T,P,D], got {dino_feature.shape}")
        if dino_feature.shape[:2] != (length, point_count):
            raise ValueError(
                f"{traj.name}: point_cloud and dino_feature must share [T,P], "
                f"got {(length, point_count)} and {dino_feature.shape[:2]}"
            )
        if self.semantic_feature_dim is not None and dino_feature.shape[-1] != self.semantic_feature_dim:
            raise ValueError(
                f"{traj.name}: expected semantic_feature_dim={self.semantic_feature_dim}, "
                f"got {dino_feature.shape[-1]}"
            )
        return dino_feature

    def _format_rgb_feature(self, feature: np.ndarray, length: int) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape[0] != length:
            raise ValueError(f"rgb_feature length {feature.shape[0]} != expected {length}")
        if feature.ndim == 1:
            feature = feature.reshape(length, 1)
        elif feature.ndim == 2:
            pass
        else:
            feature = feature.reshape(length, *feature.shape[1:-1], feature.shape[-1])
        if feature.shape[-1] < self.rgb_feature_dim:
            pad_shape = (*feature.shape[:-1], self.rgb_feature_dim - feature.shape[-1])
            pad = np.zeros(pad_shape, dtype=np.float32)
            feature = np.concatenate([feature, pad], axis=-1)
        elif feature.shape[-1] > self.rgb_feature_dim:
            feature = feature[..., : self.rgb_feature_dim]
        return feature.astype(np.float32)

    def _rgb_to_stats_feature(self, rgb: np.ndarray, length: int) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.shape[0] != length:
            raise ValueError(f"rgb length {rgb.shape[0]} != expected {length}")
        rgb = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float32)
        rgb = rgb.reshape(length, -1, rgb.shape[-1]) if rgb.ndim >= 3 else rgb.reshape(length, -1)
        mean = rgb.mean(axis=1)
        std = rgb.std(axis=1)
        return self._format_rgb_feature(np.concatenate([mean, std], axis=-1), length)

    def _find_rgb_dataset(self, group: h5py.Group):
        for key in group.keys():
            node = group[key]
            if isinstance(node, h5py.Dataset) and key.lower() == "rgb":
                return node[()]
            if isinstance(node, h5py.Group):
                found = self._find_rgb_dataset(node)
                if found is not None:
                    return found
        return None

    def _load_rgb_feature(self, traj: h5py.Group, length: int, sidecar_record=None) -> Optional[np.ndarray]:
        if sidecar_record is not None:
            for key in ("rgb_feature", "dino_feature"):
                if key in sidecar_record:
                    return self._format_rgb_feature(sidecar_record[key], length)
        for path in ("obs/rgb_feature", "obs/dino_feature", "rgb_feature", "dino_feature"):
            value = _read_feature_array(traj, path)
            if value is not None:
                return self._format_rgb_feature(value, length)
        if self.allow_raw_rgb_stats_feature and "obs" in traj:
            rgb = self._find_rgb_dataset(traj["obs"])
            if rgb is not None:
                return self._rgb_to_stats_feature(rgb, length)
        if self.allow_missing_rgb_feature:
            return np.zeros((length, self.rgb_feature_dim), dtype=np.float32)
        raise KeyError(
            "RGB context is enabled but no rgb_feature, dino_feature, or raw rgb dataset was found."
        )

    def _load_robot_state(self, traj: h5py.Group, length: int) -> np.ndarray:
        preferred = [
            "obs/agent/qpos",
            "obs/agent/qvel",
            "obs/extra/tcp_pose",
            "obs/extra/tcp_pos",
        ]
        fields = []
        for path in preferred:
            value = _read_dataset(traj, path)
            if value is not None:
                flat = _flatten_time_array(value, length)
                if flat is not None:
                    fields.append(flat)
        if not fields and "obs" in traj:
            for group_name in ("agent", "extra"):
                group = traj["obs"].get(group_name)
                if isinstance(group, h5py.Group):
                    for key in sorted(group.keys()):
                        flat = _flatten_time_array(group[key][()], length)
                        if flat is not None:
                            fields.append(flat)
        if not fields:
            return np.zeros((length, self.robot_state_dim), dtype=np.float32)
        state = np.concatenate(fields, axis=-1).astype(np.float32)
        if state.shape[-1] < self.robot_state_dim:
            pad = np.zeros((length, self.robot_state_dim - state.shape[-1]), dtype=np.float32)
            state = np.concatenate([state, pad], axis=-1)
        elif state.shape[-1] > self.robot_state_dim:
            state = state[:, : self.robot_state_dim]
        return state

    def _load_map4d_from_traj(self, traj: h5py.Group) -> Optional[np.ndarray]:
        for path in ("obs/map4d", "obs/rep", "map4d"):
            value = _read_dataset(traj, path)
            if value is not None:
                return np.asarray(value, dtype=np.float32)
        if self.actor_names is None or "env_states" not in traj or "actors" not in traj["env_states"]:
            return None
        actors = traj["env_states"]["actors"]
        missing = [name for name in self.actor_names if name not in actors]
        if missing:
            return None
        sizes = TASK_GT_SIZES.get(self.task_name)
        actor_states = [actors[name][()] for name in self.actor_names]
        if self.map4d_dim == 9:
            return _actor_states_to_pose_map4d_tensor(actor_states)
        return _actor_states_to_map4d_tensor(actor_states, sizes=sizes)

    def _load_structural_parameters(self, sidecar_record=None):
        if sidecar_record is None:
            raise ValueError("keyframe_sidecar_path is required to load size_parameters/relation_parameters")
        if "size_parameters" not in sidecar_record:
            raise KeyError("Sidecar record missing size_parameters")
        if "relation_parameters" not in sidecar_record:
            raise KeyError("Sidecar record missing relation_parameters")
        size_parameters = np.asarray(sidecar_record["size_parameters"], dtype=np.float32).reshape(-1)
        relation_parameters = np.asarray(sidecar_record["relation_parameters"], dtype=np.float32).reshape(-1)
        if size_parameters.shape[0] != self.size_parameter_dim:
            raise ValueError(
                f"Expected size_parameter_dim={self.size_parameter_dim}, got {size_parameters.shape[0]}"
            )
        if relation_parameters.shape[0] != self.relation_parameter_dim:
            raise ValueError(
                "Expected relation_parameter_dim="
                f"{self.relation_parameter_dim}, got {relation_parameters.shape[0]}"
            )
        return size_parameters.astype(np.float32), relation_parameters.astype(np.float32)

    def _load_actions(self, traj: h5py.Group) -> np.ndarray:
        actions = np.asarray(traj["actions"][()], dtype=np.float32)
        if actions.shape[-1] >= 8:
            trajectory = actions[:, :7]
            gripper = actions[:, -1:]
        elif actions.shape[-1] == 7:
            # pd_ee_delta_pose: (delta_pos[3], axis_angle[3], gripper[1])
            trajectory = np.concatenate(
                [actions[:, :3], _axis_angle_to_quaternion_np(actions[:, 3:6])],
                axis=-1,
            )
            gripper = actions[:, 6:7]
        elif actions.shape[-1] == 4:
            identity = np.broadcast_to(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (actions.shape[0], 4),
            )
            trajectory = np.concatenate([actions[:, :3], identity], axis=-1)
            gripper = actions[:, 3:4]
        else:
            raise ValueError(
                f"Expected action dim >=8, 7, or 4, got {actions.shape[-1]}"
            )
        trajectory[:, 3:7] = canonicalize_quaternion_np(trajectory[:, 3:7])
        return np.concatenate([trajectory, gripper], axis=-1).astype(np.float32)

    def _load_sidecar(self, path: Optional[str], traj_names: Sequence[str]):
        if path is None:
            return None
        sidecar = {}
        with h5py.File(path, "r") as f:
            target_format = f.attrs.get("target_format", "")
            if isinstance(target_format, bytes):
                target_format = target_format.decode("utf-8")
            compatible_formats = {
                MAP4D_DIT_TARGET_FORMAT,
                MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT,
            }
            if self.strict_target_format and target_format not in compatible_formats:
                raise ValueError(
                    f"Keyframe sidecar target_format={target_format!r} is not compatible with "
                    f"{sorted(compatible_formats)}."
                )
            for traj_name in traj_names:
                if traj_name not in f:
                    raise KeyError(f"Sidecar {path} missing group {traj_name}")
                group = f[traj_name]
                record = {key: group[key][()] for key in group.keys() if isinstance(group[key], h5py.Dataset)}
                sidecar[traj_name] = record
        return sidecar

    def _load_real_trajectories(
        self,
        demo_path: str,
        keyframe_sidecar_path: Optional[str],
        num_traj: Optional[int],
    ):
        trajectories = []
        with h5py.File(demo_path, "r") as f:
            traj_names = sorted([key for key in f.keys() if key.startswith("traj_")], key=_traj_sort_key)
            if num_traj is not None:
                traj_names = traj_names[:num_traj]
            sidecar = self._load_sidecar(keyframe_sidecar_path, traj_names)
            for traj_name in traj_names:
                traj = f[traj_name]
                actions = self._load_actions(traj)
                seq_len = actions.shape[0] + 1
                robot_state = self._load_robot_state(traj, seq_len)
                if sidecar is not None and "map4d" in sidecar[traj_name]:
                    map4d = np.asarray(sidecar[traj_name]["map4d"], dtype=np.float32)
                else:
                    map4d = self._load_map4d_from_traj(traj)
                if map4d is None:
                    raise KeyError(
                        f"{traj_name} has no map4d data and no loadable ManiSkill actor states."
                    )
                if map4d.shape[0] != seq_len:
                    raise ValueError(f"{traj_name}: map4d length {map4d.shape[0]} != actions length + 1 {seq_len}")
                if map4d.shape[1] != self.num_objects:
                    raise ValueError(f"{traj_name}: expected {self.num_objects} objects, got {map4d.shape[1]}")
                if map4d.shape[-1] != self.map4d_dim:
                    raise ValueError(
                        f"{traj_name}: expected map4d_dim={self.map4d_dim}, got {map4d.shape[-1]}"
                    )
                size_parameters, relation_parameters = self._load_structural_parameters(
                    sidecar[traj_name] if sidecar is not None else None
                )
                rgb_feature = None
                point_cloud = None
                dino_feature = None
                if self.use_rgb:
                    point_cloud = self._load_point_cloud(traj, seq_len)
                    dino_feature = self._load_dino_feature(traj, seq_len, point_cloud.shape[1])

                if sidecar is None:
                    keyframes = np.arange(seq_len, dtype=np.int64)
                    future_table = build_future_keyframe_table(
                        keyframes, num_frames=seq_len, horizon=self.horizon_keyframe
                    )
                    tcp_pose = self._load_tcp_pose(traj, seq_len)
                    keyframe_object, keyframe_tcp = gather_map4d_dit_keyframe_targets(
                        map4d, tcp_pose, future_table
                    )
                else:
                    record = sidecar[traj_name]
                    if (
                        "future_keyframe_object_targets" in record
                        and "future_keyframe_tcp_pose" in record
                    ):
                        keyframe_object = record["future_keyframe_object_targets"].astype(np.float32)
                        keyframe_tcp = record["future_keyframe_tcp_pose"].astype(np.float32)
                    else:
                        future_table = record["future_keyframe_indices"].astype(np.int64)
                        tcp_pose = record.get("tcp_pose")
                        if tcp_pose is None:
                            tcp_pose = self._load_tcp_pose(traj, seq_len)
                        keyframe_object, keyframe_tcp = gather_map4d_dit_keyframe_targets(
                            map4d, tcp_pose, future_table
                        )
                if self.keyframe_tcp_dim is not None and keyframe_tcp.shape[-1] != self.keyframe_tcp_dim:
                    raise ValueError(
                        f"{traj_name}: expected keyframe_tcp_dim={self.keyframe_tcp_dim}, "
                        f"got {keyframe_tcp.shape[-1]}"
                    )
                record = {
                    "robot_state": robot_state,
                    "actions": actions,
                    "map4d": map4d.astype(np.float32),
                    "size_parameters": size_parameters,
                    "relation_parameters": relation_parameters,
                    "keyframe_object": keyframe_object.astype(np.float32),
                    "keyframe_tcp": keyframe_tcp.astype(np.float32),
                }
                if rgb_feature is not None:
                    record["rgb_feature"] = rgb_feature.astype(np.float32)
                if point_cloud is not None:
                    record["point_cloud"] = point_cloud.astype(np.float32)
                if dino_feature is not None:
                    record["dino_feature"] = dino_feature.astype(np.float32)
                trajectories.append(record)
        if not trajectories:
            raise ValueError(f"No traj_* groups found in {demo_path}")
        return trajectories

    def _load_tcp_pose(self, traj: h5py.Group, length: int) -> np.ndarray:
        tcp_pose = _read_dataset(traj, "obs/extra/tcp_pose")
        if tcp_pose is None:
            tcp_pose = np.zeros((length, 7), dtype=np.float32)
            tcp_pose[:, 3] = 1.0
        tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
        if tcp_pose.shape[0] != length or tcp_pose.shape[-1] < 7:
            raise ValueError(f"Expected TCP pose shape [{length}, >=7], got {tcp_pose.shape}")
        tcp_pose = tcp_pose[:, :7]
        tcp_pose[:, 3:7] = canonicalize_quaternion_np(tcp_pose[:, 3:7])
        return tcp_pose

    def _build_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            action_len = traj["actions"].shape[0]
            for start in range(action_len):
                indices.append((traj_idx, start))
        return indices

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = list(self.val_indices)
        val_set.val_indices = []
        return val_set

    def get_normalizer(self, mode="limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        robot_state = np.concatenate([traj["robot_state"] for traj in self.trajectories], axis=0)
        node_poses = np.concatenate([traj["map4d"] for traj in self.trajectories], axis=0)
        size_parameters = np.stack([traj["size_parameters"] for traj in self.trajectories], axis=0)
        trajectory_pos = np.concatenate([traj["actions"][:, 0:3] for traj in self.trajectories], axis=0)
        gripper = np.concatenate([traj["actions"][:, 7:8] for traj in self.trajectories], axis=0)
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
            "node_poses": node_poses,
            "size_parameters": size_parameters,
            "trajectory_pos": trajectory_pos,
            "gripper_openness": gripper,
            "keyframe_map4d_pos": keyframe_map4d_pos,
            "keyframe_tcp_pos": keyframe_tcp_pos,
        }
        if self.trajectories[0]["keyframe_tcp"].shape[-1] == 4:
            fields["keyframe_tcp_gripper"] = np.concatenate(
                [traj["keyframe_tcp"][..., 3:4].reshape(-1, 1) for traj in self.trajectories],
                axis=0,
            )
        if self.relation_parameter_dim > 0:
            fields["relation_parameters"] = np.stack(
                [traj["relation_parameters"] for traj in self.trajectories],
                axis=0,
            )
        normalizer.fit(
            fields,
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([traj["actions"] for traj in self.trajectories], axis=0))

    def __len__(self) -> int:
        return len(self.indices)

    def _slice_with_pad(self, array: np.ndarray, start: int, horizon: int) -> np.ndarray:
        indices = np.arange(start, start + horizon)
        indices = np.clip(indices, 0, array.shape[0] - 1)
        return array[indices]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]
        current_idx = start
        obs_start = current_idx - self.n_obs_steps + 1
        action_seq = self._slice_with_pad(traj["actions"], start, self.horizon_action)
        sample = {
            "obs": {
                "robot_state": self._slice_with_pad(traj["robot_state"], obs_start, self.n_obs_steps),
                "node_poses": self._slice_with_pad(traj["map4d"], obs_start, self.n_obs_steps),
                "size_parameters": traj["size_parameters"],
                "relation_parameters": traj["relation_parameters"],
            },
            "action": {
                "trajectory": action_seq[:, 0:7],
                "gripper_openness": action_seq[:, 7:8],
            },
            "keyframe": {
                "map4d": traj["keyframe_object"][current_idx, : self.horizon_keyframe],
                "tcp": traj["keyframe_tcp"][current_idx, : self.horizon_keyframe],
            },
        }
        if self.use_rgb and "rgb_feature" in traj:
            sample["obs"]["rgb_feature"] = self._slice_with_pad(
                traj["rgb_feature"], obs_start, self.n_obs_steps
            )
        if self.use_rgb:
            sample["obs"]["point_cloud"] = self._slice_with_pad(
                traj["point_cloud"], obs_start, self.n_obs_steps
            )
            sample["obs"]["dino_feature"] = self._slice_with_pad(
                traj["dino_feature"], obs_start, self.n_obs_steps
            )
        return _to_tensor_tree(sample)


def _to_tensor_tree(value):
    if isinstance(value, dict):
        return {key: _to_tensor_tree(item) for key, item in value.items()}
    return torch.from_numpy(np.asarray(value, dtype=np.float32))
