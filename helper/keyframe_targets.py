"""Utilities for building future-keyframe auxiliary targets."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from helper.peract_keyframes import future_keyframes
except ModuleNotFoundError:
    from peract_keyframes import future_keyframes  # type: ignore


MAP4D_DIT_TARGET_FORMAT = "map4d_dit_local_delta_relative_rotation_v1"
MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT = (
    "map4d_dit_local_delta_relative_rotation_tcp_pos_gripper_v1"
)


def canonicalize_quaternion_np(quat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize WXYZ quaternions and choose the `w >= 0` representative."""
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, eps, None)
    sign = np.where(quat[..., :1] < 0.0, -1.0, 1.0).astype(np.float32)
    return (quat * sign).astype(np.float32)


def quat_conjugate_np(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply WXYZ quaternions with numpy broadcasting."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternions to rotation matrices."""
    quat = canonicalize_quaternion_np(quat)
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
    return matrix


def matrix_to_rot6d_np(matrix: np.ndarray) -> np.ndarray:
    """Use the first two rotation-matrix columns as 6D rotation."""
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1).astype(np.float32)


def rot6d_to_matrix_np(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert 6D rotation representation to an orthonormal matrix."""
    rot6d = np.asarray(rot6d, dtype=np.float32)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), eps, None)
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_orth / np.clip(np.linalg.norm(a2_orth, axis=-1, keepdims=True), eps, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def relative_quaternion_np(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Return canonical relative WXYZ quaternion from current to future."""
    current = canonicalize_quaternion_np(current)
    future = canonicalize_quaternion_np(future)
    return canonicalize_quaternion_np(quat_multiply_np(quat_conjugate_np(current), future))


def build_future_keyframe_table(
    keyframe_indices: Iterable[int],
    num_frames: int,
    horizon: int,
) -> np.ndarray:
    """Build `[num_frames, horizon]` future-keyframe indices with repeat padding."""
    table = np.zeros((num_frames, horizon), dtype=np.int64)
    for current_idx in range(num_frames):
        table[current_idx] = future_keyframes(
            keyframe_indices,
            current_idx=current_idx,
            horizon=horizon,
            fallback_idx=current_idx,
        )
    return table


def gather_keyframe_targets(
    map4d: np.ndarray,
    tcp_pose: np.ndarray,
    future_keyframe_table: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather object-state and TCP targets for keyframe future loss.

    Args:
        map4d: `[T, N, 12]`, with `size(3), position(3), rotation_6d(6)`.
        tcp_pose: `[T, tcp_dim]`.
        future_keyframe_table: `[T, H]` or `[B, H]` indices into frame time.

    Returns:
        object_targets: `[T, H, N, 9]`, where each target is
            `future_delta_pos(3) + future_rotation_6d(6)`.
        tcp_targets: `[T, H, tcp_dim]`.
    """
    map4d = np.asarray(map4d, dtype=np.float32)
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
    indices = np.asarray(future_keyframe_table, dtype=np.int64)

    if map4d.ndim != 3 or map4d.shape[-1] < 12:
        raise ValueError(f"Expected map4d shape [T, N, >=12], got {map4d.shape}")
    if tcp_pose.ndim != 2 or tcp_pose.shape[0] != map4d.shape[0]:
        raise ValueError(
            f"Expected tcp_pose shape [T, tcp_dim] with T={map4d.shape[0]}, got {tcp_pose.shape}"
        )
    if indices.ndim != 2 or indices.shape[0] != map4d.shape[0]:
        raise ValueError(
            f"Expected future_keyframe_table shape [T, H] with T={map4d.shape[0]}, got {indices.shape}"
        )

    indices = np.clip(indices, 0, map4d.shape[0] - 1)
    future_map4d = map4d[indices]
    current_pos = map4d[:, None, :, 3:6]
    future_delta_pos = future_map4d[..., 3:6] - current_pos
    future_rot_6d = future_map4d[..., 6:12]
    object_targets = np.concatenate([future_delta_pos, future_rot_6d], axis=-1)
    tcp_targets = tcp_pose[indices]
    return object_targets.astype(np.float32), tcp_targets.astype(np.float32)


def _pose_map4d_parts(map4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return object positions and rot6d from supported Map4D tensor layouts."""
    map4d = np.asarray(map4d, dtype=np.float32)
    if map4d.shape[-1] == 9:
        return map4d[..., 0:3], map4d[..., 3:9]
    if map4d.shape[-1] >= 12:
        return map4d[..., 3:6], map4d[..., 6:12]
    raise ValueError(f"Expected Map4D last dim 9 or >=12, got {map4d.shape}")


def gather_map4d_dit_keyframe_targets(
    map4d: np.ndarray,
    tcp_pose: np.ndarray,
    future_keyframe_table: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather Map4D DiT local-delta and relative-rotation keyframe targets.

    Args:
        map4d: `[T, N, 9]` pose-only with `position(3), rotation_6d(6)`,
            or legacy `[T, N, 12]` with `size(3), position(3), rotation_6d(6)`.
        tcp_pose: `[T, 7]`, with TCP `position(3), quaternion_wxyz(4)`.
        future_keyframe_table: `[T, H]` indices into frame time.

    Returns:
        object_targets: `[T, H, N, 9]`, each target is
            `local_delta_pos(3) + relative_rot6d(6)`.
        tcp_targets: `[T, H, 7]`, each target is
            `local_delta_pos(3) + delta_quat_wxyz(4)`.
    """
    map4d = np.asarray(map4d, dtype=np.float32)
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
    indices = np.asarray(future_keyframe_table, dtype=np.int64)

    if map4d.ndim != 3 or map4d.shape[-1] not in {9, 12}:
        raise ValueError(f"Expected map4d shape [T, N, 9] or [T, N, 12], got {map4d.shape}")
    if tcp_pose.ndim != 2 or tcp_pose.shape[-1] < 7 or tcp_pose.shape[0] != map4d.shape[0]:
        raise ValueError(
            f"Expected tcp_pose shape [T, >=7] with T={map4d.shape[0]}, got {tcp_pose.shape}"
        )
    if indices.ndim != 2 or indices.shape[0] != map4d.shape[0]:
        raise ValueError(
            f"Expected future_keyframe_table shape [T, H] with T={map4d.shape[0]}, got {indices.shape}"
        )

    indices = np.clip(indices, 0, map4d.shape[0] - 1)
    future_map4d = map4d[indices]

    object_pos, object_rot6d = _pose_map4d_parts(map4d)
    future_obj_pos, future_obj_rot6d = _pose_map4d_parts(future_map4d)
    current_obj_pos = object_pos[:, None]
    current_obj_rot = rot6d_to_matrix_np(object_rot6d[:, None])
    future_obj_rot = rot6d_to_matrix_np(future_obj_rot6d)
    object_local_delta_pos = np.einsum(
        "...ji,...j->...i",
        current_obj_rot,
        future_obj_pos - current_obj_pos,
    )
    object_relative_rot = np.einsum("...ji,...jk->...ik", current_obj_rot, future_obj_rot)
    object_relative_rot6d = matrix_to_rot6d_np(object_relative_rot)
    object_targets = np.concatenate(
        [object_local_delta_pos, object_relative_rot6d], axis=-1
    )

    future_tcp = tcp_pose[indices]
    current_tcp_pos = tcp_pose[:, None, 0:3]
    current_tcp_quat = canonicalize_quaternion_np(tcp_pose[:, None, 3:7])
    future_tcp_pos = future_tcp[..., 0:3]
    future_tcp_quat = canonicalize_quaternion_np(future_tcp[..., 3:7])
    current_tcp_rot = quat_to_matrix_np(current_tcp_quat)
    tcp_local_delta_pos = np.einsum(
        "...ji,...j->...i",
        current_tcp_rot,
        future_tcp_pos - current_tcp_pos,
    )
    tcp_delta_quat = relative_quaternion_np(current_tcp_quat, future_tcp_quat)
    tcp_targets = np.concatenate([tcp_local_delta_pos, tcp_delta_quat], axis=-1)
    return object_targets.astype(np.float32), tcp_targets.astype(np.float32)


def gather_map4d_dit_pos_gripper_keyframe_targets(
    map4d: np.ndarray,
    tcp_pose: np.ndarray,
    gripper: np.ndarray,
    future_keyframe_table: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather Map4D DiT targets with 4D TCP `local_delta_pos + gripper`.

    This is used for StackCube-style `pd_ee_delta_pos` debugging, where the
    robot target should stay aligned with the 4D action space.
    """
    map4d = np.asarray(map4d, dtype=np.float32)
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1, 1)
    indices = np.asarray(future_keyframe_table, dtype=np.int64)

    if map4d.ndim != 3 or map4d.shape[-1] not in {9, 12}:
        raise ValueError(f"Expected map4d shape [T, N, 9] or [T, N, 12], got {map4d.shape}")
    if tcp_pose.ndim != 2 or tcp_pose.shape[-1] < 7 or tcp_pose.shape[0] != map4d.shape[0]:
        raise ValueError(
            f"Expected tcp_pose shape [T, >=7] with T={map4d.shape[0]}, got {tcp_pose.shape}"
        )
    if gripper.shape[0] != map4d.shape[0]:
        raise ValueError(f"Expected gripper length {map4d.shape[0]}, got {gripper.shape[0]}")
    if indices.ndim != 2 or indices.shape[0] != map4d.shape[0]:
        raise ValueError(
            f"Expected future_keyframe_table shape [T, H] with T={map4d.shape[0]}, got {indices.shape}"
        )

    indices = np.clip(indices, 0, map4d.shape[0] - 1)
    future_map4d = map4d[indices]

    object_pos, object_rot6d = _pose_map4d_parts(map4d)
    future_obj_pos, future_obj_rot6d = _pose_map4d_parts(future_map4d)
    current_obj_pos = object_pos[:, None]
    current_obj_rot = rot6d_to_matrix_np(object_rot6d[:, None])
    future_obj_rot = rot6d_to_matrix_np(future_obj_rot6d)
    object_local_delta_pos = np.einsum(
        "...ji,...j->...i",
        current_obj_rot,
        future_obj_pos - current_obj_pos,
    )
    object_relative_rot = np.einsum("...ji,...jk->...ik", current_obj_rot, future_obj_rot)
    object_relative_rot6d = matrix_to_rot6d_np(object_relative_rot)
    object_targets = np.concatenate(
        [object_local_delta_pos, object_relative_rot6d], axis=-1
    )

    future_tcp = tcp_pose[indices]
    current_tcp_pos = tcp_pose[:, None, 0:3]
    current_tcp_quat = canonicalize_quaternion_np(tcp_pose[:, None, 3:7])
    future_tcp_pos = future_tcp[..., 0:3]
    current_tcp_rot = quat_to_matrix_np(current_tcp_quat)
    tcp_local_delta_pos = np.einsum(
        "...ji,...j->...i",
        current_tcp_rot,
        future_tcp_pos - current_tcp_pos,
    )
    tcp_targets = np.concatenate([tcp_local_delta_pos, gripper[indices]], axis=-1)
    return object_targets.astype(np.float32), tcp_targets.astype(np.float32)


def build_targets_for_frame(
    map4d: np.ndarray,
    tcp_pose: np.ndarray,
    keyframe_indices: Iterable[int],
    current_idx: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build future keyframe indices and targets for one current frame."""
    indices = future_keyframes(
        keyframe_indices,
        current_idx=current_idx,
        horizon=horizon,
        fallback_idx=current_idx,
    )
    table = np.zeros((len(map4d), horizon), dtype=np.int64)
    table[current_idx] = indices
    object_targets, tcp_targets = gather_keyframe_targets(map4d, tcp_pose, table)
    return indices, object_targets[current_idx], tcp_targets[current_idx]


def build_map4d_dit_targets_for_frame(
    map4d: np.ndarray,
    tcp_pose: np.ndarray,
    keyframe_indices: Iterable[int],
    current_idx: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Map4D DiT future keyframe indices and targets for one frame."""
    indices = future_keyframes(
        keyframe_indices,
        current_idx=current_idx,
        horizon=horizon,
        fallback_idx=current_idx,
    )
    table = np.zeros((len(map4d), horizon), dtype=np.int64)
    table[current_idx] = indices
    object_targets, tcp_targets = gather_map4d_dit_keyframe_targets(map4d, tcp_pose, table)
    return indices, object_targets[current_idx], tcp_targets[current_idx]
