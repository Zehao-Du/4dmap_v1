"""PerAct-style keyframe discovery for array-based demonstrations.

The original PerAct heuristic marks keyframes when the gripper state changes,
when the robot stops after motion, and at the final frame. This module keeps the
same logic but works on ManiSkill-style arrays instead of RLBench Demo objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class KeyframeConfig:
    stopping_delta: float = 0.1
    stopped_buffer_frames: int = 4
    qpos_gripper_open_threshold: float = 0.02
    min_separation: int = 1
    include_final: bool = True


def infer_gripper_open(
    *,
    actions: Optional[np.ndarray] = None,
    qpos: Optional[np.ndarray] = None,
    source: str = "auto",
    qpos_open_threshold: float = 0.02,
) -> np.ndarray:
    """Infer boolean gripper-open state for each observation frame.

    `qpos` is preferred in `auto` mode because it reflects the realized gripper
    state. If using actions, the last action dimension is treated as the gripper
    command and padded to observation length `len(actions) + 1`.
    """
    if source not in {"auto", "qpos", "action"}:
        raise ValueError(f"Unsupported gripper source: {source}")

    if source in {"auto", "qpos"} and qpos is not None and qpos.ndim == 2 and qpos.shape[1] >= 2:
        finger_width = qpos[:, -2:].mean(axis=1)
        return finger_width > qpos_open_threshold

    if source in {"auto", "action"} and actions is not None:
        action_grip = np.asarray(actions)[:, -1]
        gripper_open = action_grip > 0
        return np.concatenate([gripper_open[:1], gripper_open], axis=0)

    raise ValueError("Cannot infer gripper state: provide qpos or actions.")


def infer_joint_velocities(
    *,
    qvel: Optional[np.ndarray] = None,
    actions: Optional[np.ndarray] = None,
    tcp_pose: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a per-observation velocity-like array for stopped detection."""
    if qvel is not None:
        return np.asarray(qvel)

    if actions is not None:
        act = np.asarray(actions)
        arm_act = act[:, :-1] if act.shape[1] > 1 else act
        return np.concatenate([arm_act[:1], arm_act], axis=0)

    if tcp_pose is not None:
        tcp = np.asarray(tcp_pose)
        diff = np.diff(tcp[:, :3], axis=0, prepend=tcp[:1, :3])
        return diff

    raise ValueError("Cannot infer motion: provide qvel, actions, or tcp_pose.")


def _is_stopped(
    gripper_open: np.ndarray,
    velocities: np.ndarray,
    i: int,
    stopped_buffer: int,
    stopping_delta: float,
) -> bool:
    n = len(gripper_open)
    if n < 4 or i < 2 or i >= n - 1:
        return False

    next_is_not_final = i == (n - 2)
    gripper_state_no_change = (
        i < (n - 2)
        and gripper_open[i] == gripper_open[i + 1]
        and gripper_open[i] == gripper_open[i - 1]
        and gripper_open[i - 2] == gripper_open[i - 1]
    )
    small_delta = np.allclose(velocities[i], 0.0, atol=stopping_delta)
    return bool(
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )


def deduplicate_keyframes(indices: Iterable[int], min_separation: int = 1) -> List[int]:
    """Sort and remove keyframes that are too close to the previous one."""
    ordered = sorted({int(i) for i in indices})
    if min_separation <= 1:
        return ordered

    deduped: List[int] = []
    for idx in ordered:
        if not deduped or idx - deduped[-1] >= min_separation:
            deduped.append(idx)
        else:
            deduped[-1] = idx
    return deduped


def discover_keyframes(
    gripper_open: np.ndarray,
    velocities: np.ndarray,
    config: KeyframeConfig = KeyframeConfig(),
) -> List[int]:
    """Discover PerAct-style keyframe indices.

    The returned indices refer to observation frames, not action rows.
    """
    gripper_open = np.asarray(gripper_open).astype(bool)
    velocities = np.asarray(velocities)
    if len(gripper_open) != len(velocities):
        raise ValueError(
            f"gripper_open length {len(gripper_open)} != velocities length {len(velocities)}"
        )
    if len(gripper_open) == 0:
        return []

    keyframes: List[int] = []
    prev_gripper_open = gripper_open[0]
    stopped_buffer = 0

    for i in range(len(gripper_open)):
        stopped = _is_stopped(
            gripper_open,
            velocities,
            i,
            stopped_buffer,
            config.stopping_delta,
        )
        stopped_buffer = config.stopped_buffer_frames if stopped else stopped_buffer - 1
        last = i == (len(gripper_open) - 1)
        if i != 0 and (
            gripper_open[i] != prev_gripper_open
            or (config.include_final and last)
            or stopped
        ):
            keyframes.append(i)
        prev_gripper_open = gripper_open[i]

    if len(keyframes) > 1 and (keyframes[-1] - 1) == keyframes[-2]:
        keyframes.pop(-2)

    return deduplicate_keyframes(keyframes, config.min_separation)


def future_keyframes(
    keyframes: Iterable[int],
    current_idx: int,
    horizon: int,
    fallback_idx: Optional[int] = None,
) -> np.ndarray:
    """Return the next `horizon` keyframes after `current_idx`, padded by repeat."""
    if horizon <= 0:
        return np.zeros((0,), dtype=np.int64)

    future = [int(i) for i in keyframes if int(i) > current_idx]
    if future:
        selected = future[:horizon]
        selected.extend([selected[-1]] * (horizon - len(selected)))
    else:
        fallback = current_idx if fallback_idx is None else fallback_idx
        selected = [int(fallback)] * horizon
    return np.asarray(selected, dtype=np.int64)

