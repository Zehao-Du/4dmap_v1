"""Build sidecar data for Raw Concat + Keyframe Future Loss + TCP targets.

The output HDF5 is aligned with ManiSkill demonstration trajectories. Each
`traj_*` group contains PerAct-style keyframes, future-keyframe lookup indices,
per-frame map4d tensors, TCP targets, and optionally materialized training targets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np

from map4d.representation.maps4d.metadata import (
    TASK_METADATA_FILES,
    get_task_actor_names,
    get_task_parameter_defaults,
)

try:
    from helper.extract_keyframes import extract_traj_keyframes
    from helper.keyframe_targets import (
        MAP4D_DIT_TARGET_FORMAT,
        MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT,
        build_future_keyframe_table,
        gather_keyframe_targets,
        gather_map4d_dit_keyframe_targets,
        gather_map4d_dit_pos_gripper_keyframe_targets,
    )
except ModuleNotFoundError:
    from extract_keyframes import extract_traj_keyframes  # type: ignore
    from keyframe_targets import (  # type: ignore
        MAP4D_DIT_TARGET_FORMAT,
        MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT,
        build_future_keyframe_table,
        gather_keyframe_targets,
        gather_map4d_dit_keyframe_targets,
        gather_map4d_dit_pos_gripper_keyframe_targets,
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
    "PlugCharger-v1": (
        0.04,
        0.03,
        0.024,
        0.02,
        0.1,
        0.1,
    ),
}

TASK_SIZE_PARAMETER_DEFAULTS = {
    # Map4d_StackCube: red cube(3) + green cube(3) + desk(3)
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
    # Map4d_PlugCharger: charger body(3) + charger prong(3) + receptacle(8).
    # Receptacle decomposes into center-divider(3) + face-loop(5).
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
    # Map4d_PlugCharger: half-gap between two charger prongs.
    "PlugCharger-v1": (0.007,),
}


def _traj_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def _infer_task_name(path: str) -> str:
    for task_name in sorted(set(TASK_ACTOR_NAMES) | set(TASK_METADATA_FILES)):
        if task_name in path:
            return task_name
    raise ValueError(
        "Could not infer task name from path. Pass --task-name explicitly."
    )


def _default_actor_names(task_name: str) -> Tuple[str, ...]:
    try:
        actor_names = get_task_actor_names(task_name)
        if actor_names:
            return actor_names
    except KeyError:
        pass
    if task_name not in TASK_ACTOR_NAMES:
        raise ValueError(f"No default actor names for task {task_name}")
    return TASK_ACTOR_NAMES[task_name]


def _parse_actor_names(value: Optional[str], task_name: str) -> Tuple[str, ...]:
    if value:
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return _default_actor_names(task_name)


def _quat_wxyz_to_rotation_6d_np(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape [T, 4], got {quat.shape}")
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1e-8)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    matrix = np.empty((quat.shape[0], 3, 3), dtype=np.float32)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - z * w)
    matrix[:, 0, 2] = 2.0 * (x * z + y * w)
    matrix[:, 1, 0] = 2.0 * (x * y + z * w)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - x * w)
    matrix[:, 2, 0] = 2.0 * (x * z - y * w)
    matrix[:, 2, 1] = 2.0 * (y * z + x * w)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return np.concatenate([matrix[:, :, 0], matrix[:, :, 1]], axis=1).astype(np.float32)


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
    if sizes is not None:
        sizes_np = np.asarray(tuple(sizes), dtype=np.float32).reshape(num_objects, 3)
    else:
        sizes_np = np.zeros((num_objects, 3), dtype=np.float32)
    sizes_seq = np.broadcast_to(sizes_np, (frame_count, num_objects, 3))
    positions = np.stack([state[:, 0:3] for state in states], axis=1)
    rotations = np.stack(
        [_quat_wxyz_to_rotation_6d_np(state[:, 3:7]) for state in states],
        axis=1,
    )
    return np.concatenate([sizes_seq, positions, rotations], axis=-1).astype(np.float32)


def _actor_states_to_pose_map4d_tensor(actor_states: Sequence[np.ndarray]) -> np.ndarray:
    states = [np.asarray(state, dtype=np.float32) for state in actor_states]
    frame_count = states[0].shape[0]
    if any(state.shape[0] != frame_count for state in states):
        raise ValueError("All actor state arrays must have the same frame count.")
    positions = np.stack([state[:, 0:3] for state in states], axis=1)
    rotations = np.stack(
        [_quat_wxyz_to_rotation_6d_np(state[:, 3:7]) for state in states],
        axis=1,
    )
    return np.concatenate([positions, rotations], axis=-1).astype(np.float32)


def _load_map4d_from_traj(
    traj: h5py.Group,
    actor_names: Sequence[str],
    sizes: Optional[Iterable[float]],
) -> np.ndarray:
    actors = traj["env_states"]["actors"]
    missing = [name for name in actor_names if name not in actors]
    if missing:
        raise KeyError(f"{traj.name} missing actors: {missing}")
    actor_states = [actors[name][()] for name in actor_names]
    return _actor_states_to_map4d_tensor(actor_states, sizes=sizes)


def _load_pose_map4d_from_traj(
    traj: h5py.Group,
    actor_names: Sequence[str],
) -> np.ndarray:
    actors = traj["env_states"]["actors"]
    missing = [name for name in actor_names if name not in actors]
    if missing:
        raise KeyError(f"{traj.name} missing actors: {missing}")
    actor_states = [actors[name][()] for name in actor_names]
    return _actor_states_to_pose_map4d_tensor(actor_states)


def _task_parameters(task_name: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        size_defaults, relation_defaults = get_task_parameter_defaults(task_name)
        return (
            np.asarray(size_defaults, dtype=np.float32),
            np.asarray(relation_defaults, dtype=np.float32),
        )
    except KeyError:
        pass
    size_parameters = np.asarray(
        TASK_SIZE_PARAMETER_DEFAULTS.get(task_name, tuple()),
        dtype=np.float32,
    )
    relation_parameters = np.asarray(
        TASK_RELATION_PARAMETER_DEFAULTS.get(task_name, tuple()),
        dtype=np.float32,
    )
    return size_parameters, relation_parameters


def _write_dataset(group: h5py.Group, name: str, value: np.ndarray) -> None:
    arr = np.asarray(value)
    kwargs = {"compression": "gzip"} if arr.size > 0 else {}
    group.create_dataset(name, data=arr, **kwargs)


def _load_gripper_target(traj: h5py.Group, length: int) -> np.ndarray:
    if "actions" in traj:
        actions = np.asarray(traj["actions"][()], dtype=np.float32)
        if actions.ndim == 2 and actions.shape[0] == length - 1 and actions.shape[1] >= 1:
            gripper = actions[:, -1:]
            return np.concatenate([gripper[:1], gripper], axis=0).astype(np.float32)

    qpos = traj.get("obs", {}).get("agent", {}).get("qpos") if "obs" in traj else None
    if isinstance(qpos, h5py.Dataset):
        qpos_arr = np.asarray(qpos[()], dtype=np.float32)
        if qpos_arr.ndim == 2 and qpos_arr.shape[0] == length and qpos_arr.shape[1] >= 2:
            return qpos_arr[:, -2:].mean(axis=1, keepdims=True).astype(np.float32)

    return np.zeros((length, 1), dtype=np.float32)


def build_keyframe_aux_dataset(
    demo_path: str,
    output_path: str,
    *,
    task_name: str,
    actor_names: Sequence[str],
    future_horizon: int,
    num_traj: Optional[int] = None,
    gripper_source: str = "auto",
    stopping_delta: float = 0.1,
    min_separation: int = 1,
    materialize_targets: bool = True,
    tcp_target: str = "pose",
    target_format: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    try:
        from helper.peract_keyframes import KeyframeConfig
    except ModuleNotFoundError:
        from peract_keyframes import KeyframeConfig  # type: ignore

    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = KeyframeConfig(
        stopping_delta=stopping_delta,
        min_separation=min_separation,
    )
    sizes = TASK_GT_SIZES.get(task_name)
    summary_rows = []

    with h5py.File(demo_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        traj_names = sorted([k for k in f_in.keys() if k.startswith("traj_")], key=_traj_sort_key)
        if num_traj is not None:
            traj_names = traj_names[:num_traj]

        f_out.attrs["source_demo_path"] = demo_path
        f_out.attrs["task_name"] = task_name
        f_out.attrs["actor_names"] = json.dumps(list(actor_names))
        f_out.attrs["future_horizon"] = int(future_horizon)
        if tcp_target not in {"pose", "pos", "pos_gripper"}:
            raise ValueError(f"tcp_target must be 'pose', 'pos', or 'pos_gripper', got {tcp_target!r}")
        if target_format is None:
            if tcp_target == "pose":
                target_format = "object_delta_pos_rot6d_plus_tcp_pose"
            elif tcp_target == "pos":
                target_format = "object_delta_pos_rot6d_plus_tcp_pos"
            else:
                target_format = "object_delta_pos_rot6d_plus_tcp_pos_gripper"
        if target_format not in {
            MAP4D_DIT_TARGET_FORMAT,
            MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT,
            "object_delta_pos_rot6d_plus_tcp_pose",
            "object_delta_pos_rot6d_plus_tcp_pos",
            "object_delta_pos_rot6d_plus_tcp_pos_gripper",
        }:
            raise ValueError(f"Unsupported target_format={target_format!r}")
        if target_format == MAP4D_DIT_TARGET_FORMAT and tcp_target != "pose":
            raise ValueError(f"{MAP4D_DIT_TARGET_FORMAT} requires tcp_target='pose'")
        if target_format == MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT and tcp_target != "pos_gripper":
            raise ValueError(
                f"{MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT} requires tcp_target='pos_gripper'"
            )
        f_out.attrs["tcp_target"] = tcp_target
        f_out.attrs["tcp_dim"] = 7 if tcp_target == "pose" else 4 if tcp_target == "pos_gripper" else 3
        f_out.attrs["target_format"] = target_format

        for traj_name in traj_names:
            traj = f_in[traj_name]
            if target_format in {MAP4D_DIT_TARGET_FORMAT, MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT}:
                map4d = _load_pose_map4d_from_traj(traj, actor_names)
                size_parameters, relation_parameters = _task_parameters(task_name)
                group_map4d_dim = 9
            else:
                map4d = _load_map4d_from_traj(traj, actor_names, sizes)
                size_parameters, relation_parameters = _task_parameters(task_name)
                group_map4d_dim = 12
            tcp_pose = traj["obs"]["extra"]["tcp_pose"][()].astype(np.float32)
            if tcp_pose.shape[0] != map4d.shape[0]:
                raise ValueError(
                    f"{traj_name}: tcp_pose length {tcp_pose.shape[0]} != map4d length {map4d.shape[0]}"
                )
            gripper_target = _load_gripper_target(traj, map4d.shape[0])
            if tcp_target == "pose":
                tcp_target_seq = tcp_pose
            elif tcp_target == "pos":
                tcp_target_seq = tcp_pose[:, :3]
            else:
                tcp_target_seq = np.concatenate([tcp_pose[:, :3], gripper_target], axis=-1)

            keyframe_data = extract_traj_keyframes(
                traj,
                gripper_source=gripper_source,
                config=config,
            )
            keyframes = keyframe_data["keyframe_indices"].astype(np.int64)
            future_table = build_future_keyframe_table(
                keyframes,
                num_frames=map4d.shape[0],
                horizon=future_horizon,
            )

            group = f_out.create_group(traj_name)
            group.attrs["num_frames"] = int(map4d.shape[0])
            group.attrs["num_keyframes"] = int(len(keyframes))
            group.attrs["tcp_target"] = tcp_target
            group.attrs["map4d_dim"] = int(group_map4d_dim)
            group.attrs["size_parameter_dim"] = int(size_parameters.shape[0])
            group.attrs["relation_parameter_dim"] = int(relation_parameters.shape[0])
            _write_dataset(group, "map4d", map4d)
            _write_dataset(group, "size_parameters", size_parameters)
            _write_dataset(group, "relation_parameters", relation_parameters)
            _write_dataset(group, "tcp_pose", tcp_pose)
            if tcp_target == "pos":
                _write_dataset(group, "tcp_pos", tcp_target_seq)
            elif tcp_target == "pos_gripper":
                _write_dataset(group, "tcp_pos_gripper", tcp_target_seq)
                _write_dataset(group, "gripper_target", gripper_target)
            _write_dataset(group, "keyframe_indices", keyframes)
            _write_dataset(group, "keyframe_tcp_pose", tcp_target_seq[keyframes])
            _write_dataset(group, "future_keyframe_indices", future_table)

            if materialize_targets:
                if target_format == MAP4D_DIT_TARGET_FORMAT:
                    object_targets, tcp_targets = gather_map4d_dit_keyframe_targets(
                        map4d,
                        tcp_pose,
                        future_table,
                    )
                elif target_format == MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT:
                    object_targets, tcp_targets = gather_map4d_dit_pos_gripper_keyframe_targets(
                        map4d,
                        tcp_pose,
                        gripper_target,
                        future_table,
                    )
                else:
                    object_targets, tcp_targets = gather_keyframe_targets(
                        map4d,
                        tcp_target_seq,
                        future_table,
                    )
                _write_dataset(group, "future_keyframe_object_targets", object_targets)
                _write_dataset(group, "future_keyframe_tcp_pose", tcp_targets)

            summary_rows.append(
                {
                    "traj": traj_name,
                    "num_frames": int(map4d.shape[0]),
                    "num_keyframes": int(len(keyframes)),
                    "first_keyframes": keyframes[:10].tolist(),
                    "last_keyframe": int(keyframes[-1]) if len(keyframes) else None,
                }
            )

    counts = [row["num_keyframes"] for row in summary_rows]
    return {
        "demo_path": demo_path,
        "output_path": output_path,
        "task_name": task_name,
        "actor_names": list(actor_names),
        "future_horizon": future_horizon,
        "tcp_target": tcp_target,
        "target_format": target_format,
        "tcp_dim": 7 if tcp_target == "pose" else 4 if tcp_target == "pos_gripper" else 3,
        "map4d_dim": 9
        if target_format in {MAP4D_DIT_TARGET_FORMAT, MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT}
        else 12,
        "size_parameter_dim": int(_task_parameters(task_name)[0].shape[0]),
        "relation_parameter_dim": int(_task_parameters(task_name)[1].shape[0]),
        "materialize_targets": materialize_targets,
        "num_trajectories": len(summary_rows),
        "min_keyframes": int(np.min(counts)) if counts else 0,
        "mean_keyframes": float(np.mean(counts)) if counts else 0.0,
        "max_keyframes": int(np.max(counts)) if counts else 0,
        "trajectories": summary_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--actor-names", default=None, help="Comma-separated actor names; defaults by task.")
    parser.add_argument("--future-horizon", type=int, default=4)
    parser.add_argument("--num-traj", type=int, default=None)
    parser.add_argument("--gripper-source", choices=["auto", "qpos", "action"], default="auto")
    parser.add_argument("--stopping-delta", type=float, default=0.1)
    parser.add_argument("--min-separation", type=int, default=1)
    parser.add_argument(
        "--tcp-target",
        choices=["pose", "pos", "pos_gripper"],
        default="pose",
        help=(
            "Use full TCP pose (7D), TCP position only (3D), or "
            "TCP position plus gripper (4D) for future keyframe targets."
        ),
    )
    parser.add_argument(
        "--target-format",
        default=None,
        choices=[
            MAP4D_DIT_TARGET_FORMAT,
            MAP4D_DIT_TCP_POS_GRIPPER_TARGET_FORMAT,
            "object_delta_pos_rot6d_plus_tcp_pose",
            "object_delta_pos_rot6d_plus_tcp_pos",
            "object_delta_pos_rot6d_plus_tcp_pos_gripper",
        ],
        help=(
            "Materialized keyframe target convention. Defaults to the legacy "
            "ACT/DP format matching --tcp-target; pass "
            f"{MAP4D_DIT_TARGET_FORMAT} for Map4D DiT."
        ),
    )
    parser.add_argument("--no-materialize-targets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_name = args.task_name or _infer_task_name(args.demo_path)
    actor_names = _parse_actor_names(args.actor_names, task_name)
    summary = build_keyframe_aux_dataset(
        args.demo_path,
        args.output_path,
        task_name=task_name,
        actor_names=actor_names,
        future_horizon=args.future_horizon,
        num_traj=args.num_traj,
        gripper_source=args.gripper_source,
        stopping_delta=args.stopping_delta,
        min_separation=args.min_separation,
        tcp_target=args.tcp_target,
        target_format=args.target_format,
        materialize_targets=not args.no_materialize_targets,
        overwrite=args.overwrite,
    )

    summary_path = Path(args.summary_json) if args.summary_json else Path(args.output_path).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "Built {output_path}; trajectories={num_trajectories}, keyframes min/mean/max="
        "{min_keyframes}/{mean_keyframes:.2f}/{max_keyframes}".format(**summary)
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
