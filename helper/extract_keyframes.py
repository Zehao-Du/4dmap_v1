"""Extract PerAct-style keyframes from ManiSkill HDF5 demonstrations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np

try:
    from helper.peract_keyframes import (
        KeyframeConfig,
        discover_keyframes,
        infer_gripper_open,
        infer_joint_velocities,
    )
except ModuleNotFoundError:
    from peract_keyframes import (  # type: ignore
        KeyframeConfig,
        discover_keyframes,
        infer_gripper_open,
        infer_joint_velocities,
    )


def _traj_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def _optional_dataset(group: h5py.Group, path: str) -> Optional[np.ndarray]:
    if path in group:
        return group[path][()]
    return None


def extract_traj_keyframes(
    traj: h5py.Group,
    *,
    gripper_source: str = "auto",
    config: KeyframeConfig = KeyframeConfig(),
) -> Dict[str, np.ndarray]:
    """Extract keyframe indices and TCP poses for one `traj_*` group."""
    actions = _optional_dataset(traj, "actions")
    qpos = _optional_dataset(traj, "obs/agent/qpos")
    qvel = _optional_dataset(traj, "obs/agent/qvel")
    tcp_pose = _optional_dataset(traj, "obs/extra/tcp_pose")

    gripper_open = infer_gripper_open(
        actions=actions,
        qpos=qpos,
        source=gripper_source,
        qpos_open_threshold=config.qpos_gripper_open_threshold,
    )
    velocities = infer_joint_velocities(qvel=qvel, actions=actions, tcp_pose=tcp_pose)
    keyframes = np.asarray(discover_keyframes(gripper_open, velocities, config), dtype=np.int64)

    result: Dict[str, np.ndarray] = {
        "keyframe_indices": keyframes,
        "gripper_open": gripper_open.astype(np.bool_),
    }
    if tcp_pose is not None:
        result["keyframe_tcp_pose"] = tcp_pose[keyframes]
    if actions is not None:
        result["length_actions"] = np.asarray(actions.shape[0], dtype=np.int64)
    result["length_observations"] = np.asarray(len(gripper_open), dtype=np.int64)
    return result


def extract_file_keyframes(
    demo_path: str,
    *,
    num_traj: Optional[int] = None,
    gripper_source: str = "auto",
    config: KeyframeConfig = KeyframeConfig(),
) -> Dict[str, object]:
    """Extract keyframes for every trajectory in a ManiSkill demo HDF5 file."""
    rows: List[Dict[str, object]] = []
    with h5py.File(demo_path, "r") as f:
        traj_names = sorted([k for k in f.keys() if k.startswith("traj_")], key=_traj_sort_key)
        if num_traj is not None:
            traj_names = traj_names[:num_traj]

        for name in traj_names:
            extracted = extract_traj_keyframes(
                f[name],
                gripper_source=gripper_source,
                config=config,
            )
            keyframes = extracted["keyframe_indices"]
            tcp = extracted.get("keyframe_tcp_pose")
            rows.append(
                {
                    "traj": name,
                    "num_keyframes": int(len(keyframes)),
                    "length_actions": int(extracted.get("length_actions", -1)),
                    "length_observations": int(extracted["length_observations"]),
                    "keyframe_indices": keyframes.tolist(),
                    "keyframe_tcp_pose": tcp.tolist() if tcp is not None else None,
                }
            )

    counts = [int(row["num_keyframes"]) for row in rows]
    return {
        "demo_path": demo_path,
        "num_trajectories": len(rows),
        "mean_keyframes": float(np.mean(counts)) if counts else 0.0,
        "min_keyframes": int(np.min(counts)) if counts else 0,
        "max_keyframes": int(np.max(counts)) if counts else 0,
        "trajectories": rows,
    }


def save_npz(summary: Dict[str, object], output_path: str) -> None:
    """Save padded keyframe index and TCP arrays for fast loading."""
    rows = summary["trajectories"]
    assert isinstance(rows, list)
    max_keyframes = max((len(row["keyframe_indices"]) for row in rows), default=0)
    traj_names = np.asarray([row["traj"] for row in rows], dtype=object)
    indices = np.full((len(rows), max_keyframes), -1, dtype=np.int64)

    tcp_dim = 0
    for row in rows:
        tcp = row.get("keyframe_tcp_pose")
        if tcp:
            tcp_dim = len(tcp[0])
            break
    tcp_pose = np.full((len(rows), max_keyframes, tcp_dim), np.nan, dtype=np.float32)

    lengths = np.zeros((len(rows),), dtype=np.int64)
    for i, row in enumerate(rows):
        kf = np.asarray(row["keyframe_indices"], dtype=np.int64)
        lengths[i] = len(kf)
        indices[i, : len(kf)] = kf
        tcp = row.get("keyframe_tcp_pose")
        if tcp is not None and tcp_dim > 0:
            tcp_arr = np.asarray(tcp, dtype=np.float32)
            tcp_pose[i, : tcp_arr.shape[0], : tcp_arr.shape[1]] = tcp_arr

    np.savez_compressed(
        output_path,
        traj_names=traj_names,
        keyframe_indices=indices,
        keyframe_lengths=lengths,
        keyframe_tcp_pose=tcp_pose,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", required=True, help="Path to ManiSkill .h5 demonstration file.")
    parser.add_argument("--output-dir", default=None, help="Directory for keyframes.json and keyframes.npz.")
    parser.add_argument("--output-json", default=None, help="Optional explicit JSON output path.")
    parser.add_argument("--output-npz", default=None, help="Optional explicit NPZ output path.")
    parser.add_argument("--num-traj", type=int, default=None, help="Limit number of trajectories.")
    parser.add_argument("--gripper-source", choices=["auto", "qpos", "action"], default="auto")
    parser.add_argument("--stopping-delta", type=float, default=0.1)
    parser.add_argument("--stopped-buffer-frames", type=int, default=4)
    parser.add_argument("--qpos-gripper-open-threshold", type=float, default=0.02)
    parser.add_argument("--min-separation", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = KeyframeConfig(
        stopping_delta=args.stopping_delta,
        stopped_buffer_frames=args.stopped_buffer_frames,
        qpos_gripper_open_threshold=args.qpos_gripper_open_threshold,
        min_separation=args.min_separation,
    )
    summary = extract_file_keyframes(
        args.demo_path,
        num_traj=args.num_traj,
        gripper_source=args.gripper_source,
        config=config,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.demo_path).with_suffix("").parent / "keyframes"
    output_json = Path(args.output_json) if args.output_json else output_dir / "keyframes.json"
    output_npz = Path(args.output_npz) if args.output_npz else output_dir / "keyframes.npz"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_npz(summary, str(output_npz))

    print(
        "Extracted {num_trajectories} trajectories; keyframes min/mean/max = "
        "{min_keyframes}/{mean_keyframes:.2f}/{max_keyframes}".format(**summary)
    )
    print(f"Wrote {output_json}")
    print(f"Wrote {output_npz}")


if __name__ == "__main__":
    main()
