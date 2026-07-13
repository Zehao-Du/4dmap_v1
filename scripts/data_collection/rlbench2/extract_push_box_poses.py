#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "rlbench2" / "bimanual_push_box.train.squashfs"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "rlbench2" / "bimanual_push_box_train_poses.npz"
DEFAULT_CSV = PROJECT_ROOT / "dataset" / "rlbench2" / "bimanual_push_box_train_poses.csv"
BOX_SIZE_XYZ = np.array([0.192, 0.384, 0.128], dtype=np.float32)


def _episode_index(path: Path | str) -> int:
    match = re.search(r"episode(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def _list_low_dim_files_from_squashfs(squashfs: Path) -> list[str]:
    out = subprocess.check_output(["unsquashfs", "-lc", str(squashfs)], text=True)
    files = []
    for line in out.splitlines():
        file_path = line.strip()
        if not file_path.endswith("/low_dim_obs.pkl"):
            continue
        if file_path.startswith("squashfs-root/"):
            file_path = file_path[len("squashfs-root/") :]
        files.append(file_path)
    return sorted(files, key=_episode_index)


def _extract_low_dim_files(squashfs: Path, files: list[str], dest: Path) -> Path:
    extract_list = dest / "extract_files.txt"
    extract_list.write_text("\n".join(files) + "\n")
    subprocess.check_call(
        [
            "unsquashfs",
            "-q",
            "-n",
            "-f",
            "-d",
            str(dest),
            "-extract-file",
            str(extract_list),
            str(squashfs),
        ]
    )
    return dest


def _low_dim_files_from_dir(root: Path) -> list[Path]:
    files = sorted(root.glob("**/all_variations/episodes/episode*/low_dim_obs.pkl"), key=_episode_index)
    if not files:
        files = sorted(root.glob("all_variations/episodes/episode*/low_dim_obs.pkl"), key=_episode_index)
    return files


def _load_pose_sequence(low_dim_path: Path) -> np.ndarray:
    with low_dim_path.open("rb") as f:
        demo = pickle.load(f)
    poses = []
    for obs in demo:
        state = np.asarray(obs.task_low_dim_state, dtype=np.float32).reshape(-1)
        poses.append(state[:7])
    return np.stack(poses, axis=0)


def _collect(low_dim_files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_ids = []
    frame_ids = []
    poses = []
    for low_dim in tqdm(low_dim_files, desc="extract push-box poses", dynamic_ncols=True):
        ep = _episode_index(low_dim)
        seq = _load_pose_sequence(low_dim)
        episode_ids.extend([ep] * len(seq))
        frame_ids.extend(range(len(seq)))
        poses.append(seq)
    return (
        np.asarray(episode_ids, dtype=np.int32),
        np.asarray(frame_ids, dtype=np.int32),
        np.concatenate(poses, axis=0).astype(np.float32),
    )


def _write_csv(path: Path, episode_ids: np.ndarray, frame_ids: np.ndarray, poses_xyzw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "size_x", "size_y", "size_z"])
        rows = zip(episode_ids, frame_ids, poses_xyzw)
        rows = tqdm(rows, total=len(frame_ids), desc="write push-box pose csv", dynamic_ncols=True)
        for ep, frame, pose in rows:
            writer.writerow([int(ep), int(frame), *map(float, pose), *map(float, BOX_SIZE_XYZ)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-frame push-box poses for Map4D from RLBench2 low_dim_obs.pkl.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="A .squashfs file or extracted RLBench2 task root.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--keep-low-dim-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.input.suffix == ".squashfs":
        files = _list_low_dim_files_from_squashfs(args.input)
        if not files:
            raise FileNotFoundError(f"No low_dim_obs.pkl found in {args.input}")
        if args.keep_low_dim_dir is None:
            with tempfile.TemporaryDirectory(prefix="rlbench2_low_dim_") as tmp:
                low_dim_root = _extract_low_dim_files(args.input, files, Path(tmp))
                low_dim_files = _low_dim_files_from_dir(low_dim_root)
                episode_ids, frame_ids, poses_xyzw = _collect(low_dim_files)
        else:
            args.keep_low_dim_dir.mkdir(parents=True, exist_ok=True)
            low_dim_root = _extract_low_dim_files(args.input, files, args.keep_low_dim_dir)
            low_dim_files = _low_dim_files_from_dir(low_dim_root)
            episode_ids, frame_ids, poses_xyzw = _collect(low_dim_files)
    else:
        low_dim_files = _low_dim_files_from_dir(args.input)
        if not low_dim_files:
            raise FileNotFoundError(f"No low_dim_obs.pkl found under {args.input}")
        episode_ids, frame_ids, poses_xyzw = _collect(low_dim_files)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        episode=episode_ids,
        frame=frame_ids,
        pose_xyzw=poses_xyzw,
        position=poses_xyzw[:, :3],
        quaternion_xyzw=poses_xyzw[:, 3:7],
        size_xyz=BOX_SIZE_XYZ,
    )
    _write_csv(args.csv, episode_ids, frame_ids, poses_xyzw)
    print(f"episodes={len(np.unique(episode_ids))} frames={len(frame_ids)}")
    print(args.output)
    print(args.csv)


if __name__ == "__main__":
    main()
