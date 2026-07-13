#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPS4D_DIR = PROJECT_ROOT / "map4d" / "representation" / "maps4d"
if str(MAPS4D_DIR) not in sys.path:
    sys.path.insert(0, str(MAPS4D_DIR))

from rlbench2_push_box import Map4d_RLBench2PushBox  # noqa: E402


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _cuboid_vertices(center: np.ndarray, size: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    half = np.asarray(size, dtype=np.float64) / 2.0
    corners = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    return corners @ _quat_xyzw_to_matrix(quat_xyzw).T + center


def _add_cuboid(ax, vertices: np.ndarray, color: str, alpha: float, label: str) -> None:
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]
    poly = Poly3DCollection(faces, facecolors=color, edgecolors="#222222", linewidths=0.8, alpha=alpha)
    ax.add_collection3d(poly)
    center = vertices.mean(axis=0)
    ax.text(center[0], center[1], center[2] + 0.09, label, fontsize=10, ha="center", color="#222222")


def _set_equal_axes(ax, points: np.ndarray, padding: float = 0.16) -> None:
    mins = points.min(axis=0) - padding
    maxs = points.max(axis=0) + padding
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)


def _load_demo_pose_sequence(low_dim_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with low_dim_path.open("rb") as f:
        demo = pickle.load(f)
    states = np.concatenate([np.asarray(obs.task_low_dim_state, dtype=np.float64).reshape(1, -1) for obs in demo], axis=0)
    # RLBench object poses are xyz + quaternion xyzw in the task low-dimensional state.
    return states[:, 0:3], states[:, 3:7]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low-dim",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "rlbench2" / "squashfs-root" / "all_variations" / "episodes" / "episode0" / "low_dim_obs.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "visualizations" / "rlbench2_push_box_representation.png",
    )
    parser.add_argument("--size", type=float, nargs=3, default=(0.12, 0.12, 0.12), metavar=("L", "H", "W"))
    args = parser.parse_args()

    positions, quat_xyzw = _load_demo_pose_sequence(args.low_dim)
    size = np.asarray(args.size, dtype=np.float64)
    initial_pos = positions[0]
    initial_quat_xyzw = quat_xyzw[0]
    final_pos = positions[-1]
    final_quat_xyzw = quat_xyzw[-1]

    # Instantiate the actual Map4D representation class. It expects quaternion wxyz.
    quat_wxyz = np.concatenate([initial_quat_xyzw[3:4], initial_quat_xyzw[:3]], axis=0)
    rep = Map4d_RLBench2PushBox(
        positions=torch.tensor(initial_pos[None, :], dtype=torch.float32),
        rotations=torch.tensor(quat_wxyz[None, :], dtype=torch.float32),
        size_parameters=torch.tensor(size[None, :], dtype=torch.float32),
        relation_parameters=torch.empty((1, 0), dtype=torch.float32),
    )

    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("RLBench2 Push Box Map4D Representation", pad=18)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    initial_vertices = _cuboid_vertices(initial_pos, size, initial_quat_xyzw)
    final_vertices = _cuboid_vertices(final_pos, size, final_quat_xyzw)
    _add_cuboid(ax, initial_vertices, "#4C78A8", 0.55, "push box node")
    _add_cuboid(ax, final_vertices, "#F58518", 0.22, "episode final")

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#E45756", linewidth=2.0, label="episode0 box path")
    ax.scatter(initial_pos[0], initial_pos[1], initial_pos[2], s=40, color="#4C78A8", label="initial pose")
    ax.scatter(final_pos[0], final_pos[1], final_pos[2], s=40, color="#F58518", label="final pose")

    all_points = np.concatenate([positions, initial_vertices, final_vertices], axis=0)
    _set_equal_axes(ax, all_points)
    ax.view_init(elev=24, azim=-56)
    ax.legend(loc="upper left")

    text = (
        f"Representation: {rep.representation_name}\n"
        f"Objects: {len(rep.Objects)}  Nodes: {rep.N}  Edges: {rep.M}\n"
        f"Node semantic: {rep.Nodes[0].Node_Semantic}\n"
        f"Size parameters [L,H,W]: {size.tolist()}"
    )
    fig.text(0.03, 0.03, text, fontsize=10, family="monospace", va="bottom")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
