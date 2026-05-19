from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

from map4d.construction.foundationpose import FoundationPoseLoader


DATA_ROOT = (
    REPO_ROOT.parent
    / "dataset"
    / "ManiSkill"
    / "StackCube-v1"
    / "motionplanning"
)


def default_stackcube_dataset() -> pathlib.Path:
    candidates = [
        DATA_ROOT / "StackCube.rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.h5",
        DATA_ROOT / "StackCube.rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.ep00002_00002.h5",
        DATA_ROOT / "StackCube.rgb.pd_ee_delta_pose.physx_cpu.h5",
        DATA_ROOT / "StackCube.h5",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


DEFAULT_DATASET = default_stackcube_dataset()


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def actor_state_to_world_pose(actor_state: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = actor_state[:3].astype(np.float32)
    pose[:3, :3] = quat_wxyz_to_mat(actor_state[3:7]).astype(np.float32)
    return pose


def make_w2c_4x4(extrinsic_cv: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :4] = extrinsic_cv.astype(np.float32)
    return T


def project_point(K: np.ndarray, w2c: np.ndarray, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz1 = np.concatenate([xyz_world.astype(np.float32), np.ones(1, dtype=np.float32)])
    xyz_cam = w2c @ xyz1
    uvw = K @ xyz_cam[:3]
    uv = uvw[:2] / uvw[2]
    return uv, xyz_cam[:3]


def infer_segmentation_id(seg: np.ndarray, uv: np.ndarray, radius: int = 4) -> int:
    h, w = seg.shape
    u = int(np.clip(round(float(uv[0])), 0, w - 1))
    v = int(np.clip(round(float(uv[1])), 0, h - 1))
    patch = seg[max(0, v - radius) : min(h, v + radius + 1), max(0, u - radius) : min(w, u + radius + 1)]
    vals, counts = np.unique(patch, return_counts=True)
    order = np.argsort(counts)[::-1]
    for idx in order:
        val = int(vals[idx])
        if val > 0:
            return val
    return int(seg[v, u])


def pose_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    r = pred[:3, :3] @ gt[:3, :3].T
    trace = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    rot_deg = math.degrees(math.acos(trace))
    trans_m = float(np.linalg.norm(pred[:3, 3] - gt[:3, 3]))
    return {
        "translation_error_m": trans_m,
        "rotation_error_deg": rot_deg,
    }


def symmetry_aware_yaw_error_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    def yaw_z(T: np.ndarray) -> float:
        return math.degrees(math.atan2(float(T[1, 0]), float(T[0, 0])))

    raw = yaw_z(pred) - yaw_z(gt)
    raw = (raw + 180.0) % 360.0 - 180.0
    cands = [abs(raw - k * 90.0) for k in range(-4, 5)]
    return min((((c + 180.0) % 360.0) - 180.0) for c in cands)


def save_mask_png(path: pathlib.Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return _jsonable(asdict(x))
    if isinstance(x, pathlib.Path):
        return str(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FoundationPose on a StackCube dataset frame.")
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET)
    parser.add_argument("--traj", type=str, default="traj_0")
    parser.add_argument("--camera", type=str, default="base_camera", choices=["base_camera", "hand_camera"])
    parser.add_argument("--frame", type=int, default=10)
    parser.add_argument("--cube-size", type=float, default=0.04, help="Cube side length in meters.")
    parser.add_argument("--refine-iter", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "outputs" / "foundationpose_stackcube_validation_latest",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = FoundationPoseLoader(device=args.device)
    validation = loader.validate(check_runtime=True)

    mesh = trimesh.creation.box(extents=(args.cube_size, args.cube_size, args.cube_size))
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mesh.faces = np.asarray(mesh.faces, dtype=np.int64)
    _ = mesh.vertex_normals
    mesh.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    from Utils import draw_posed_3d_box, draw_xyz_axis

    glctx = loader.create_glctx()

    with h5py.File(dataset, "r") as f:
        base = f[args.traj]
        rgb = base[f"obs/sensor_data/{args.camera}/rgb"][args.frame].astype(np.uint8)
        depth_mm = base[f"obs/sensor_data/{args.camera}/depth"][args.frame, ..., 0].astype(np.int32)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        seg = base[f"obs/sensor_data/{args.camera}/segmentation"][args.frame, ..., 0].astype(np.int32)
        K = base[f"obs/sensor_param/{args.camera}/intrinsic_cv"][args.frame].astype(np.float32)
        w2c = make_w2c_4x4(base[f"obs/sensor_param/{args.camera}/extrinsic_cv"][args.frame])

        results: dict[str, Any] = {}
        overlay_gt = rgb.copy()
        overlay_pred = rgb.copy()

        for cube_name in ("cubeA", "cubeB"):
            cube_dir = output_dir / cube_name
            cube_dir.mkdir(parents=True, exist_ok=True)

            actor_state = base[f"env_states/actors/{cube_name}"][args.frame].astype(np.float32)
            world_pose = actor_state_to_world_pose(actor_state)
            gt_pose = w2c @ world_pose

            uv, xyz_cam = project_point(K, w2c, actor_state[:3])
            seg_id = infer_segmentation_id(seg, uv)
            mask = seg == seg_id

            estimator = loader.load_estimator(mesh=mesh, debug=2, debug_dir=cube_dir, glctx=glctx)
            pred_pose = estimator.register(
                K=K,
                rgb=rgb,
                depth=depth_m,
                ob_mask=mask.astype(np.uint8),
                iteration=args.refine_iter,
            )

            bbox = np.array(
                [
                    [-args.cube_size / 2, -args.cube_size / 2, -args.cube_size / 2],
                    [args.cube_size / 2, args.cube_size / 2, args.cube_size / 2],
                ],
                dtype=np.float32,
            )

            overlay_gt = draw_posed_3d_box(K, img=overlay_gt, ob_in_cam=gt_pose, bbox=bbox)
            overlay_gt = draw_xyz_axis(overlay_gt, ob_in_cam=gt_pose, scale=args.cube_size * 1.2, K=K, thickness=2, transparency=0, is_input_rgb=True)
            overlay_pred = draw_posed_3d_box(K, img=overlay_pred, ob_in_cam=pred_pose, bbox=bbox)
            overlay_pred = draw_xyz_axis(overlay_pred, ob_in_cam=pred_pose, scale=args.cube_size * 1.2, K=K, thickness=2, transparency=0, is_input_rgb=True)

            imageio.imwrite(cube_dir / "rgb.png", rgb)
            cv2.imwrite(str(cube_dir / "depth_mm.png"), depth_mm.astype(np.uint16))
            save_mask_png(cube_dir / "mask.png", mask)

            metrics = pose_metrics(pred_pose, gt_pose)
            metrics["cube_symmetry_aware_yaw_error_deg_mod_90"] = float(abs(symmetry_aware_yaw_error_deg(pred_pose, gt_pose)))
            metrics["mask_pixels"] = int(mask.sum())
            metrics["segmentation_id"] = int(seg_id)
            metrics["projected_center_uv"] = [float(uv[0]), float(uv[1])]
            metrics["center_in_camera_m"] = [float(v) for v in xyz_cam]
            metrics["gt_pose"] = gt_pose
            metrics["pred_pose"] = pred_pose
            results[cube_name] = metrics

    imageio.imwrite(output_dir / "frame_rgb.png", rgb)
    cv2.imwrite(str(output_dir / "frame_depth_mm.png"), depth_mm.astype(np.uint16))
    imageio.imwrite(output_dir / "overlay_gt_both.png", overlay_gt)
    imageio.imwrite(output_dir / "overlay_pred_both.png", overlay_pred)

    summary = {
        "dataset": dataset,
        "traj": args.traj,
        "camera": args.camera,
        "frame": args.frame,
        "cube_size_m": args.cube_size,
        "validation": validation,
        "inputs_available": {
            "rgb": True,
            "depth": True,
            "segmentation": True,
            "camera_intrinsics": True,
            "camera_extrinsics": True,
            "object_mesh": True,
            "object_gt_pose": True,
        },
        "notes": [
            "Masks were derived from the dataset segmentation by auto-identifying the segmentation id at the GT-projected object center.",
            "Cube geometry was supplied as a canonical 4 cm box mesh built in-script, matching ManiSkill StackCube actor dimensions.",
            "Direct rotation error is not fully meaningful for cubes because of geometric symmetry; a modulo-90 yaw error is also reported.",
        ],
        "results": results,
        "outputs": {
            "frame_rgb": output_dir / "frame_rgb.png",
            "frame_depth_mm": output_dir / "frame_depth_mm.png",
            "overlay_gt_both": output_dir / "overlay_gt_both.png",
            "overlay_pred_both": output_dir / "overlay_pred_both.png",
            "cubeA_dir": output_dir / "cubeA",
            "cubeB_dir": output_dir / "cubeB",
        },
    }

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)

    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
