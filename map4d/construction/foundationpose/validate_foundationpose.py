from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import asdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

from map4d.construction.foundationpose import FoundationPoseLoader


def _to_jsonable(obj):
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _default_mesh_path(repo_root: pathlib.Path) -> pathlib.Path | None:
    candidates = [
        repo_root / "third_party" / "RLBench" / "rlbench" / "assets" / "procedural_objects" / "705" / "705.obj",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _rotation_error_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    r = pred[:3, :3] @ gt[:3, :3].T
    trace = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def _translation_error_m(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred[:3, 3] - gt[:3, 3]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[3])
    parser.add_argument("--mesh", type=pathlib.Path, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent / "validation_outputs" / "latest")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=600.0)
    parser.add_argument("--fy", type=float, default=600.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--refine-iter", type=int, default=3)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = FoundationPoseLoader(device=args.device)
    validation = loader.validate(check_runtime=True)

    scorer = loader.load_scorer()
    refiner = loader.load_refiner()

    mesh_path = args.mesh.resolve() if args.mesh is not None else _default_mesh_path(repo_root)
    if mesh_path is None:
        mesh = trimesh.creation.box(extents=(0.12, 0.12, 0.12))
    else:
        mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mesh.faces = np.asarray(mesh.faces, dtype=np.int64)
    extents = np.asarray(mesh.extents, dtype=np.float32)
    max_extent = float(np.max(extents)) if extents.size else 1.0
    if max_extent > 0:
        mesh.apply_scale(0.12 / max_extent)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    _ = mesh.vertex_normals
    mesh.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    estimator = loader.load_estimator(mesh=mesh, debug=2, debug_dir=output_dir)

    from Utils import (
        draw_posed_3d_box,
        draw_xyz_axis,
        glcam_in_cvcam,
        make_mesh_tensors,
        nvdiffrast_render,
    )

    K = np.array(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    gt_pose = np.eye(4, dtype=np.float32)
    gt_pose[:3, :3] = np.array(
        [
            [0.81379765, -0.54383814, -0.20487413],
            [0.4698463, 0.8231729, -0.3187958],
            [0.34202015, 0.16317591, 0.9254166],
        ],
        dtype=np.float32,
    )
    gt_pose[:3, 3] = np.array([0.02, -0.01, 0.85], dtype=np.float32)

    glctx = loader.create_glctx()
    color_t, depth_t, _ = nvdiffrast_render(
        K=K,
        H=args.height,
        W=args.width,
        ob_in_cams=torch.as_tensor(gt_pose, device=args.device, dtype=torch.float32)[None],
        glctx=glctx,
        mesh_tensors=make_mesh_tensors(mesh, device=args.device),
    )
    rgb = (color_t[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    depth = depth_t[0].detach().cpu().numpy().astype(np.float32)
    mask = depth > 1e-4

    pred_pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask.astype(np.uint8), iteration=args.refine_iter)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
    gt_center_pose = gt_pose @ np.linalg.inv(to_origin)
    pred_center_pose = pred_pose @ np.linalg.inv(to_origin)

    overlay_pred = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=pred_center_pose, bbox=bbox)
    overlay_pred = draw_xyz_axis(overlay_pred, ob_in_cam=pred_center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
    overlay_gt = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=gt_center_pose, bbox=bbox)
    overlay_gt = draw_xyz_axis(overlay_gt, ob_in_cam=gt_center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)

    imageio.imwrite(output_dir / "synthetic_rgb.png", rgb)
    cv2.imwrite(str(output_dir / "synthetic_depth_mm.png"), (depth * 1000.0).astype(np.uint16))
    cv2.imwrite(str(output_dir / "synthetic_mask.png"), (mask.astype(np.uint8) * 255))
    imageio.imwrite(output_dir / "overlay_gt.png", overlay_gt)
    imageio.imwrite(output_dir / "overlay_pred.png", overlay_pred)

    report = {
        "validation": _to_jsonable(asdict(validation)),
        "mesh_path": str(mesh_path),
        "output_dir": str(output_dir),
        "synthetic_rgb": str(output_dir / "synthetic_rgb.png"),
        "synthetic_depth": str(output_dir / "synthetic_depth_mm.png"),
        "synthetic_mask": str(output_dir / "synthetic_mask.png"),
        "overlay_gt": str(output_dir / "overlay_gt.png"),
        "overlay_pred": str(output_dir / "overlay_pred.png"),
        "estimator_debug_files": [
            str(output_dir / "color.png"),
            str(output_dir / "depth.png"),
            str(output_dir / "ob_mask.png"),
            str(output_dir / "scene_raw.ply"),
            str(output_dir / "scene_complete.ply"),
            str(output_dir / "vis_refiner.png"),
            str(output_dir / "vis_score.png"),
        ],
        "gt_pose": gt_pose.tolist(),
        "pred_pose": pred_pose.tolist(),
        "translation_error_m": _translation_error_m(pred_pose, gt_pose),
        "rotation_error_deg": _rotation_error_deg(pred_pose, gt_pose),
        "weights_loaded": True,
        "forward_inference_ran": True,
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
