#!/usr/bin/env python
"""Build PPI-style point clouds inside ManiSkill RGB-D demonstration HDF5 files.

The script reads per-camera RGB, depth, and camera parameters from ManiSkill
trajectories and writes:

    traj_*/obs/point_cloud/<camera>     [T, points_per_camera, 6]  xyzrgb
    traj_*/obs/point_cloud/fused        [T, num_points, 6]         xyzrgb
    traj_*/obs/point_cloud_source/fused/camera_index [T, num_points]
    traj_*/obs/point_cloud_source/fused/pixel_uv     [T, num_points, 2]
    traj_*/obs/tcp_trajectory/pose      [T, 7]
    traj_*/obs/tcp_trajectory/pos       [T, 3]

Point clouds follow the PPI preprocessing style: depth images are back-projected
to world coordinates, filtered by a workspace bounding box, and randomly sampled
to a fixed number of points.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np


DEFAULT_CAMERAS = ("base_camera", "hand_camera")
DEFAULT_BBOX = {
    "StackCube-v1": ((-0.8, -0.6, -0.05), (0.5, 0.6, 0.8)),
    "PlugCharger-v1": ((-0.8, -0.6, -0.05), (0.5, 0.6, 0.8)),
}


def _traj_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def _parse_cameras(value: str) -> Optional[Tuple[str, ...]]:
    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_bbox(value: str, task_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if value.strip().lower() == "auto":
        mins, maxs = DEFAULT_BBOX.get(task_name, DEFAULT_BBOX["StackCube-v1"])
        return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)

    parts = [float(x) for x in value.replace(",", " ").split()]
    if len(parts) != 6:
        raise ValueError("--bbox must be 'auto' or six numbers: xmin ymin zmin xmax ymax zmax")
    mins = np.asarray(parts[:3], dtype=np.float32)
    maxs = np.asarray(parts[3:], dtype=np.float32)
    if np.any(maxs <= mins):
        raise ValueError(f"Invalid bbox min/max: {mins} {maxs}")
    return mins, maxs


def _discover_cameras(traj: h5py.Group, requested: Optional[Sequence[str]]) -> List[str]:
    sensor_data = traj.get("obs", {}).get("sensor_data") if "obs" in traj else None
    sensor_param = traj.get("obs", {}).get("sensor_param") if "obs" in traj else None
    if not isinstance(sensor_data, h5py.Group) or not isinstance(sensor_param, h5py.Group):
        raise KeyError(f"{traj.name} must contain obs/sensor_data and obs/sensor_param")

    if requested is not None:
        cameras = list(requested)
    else:
        cameras = [name for name in DEFAULT_CAMERAS if name in sensor_data]
        if not cameras:
            cameras = sorted(sensor_data.keys())

    missing = []
    for camera in cameras:
        if (
            camera not in sensor_data
            or "rgb" not in sensor_data[camera]
            or "depth" not in sensor_data[camera]
            or camera not in sensor_param
            or "intrinsic_cv" not in sensor_param[camera]
            or "extrinsic_cv" not in sensor_param[camera]
        ):
            missing.append(camera)
    if missing:
        raise KeyError(f"{traj.name} missing RGB-D or camera params for cameras: {missing}")
    return cameras


def _copy_h5_file(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key in src.keys():
        src.copy(src[key], dst, name=key)


def _write_dataset(group: h5py.Group, name: str, value: np.ndarray, overwrite: bool) -> None:
    if name in group:
        if not overwrite:
            raise FileExistsError(f"{group.name}/{name} exists. Pass --overwrite to replace it.")
        del group[name]
    group.create_dataset(name, data=value, compression="gzip", shuffle=True)


def _require_clean_group(parent: h5py.Group, name: str, overwrite: bool) -> h5py.Group:
    if name in parent:
        if isinstance(parent[name], h5py.Dataset):
            if not overwrite:
                raise FileExistsError(f"{parent.name}/{name} exists as a dataset.")
            del parent[name]
        elif overwrite:
            del parent[name]
    return parent.require_group(name)


def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_m = depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        depth_m = depth_m / 1024.0
    return depth_m


def _pointcloud_from_depth(
    depth_m: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
) -> np.ndarray:
    """Back-project depth to world points using ManiSkill OpenCV camera params.

    This mirrors PPI's projection inversion: inv(K @ [R|t]) maps depth-scaled
    pixel coordinates [u*z, v*z, z] back to world coordinates.
    """

    height, width = depth_m.shape
    u = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    v = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    pixels = np.stack((u * depth_m, v * depth_m, depth_m), axis=-1)

    projection = intrinsic_cv.astype(np.float32) @ extrinsic_cv.astype(np.float32)
    projection_h = np.concatenate(
        [projection, np.asarray([[0, 0, 0, 1]], dtype=np.float32)], axis=0
    )
    projection_inv = np.linalg.inv(projection_h)[:3]
    pixels_h = np.concatenate(
        [pixels.reshape(-1, 3), np.ones((height * width, 1), dtype=np.float32)], axis=1
    )
    return (pixels_h @ projection_inv.T).astype(np.float32)


def _sample_points(
    xyzrgb: np.ndarray,
    pixel_uv: np.ndarray,
    num_points: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    xyz = xyzrgb[:, :3]
    valid = np.isfinite(xyz).all(axis=1)
    valid &= xyz[:, 2] > bbox_min[2]
    inside = valid & np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)
    candidate_indices = np.flatnonzero(inside)
    used_fallback = False

    if len(candidate_indices) == 0:
        candidate_indices = np.flatnonzero(valid)
        used_fallback = True
    if len(candidate_indices) == 0:
        raise ValueError("No valid depth points available for point cloud sampling.")

    replace = len(candidate_indices) < num_points
    sampled_indices = rng.choice(candidate_indices, size=num_points, replace=replace)
    return (
        xyzrgb[sampled_indices].astype(np.float32),
        pixel_uv[sampled_indices].astype(np.int32),
        used_fallback or replace,
    )


def _points_per_camera(total: int, num_cameras: int) -> List[int]:
    base = total // num_cameras
    rem = total % num_cameras
    return [base + (1 if idx < rem else 0) for idx in range(num_cameras)]


def _build_traj_pointcloud(
    traj: h5py.Group,
    cameras: Sequence[str],
    num_points: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    seed: int,
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
    Dict[str, int],
]:
    per_camera_counts = _points_per_camera(num_points, len(cameras))
    per_camera_clouds: Dict[str, List[np.ndarray]] = {camera: [] for camera in cameras}
    per_camera_pixels: Dict[str, List[np.ndarray]] = {camera: [] for camera in cameras}
    per_camera_indices: Dict[str, List[np.ndarray]] = {camera: [] for camera in cameras}
    fused_frames: List[np.ndarray] = []
    fused_pixel_frames: List[np.ndarray] = []
    fused_camera_index_frames: List[np.ndarray] = []
    fallback_counts = {camera: 0 for camera in cameras}

    num_frames = traj["obs"]["sensor_data"][cameras[0]]["rgb"].shape[0]
    for frame_idx in range(num_frames):
        frame_clouds = []
        frame_pixels = []
        frame_camera_indices = []
        for camera_idx, (camera, camera_points) in enumerate(zip(cameras, per_camera_counts)):
            rgb = traj["obs"]["sensor_data"][camera]["rgb"][frame_idx]
            depth = traj["obs"]["sensor_data"][camera]["depth"][frame_idx]
            intrinsic = traj["obs"]["sensor_param"][camera]["intrinsic_cv"][frame_idx]
            extrinsic = traj["obs"]["sensor_param"][camera]["extrinsic_cv"][frame_idx]

            depth_m = _depth_to_meters(depth)
            xyz = _pointcloud_from_depth(depth_m, intrinsic, extrinsic)
            height, width = depth_m.shape
            u = np.tile(np.arange(width, dtype=np.int32), (height, 1))
            v = np.tile(np.arange(height, dtype=np.int32)[:, None], (1, width))
            pixel_uv = np.stack((u, v), axis=-1).reshape(-1, 2)
            rgb_flat = rgb.reshape(-1, 3).astype(np.float32)
            xyzrgb = np.concatenate([xyz, rgb_flat], axis=1)

            rng = np.random.default_rng(seed + frame_idx * 9973 + camera_idx * 101)
            sampled, sampled_pixel_uv, fallback = _sample_points(
                xyzrgb,
                pixel_uv,
                camera_points,
                bbox_min,
                bbox_max,
                rng,
            )
            if fallback:
                fallback_counts[camera] += 1
            per_camera_clouds[camera].append(sampled)
            per_camera_pixels[camera].append(sampled_pixel_uv)
            per_camera_indices[camera].append(
                np.full((sampled.shape[0],), camera_idx, dtype=np.int16)
            )
            frame_clouds.append(sampled)
            frame_pixels.append(sampled_pixel_uv)
            frame_camera_indices.append(per_camera_indices[camera][-1])
        fused_frames.append(np.concatenate(frame_clouds, axis=0).astype(np.float32))
        fused_pixel_frames.append(np.concatenate(frame_pixels, axis=0).astype(np.int32))
        fused_camera_index_frames.append(np.concatenate(frame_camera_indices, axis=0).astype(np.int16))

    per_camera_arrays = {
        camera: np.stack(frames, axis=0).astype(np.float32)
        for camera, frames in per_camera_clouds.items()
    }
    per_camera_sources = {
        camera: {
            "pixel_uv": np.stack(per_camera_pixels[camera], axis=0).astype(np.int32),
            "camera_index": np.stack(per_camera_indices[camera], axis=0).astype(np.int16),
        }
        for camera in cameras
    }
    fused = np.stack(fused_frames, axis=0).astype(np.float32)
    fused_source = {
        "pixel_uv": np.stack(fused_pixel_frames, axis=0).astype(np.int32),
        "camera_index": np.stack(fused_camera_index_frames, axis=0).astype(np.int16),
    }
    return per_camera_arrays, fused, per_camera_sources, fused_source, fallback_counts


def build_pointcloud_dataset(
    demo_path: Path,
    output_path: Path,
    *,
    task_name: str,
    cameras: Optional[Sequence[str]],
    num_points: int,
    bbox: Tuple[np.ndarray, np.ndarray],
    num_traj: Optional[int],
    overwrite: bool,
    in_place: bool,
    seed: int,
) -> Dict[str, object]:
    if output_path.exists() and not overwrite and not in_place:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")

    mode = "r+" if in_place else "r"
    with h5py.File(demo_path, mode) as f_in:
        traj_names = sorted([key for key in f_in.keys() if key.startswith("traj_")], key=_traj_sort_key)
        if num_traj is not None:
            traj_names = traj_names[:num_traj]
        if not traj_names:
            raise ValueError(f"No traj_* groups found in {demo_path}")

        if in_place:
            f_out = f_in
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            f_out = h5py.File(output_path, "w")
            _copy_h5_file(f_in, f_out)

        summary_rows = []
        bbox_min, bbox_max = bbox
        try:
            f_out.attrs["pointcloud_source_demo_path"] = str(demo_path)
            f_out.attrs["pointcloud_style"] = "ppi_random_sample_bbox"
            f_out.attrs["pointcloud_num_points"] = int(num_points)
            f_out.attrs["pointcloud_bbox_min"] = bbox_min
            f_out.attrs["pointcloud_bbox_max"] = bbox_max

            for local_idx, traj_name in enumerate(traj_names):
                traj_in = f_in[traj_name]
                traj_out = f_out[traj_name]
                camera_names = _discover_cameras(traj_in, cameras)
                per_camera, fused, per_camera_sources, fused_source, fallback_counts = _build_traj_pointcloud(
                    traj_in,
                    camera_names,
                    num_points,
                    bbox_min,
                    bbox_max,
                    seed + local_idx * 1000003,
                )

                obs_out = traj_out.require_group("obs")
                pointcloud_group = _require_clean_group(obs_out, "point_cloud", overwrite)
                for camera_name in camera_names:
                    _write_dataset(pointcloud_group, camera_name, per_camera[camera_name], overwrite)
                _write_dataset(pointcloud_group, "fused", fused, overwrite)

                source_group = _require_clean_group(obs_out, "point_cloud_source", overwrite)
                fused_source_group = _require_clean_group(source_group, "fused", overwrite)
                _write_dataset(
                    fused_source_group,
                    "camera_index",
                    fused_source["camera_index"],
                    overwrite,
                )
                _write_dataset(fused_source_group, "pixel_uv", fused_source["pixel_uv"], overwrite)
                for camera_name in camera_names:
                    camera_source_group = _require_clean_group(source_group, camera_name, overwrite)
                    _write_dataset(
                        camera_source_group,
                        "camera_index",
                        per_camera_sources[camera_name]["camera_index"],
                        overwrite,
                    )
                    _write_dataset(
                        camera_source_group,
                        "pixel_uv",
                        per_camera_sources[camera_name]["pixel_uv"],
                        overwrite,
                    )

                tcp_pose = traj_in["obs"]["extra"]["tcp_pose"][()].astype(np.float32)
                tcp_group = _require_clean_group(obs_out, "tcp_trajectory", overwrite)
                _write_dataset(tcp_group, "pose", tcp_pose, overwrite)
                _write_dataset(tcp_group, "pos", tcp_pose[:, :3], overwrite)

                traj_out.attrs["pointcloud_cameras"] = json.dumps(camera_names)
                traj_out.attrs["pointcloud_shape"] = json.dumps(list(fused.shape))
                summary_rows.append(
                    {
                        "traj": traj_name,
                        "num_frames": int(fused.shape[0]),
                        "cameras": camera_names,
                        "per_camera_shapes": {
                            camera: list(per_camera[camera].shape) for camera in camera_names
                        },
                        "fused_shape": list(fused.shape),
                        "fused_camera_index_shape": list(fused_source["camera_index"].shape),
                        "fused_pixel_uv_shape": list(fused_source["pixel_uv"].shape),
                        "tcp_pose_shape": list(tcp_pose.shape),
                        "fallback_counts": fallback_counts,
                    }
                )
                print(
                    f"{traj_name}: cameras={camera_names}, point_cloud={tuple(fused.shape)}, "
                    f"tcp_pose={tuple(tcp_pose.shape)}",
                    flush=True,
                )
        finally:
            if not in_place:
                f_out.close()

    return {
        "demo_path": str(demo_path),
        "output_path": str(demo_path if in_place else output_path),
        "task_name": task_name,
        "num_points": int(num_points),
        "bbox_min": bbox[0].tolist(),
        "bbox_max": bbox[1].tolist(),
        "num_trajectories": len(summary_rows),
        "trajectories": summary_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--task-name", default="StackCube-v1")
    parser.add_argument("--cameras", default="base_camera,hand_camera")
    parser.add_argument("--num-points", type=int, default=6144)
    parser.add_argument(
        "--bbox",
        default="auto",
        help="auto or six numbers: xmin ymin zmin xmax ymax zmax",
    )
    parser.add_argument("--num-traj", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_path = Path(args.demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")

    if args.in_place:
        output_path = demo_path
    elif args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = demo_path.with_suffix("").with_name(f"{demo_path.stem}.with_pointcloud.h5")

    summary = build_pointcloud_dataset(
        demo_path,
        output_path,
        task_name=args.task_name,
        cameras=_parse_cameras(args.cameras),
        num_points=args.num_points,
        bbox=_parse_bbox(args.bbox, args.task_name),
        num_traj=args.num_traj,
        overwrite=args.overwrite,
        in_place=args.in_place,
        seed=args.seed,
    )

    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else Path(summary["output_path"]).with_suffix(".pointcloud.summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Wrote {summary['output_path']}; trajectories={summary['num_trajectories']}; "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
