#!/usr/bin/env python
"""Build Map4D DiT GT-map sidecar data.

The output sidecar copies an existing Map4D DiT keyframe sidecar and validates
that the demo HDF5 contains the Semantic Field inputs required by Map4DDiT.

Important: this script does not run Map4DEncoder and does not write
``traj_*/map_node_feature``. Map encoder features are produced online inside
Map4DDiT so the map encoder is trained jointly with the denoising network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAPS4D_DIR = PROJECT_ROOT / "map4d" / "representation" / "maps4d"
for path in (PROJECT_ROOT, MAPS4D_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import numpy as np

from helper.build_keyframe_aux_dataset import (
    _infer_task_name,
    _parse_actor_names,
)
from map4d.representation.maps4d.metadata import TASK_METADATA_FILES, get_task_num_map_nodes


GT_MAP_FORMAT = "map4d_gt_pose_sidecar_v1"
SEMANTIC_FIELD_FORMAT = "rgbd_points_plus_map4d_node_centers_v1"


def _traj_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def _copy_h5_tree(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key in src.keys():
        src.copy(src[key], dst, name=key)


def _read_sidecar_map(sidecar: Optional[h5py.File], traj_name: str) -> Optional[np.ndarray]:
    if sidecar is None or traj_name not in sidecar:
        return None
    group = sidecar[traj_name]
    if "map4d" not in group:
        return None
    return np.asarray(group["map4d"][()], dtype=np.float32)


def _representation_json_path(task_name: str) -> str:
    filename = TASK_METADATA_FILES.get(task_name)
    if filename is None:
        raise KeyError(f"No Map4D representation JSON registered for task {task_name!r}")
    return f"map4d/representation/maps4d/{filename}"


def _validate_no_precomputed_map_features(sidecar: h5py.File) -> None:
    stale_paths = []
    for traj_name in sidecar.keys():
        if not traj_name.startswith("traj_"):
            continue
        group = sidecar[traj_name]
        if isinstance(group, h5py.Group) and "map_node_feature" in group:
            stale_paths.append(f"{traj_name}/map_node_feature")
    if stale_paths:
        raise ValueError(
            "Input sidecar contains precomputed map encoder features. "
            "Rebuild the keyframe sidecar without these datasets: "
            + ", ".join(stale_paths[:8])
        )


def _load_pose_map4d(
    sidecar: h5py.File,
    traj_name: str,
) -> np.ndarray:
    sidecar_map = _read_sidecar_map(sidecar, traj_name)
    if sidecar_map is None:
        raise KeyError(f"Input sidecar missing {traj_name}/map4d")
    if sidecar_map.ndim != 3 or sidecar_map.shape[-1] not in {7, 9, 10, 12}:
        raise ValueError(f"{traj_name}: expected sidecar map4d [T,N,7/9/10/12], got {sidecar_map.shape}")
    if sidecar_map.shape[-1] in {7, 9}:
        return sidecar_map.astype(np.float32)
    if sidecar_map.shape[-1] == 10:
        return sidecar_map[..., 3:10].astype(np.float32)
    return sidecar_map[..., 3:12].astype(np.float32)


def _validate_semantic_field(
    traj: h5py.Group,
    *,
    pointcloud_path: str,
    dino_feature_path: str,
    map4d: np.ndarray,
) -> Dict[str, object]:
    point_node = traj
    for part in pointcloud_path.split("/"):
        if not isinstance(point_node, h5py.Group) or part not in point_node:
            raise KeyError(f"{traj.name} missing {pointcloud_path}")
        point_node = point_node[part]
    if not isinstance(point_node, h5py.Dataset):
        raise TypeError(f"{traj.name}/{pointcloud_path} must be an HDF5 dataset")

    dino_node = traj
    for part in dino_feature_path.split("/"):
        if not isinstance(dino_node, h5py.Group) or part not in dino_node:
            raise KeyError(f"{traj.name} missing {dino_feature_path}")
        dino_node = dino_node[part]
    if not isinstance(dino_node, h5py.Dataset):
        raise TypeError(
            f"{traj.name}/{dino_feature_path} must be a per-point HDF5 dataset, not a group"
        )

    point_shape = tuple(point_node.shape)
    dino_shape = tuple(dino_node.shape)
    if len(point_shape) != 3 or point_shape[-1] < 3:
        raise ValueError(f"{traj.name}/{pointcloud_path} must be [T,P,>=3], got {point_shape}")
    if len(dino_shape) != 3:
        raise ValueError(f"{traj.name}/{dino_feature_path} must be [T,P,D], got {dino_shape}")
    if point_shape[:2] != dino_shape[:2]:
        raise ValueError(
            f"{traj.name}: point_cloud and dino_feature must share [T,P], got {point_shape} and {dino_shape}"
        )
    if map4d.ndim != 3 or map4d.shape[-1] < 3:
        raise ValueError(f"{traj.name}: map4d must be [T,N,>=3], got {map4d.shape}")
    node_count = int(map4d.shape[1])
    if point_shape[0] != map4d.shape[0]:
        raise ValueError(f"{traj.name}: semantic field T={point_shape[0]} != map4d T={map4d.shape[0]}")
    if point_shape[1] <= node_count:
        raise ValueError(f"{traj.name}: semantic field point count {point_shape[1]} must exceed N={node_count}")
    source_group = traj.get("obs", {}).get("semantic_field_source") if "obs" in traj else None
    if not isinstance(source_group, h5py.Group):
        raise KeyError(f"{traj.name} missing obs/semantic_field_source")
    if "token_type" not in source_group:
        raise KeyError(f"{traj.name} missing obs/semantic_field_source/token_type")
    token_type = np.asarray(source_group["token_type"][()])
    if token_type.shape != point_shape[:2]:
        raise ValueError(
            f"{traj.name}: token_type shape {token_type.shape} must match semantic field [T,P+N] {point_shape[:2]}"
        )
    if not np.all(token_type[:, -node_count:] == 1):
        raise ValueError(f"{traj.name}: last {node_count} token_type entries must be Map4D node centers")
    if not np.all(token_type[:, : -node_count] == 0):
        raise ValueError(f"{traj.name}: RGB-D token_type prefix must be 0")
    if "node_center_token_indices" not in source_group:
        raise KeyError(f"{traj.name} missing obs/semantic_field_source/node_center_token_indices")
    expected_indices = np.arange(point_shape[1] - node_count, point_shape[1], dtype=np.int32)
    actual_indices = np.asarray(source_group["node_center_token_indices"][()], dtype=np.int32).reshape(-1)
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            f"{traj.name}: node_center_token_indices must be {expected_indices.tolist()}, "
            f"got {actual_indices.tolist()}"
        )
    point_xyz_tail = np.asarray(point_node[:, -node_count:, :3], dtype=np.float32)
    if not np.allclose(point_xyz_tail, map4d[..., 0:3], atol=1e-5):
        raise ValueError(f"{traj.name}: last {node_count} point_cloud xyz rows must match map4d positions")
    _require_dino_provenance(traj.name, dino_node)
    return {
        "point_cloud_shape": point_shape,
        "dino_feature_shape": dino_shape,
        "rgbd_point_count": int(point_shape[1] - node_count),
        "node_center_count": node_count,
        "semantic_field_format": SEMANTIC_FIELD_FORMAT,
    }


def _validate_pointcloud_field(
    traj: h5py.Group,
    *,
    pointcloud_path: str,
) -> Dict[str, object]:
    point_node = traj
    for part in pointcloud_path.split("/"):
        if not isinstance(point_node, h5py.Group) or part not in point_node:
            raise KeyError(f"{traj.name} missing {pointcloud_path}")
        point_node = point_node[part]
    if not isinstance(point_node, h5py.Dataset):
        raise TypeError(f"{traj.name}/{pointcloud_path} must be an HDF5 dataset")

    point_shape = tuple(point_node.shape)
    if len(point_shape) != 3 or point_shape[-1] < 3:
        raise ValueError(f"{traj.name}/{pointcloud_path} must be [T,P,>=3], got {point_shape}")
    return {
        "point_cloud_shape": point_shape,
    }


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _require_dino_provenance(traj_name: str, feature: h5py.Dataset) -> None:
    flattened = {}
    for node in (feature.file, feature.parent, feature, feature.parent.parent):
        if node is None:
            continue
        for key, value in node.attrs.items():
            flattened[str(key)] = _decode_attr(value)

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


def build_map4d_context_dataset(
    demo_path: Path,
    output_path: Path,
    *,
    task_name: str,
    actor_names: Sequence[str],
    input_sidecar_path: Optional[Path],
    num_traj: Optional[int],
    pointcloud_path: str,
    dino_feature_path: str,
    require_semantic_field: bool,
    overwrite: bool,
) -> Dict[str, object]:
    if not require_semantic_field:
        raise ValueError("--skip-semantic-field-validation is disabled; unified Semantic Field validation is required.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
    if input_sidecar_path is None:
        raise ValueError("--input-sidecar-path is required; final context sidecar must copy GT map/keyframe targets")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_map_nodes = get_task_num_map_nodes(task_name)

    summary_rows = []
    with h5py.File(demo_path, "r") as f_demo:
        sidecar_in = h5py.File(input_sidecar_path, "r")
        try:
            with h5py.File(output_path, "w") as f_out:
                _validate_no_precomputed_map_features(sidecar_in)
                _copy_h5_tree(sidecar_in, f_out)

                traj_names = sorted([k for k in f_demo.keys() if k.startswith("traj_")], key=_traj_sort_key)
                if num_traj is not None:
                    traj_names = traj_names[:num_traj]

                f_out.attrs["source_demo_path"] = str(demo_path)
                if input_sidecar_path is not None:
                    f_out.attrs["source_sidecar_path"] = str(input_sidecar_path)
                f_out.attrs["task_name"] = task_name
                f_out.attrs["representation_json"] = sidecar_in.attrs.get(
                    "representation_json",
                    _representation_json_path(task_name),
                )
                f_out.attrs["map_context_format"] = GT_MAP_FORMAT
                f_out.attrs["num_map_nodes"] = int(num_map_nodes)
                f_out.attrs["map_encoder_location"] = "online_in_Map4DDiT"
                f_out.attrs["semantic_field_format"] = SEMANTIC_FIELD_FORMAT

                for traj_name in traj_names:
                    traj = f_demo[traj_name]
                    pose_map4d = _load_pose_map4d(sidecar_in, traj_name)
                    if require_semantic_field:
                        semantic_info = _validate_semantic_field(
                            traj,
                            pointcloud_path=pointcloud_path,
                            dino_feature_path=dino_feature_path,
                            map4d=pose_map4d,
                        )
                    else:
                        semantic_info = _validate_pointcloud_field(
                            traj,
                            pointcloud_path=pointcloud_path,
                        )

                    group = f_out.require_group(traj_name)
                    group.attrs["num_frames"] = int(pose_map4d.shape[0])
                    group.attrs["num_target_nodes"] = int(pose_map4d.shape[1])
                    group.attrs["map4d_dim"] = int(pose_map4d.shape[-1])
                    group.attrs["num_map_nodes"] = int(num_map_nodes)
                    group.attrs["map_context_format"] = GT_MAP_FORMAT

                    row = {
                        "traj": traj_name,
                        "num_frames": int(pose_map4d.shape[0]),
                        "map4d_shape": list(pose_map4d.shape),
                        "num_target_nodes": int(pose_map4d.shape[1]),
                        "num_map_nodes": int(num_map_nodes),
                    }
                    if semantic_info is not None:
                        for key, value in semantic_info.items():
                            row[key] = list(value) if isinstance(value, tuple) else value
                    summary_rows.append(row)
        finally:
            sidecar_in.close()

    if not summary_rows:
        raise ValueError(f"No traj_* groups found in {demo_path}")

    return {
        "demo_path": str(demo_path),
        "input_sidecar_path": str(input_sidecar_path) if input_sidecar_path else None,
        "output_path": str(output_path),
        "task_name": task_name,
        "actor_names": list(actor_names),
        "representation_json": _representation_json_path(task_name),
        "map_context_format": GT_MAP_FORMAT,
        "num_map_nodes": int(num_map_nodes),
        "num_trajectories": len(summary_rows),
        "require_semantic_field": bool(require_semantic_field),
        "pointcloud_path": pointcloud_path,
        "dino_feature_path": dino_feature_path,
        "trajectories": summary_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument(
        "--input-sidecar-path",
        type=Path,
        required=True,
        help="Existing keyframe sidecar to copy into the final GT-map sidecar.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--actor-names", default=None)
    parser.add_argument("--num-traj", type=int, default=None)
    parser.add_argument("--pointcloud-path", default="obs/point_cloud/fused")
    parser.add_argument("--dino-feature-path", default="obs/dino_feature")
    parser.add_argument(
        "--skip-semantic-field-validation",
        action="store_true",
        help="Only validate point clouds. Use for online_dinov3 training datasets without precomputed obs/dino_feature.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_name = args.task_name or _infer_task_name(str(args.demo_path))
    actor_names = _parse_actor_names(args.actor_names, task_name)
    summary = build_map4d_context_dataset(
        args.demo_path,
        args.output_path,
        task_name=task_name,
        actor_names=actor_names,
        input_sidecar_path=args.input_sidecar_path,
        num_traj=args.num_traj,
        pointcloud_path=args.pointcloud_path,
        dino_feature_path=args.dino_feature_path,
        require_semantic_field=not args.skip_semantic_field_validation,
        overwrite=args.overwrite,
    )

    summary_path = args.summary_json or args.output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "Built {output_path}; trajectories={num_trajectories}, "
        "num_map_nodes={num_map_nodes}, map_context_format={map_context_format}".format(**summary)
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
