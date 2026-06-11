#!/usr/bin/env python
"""Build or validate per-point DINO Semantic Field features for Map4D DiT.

The current Map4DDiT expects:

    traj_*/obs/point_cloud/fused  [T, P, >=3]
    traj_*/obs/dino_feature       [T, P, D_sem]

Use mode=dinov3 to generate DINO patch-token features aligned to each sampled
point via traj_*/obs/point_cloud_source/fused/{camera_index,pixel_uv}. Use
mode=existing_dino to only validate a precomputed per-point DINO feature.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DP_ROOT = PROJECT_ROOT / "baselines" / "diffusion_policy"
for _path in (PROJECT_ROOT, DP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _traj_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def _split_h5_path(path: str) -> List[str]:
    parts = [part for part in path.strip("/").split("/") if part]
    if not parts:
        raise ValueError("HDF5 path must not be empty")
    return parts


def _get_dataset(root: h5py.Group, path: str) -> h5py.Dataset:
    node = root
    for part in _split_h5_path(path):
        if not isinstance(node, h5py.Group) or part not in node:
            raise KeyError(f"{root.name} missing {path}")
        node = node[part]
    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"{root.name}/{path} must be an HDF5 dataset")
    return node


def _require_parent_group(root: h5py.Group, path: str) -> Tuple[h5py.Group, str]:
    parts = _split_h5_path(path)
    group = root
    for part in parts[:-1]:
        if part in group and isinstance(group[part], h5py.Dataset):
            raise TypeError(f"{group.name}/{part} exists as a dataset; expected a group")
        group = group.require_group(part)
    return group, parts[-1]


def _write_dataset(group: h5py.Group, name: str, value: np.ndarray, overwrite: bool) -> h5py.Dataset:
    if name in group:
        if not overwrite:
            raise FileExistsError(f"{group.name}/{name} exists. Pass --overwrite to replace it.")
        del group[name]
    return group.create_dataset(name, data=value, compression="gzip", shuffle=True)


def _parse_cameras(value: str) -> Optional[Tuple[str, ...]]:
    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _validate_shapes(
    traj_name: str,
    pointcloud: h5py.Dataset,
    feature: h5py.Dataset,
) -> Dict[str, object]:
    point_shape = tuple(pointcloud.shape)
    feature_shape = tuple(feature.shape)
    if len(point_shape) != 3 or point_shape[-1] < 3:
        raise ValueError(f"{traj_name}: point cloud must be [T,P,>=3], got {point_shape}")
    if len(feature_shape) != 3:
        raise ValueError(f"{traj_name}: semantic feature must be [T,P,D], got {feature_shape}")
    if point_shape[:2] != feature_shape[:2]:
        raise ValueError(
            f"{traj_name}: point cloud and semantic feature must share [T,P], "
            f"got {point_shape} and {feature_shape}"
        )
    return {
        "traj": traj_name,
        "point_cloud_shape": list(point_shape),
        "semantic_feature_shape": list(feature_shape),
        "semantic_feature_dim": int(feature_shape[-1]),
    }


def _attrs_to_dicts(*nodes) -> List[Dict[str, object]]:
    attrs = []
    for node in nodes:
        if node is None:
            continue
        attrs.append({key: value for key, value in node.attrs.items()})
    return attrs


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _require_dino_provenance(traj_name: str, feature: h5py.Dataset) -> Dict[str, object]:
    attr_dicts = _attrs_to_dicts(feature.file, feature.parent, feature, feature.parent.parent)
    flattened = {}
    for attrs in attr_dicts:
        for key, value in attrs.items():
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

    return {
        "semantic_feature_source": flattened.get("semantic_feature_source"),
        "semantic_feature_model": flattened.get("semantic_feature_model")
        or flattened.get("dinov3_model")
        or flattened.get("model"),
        "feature_type": flattened.get("feature_type"),
    }


def _load_dinov3_helpers():
    from scripts.data_collection.build_dinov3_features import (  # noqa: PLC0415
        MODEL_WEIGHT_FILES,
        _find_weights_path,
        _load_backbone,
        _parse_image_size,
        _preprocess_rgb,
        _target_size,
    )

    return {
        "MODEL_WEIGHT_FILES": MODEL_WEIGHT_FILES,
        "_find_weights_path": _find_weights_path,
        "_load_backbone": _load_backbone,
        "_parse_image_size": _parse_image_size,
        "_preprocess_rgb": _preprocess_rgb,
        "_target_size": _target_size,
    }


def _pointcloud_cameras(traj: h5py.Group, requested: Optional[Sequence[str]]) -> List[str]:
    raw = traj.attrs.get("pointcloud_cameras")
    if raw is None:
        raise KeyError(f"{traj.name} missing pointcloud_cameras attr. Rebuild point clouds first.")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    cameras = json.loads(str(raw))
    if not isinstance(cameras, list) or not all(isinstance(item, str) for item in cameras):
        raise ValueError(f"{traj.name} pointcloud_cameras attr must be a JSON string list, got {raw!r}")
    if requested is not None and list(requested) != cameras:
        raise ValueError(
            f"{traj.name}: requested cameras {list(requested)} must match pointcloud source order {cameras}"
        )
    return cameras


def _get_patch_size(backbone) -> Tuple[int, int]:
    patch_size = getattr(backbone, "patch_size", None)
    if patch_size is None and hasattr(backbone, "patch_embed"):
        patch_size = getattr(backbone.patch_embed, "patch_size", None)
    if patch_size is None:
        raise AttributeError("DINOv3 backbone does not expose patch_size or patch_embed.patch_size")
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        return int(patch_size[0]), int(patch_size[1])
    raise TypeError(f"Unsupported DINOv3 patch_size={patch_size!r}")


def _encode_patch_grid(
    rgb: np.ndarray,
    *,
    backbone,
    device,
    batch_size: int,
    image_size: Optional[Tuple[int, int]],
    multiple: int,
    amp: bool,
    preprocess_rgb,
    target_size_fn,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    import torch  # noqa: PLC0415

    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB shape [T,H,W,3], got {rgb.shape}")
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    resize_to = target_size_fn(height, width, image_size=image_size, multiple=multiple)
    target_h, target_w = resize_to if resize_to is not None else (height, width)
    patch_h, patch_w = _get_patch_size(backbone)
    if target_h % patch_h != 0 or target_w % patch_w != 0:
        raise ValueError(
            f"DINO input size {(target_h, target_w)} must be divisible by patch size {(patch_h, patch_w)}"
        )
    grid_h, grid_w = target_h // patch_h, target_w // patch_w

    chunks = []
    use_amp = amp and device.type == "cuda"
    for start in range(0, rgb.shape[0], batch_size):
        batch = rgb[start : start + batch_size]
        x = preprocess_rgb(batch, device=device, image_size=image_size, multiple=multiple)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_amp):
            feats = backbone.forward_features(x)
            tokens = feats["x_norm_patchtokens"]
        if tokens.ndim != 3:
            raise ValueError(f"DINO patch tokens must be [B,N,D], got {tuple(tokens.shape)}")
        if int(tokens.shape[1]) != grid_h * grid_w:
            raise ValueError(
                f"DINO patch token count {int(tokens.shape[1])} does not match grid "
                f"{grid_h}x{grid_w} for input {(target_h, target_w)}"
            )
        tokens = tokens.reshape(tokens.shape[0], grid_h, grid_w, tokens.shape[-1])
        chunks.append(tokens.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32), (target_h, target_w)


def _assign_camera_patch_features(
    out: np.ndarray,
    filled: np.ndarray,
    *,
    patch_grid: np.ndarray,
    camera_index: np.ndarray,
    pixel_uv: np.ndarray,
    camera_idx: int,
    image_hw: Tuple[int, int],
) -> None:
    height, width = image_hw
    grid_h, grid_w = int(patch_grid.shape[1]), int(patch_grid.shape[2])
    for frame_idx in range(out.shape[0]):
        point_indices = np.flatnonzero(camera_index[frame_idx] == camera_idx)
        if point_indices.size == 0:
            continue
        uv = pixel_uv[frame_idx, point_indices]
        if np.any(uv[:, 0] < 0) or np.any(uv[:, 0] >= width) or np.any(uv[:, 1] < 0) or np.any(uv[:, 1] >= height):
            raise ValueError(
                f"camera_idx={camera_idx} frame={frame_idx}: pixel_uv out of bounds for image {(height, width)}"
            )
        patch_x = np.floor((uv[:, 0].astype(np.float32) + 0.5) * grid_w / width).astype(np.int64)
        patch_y = np.floor((uv[:, 1].astype(np.float32) + 0.5) * grid_h / height).astype(np.int64)
        patch_x = np.clip(patch_x, 0, grid_w - 1)
        patch_y = np.clip(patch_y, 0, grid_h - 1)
        out[frame_idx, point_indices] = patch_grid[frame_idx, patch_y, patch_x]
        filled[frame_idx, point_indices] = True


def _generate_dinov3_point_features(
    demo_path: Path,
    *,
    pointcloud_path: str,
    source_path: str,
    output_path: str,
    num_traj: Optional[int],
    model: str,
    weights_path: Optional[str],
    third_party_dir: Path,
    cameras: Optional[Sequence[str]],
    batch_size: int,
    device_name: str,
    image_size_value: Optional[str],
    multiple: int,
    amp: bool,
    overwrite: bool,
    output_dtype: str,
) -> Dict[str, object]:
    helpers = _load_dinov3_helpers()
    if model not in helpers["MODEL_WEIGHT_FILES"]:
        raise ValueError(f"Unsupported DINOv3 model={model!r}. Use one of {sorted(helpers['MODEL_WEIGHT_FILES'])}")
    weights = helpers["_find_weights_path"](model, weights_path)
    if not third_party_dir.exists():
        raise FileNotFoundError(f"DINOv3 third_party dir not found: {third_party_dir}")

    import torch  # noqa: PLC0415

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"Loading DINOv3 backbone once: model={model}, device={device}, weights={weights}",
        flush=True,
    )
    backbone = helpers["_load_backbone"](str(third_party_dir), str(weights), model)
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    image_size = helpers["_parse_image_size"](image_size_value)
    print(f"Loaded DINOv3 backbone: embed_dim={int(backbone.embed_dim)}", flush=True)

    if output_dtype == "float32":
        h5_dtype = np.float32
    elif output_dtype == "float16":
        h5_dtype = np.float16
    else:
        raise ValueError(f"Unsupported output_dtype={output_dtype!r}. Use float32 or float16.")

    rows = []
    feature_dims = set()
    with h5py.File(demo_path, "r+") as f:
        f.attrs["semantic_feature_source"] = "dinov3_patch_token_projected_to_pointcloud"
        f.attrs["semantic_feature_model"] = model
        f.attrs["feature_type"] = "dinov3"
        f.attrs["semantic_feature_pointcloud_path"] = pointcloud_path
        f.attrs["semantic_feature_source_path"] = source_path

        traj_names = sorted([key for key in f.keys() if key.startswith("traj_")], key=_traj_sort_key)
        if num_traj is not None:
            traj_names = traj_names[:num_traj]
        if not traj_names:
            raise ValueError(f"No traj_* groups found in {demo_path}")

        for traj_name in traj_names:
            traj = f[traj_name]
            camera_names = _pointcloud_cameras(traj, cameras)
            pointcloud = _get_dataset(traj, pointcloud_path)
            camera_index = _get_dataset(traj, f"{source_path}/camera_index")[()]
            pixel_uv = _get_dataset(traj, f"{source_path}/pixel_uv")[()]
            point_shape = tuple(pointcloud.shape)
            if camera_index.shape != point_shape[:2]:
                raise ValueError(
                    f"{traj_name}: camera_index shape {camera_index.shape} must match point cloud [T,P] {point_shape[:2]}"
                )
            if pixel_uv.shape != point_shape[:2] + (2,):
                raise ValueError(
                    f"{traj_name}: pixel_uv shape {pixel_uv.shape} must match point cloud [T,P,2] {point_shape[:2] + (2,)}"
                )
            if int(camera_index.min()) < 0 or int(camera_index.max()) >= len(camera_names):
                raise ValueError(
                    f"{traj_name}: camera_index range [{int(camera_index.min())}, {int(camera_index.max())}] "
                    f"is incompatible with cameras={camera_names}"
                )

            semantic_feature = None
            filled = np.zeros(point_shape[:2], dtype=bool)
            camera_feature_shapes = {}
            resize_shapes = {}
            for camera_idx, camera_name in enumerate(camera_names):
                rgb = _get_dataset(traj, f"obs/sensor_data/{camera_name}/rgb")[()]
                if rgb.shape[0] != point_shape[0]:
                    raise ValueError(
                        f"{traj_name}: camera {camera_name} RGB length {rgb.shape[0]} "
                        f"!= point cloud length {point_shape[0]}"
                    )
                patch_grid, resized_hw = _encode_patch_grid(
                    rgb,
                    backbone=backbone,
                    device=device,
                    batch_size=batch_size,
                    image_size=image_size,
                    multiple=multiple,
                    amp=amp,
                    preprocess_rgb=helpers["_preprocess_rgb"],
                    target_size_fn=helpers["_target_size"],
                )
                if semantic_feature is None:
                    semantic_feature = np.empty(point_shape[:2] + (patch_grid.shape[-1],), dtype=np.float32)
                elif semantic_feature.shape[-1] != patch_grid.shape[-1]:
                    raise ValueError(
                        f"{traj_name}: inconsistent DINO dims, got {semantic_feature.shape[-1]} "
                        f"and {patch_grid.shape[-1]}"
                    )
                _assign_camera_patch_features(
                    semantic_feature,
                    filled,
                    patch_grid=patch_grid,
                    camera_index=camera_index,
                    pixel_uv=pixel_uv,
                    camera_idx=camera_idx,
                    image_hw=(int(rgb.shape[1]), int(rgb.shape[2])),
                )
                camera_feature_shapes[camera_name] = list(patch_grid.shape)
                resize_shapes[camera_name] = list(resized_hw)

            if semantic_feature is None:
                raise ValueError(f"{traj_name}: no camera features were generated")
            if not filled.all():
                missing = int((~filled).sum())
                raise ValueError(f"{traj_name}: {missing} point features were not assigned from any camera")

            parent, dataset_name = _require_parent_group(traj, output_path)
            feature_ds = _write_dataset(parent, dataset_name, semantic_feature.astype(h5_dtype), overwrite=overwrite)
            feature_ds.attrs["semantic_feature_source"] = "dinov3_patch_token_projected_to_pointcloud"
            feature_ds.attrs["semantic_feature_model"] = model
            feature_ds.attrs["feature_type"] = "dinov3"
            feature_ds.attrs["semantic_feature_dtype"] = output_dtype
            feature_ds.attrs["weights_path"] = str(weights)
            feature_ds.attrs["third_party_dir"] = str(third_party_dir)
            feature_ds.attrs["semantic_alignment"] = "pointcloud_source_pixel_uv_nearest_patch"
            feature_ds.attrs["pointcloud_path"] = pointcloud_path
            feature_ds.attrs["pointcloud_source_path"] = source_path
            feature_ds.attrs["dinov3_cameras"] = json.dumps(camera_names)
            feature_ds.attrs["dinov3_input_image_size"] = image_size_value or "auto_multiple"
            feature_ds.attrs["dinov3_multiple"] = int(multiple)

            row = _validate_shapes(traj_name, pointcloud, feature_ds)
            row.update(_require_dino_provenance(traj_name, feature_ds))
            row.update(
                {
                    "cameras": camera_names,
                    "camera_patch_feature_shapes": camera_feature_shapes,
                    "camera_resized_hw": resize_shapes,
                }
            )
            rows.append(row)
            feature_dims.add(row["semantic_feature_dim"])
            print(
                f"{traj_name}: point_cloud={tuple(pointcloud.shape)}, "
                f"semantic_feature={tuple(feature_ds.shape)}, cameras={camera_names}",
                flush=True,
            )

    if len(feature_dims) != 1:
        raise ValueError(f"All trajectories must have the same semantic feature dim, got {sorted(feature_dims)}")

    return {
        "demo_path": str(demo_path),
        "mode": "dinov3",
        "pointcloud_path": pointcloud_path,
        "pointcloud_source_path": source_path,
        "semantic_feature_path": output_path,
        "semantic_feature_dim": int(next(iter(feature_dims))),
        "semantic_feature_model": model,
        "semantic_feature_dtype": output_dtype,
        "weights_path": str(weights),
        "third_party_dir": str(third_party_dir),
        "num_trajectories": len(rows),
        "trajectories": rows,
    }


def build_point_semantic_features(
    demo_path: Path,
    *,
    mode: str,
    pointcloud_path: str,
    source_path: str,
    output_path: str,
    num_traj: Optional[int],
    model: str,
    weights_path: Optional[str],
    third_party_dir: Path,
    cameras: Optional[Sequence[str]],
    batch_size: int,
    device_name: str,
    image_size: Optional[str],
    multiple: int,
    amp: bool,
    overwrite: bool,
    output_dtype: str,
) -> Dict[str, object]:
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")
    if mode == "dinov3":
        return _generate_dinov3_point_features(
            demo_path,
            pointcloud_path=pointcloud_path,
            source_path=source_path,
            output_path=output_path,
            num_traj=num_traj,
            model=model,
            weights_path=weights_path,
            third_party_dir=third_party_dir,
            cameras=cameras,
            batch_size=batch_size,
            device_name=device_name,
            image_size_value=image_size,
            multiple=multiple,
            amp=amp,
            overwrite=overwrite,
            output_dtype=output_dtype,
        )
    if mode != "existing_dino":
        raise ValueError(f"Unsupported mode={mode!r}. Use dinov3 or existing_dino.")

    rows = []
    feature_dims = set()
    with h5py.File(demo_path, "r") as f:
        traj_names = sorted([key for key in f.keys() if key.startswith("traj_")], key=_traj_sort_key)
        if num_traj is not None:
            traj_names = traj_names[:num_traj]
        if not traj_names:
            raise ValueError(f"No traj_* groups found in {demo_path}")

        for traj_name in traj_names:
            traj = f[traj_name]
            pointcloud = _get_dataset(traj, pointcloud_path)
            feature = _get_dataset(traj, output_path)

            row = _validate_shapes(traj_name, pointcloud, feature)
            row.update(_require_dino_provenance(traj_name, feature))
            feature_dims.add(row["semantic_feature_dim"])
            rows.append(row)
            print(
                f"{traj_name}: point_cloud={tuple(pointcloud.shape)}, "
                f"semantic_feature={tuple(feature.shape)}",
                flush=True,
            )

    if len(feature_dims) != 1:
        raise ValueError(f"All trajectories must have the same semantic feature dim, got {sorted(feature_dims)}")

    return {
        "demo_path": str(demo_path),
        "mode": mode,
        "pointcloud_path": pointcloud_path,
        "pointcloud_source_path": None,
        "semantic_feature_path": output_path,
        "semantic_feature_dim": int(next(iter(feature_dims))),
        "num_trajectories": len(rows),
        "trajectories": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", required=True, type=Path)
    parser.add_argument("--mode", choices=["dinov3", "existing_dino"], default="dinov3")
    parser.add_argument("--pointcloud-path", default="obs/point_cloud/fused")
    parser.add_argument("--pointcloud-source-path", default="obs/point_cloud_source/fused")
    parser.add_argument("--output-path", default="obs/dino_feature")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--num-traj", type=int, default=None)
    parser.add_argument("--model", default="dinov3_vits16")
    parser.add_argument("--weights-path", default=None)
    parser.add_argument(
        "--third-party-dir",
        default=str(PROJECT_ROOT / "map4d" / "backbone" / "model" / "vision" / "dinov3"),
    )
    parser.add_argument("--cameras", default="auto", help="Must match pointcloud_cameras order, or auto.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--image-size", default=None, help="Optional DINO input resize, e.g. 224 or 224x224.")
    parser.add_argument("--multiple", type=int, default=16, help="Auto-resize H/W to this multiple.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dtype", default="float32", choices=["float32", "float16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_point_semantic_features(
        args.demo_path,
        mode=args.mode,
        pointcloud_path=args.pointcloud_path,
        source_path=args.pointcloud_source_path,
        output_path=args.output_path,
        num_traj=args.num_traj,
        model=args.model,
        weights_path=args.weights_path,
        third_party_dir=Path(args.third_party_dir),
        cameras=_parse_cameras(args.cameras),
        batch_size=args.batch_size,
        device_name=args.device,
        image_size=args.image_size,
        multiple=args.multiple,
        amp=not args.no_amp,
        overwrite=args.overwrite,
        output_dtype=args.output_dtype,
    )

    summary_path = args.summary_json or args.demo_path.with_suffix(".semantic_field.summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "Prepared per-point DINO Semantic Field features in {demo_path}; trajectories={num_trajectories}, "
        "semantic_feature_dim={semantic_feature_dim}; summary={summary_path}".format(
            summary_path=summary_path,
            **summary,
        )
    )


if __name__ == "__main__":
    main()
