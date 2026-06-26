#!/usr/bin/env python
"""Build or validate per-point DINO Semantic Field features for Map4D DiT.

The current Map4DDiT expects:

    traj_*/obs/point_cloud/fused  [T, P+N, >=3]
    traj_*/obs/dino_feature       [T, P+N, D_sem]

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
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DP_ROOT = PROJECT_ROOT / "baselines" / "diffusion_policy"
for _path in (PROJECT_ROOT, DP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from diffusion_policy.dinov3_encoder import _IMAGENET_MEAN, _IMAGENET_STD, _load_backbone


MODEL_WEIGHT_FILES = {
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
}


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


def _require_clean_group(parent: h5py.Group, name: str, overwrite: bool) -> h5py.Group:
    if name in parent:
        if isinstance(parent[name], h5py.Dataset):
            if not overwrite:
                raise FileExistsError(f"{parent.name}/{name} exists as a dataset.")
            del parent[name]
        elif overwrite:
            del parent[name]
    return parent.require_group(name)


def _parse_cameras(value: str) -> Optional[Tuple[str, ...]]:
    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _read_sidecar_map4d(sidecar: h5py.File, traj_name: str) -> np.ndarray:
    if traj_name not in sidecar:
        raise KeyError(f"Map4D sidecar missing group {traj_name}")
    group = sidecar[traj_name]
    if "map4d" not in group:
        raise KeyError(f"Map4D sidecar missing {traj_name}/map4d")
    map4d = np.asarray(group["map4d"][()], dtype=np.float32)
    if map4d.ndim != 3 or map4d.shape[-1] < 3:
        raise ValueError(f"{traj_name}/map4d must be [T,N,>=3], got {map4d.shape}")
    return map4d


def _semantic_source_group(traj: h5py.Group, overwrite: bool) -> h5py.Group:
    obs = traj.require_group("obs")
    return _require_clean_group(obs, "semantic_field_source", overwrite)


def _reject_existing_node_tokens(
    traj: h5py.Group,
    pointcloud: np.ndarray,
    *,
    expected_nodes: int,
) -> None:
    source = traj.get("obs", {}).get("semantic_field_source") if "obs" in traj else None
    if not isinstance(source, h5py.Group) or "token_type" not in source:
        return
    token_type = np.asarray(source["token_type"][()])
    if token_type.ndim != 2 or token_type.shape != pointcloud.shape[:2]:
        return
    if expected_nodes <= 0 or token_type.shape[1] <= expected_nodes:
        return
    if not np.all(token_type[:, -expected_nodes:] == 1):
        return
    if not np.all(token_type[:, : -expected_nodes] == 0):
        return
    raise ValueError(
        f"{traj.name}: {pointcloud.shape} already appears to be a unified Semantic Field. "
        "Rebuild obs/point_cloud/fused from RGB-D points before regenerating semantic features."
    )


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


def _find_weights_path(model: str, explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"DINOv3 weights not found: {path}")

    env_path = os.environ.get("DINOV3_WEIGHTS_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"DINOV3_WEIGHTS_PATH does not exist: {path}")

    filename = MODEL_WEIGHT_FILES.get(model)
    candidates = []
    if filename:
        candidates.extend(
            [
                Path("/data2/zehao/models/DINOv3/DINOv3_ViT_LVD_1689M") / filename,
                Path("/data2/zehao/models/dinov3") / filename,
                Path("/data2/zehao/models/DINOv3") / filename,
                Path("/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/foundation_models") / filename,
                PROJECT_ROOT / "checkpoints" / "dinov3" / filename,
                PROJECT_ROOT.parent / "checkpoints" / "dinov3" / filename,
                Path(
                    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/"
                    "zehao/4dmap/4dmap_policy/checkpoints/dinov3"
                )
                / filename,
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    hint = "\n  ".join(str(path) for path in candidates) if candidates else "(no known filename)"
    raise FileNotFoundError(
        "DINOv3 weights not found. Pass --weights-path or set DINOV3_WEIGHTS_PATH. "
        f"Checked:\n  {hint}"
    )


def _parse_image_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None or value == "":
        return None
    if "x" in value:
        height, width = value.lower().split("x", 1)
        return int(height), int(width)
    size = int(value)
    return size, size


def _target_size(
    height: int,
    width: int,
    *,
    image_size: Optional[Tuple[int, int]],
    multiple: int,
) -> Optional[Tuple[int, int]]:
    if image_size is not None:
        return image_size
    if multiple <= 1 or (height % multiple == 0 and width % multiple == 0):
        return None
    return (
        int(np.ceil(height / multiple) * multiple),
        int(np.ceil(width / multiple) * multiple),
    )


def _preprocess_rgb(
    rgb: np.ndarray,
    *,
    device: torch.device,
    image_size: Optional[Tuple[int, int]],
    multiple: int,
) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB shape [B,H,W,3], got {rgb.shape}")
    tensor = torch.as_tensor(rgb, device=device)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor.float()
        if tensor.max() > 2.0:
            tensor = tensor.div(255.0)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()

    resize_to = _target_size(
        tensor.shape[-2], tensor.shape[-1], image_size=image_size, multiple=multiple
    )
    if resize_to is not None:
        tensor = F.interpolate(tensor, size=resize_to, mode="bilinear", align_corners=False)

    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


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


def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_m = depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        depth_m = depth_m / 1024.0
    return depth_m


def _project_world_points(
    xyz: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32)
    ones = np.ones((*xyz.shape[:-1], 1), dtype=np.float32)
    xyz_h = np.concatenate([xyz, ones], axis=-1)
    camera = xyz_h @ extrinsic_cv.astype(np.float32).T
    z = camera[..., 2]
    pixel_h = camera[..., :3] @ intrinsic_cv.astype(np.float32).T
    uv = pixel_h[..., :2] / np.clip(pixel_h[..., 2:3], 1e-8, None)
    valid = np.isfinite(uv).all(axis=-1) & np.isfinite(z) & (z > 1e-6)
    return uv.astype(np.float32), z.astype(np.float32), valid


def _sample_patch_grid_at_uv(
    patch_grid_frame: np.ndarray,
    uv: np.ndarray,
    *,
    image_hw: Tuple[int, int],
) -> np.ndarray:
    height, width = image_hw
    grid_h, grid_w = int(patch_grid_frame.shape[0]), int(patch_grid_frame.shape[1])
    patch_x = np.floor((uv[..., 0].astype(np.float32) + 0.5) * grid_w / width).astype(np.int64)
    patch_y = np.floor((uv[..., 1].astype(np.float32) + 0.5) * grid_h / height).astype(np.int64)
    patch_x = np.clip(patch_x, 0, grid_w - 1)
    patch_y = np.clip(patch_y, 0, grid_h - 1)
    return patch_grid_frame[patch_y, patch_x]


def _node_center_semantic_features(
    traj: h5py.Group,
    *,
    node_xyz: np.ndarray,
    camera_names: Sequence[str],
    camera_patch_grids: Dict[str, np.ndarray],
    max_views: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_views <= 0:
        raise ValueError("max_views must be positive")
    if node_xyz.ndim != 3 or node_xyz.shape[-1] != 3:
        raise ValueError(f"node_xyz must be [T,N,3], got {node_xyz.shape}")
    time_steps, num_nodes, _ = node_xyz.shape
    feature_dim = int(next(iter(camera_patch_grids.values())).shape[-1])
    node_feature = np.zeros((time_steps, num_nodes, feature_dim), dtype=np.float32)
    node_rgb = np.zeros((time_steps, num_nodes, 3), dtype=np.float32)
    node_camera_index = np.full((time_steps, num_nodes, max_views), -1, dtype=np.int16)
    node_pixel_uv = np.full((time_steps, num_nodes, max_views, 2), -1, dtype=np.int32)
    node_camera_weight = np.zeros((time_steps, num_nodes, max_views), dtype=np.float32)

    for frame_idx in range(time_steps):
        candidates = [[] for _ in range(num_nodes)]
        for camera_idx, camera_name in enumerate(camera_names):
            rgb = np.asarray(traj[f"obs/sensor_data/{camera_name}/rgb"][frame_idx])
            depth = _depth_to_meters(traj[f"obs/sensor_data/{camera_name}/depth"][frame_idx])
            intrinsic = np.asarray(
                traj[f"obs/sensor_param/{camera_name}/intrinsic_cv"][frame_idx],
                dtype=np.float32,
            )
            extrinsic = np.asarray(
                traj[f"obs/sensor_param/{camera_name}/extrinsic_cv"][frame_idx],
                dtype=np.float32,
            )
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            uv, z, valid = _project_world_points(node_xyz[frame_idx], intrinsic, extrinsic)
            in_bounds = (
                valid
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
            visible_indices = np.flatnonzero(in_bounds)
            if visible_indices.size == 0:
                continue
            uv_int = np.rint(uv[visible_indices]).astype(np.int32)
            uv_int[:, 0] = np.clip(uv_int[:, 0], 0, width - 1)
            uv_int[:, 1] = np.clip(uv_int[:, 1], 0, height - 1)
            depth_at_uv = depth[uv_int[:, 1], uv_int[:, 0]]
            depth_delta = np.abs(depth_at_uv.astype(np.float32) - z[visible_indices])
            distance = np.linalg.norm(node_xyz[frame_idx, visible_indices] - extrinsic[:3, 3][None], axis=1)
            weight = 1.0 / ((depth_delta + 1e-3) * (distance + 1e-3))
            patch_feature = _sample_patch_grid_at_uv(
                camera_patch_grids[camera_name][frame_idx],
                uv[visible_indices],
                image_hw=(height, width),
            )
            rgb_feature = rgb[uv_int[:, 1], uv_int[:, 0]].astype(np.float32)
            for local_idx, node_idx in enumerate(visible_indices):
                candidates[int(node_idx)].append(
                    (
                        float(weight[local_idx]),
                        int(camera_idx),
                        uv_int[local_idx].copy(),
                        patch_feature[local_idx].astype(np.float32),
                        rgb_feature[local_idx].astype(np.float32),
                    )
                )

        for node_idx, node_candidates in enumerate(candidates):
            if not node_candidates:
                raise ValueError(
                    f"{traj.name}: node center frame={frame_idx}, node={node_idx} "
                    "does not project into any configured camera"
                )
            node_candidates.sort(key=lambda item: item[0], reverse=True)
            selected = node_candidates[:max_views]
            weights = np.asarray([item[0] for item in selected], dtype=np.float32)
            weights = weights / np.clip(weights.sum(), 1e-8, None)
            features = np.stack([item[3] for item in selected], axis=0)
            rgbs = np.stack([item[4] for item in selected], axis=0)
            node_feature[frame_idx, node_idx] = np.sum(features * weights[:, None], axis=0)
            node_rgb[frame_idx, node_idx] = np.sum(rgbs * weights[:, None], axis=0)
            for view_idx, (candidate, normalized_weight) in enumerate(zip(selected, weights)):
                _, camera_idx, uv_int, _, _ = candidate
                node_camera_index[frame_idx, node_idx, view_idx] = camera_idx
                node_pixel_uv[frame_idx, node_idx, view_idx] = uv_int
                node_camera_weight[frame_idx, node_idx, view_idx] = normalized_weight

    return node_feature, node_rgb, node_camera_index, node_pixel_uv, node_camera_weight


def _generate_dinov3_point_features(
    demo_path: Path,
    *,
    map4d_sidecar_path: Path,
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
    node_center_max_views: int,
) -> Dict[str, object]:
    if model not in MODEL_WEIGHT_FILES:
        raise ValueError(f"Unsupported DINOv3 model={model!r}. Use one of {sorted(MODEL_WEIGHT_FILES)}")
    weights = _find_weights_path(model, weights_path)
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
    backbone = _load_backbone(str(third_party_dir), str(weights), model)
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    image_size = _parse_image_size(image_size_value)
    print(f"Loaded DINOv3 backbone: embed_dim={int(backbone.embed_dim)}", flush=True)

    if output_dtype == "float32":
        h5_dtype = np.float32
    elif output_dtype == "float16":
        h5_dtype = np.float16
    else:
        raise ValueError(f"Unsupported output_dtype={output_dtype!r}. Use float32 or float16.")

    rows = []
    feature_dims = set()
    with h5py.File(demo_path, "r+") as f, h5py.File(map4d_sidecar_path, "r") as sidecar:
        f.attrs["semantic_field_format"] = "rgbd_points_plus_map4d_node_centers_v1"
        f.attrs["semantic_feature_source"] = "dinov3_patch_token_projected_to_semantic_field"
        f.attrs["semantic_feature_model"] = model
        f.attrs["feature_type"] = "dinov3"
        f.attrs["semantic_feature_pointcloud_path"] = pointcloud_path
        f.attrs["semantic_feature_source_path"] = source_path
        f.attrs["semantic_feature_map4d_sidecar_path"] = str(map4d_sidecar_path)

        traj_names = sorted([key for key in f.keys() if key.startswith("traj_")], key=_traj_sort_key)
        if num_traj is not None:
            traj_names = traj_names[:num_traj]
        if not traj_names:
            raise ValueError(f"No traj_* groups found in {demo_path}")

        for traj_name in traj_names:
            traj = f[traj_name]
            map4d = _read_sidecar_map4d(sidecar, traj_name)
            node_xyz = map4d[..., 0:3].astype(np.float32)
            camera_names = _pointcloud_cameras(traj, cameras)
            pointcloud_ds = _get_dataset(traj, pointcloud_path)
            pointcloud = np.asarray(pointcloud_ds[()], dtype=np.float32)
            _reject_existing_node_tokens(
                traj,
                pointcloud,
                expected_nodes=node_xyz.shape[1],
            )
            camera_index = _get_dataset(traj, f"{source_path}/camera_index")[()]
            pixel_uv = _get_dataset(traj, f"{source_path}/pixel_uv")[()]
            point_shape = tuple(pointcloud.shape)
            if node_xyz.shape[0] != point_shape[0]:
                raise ValueError(
                    f"{traj_name}: map4d length {node_xyz.shape[0]} != point cloud length {point_shape[0]}"
                )
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
            camera_patch_grids = {}
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
                    preprocess_rgb=_preprocess_rgb,
                    target_size_fn=_target_size,
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
                camera_patch_grids[camera_name] = patch_grid
                camera_feature_shapes[camera_name] = list(patch_grid.shape)
                resize_shapes[camera_name] = list(resized_hw)

            if semantic_feature is None:
                raise ValueError(f"{traj_name}: no camera features were generated")
            if not filled.all():
                missing = int((~filled).sum())
                raise ValueError(f"{traj_name}: {missing} point features were not assigned from any camera")

            (
                node_feature,
                node_rgb,
                node_camera_index,
                node_pixel_uv,
                node_camera_weight,
            ) = _node_center_semantic_features(
                traj,
                node_xyz=node_xyz,
                camera_names=camera_names,
                camera_patch_grids=camera_patch_grids,
                max_views=node_center_max_views,
            )
            unified_pointcloud = np.concatenate(
                [
                    pointcloud,
                    np.concatenate([node_xyz, node_rgb], axis=-1).astype(np.float32),
                ],
                axis=1,
            )
            unified_feature = np.concatenate([semantic_feature, node_feature], axis=1)
            token_type = np.zeros(unified_pointcloud.shape[:2], dtype=np.int8)
            token_type[:, point_shape[1] :] = 1
            node_center_token_indices = np.arange(
                point_shape[1],
                point_shape[1] + node_xyz.shape[1],
                dtype=np.int32,
            )

            point_parent, point_dataset_name = _require_parent_group(traj, pointcloud_path)
            point_ds = _write_dataset(
                point_parent,
                point_dataset_name,
                unified_pointcloud.astype(np.float32),
                overwrite=overwrite,
            )
            point_ds.attrs["semantic_field_format"] = "rgbd_points_plus_map4d_node_centers_v1"
            point_ds.attrs["rgbd_point_count"] = int(point_shape[1])
            point_ds.attrs["node_center_count"] = int(node_xyz.shape[1])

            parent, dataset_name = _require_parent_group(traj, output_path)
            feature_ds = _write_dataset(parent, dataset_name, unified_feature.astype(h5_dtype), overwrite=overwrite)
            feature_ds.attrs["semantic_feature_source"] = "dinov3_patch_token_projected_to_semantic_field"
            feature_ds.attrs["semantic_feature_model"] = model
            feature_ds.attrs["feature_type"] = "dinov3"
            feature_ds.attrs["semantic_feature_dtype"] = output_dtype
            feature_ds.attrs["weights_path"] = str(weights)
            feature_ds.attrs["third_party_dir"] = str(third_party_dir)
            feature_ds.attrs["semantic_alignment"] = "rgbd_point_source_plus_projected_map4d_node_centers"
            feature_ds.attrs["pointcloud_path"] = pointcloud_path
            feature_ds.attrs["pointcloud_source_path"] = source_path
            feature_ds.attrs["map4d_sidecar_path"] = str(map4d_sidecar_path)
            feature_ds.attrs["rgbd_point_count"] = int(point_shape[1])
            feature_ds.attrs["node_center_count"] = int(node_xyz.shape[1])
            feature_ds.attrs["dinov3_cameras"] = json.dumps(camera_names)
            feature_ds.attrs["dinov3_input_image_size"] = image_size_value or "auto_multiple"
            feature_ds.attrs["dinov3_multiple"] = int(multiple)

            source_group = _semantic_source_group(traj, overwrite)
            _write_dataset(source_group, "token_type", token_type, overwrite)
            _write_dataset(source_group, "node_center_token_indices", node_center_token_indices, overwrite)
            _write_dataset(source_group, "node_center_camera_index", node_camera_index, overwrite)
            _write_dataset(source_group, "node_center_pixel_uv", node_pixel_uv, overwrite)
            _write_dataset(source_group, "node_center_camera_weight", node_camera_weight, overwrite)
            source_group.attrs["semantic_field_format"] = "rgbd_points_plus_map4d_node_centers_v1"
            source_group.attrs["token_type_0"] = "rgbd_point"
            source_group.attrs["token_type_1"] = "map4d_node_center"
            source_group.attrs["rgbd_point_count"] = int(point_shape[1])
            source_group.attrs["node_center_count"] = int(node_xyz.shape[1])

            row = _validate_shapes(traj_name, point_ds, feature_ds)
            row.update(_require_dino_provenance(traj_name, feature_ds))
            row.update(
                {
                    "cameras": camera_names,
                    "camera_patch_feature_shapes": camera_feature_shapes,
                    "camera_resized_hw": resize_shapes,
                    "rgbd_point_count": int(point_shape[1]),
                    "node_center_count": int(node_xyz.shape[1]),
                    "semantic_field_point_count": int(unified_pointcloud.shape[1]),
                }
            )
            rows.append(row)
            feature_dims.add(row["semantic_feature_dim"])
            print(
                f"{traj_name}: point_cloud={tuple(point_ds.shape)}, "
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
        "map4d_sidecar_path": str(map4d_sidecar_path),
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
    map4d_sidecar_path: Optional[Path],
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
    node_center_max_views: int,
) -> Dict[str, object]:
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")
    if map4d_sidecar_path is None:
        raise ValueError("--map4d-sidecar-path is required for unified Semantic Field data")
    if not map4d_sidecar_path.exists():
        raise FileNotFoundError(f"Map4D sidecar file not found: {map4d_sidecar_path}")
    if mode == "dinov3":
        return _generate_dinov3_point_features(
            demo_path,
            map4d_sidecar_path=map4d_sidecar_path,
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
            node_center_max_views=node_center_max_views,
        )
    if mode != "existing_dino":
        raise ValueError(f"Unsupported mode={mode!r}. Use dinov3 or existing_dino.")

    rows = []
    feature_dims = set()
    with h5py.File(demo_path, "r") as f, h5py.File(map4d_sidecar_path, "r") as sidecar:
        traj_names = sorted([key for key in f.keys() if key.startswith("traj_")], key=_traj_sort_key)
        if num_traj is not None:
            traj_names = traj_names[:num_traj]
        if not traj_names:
            raise ValueError(f"No traj_* groups found in {demo_path}")

        for traj_name in traj_names:
            traj = f[traj_name]
            map4d = _read_sidecar_map4d(sidecar, traj_name)
            node_count = int(map4d.shape[1])
            pointcloud = _get_dataset(traj, pointcloud_path)
            feature = _get_dataset(traj, output_path)

            row = _validate_shapes(traj_name, pointcloud, feature)
            source = traj.get("obs", {}).get("semantic_field_source") if "obs" in traj else None
            if not isinstance(source, h5py.Group) or "token_type" not in source:
                raise KeyError(f"{traj_name} missing obs/semantic_field_source/token_type")
            token_type = np.asarray(source["token_type"][()])
            if token_type.shape != tuple(pointcloud.shape[:2]):
                raise ValueError(
                    f"{traj_name}: token_type shape {token_type.shape} must match semantic field "
                    f"[T,P+N] {tuple(pointcloud.shape[:2])}"
                )
            if token_type.shape[1] <= node_count:
                raise ValueError(f"{traj_name}: semantic field has no RGB-D prefix before {node_count} nodes")
            if not np.all(token_type[:, -node_count:] == 1):
                raise ValueError(f"{traj_name}: last {node_count} token_type entries must be node centers")
            if not np.all(token_type[:, : -node_count] == 0):
                raise ValueError(f"{traj_name}: RGB-D token_type prefix must be 0")
            if not np.allclose(pointcloud[:, -node_count:, :3], map4d[..., 0:3], atol=1e-5):
                raise ValueError(f"{traj_name}: last {node_count} point_cloud xyz rows must match map4d positions")
            row.update(_require_dino_provenance(traj_name, feature))
            row["rgbd_point_count"] = int(pointcloud.shape[1] - node_count)
            row["node_center_count"] = node_count
            row["semantic_field_point_count"] = int(pointcloud.shape[1])
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
        "map4d_sidecar_path": str(map4d_sidecar_path),
        "semantic_feature_dim": int(next(iter(feature_dims))),
        "num_trajectories": len(rows),
        "trajectories": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", required=True, type=Path)
    parser.add_argument("--mode", choices=["dinov3", "existing_dino"], default="dinov3")
    parser.add_argument(
        "--map4d-sidecar-path",
        type=Path,
        required=True,
        help="Keyframe/context sidecar containing traj_*/map4d. Node centers are appended from map4d[...,0:3].",
    )
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
    parser.add_argument("--node-center-max-views", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_point_semantic_features(
        args.demo_path,
        mode=args.mode,
        map4d_sidecar_path=args.map4d_sidecar_path,
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
        node_center_max_views=args.node_center_max_views,
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
