#!/usr/bin/env python
"""Generate frozen DINOv3 features for ManiSkill demonstration HDF5 files.

The default output is a compact sidecar HDF5:

    traj_0/dino_feature/<camera>        [T, embed_dim]
    traj_0/dinov3/<camera>/patch_mean   [T, embed_dim]

Use --embed-in-output-demo to copy the source demo and write
obs/dino_feature/<camera> inside each trajectory, which is directly readable by
ManiSkillMap4DDataset as separate camera tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DP_ROOT = PROJECT_ROOT / "baselines" / "diffusion_policy"
for path in (PROJECT_ROOT, DP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import numpy as np
import torch
import torch.nn.functional as F

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


def _task_traj_name(task_name: str) -> str:
    mapping = {
        "StackCube-v1": "StackCube",
        "PlugCharger-v1": "PlugCharger",
    }
    return mapping.get(task_name, task_name.removesuffix("-v1"))


def _find_demo_path(
    *,
    root_dir: Path,
    task_name: str,
    control_mode: str,
    dataset_dir: Optional[Path],
) -> Path:
    data_dir = dataset_dir or root_dir / "dataset" / "ManiSkill" / task_name / "motionplanning"
    traj_name = _task_traj_name(task_name)
    if control_mode == "auto":
        modes = ("pd_ee_delta_pose", "pd_ee_delta_pos")
    else:
        modes = (control_mode,)
    tried = []
    for mode in modes:
        candidate = data_dir / f"{traj_name}.rgb.{mode}.physx_cpu.filtered.h5"
        tried.append(candidate)
        if candidate.exists():
            return candidate
    tried_text = "\n  ".join(str(path) for path in tried)
    raise FileNotFoundError(
        f"Filtered demo file not found under {data_dir}. Tried:\n  {tried_text}"
    )


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
                Path("/data2/zehao/MAP4D/checkpoints/dinov3") / filename,
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
    model_root = Path("/data2/zehao/models")
    if filename and model_root.exists():
        matches = sorted(model_root.rglob(filename))
        if matches:
            return matches[0]
    hint = "\n  ".join(str(path) for path in candidates) if candidates else "(no known filename)"
    raise FileNotFoundError(
        "DINOv3 weights not found. Pass --weights-path or set DINOV3_WEIGHTS_PATH. "
        f"Checked:\n  {hint}"
    )


def _parse_cameras(value: str) -> Optional[Tuple[str, ...]]:
    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _discover_cameras(traj: h5py.Group, requested: Optional[Sequence[str]]) -> List[str]:
    sensor_data = traj.get("obs", {}).get("sensor_data") if "obs" in traj else None
    if not isinstance(sensor_data, h5py.Group):
        raise KeyError(f"{traj.name} has no obs/sensor_data group")

    if requested is not None:
        missing = [
            name
            for name in requested
            if name not in sensor_data or "rgb" not in sensor_data[name]
        ]
        if missing:
            raise KeyError(f"{traj.name} missing requested RGB cameras: {missing}")
        return list(requested)

    cameras = [
        name
        for name in sorted(sensor_data.keys())
        if isinstance(sensor_data[name], h5py.Group) and "rgb" in sensor_data[name]
    ]
    if not cameras:
        raise KeyError(f"{traj.name} contains no RGB cameras under obs/sensor_data")
    return cameras


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


def _pool_features(feats: Dict[str, torch.Tensor], pool: str) -> torch.Tensor:
    if pool == "patch_mean":
        return feats["x_norm_patchtokens"].mean(dim=1)
    if pool == "cls":
        return feats["x_norm_clstoken"]
    raise ValueError(f"Unsupported pool={pool!r}")


def _encode_rgb_sequence(
    rgb: np.ndarray,
    *,
    backbone: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: Optional[Tuple[int, int]],
    multiple: int,
    pool: str,
    amp: bool,
) -> np.ndarray:
    features = []
    use_amp = amp and device.type == "cuda"
    for start in range(0, rgb.shape[0], batch_size):
        batch = rgb[start : start + batch_size]
        x = _preprocess_rgb(
            batch,
            device=device,
            image_size=image_size,
            multiple=multiple,
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_amp):
            feats = backbone.forward_features(x)
            pooled = _pool_features(feats, pool)
        features.append(pooled.float().cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32)


def _write_dataset(group: h5py.Group, name: str, value: np.ndarray, overwrite: bool) -> None:
    if name in group:
        if not overwrite:
            raise FileExistsError(f"{group.name}/{name} exists. Pass --overwrite to replace it.")
        del group[name]
    group.create_dataset(name, data=value, compression="gzip")


def _require_group(parent: h5py.Group, name: str, overwrite: bool) -> h5py.Group:
    if name in parent and isinstance(parent[name], h5py.Dataset):
        if not overwrite:
            raise FileExistsError(f"{parent.name}/{name} exists as a dataset. Pass --overwrite to replace it.")
        del parent[name]
    return parent.require_group(name)


def _copy_h5_file(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key in src.keys():
        src.copy(src[key], dst, name=key)


def build_dinov3_features(
    demo_path: Path,
    output_path: Path,
    *,
    model: str,
    weights_path: Path,
    third_party_dir: Path,
    cameras: Optional[Sequence[str]],
    batch_size: int,
    num_traj: Optional[int],
    device_name: str,
    image_size: Optional[Tuple[int, int]],
    multiple: int,
    pool: str,
    amp: bool,
    overwrite: bool,
    embed_in_output_demo: bool,
    in_place: bool,
    concat_camera_features: bool,
) -> Dict[str, object]:
    if output_path.exists() and not overwrite and not in_place:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    backbone = _load_backbone(str(third_party_dir), str(weights_path), model)
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    summary_rows = []
    mode = "r+" if in_place else "r"
    out_mode = None if in_place else "w"

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
            f_out = h5py.File(output_path, out_mode)
            if embed_in_output_demo:
                _copy_h5_file(f_in, f_out)

        try:
            f_out.attrs["source_demo_path"] = str(demo_path)
            f_out.attrs["feature_type"] = "dinov3"
            f_out.attrs["model"] = model
            f_out.attrs["weights_path"] = str(weights_path)
            f_out.attrs["third_party_dir"] = str(third_party_dir)
            f_out.attrs["pool"] = pool
            f_out.attrs["output_kind"] = (
                "in_place_demo"
                if in_place
                else "embedded_demo"
                if embed_in_output_demo
                else "feature_sidecar"
            )
            if not embed_in_output_demo and not in_place:
                f_out.attrs["target_format"] = "dinov3_feature_sidecar"

            for traj_name in traj_names:
                source_traj = f_in[traj_name]
                camera_names = _discover_cameras(source_traj, cameras)
                camera_features = []
                camera_shapes = {}
                expected_len = None
                for camera_name in camera_names:
                    rgb = source_traj["obs"]["sensor_data"][camera_name]["rgb"][()]
                    if expected_len is None:
                        expected_len = rgb.shape[0]
                    elif rgb.shape[0] != expected_len:
                        raise ValueError(
                            f"{traj_name}: camera {camera_name} length {rgb.shape[0]} "
                            f"!= expected {expected_len}"
                        )
                    feature = _encode_rgb_sequence(
                        rgb,
                        backbone=backbone,
                        device=device,
                        batch_size=batch_size,
                        image_size=image_size,
                        multiple=multiple,
                        pool=pool,
                        amp=amp,
                    )
                    camera_features.append(feature)
                    camera_shapes[camera_name] = list(feature.shape)

                    traj_out = f_out[traj_name] if traj_name in f_out else f_out.create_group(traj_name)
                    dinov3_group = traj_out.require_group("dinov3").require_group(camera_name)
                    _write_dataset(dinov3_group, pool, feature, overwrite=overwrite)

                stacked = np.stack(camera_features, axis=1)
                traj_out = f_out[traj_name] if traj_name in f_out else f_out.create_group(traj_name)
                if concat_camera_features:
                    concat = np.concatenate(camera_features, axis=-1)
                    if embed_in_output_demo or in_place:
                        obs_group = traj_out.require_group("obs")
                        _write_dataset(obs_group, "dino_feature", concat, overwrite=overwrite)
                    else:
                        _write_dataset(traj_out, "dino_feature", concat, overwrite=overwrite)
                else:
                    if embed_in_output_demo or in_place:
                        feature_group = _require_group(traj_out.require_group("obs"), "dino_feature", overwrite)
                    else:
                        feature_group = _require_group(traj_out, "dino_feature", overwrite)
                    for camera_name, feature in zip(camera_names, camera_features):
                        _write_dataset(feature_group, camera_name, feature, overwrite=overwrite)

                traj_out.attrs["dinov3_cameras"] = json.dumps(camera_names)
                traj_out.attrs["dinov3_feature_dim"] = int(stacked.shape[-1])
                traj_out.attrs["dinov3_num_cameras"] = int(stacked.shape[1])
                traj_out.attrs["dinov3_num_frames"] = int(stacked.shape[0])
                summary_rows.append(
                    {
                        "traj": traj_name,
                        "num_frames": int(stacked.shape[0]),
                        "cameras": camera_names,
                        "camera_feature_shapes": camera_shapes,
                        "feature_shape": list(stacked.shape),
                        "concat_shape": list(np.concatenate(camera_features, axis=-1).shape)
                        if concat_camera_features
                        else None,
                    }
                )
                print(
                    f"{traj_name}: frames={stacked.shape[0]}, cameras={camera_names}, "
                    f"dino_feature={tuple(stacked.shape)}",
                    flush=True,
                )
        finally:
            if not in_place:
                f_out.close()

    embed_dim = int(backbone.embed_dim)
    return {
        "demo_path": str(demo_path),
        "output_path": str(demo_path if in_place else output_path),
        "model": model,
        "weights_path": str(weights_path),
        "third_party_dir": str(third_party_dir),
        "pool": pool,
        "embed_dim": embed_dim,
        "feature_dim": embed_dim,
        "num_trajectories": len(summary_rows),
        "embed_in_output_demo": embed_in_output_demo,
        "in_place": in_place,
        "concat_camera_features": concat_camera_features,
        "trajectories": summary_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", default=None, help="Input ManiSkill HDF5 demo.")
    parser.add_argument("--output-path", default=None, help="Output sidecar/demo HDF5 path.")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--root-dir", default="/data2/zehao/MAP4D")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--task-name", default="StackCube-v1")
    parser.add_argument(
        "--control-mode",
        default="auto",
        choices=["auto", "pd_ee_delta_pose", "pd_ee_delta_pos"],
    )
    parser.add_argument("--model", default="dinov3_vits16", choices=sorted(MODEL_WEIGHT_FILES))
    parser.add_argument("--weights-path", default=None)
    parser.add_argument(
        "--third-party-dir",
        default=str(PROJECT_ROOT / "map4d" / "backbone" / "model" / "vision" / "dinov3"),
    )
    parser.add_argument("--cameras", default="auto", help="Comma-separated cameras, or auto.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-traj", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--image-size", default=None, help="Optional resize, e.g. 224 or 224x224.")
    parser.add_argument("--multiple", type=int, default=16, help="Auto-resize H/W to this multiple.")
    parser.add_argument("--pool", default="patch_mean", choices=["patch_mean", "cls"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--embed-in-output-demo",
        action="store_true",
        help="Copy the full source demo and write per-camera traj/obs/dino_feature into the copy.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write per-camera obs/dino_feature into --demo-path directly.",
    )
    parser.add_argument(
        "--concat-camera-features",
        action="store_true",
        help="Legacy mode: concatenate camera features into one [T, num_cameras * dim] dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    demo_path = (
        Path(args.demo_path)
        if args.demo_path
        else _find_demo_path(
            root_dir=root_dir,
            task_name=args.task_name,
            control_mode=args.control_mode,
            dataset_dir=dataset_dir,
        )
    )
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    if args.in_place:
        output_path = demo_path
    elif args.output_path:
        output_path = Path(args.output_path)
    elif args.embed_in_output_demo:
        output_path = demo_path.with_suffix("").with_name(
            f"{demo_path.stem}.with_dinov3_{args.model}.h5"
        )
    else:
        output_path = demo_path.with_suffix("").with_name(
            f"{demo_path.stem}.dinov3_{args.model}.h5"
        )

    weights_path = _find_weights_path(args.model, args.weights_path)
    third_party_dir = Path(args.third_party_dir)
    if not third_party_dir.exists():
        raise FileNotFoundError(f"DINOv3 third_party dir not found: {third_party_dir}")

    summary = build_dinov3_features(
        demo_path,
        output_path,
        model=args.model,
        weights_path=weights_path,
        third_party_dir=third_party_dir,
        cameras=_parse_cameras(args.cameras),
        batch_size=args.batch_size,
        num_traj=args.num_traj,
        device_name=args.device,
        image_size=_parse_image_size(args.image_size),
        multiple=args.multiple,
        pool=args.pool,
        amp=not args.no_amp,
        overwrite=args.overwrite,
        embed_in_output_demo=args.embed_in_output_demo,
        in_place=args.in_place,
        concat_camera_features=args.concat_camera_features,
    )

    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else Path(summary["output_path"]).with_suffix(".summary.json")
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
