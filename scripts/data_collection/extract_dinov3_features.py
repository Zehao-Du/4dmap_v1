"""Extract DINOv3 CLS-token features from ManiSkill RGB demo HDF5 files.

Produces a sidecar HDF5 with per-trajectory rgb_feature arrays that can be
loaded by the Map4D DiT dataset (ManiSkillMap4DDataset).

Usage:
    python scripts/data_collection/extract_dinov3_features.py \
        --demo-path dataset/ManiSkill/.../demo.h5 \
        --output-path dataset/ManiSkill/.../demo.dinov3_s16.h5 \
        --model-name vit_small_patch16_dinov3 \
        --image-size 224 \
        --batch-size 64

If --output-path is omitted it defaults to <demo>.dinov3_s16.h5.
The output HDF5 has one group per trajectory with a single dataset:
    traj_0/rgb_feature: float32 (T, feat_dim)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Extract DINOv3 features from ManiSkill demos")
    p.add_argument("--demo-path", type=str, required=True, help="Input HDF5 demo file")
    p.add_argument("--output-path", type=str, default=None, help="Output sidecar HDF5")
    p.add_argument(
        "--model-name",
        type=str,
        default="vit_small_patch16_dinov3",
        choices=[
            "vit_small_patch16_dinov3",
            "vit_small_plus_patch16_dinov3",
            "vit_base_patch16_dinov3",
            "vit_large_patch16_dinov3",
        ],
        help="timm model name",
    )
    p.add_argument("--image-size", type=int, default=224, help="Resize images to this size")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    p.add_argument("--num-traj", type=int, default=None, help="Limit number of trajectories")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    p.add_argument(
        "--camera-key",
        type=str,
        default=None,
        help="HDF5 path to RGB images (auto-detected if omitted)",
    )
    return p.parse_args()


def find_rgb_dataset(group: h5py.Group, prefer_key: str | None = None) -> str | None:
    """Find the path to RGB image data inside a trajectory group."""
    if prefer_key and prefer_key in group:
        return prefer_key
    candidates = [
        "obs/sensor_data/base_camera/rgb",
        "obs/sensor_data/hand_camera/rgb",
        "obs/image/base_camera/rgb",
        "obs/image/hand_camera/rgb",
        "obs/rgb",
    ]
    for path in candidates:
        node = group
        parts = path.split("/")
        found = True
        for part in parts:
            if isinstance(node, h5py.Group) and part in node:
                node = node[part]
            else:
                found = False
                break
        if found and isinstance(node, h5py.Dataset):
            return path
    # Recursive search
    result = []

    def _search(g, prefix=""):
        for key in g.keys():
            full = f"{prefix}/{key}" if prefix else key
            if isinstance(g[key], h5py.Dataset) and key.lower() == "rgb":
                result.append(full)
            elif isinstance(g[key], h5py.Group):
                _search(g[key], full)

    _search(group)
    return result[0] if result else None


def build_transform(image_size: int):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(image_size),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    images: np.ndarray,
    transform: T.Compose,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract CLS token features from a batch of RGB images.

    Args:
        images: uint8 array of shape (T, H, W, 3) or (T, C, H, W)
    Returns:
        features: float32 array of shape (T, feat_dim)
    """
    T_len = images.shape[0]
    all_features = []

    for start in range(0, T_len, batch_size):
        end = min(start + batch_size, T_len)
        batch = images[start:end]
        if batch.ndim == 4 and batch.shape[-1] == 3:
            # (B, H, W, 3) -> (B, 3, H, W)
            batch = np.transpose(batch, (0, 3, 1, 2))
        batch_t = torch.from_numpy(batch).float().to(device) / 255.0
        batch_t = transform(batch_t)
        # forward_head with no head -> CLS token
        features = model(batch_t)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0).astype(np.float32)


def main():
    args = parse_args()

    if args.output_path is None:
        stem = Path(args.demo_path).stem
        args.output_path = str(Path(args.demo_path).parent / f"{stem}.dinov3_s16.h5")

    if os.path.exists(args.output_path) and not args.overwrite:
        print(f"Output already exists: {args.output_path}. Use --overwrite to replace.")
        sys.exit(1)

    print(f"Loading model: {args.model_name}")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    feat_dim = model.embed_dim
    print(f"  embed_dim={feat_dim}, image_size={args.image_size}, device={device}")

    transform = build_transform(args.image_size)

    print(f"Reading demos from: {args.demo_path}")
    with h5py.File(args.demo_path, "r") as f_in:
        traj_names = sorted(
            [k for k in f_in.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if args.num_traj is not None:
            traj_names = traj_names[: args.num_traj]
        print(f"  Trajectories to process: {len(traj_names)}")

        # Detect camera key from first trajectory
        camera_key = args.camera_key
        if camera_key is None:
            camera_key = find_rgb_dataset(f_in[traj_names[0]])
            if camera_key is None:
                print("ERROR: Could not find RGB data in the demo file.")
                print("  Available keys in traj_0:")
                f_in[traj_names[0]].visititems(lambda name, obj: print(f"    {name}"))
                sys.exit(1)
        print(f"  Using camera key: {camera_key}")

        # Check image shape
        sample = f_in[traj_names[0]][camera_key]
        print(f"  Image shape per frame: {sample.shape[1:]} (T={sample.shape[0]})")

        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with h5py.File(args.output_path, "w") as f_out:
            f_out.attrs["model_name"] = args.model_name
            f_out.attrs["image_size"] = args.image_size
            f_out.attrs["feat_dim"] = feat_dim
            f_out.attrs["camera_key"] = camera_key

            for traj_name in tqdm(traj_names, desc="Extracting features"):
                rgb_data = f_in[traj_name][camera_key][()]
                features = extract_features(model, rgb_data, transform, args.batch_size, device)
                group = f_out.create_group(traj_name)
                group.create_dataset(
                    "rgb_feature",
                    data=features,
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=4,
                )

    print(f"\nDone. Output: {args.output_path}")
    print(f"  Shape per trajectory: (T, {feat_dim})")
    print(f"  Total trajectories: {len(traj_names)}")


if __name__ == "__main__":
    main()
