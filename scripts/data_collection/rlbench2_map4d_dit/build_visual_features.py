#!/usr/bin/env python3
"""Build RLBench2 point-cloud and feature npy files for Map4D DiT datasets."""

from __future__ import annotations

import argparse
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rlbench.backend.const import DEPTH_SCALE
from rlbench.backend.utils import image_to_float_array
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CAMERAS = (
    "front",
    "over_shoulder_left",
    "over_shoulder_right",
    "overhead",
    "wrist_left",
    "wrist_right",
)

DEFAULT_DINOV2_REPO = (
    PROJECT_ROOT.parent / "PPI" / "repos" / "dinov2"
)
DEFAULT_DINOV2_WEIGHTS = (
    PROJECT_ROOT.parent
    / "PPI"
    / "outputs"
    / "2026-06-20"
    / "02-32-05"
    / "pretrained_models"
    / "hub"
    / "checkpoints"
    / "dinov2_vits14_pretrain.pth"
)


def _episode_index(path: Path | str) -> int:
    match = re.search(r"episode(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def _list_members(squashfs: Path) -> list[str]:
    out = subprocess.check_output(["unsquashfs", "-lc", str(squashfs)], text=True)
    members = []
    for line in out.splitlines():
        item = line.strip()
        if item.startswith("squashfs-root/"):
            item = item[len("squashfs-root/") :]
        members.append(item)
    return members


def _extract_members(squashfs: Path, members: list[str], dest: Path) -> None:
    extract_list = dest / "extract_files.txt"
    extract_list.write_text("\n".join(members) + "\n")
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


def _prepare_raw_dir(input_path: Path, output_raw_dir: Path, max_episodes: int | None) -> Path:
    if input_path.suffix != ".squashfs":
        return input_path

    members = _list_members(input_path)
    selected = []
    for member in members:
        if "/episodes/episode" not in member:
            continue
        if not (
            member.endswith("low_dim_obs.pkl")
            or member.endswith("variation_descriptions.pkl")
            or "_rgb/rgb_" in member
            or "_depth/depth_" in member
        ):
            continue
        ep = _episode_index(member)
        if max_episodes is not None and ep >= max_episodes:
            continue
        selected.append(member)
    if not selected:
        raise FileNotFoundError(f"No usable RLBench2 episode members found in {input_path}")

    output_raw_dir.mkdir(parents=True, exist_ok=True)
    _extract_members(input_path, selected, output_raw_dir)
    raw_root = output_raw_dir / "all_variations" / "episodes"
    if not raw_root.exists():
        raise FileNotFoundError(f"Extraction did not create {raw_root}")
    return raw_root


def _decode_depth(path: Path, near: float, far: float) -> np.ndarray:
    image = np.asarray(Image.open(path))
    depth = image_to_float_array(image, DEPTH_SCALE)
    return (near + depth.astype(np.float32) * (far - near)).astype(np.float32)


def _load_rgb(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return rgb


def _load_rgb_u8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _camera_points(depth: np.ndarray, rgb: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float32)
    z = depth.reshape(-1)
    valid = np.isfinite(z) & (z > 1e-4)
    if not np.any(valid):
        return np.zeros((0, 6), dtype=np.float32)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = (u.reshape(-1) - cx) * z / fx
    y = (v.reshape(-1) - cy) * z / fy
    cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)[valid]
    world = cam @ np.asarray(extrinsics, dtype=np.float32).T
    color = rgb.reshape(-1, 3)[valid]
    return np.concatenate([world[:, :3], color], axis=-1).astype(np.float32)


def _sample_points(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError("No valid RGB-D points were reconstructed; point-cloud fallback is not allowed")
    replace = points.shape[0] < num_points
    idx = rng.choice(points.shape[0], size=num_points, replace=replace)
    return points[idx].astype(np.float32)


def _world_to_camera_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.shape == (4, 4):
        extrinsics = extrinsics[:3]
    if extrinsics.shape != (3, 4):
        raise ValueError(f"camera extrinsics must be [3,4] or [4,4], got {extrinsics.shape}")
    c = extrinsics[:3, 3:4]
    r = extrinsics[:3, :3]
    r_inv = r.T
    r_inv_c = r_inv @ c
    return np.concatenate([r_inv, -r_inv_c], axis=-1).astype(np.float32)


def _resolve_dinov2_path(path: str | None, default: Path, name: str) -> Path:
    resolved = Path(path).expanduser() if path else default
    if not resolved.exists():
        raise FileNotFoundError(
            f"{name} not found: {resolved}. Set the corresponding --dino-* argument."
        )
    return resolved


class Dinov2PointFeatureExtractor:
    def __init__(
        self,
        *,
        repo_path: Path,
        weights_path: Path,
        device: str,
        patch_h: int,
        patch_w: int,
        batch_size: int,
    ) -> None:
        import torch
        import torchvision.transforms as T
        from map4d.backbone.model.vision.semantic_feature_extractor import (
            interpolate_feats,
            project_points_coords,
        )

        self.torch = torch
        self.transforms = T
        self.interpolate_feats = interpolate_feats
        self.project_points_coords = project_points_coords
        self.device = torch.device(device)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"DINO batch_size must be positive, got {self.batch_size}")
        self.feat_dim = 384
        self.mu = 0.02
        self.model = torch.hub.load(
            str(repo_path),
            "dinov2_vits14",
            source="local",
            skip_validation=True,
        )
        state_dict = torch.load(str(weights_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.transform = T.Compose(
            [
                T.Resize((self.patch_h * 14, self.patch_w * 14)),
                T.CenterCrop((self.patch_h * 14, self.patch_w * 14)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _extract_patch_features(self, images: np.ndarray):
        torch = self.torch
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected images [K,H,W,3], got {images.shape}")
        patch_features = []
        with torch.no_grad():
            for start in range(0, images.shape[0], self.batch_size):
                end = min(start + self.batch_size, images.shape[0])
                tensor = torch.zeros(
                    (end - start, 3, self.patch_h * 14, self.patch_w * 14),
                    device=self.device,
                )
                for idx, image_idx in enumerate(range(start, end)):
                    tensor[idx] = self.transform(Image.fromarray(images[image_idx]))[:3]
                features_dict = self.model.forward_features(tensor)
                features = features_dict["x_norm_patchtokens"]
                patch_features.append(features.reshape((end - start, self.patch_h, self.patch_w, self.feat_dim)))
        return torch.cat(patch_features, dim=0)

    @staticmethod
    def _valid_feature_rows(feature: np.ndarray) -> np.ndarray:
        return np.isfinite(feature).all(axis=-1) & (np.abs(feature).sum(axis=-1) > 0)

    def _project_patch_features(self, points_xyz: np.ndarray, obs: dict, patch_features) -> np.ndarray:
        torch = self.torch
        pts = torch.as_tensor(points_xyz, dtype=torch.float32, device=self.device)
        pose = torch.as_tensor(obs["pose"], dtype=torch.float32, device=self.device)
        intrinsics = torch.as_tensor(obs["K"], dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(obs["depth"], dtype=torch.float32, device=self.device)
        pts_2d, valid_mask, pts_depth = self.project_points_coords(pts, pose, intrinsics)
        pts_depth = pts_depth[..., 0]
        inter_depth = self.interpolate_feats(
            depth.unsqueeze(1),
            pts_2d,
            h=obs["depth"].shape[1],
            w=obs["depth"].shape[2],
            padding_mode="zeros",
            align_corners=True,
            inter_mode="nearest",
        )[..., 0]
        dist = inter_depth - pts_depth
        dist_valid = (inter_depth > 0.0) & valid_mask & (dist > -self.mu)
        dist_weight = torch.exp(torch.clamp(self.mu - torch.abs(dist), max=0) / self.mu)
        all_invalid = dist_valid.float().sum(0) == 0
        inter_feat = self.interpolate_feats(
            patch_features.permute(0, 3, 1, 2),
            pts_2d,
            h=obs["depth"].shape[1],
            w=obs["depth"].shape[2],
            padding_mode="zeros",
            align_corners=True,
            inter_mode="bilinear",
        )
        denom = dist_valid.float().sum(0).unsqueeze(-1)
        if torch.any(denom <= 0):
            raise ValueError(
                f"DINO projection failed for {int(torch.sum(denom <= 0).item())} sampled points"
            )
        feat = (inter_feat * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) / denom
        feat[all_invalid] = 0.0
        feature_np = feat.detach().cpu().numpy().astype(np.float32)
        if not np.all(self._valid_feature_rows(feature_np)):
            invalid = int(np.sum(~self._valid_feature_rows(feature_np)))
            raise ValueError(f"DINO feature contains {invalid} invalid/zero point rows")
        return feature_np

    def extract(self, points_xyz: np.ndarray, obs: dict) -> np.ndarray:
        patch_features = self._extract_patch_features(obs["color"])
        return self._project_patch_features(points_xyz, obs, patch_features)

    def extract_many(self, points_xyz_list: list[np.ndarray], obs_list: list[dict]) -> list[np.ndarray]:
        if len(points_xyz_list) != len(obs_list):
            raise ValueError(
                f"points/obs batch length mismatch: {len(points_xyz_list)} vs {len(obs_list)}"
            )
        if not obs_list:
            return []
        camera_count = int(obs_list[0]["color"].shape[0])
        if camera_count <= 0:
            raise ValueError("DINO frame batch must contain at least one camera")
        for idx, obs in enumerate(obs_list):
            if int(obs["color"].shape[0]) != camera_count:
                raise ValueError(
                    f"Frame {idx} camera count {obs['color'].shape[0]} does not match {camera_count}"
                )
        colors = np.concatenate([obs["color"] for obs in obs_list], axis=0)
        patch_features = self._extract_patch_features(colors)
        if patch_features.shape[0] != len(obs_list) * camera_count:
            raise ValueError(
                f"DINO patch feature batch mismatch: got {patch_features.shape[0]}, "
                f"expected {len(obs_list) * camera_count}"
            )
        frame_patch_features = patch_features.reshape(
            len(obs_list),
            camera_count,
            self.patch_h,
            self.patch_w,
            self.feat_dim,
        )
        features = []
        for points_xyz, obs, frame_features in zip(points_xyz_list, obs_list, frame_patch_features):
            features.append(self._project_patch_features(points_xyz, obs, frame_features))
        return features


def _build_episode(
    episode_dir: Path,
    pcd_episode_dir: Path,
    dino_episode_dir: Path,
    pcd_type: str,
    cameras: tuple[str, ...],
    num_points: int,
    feature_extractor: Dinov2PointFeatureExtractor,
    max_frames: int | None,
    seed: int,
    frames_per_dino_batch: int,
) -> tuple[int, int]:
    ep = _episode_index(episode_dir)
    with (episode_dir / "low_dim_obs.pkl").open("rb") as f:
        demo = pickle.load(f)
    frame_count = len(demo) if max_frames is None else min(len(demo), max_frames)
    pcd_out = pcd_episode_dir / f"episode{ep}" / pcd_type
    dino_out = dino_episode_dir / f"episode{ep}" / pcd_type
    pcd_out.mkdir(parents=True, exist_ok=True)
    dino_out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + ep)
    written = 0
    frame_iter = tqdm(
        range(0, frame_count, frames_per_dino_batch),
        desc=f"episode{ep}",
        dynamic_ncols=True,
        leave=False,
        total=(frame_count + frames_per_dino_batch - 1) // frames_per_dino_batch,
    )
    for batch_start in frame_iter:
        batch_frames = range(batch_start, min(batch_start + frames_per_dino_batch, frame_count))
        batch_point_clouds = []
        batch_feature_obs = []
        batch_frame_ids = []
        for frame in batch_frames:
            obs = demo[frame]
            frame_points = []
            frame_images = []
            frame_depths = []
            frame_extrinsics = []
            frame_intrinsics = []
            for camera in cameras:
                rgb_path = episode_dir / f"{camera}_rgb" / f"rgb_{frame:04d}.png"
                depth_path = episode_dir / f"{camera}_depth" / f"depth_{frame:04d}.png"
                if not rgb_path.exists() or not depth_path.exists():
                    raise FileNotFoundError(f"Missing RGB-D files for {camera} frame {frame} in {episode_dir}")
                misc = obs.misc
                intrinsics = misc[f"{camera}_camera_intrinsics"]
                extrinsics = misc[f"{camera}_camera_extrinsics"]
                near = float(misc[f"{camera}_camera_near"])
                far = float(misc[f"{camera}_camera_far"])
                depth = _decode_depth(depth_path, near=near, far=far)
                rgb = _load_rgb(rgb_path)
                frame_points.append(_camera_points(depth, rgb, intrinsics, extrinsics))
                frame_images.append(_load_rgb_u8(rgb_path))
                frame_depths.append(depth)
                frame_intrinsics.append(np.asarray(intrinsics, dtype=np.float32))
                frame_extrinsics.append(_world_to_camera_extrinsics(extrinsics))
            if not frame_points:
                raise ValueError(f"No camera points were reconstructed for {episode_dir} frame {frame}")
            fused = np.concatenate(frame_points, axis=0)
            point_cloud = _sample_points(fused, num_points=num_points, rng=rng)
            batch_point_clouds.append(point_cloud)
            batch_feature_obs.append(
                {
                    "color": np.stack(frame_images, axis=0),
                    "depth": np.stack(frame_depths, axis=0),
                    "pose": np.stack(frame_extrinsics, axis=0),
                    "K": np.stack(frame_intrinsics, axis=0),
                }
            )
            batch_frame_ids.append(frame)
        dino_features = feature_extractor.extract_many(
            [point_cloud[:, :3] for point_cloud in batch_point_clouds],
            batch_feature_obs,
        )
        for frame, point_cloud, dino_feature in zip(batch_frame_ids, batch_point_clouds, dino_features):
            if dino_feature.shape != (point_cloud.shape[0], feature_extractor.feat_dim):
                raise ValueError(
                    f"DINO feature shape {dino_feature.shape} does not match point cloud {point_cloud.shape}"
                )
            np.save(pcd_out / f"step{frame:03d}.npy", point_cloud)
            np.save(dino_out / f"step{frame:03d}.npy", dino_feature)
            written += 1
    return ep, written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="RLBench2 .squashfs or extracted episodes dir.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pcd-type", default="rgb_pcd_rps6144")
    parser.add_argument("--num-points", type=int, default=6144)
    parser.add_argument("--feature-dim", type=int, default=384, help=argparse.SUPPRESS)
    parser.add_argument(
        "--feature-mode",
        choices=["dinov2"],
        default="dinov2",
        help="Generate real per-point DINO features.",
    )
    parser.add_argument("--dinov2-repo-path", default=str(DEFAULT_DINOV2_REPO))
    parser.add_argument("--dinov2-weights-path", default=str(DEFAULT_DINOV2_WEIGHTS))
    parser.add_argument("--dino-device", default="cuda:0")
    parser.add_argument("--dino-batch-size", type=int, default=32)
    parser.add_argument("--dino-patch-h", type=int, default=64)
    parser.add_argument("--dino-patch-w", type=int, default=64)
    parser.add_argument("--cameras", default="front,over_shoulder_left,over_shoulder_right,overhead,wrist_left,wrist_right")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    cameras = tuple(item.strip() for item in args.cameras.split(",") if item.strip())
    unknown = sorted(set(cameras) - set(CAMERAS))
    if unknown:
        raise ValueError(f"Unknown cameras: {unknown}")
    if args.dino_batch_size <= 0:
        raise ValueError(f"--dino-batch-size must be positive, got {args.dino_batch_size}")
    if args.dino_batch_size % len(cameras) != 0:
        raise ValueError(
            f"--dino-batch-size must be a multiple of camera count {len(cameras)}, "
            f"got {args.dino_batch_size}"
        )
    frames_per_dino_batch = args.dino_batch_size // len(cameras)
    print(
        f"dino_batch_size={args.dino_batch_size} cameras={len(cameras)} "
        f"frames_per_dino_batch={frames_per_dino_batch}",
        flush=True,
    )

    raw_parent = args.output_dir / "raw"
    raw_dir = _prepare_raw_dir(args.input, raw_parent, args.max_episodes)
    if args.copy_raw and args.input.suffix != ".squashfs":
        target = raw_parent / "all_variations" / "episodes"
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(args.input, target)
        raw_dir = target

    pcd_root = args.output_dir / "point_cloud"
    dino_root = args.output_dir / "dino_feature"
    dino_repo = _resolve_dinov2_path(args.dinov2_repo_path, DEFAULT_DINOV2_REPO, "DINOv2 repo")
    dino_weights = _resolve_dinov2_path(args.dinov2_weights_path, DEFAULT_DINOV2_WEIGHTS, "DINOv2 weights")
    feature_extractor = Dinov2PointFeatureExtractor(
        repo_path=dino_repo,
        weights_path=dino_weights,
        device=args.dino_device,
        patch_h=args.dino_patch_h,
        patch_w=args.dino_patch_w,
        batch_size=args.dino_batch_size,
    )
    episode_dirs = sorted(raw_dir.glob("episode*"), key=_episode_index)
    if args.max_episodes is not None:
        episode_dirs = [path for path in episode_dirs if _episode_index(path) < args.max_episodes]
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {raw_dir}")

    total_frames = 0
    episode_iter = tqdm(
        episode_dirs,
        desc="episodes",
        dynamic_ncols=True,
    )
    for episode_dir in episode_iter:
        ep, frames = _build_episode(
            episode_dir,
            pcd_root,
            dino_root,
            args.pcd_type,
            cameras,
            args.num_points,
            feature_extractor,
            args.max_frames,
            args.seed,
            frames_per_dino_batch,
        )
        total_frames += frames
        print(f"episode{ep}: frames={frames}")

    manifest = args.output_dir / "visual_features.env"
    with manifest.open("w") as f:
        f.write(f"RLBENCH2_RAW_DIR={str(raw_dir)!r}\n")
        f.write(f"RLBENCH2_PCD_PATH={str(pcd_root)!r}\n")
        f.write(f"RLBENCH2_DINO_PATH={str(dino_root)!r}\n")
        f.write(f"RLBENCH2_PCD_TYPE={args.pcd_type!r}\n")
        f.write(f"POINTCLOUD_NUM_POINTS={str(args.num_points)!r}\n")
        f.write(f"DINO_BATCH_SIZE={str(args.dino_batch_size)!r}\n")
        f.write(f"FRAMES_PER_DINO_BATCH={str(frames_per_dino_batch)!r}\n")
        f.write("SEMANTIC_FEATURE_DIM='384'\n")
        f.write("SEMANTIC_FEATURE_SOURCE='dinov2_vits14'\n")
    print(f"episodes={len(episode_dirs)} frames={total_frames}")
    print(f"raw_dir={raw_dir}")
    print(f"pcd_path={pcd_root}")
    print(f"dino_path={dino_root}")
    print(manifest)


if __name__ == "__main__":
    main()
