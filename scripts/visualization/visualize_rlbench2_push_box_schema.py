#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_META = PROJECT_ROOT / "map4d" / "representation" / "maps4d" / "rlbench2_push_box.json"
DEFAULT_EPISODE = PROJECT_ROOT / "dataset" / "rlbench2" / "squashfs-root" / "all_variations" / "episodes" / "episode0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "visualizations" / "rlbench2_push_box_map_overlay.png"

CAMERAS = [
    "front",
    "overhead",
    "over_shoulder_left",
    "over_shoulder_right",
    "wrist_left",
    "wrist_right",
]
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
# Read from CoppeliaSim Shape("cube").get_bounding_box() on 2026-07-07.
# BBox: [-0.096, 0.096, -0.192, 0.192, -0.064, 0.064]
SIM_BOX_SIZE_XYZ = np.array([0.192, 0.384, 0.128], dtype=np.float64)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
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


def cuboid_vertices(center: np.ndarray, quat_xyzw: np.ndarray, size_xyz: np.ndarray) -> np.ndarray:
    half = size_xyz / 2.0
    local = np.array(
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
    return local @ quat_xyzw_to_matrix(quat_xyzw).T + center


def project_points(points_world: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world)
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    points_cam = (world_to_camera @ points_h.T).T[:, :3]
    valid = np.abs(points_cam[:, 2]) > 1e-8
    pixels_h = (intrinsics @ points_cam.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels, valid


def draw_map_overlay(image: Image.Image, pixels: np.ndarray, valid: np.ndarray, semantic: str, camera: str, frame: int) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    def near_frame(point: np.ndarray) -> bool:
        return -width <= point[0] <= 2 * width and -height <= point[1] <= 2 * height

    for a, b in BOX_EDGES:
        if valid[a] and valid[b] and near_frame(pixels[a]) and near_frame(pixels[b]):
            draw.line([tuple(pixels[a]), tuple(pixels[b])], fill=(255, 35, 35, 240), width=3)

    for point, is_valid in zip(pixels, valid):
        if is_valid and near_frame(point):
            x, y = point
            r = 4
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 230, 35, 240), outline=(0, 0, 0, 255))

    if np.any(valid):
        center = np.nanmean(pixels[valid], axis=0)
    else:
        center = np.array([12.0, 32.0], dtype=np.float64)
    tx = float(np.clip(center[0] + 8, 4, width - 150))
    ty = float(np.clip(center[1] - 24, 4, height - 24))
    draw.rectangle((tx - 4, ty - 3, tx + 140, ty + 18), fill=(0, 0, 0, 155))
    draw.text((tx, ty), f"Map4D: {semantic}", fill=(255, 255, 255, 255))
    draw.text((8, 8), f"{camera} frame {frame}", fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image


def load_demo(episode_dir: Path):
    with (episode_dir / "low_dim_obs.pkl").open("rb") as f:
        return pickle.load(f)


def load_semantic(metadata: Path) -> str:
    task_meta = json.loads(metadata.read_text())["rlbench2_push_box"]
    return task_meta["graph"]["node_semantics"][0]


def compose_grid(images: list[Image.Image], columns: int = 3) -> Image.Image:
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (cell_w * columns, cell_h * rows), (24, 24, 24))
    for idx, image in enumerate(images):
        canvas.paste(image, ((idx % columns) * cell_w, (idx // columns) * cell_h))
    return canvas


def output_for_camera(output: Path, camera: str) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}_{camera}{output.suffix}")
    return output / f"rlbench2_push_box_map_overlay_{camera}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay RLBench2 push-box Map4D cuboid on saved RGB frames.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera", choices=[*CAMERAS, "all"], default="front")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--size-xyz",
        type=float,
        nargs=3,
        default=SIM_BOX_SIZE_XYZ.tolist(),
        metavar=("X", "Y", "Z"),
        help="Box size in simulator/object local xyz meters. Default is read from CoppeliaSim cube bbox.",
    )
    args = parser.parse_args()

    semantic = load_semantic(args.metadata)
    demo = load_demo(args.episode_dir)
    obs = demo[args.frame]
    state = np.asarray(obs.task_low_dim_state, dtype=np.float64).reshape(-1)
    center = state[0:3]
    quat_xyzw = state[3:7]
    vertices = cuboid_vertices(center, quat_xyzw, np.asarray(args.size_xyz, dtype=np.float64))

    cameras = CAMERAS if args.camera == "all" else [args.camera]
    overlays = []
    for camera in cameras:
        image_path = args.episode_dir / f"{camera}_rgb" / f"rgb_{args.frame:04d}.png"
        image = Image.open(image_path)
        intrinsics = np.asarray(obs.misc[f"{camera}_camera_intrinsics"], dtype=np.float64)
        camera_to_world = np.asarray(obs.misc[f"{camera}_camera_extrinsics"], dtype=np.float64)
        pixels, valid = project_points(vertices, intrinsics, camera_to_world)
        overlays.append(draw_map_overlay(image, pixels, valid, semantic, camera, args.frame))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.camera == "all":
        compose_grid(overlays).save(args.output)
        print(args.output)
    else:
        output = output_for_camera(args.output, args.camera)
        output.parent.mkdir(parents=True, exist_ok=True)
        overlays[0].save(output)
        print(output)


if __name__ == "__main__":
    main()
