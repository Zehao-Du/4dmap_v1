from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_MAPS4D_DIR = _REPO_ROOT / "map4d" / "representation" / "maps4d"
_DATA_ROOT = _REPO_ROOT.parent / "dataset" / "ManiSkill"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_MAPS4D_DIR) not in sys.path:
    sys.path.insert(0, str(_MAPS4D_DIR))


DEFAULT_TRAJ_PATH = (
    _DATA_ROOT
    / "StackCube-v1"
    / "motionplanning"
    / "StackCube.rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.ep00002_00002.h5"
)
DEFAULT_OUTPUT_DIR = _THIS_FILE.parent / "validation_outputs" / "sam2_stackcube_ep00002"

PROMPT_COLORS = {
    "red cube": (220, 30, 30),
    "green cube": (40, 200, 40),
}


def load_runtime_dependencies() -> None:
    global h5py, np, torch, Image, ImageDraw

    import h5py
    import numpy as np
    import torch
    from PIL import Image, ImageDraw


@dataclass
class ObjectBootstrap:
    prompt: str
    actor_name: str
    obj_id: int
    seg_id: int
    box_xyxy: np.ndarray
    gt_mask: np.ndarray
    sam_mask: np.ndarray
    sam_score: float
    mean_rgb: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a minimal SAM2 StackCube pipeline on one real ManiSkill trajectory.")
    parser.add_argument("--traj-path", type=pathlib.Path, default=DEFAULT_TRAJ_PATH)
    parser.add_argument("--camera", type=str, default="base_camera")
    parser.add_argument("--frame-limit", type=int, default=12, help="How many frames to export / track.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None, help="SAM2 device: cuda, cuda:N, or cpu.")
    parser.add_argument("--sam2-version", type=str, default="sam2.1-hiera-large")
    return parser.parse_args()


def build_stackcube_prompts() -> List[str]:
    import torch
    from maniskill_stackcube import Map4d_StackCube

    sizes = torch.ones((1, 9), dtype=torch.float32)
    positions = torch.zeros((1, 9), dtype=torch.float32)
    rotations = torch.zeros((1, 18), dtype=torch.float32)
    structure_map = Map4d_StackCube(sizes, positions, rotations, clip_model=None)
    prompts = [item["text_prompt"] for item in structure_map.Subgraph_Prompts]
    prompts = [prompt for prompt in prompts if prompt in {"red cube", "green cube"}]
    if prompts != ["red cube", "green cube"]:
        raise RuntimeError(f"Unexpected StackCube prompts: {prompts}")
    return prompts


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


def infer_segmentation_id(seg: np.ndarray, uv: np.ndarray, radius: int = 5) -> int:
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


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Cannot derive box from empty mask")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def classify_color_prompt(mean_rgb: np.ndarray) -> str:
    r, g, b = [float(x) for x in mean_rgb]
    if r > g and r > b:
        return "red cube"
    if g > r and g > b:
        return "green cube"
    raise RuntimeError(f"Could not map actor color from mean_rgb={mean_rgb.tolist()}")


def export_video_frames(rgb_frames: np.ndarray, output_dir: pathlib.Path) -> pathlib.Path:
    frames_dir = output_dir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(rgb_frames):
        Image.fromarray(frame.astype(np.uint8)).save(frames_dir / f"{idx:05d}.jpg")
    return frames_dir


def alpha_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_box(image: np.ndarray, box_xyxy: np.ndarray, color: Tuple[int, int, int], width: int = 3) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    draw.rectangle([float(v) for v in box_xyxy], outline=color, width=width)
    return np.array(pil)


def add_label(image: np.ndarray, text: str, xy: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    x, y = int(xy[0]), int(xy[1])
    draw.rectangle([x, y, x + 210, y + 20], fill=(0, 0, 0))
    draw.text((x + 4, y + 3), text, fill=color)
    return np.array(pil)


def save_first_frame_visualization(
    image: np.ndarray,
    bootstraps: List[ObjectBootstrap],
    output_path: pathlib.Path,
) -> None:
    vis = image.copy()
    for item in bootstraps:
        color = PROMPT_COLORS.get(item.prompt, (255, 255, 0))
        vis = alpha_overlay(vis, item.sam_mask, color)
        vis = draw_box(vis, item.box_xyxy, color)
        x1, y1, _, _ = item.box_xyxy.astype(int)
        vis = add_label(vis, f"{item.prompt} | score={item.sam_score:.3f}", (x1, max(0, y1 - 22)), color)
    Image.fromarray(vis).save(output_path)


def save_tracked_frame_visualization(
    image: np.ndarray,
    bootstraps: List[ObjectBootstrap],
    tracked_masks: Dict[int, np.ndarray],
    gt_masks: Dict[str, np.ndarray],
    ious: Dict[str, float],
    output_path: pathlib.Path,
    frame_idx: int,
) -> None:
    vis = image.copy()
    for item in bootstraps:
        color = PROMPT_COLORS.get(item.prompt, (255, 255, 0))
        pred_mask = tracked_masks[item.obj_id]
        vis = alpha_overlay(vis, pred_mask, color)
        pred_box = mask_to_box(pred_mask)
        vis = draw_box(vis, pred_box, color)
        x1, y1, _, _ = pred_box.astype(int)
        vis = add_label(vis, f"f={frame_idx} {item.prompt} IoU={ious[item.prompt]:.3f}", (x1, max(0, y1 - 22)), color)
    Image.fromarray(vis).save(output_path)


def bootstrap_first_frame(
    h5_group,
    camera_name: str,
    image_predictor,
    prompts: List[str],
) -> tuple[List[ObjectBootstrap], np.ndarray]:
    rgb0 = h5_group[f"obs/sensor_data/{camera_name}/rgb"][0].astype(np.uint8)
    seg0 = h5_group[f"obs/sensor_data/{camera_name}/segmentation"][0, ..., 0].astype(np.int32)
    K0 = h5_group[f"obs/sensor_param/{camera_name}/intrinsic_cv"][0].astype(np.float32)
    w2c0 = make_w2c_4x4(h5_group[f"obs/sensor_param/{camera_name}/extrinsic_cv"][0])

    actor_infos = []
    for actor_name in ("cubeA", "cubeB"):
        actor_state = h5_group[f"env_states/actors/{actor_name}"][0].astype(np.float32)
        uv, _ = project_point(K0, w2c0, actor_state[:3])
        seg_id = infer_segmentation_id(seg0, uv)
        gt_mask = seg0 == seg_id
        mean_rgb = rgb0[gt_mask].mean(axis=0)
        actor_infos.append(
            {
                "actor_name": actor_name,
                "seg_id": int(seg_id),
                "gt_mask": gt_mask,
                "box_xyxy": mask_to_box(gt_mask),
                "mean_rgb": mean_rgb.astype(np.float32),
                "prompt_guess": classify_color_prompt(mean_rgb),
            }
        )

    prompt_to_actor = {}
    fallback_prompt_to_actor_name = {
        "red cube": "cubeA",
        "green cube": "cubeB",
    }
    for prompt in prompts:
        matches = [info for info in actor_infos if info["prompt_guess"] == prompt]
        if len(matches) == 1:
            prompt_to_actor[prompt] = matches[0]
            continue

        fallback_actor_name = fallback_prompt_to_actor_name.get(prompt)
        fallback_matches = [info for info in actor_infos if info["actor_name"] == fallback_actor_name]
        if len(fallback_matches) == 1:
            prompt_to_actor[prompt] = fallback_matches[0]
            continue

        raise RuntimeError(
            f"Prompt-to-actor mapping failed for prompt={prompt!r}; matches={len(matches)}. "
            f"Actor guesses={[info['prompt_guess'] for info in actor_infos]}"
        )

    image_predictor.set_image(rgb0)
    bootstraps: List[ObjectBootstrap] = []
    for obj_id, prompt in enumerate(prompts, start=1):
        info = prompt_to_actor[prompt]
        masks, scores, _ = image_predictor.predict(
            box=info["box_xyxy"][None, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        sam_mask = masks[best_idx].astype(bool)
        bootstraps.append(
            ObjectBootstrap(
                prompt=prompt,
                actor_name=info["actor_name"],
                obj_id=obj_id,
                seg_id=info["seg_id"],
                box_xyxy=info["box_xyxy"],
                gt_mask=info["gt_mask"],
                sam_mask=sam_mask,
                sam_score=float(scores[best_idx]),
                mean_rgb=info["mean_rgb"],
            )
        )
    return bootstraps, rgb0


def compute_gt_masks_for_frame(h5_group, camera_name: str, frame_idx: int, bootstraps: List[ObjectBootstrap]) -> Dict[str, np.ndarray]:
    seg = h5_group[f"obs/sensor_data/{camera_name}/segmentation"][frame_idx, ..., 0].astype(np.int32)
    K = h5_group[f"obs/sensor_param/{camera_name}/intrinsic_cv"][frame_idx].astype(np.float32)
    w2c = make_w2c_4x4(h5_group[f"obs/sensor_param/{camera_name}/extrinsic_cv"][frame_idx])
    out = {}
    for item in bootstraps:
        actor_state = h5_group[f"env_states/actors/{item.actor_name}"][frame_idx].astype(np.float32)
        uv, _ = project_point(K, w2c, actor_state[:3])
        seg_id = infer_segmentation_id(seg, uv)
        out[item.prompt] = seg == seg_id
    return out


def track_video(
    frames_dir: pathlib.Path,
    video_predictor,
    bootstraps: List[ObjectBootstrap],
    frame_limit: int,
) -> Dict[int, Dict[int, np.ndarray]]:
    inference_state = video_predictor.init_state(str(frames_dir))
    for item in bootstraps:
        video_predictor.add_new_mask(
            inference_state,
            frame_idx=0,
            obj_id=item.obj_id,
            mask=item.sam_mask.astype(np.uint8),
        )

    tracked: Dict[int, Dict[int, np.ndarray]] = {}
    for frame_idx, object_ids, mask_logits in video_predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=frame_limit,
    ):
        tracked[frame_idx] = {}
        for i, obj_id in enumerate(object_ids):
            tracked[frame_idx][int(obj_id)] = (mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(bool)
    return tracked


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    from map4d.construction.SAM2.sam2_loader import (
        load_sam2_image_predictor,
        load_sam2_video_predictor,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = build_stackcube_prompts()
    image_predictor = load_sam2_image_predictor(version=args.sam2_version, device=args.device)
    video_predictor = load_sam2_video_predictor(version=args.sam2_version, device=args.device)

    with h5py.File(str(args.traj_path), "r") as f:
        traj = f["traj_0"]
        total_frames = int(traj[f"obs/sensor_data/{args.camera}/rgb"].shape[0])
        frame_limit = min(int(args.frame_limit), total_frames)
        rgb_frames = traj[f"obs/sensor_data/{args.camera}/rgb"][:frame_limit].astype(np.uint8)
        frames_dir = export_video_frames(rgb_frames, output_dir)

        bootstraps, first_image = bootstrap_first_frame(traj, args.camera, image_predictor, prompts)
        save_first_frame_visualization(first_image, bootstraps, output_dir / "first_frame_prompted_masks.png")

        tracked = track_video(frames_dir, video_predictor, bootstraps, frame_limit=frame_limit)

        selected_frames = sorted(set([0, min(4, frame_limit - 1), min(8, frame_limit - 1), frame_limit - 1]))
        per_frame_metrics = []
        for frame_idx in range(frame_limit):
            gt_masks = compute_gt_masks_for_frame(traj, args.camera, frame_idx, bootstraps)
            frame_entry = {"frame_idx": frame_idx, "objects": {}}
            for item in bootstraps:
                pred_mask = tracked[frame_idx][item.obj_id]
                gt_mask = gt_masks[item.prompt]
                frame_entry["objects"][item.prompt] = {
                    "actor_name": item.actor_name,
                    "iou_vs_segmentation": iou(pred_mask, gt_mask),
                    "pred_area": int(pred_mask.sum()),
                    "gt_area": int(gt_mask.sum()),
                }
            per_frame_metrics.append(frame_entry)

            if frame_idx in selected_frames:
                image = traj[f"obs/sensor_data/{args.camera}/rgb"][frame_idx].astype(np.uint8)
                ious = {
                    item.prompt: frame_entry["objects"][item.prompt]["iou_vs_segmentation"]
                    for item in bootstraps
                }
                save_tracked_frame_visualization(
                    image=image,
                    bootstraps=bootstraps,
                    tracked_masks=tracked[frame_idx],
                    gt_masks=gt_masks,
                    ious=ious,
                    output_path=output_dir / f"tracked_frame_{frame_idx:05d}.png",
                    frame_idx=frame_idx,
                )

    initial_metrics = {
        item.prompt: {
            "actor_name": item.actor_name,
            "seg_id": item.seg_id,
            "sam_score": item.sam_score,
            "iou_vs_segmentation": iou(item.sam_mask, item.gt_mask),
            "box_xyxy": [float(v) for v in item.box_xyxy.tolist()],
            "mean_rgb": [float(v) for v in item.mean_rgb.tolist()],
        }
        for item in bootstraps
    }
    summary = {
        "traj_path": str(args.traj_path.resolve()),
        "camera": args.camera,
        "frame_limit": frame_limit,
        "sam2_version": args.sam2_version,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "map_class": "Map4d_StackCube",
        "prompts": prompts,
        "note": (
            "SAM2 itself is not text-promptable. This validator uses the map prompts ['red cube', 'green cube'] "
            "to resolve object identity on the first frame, then uses dataset segmentation-derived boxes as SAM2 geometric prompts."
        ),
        "initial_prompt_metrics": initial_metrics,
        "per_frame_metrics": per_frame_metrics,
        "selected_visualization_frames": selected_frames,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] SAM2 StackCube validation complete")
    print(f"  traj_path   : {args.traj_path.resolve()}")
    print(f"  prompts     : {prompts}")
    print(f"  output_dir  : {output_dir}")
    for prompt, metrics in initial_metrics.items():
        print(
            f"  init {prompt:10s} | actor={metrics['actor_name']} | "
            f"score={metrics['sam_score']:.3f} | first-frame IoU={metrics['iou_vs_segmentation']:.3f}"
        )


if __name__ == "__main__":
    main()
