#!/usr/bin/env python3
"""Evaluate a Map4D DiT checkpoint with ManiSkill rollout."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import torch
from omegaconf import OmegaConf

from map4d.backbone.eval_maniskill import build_rollout_evaluator
from map4d.backbone.policy.map4d_dit_policy import Map4DDiTPolicy
from map4d.representation.maps4d.metadata import get_task_metadata_value


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("map4d_meta", get_task_metadata_value, replace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the training .hydra/config.yaml.")
    parser.add_argument("--checkpoint", required=True, help="Path to a Map4D DiT .pth.tar checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory for rollout_metrics.jsonl.")
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--sim-backend", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--pointcloud-bbox", default=None, help="'auto' or six numbers: xmin ymin zmin xmax ymax zmax.")
    parser.add_argument("--seed", type=int, default=None, help="Optional ManiSkill reset seed for rollout eval.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-raw-model", action="store_true", help="Load model_state_dict instead of EMA.")
    parser.add_argument("--normalizer-cache", default=None, help="Optional path to load/save normalizer state.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    for stale_key in (
        "dinov3_model",
        "dinov3_weights_path",
        "dinov3_third_party_dir",
        "dinov3_image_size",
        "dinov3_input_multiple",
        "dinov3_amp",
    ):
        if stale_key in cfg.policy.model_cfg:
            del cfg.policy.model_cfg[stale_key]
    cfg.rollout.enabled = True
    cfg.rollout.num_eval_episodes = int(args.num_eval_episodes)
    cfg.rollout.num_eval_envs = int(args.num_eval_envs)
    cfg.rollout.close_after_eval = True
    cfg.logging.mode = "disabled"
    cfg.dataloader.num_workers = 0
    cfg.val_dataloader.num_workers = 0
    if args.num_inference_steps is not None:
        cfg.policy.num_inference_steps = int(args.num_inference_steps)
    if args.max_episode_steps is not None:
        cfg.rollout.max_episode_steps = int(args.max_episode_steps)
    if args.sim_backend is not None:
        cfg.rollout.sim_backend = args.sim_backend
    if args.image_size is not None:
        cfg.rollout.image_size = int(args.image_size)
    if args.pointcloud_bbox is not None:
        cfg.rollout.pointcloud_bbox = args.pointcloud_bbox
    if args.seed is not None:
        cfg.rollout.seed = int(args.seed)

    requested_device = args.device
    if requested_device == "cuda":
        requested_device = "cuda:0"
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_key = "model_state_dict" if args.use_raw_model else "ema_model_state_dict"
    if state_key not in checkpoint or checkpoint[state_key] is None:
        raise KeyError(f"Checkpoint does not contain usable {state_key}: {args.checkpoint}")

    policy: Map4DDiTPolicy = hydra.utils.instantiate(cfg.policy)
    state_dict = checkpoint[state_key]
    has_normalizer = any(key.startswith("normalizer.") for key in state_dict)
    if not has_normalizer:
        if args.normalizer_cache is None or not pathlib.Path(args.normalizer_cache).is_file():
            raise KeyError(
                f"Checkpoint does not contain normalizer state and --normalizer-cache was not provided: {args.checkpoint}"
            )
        from map4d.backbone.model.common.normalizer import LinearNormalizer

        normalizer = LinearNormalizer()
        normalizer.load_state_dict(torch.load(args.normalizer_cache, map_location="cpu", weights_only=False))
        policy.set_normalizer(normalizer)
    policy.load_state_dict(state_dict, strict=True)
    policy.to(device)
    policy.eval()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = build_rollout_evaluator(cfg, device=device, output_dir=str(output_dir))
    if evaluator is None:
        raise RuntimeError("rollout evaluator was not constructed")
    try:
        metrics = evaluator.evaluate(
            policy,
            epoch=int(checkpoint.get("epoch", -1)),
            iteration=int(checkpoint.get("global_step", -1)),
        )
    finally:
        evaluator.close()
    print(metrics)


if __name__ == "__main__":
    main()
