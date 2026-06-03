"""Evaluate DP + map4d checkpoint with GT map4d from the environment."""
import os
import sys
import random
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
import tyro

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs

import train_rgbd as dp_train

# Set module-level device that dp_train.Agent.compute_loss references (not used during eval)
dp_train.device = None


@dataclass
class Args:
    checkpoint: str = ""
    """path to checkpoint .pt file"""
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    env_id: str = "StackCube-v1"
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    max_episode_steps: int = 1000
    sim_backend: str = "physx_cpu"
    control_mode: str = "pd_ee_delta_pos"
    obs_mode: str = "rgb"

    # DP architecture (must match training)
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    visual_encoder: str = "plain_conv"

    # 4D Map arguments (must match training)
    use_map4d: bool = True
    map4d_source: str = "maniskill_gt"
    map4d_task_name: str = "StackCube-v1"
    map4d_strict: bool = True
    map4d_pre_horizon: Optional[int] = None
    map4d_future_horizon: int = 3
    map4d_num_objects: int = 3
    map4d_feature_dim: int = 128
    map4d_node_dim: int = 128
    map4d_relation_dim: int = 64
    map4d_temporal_dim: int = 128
    map4d_pose_weight: float = 1.0
    map4d_penetration_weight: float = 0.1
    map4d_kinematic_weight: float = 0.1
    map4d_vel_limit: float = 0.5
    map4d_acc_limit: float = 1.0
    map4d_rot_vel_limit: float = 1.0
    map4d_rot_acc_limit: float = 2.0
    map4d_penetration_margin: float = 0.0

    # DINOv3 paths (if using dinov3 encoder)
    dinov3_weights_path: str = "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy/checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    dinov3_third_party_dir: str = "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy/third_party/dinov3"


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.checkpoint, "--checkpoint is required"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        control_mode=args.control_mode, reward_mode="sparse",
        obs_mode=args.obs_mode, render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(
        args.env_id, args.num_eval_envs, args.sim_backend,
        env_kwargs, other_kwargs, wrappers=[FlattenRGBDObservationWrapper],
        map4d_source=args.map4d_source if args.use_map4d else None,
        map4d_task_name=args.map4d_task_name,
        map4d_strict=args.map4d_strict,
    )

    print("Creating agent and loading checkpoint...")
    # Use the Agent class from the training script directly
    agent = dp_train.Agent(envs, args).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(ckpt['ema_agent'], strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    print(f"Evaluating {args.num_eval_episodes} episodes with GT map4d...")
    eval_metrics = evaluate(args.num_eval_episodes, agent, envs, device, args.sim_backend)

    print("\n=== Evaluation Results ===")
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
        print(f"  {k}: {eval_metrics[k]:.4f}")
