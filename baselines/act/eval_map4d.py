"""Evaluate ACT + map4d checkpoint with GT map4d from the environment."""
import os
import sys
import json
import time
import random
from dataclasses import dataclass
from functools import partial
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
import tyro

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from act.evaluate_map4d import evaluate_map4d
from act.make_env import make_eval_envs

from map4d.map4d_encoder import Map4d_Encoder
from map4d.encoder import PhysicsLosses

import gymnasium as gym
import torch.nn as nn
import torchvision.transforms as T
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv
from typing import Dict

from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE


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
    include_depth: bool = False
    temporal_agg: bool = True

    # ACT architecture (must match training)
    position_embedding: str = "sine"
    backbone: str = "resnet18"
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False
    kl_weight: float = 10

    # 4D Map arguments (must match training)
    use_map4d: bool = True
    map4d_source: str = "maniskill_gt"
    map4d_task_name: str = "StackCube-v1"
    map4d_pre_horizon: int = 6
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

    # Normalization stats from training data
    demo_path: str = ""
    """demo h5 to compute norm stats (same as training)"""
    num_demos: Optional[int] = None


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, rgb=True, depth=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images_rgb = []
        images_depth = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                resized_rgb = self.transforms(cam_data["rgb"].permute(0, 3, 1, 2))
                images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = (cam_data["depth"].to(torch.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(depth.permute(0, 3, 1, 2))
                images_depth.append(resized_depth)
        rgb = torch.stack(images_rgb, dim=1)
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1)
        observation = common.flatten_state_dict(observation, use_torch=True)
        ret = dict()
        if self.include_state:
            ret["state"] = observation
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = rgb
        elif self.include_rgb and self.include_depth:
            ret["rgb"] = rgb
            ret["depth"] = depth
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = depth
        return ret


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.state_dim = env.single_observation_space['state'].shape[0]
        self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.use_map4d = args.use_map4d
        self.map4d_feature_dim = args.map4d_feature_dim if args.use_map4d else 0
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.use_map4d:
            self.map4d_encoder = Map4d_Encoder(
                num_objects=args.map4d_num_objects,
                pre_horizon=args.map4d_pre_horizon,
                future_horizon=args.map4d_future_horizon,
                node_dim=args.map4d_node_dim,
                relation_dim=args.map4d_relation_dim,
                temporal_dim=args.map4d_temporal_dim,
                feature_dim=args.map4d_feature_dim,
            )
        else:
            self.map4d_encoder = None

        backbones = [build_backbone(args)]
        transformer = build_transformer(args)
        encoder = build_encoder(args)

        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
            map4d_dim=self.map4d_feature_dim,
        )

    def _encode_map4d(self, obs):
        if not self.use_map4d:
            return None
        if 'map4d' not in obs:
            raise ValueError("map4d observation is required when use_map4d=True")
        map4d_seq = obs['map4d']
        map_feature = self.map4d_encoder(map4d_seq=map4d_seq)
        if map_feature.dim() == 3:
            map_feature = map_feature[:, -1]
        return map_feature

    def get_action(self, obs):
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])
        if hasattr(self, 'include_depth') and self.include_depth:
            obs['depth'] = obs['depth'].float()

        map_feature = self._encode_map4d(obs)

        obs_for_model = {k: v for k, v in obs.items() if k not in ('map4d', 'future_map4d')}
        if map_feature is not None:
            obs_for_model['map4d_feature'] = map_feature
        a_hat, (_, _) = self.model(obs_for_model)
        return a_hat


def load_norm_stats(demo_path, num_queries, num_demos, include_depth, control_mode):
    """Load normalization stats from dataset using the training Dataset class."""
    import train_rgbd_map4d
    # The Dataset class references the global `args` for control_mode
    train_rgbd_map4d.args = type('Args', (), {'control_mode': control_mode})()
    dataset = train_rgbd_map4d.SmallDemoDataset_ACTPolicy(
        demo_path, num_queries, num_traj=num_demos,
        include_depth=include_depth, use_map4d=False,
    )
    return dataset.norm_stats


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.checkpoint, "--checkpoint is required"
    assert args.demo_path, "--demo-path is required for normalization stats"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        control_mode=args.control_mode, reward_mode="sparse",
        obs_mode="rgbd" if args.include_depth else "rgb",
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    wrappers = [partial(FlattenRGBDObservationWrapper, depth=args.include_depth)]
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend,
                          env_kwargs, None, wrappers=wrappers,
                          map4d_pre_horizon=args.map4d_pre_horizon if args.use_map4d else 0,
                          map4d_task_name=args.map4d_task_name)

    print("Loading normalization stats from demo data...")
    norm_stats = load_norm_stats(args.demo_path, args.num_queries, args.num_demos, args.include_depth, args.control_mode)

    print("Creating agent and loading checkpoint...")
    agent = Agent(envs, args).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(ckpt['ema_agent'])
    print(f"Loaded checkpoint: {args.checkpoint}")

    eval_kwargs = dict(
        stats=norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
        max_timesteps=args.max_episode_steps, device=device, sim_backend=args.sim_backend,
        map4d_pre_horizon=args.map4d_pre_horizon,
    )

    print(f"Evaluating {args.num_eval_episodes} episodes with GT map4d...")
    eval_metrics = evaluate_map4d(args.num_eval_episodes, agent, envs, eval_kwargs)

    print("\n=== Evaluation Results ===")
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
        print(f"  {k}: {eval_metrics[k]:.4f}")
