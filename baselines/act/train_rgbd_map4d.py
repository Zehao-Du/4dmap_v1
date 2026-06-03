ALGO_NAME = 'BC_ACT_rgbd_map4d'

import argparse
import os
import sys
import random
from distutils.util import strtobool
from functools import partial
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from act.evaluate_map4d import evaluate_map4d
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from act.utils import IterationBasedBatchSampler, worker_init_fn
from act.make_env import make_eval_envs
try:
    from diffusers.training_utils import EMAModel
except (ImportError, RuntimeError):
    class EMAModel:
        def __init__(self, parameters, power=0.75):
            self.decay = float(power)
            self.shadow_params = [p.detach().clone() for p in parameters]

        def step(self, parameters):
            with torch.no_grad():
                for shadow, param in zip(self.shadow_params, parameters):
                    shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

        def copy_to(self, parameters):
            with torch.no_grad():
                for shadow, param in zip(self.shadow_params, parameters):
                    param.copy_(shadow)
from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import tyro

# Add repo root to path for map4d imports
_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_BASELINE_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from map4d.map4d_encoder import Map4d_Encoder
from map4d.encoder import PhysicsLosses

# --------------------------------------------------------------------------- #
# Map4D GT helpers (from DP train_rgbd.py)
# --------------------------------------------------------------------------- #
STACKCUBE_GT_SIZES_MANISKILL_XYZ = (
    0.04, 0.04, 0.04,
    0.04, 0.04, 0.04,
    1.2090764, 2.4178784, 0.91964292762787,
)

PLUGCHARGER_GT_SIZES_MANISKILL_XYZ = (
    0.04, 0.03, 0.024,       # charger body (2 * _base_size)
    0.02, 0.1, 0.1,          # receptacle (2 * _receptacle_size)
)

TASK_GT_SIZES = {
    "StackCube-v1": STACKCUBE_GT_SIZES_MANISKILL_XYZ,
    "PlugCharger-v1": PLUGCHARGER_GT_SIZES_MANISKILL_XYZ,
}


def _quat_wxyz_to_rotation_6d_np(quat_wxyz):
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape [T, 4], got {quat.shape}")
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1e-8)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    matrix = np.empty((quat.shape[0], 3, 3), dtype=np.float32)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - z * w)
    matrix[:, 0, 2] = 2.0 * (x * z + y * w)
    matrix[:, 1, 0] = 2.0 * (x * y + z * w)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - x * w)
    matrix[:, 2, 0] = 2.0 * (x * z - y * w)
    matrix[:, 2, 1] = 2.0 * (y * z + x * w)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return np.concatenate([matrix[:, :, 0], matrix[:, :, 1]], axis=1).astype(np.float32)


def _actor_states_to_map4d_tensor(actor_states, *, sizes=None):
    states = [np.asarray(state, dtype=np.float32) for state in actor_states]
    num_objects = len(states)
    frame_count = states[0].shape[0]
    if any(state.shape[0] != frame_count for state in states):
        raise ValueError("All actor state arrays must have the same frame count.")
    if sizes is not None:
        sizes_np = np.asarray(sizes, dtype=np.float32).reshape(num_objects, 3)
    else:
        sizes_np = np.zeros((num_objects, 3), dtype=np.float32)
    sizes_seq = np.broadcast_to(sizes_np, (frame_count, num_objects, 3))
    positions = np.stack([state[:, 0:3] for state in states], axis=1)
    rotations = np.stack(
        [_quat_wxyz_to_rotation_6d_np(state[:, 3:7]) for state in states], axis=1,
    )
    return np.concatenate([sizes_seq, positions, rotations], axis=-1).astype(np.float32)


TASK_ACTOR_NAMES = {
    "StackCube-v1": ("cubeA", "cubeB", "table-workspace"),
    "PlugCharger-v1": ("charger", "receptacle"),
}


def _load_maniskill_gt_map4d_tensors(
    data_path, *, num_traj=None, task_name="StackCube-v1",
    actor_names=None, device=None,
):
    if actor_names is None:
        actor_names = TASK_ACTOR_NAMES.get(task_name)
        if actor_names is None:
            raise ValueError(f"No default actor_names for task {task_name}. Provide actor_names explicitly.")
    sizes = TASK_GT_SIZES.get(task_name)
    import h5py
    with h5py.File(data_path, "r") as f:
        traj_keys = [key for key in f.keys() if key.startswith("traj_")]
        traj_keys = sorted(traj_keys, key=lambda key: int(key.split("_")[-1]))
        if num_traj is not None:
            traj_keys = traj_keys[:num_traj]
        map4d_tensors = []
        for traj_key in traj_keys:
            actors = f[traj_key]["env_states"]["actors"]
            missing = [name for name in actor_names if name not in actors]
            if missing:
                raise KeyError(f"{traj_key} missing ManiSkill GT actors: {missing}")
            actor_states = [actors[name][()] for name in actor_names]
            map4d_np = _actor_states_to_map4d_tensor(actor_states, sizes=sizes)
            map4d_tensors.append(torch.as_tensor(map4d_np, device=device))
    return map4d_tensors


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    env_id: str = "StackCube-v1"
    demo_path: str = 'pickcube.trajectory.rgbd.pd_joint_delta_pos.cpu.h5'
    num_demos: Optional[int] = None
    total_iters: int = 1_000_000
    batch_size: int = 256

    # ACT specific arguments
    lr: float = 1e-4
    kl_weight: float = 10
    temporal_agg: bool = True

    # Backbone
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = True

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    # Environment
    max_episode_steps: Optional[int] = None
    log_freq: int = 1000
    eval_freq: int = 5000
    save_freq: Optional[int] = None
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "cpu"
    num_dataload_workers: int = 0
    control_mode: str = 'pd_joint_delta_pos'
    demo_type: Optional[str] = None

    # 4D Map arguments
    use_map4d: bool = True
    map4d_raw_concat: bool = False
    """bypass encoder: flatten raw map4d and concat to state directly"""
    map4d_as_tokens: bool = False
    """inject map4d frames as individual tokens into transformer memory"""
    map4d_mlp_token: bool = False
    """flatten all map4d frames and project via MLP into one context token"""
    map4d_aux_loss: bool = False
    """add auxiliary future prediction loss on policy internal feature"""
    map4d_aux_weight: float = 1.0
    """weight for auxiliary future prediction loss"""
    map4d_source: str = "maniskill_gt"
    map4d_task_name: str = "StackCube-v1"
    map4d_pre_horizon: int = 6
    map4d_future_horizon: int = 3
    map4d_num_objects: int = 3
    map4d_feature_dim: int = 128
    map4d_node_dim: int = 128
    map4d_relation_dim: int = 64
    map4d_temporal_dim: int = 128
    map4d_encoder_type: str = "gru"
    """encoder backbone: 'gru' or 'transformer'"""
    map4d_pose_weight: float = 1.0
    map4d_penetration_weight: float = 0.1
    map4d_kinematic_weight: float = 0.1
    map4d_vel_limit: float = 0.5
    map4d_acc_limit: float = 1.0
    map4d_rot_vel_limit: float = 1.0
    map4d_rot_acc_limit: float = 2.0
    map4d_penetration_margin: float = 0.0


# --------------------------------------------------------------------------- #
# Observation wrapper (same as train_rgbd.py)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SmallDemoDataset_ACTPolicy(Dataset):
    def __init__(self, data_path, num_queries, num_traj, include_depth=True,
                 use_map4d=False, map4d_source="maniskill_gt", map4d_task_name="StackCube-v1",
                 map4d_pre_horizon=6, map4d_future_horizon=3):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from act.utils import load_demo_dataset
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        print('Raw trajectory loaded, start to pre-process the observations...')

        self.include_depth = include_depth
        self.use_map4d = use_map4d
        self.map4d_pre_horizon = map4d_pre_horizon
        self.map4d_future_horizon = map4d_future_horizon
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])

        # Load map4d GT tensors
        if self.use_map4d:
            print("Loading map4d GT tensors...")
            self.map4d_tensors = _load_maniskill_gt_map4d_tensors(
                data_path, num_traj=num_traj, task_name=map4d_task_name
            )
            print(f"Loaded {len(self.map4d_tensors)} map4d trajectories, "
                  f"shape[0]: {self.map4d_tensors[0].shape}")

        # Pre-process the observations
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories['observations']:
            obs_traj_dict = self.process_obs(obs_traj_dict)
            obs_traj_dict_list.append(obs_traj_dict)
        trajectories['observations'] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict.keys())

        # Pre-process the actions
        for i in range(len(trajectories['actions'])):
            trajectories['actions'][i] = torch.Tensor(trajectories['actions'][i])
        print('Obs/action pre-processing is done.')

        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,))

        self.slices = []
        self.num_traj = len(trajectories['actions'])
        for traj_idx in range(self.num_traj):
            episode_len = trajectories['actions'][traj_idx].shape[0]
            self.slices += [(traj_idx, ts) for ts in range(episode_len)]

        print(f"Length of Dataset: {len(self.slices)}")

        self.num_queries = num_queries
        self.trajectories = trajectories
        self.delta_control = 'delta' in args.control_mode
        self.norm_stats = self.get_norm_stats() if not self.delta_control else None

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]

        state = self.trajectories['observations'][traj_idx]['state'][ts]
        act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
        action_len = act_seq.shape[0]

        if action_len < self.num_queries:
            if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries-action_len, 1)], dim=0)
            elif not self.delta_control:
                target = act_seq[-1]
                act_seq = torch.cat([act_seq, target.repeat(self.num_queries-action_len, 1)], dim=0)

        if not self.delta_control:
            state = (state - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
            act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        if self.include_depth:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            depth = self.trajectories['observations'][traj_idx]['depth'][ts]
            obs = dict(state=state, rgb=rgb, depth=depth)
        else:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            obs = dict(state=state, rgb=rgb)

        # Map4d slicing
        if self.use_map4d:
            map4d_traj = self.map4d_tensors[traj_idx]  # (T, N, feat)
            T_len = map4d_traj.shape[0]
            # Slice pre_horizon frames ending at ts (inclusive)
            start = max(0, ts + 1 - self.map4d_pre_horizon)
            end = ts + 1
            map4d_seq = map4d_traj[start:end]
            # Pad if not enough frames at the beginning
            if map4d_seq.shape[0] < self.map4d_pre_horizon:
                pad_len = self.map4d_pre_horizon - map4d_seq.shape[0]
                map4d_seq = torch.cat([map4d_seq[:1].expand(pad_len, -1, -1), map4d_seq], dim=0)
            obs['map4d'] = map4d_seq  # (pre_horizon, N, feat)

            # Future map4d for auxiliary loss
            fut_start = ts + 1
            fut_end = min(T_len, fut_start + self.map4d_future_horizon)
            future_map4d = map4d_traj[fut_start:fut_end]
            if future_map4d.shape[0] < self.map4d_future_horizon:
                pad_len = self.map4d_future_horizon - future_map4d.shape[0]
                if future_map4d.shape[0] > 0:
                    future_map4d = torch.cat([future_map4d, future_map4d[-1:].expand(pad_len, -1, -1)], dim=0)
                else:
                    future_map4d = map4d_traj[ts:ts+1].expand(self.map4d_future_horizon, -1, -1)
            obs['future_map4d'] = future_map4d  # (future_horizon, N, feat)

        return {
            'observations': obs,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)

    def process_obs(self, obs_dict):
        sensor_data = obs_dict.pop("sensor_data")
        del obs_dict["sensor_param"]
        images_rgb = []
        images_depth = []
        for cam_data in sensor_data.values():
            rgb = torch.from_numpy(cam_data["rgb"])
            resized_rgb = self.transforms(rgb.permute(0, 3, 1, 2))
            images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = torch.Tensor(cam_data["depth"].astype(np.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(depth.permute(0, 3, 1, 2))
                images_depth.append(resized_depth)
        rgb = torch.stack(images_rgb, dim=1)
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1)
        obs_dict['extra'] = {k: v[:, None] if len(v.shape) == 1 else v for k, v in obs_dict['extra'].items()}
        obs_dict = common.flatten_state_dict(obs_dict, use_torch=True)
        processed_obs = dict(state=obs_dict, rgb=rgb, depth=depth) if self.include_depth else dict(state=obs_dict, rgb=rgb)
        return processed_obs

    def get_norm_stats(self):
        all_state_data = []
        all_action_data = []
        for traj_idx, ts in self.slices:
            state = self.trajectories['observations'][traj_idx]['state'][ts]
            act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
            action_len = act_seq.shape[0]
            if action_len < self.num_queries:
                target_pos = act_seq[-1]
                act_seq = torch.cat([act_seq, target_pos.repeat(self.num_queries-action_len, 1)], dim=0)
            all_state_data.append(state)
            all_action_data.append(act_seq)
        all_state_data = torch.stack(all_state_data)
        all_action_data = torch.concatenate(all_action_data)
        state_mean = all_state_data.mean(dim=0, keepdim=True)
        state_std = all_state_data.std(dim=0, keepdim=True)
        state_std = torch.clip(state_std, 1e-2, np.inf)
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf)
        stats = {"action_mean": action_mean, "action_std": action_std,
                 "state_mean": state_mean, "state_std": state_std,
                 "example_state": state}
        return stats


# --------------------------------------------------------------------------- #
# Agent with Map4D
# --------------------------------------------------------------------------- #
class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        assert len(env.single_observation_space['state'].shape) == 1
        assert len(env.single_observation_space['rgb'].shape) == 4
        assert len(env.single_action_space.shape) == 1

        self.state_dim = env.single_observation_space['state'].shape[0]
        self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.use_map4d = args.use_map4d
        self.map4d_raw_concat = args.map4d_raw_concat if args.use_map4d else False
        self.map4d_as_tokens = args.map4d_as_tokens if args.use_map4d else False
        self.map4d_mlp_token = args.map4d_mlp_token if args.use_map4d else False
        self.map4d_feature_dim = args.map4d_feature_dim if args.use_map4d else 0
        self.include_depth = args.include_depth
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # Map4D encoder (skip if raw_concat, as_tokens, or mlp_token mode)
        if self.use_map4d and not self.map4d_raw_concat and not self.map4d_as_tokens and not self.map4d_mlp_token:
            encoder_kwargs = dict(
                num_objects=args.map4d_num_objects,
                node_dim=args.map4d_node_dim,
                relation_dim=args.map4d_relation_dim,
                temporal_dim=args.map4d_temporal_dim,
                feature_dim=args.map4d_feature_dim,
            )
            if args.map4d_encoder_type == "transformer":
                from map4d.encoder.geometric_transformer_encoder import GeometricTransformerEncoder
                geo_encoder = GeometricTransformerEncoder(**encoder_kwargs)
            else:
                from map4d.encoder.geometric_encoder import GeometricEncoder
                geo_encoder = GeometricEncoder(**encoder_kwargs)
            self.map4d_encoder = Map4d_Encoder(
                encoder=geo_encoder,
                num_objects=args.map4d_num_objects,
                pre_horizon=args.map4d_pre_horizon,
                future_horizon=args.map4d_future_horizon,
                feature_dim=args.map4d_feature_dim,
            )
            self.map4d_losses = PhysicsLosses(
                num_objects=args.map4d_num_objects,
                pose_weight=args.map4d_pose_weight,
                penetration_weight=args.map4d_penetration_weight,
                kinematic_weight=args.map4d_kinematic_weight,
                vel_limit=args.map4d_vel_limit,
                acc_limit=args.map4d_acc_limit,
                rot_vel_limit=args.map4d_rot_vel_limit,
                rot_acc_limit=args.map4d_rot_acc_limit,
                penetration_margin=args.map4d_penetration_margin,
            )
        else:
            self.map4d_encoder = None
            self.map4d_losses = None

        # CNN backbone
        backbones = []
        backbone = build_backbone(args)
        backbones.append(backbone)

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_encoder(args)

        # raw_concat: expand state_dim; encoder mode: use map4d_dim as context token
        if self.map4d_raw_concat:
            raw_map4d_dim = args.map4d_num_objects * 12
            model_state_dim = self.state_dim + raw_map4d_dim
            model_map4d_dim = 0
        elif self.map4d_as_tokens:
            model_state_dim = self.state_dim
            model_map4d_dim = 0
        elif self.map4d_mlp_token or self.use_map4d:
            model_state_dim = self.state_dim
            model_map4d_dim = self.map4d_feature_dim
        else:
            model_state_dim = self.state_dim
            model_map4d_dim = 0

        # mlp_token: flatten all frames → MLP → map4d_feature_dim
        if self.map4d_mlp_token:
            mlp_input_dim = args.map4d_pre_horizon * args.map4d_num_objects * 12
            self.map4d_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, self.map4d_feature_dim),
            )

        # map4d as tokens params
        if self.map4d_as_tokens:
            map4d_token_dim = 12  # per-object per-frame: size(3)+pos(3)+rot(6)
            map4d_max_tokens = args.map4d_pre_horizon * args.map4d_num_objects
        else:
            map4d_token_dim = 0
            map4d_max_tokens = 0

        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=model_state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
            map4d_dim=model_map4d_dim,
            map4d_token_dim=map4d_token_dim,
            map4d_max_tokens=map4d_max_tokens,
        )

        # Auxiliary future prediction head (raw_concat + aux_loss mode)
        self.map4d_aux_loss = args.map4d_aux_loss if args.use_map4d else False
        self.map4d_aux_weight = args.map4d_aux_weight
        if self.map4d_aux_loss:
            self.future_horizon = args.map4d_future_horizon
            self.map4d_num_objects = args.map4d_num_objects
            # Learned query + cross-attention readout from full encoder memory
            self.aux_query = nn.Parameter(torch.randn(1, 1, args.hidden_dim))
            self.aux_cross_attn = nn.MultiheadAttention(
                embed_dim=args.hidden_dim, num_heads=4, batch_first=True,
            )
            self.aux_norm = nn.LayerNorm(args.hidden_dim)
            self.future_pred_head = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, self.future_horizon * self.map4d_num_objects * 9),
            )

    def _encode_map4d(self, obs, training=False):
        if not self.use_map4d:
            return None, None
        if 'map4d' not in obs:
            raise ValueError("map4d observation is required when use_map4d=True")
        if self.map4d_raw_concat:
            return None, None
        map4d_seq = obs['map4d']
        if training:
            future_map4d_seq = obs.get('future_map4d')
            map_feature, map_aux = self.map4d_encoder.forward_with_aux(
                map4d_seq=map4d_seq, future_map4d_seq=future_map4d_seq
            )
        else:
            map_feature = self.map4d_encoder(map4d_seq=map4d_seq)
            map_aux = None
        # map_feature: [B, T, feature_dim] — take last timestep
        if map_feature.dim() == 3:
            map_feature = map_feature[:, -1]
        return map_feature, map_aux

    def _get_raw_map4d_state(self, obs):
        """Flatten last frame of map4d and return for state concat."""
        map4d_seq = obs['map4d']  # (B, pre_horizon, N, 12)
        last_frame = map4d_seq[:, -1]  # (B, N, 12)
        return last_frame.reshape(last_frame.shape[0], -1)  # (B, N*12)

    def compute_loss(self, obs, action_seq):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])
        if self.include_depth:
            obs['depth'] = obs['depth'].float()

        if self.map4d_as_tokens:
            obs_for_model = {k: v for k, v in obs.items() if k not in ('map4d', 'future_map4d')}
            # Flatten map4d (B, T, N, 12) → (B, T*N, 12) as tokens
            map4d_seq = obs['map4d']
            B, T, N, D = map4d_seq.shape
            obs_for_model['map4d_tokens'] = map4d_seq.reshape(B, T * N, D)
            a_hat, (mu, logvar), encoder_memory = self.model(obs_for_model, action_seq)
            map_aux = None
        elif self.map4d_mlp_token:
            obs_for_model = {k: v for k, v in obs.items() if k not in ('map4d', 'future_map4d')}
            map4d_seq = obs['map4d']  # (B, T, N, 12)
            map4d_flat = map4d_seq.flatten(start_dim=1)  # (B, T*N*12)
            obs_for_model['map4d_feature'] = self.map4d_mlp(map4d_flat)  # (B, feature_dim)
            a_hat, (mu, logvar), encoder_memory = self.model(obs_for_model, action_seq)
            map_aux = None
        elif self.map4d_raw_concat:
            raw_map4d = self._get_raw_map4d_state(obs)
            obs_for_model = {k: v for k, v in obs.items() if k not in ('map4d', 'future_map4d')}
            obs_for_model['state'] = torch.cat([obs['state'], raw_map4d], dim=-1)
            a_hat, (mu, logvar), encoder_memory = self.model(obs_for_model, action_seq)
            map_aux = None
        else:
            map_feature, map_aux = self._encode_map4d(obs, training=True)
            obs_for_model = {k: v for k, v in obs.items() if k not in ('map4d', 'future_map4d')}
            if map_feature is not None:
                obs_for_model['map4d_feature'] = map_feature
            a_hat, (mu, logvar), encoder_memory = self.model(obs_for_model, action_seq)

        # compute l1 loss and kl loss
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()

        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

        # Add map4d encoder auxiliary loss (encoder mode only)
        if self.use_map4d and not self.map4d_raw_concat and self.map4d_losses is not None and map_aux is not None:
            map_losses = self.map4d_losses(map_aux)
            loss_dict['map4d_loss'] = map_losses["total"]
            loss_dict['loss'] = loss_dict['loss'] + map_losses["total"]

        # Add future prediction auxiliary loss (raw_concat + aux_loss mode)
        if self.map4d_aux_loss and 'future_map4d' in obs:
            future_map4d = obs['future_map4d']  # (B, H, N, 12)
            current_map4d = obs['map4d'][:, -1]  # (B, N, 12)
            # Position: predict delta (subtraction is valid for positions)
            gt_future_pos = future_map4d[..., 3:6]   # (B, H, N, 3)
            current_pos = current_map4d[:, :, 3:6].unsqueeze(1)  # (B, 1, N, 3)
            gt_delta_pos = gt_future_pos - current_pos  # (B, H, N, 3)
            # Rotation: predict absolute future rotation 6D (avoid delta subtraction bug)
            gt_future_rot = future_map4d[..., 6:12]  # (B, H, N, 6)
            gt_target = torch.cat([gt_delta_pos, gt_future_rot], dim=-1).flatten(start_dim=1)  # (B, H*N*9)
            # Cross-attention readout from full encoder memory
            B = encoder_memory.shape[1]
            memory_kv = encoder_memory.permute(1, 0, 2)  # (B, seq_len, hidden_dim)
            query = self.aux_query.expand(B, -1, -1)  # (B, 1, hidden_dim)
            readout, _ = self.aux_cross_attn(query, memory_kv, memory_kv)  # (B, 1, hidden_dim)
            readout = self.aux_norm(readout.squeeze(1))  # (B, hidden_dim)
            pred_future = self.future_pred_head(readout)
            aux_loss = F.l1_loss(pred_future, gt_target)
            loss_dict['aux_future_loss'] = aux_loss
            loss_dict['loss'] = loss_dict['loss'] + aux_loss * self.map4d_aux_weight

        return loss_dict

    def get_action(self, obs):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])
        if self.include_depth:
            obs['depth'] = obs['depth'].float()

        if self.map4d_as_tokens:
            obs_for_model = {k: v for k, v in obs.items() if k not in ('map4d', 'future_map4d')}
            map4d_seq = obs['map4d']
            B, T, N, D = map4d_seq.shape
            obs_for_model['map4d_tokens'] = map4d_seq.reshape(B, T * N, D)
            a_hat, (_, _), _ = self.model(obs_for_model)
        elif self.map4d_mlp_token:
            obs_for_model = {k: v for k, v in obs.items() if k not in ('map4d', 'future_map4d')}
            map4d_seq = obs['map4d']
            map4d_flat = map4d_seq.flatten(start_dim=1)
            obs_for_model['map4d_feature'] = self.map4d_mlp(map4d_flat)
            a_hat, (_, _), _ = self.model(obs_for_model)
        elif self.map4d_raw_concat:
            raw_map4d = self._get_raw_map4d_state(obs)
            obs_for_model = {k: v for k, v in obs.items() if k not in ('map4d', 'future_map4d')}
            obs_for_model['state'] = torch.cat([obs['state'], raw_map4d], dim=-1)
            a_hat, (_, _), _ = self.model(obs_for_model)
        else:
            map_feature, _ = self._encode_map4d(obs, training=False)
            obs_for_model = {k: v for k, v in obs.items() if k not in ('map4d', 'future_map4d')}
            if map_feature is not None:
                obs_for_model['map4d_feature'] = map_feature
            a_hat, (_, _), _ = self.model(obs_for_model)
        return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'norm_stats': dataset.norm_stats,
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse",
                      obs_mode="rgbd" if args.include_depth else "rgb", render_mode="rgb_array")
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    wrappers = [partial(FlattenRGBDObservationWrapper, depth=args.include_depth)]
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, None,
                          video_dir=f'runs/{run_name}/videos' if args.capture_video else None, wrappers=wrappers,
                          map4d_pre_horizon=args.map4d_pre_horizon if args.use_map4d else 0,
                          map4d_task_name=args.map4d_task_name)

    # dataloader setup
    dataset = SmallDemoDataset_ACTPolicy(
        args.demo_path, args.num_queries, num_traj=args.num_demos,
        include_depth=args.include_depth,
        use_map4d=args.use_map4d, map4d_source=args.map4d_source,
        map4d_task_name=args.map4d_task_name,
        map4d_pre_horizon=args.map4d_pre_horizon,
        map4d_future_horizon=args.map4d_future_horizon,
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset, batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    if args.num_demos is None:
        args.num_demos = dataset.num_traj

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs,
                                      env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=config, name=run_name,
                   save_code=True, group="ACT_map4d", tags=["act", "map4d"])
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # agent setup
    agent = Agent(envs, args).to(device)

    # optimizer setup
    param_dicts = [
        {"params": [p for n, p in agent.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)

    lr_drop = int((2/3)*args.total_iters)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)

    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    eval_kwargs = dict(
        stats=dataset.norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
        max_timesteps=args.max_episode_steps, device=device, sim_backend=args.sim_backend,
        map4d_pre_horizon=args.map4d_pre_horizon
    )

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    for cur_iter, data_batch in enumerate(train_dataloader):
        last_tick = time.time()
        obs_batch_dict = data_batch['observations']
        obs_batch_dict = {k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()}
        act_batch = data_batch['actions'].cuda(non_blocking=True)

        loss_dict = agent.compute_loss(obs=obs_batch_dict, action_seq=act_batch)
        total_loss = loss_dict['loss']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        ema.step(agent.parameters())
        timings["update"] += time.time() - last_tick

        # Evaluation
        if cur_iter % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate_map4d(args.num_eval_episodes, ema_agent, envs, eval_kwargs)
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], cur_iter)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(f'New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.')

        if cur_iter % args.log_freq == 0:
            loss_str = f"Iteration {cur_iter}, loss: {total_loss.item():.6f}"
            if 'map4d_loss' in loss_dict:
                loss_str += f", map4d_loss: {loss_dict['map4d_loss'].item():.6f}"
            if 'aux_future_loss' in loss_dict:
                loss_str += f", aux_future_loss: {loss_dict['aux_future_loss'].item():.6f}"
            print(loss_str)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)
            if 'map4d_loss' in loss_dict:
                writer.add_scalar("losses/map4d_loss", loss_dict['map4d_loss'].item(), cur_iter)
            if 'aux_future_loss' in loss_dict:
                writer.add_scalar("losses/aux_future_loss", loss_dict['aux_future_loss'].item(), cur_iter)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, cur_iter)

        if args.save_freq is not None and cur_iter % args.save_freq == 0:
            save_ckpt(run_name, str(cur_iter))

    envs.close()
    writer.close()
