"""ManiSkill rollout evaluation for the standalone Map4D DiT policy."""

from __future__ import annotations

import json
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from map4d.backbone.policy.map4d_dit_policy import Map4DDiTPolicy
from helper.keyframe_targets import matrix_to_rot6d_np, quat_to_matrix_np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DP_ROOT = PROJECT_ROOT / "baselines" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from diffusion_policy.dinov3_encoder import _IMAGENET_MEAN, _IMAGENET_STD, _load_backbone


TASK_SIZE_PARAMETERS = {
    "StackCube-v1": np.asarray(
        [
            0.04,
            0.04,
            0.04,
            0.04,
            0.04,
            0.04,
            1.2090764,
            2.4178784,
            0.91964292762787,
        ],
        dtype=np.float32,
    ),
    "PlugCharger-v1": np.asarray(
        [
            0.04,
            0.03,
            0.024,
            0.016,
            0.0015,
            0.0064,
            0.02,
            0.004,
            0.04,
            0.004,
            0.045,
            0.025,
            0.1,
            0.1,
        ],
        dtype=np.float32,
    ),
}

TASK_RELATION_PARAMETERS = {
    "StackCube-v1": np.zeros((0,), dtype=np.float32),
    "PlugCharger-v1": np.asarray([0.007], dtype=np.float32),
}

STACKCUBE_TABLE_POSE = np.asarray(
    [-0.12, 0.0, -0.9196429, 0.70710677, 0.0, 0.0, 0.70710677],
    dtype=np.float32,
)


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_tensor(value, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _append_episode_metrics(eval_metrics, episode_info):
    for key, value in episode_info.items():
        if key.startswith("_"):
            continue
        if torch.is_tensor(value):
            value = value.float().detach().cpu().numpy()
        eval_metrics[key].append(value)


def _metric_done(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.any().item())
    return bool(np.asarray(value).any())


def _metric_all(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.all().item())
    return bool(np.asarray(value).all())


class OnlineDinoV3FeatureExtractor:
    def __init__(
        self,
        *,
        model: str,
        weights_path: str,
        third_party_dir: str,
        camera_names: Sequence[str],
        image_size: int,
        device: torch.device,
    ):
        self.camera_names = tuple(camera_names)
        self.image_size = int(image_size)
        self.device = device
        self.backbone = _load_backbone(third_party_dir, weights_path, model)
        self.backbone.to(device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.embed_dim = int(self.backbone.embed_dim)
        self.mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, obs: Dict[str, object]) -> np.ndarray:
        sensor_data = obs["sensor_data"]
        features = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            for camera_name in self.camera_names:
                rgb = sensor_data[camera_name]["rgb"]
                x = _to_tensor(rgb, self.device)
                if x.ndim == 3:
                    x = x[None]
                if x.shape[-1] == 3:
                    x = x.permute(0, 3, 1, 2).contiguous()
                if x.max() > 2.0:
                    x = x / 255.0
                if x.shape[-2:] != (self.image_size, self.image_size):
                    x = F.interpolate(
                        x,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                x = (x - self.mean) / self.std
                feats = self.backbone.forward_features(x)
                features.append(feats["x_norm_patchtokens"].mean(dim=1).float())
        return torch.cat(features, dim=-1).cpu().numpy().astype(np.float32)


class Map4DDiTManiSkillEvaluator:
    def __init__(self, cfg: DictConfig, *, device: torch.device, output_dir: str):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.env = None
        self.dino = None
        self.robot_history = []
        self.map4d_history = []
        self.rgb_history = []

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.robot_history = []
        self.map4d_history = []
        self.rgb_history = []
        gc.collect()

    def _ensure_env(self):
        if self.env is not None:
            return
        import mani_skill.envs  # noqa: F401

        env_kwargs = dict(
            control_mode=self.cfg.control_mode,
            reward_mode="sparse",
            obs_mode=self.cfg.obs_mode,
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
            max_episode_steps=int(self.cfg.max_episode_steps),
        )
        env = gym.make(
            self.cfg.env_id,
            num_envs=int(self.cfg.num_eval_envs),
            sim_backend=self.cfg.sim_backend,
            reconfiguration_freq=1,
            **env_kwargs,
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        self.max_episode_steps = int(max_episode_steps or self.cfg.max_episode_steps)
        self.env = ManiSkillVectorEnv(
            env,
            ignore_terminations=True,
            record_metrics=True,
        )

        if bool(self.cfg.use_rgb):
            self.dino = OnlineDinoV3FeatureExtractor(
                model=self.cfg.dinov3_model,
                weights_path=self.cfg.dinov3_weights_path,
                third_party_dir=self.cfg.dinov3_third_party_dir,
                camera_names=tuple(self.cfg.camera_names),
                image_size=int(self.cfg.image_size),
                device=self.device,
            )

    def evaluate(
        self,
        policy: Map4DDiTPolicy,
        epoch: int,
        iteration: Optional[int] = None,
    ) -> Dict[str, float]:
        self._ensure_env()
        policy.eval()
        eval_metrics = defaultdict(list)
        progress = tqdm(total=int(self.cfg.num_eval_episodes), desc=f"Rollout epoch {epoch}")
        obs, info = self.env.reset()
        self._reset_history(obs)
        episodes = 0

        try:
            with torch.no_grad():
                while episodes < int(self.cfg.num_eval_episodes):
                    policy_obs = self._policy_obs()
                    result = policy.predict_action(policy_obs)
                    actions = self._env_actions(result["action"])
                    for step_idx in range(actions.shape[1]):
                        obs, reward, terminated, truncated, info = self.env.step(actions[:, step_idx])
                        self._append_history(obs)
                        if _metric_done(truncated):
                            break

                    if _metric_done(truncated):
                        assert _metric_all(truncated), (
                            "all episodes should truncate at the same time for fair evaluation with other algorithms"
                        )
                        self._collect_metrics(eval_metrics, info)
                        episodes += self.env.num_envs
                        progress.update(self.env.num_envs)
                        self._reset_history(obs)
        finally:
            progress.close()
            if bool(self.cfg.get("close_after_eval", True)):
                self.close()

        mean_metrics = {}
        for key, values in eval_metrics.items():
            mean_metrics[key] = float(np.mean(np.stack(values)))
        self._write_metrics(epoch, iteration, mean_metrics)
        return mean_metrics

    def _reset_history(self, obs):
        robot_state = self._robot_state(obs)
        map4d = self._map4d()
        rgb_feature = self._rgb_feature(obs)
        horizon = int(self.cfg.obs_horizon)
        self.robot_history = [robot_state.copy() for _ in range(horizon)]
        self.map4d_history = [map4d.copy() for _ in range(horizon)]
        self.rgb_history = [rgb_feature.copy() for _ in range(horizon)]

    def _append_history(self, obs):
        self.robot_history.append(self._robot_state(obs))
        self.map4d_history.append(self._map4d())
        self.rgb_history.append(self._rgb_feature(obs))
        horizon = int(self.cfg.obs_horizon)
        self.robot_history = self.robot_history[-horizon:]
        self.map4d_history = self.map4d_history[-horizon:]
        self.rgb_history = self.rgb_history[-horizon:]

    def _policy_obs(self) -> Dict[str, torch.Tensor]:
        batch_size = self.env.num_envs
        task_name = self.cfg.env_id
        obs = {
            "robot_state": torch.as_tensor(np.stack(self.robot_history, axis=1), device=self.device),
            "map4d": torch.as_tensor(np.stack(self.map4d_history, axis=1), device=self.device),
            "size_parameters": torch.as_tensor(
                np.broadcast_to(
                    TASK_SIZE_PARAMETERS[task_name],
                    (batch_size, TASK_SIZE_PARAMETERS[task_name].shape[0]),
                ).copy(),
                device=self.device,
            ),
            "relation_parameters": torch.as_tensor(
                np.broadcast_to(
                    TASK_RELATION_PARAMETERS[task_name],
                    (batch_size, TASK_RELATION_PARAMETERS[task_name].shape[0]),
                ).copy(),
                device=self.device,
            ),
        }
        if bool(self.cfg.use_rgb):
            obs["rgb_feature"] = torch.as_tensor(np.stack(self.rgb_history, axis=1), device=self.device)
        return obs

    def _robot_state(self, obs) -> np.ndarray:
        agent = obs["agent"]
        extra = obs.get("extra", {})
        fields = []
        for value in (agent.get("qpos"), agent.get("qvel"), extra.get("tcp_pose"), extra.get("tcp_pos")):
            if value is None:
                continue
            arr = _to_numpy(value).reshape(self.env.num_envs, -1).astype(np.float32)
            fields.append(arr)
        if not fields:
            state = np.zeros((self.env.num_envs, int(self.cfg.robot_state_dim)), dtype=np.float32)
        else:
            state = np.concatenate(fields, axis=-1).astype(np.float32)
        target_dim = int(self.cfg.robot_state_dim)
        if state.shape[-1] < target_dim:
            pad = np.zeros((state.shape[0], target_dim - state.shape[-1]), dtype=np.float32)
            state = np.concatenate([state, pad], axis=-1)
        elif state.shape[-1] > target_dim:
            state = state[:, :target_dim]
        return state.astype(np.float32)

    def _rgb_feature(self, obs) -> np.ndarray:
        if not bool(self.cfg.use_rgb):
            return np.zeros((self.env.num_envs, 0), dtype=np.float32)
        feature = self.dino(obs)
        expected = int(self.cfg.rgb_feature_dim)
        if feature.shape[-1] < expected:
            pad = np.zeros((feature.shape[0], expected - feature.shape[-1]), dtype=np.float32)
            feature = np.concatenate([feature, pad], axis=-1)
        elif feature.shape[-1] > expected:
            feature = feature[:, :expected]
        return feature.astype(np.float32)

    def _map4d(self) -> np.ndarray:
        base_env = self.env.base_env
        if self.cfg.env_id == "StackCube-v1":
            poses = [
                self._actor_pose(base_env.cubeA),
                self._actor_pose(base_env.cubeB),
                self._table_pose(base_env),
            ]
        else:
            poses = [
                self._actor_pose(base_env.charger),
                self._actor_pose(base_env.receptacle),
            ]
        pos = np.stack([pose[:, 0:3] for pose in poses], axis=1)
        rot = np.stack(
            [matrix_to_rot6d_np(quat_to_matrix_np(pose[:, 3:7])) for pose in poses],
            axis=1,
        )
        return np.concatenate([pos, rot], axis=-1).astype(np.float32)

    def _actor_pose(self, actor) -> np.ndarray:
        raw_pose = _to_numpy(actor.pose.raw_pose).astype(np.float32)
        if raw_pose.ndim == 1:
            raw_pose = raw_pose[None]
        return raw_pose[:, :7]

    def _table_pose(self, env) -> np.ndarray:
        table_scene = getattr(env, "table_scene", None)
        table_actor = getattr(table_scene, "table", None)
        if table_actor is not None:
            return self._actor_pose(table_actor)
        return np.broadcast_to(STACKCUBE_TABLE_POSE, (self.env.num_envs, 7)).astype(np.float32)

    def _env_actions(self, action: torch.Tensor):
        action = action.detach()
        if self.cfg.action_mode == "pos_gripper":
            action = torch.cat([action[..., 0:3], action[..., 7:8]], dim=-1)
        elif self.cfg.action_mode == "pose":
            action = action[..., :7]
        else:
            raise ValueError(f"Unsupported rollout action_mode={self.cfg.action_mode!r}")
        action = self._clip_action(action)
        if self.cfg.sim_backend == "physx_cpu":
            return action.cpu().numpy()
        return action

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        space = getattr(self.env, "single_action_space", self.env.action_space)
        if not isinstance(space, gym.spaces.Box):
            return action
        low = torch.as_tensor(space.low, device=action.device, dtype=action.dtype)
        high = torch.as_tensor(space.high, device=action.device, dtype=action.dtype)
        while low.ndim < action.ndim:
            low = low.unsqueeze(0)
            high = high.unsqueeze(0)
        return torch.max(torch.min(action, high), low)

    def _collect_metrics(self, eval_metrics, info):
        if "final_info" in info and isinstance(info["final_info"], dict):
            _append_episode_metrics(eval_metrics, info["final_info"]["episode"])
        elif "final_info" in info:
            for final_info in info["final_info"]:
                _append_episode_metrics(eval_metrics, final_info["episode"])
        elif "episode" in info:
            _append_episode_metrics(eval_metrics, info["episode"])
        else:
            raise KeyError(f"Expected episode metrics in info, got keys: {list(info.keys())}")

    def _metrics_record(self, epoch: int, iteration: Optional[int], metrics: Dict[str, float]):
        return {
            "epoch": int(epoch),
            "iteration": int(epoch if iteration is None else iteration),
            "num_eval_episodes": int(self.cfg.num_eval_episodes),
            "run_name": str(self.cfg.run_name),
            **metrics,
        }

    def _write_metrics(self, epoch: int, iteration: Optional[int], metrics: Dict[str, float]):
        record = self._metrics_record(epoch, iteration, metrics)
        run_dir_record = {
            "time": time.time(),
            "num_eval_envs": int(self.cfg.num_eval_envs),
            **record,
        }

        path = Path(self.output_dir) / "rollout_metrics.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run_dir_record, sort_keys=True) + "\n")

        metrics_dir = Path(str(self.cfg.metrics_dir))
        if not metrics_dir.is_absolute():
            metrics_dir = PROJECT_ROOT / metrics_dir
        metrics_dir.mkdir(parents=True, exist_ok=True)
        eval_metrics_path = metrics_dir / f"{self.cfg.run_name}.jsonl"
        with eval_metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"Eval metrics written to {eval_metrics_path}")


def build_rollout_evaluator(cfg: DictConfig, *, device: torch.device, output_dir: str) -> Optional[Map4DDiTManiSkillEvaluator]:
    rollout_cfg = cfg.get("rollout")
    if rollout_cfg is None or not bool(rollout_cfg.enabled):
        return None
    resolved = OmegaConf.to_container(rollout_cfg, resolve=True)
    resolved["env_id"] = cfg.task.name
    resolved["obs_horizon"] = int(cfg.n_obs_steps)
    resolved["robot_state_dim"] = int(cfg.robot_state_dim)
    resolved["rgb_feature_dim"] = int(cfg.policy.model_cfg.rgb_feature_dim)
    resolved["use_rgb"] = bool(cfg.policy.model_cfg.use_rgb)
    return Map4DDiTManiSkillEvaluator(OmegaConf.create(resolved), device=device, output_dir=output_dir)
