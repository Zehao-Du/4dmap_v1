"""ManiSkill environment evaluation for the Map4D DiT policy."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 - register envs


TASK_CONFIG = {
    "StackCube-v1": {
        "control_mode": "pd_ee_delta_pos",
        "max_episode_steps": 1000,
        "action_dim": 4,
        "num_objects": 3,
    },
    "PlugCharger-v1": {
        "control_mode": "pd_ee_delta_pose",
        "max_episode_steps": 400,
        "action_dim": 7,
        "num_objects": 2,
    },
}

TASK_SIZE_PARAMETERS = {
    "StackCube-v1": np.array([
        0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
        1.2090764, 2.4178784, 0.91964292762787,
    ], dtype=np.float32),
    "PlugCharger-v1": np.array([
        0.04, 0.03, 0.024, 0.016, 0.0015, 0.0064,
        0.02, 0.004, 0.04, 0.004, 0.045, 0.025, 0.1, 0.1,
    ], dtype=np.float32),
}

TASK_RELATION_PARAMETERS = {
    "StackCube-v1": np.array([], dtype=np.float32),
    "PlugCharger-v1": np.array([0.007], dtype=np.float32),
}


def _quat_wxyz_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternion to axis-angle. Shape (..., 4) -> (..., 3)."""
    w = np.clip(quat[..., 0:1], -1.0, 1.0)
    xyz = quat[..., 1:4]
    angle = 2.0 * np.arccos(np.abs(w))
    sin_half = np.sin(angle / 2.0)
    small = sin_half < 1e-7
    axis = np.where(small, np.zeros_like(xyz), xyz / np.maximum(sin_half, 1e-8))
    sign = np.where(w < 0, -1.0, 1.0)
    return (axis * angle * sign).astype(np.float32)


def _quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert a single WXYZ quaternion to 6D rotation."""
    quat = quat / max(np.linalg.norm(quat), 1e-8)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    r20 = 2.0 * (x * z - y * w)
    r01 = 2.0 * (x * y - z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r21 = 2.0 * (y * z + x * w)
    return np.array([r00, r10, r20, r01, r11, r21], dtype=np.float32)


def _get_pose(actor) -> np.ndarray:
    raw_pose = actor.pose.raw_pose
    if torch.is_tensor(raw_pose):
        raw_pose = raw_pose.detach().cpu().numpy()
    raw_pose = np.asarray(raw_pose, dtype=np.float32)
    if raw_pose.ndim == 2:
        raw_pose = raw_pose[0]
    return raw_pose[:7]


def _get_map4d_stackcube(env) -> np.ndarray:
    """Get 9-dim map4d (pos+rot6d) for StackCube. Shape (3, 9)."""
    cube_a_pose = _get_pose(env.cubeA)
    cube_b_pose = _get_pose(env.cubeB)
    table_pose = _get_pose(env.table_scene.table)
    poses = [cube_a_pose, cube_b_pose, table_pose]
    positions = np.stack([p[:3] for p in poses], axis=0)
    rotations = np.stack([_quat_wxyz_to_rot6d(p[3:7]) for p in poses], axis=0)
    return np.concatenate([positions, rotations], axis=-1).astype(np.float32)


def _get_map4d_plugcharger(env) -> np.ndarray:
    """Get 9-dim map4d (pos+rot6d) for PlugCharger. Shape (2, 9)."""
    charger_pose = _get_pose(env.charger)
    receptacle_pose = _get_pose(env.receptacle)
    poses = [charger_pose, receptacle_pose]
    positions = np.stack([p[:3] for p in poses], axis=0)
    rotations = np.stack([_quat_wxyz_to_rot6d(p[3:7]) for p in poses], axis=0)
    return np.concatenate([positions, rotations], axis=-1).astype(np.float32)


class DiTEvalObservationWrapper(gym.Wrapper):
    """Wraps ManiSkill env to produce observations for the Map4D DiT policy."""

    def __init__(self, env, *, task_name: str, n_obs_steps: int = 2, robot_state_dim: int = 32):
        super().__init__(env)
        self.task_name = task_name
        self.n_obs_steps = n_obs_steps
        self.robot_state_dim = robot_state_dim
        self._map4d_history = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        current_map4d = self._current_map4d()
        self._map4d_history = [current_map4d.copy() for _ in range(self.n_obs_steps)]
        return self._build_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._map4d_history.append(self._current_map4d())
        self._map4d_history = self._map4d_history[-self.n_obs_steps:]
        return self._build_obs(obs), reward, terminated, truncated, info

    def _build_obs(self, raw_obs) -> Dict[str, np.ndarray]:
        robot_state = self._get_robot_state(raw_obs)
        map4d = np.stack(self._map4d_history, axis=0).astype(np.float32)
        return {
            "robot_state": robot_state,
            "map4d": map4d,
        }

    def _get_robot_state(self, obs) -> np.ndarray:
        """Extract robot state from raw obs, pad/truncate to robot_state_dim."""
        if isinstance(obs, dict):
            parts = []
            if "agent" in obs and isinstance(obs["agent"], dict):
                for k in ("qpos", "qvel"):
                    if k in obs["agent"]:
                        parts.append(np.asarray(obs["agent"][k], dtype=np.float32).flatten())
            if "extra" in obs and isinstance(obs["extra"], dict):
                if "tcp_pose" in obs["extra"]:
                    parts.append(np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).flatten())
            if not parts:
                # Flattened obs
                if "state" in obs:
                    parts.append(np.asarray(obs["state"], dtype=np.float32).flatten())
            state = np.concatenate(parts) if parts else np.zeros(self.robot_state_dim, dtype=np.float32)
        else:
            state = np.asarray(obs, dtype=np.float32).flatten()
        # Pad or truncate
        if state.shape[0] < self.robot_state_dim:
            state = np.concatenate([state, np.zeros(self.robot_state_dim - state.shape[0], dtype=np.float32)])
        elif state.shape[0] > self.robot_state_dim:
            state = state[:self.robot_state_dim]
        return state

    def _current_map4d(self) -> np.ndarray:
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        if self.task_name == "StackCube-v1":
            return _get_map4d_stackcube(env)
        elif self.task_name == "PlugCharger-v1":
            return _get_map4d_plugcharger(env)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")


def _convert_action(action_8d: np.ndarray, task_name: str) -> np.ndarray:
    """Convert DiT 8-dim output (pos3+quat4+gripper1) to env action."""
    pos = action_8d[..., :3]
    quat = action_8d[..., 3:7]
    gripper = action_8d[..., 7:8]

    task_cfg = TASK_CONFIG[task_name]
    if task_cfg["action_dim"] == 7:
        # pos + axis_angle + gripper
        axis_angle = _quat_wxyz_to_axis_angle(quat)
        return np.concatenate([pos, axis_angle, gripper], axis=-1).astype(np.float32)
    elif task_cfg["action_dim"] == 4:
        # pos + gripper (ignore rotation)
        return np.concatenate([pos, gripper], axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported action_dim={task_cfg['action_dim']}")


@torch.no_grad()
def evaluate_maniskill(
    policy,
    task_name: str,
    *,
    num_eval_episodes: int = 100,
    num_eval_envs: int = 10,
    n_obs_steps: int = 2,
    robot_state_dim: int = 32,
    size_parameter_dim: int = 0,
    relation_parameter_dim: int = 0,
    device: torch.device = torch.device("cuda"),
    use_rgb: bool = False,
    rgb_feature_dim: int = 384,
) -> Dict[str, float]:
    """Run env evaluation and return metrics."""
    task_cfg = TASK_CONFIG[task_name]
    max_episode_steps = task_cfg["max_episode_steps"]

    # Create envs
    from mani_skill.utils.wrappers import CPUGymWrapper

    def make_env(seed):
        def thunk():
            env = gym.make(
                task_name,
                obs_mode="state",
                control_mode=task_cfg["control_mode"],
                render_mode="rgb_array",
                sim_backend="cpu",
                max_episode_steps=max_episode_steps,
                reconfiguration_freq=1,
            )
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            env = DiTEvalObservationWrapper(
                env,
                task_name=task_name,
                n_obs_steps=n_obs_steps,
                robot_state_dim=robot_state_dim,
            )
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv(
        [make_env(seed) for seed in range(num_eval_envs)]
    )

    # Constant parameters
    size_params = TASK_SIZE_PARAMETERS.get(task_name, np.zeros(size_parameter_dim, dtype=np.float32))
    relation_params = TASK_RELATION_PARAMETERS.get(task_name, np.zeros(relation_parameter_dim, dtype=np.float32))
    if size_parameter_dim > 0 and size_params.shape[0] != size_parameter_dim:
        size_params = size_params[:size_parameter_dim] if size_params.shape[0] > size_parameter_dim else np.pad(size_params, (0, size_parameter_dim - size_params.shape[0]))
    if relation_parameter_dim > 0 and relation_params.shape[0] != relation_parameter_dim:
        relation_params = relation_params[:relation_parameter_dim] if relation_params.shape[0] > relation_parameter_dim else np.pad(relation_params, (0, relation_parameter_dim - relation_params.shape[0]))

    policy.eval()
    eval_metrics = defaultdict(list)
    obs_batch, _ = envs.reset()
    eps_count = 0
    n_action_steps = policy.n_action_steps
    action_buffer = None
    action_idx = 0

    while eps_count < num_eval_episodes:
        if action_buffer is None or action_idx >= n_action_steps:
            # Build policy input
            obs_dict = {
                "robot_state": torch.from_numpy(
                    np.stack(obs_batch["robot_state"])
                ).float().unsqueeze(1).expand(-1, n_obs_steps, -1).to(device),
                "map4d": torch.from_numpy(
                    np.stack(obs_batch["map4d"])
                ).float().to(device),
                "size_parameters": torch.from_numpy(
                    np.tile(size_params, (num_eval_envs, 1))
                ).float().to(device),
            }
            if relation_parameter_dim > 0:
                obs_dict["relation_parameters"] = torch.from_numpy(
                    np.tile(relation_params, (num_eval_envs, 1))
                ).float().to(device)
            if use_rgb:
                obs_dict["rgb_feature"] = torch.zeros(
                    num_eval_envs, n_obs_steps, rgb_feature_dim, device=device
                )

            result = policy.predict_action(obs_dict)
            action_buffer = result["action"].cpu().numpy()
            action_idx = 0

        # Step env
        actions_8d = action_buffer[:, action_idx]
        env_actions = np.stack([
            _convert_action(actions_8d[i], task_name)
            for i in range(num_eval_envs)
        ])
        action_idx += 1

        obs_batch, _, _, truncated, info = envs.step(env_actions)

        if np.any(truncated):
            if "final_info" in info and isinstance(info["final_info"], dict):
                for k, v in info["final_info"]["episode"].items():
                    eval_metrics[k].append(v)
            elif "final_info" in info and isinstance(info["final_info"], (list, tuple)):
                for i, fi in enumerate(info["final_info"]):
                    if fi is not None and "episode" in fi:
                        for k, v in fi["episode"].items():
                            eval_metrics[k].append(v)
            eps_count += int(np.sum(truncated))
            action_buffer = None
            action_idx = 0

    envs.close()

    result = {}
    for k, v in eval_metrics.items():
        arr = np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)
        result[k] = float(arr.mean())
    return result
