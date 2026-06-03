from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import mani_skill.envs
import numpy as np
import torch
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers import RecordEpisode, CPUGymWrapper


TASK_NUM_OBJECTS = {
    "StackCube-v1": 3,
    "PlugCharger-v1": 2,
}


class ACTMap4dObservationWrapper(gym.Wrapper):
    """Inject GT map4d into ACT observations at each step."""

    def __init__(self, env, *, map4d_pre_horizon: int = 6, task_name: str = "StackCube-v1"):
        if task_name not in TASK_NUM_OBJECTS:
            raise ValueError(f"ACTMap4dObservationWrapper does not support {task_name}")
        super().__init__(env)
        self.map4d_pre_horizon = map4d_pre_horizon
        self.task_name = task_name
        self.num_objects = TASK_NUM_OBJECTS[task_name]
        self._history = []
        if isinstance(self.observation_space, spaces.Dict):
            obs_spaces = dict(self.observation_space.spaces)
            obs_spaces["map4d"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(map4d_pre_horizon, self.num_objects, 12),
                dtype=np.float32,
            )
            self.observation_space = spaces.Dict(obs_spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        current = self._current_map4d()
        self._history = [current.copy() for _ in range(self.map4d_pre_horizon)]
        return self._inject(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._history.append(self._current_map4d())
        self._history = self._history[-self.map4d_pre_horizon:]
        return self._inject(obs), reward, terminated, truncated, info

    def _inject(self, obs):
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["map4d"] = np.stack(self._history, axis=0).astype(np.float32)
        return obs

    def _current_map4d(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if self.task_name == "StackCube-v1":
            return self._map4d_stackcube(env)
        elif self.task_name == "PlugCharger-v1":
            return self._map4d_plugcharger(env)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

    def _map4d_stackcube(self, env):
        cube_a_pose = self._get_pose(env.cubeA)
        cube_b_pose = self._get_pose(env.cubeB)
        table_pose = self._get_pose(env.table_scene.table)
        cube_half_size = env.cube_half_size
        if torch.is_tensor(cube_half_size):
            cube_half_size = cube_half_size.detach().cpu().numpy()
        cube_size = np.asarray(cube_half_size, dtype=np.float32).reshape(-1) * 2.0
        table_size = np.array([
            float(env.table_scene.table_length),
            float(env.table_scene.table_width),
            float(env.table_scene.table_height),
        ], dtype=np.float32)
        sizes = np.stack([cube_size, cube_size, table_size], axis=0)
        poses = [cube_a_pose, cube_b_pose, table_pose]
        positions = np.stack([p[:3] for p in poses], axis=0)
        rotations = np.stack([self._quat_wxyz_to_rot6d(p[3:7]) for p in poses], axis=0)
        return np.concatenate([sizes, positions, rotations], axis=-1).astype(np.float32)

    def _map4d_plugcharger(self, env):
        charger_pose = self._get_pose(env.charger)
        receptacle_pose = self._get_pose(env.receptacle)
        base_size = np.asarray(env._base_size, dtype=np.float32) * 2.0
        receptacle_size = np.asarray(env._receptacle_size, dtype=np.float32) * 2.0
        sizes = np.stack([base_size, receptacle_size], axis=0)
        poses = [charger_pose, receptacle_pose]
        positions = np.stack([p[:3] for p in poses], axis=0)
        rotations = np.stack([self._quat_wxyz_to_rot6d(p[3:7]) for p in poses], axis=0)
        return np.concatenate([sizes, positions, rotations], axis=-1).astype(np.float32)

    @staticmethod
    def _get_pose(actor):
        raw_pose = actor.pose.raw_pose
        if torch.is_tensor(raw_pose):
            raw_pose = raw_pose.detach().cpu().numpy()
        raw_pose = np.asarray(raw_pose, dtype=np.float32)
        if raw_pose.ndim == 2:
            raw_pose = raw_pose[0]
        return raw_pose[:7]

    @staticmethod
    def _quat_wxyz_to_rot6d(quat):
        quat = np.asarray(quat, dtype=np.float32)
        quat = quat / max(np.linalg.norm(quat), 1e-8)
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        r00 = 1.0 - 2.0 * (y * y + z * z)
        r10 = 2.0 * (x * y + z * w)
        r20 = 2.0 * (x * z - y * w)
        r01 = 2.0 * (x * y - z * w)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r21 = 2.0 * (y * z + x * w)
        return np.array([r00, r10, r20, r01, r11, r21], dtype=np.float32)


def make_eval_envs(env_id, num_envs: int, sim_backend: str, env_kwargs: dict, other_kwargs: dict,
                   video_dir: Optional[str] = None, wrappers: list[gym.Wrapper] = [],
                   map4d_pre_horizon: int = 0, map4d_task_name: str = "StackCube-v1"):
    """Create vectorized environment for evaluation and/or recording videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu"
        env_kwargs: the environment kwargs.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
        map4d_pre_horizon: if > 0, inject GT map4d with this temporal horizon.
        map4d_task_name: task for map4d extraction.
    """
    if sim_backend == "physx_cpu":
        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict()):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if map4d_pre_horizon > 0:
                    env = ACTMap4dObservationWrapper(
                        env, map4d_pre_horizon=map4d_pre_horizon, task_name=map4d_task_name
                    )
                if video_dir:
                    env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False,
                                        info_on_video=True, source_type="act",
                                        source_desc="act evaluation rollout")
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk
        vector_cls = gym.vector.SyncVectorEnv if num_envs == 1 else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        env = vector_cls([cpu_make_env(env_id, seed, video_dir if seed == 0 else None, env_kwargs, other_kwargs) for seed in range(num_envs)])
    else:
        env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend, reconfiguration_freq=1, **env_kwargs)
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if map4d_pre_horizon > 0:
            env = ACTMap4dObservationWrapper(
                env, map4d_pre_horizon=map4d_pre_horizon, task_name=map4d_task_name
            )
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, save_video=True,
                                source_type="act", source_desc="act evaluation rollout",
                                max_steps_per_video=max_episode_steps)
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env
