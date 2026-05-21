from typing import Optional
import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


STACKCUBE_FALLBACK_TABLE_STATE = np.asarray(
    [-0.12, 0.0, -0.9196429, 0.70710677, 0.0, 0.0, 0.70710677],
    dtype=np.float32,
)
STACKCUBE_FALLBACK_SIZES = np.asarray(
    [
        [0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04],
        [1.2090764, 2.4178784, 0.91964292762787],
    ],
    dtype=np.float32,
)


def _quat_wxyz_to_rotation_6d(quat_wxyz):
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    if quat.ndim == 1:
        quat = quat[None]
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


class ManiSkillGTMap4dObservationWrapper(gym.Wrapper):
    """Inject StackCube GT 4D map tensors into FrameStack observations."""

    def __init__(
        self,
        env,
        *,
        obs_horizon: int,
        task_name: str = "StackCube-v1",
        strict: bool = True,
    ):
        if task_name != "StackCube-v1":
            raise ValueError("ManiSkillGTMap4dObservationWrapper supports StackCube-v1 only.")
        super().__init__(env)
        self.obs_horizon = int(obs_horizon)
        self.strict = bool(strict)
        self._history = []
        self.observation_space = gym.spaces.Dict(
            {
                **self.env.observation_space.spaces,
                "map4d": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.obs_horizon, 3, 12),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        current = self._current_map4d()
        self._history = [current.copy() for _ in range(self.obs_horizon)]
        return self._inject(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._history.append(self._current_map4d())
        self._history = self._history[-self.obs_horizon :]
        return self._inject(obs), reward, terminated, truncated, info

    def _inject(self, obs):
        obs = dict(obs)
        obs["map4d"] = np.stack(self._history, axis=0).astype(np.float32)
        return obs

    def _current_map4d(self):
        env = self.env.unwrapped
        try:
            cube_a = self._actor_state(getattr(env, "cubeA"))
            cube_b = self._actor_state(getattr(env, "cubeB"))
            table = self._table_state(env)
            sizes = self._current_sizes(env)
        except Exception:
            if self.strict:
                raise
            cube_a = np.zeros(7, dtype=np.float32)
            cube_b = np.zeros(7, dtype=np.float32)
            cube_a[3] = 1.0
            cube_b[3] = 1.0
            table = STACKCUBE_FALLBACK_TABLE_STATE
            sizes = STACKCUBE_FALLBACK_SIZES
        states = [cube_a, cube_b, table]
        positions = np.stack([state[:3] for state in states], axis=0)
        rotations = np.stack([_quat_wxyz_to_rotation_6d(state[3:7])[0] for state in states], axis=0)
        return np.concatenate([sizes, positions, rotations], axis=-1).astype(np.float32)

    @staticmethod
    def _current_sizes(env):
        cube_half_size = getattr(env, "cube_half_size", None)
        if torch.is_tensor(cube_half_size):
            cube_half_size = cube_half_size.detach().cpu().numpy()
        cube_size = np.asarray(cube_half_size, dtype=np.float32).reshape(-1) * 2.0
        table_scene = getattr(env, "table_scene", None)
        table_size = np.asarray(
            [
                float(getattr(table_scene, "table_length")),
                float(getattr(table_scene, "table_width")),
                float(getattr(table_scene, "table_height")),
            ],
            dtype=np.float32,
        )
        return np.stack([cube_size, cube_size, table_size], axis=0).astype(np.float32)

    @classmethod
    def _table_state(cls, env):
        table_scene = getattr(env, "table_scene", None)
        table_actor = getattr(table_scene, "table", None)
        if table_actor is None:
            return STACKCUBE_FALLBACK_TABLE_STATE
        return cls._actor_state(table_actor)

    @staticmethod
    def _actor_state(actor):
        raw_pose = actor.pose.raw_pose
        if torch.is_tensor(raw_pose):
            raw_pose = raw_pose.detach().cpu().numpy()
        raw_pose = np.asarray(raw_pose, dtype=np.float32)
        if raw_pose.ndim == 2:
            raw_pose = raw_pose[0]
        return raw_pose[:7].astype(np.float32)

def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
    map4d_source: Optional[str] = None,
    map4d_task_name: str = "StackCube-v1",
    map4d_strict: bool = True,
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "physx_cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict()
        ):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if map4d_source == "maniskill_gt":
                    env = ManiSkillGTMap4dObservationWrapper(
                        env,
                        obs_horizon=other_kwargs["obs_horizon"],
                        task_name=map4d_task_name,
                        strict=map4d_strict,
                    )
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="diffusion_policy",
                        source_desc="diffusion_policy evaluation rollout",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        )
        env = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                    other_kwargs,
                )
                for seed in range(num_envs)
            ]
        )
    else:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
        if map4d_source == "maniskill_gt":
            env = ManiSkillGTMap4dObservationWrapper(
                env,
                obs_horizon=other_kwargs["obs_horizon"],
                task_name=map4d_task_name,
                strict=map4d_strict,
            )
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="diffusion_policy",
                source_desc="diffusion_policy evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env
