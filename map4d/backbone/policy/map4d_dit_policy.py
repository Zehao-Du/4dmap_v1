"""Policy wrapper for the standalone Map4D DiT backbone."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf

from map4d.backbone.model.common.normalizer import LinearNormalizer
from map4d.backbone.model.diffusion.map4d_dit import Map4DDiT, normalize_quaternion
from map4d.backbone.policy.base_policy import BasePolicy


class Map4DDiTPolicy(BasePolicy):
    """Joint diffusion policy for trajectory and Map4D keyframe latent tokens."""

    def __init__(
        self,
        *,
        model_cfg,
        noise_scheduler_cfg,
        action_horizon: int,
        keyframe_horizon: int,
        n_action_steps: Optional[int] = None,
        n_obs_steps: int = 2,
        num_inference_steps: Optional[int] = None,
        trajectory_loss_weight: float = 1.0,
        keyframe_tcp_loss_weight: float = 1.0,
        keyframe_map4d_loss_weight: Optional[float] = None,
        keyframe_object_loss_weight: Optional[float] = None,
        gripper_loss_weight: float = 1.0,
    ):
        super().__init__()
        if not isinstance(model_cfg, dict):
            model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        self.model = Map4DDiT(
            action_horizon=action_horizon,
            keyframe_horizon=keyframe_horizon,
            obs_horizon=n_obs_steps,
            **model_cfg,
        )
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_cfg.num_train_timesteps,
            beta_schedule=getattr(noise_scheduler_cfg, "position_beta_schedule", "scaled_linear"),
            prediction_type=noise_scheduler_cfg.prediction_type,
            clip_sample=False,
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_cfg.num_train_timesteps,
            beta_schedule=getattr(noise_scheduler_cfg, "rotation_beta_schedule", "squaredcos_cap_v2"),
            prediction_type=noise_scheduler_cfg.prediction_type,
            clip_sample=False,
        )
        self.noise_scheduler_cfg = noise_scheduler_cfg
        self.action_horizon = int(action_horizon)
        self.keyframe_horizon = int(keyframe_horizon)
        self.n_action_steps = int(n_action_steps or action_horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.num_inference_steps = int(num_inference_steps or noise_scheduler_cfg.num_train_timesteps)
        if keyframe_map4d_loss_weight is None:
            keyframe_map4d_loss_weight = 0.3 if keyframe_object_loss_weight is None else keyframe_object_loss_weight
        self.loss_weights = {
            "trajectory": float(trajectory_loss_weight),
            "keyframe_tcp": float(keyframe_tcp_loss_weight),
            "keyframe_map4d": float(keyframe_map4d_loss_weight),
            "gripper": float(gripper_loss_weight),
        }
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _has_norm(self, key: str) -> bool:
        return key in self.normalizer.params_dict

    def _normalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if self._has_norm(key):
            return self.normalizer[key].normalize(value)
        return value

    def _unnormalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if self._has_norm(key):
            return self.normalizer[key].unnormalize(value)
        return value

    def _normalize_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {key: value.to(device=self.device, dtype=self.dtype) for key, value in obs.items()}
        if "robot_state" in out:
            out["robot_state"] = self._normalize("robot_state", out["robot_state"])
        return out

    def _normalize_targets(self, batch) -> Dict[str, torch.Tensor]:
        trajectory = batch["action"]["trajectory"].clone()
        trajectory[..., 0:3] = self._normalize("trajectory_pos", trajectory[..., 0:3])
        trajectory[..., 3:7] = normalize_quaternion(trajectory[..., 3:7])

        keyframe_map4d = batch["keyframe"]["map4d"].clone()
        keyframe_map4d[..., 0:3] = self._normalize("keyframe_map4d_pos", keyframe_map4d[..., 0:3])

        keyframe_tcp = batch["keyframe"]["tcp"].clone()
        keyframe_tcp[..., 0:3] = self._normalize("keyframe_tcp_pos", keyframe_tcp[..., 0:3])
        if keyframe_tcp.shape[-1] == 7:
            keyframe_tcp[..., 3:7] = normalize_quaternion(keyframe_tcp[..., 3:7])
        elif keyframe_tcp.shape[-1] == 4:
            keyframe_tcp[..., 3:4] = self._normalize(
                "keyframe_tcp_gripper", keyframe_tcp[..., 3:4]
            )
        else:
            raise ValueError(f"Unsupported keyframe TCP dim {keyframe_tcp.shape[-1]}")

        gripper = batch["action"]["gripper_openness"].clone()
        gripper = self._normalize("gripper_openness", gripper)
        return {
            "trajectory": trajectory,
            "keyframe_map4d": keyframe_map4d,
            "keyframe_tcp": keyframe_tcp,
            "gripper_openness": gripper,
        }

    def _unnormalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        out = trajectory.clone()
        out[..., 0:3] = self._unnormalize("trajectory_pos", out[..., 0:3])
        out[..., 3:7] = normalize_quaternion(out[..., 3:7])
        return out

    def _unnormalize_gripper(self, gripper: torch.Tensor) -> torch.Tensor:
        return self._unnormalize("gripper_openness", gripper)

    def _add_noise(self, targets: Dict[str, torch.Tensor], timesteps: torch.Tensor):
        trajectory = targets["trajectory"]
        keyframe_map4d = targets["keyframe_map4d"]
        keyframe_tcp = targets["keyframe_tcp"]
        noise = {
            "trajectory": torch.randn_like(trajectory),
            "keyframe_map4d": torch.randn_like(keyframe_map4d),
            "keyframe_tcp": torch.randn_like(keyframe_tcp),
        }

        noisy_trajectory = torch.cat(
            [
                self.position_noise_scheduler.add_noise(
                    trajectory[..., 0:3], noise["trajectory"][..., 0:3], timesteps
                ),
                normalize_quaternion(
                    self.rotation_noise_scheduler.add_noise(
                        trajectory[..., 3:7], noise["trajectory"][..., 3:7], timesteps
                    )
                ),
            ],
            dim=-1,
        )
        noisy_map4d = torch.cat(
            [
                self.position_noise_scheduler.add_noise(
                    keyframe_map4d[..., 0:3], noise["keyframe_map4d"][..., 0:3], timesteps
                ),
                self.rotation_noise_scheduler.add_noise(
                    keyframe_map4d[..., 3:9], noise["keyframe_map4d"][..., 3:9], timesteps
                ),
            ],
            dim=-1,
        )
        tcp_pos = self.position_noise_scheduler.add_noise(
            keyframe_tcp[..., 0:3], noise["keyframe_tcp"][..., 0:3], timesteps
        )
        if keyframe_tcp.shape[-1] == 7:
            tcp_tail = normalize_quaternion(
                self.rotation_noise_scheduler.add_noise(
                    keyframe_tcp[..., 3:7], noise["keyframe_tcp"][..., 3:7], timesteps
                )
            )
        elif keyframe_tcp.shape[-1] == 4:
            tcp_tail = self.position_noise_scheduler.add_noise(
                keyframe_tcp[..., 3:4], noise["keyframe_tcp"][..., 3:4], timesteps
            )
        else:
            raise ValueError(f"Unsupported keyframe TCP dim {keyframe_tcp.shape[-1]}")
        noisy_tcp = torch.cat([tcp_pos, tcp_tail], dim=-1)
        noisy = {
            "trajectory": noisy_trajectory,
            "keyframe_map4d": noisy_map4d,
            "keyframe_tcp": noisy_tcp,
        }
        return noisy, noise

    def forward(self, batch):
        obs = self._normalize_obs(batch["obs"])
        targets = self._normalize_targets(batch)
        batch_size = targets["trajectory"].shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler_cfg.num_train_timesteps,
            (batch_size,),
            device=targets["trajectory"].device,
        ).long()

        noisy, noise = self._add_noise(targets, timesteps)
        pred = self.model(noisy, timesteps, obs)

        trajectory_loss = F.l1_loss(pred["trajectory"], noise["trajectory"])
        keyframe_tcp_loss = F.l1_loss(pred["keyframe_tcp"], noise["keyframe_tcp"])
        keyframe_map4d_loss = F.l1_loss(pred["keyframe_map4d"], noise["keyframe_map4d"])
        gripper_loss = F.l1_loss(pred["gripper_openness"], targets["gripper_openness"])
        total_loss = (
            self.loss_weights["trajectory"] * trajectory_loss
            + self.loss_weights["keyframe_tcp"] * keyframe_tcp_loss
            + self.loss_weights["keyframe_map4d"] * keyframe_map4d_loss
            + self.loss_weights["gripper"] * gripper_loss
        )
        return total_loss, {
            "bc_loss": float(total_loss.detach().cpu()),
            "trajectory_noise_l1": float(trajectory_loss.detach().cpu()),
            "keyframe_tcp_noise_l1": float(keyframe_tcp_loss.detach().cpu()),
            "keyframe_map4d_noise_l1": float(keyframe_map4d_loss.detach().cpu()),
            "gripper_l1": float(gripper_loss.detach().cpu()),
        }

    def _sample_step(self, sample, pred, t):
        trajectory_pos = self.position_noise_scheduler.step(
            pred["trajectory"][..., 0:3], t, sample["trajectory"][..., 0:3]
        ).prev_sample
        trajectory_quat = self.rotation_noise_scheduler.step(
            pred["trajectory"][..., 3:7], t, sample["trajectory"][..., 3:7]
        ).prev_sample
        map4d_pos = self.position_noise_scheduler.step(
            pred["keyframe_map4d"][..., 0:3], t, sample["keyframe_map4d"][..., 0:3]
        ).prev_sample
        map4d_rot = self.rotation_noise_scheduler.step(
            pred["keyframe_map4d"][..., 3:9], t, sample["keyframe_map4d"][..., 3:9]
        ).prev_sample
        tcp_pos = self.position_noise_scheduler.step(
            pred["keyframe_tcp"][..., 0:3], t, sample["keyframe_tcp"][..., 0:3]
        ).prev_sample
        if sample["keyframe_tcp"].shape[-1] == 7:
            tcp_tail = normalize_quaternion(
                self.rotation_noise_scheduler.step(
                    pred["keyframe_tcp"][..., 3:7], t, sample["keyframe_tcp"][..., 3:7]
                ).prev_sample
            )
        elif sample["keyframe_tcp"].shape[-1] == 4:
            tcp_tail = self.position_noise_scheduler.step(
                pred["keyframe_tcp"][..., 3:4], t, sample["keyframe_tcp"][..., 3:4]
            ).prev_sample
        else:
            raise ValueError(f"Unsupported keyframe TCP dim {sample['keyframe_tcp'].shape[-1]}")
        return {
            "trajectory": torch.cat([trajectory_pos, normalize_quaternion(trajectory_quat)], dim=-1),
            "keyframe_map4d": torch.cat([map4d_pos, map4d_rot], dim=-1),
            "keyframe_tcp": torch.cat([tcp_pos, tcp_tail], dim=-1),
        }

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = self._normalize_obs(obs_dict)
        value = obs["robot_state"]
        batch_size = value.shape[0]
        device = self.device
        dtype = self.dtype
        num_arms = int(getattr(self.model, "num_arms", 1))

        if num_arms == 1:
            trajectory_shape = (batch_size, self.action_horizon, 7)
            tcp_shape = (batch_size, self.keyframe_horizon, self.model.tcp_dim)
        else:
            trajectory_shape = (batch_size, num_arms, self.action_horizon, 7)
            tcp_shape = (batch_size, num_arms, self.keyframe_horizon, self.model.tcp_dim)

        sample = {
            "trajectory": torch.randn(*trajectory_shape, device=device, dtype=dtype),
            "keyframe_map4d": torch.randn(
                batch_size,
                self.keyframe_horizon,
                self.model.num_objects,
                9,
                device=device,
                dtype=dtype,
            ),
            "keyframe_tcp": torch.randn(*tcp_shape, device=device, dtype=dtype),
        }
        sample["trajectory"][..., 3:7] = normalize_quaternion(sample["trajectory"][..., 3:7])
        if self.model.tcp_dim == 7:
            sample["keyframe_tcp"][..., 3:7] = normalize_quaternion(sample["keyframe_tcp"][..., 3:7])

        self.position_noise_scheduler.set_timesteps(self.num_inference_steps)
        self.rotation_noise_scheduler.set_timesteps(self.num_inference_steps)
        latest_pred = None
        for t in self.position_noise_scheduler.timesteps:
            timestep = t * torch.ones(batch_size, device=device, dtype=torch.long)
            latest_pred = self.model(sample, timestep, obs)
            sample = self._sample_step(sample, latest_pred, t)

        trajectory_pred = self._unnormalize_trajectory(sample["trajectory"])
        gripper_pred = self._unnormalize_gripper(latest_pred["gripper_openness"])
        action_pred = torch.cat([trajectory_pred, gripper_pred], dim=-1)
        if num_arms == 1:
            action = action_pred[:, : self.n_action_steps]
        else:
            action = action_pred[:, :, : self.n_action_steps]
        return {
            "action": action,
            "action_pred": action_pred,
            "trajectory_pred": trajectory_pred,
            "gripper_openness_pred": gripper_pred,
            "keyframe_map4d_latent": sample["keyframe_map4d"],
            "keyframe_tcp_latent": sample["keyframe_tcp"],
        }
