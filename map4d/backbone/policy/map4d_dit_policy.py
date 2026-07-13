"""Policy wrapper for the standalone Map4D DiT backbone."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf

from map4d.backbone.model.common.normalizer import LinearNormalizer
from map4d.backbone.model.diffusion.mask_generator import LowdimMaskGenerator
from map4d.backbone.model.diffusion.map4d_dit import Map4DDiT
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
        self.mask_generator = LowdimMaskGenerator(
            action_dim=int(self.model.trajectory_dim),
            obs_dim=0,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=True,
        )

    def _normalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.shape[-1] == 4:
            return torch.cat(
                [
                    self.normalizer["trajectory_pos"].normalize(trajectory[..., 0:3]),
                    self.normalizer["gripper_openness"].normalize(trajectory[..., 3:4]),
                ],
                dim=-1,
            )
        if trajectory.shape[-1] == 7:
            return torch.cat(
                [
                    self.normalizer["trajectory_pos"].normalize(trajectory[..., 0:3]),
                    trajectory[..., 3:7],
                ],
                dim=-1,
            )
        raise ValueError(f"Unsupported trajectory dim {trajectory.shape[-1]}")

    def _unnormalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.shape[-1] == 4:
            return torch.cat(
                [
                    self.normalizer["trajectory_pos"].unnormalize(trajectory[..., 0:3]),
                    self.normalizer["gripper_openness"].unnormalize(trajectory[..., 3:4]),
                ],
                dim=-1,
            )
        if trajectory.shape[-1] == 7:
            return torch.cat(
                [
                    self.normalizer["trajectory_pos"].unnormalize(trajectory[..., 0:3]),
                    trajectory[..., 3:7],
                ],
                dim=-1,
            )
        raise ValueError(f"Unsupported trajectory dim {trajectory.shape[-1]}")

    def _normalize_keyframe_tcp(self, keyframe_tcp: torch.Tensor) -> torch.Tensor:
        if keyframe_tcp.shape[-1] == 4:
            return torch.cat(
                [
                    self.normalizer["keyframe_tcp_pos"].normalize(keyframe_tcp[..., 0:3]),
                    self.normalizer["keyframe_tcp_gripper"].normalize(keyframe_tcp[..., 3:4]),
                ],
                dim=-1,
            )
        if keyframe_tcp.shape[-1] == 7:
            return torch.cat(
                [
                    self.normalizer["keyframe_tcp_pos"].normalize(keyframe_tcp[..., 0:3]),
                    keyframe_tcp[..., 3:7],
                ],
                dim=-1,
            )
        raise ValueError(f"Unsupported keyframe TCP dim {keyframe_tcp.shape[-1]}")

    def _unnormalize_keyframe_tcp(self, keyframe_tcp: torch.Tensor) -> torch.Tensor:
        if keyframe_tcp.shape[-1] == 4:
            return torch.cat(
                [
                    self.normalizer["keyframe_tcp_pos"].unnormalize(keyframe_tcp[..., 0:3]),
                    self.normalizer["keyframe_tcp_gripper"].unnormalize(keyframe_tcp[..., 3:4]),
                ],
                dim=-1,
            )
        if keyframe_tcp.shape[-1] == 7:
            return torch.cat(
                [
                    self.normalizer["keyframe_tcp_pos"].unnormalize(keyframe_tcp[..., 0:3]),
                    keyframe_tcp[..., 3:7],
                ],
                dim=-1,
            )
        raise ValueError(f"Unsupported keyframe TCP dim {keyframe_tcp.shape[-1]}")

    # ========= inference / testing ============
    @torch.no_grad()
    def conditional_sample_map4d_dit(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        obs: Dict[str, torch.Tensor],
    ):
        """Testing denoising loop, matching PPI's conditional_sample_* entry."""
        self.position_noise_scheduler.set_timesteps(self.num_inference_steps)
        self.rotation_noise_scheduler.set_timesteps(self.num_inference_steps)

        batch_size = condition_data.shape[0]
        device = condition_data.device
        dtype = condition_data.dtype
        num_arms = int(getattr(self.model, "num_arms", 1))

        if num_arms == 1:
            tcp_shape = (batch_size, self.keyframe_horizon, self.model.tcp_dim)
        else:
            tcp_shape = (batch_size, num_arms, self.keyframe_horizon, self.model.tcp_dim)

        trajectory = torch.randn_like(condition_data)
        keyframe_tcp = torch.randn(*tcp_shape, device=device, dtype=dtype)
        if condition_data.shape[-1] == 4:
            trajectory[..., 3:4].zero_()
        trajectory = torch.where(condition_mask, condition_data, trajectory)

        sample = {"trajectory": trajectory, "keyframe_tcp": keyframe_tcp}
        latest_pred = None
        for t in self.position_noise_scheduler.timesteps:
            timestep = t * torch.ones(batch_size, device=device, dtype=torch.long)
            latest_pred = self.model(sample, timestep, obs)

            trajectory_pos = self.position_noise_scheduler.step(
                latest_pred["trajectory"][..., 0:3], t, sample["trajectory"][..., 0:3]
            ).prev_sample
            if sample["trajectory"].shape[-1] == 7:
                trajectory_tail = self.rotation_noise_scheduler.step(
                    latest_pred["trajectory"][..., 3:7], t, sample["trajectory"][..., 3:7]
                ).prev_sample
            elif sample["trajectory"].shape[-1] == 4:
                trajectory_tail = torch.zeros_like(latest_pred["trajectory"][..., 3:4])
            else:
                raise ValueError(f"Unsupported trajectory dim {sample['trajectory'].shape[-1]}")

            tcp_pos = self.position_noise_scheduler.step(
                latest_pred["keyframe_tcp"][..., 0:3], t, sample["keyframe_tcp"][..., 0:3]
            ).prev_sample
            if sample["keyframe_tcp"].shape[-1] == 7:
                tcp_tail = self.rotation_noise_scheduler.step(
                    latest_pred["keyframe_tcp"][..., 3:7], t, sample["keyframe_tcp"][..., 3:7]
                ).prev_sample
            elif sample["keyframe_tcp"].shape[-1] == 4:
                tcp_tail = self.position_noise_scheduler.step(
                    latest_pred["keyframe_tcp"][..., 3:4], t, sample["keyframe_tcp"][..., 3:4]
                ).prev_sample
            else:
                raise ValueError(f"Unsupported keyframe TCP dim {sample['keyframe_tcp'].shape[-1]}")

            trajectory = torch.cat([trajectory_pos, trajectory_tail], dim=-1)
            trajectory = torch.where(condition_mask, condition_data, trajectory)
            sample = {
                "trajectory": trajectory,
                "keyframe_tcp": torch.cat([tcp_pos, tcp_tail], dim=-1),
            }
        return sample, latest_pred

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Testing entry: normalize obs, then denoise action/TCP."""
        nobs = {
            key: self.normalizer.normalize({key: value})[key] if key in self.normalizer.params_dict else value
            for key, value in obs_dict.items()
        }
        value = nobs["robot_state"]
        batch_size = value.shape[0]
        device = self.device
        dtype = self.dtype
        num_arms = int(getattr(self.model, "num_arms", 1))

        if num_arms == 1:
            trajectory_shape = (batch_size, self.action_horizon, int(self.model.trajectory_dim))
        else:
            trajectory_shape = (batch_size, num_arms, self.action_horizon, int(self.model.trajectory_dim))

        condition_data = torch.zeros(*trajectory_shape, device=device, dtype=dtype)
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        sample, latest_pred = self.conditional_sample_map4d_dit(condition_data, condition_mask, nobs)

        ntrajectory_pred = sample["trajectory"].clone()
        if ntrajectory_pred.shape[-1] == 7:
            ngripper_pred = latest_pred["gripper_openness"]
            trajectory_pred = self._unnormalize_trajectory(ntrajectory_pred)
            gripper_pred = self.normalizer["gripper_openness"].unnormalize(ngripper_pred)
            action_pred = torch.cat([trajectory_pred, gripper_pred], dim=-1)
        elif ntrajectory_pred.shape[-1] == 4:
            ngripper_pred = latest_pred["trajectory"][..., 3:4]
            ntrajectory_pred[..., 3:4] = ngripper_pred
            trajectory_pred = self._unnormalize_trajectory(ntrajectory_pred)
            gripper_pred = trajectory_pred[..., 3:4]
            action_pred = trajectory_pred
        else:
            raise ValueError(f"Unsupported trajectory dim {ntrajectory_pred.shape[-1]}")

        keyframe_node_position_pred = self.normalizer["keyframe_map4d_pos"].unnormalize(
            latest_pred["keyframe_node_position"]
        )
        keyframe_tcp_latent = self._unnormalize_keyframe_tcp(sample["keyframe_tcp"])

        if num_arms == 1:
            action = action_pred[:, : self.n_action_steps]
        else:
            action = action_pred[:, :, : self.n_action_steps]
        return {
            "action": action,
            "action_pred": action_pred,
            "trajectory_pred": trajectory_pred,
            "gripper_openness_pred": gripper_pred,
            "keyframe_node_position_pred": keyframe_node_position_pred,
            "keyframe_tcp_latent": keyframe_tcp_latent,
        }

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Training setup: copy dataset normalizer into the policy."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch):
        """Training entry: normalize obs, add diffusion noise, predict noise, compute losses."""
        nobs = self.normalizer.normalize(batch["obs"])
        ntrajectory = self._normalize_trajectory(batch["action"]["trajectory"])
        if ntrajectory.shape[-1] not in {4, 7}:
            raise ValueError(f"Unsupported trajectory dim {ntrajectory.shape[-1]}")

        nkeyframe_node_position = self.normalizer["keyframe_map4d_pos"].normalize(
            batch["keyframe"]["map4d"][..., 0:3]
        )

        nkeyframe_tcp = self._normalize_keyframe_tcp(batch["keyframe"]["tcp"])
        if nkeyframe_tcp.shape[-1] not in {4, 7}:
            raise ValueError(f"Unsupported keyframe TCP dim {nkeyframe_tcp.shape[-1]}")

        if ntrajectory.shape[-1] == 4:
            ngripper = ntrajectory[..., 3:4]
        else:
            ngripper = self.normalizer["gripper_openness"].normalize(batch["action"]["gripper_openness"])

        batch_size = ntrajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler_cfg.num_train_timesteps,
            (batch_size,),
            device=ntrajectory.device,
        ).long()

        noise = {
            "trajectory": torch.randn_like(ntrajectory),
            "keyframe_tcp": torch.randn_like(nkeyframe_tcp),
        }
        trajectory_pos = self.position_noise_scheduler.add_noise(
            ntrajectory[..., 0:3], noise["trajectory"][..., 0:3], timesteps
        )
        if ntrajectory.shape[-1] == 7:
            trajectory_tail = self.rotation_noise_scheduler.add_noise(
                ntrajectory[..., 3:7], noise["trajectory"][..., 3:7], timesteps
            )
        else:
            trajectory_tail = torch.zeros_like(ntrajectory[..., 3:4])
            noise["trajectory"][..., 3:4].zero_()
        tcp_pos = self.position_noise_scheduler.add_noise(
            nkeyframe_tcp[..., 0:3], noise["keyframe_tcp"][..., 0:3], timesteps
        )
        if nkeyframe_tcp.shape[-1] == 7:
            tcp_tail = self.rotation_noise_scheduler.add_noise(
                nkeyframe_tcp[..., 3:7], noise["keyframe_tcp"][..., 3:7], timesteps
            )
        else:
            tcp_tail = self.position_noise_scheduler.add_noise(
                nkeyframe_tcp[..., 3:4], noise["keyframe_tcp"][..., 3:4], timesteps
            )

        noisy = {
            "trajectory": torch.cat([trajectory_pos, trajectory_tail], dim=-1),
            "keyframe_tcp": torch.cat([tcp_pos, tcp_tail], dim=-1),
        }
        if ntrajectory.ndim == 4:
            batch_size, num_arms, horizon, dim = ntrajectory.shape
            condition_mask = self.mask_generator((batch_size * num_arms, horizon, dim)).reshape(
                batch_size, num_arms, horizon, dim
            )
        else:
            condition_mask = self.mask_generator(ntrajectory.shape)
        if ntrajectory.shape[-1] == 4:
            condition_mask = condition_mask.clone()
            condition_mask[..., 3:4] = False
        noisy["trajectory"] = torch.where(condition_mask, ntrajectory, noisy["trajectory"])
        pred = self.model(noisy, timesteps, nobs)

        trajectory_pos_loss = F.l1_loss(pred["trajectory"][..., 0:3], noise["trajectory"][..., 0:3])
        if ntrajectory.shape[-1] == 7:
            trajectory_rot_loss = F.l1_loss(pred["trajectory"][..., 3:7], noise["trajectory"][..., 3:7])
            trajectory_loss = F.l1_loss(pred["trajectory"], noise["trajectory"])
            gripper_loss = F.l1_loss(pred["gripper_openness"], ngripper)
        else:
            trajectory_rot_loss = pred["trajectory"].new_tensor(0.0)
            trajectory_loss = trajectory_pos_loss
            gripper_loss = F.l1_loss(pred["trajectory"][..., 3:4], ntrajectory[..., 3:4])

        if nkeyframe_tcp.shape[-1] == 7:
            keyframe_tcp_pos_loss = F.l1_loss(pred["keyframe_tcp"][..., 0:3], noise["keyframe_tcp"][..., 0:3])
            keyframe_tcp_rot_loss = F.l1_loss(pred["keyframe_tcp"][..., 3:7], noise["keyframe_tcp"][..., 3:7])
        else:
            keyframe_tcp_pos_loss = F.l1_loss(pred["keyframe_tcp"][..., 0:3], noise["keyframe_tcp"][..., 0:3])
            keyframe_tcp_rot_loss = pred["keyframe_tcp"].new_tensor(0.0)
        keyframe_tcp_loss = F.l1_loss(pred["keyframe_tcp"], noise["keyframe_tcp"])
        keyframe_node_position_loss = F.l1_loss(pred["keyframe_node_position"], nkeyframe_node_position)
        total_loss = (
            self.loss_weights["trajectory"] * trajectory_loss
            + self.loss_weights["keyframe_tcp"] * keyframe_tcp_loss
            + self.loss_weights["keyframe_map4d"] * keyframe_node_position_loss
            + self.loss_weights["gripper"] * gripper_loss
        )
        return total_loss, {
            "bc_loss": float(total_loss.detach().cpu()),
            "trajectory_noise_l1": float(trajectory_loss.detach().cpu()),
            "trajectory_pos_noise_l1": float(trajectory_pos_loss.detach().cpu()),
            "trajectory_rot_noise_l1": float(trajectory_rot_loss.detach().cpu()),
            "keyframe_tcp_noise_l1": float(keyframe_tcp_loss.detach().cpu()),
            "keyframe_tcp_pos_noise_l1": float(keyframe_tcp_pos_loss.detach().cpu()),
            "keyframe_tcp_rot_noise_l1": float(keyframe_tcp_rot_loss.detach().cpu()),
            "keyframe_node_position_l1": float(keyframe_node_position_loss.detach().cpu()),
            "gripper_l1": float(gripper_loss.detach().cpu()),
        }
