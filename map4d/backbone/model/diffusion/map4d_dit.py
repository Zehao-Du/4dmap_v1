"""Map4D DiT denoiser for joint trajectory and keyframe latent planning."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from map4d.backbone.model.diffusion.conditional_dit1d import DiTBlock1D, _modulate
from map4d.backbone.model.diffusion.positional_embedding import SinusoidalPosEmb


def normalize_quaternion(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize WXYZ quaternions and choose the `w >= 0` representative."""
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)
    sign = torch.where(quat[..., :1] < 0.0, -1.0, 1.0).to(dtype=quat.dtype, device=quat.device)
    return quat * sign


def rot6d_to_matrix(rot6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert 6D rotation representation to a proper rotation matrix."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


class Map4DDiT(nn.Module):
    """DiT backbone for Map4D policy.

    Noisy target tokens are trajectory, keyframe-object, and keyframe-TCP tokens.
    Context tokens come from optional RGB features, Map4D object history, object
    sizes, and pairwise relation features. Robot state and diffusion timestep are
    used as global AdaLN conditioning.
    """

    trajectory_dim = 7
    object_dim = 9

    def __init__(
        self,
        *,
        robot_state_dim: int,
        num_objects: int,
        action_horizon: int,
        keyframe_horizon: int,
        obs_horizon: int = 2,
        map4d_dim: int = 9,
        size_parameter_dim: int = 0,
        rgb_feature_dim: int = 0,
        relation_parameter_dim: int = 0,
        tcp_dim: int = 7,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        diffusion_step_embed_dim: int = 256,
        use_rgb: bool = False,
        max_context_tokens: int = 1024,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if use_rgb and rgb_feature_dim <= 0:
            raise ValueError("rgb_feature_dim must be positive when use_rgb=True")

        self.robot_state_dim = int(robot_state_dim)
        self.num_objects = int(num_objects)
        self.action_horizon = int(action_horizon)
        self.keyframe_horizon = int(keyframe_horizon)
        self.obs_horizon = int(obs_horizon)
        self.map4d_dim = int(map4d_dim)
        self.size_parameter_dim = int(size_parameter_dim)
        self.rgb_feature_dim = int(rgb_feature_dim)
        self.relation_parameter_dim = int(relation_parameter_dim)
        self.tcp_dim = int(tcp_dim)
        self.embed_dim = int(embed_dim)
        self.use_rgb = bool(use_rgb)
        self.max_context_tokens = int(max_context_tokens)
        if self.tcp_dim not in {4, 7}:
            raise ValueError(f"tcp_dim must be 4 or 7, got {self.tcp_dim}")

        self.trajectory_proj = nn.Linear(self.trajectory_dim, embed_dim)
        self.object_proj = nn.Linear(self.object_dim, embed_dim)
        self.tcp_proj = nn.Linear(self.tcp_dim, embed_dim)
        self.map4d_proj = nn.Sequential(
            nn.Linear(map4d_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.size_proj = (
            nn.Sequential(
                nn.Linear(size_parameter_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            if size_parameter_dim > 0
            else None
        )
        self.relation_proj = (
            nn.Sequential(
                nn.Linear(relation_parameter_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            if relation_parameter_dim > 0
            else None
        )
        self.rgb_proj = (
            nn.Sequential(
                nn.Linear(rgb_feature_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            if use_rgb
            else None
        )

        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_state_dim * obs_horizon, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        target_tokens = action_horizon + keyframe_horizon * num_objects + keyframe_horizon
        self.target_pos_embed = nn.Parameter(torch.zeros(1, target_tokens, embed_dim))
        self.context_pos_embed = nn.Parameter(torch.zeros(1, max_context_tokens, embed_dim))
        self.type_embed = nn.Parameter(torch.zeros(6, embed_dim))

        self.blocks = nn.ModuleList(
            [
                DiTBlock1D(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.trajectory_head = nn.Linear(embed_dim, self.trajectory_dim)
        self.object_head = nn.Linear(embed_dim, self.object_dim)
        self.tcp_head = nn.Linear(embed_dim, self.tcp_dim)
        self.gripper_head = nn.Linear(embed_dim, 1)

        nn.init.normal_(self.target_pos_embed, std=0.02)
        nn.init.normal_(self.context_pos_embed, std=0.02)
        nn.init.normal_(self.type_embed, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

    def _make_timestep(self, timestep: Union[torch.Tensor, float, int], batch_size: int, device):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device=device)
        else:
            timestep = timestep.to(device=device)
        return timestep.expand(batch_size).long()

    def _context_tokens(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        robot_state = obs["robot_state"]
        map4d = obs["map4d"]
        if robot_state.ndim != 3:
            raise ValueError(f"obs.robot_state must have shape [B, T, D], got {robot_state.shape}")
        if map4d.ndim != 4:
            raise ValueError(f"obs.map4d must have shape [B, T, N, D], got {map4d.shape}")
        if map4d.shape[2] != self.num_objects:
            raise ValueError(f"Expected {self.num_objects} map4d objects, got {map4d.shape[2]}")
        if map4d.shape[-1] != self.map4d_dim:
            raise ValueError(f"Expected map4d_dim={self.map4d_dim}, got {map4d.shape[-1]}")

        robot_state = robot_state[:, -self.obs_horizon :]
        if robot_state.shape[1] < self.obs_horizon:
            pad = robot_state[:, :1].expand(-1, self.obs_horizon - robot_state.shape[1], -1)
            robot_state = torch.cat([pad, robot_state], dim=1)
        global_cond = self.robot_encoder(robot_state.reshape(robot_state.shape[0], -1))

        map4d = map4d[:, -self.obs_horizon :]
        map_tokens = self.map4d_proj(map4d.reshape(map4d.shape[0], -1, map4d.shape[-1]))
        map_tokens = map_tokens + self.type_embed[3]

        context = [map_tokens]
        if self.size_proj is not None:
            size_parameters = obs.get("size_parameters")
            if size_parameters is None:
                raise ValueError("Map4DDiT requires obs.size_parameters when size_parameter_dim > 0")
            if size_parameters.shape[-1] != self.size_parameter_dim:
                raise ValueError(
                    f"Expected size_parameter_dim={self.size_parameter_dim}, got {size_parameters.shape[-1]}"
                )
            context.append(self.size_proj(size_parameters)[:, None, :] + self.type_embed[4])
        if self.relation_proj is not None:
            relation_parameters = obs.get("relation_parameters")
            if relation_parameters is None:
                raise ValueError("Map4DDiT requires obs.relation_parameters when relation_parameter_dim > 0")
            if relation_parameters.shape[-1] != self.relation_parameter_dim:
                raise ValueError(
                    "Expected relation_parameter_dim="
                    f"{self.relation_parameter_dim}, got {relation_parameters.shape[-1]}"
                )
            context.append(self.relation_proj(relation_parameters)[:, None, :] + self.type_embed[5])

        if self.use_rgb:
            rgb_feature = obs.get("rgb_feature")
            if rgb_feature is None:
                raise ValueError("Map4DDiT was configured with use_rgb=True but obs.rgb_feature is missing")
            rgb_feature = rgb_feature.reshape(rgb_feature.shape[0], -1, rgb_feature.shape[-1])
            context.append(self.rgb_proj(rgb_feature) + self.type_embed[3])

        context_tokens = torch.cat(context, dim=1)
        if context_tokens.shape[1] > self.max_context_tokens:
            raise ValueError(
                f"Context token count {context_tokens.shape[1]} exceeds max_context_tokens={self.max_context_tokens}"
            )
        context_tokens = context_tokens + self.context_pos_embed[:, : context_tokens.shape[1]]
        return context_tokens, global_cond

    def forward(
        self,
        noisy_targets: Dict[str, torch.Tensor],
        timestep: Union[torch.Tensor, float, int],
        obs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        trajectory = noisy_targets["trajectory"]
        keyframe_object = noisy_targets["keyframe_object"]
        keyframe_tcp = noisy_targets["keyframe_tcp"]
        if trajectory.shape[1:] != (self.action_horizon, self.trajectory_dim):
            raise ValueError(
                f"trajectory must have shape [B, {self.action_horizon}, {self.trajectory_dim}], "
                f"got {trajectory.shape}"
            )
        if keyframe_object.shape[1:] != (self.keyframe_horizon, self.num_objects, self.object_dim):
            raise ValueError(
                "keyframe_object must have shape "
                f"[B, {self.keyframe_horizon}, {self.num_objects}, {self.object_dim}], "
                f"got {keyframe_object.shape}"
            )
        if keyframe_tcp.shape[1:] != (self.keyframe_horizon, self.tcp_dim):
            raise ValueError(
                f"keyframe_tcp must have shape [B, {self.keyframe_horizon}, {self.tcp_dim}], "
                f"got {keyframe_tcp.shape}"
            )

        batch_size = trajectory.shape[0]
        device = trajectory.device
        context_tokens, global_cond = self._context_tokens(obs)
        timesteps = self._make_timestep(timestep, batch_size, device)
        cond = self.time_encoder(timesteps) + global_cond

        traj_tokens = self.trajectory_proj(trajectory) + self.type_embed[0]
        object_tokens = self.object_proj(keyframe_object.reshape(batch_size, -1, self.object_dim))
        object_tokens = object_tokens + self.type_embed[1]
        tcp_tokens = self.tcp_proj(keyframe_tcp) + self.type_embed[2]
        target_tokens = torch.cat([traj_tokens, object_tokens, tcp_tokens], dim=1)
        target_tokens = target_tokens + self.target_pos_embed[:, : target_tokens.shape[1]]
        x = torch.cat([target_tokens, context_tokens], dim=1)

        for block in self.blocks:
            x = block(x, cond)

        shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)

        target_x = x[:, : target_tokens.shape[1]]
        traj_end = self.action_horizon
        object_end = traj_end + self.keyframe_horizon * self.num_objects
        traj_feat = target_x[:, :traj_end]
        object_feat = target_x[:, traj_end:object_end]
        tcp_feat = target_x[:, object_end:]

        pred_trajectory = self.trajectory_head(traj_feat)
        pred_object = self.object_head(object_feat).reshape(
            batch_size, self.keyframe_horizon, self.num_objects, self.object_dim
        )
        pred_tcp = self.tcp_head(tcp_feat)
        pred_gripper = self.gripper_head(traj_feat)

        return {
            "trajectory": pred_trajectory,
            "keyframe_object": pred_object,
            "keyframe_tcp": pred_tcp,
            "gripper_openness": pred_gripper,
            "trajectory_features": traj_feat,
        }


class DiffusionHeadMap4D(nn.Module):
    def __init__(self,
                 embedding_dim=120                                                                                                                               ,
                 num_attn_heads=8,
                 use_instruction=True,
                 rotation_parametrization='quat',
                 nhist=1,
                 lang_enhanced=False,
                 horizon_keyframe=2,
                 horizon_continuous=3
    ):
        super().__init__()
    
    def forward(self, trajectory_left, trajectory_right, timestep,
            fixed_inputs):
        # set_trace()
        (pcd_coord, pcd_feat, lang_feat, state_feat, sampled_pcd_coord, sampled_pcd_feat, pointflow_feat, pointflow_coords) = fixed_inputs
        
        # Trajectory features(noisy actions, concatenation of keypose and continuous actions)
        traj_feats_left = self.traj_encoder(trajectory_left)
        traj_feats_right = self.traj_encoder(trajectory_right)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats_left.size(1), device=traj_feats_left.device)
        )[None].repeat(len(traj_feats_left), 1, 1)
        
        if self.use_instruction:
            traj_feats_left, _ = self.traj_lang_attention_left[0](
                seq1=traj_feats_left, seq1_key_padding_mask=None,
                seq2=lang_feat, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )

            traj_feats_right, _ = self.traj_lang_attention_right[0](
                seq1=traj_feats_right, seq1_key_padding_mask=None,
                seq2=lang_feat, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
            
        traj_feats_left = traj_feats_left + traj_time_pos
        traj_feats_right = traj_feats_right + traj_time_pos

        traj_feats_left = einops.rearrange(traj_feats_left, 'b l c -> l b c')
        traj_feats_right = einops.rearrange(traj_feats_right, 'b l c -> l b c')
        pcd_feat = einops.rearrange(pcd_feat, 'b l c -> l b c')
        sampled_pcd_feat = einops.rearrange(sampled_pcd_feat, 'b l c -> l b c')
        state_feat = einops.rearrange(state_feat, 'b l c -> l b c')
        pointflow_feat = einops.rearrange(pointflow_feat, 'b l c -> l b c')
        
        pos_pred_left, rot_pred_left, openess_pred_left, pos_pred_right, rot_pred_right, openess_pred_right, position_point_flow = self.prediction_head(
            trajectory_left[..., :3], traj_feats_left,
            trajectory_right[..., :3], traj_feats_right,
            pcd_coord[..., :3], pcd_feat,
            timestep, state_feat,
            sampled_pcd_coord, sampled_pcd_feat,
            lang_feat, pointflow_feat, pointflow_coords
        )
        return ([torch.cat((pos_pred_left, rot_pred_left, openess_pred_left), -1)],[torch.cat((pos_pred_right, rot_pred_right, openess_pred_right), -1)],[position_point_flow])
