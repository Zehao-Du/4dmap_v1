"""Map4D staged denoiser for trajectory and keyframe latent planning."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from map4d.backbone.model.diffusion.conditional_dit1d import _modulate
from map4d.backbone.model.diffusion.diffuser_actor_utils.layers import (
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfAttentionModule,
)
from map4d.backbone.model.diffusion.diffuser_actor_utils.position_encodings import RotaryPositionEncoding3D
from map4d.backbone.model.diffusion.positional_embedding import SinusoidalPosEmb
from map4d.encoder.map_encoder import Map4DEncoder

MAPS4D_DIR = Path(__file__).resolve().parents[3] / "representation" / "maps4d"
if str(MAPS4D_DIR) not in sys.path:
    sys.path.insert(0, str(MAPS4D_DIR))
from maniskill_plugcharger import Map4d_PlugCharger  # noqa: E402
from maniskill_stackcube import Map4d_StackCube  # noqa: E402


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
    """PPI-style staged denoiser for Map4D policy.

    The model keeps the PPI information flow:
      unified 3D context field -> keyframe node pose
      context + node hidden -> keyframe TCP pose
      context + node/TCP hidden -> trajectory/action

    Semantic Field points and map node features are treated as one read-only 3D
    context field for cross-attention.
    """

    trajectory_dim = 7
    node_dim = 9
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
        embed_dim: int = 240,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        diffusion_step_embed_dim: int = 240,
        use_rgb: bool = False,
        max_context_tokens: int = 1024,
        semantic_feature_dim: Optional[int] = None,
        semantic_feature_mode: str = "precomputed",
        dinov3_model: str = "dinov3_vits16",
        dinov3_weights_path: Optional[str] = None,
        dinov3_third_party_dir: Optional[str] = None,
        dinov3_image_size: int = 224,
        dinov3_input_multiple: int = 16,
        dinov3_amp: bool = True,
        point_dim: int = 3,
        map_feature_dim: int = 240,
        n_map_step: Optional[int] = None,
        num_arms: int = 1,
        num_map_nodes: Optional[int] = None,
        use_map_encoder: bool = False,
        map_name: Optional[str] = None,
        map_encoder_hidden_dim: Optional[int] = None,
        map_encoder_num_layers: int = 4,
        map_encoder_num_heads: int = 8,
        detach_stage_features: bool = True,
        cross_attn_layers: int = 2,
        self_attn_layers: Optional[int] = None,
        use_map_node_pos_embed: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if embed_dim % 6 != 0:
            raise ValueError("embed_dim must be divisible by 6 for PPI-style 3D rotary position encoding")

        self.robot_state_dim = int(robot_state_dim)
        self.num_objects = int(num_objects)
        self.num_target_nodes = int(num_objects)
        self.action_horizon = int(action_horizon)
        self.keyframe_horizon = int(keyframe_horizon)
        self.obs_horizon = int(obs_horizon)
        self.map4d_dim = int(map4d_dim)
        self.size_parameter_dim = int(size_parameter_dim)
        self.relation_parameter_dim = int(relation_parameter_dim)
        self.tcp_dim = int(tcp_dim)
        self.embed_dim = int(embed_dim)
        self.max_context_tokens = int(max_context_tokens)
        self.point_dim = int(point_dim)
        self.semantic_feature_dim = int(semantic_feature_dim or rgb_feature_dim or 384)
        self.semantic_feature_mode = str(semantic_feature_mode)
        if self.semantic_feature_mode not in {"precomputed", "online_dinov3"}:
            raise ValueError(
                "semantic_feature_mode must be 'precomputed' or 'online_dinov3', "
                f"got {self.semantic_feature_mode!r}"
            )
        self.dinov3_model = str(dinov3_model)
        self.dinov3_weights_path = "" if dinov3_weights_path is None else str(dinov3_weights_path)
        self.dinov3_third_party_dir = "" if dinov3_third_party_dir is None else str(dinov3_third_party_dir)
        self.dinov3_image_size = None if dinov3_image_size is None else int(dinov3_image_size)
        self.dinov3_input_multiple = int(dinov3_input_multiple)
        self.dinov3_amp = bool(dinov3_amp)
        object.__setattr__(self, "_dinov3_backbone", None)
        self.map_feature_dim = int(map_feature_dim)
        self.n_map_step = int(n_map_step or obs_horizon)
        self.num_arms = int(num_arms)
        self.num_map_nodes = None if num_map_nodes is None else int(num_map_nodes)
        self.detach_stage_features = bool(detach_stage_features)
        self.use_map_encoder = bool(use_map_encoder)
        self.map_name = map_name
        if self.tcp_dim not in {4, 7}:
            raise ValueError(f"tcp_dim must be 4 or 7, got {self.tcp_dim}")
        if self.point_dim != 3:
            raise ValueError("Map4DDiT currently requires point_dim=3")
        if self.num_arms <= 0:
            raise ValueError("num_arms must be positive")
        if self.n_map_step <= 0:
            raise ValueError("n_map_step must be positive")
        if self.semantic_feature_mode == "online_dinov3":
            if not self.dinov3_weights_path:
                raise ValueError("online_dinov3 requires dinov3_weights_path")
            if not Path(self.dinov3_weights_path).expanduser().exists():
                raise FileNotFoundError(f"DINOv3 weights not found: {self.dinov3_weights_path}")
            if not self.dinov3_third_party_dir:
                raise ValueError("online_dinov3 requires dinov3_third_party_dir")
            if not Path(self.dinov3_third_party_dir).expanduser().exists():
                raise FileNotFoundError(f"DINOv3 third_party dir not found: {self.dinov3_third_party_dir}")
            if self.dinov3_image_size is not None and self.dinov3_image_size <= 0:
                raise ValueError("dinov3_image_size must be positive or null")
            if self.dinov3_input_multiple <= 0:
                raise ValueError("dinov3_input_multiple must be positive")
            self.register_buffer(
                "_dinov3_mean",
                torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_dinov3_std",
                torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )

        self.trajectory_proj = nn.Linear(self.trajectory_dim, embed_dim)
        self.node_proj = nn.Linear(self.node_dim, embed_dim)
        self.object_proj = self.node_proj
        self.tcp_proj = nn.Linear(self.tcp_dim, embed_dim)
        self.semantic_field_proj = nn.Sequential(
            nn.Linear(self.point_dim + self.semantic_feature_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.map_feature_proj = nn.Sequential(
            nn.Linear(self.map_feature_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.map_encoder = None
        if self.use_map_encoder:
            self.map_encoder = Map4DEncoder(
                map_name=map_name,
                num_nodes=self.num_map_nodes,
                hidden_dim=int(map_encoder_hidden_dim or self.map_feature_dim),
                num_layers=int(map_encoder_num_layers),
                num_heads=int(map_encoder_num_heads),
            )
            if int(map_encoder_hidden_dim or self.map_feature_dim) != self.map_feature_dim:
                raise ValueError("map_encoder_hidden_dim must equal map_feature_dim")

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

        self.relative_pe_layer = RotaryPositionEncoding3D(embed_dim)

        self.trajectory_pos_embed = nn.Parameter(torch.zeros(1, 1, action_horizon, embed_dim))
        self.keyframe_time_embed = nn.Parameter(torch.zeros(1, keyframe_horizon, 1, embed_dim))
        self.node_id_embed = nn.Parameter(torch.zeros(1, 1, self.num_target_nodes, embed_dim))
        self.tcp_pos_embed = nn.Parameter(torch.zeros(1, 1, keyframe_horizon, embed_dim))
        self.arm_embed = nn.Parameter(torch.zeros(1, self.num_arms, 1, embed_dim))
        self.target_type_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.source_embed = nn.Parameter(torch.zeros(2, embed_dim))
        self.map_time_pos_embed = nn.Parameter(torch.zeros(1, self.n_map_step, 1, embed_dim))
        self.map_node_pos_embed = (
            nn.Parameter(torch.zeros(1, 1, self.num_map_nodes, embed_dim))
            if use_map_node_pos_embed and self.num_map_nodes is not None
            else None
        )
        self.map4d_proj = None
        self.size_proj = None
        self.relation_proj = None
        self.rgb_proj = None

        self_attn_layers = int(self_attn_layers or depth)
        self.node_context_attn = FFWRelativeCrossAttentionModule(
            embed_dim, num_heads, num_layers=int(cross_attn_layers), use_adaln=True
        )
        self.node_self_attn = FFWRelativeSelfAttentionModule(
            embed_dim, num_heads, num_layers=self_attn_layers, use_adaln=True
        )
        self.tcp_context_attn = FFWRelativeCrossAttentionModule(
            embed_dim, num_heads, num_layers=int(cross_attn_layers), use_adaln=True
        )
        self.tcp_self_attn = FFWRelativeSelfAttentionModule(
            embed_dim, num_heads, num_layers=self_attn_layers, use_adaln=True
        )
        self.action_context_attn = FFWRelativeCrossAttentionModule(
            embed_dim, num_heads, num_layers=int(cross_attn_layers), use_adaln=True
        )
        self.action_self_attn = FFWRelativeSelfAttentionModule(
            embed_dim, num_heads, num_layers=self_attn_layers, use_adaln=True
        )

        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.trajectory_head = nn.Linear(embed_dim, self.trajectory_dim)
        self.node_head = nn.Linear(embed_dim, self.node_dim)
        self.object_head = self.node_head
        self.tcp_head = nn.Linear(embed_dim, self.tcp_dim)
        self.gripper_head = nn.Linear(embed_dim, 1)

        for param in (
            self.trajectory_pos_embed,
            self.keyframe_time_embed,
            self.node_id_embed,
            self.tcp_pos_embed,
            self.arm_embed,
            self.target_type_embed,
            self.source_embed,
            self.map_time_pos_embed,
        ):
            nn.init.normal_(param, std=0.02)
        if self.map_node_pos_embed is not None:
            nn.init.normal_(self.map_node_pos_embed, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

    def _load_online_dinov3_backbone(self, device: torch.device):
        backbone = object.__getattribute__(self, "_dinov3_backbone")
        if backbone is None:
            project_root = Path(__file__).resolve().parents[4]
            dp_root = project_root / "baselines" / "diffusion_policy"
            if str(dp_root) not in sys.path:
                sys.path.insert(0, str(dp_root))
            from diffusion_policy.dinov3_encoder import _load_backbone  # noqa: PLC0415

            backbone = _load_backbone(
                self.dinov3_third_party_dir,
                self.dinov3_weights_path,
                self.dinov3_model,
            )
            backbone.eval()
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
            embed_dim = int(getattr(backbone, "embed_dim", 0))
            if embed_dim != self.semantic_feature_dim:
                raise ValueError(
                    f"DINOv3 embed_dim={embed_dim} does not match "
                    f"semantic_feature_dim={self.semantic_feature_dim}"
                )
            object.__setattr__(self, "_dinov3_backbone", backbone)
        backbone.to(device)
        backbone.eval()
        return backbone

    @staticmethod
    def _dinov3_patch_size(backbone) -> Tuple[int, int]:
        patch_size = getattr(backbone, "patch_size", None)
        if patch_size is None and hasattr(backbone, "patch_embed"):
            patch_size = getattr(backbone.patch_embed, "patch_size", None)
        if patch_size is None:
            raise AttributeError("DINOv3 backbone does not expose patch_size or patch_embed.patch_size")
        if isinstance(patch_size, int):
            return int(patch_size), int(patch_size)
        if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            return int(patch_size[0]), int(patch_size[1])
        raise TypeError(f"Unsupported DINOv3 patch_size={patch_size!r}")

    def _online_dinov3_point_features(
        self,
        obs: Dict[str, torch.Tensor],
        point_cloud: torch.Tensor,
    ) -> torch.Tensor:
        rgb = obs.get("rgb")
        camera_index = obs.get("point_camera_index")
        pixel_uv = obs.get("point_pixel_uv")
        if rgb is None or camera_index is None or pixel_uv is None:
            raise ValueError(
                "online_dinov3 requires obs.rgb, obs.point_camera_index, and obs.point_pixel_uv"
            )
        if rgb.ndim != 6 or rgb.shape[-1] != 3:
            raise ValueError(f"obs.rgb must have shape [B,T,C,H,W,3], got {rgb.shape}")
        if camera_index.ndim != 3:
            raise ValueError(f"obs.point_camera_index must have shape [B,T,P], got {camera_index.shape}")
        if pixel_uv.ndim != 4 or pixel_uv.shape[-1] != 2:
            raise ValueError(f"obs.point_pixel_uv must have shape [B,T,P,2], got {pixel_uv.shape}")
        batch_size, steps, point_count = point_cloud.shape[:3]
        if rgb.shape[:2] != (batch_size, steps):
            raise ValueError(f"obs.rgb [B,T] {rgb.shape[:2]} must match point_cloud {(batch_size, steps)}")
        if camera_index.shape != (batch_size, steps, point_count):
            raise ValueError(
                f"obs.point_camera_index shape {camera_index.shape} must match [B,T,P] "
                f"{(batch_size, steps, point_count)}"
            )
        if pixel_uv.shape[:3] != (batch_size, steps, point_count):
            raise ValueError(
                f"obs.point_pixel_uv shape {pixel_uv.shape} must match [B,T,P,2] "
                f"{(batch_size, steps, point_count, 2)}"
            )

        device = point_cloud.device
        dtype = point_cloud.dtype
        backbone = self._load_online_dinov3_backbone(device)
        camera_count = int(rgb.shape[2])
        image_h, image_w = int(rgb.shape[3]), int(rgb.shape[4])
        target_h, target_w = (
            (self.dinov3_image_size, self.dinov3_image_size)
            if self.dinov3_image_size is not None
            else (image_h, image_w)
        )
        if self.dinov3_image_size is None and self.dinov3_input_multiple > 1:
            target_h = ((target_h + self.dinov3_input_multiple - 1) // self.dinov3_input_multiple) * self.dinov3_input_multiple
            target_w = ((target_w + self.dinov3_input_multiple - 1) // self.dinov3_input_multiple) * self.dinov3_input_multiple
        patch_h, patch_w = self._dinov3_patch_size(backbone)
        if target_h % patch_h != 0 or target_w % patch_w != 0:
            raise ValueError(
                f"DINO input size {(target_h, target_w)} must be divisible by patch size {(patch_h, patch_w)}"
            )
        grid_h, grid_w = target_h // patch_h, target_w // patch_w

        x = rgb.reshape(batch_size * steps * camera_count, image_h, image_w, 3).float().div_(255.0)
        x = x.permute(0, 3, 1, 2).contiguous()
        if (target_h, target_w) != (image_h, image_w):
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        mean = self._dinov3_mean.to(device=device, dtype=x.dtype)
        std = self._dinov3_std.to(device=device, dtype=x.dtype)
        x = (x - mean) / std

        use_amp = self.dinov3_amp and device.type == "cuda"
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            feats = backbone.forward_features(x)
            patch_tokens = feats["x_norm_patchtokens"]
        if patch_tokens.ndim != 3:
            raise ValueError(f"DINO patch tokens must be [B,N,D], got {patch_tokens.shape}")
        if patch_tokens.shape[1] != grid_h * grid_w:
            raise ValueError(
                f"DINO patch token count {patch_tokens.shape[1]} does not match grid {grid_h}x{grid_w}"
            )
        if patch_tokens.shape[-1] != self.semantic_feature_dim:
            raise ValueError(
                f"DINO feature dim {patch_tokens.shape[-1]} does not match "
                f"semantic_feature_dim={self.semantic_feature_dim}"
            )

        camera_index = camera_index.to(device=device).long()
        pixel_uv = pixel_uv.to(device=device).long()
        if torch.any(camera_index < 0) or torch.any(camera_index >= camera_count):
            raise ValueError(f"point_camera_index must be in [0,{camera_count}), got invalid values")
        u = pixel_uv[..., 0]
        v = pixel_uv[..., 1]
        if torch.any(u < 0) or torch.any(u >= image_w) or torch.any(v < 0) or torch.any(v >= image_h):
            raise ValueError(f"point_pixel_uv contains values outside image size {(image_h, image_w)}")

        patch_x = torch.floor((u.float() + 0.5) * grid_w / image_w).long().clamp_(0, grid_w - 1)
        patch_y = torch.floor((v.float() + 0.5) * grid_h / image_h).long().clamp_(0, grid_h - 1)
        patch_index = patch_y * grid_w + patch_x
        patch_tokens = patch_tokens.reshape(
            batch_size * steps,
            camera_count,
            grid_h * grid_w,
            self.semantic_feature_dim,
        )
        flat_camera = camera_index.reshape(batch_size * steps, point_count)
        flat_patch = patch_index.reshape(batch_size * steps, point_count)
        flat_bt = torch.arange(batch_size * steps, device=device)[:, None].expand(-1, point_count)
        feature = patch_tokens[flat_bt, flat_camera, flat_patch]
        return feature.reshape(batch_size, steps, point_count, self.semantic_feature_dim).to(dtype=dtype)

    def _make_timestep(self, timestep: Union[torch.Tensor, float, int], batch_size: int, device):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device=device)
        else:
            timestep = timestep.to(device=device)
        return timestep.expand(batch_size).long()

    def _global_condition(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        robot_state = obs["robot_state"]
        if robot_state.ndim != 3:
            raise ValueError(f"obs.robot_state must have shape [B, T, D], got {robot_state.shape}")
        robot_state = robot_state[:, -self.obs_horizon :]
        if robot_state.shape[1] < self.obs_horizon:
            pad = robot_state[:, :1].expand(-1, self.obs_horizon - robot_state.shape[1], -1)
            robot_state = torch.cat([pad, robot_state], dim=1)
        return self.robot_encoder(robot_state.reshape(robot_state.shape[0], -1))

    def _semantic_context(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        point_cloud = obs.get("point_cloud")
        if point_cloud is None:
            raise ValueError("Map4DDiT requires obs.point_cloud")
        if point_cloud.ndim == 3:
            point_cloud = point_cloud[:, None]
        if point_cloud.ndim != 4:
            raise ValueError(f"obs.point_cloud must have shape [B, T, P, C], got {point_cloud.shape}")

        if self.semantic_feature_mode == "precomputed":
            dino_feature = obs.get("dino_feature")
            if dino_feature is None:
                raise ValueError("precomputed semantic_feature_mode requires obs.dino_feature")
        elif self.semantic_feature_mode == "online_dinov3":
            if "dino_feature" in obs:
                raise ValueError("online_dinov3 mode expects raw RGB/source indices, not obs.dino_feature")
            dino_feature = self._online_dinov3_point_features(obs, point_cloud)
        else:
            raise ValueError(f"Unsupported semantic_feature_mode={self.semantic_feature_mode!r}")

        if dino_feature.ndim == 3:
            dino_feature = dino_feature[:, None]
        if dino_feature.ndim != 4:
            raise ValueError(f"obs.dino_feature must have shape [B, T, P, D], got {dino_feature.shape}")
        if point_cloud.shape[:3] != dino_feature.shape[:3]:
            raise ValueError(
                f"point_cloud and dino_feature must share [B,T,P], got {point_cloud.shape} and {dino_feature.shape}"
            )
        if point_cloud.shape[-1] < self.point_dim:
            raise ValueError(f"point_cloud last dim must be >= {self.point_dim}, got {point_cloud.shape[-1]}")
        if dino_feature.shape[-1] != self.semantic_feature_dim:
            raise ValueError(
                f"Expected semantic_feature_dim={self.semantic_feature_dim}, got {dino_feature.shape[-1]}"
            )

        xyz = point_cloud[..., : self.point_dim].reshape(point_cloud.shape[0], -1, self.point_dim)
        feature = dino_feature.reshape(dino_feature.shape[0], -1, self.semantic_feature_dim)
        token = self.semantic_field_proj(torch.cat([xyz, feature], dim=-1))
        token = token + self.source_embed[0]
        return xyz, token

    def _expand_static_parameter(
        self,
        value: torch.Tensor,
        *,
        batch_size: int,
        steps: int,
        dim: int,
        name: str,
    ) -> torch.Tensor:
        if dim == 0:
            return value.new_zeros(batch_size * steps, 0)
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(batch_size, -1)
        if value.ndim == 2:
            if value.shape != (batch_size, dim):
                raise ValueError(f"obs.{name} must have shape [B,{dim}] or [{dim}], got {value.shape}")
            value = value[:, None].expand(-1, steps, -1)
        elif value.ndim == 3:
            if value.shape != (batch_size, steps, dim):
                raise ValueError(f"obs.{name} must have shape [B,T,{dim}], got {value.shape}")
        else:
            raise ValueError(f"obs.{name} must have 1, 2, or 3 dims, got {value.shape}")
        return value.reshape(batch_size * steps, dim)

    def _map_representation_from_gt(self, obs: Dict[str, torch.Tensor]):
        if self.map_encoder is None:
            raise ValueError("Online GT map encoding requires Map4DDiT(use_map_encoder=True)")
        if self.map_name is None:
            raise ValueError("Online GT map encoding requires model_cfg.map_name")

        node_poses = obs.get("node_poses")
        if node_poses is None:
            raise ValueError("Map4DDiT requires obs.node_poses for online map encoding")
        if node_poses.ndim == 3:
            node_poses = node_poses[:, None]
        if node_poses.ndim != 4:
            raise ValueError(f"obs.node_poses must have shape [B,T,N,9], got {node_poses.shape}")
        if node_poses.shape[1] != self.n_map_step:
            raise ValueError(f"Expected n_map_step={self.n_map_step}, got obs.node_poses T={node_poses.shape[1]}")
        if node_poses.shape[2] != self.num_target_nodes:
            raise ValueError(
                f"Expected {self.num_target_nodes} target nodes in obs.node_poses, got {node_poses.shape[2]}"
            )
        if node_poses.shape[-1] != 9:
            raise ValueError(f"Online map encoder expects node pose dim 9, got {node_poses.shape[-1]}")

        batch_size, steps = node_poses.shape[:2]
        flat = node_poses.reshape(batch_size * steps, self.num_target_nodes, 9)
        positions = flat[..., 0:3].reshape(batch_size * steps, -1)
        rotations = flat[..., 3:9].reshape(batch_size * steps, -1)

        size_parameters = obs.get("size_parameters")
        if size_parameters is None:
            raise ValueError("Map4DDiT requires obs.size_parameters for online map encoding")
        relation_parameters = obs.get("relation_parameters")
        if relation_parameters is None:
            raise ValueError("Map4DDiT requires obs.relation_parameters for online map encoding")
        sizes = self._expand_static_parameter(
            size_parameters.to(device=node_poses.device, dtype=node_poses.dtype),
            batch_size=batch_size,
            steps=steps,
            dim=self.size_parameter_dim,
            name="size_parameters",
        )
        relations = self._expand_static_parameter(
            relation_parameters.to(device=node_poses.device, dtype=node_poses.dtype),
            batch_size=batch_size,
            steps=steps,
            dim=self.relation_parameter_dim,
            name="relation_parameters",
        )

        if self.map_name == "StackCube-v1":
            return Map4d_StackCube(sizes, positions, rotations, clip_model=None)
        if self.map_name == "PlugCharger-v1":
            return Map4d_PlugCharger(positions, rotations, sizes, relations, clip_model=None)
        raise ValueError(f"Unsupported map_name={self.map_name!r}")

    def _encoded_map_nodes_from_gt(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_poses = obs["node_poses"]
        if node_poses.ndim == 3:
            batch_size, steps = node_poses.shape[0], 1
        else:
            batch_size, steps = node_poses.shape[:2]
        rep = self._map_representation_from_gt(obs)
        encoded = self.map_encoder(rep)
        if encoded.ndim != 3:
            raise ValueError(f"Map4DEncoder output must be [B*T,N,3+D], got {encoded.shape}")
        return encoded.reshape(batch_size, steps, encoded.shape[1], encoded.shape[2])

    def _map_context(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        stale_keys = {"map_node_feature", "map_feature", "map_graph", "map_graph_seq"} & set(obs.keys())
        if stale_keys:
            raise ValueError(
                "Map4DDiT no longer accepts precomputed map features or map graphs in obs; "
                f"remove {sorted(stale_keys)} and pass GT obs.node_poses/size_parameters/relation_parameters."
            )

        encoded_map_nodes = self._encoded_map_nodes_from_gt(obs)
        if not torch.is_tensor(encoded_map_nodes):
            raise TypeError("Map4DEncoder output must be a torch.Tensor")
        if encoded_map_nodes.ndim == 3:
            encoded_map_nodes = encoded_map_nodes[:, None]
        if encoded_map_nodes.ndim != 4:
            raise ValueError(
                f"Map4DEncoder output must have shape [B, T, N, 3 + D_map], got {encoded_map_nodes.shape}"
            )
        if encoded_map_nodes.shape[1] != self.n_map_step:
            raise ValueError(f"Expected n_map_step={self.n_map_step}, got {encoded_map_nodes.shape[1]}")
        if self.num_map_nodes is not None and encoded_map_nodes.shape[2] != self.num_map_nodes:
            raise ValueError(f"Expected num_map_nodes={self.num_map_nodes}, got {encoded_map_nodes.shape[2]}")
        if encoded_map_nodes.shape[-1] != 3 + self.map_feature_dim:
            raise ValueError(
                f"Expected map feature dim {3 + self.map_feature_dim}, got {encoded_map_nodes.shape[-1]}"
            )

        map_xyz = encoded_map_nodes[..., :3]
        map_feature = encoded_map_nodes[..., 3:]
        map_token = self.map_feature_proj(map_feature)
        map_token = map_token + self.map_time_pos_embed[:, : map_token.shape[1]]
        if self.map_node_pos_embed is not None:
            map_token = map_token + self.map_node_pos_embed[:, :, : map_token.shape[2]]
        map_token = map_token + self.source_embed[1]
        return map_xyz.reshape(map_xyz.shape[0], -1, 3), map_token.reshape(map_token.shape[0], -1, self.embed_dim)

    def _context_field(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_cond = self._global_condition(obs)
        semantic_xyz, semantic_token = self._semantic_context(obs)
        map_xyz, map_token = self._map_context(obs)
        context_xyz = torch.cat([semantic_xyz, map_xyz], dim=1)
        context_token = torch.cat([semantic_token, map_token], dim=1)
        if context_token.shape[1] > self.max_context_tokens:
            raise ValueError(
                f"Context token count {context_token.shape[1]} exceeds max_context_tokens={self.max_context_tokens}"
            )
        return context_xyz, context_token, global_cond

    def _finalize(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
        return _modulate(self.final_norm(x), shift, scale)

    @staticmethod
    def _to_seq(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1)

    @staticmethod
    def _to_batch(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1)

    def _prepare_targets(self, noisy_targets: Dict[str, torch.Tensor]):
        trajectory = noisy_targets["trajectory"]
        keyframe_node = noisy_targets.get(
            "keyframe_node",
            noisy_targets.get("keyframe_map4d", noisy_targets.get("keyframe_object")),
        )
        keyframe_tcp = noisy_targets["keyframe_tcp"]
        if keyframe_node is None:
            raise ValueError("noisy_targets must include keyframe_node or keyframe_map4d")

        squeeze_trajectory_arm = False
        squeeze_tcp_arm = False
        if trajectory.ndim == 3:
            trajectory = trajectory[:, None]
            squeeze_trajectory_arm = True
        if keyframe_tcp.ndim == 3:
            keyframe_tcp = keyframe_tcp[:, None]
            squeeze_tcp_arm = True

        if trajectory.shape[1:] != (self.num_arms, self.action_horizon, self.trajectory_dim):
            raise ValueError(
                "trajectory must have shape "
                f"[B, {self.num_arms}, {self.action_horizon}, {self.trajectory_dim}], got {trajectory.shape}"
            )
        if keyframe_node.shape[1:] != (self.keyframe_horizon, self.num_target_nodes, self.node_dim):
            raise ValueError(
                "keyframe_node/keyframe_map4d must have shape "
                f"[B, {self.keyframe_horizon}, {self.num_target_nodes}, {self.node_dim}], "
                f"got {keyframe_node.shape}"
            )
        if keyframe_tcp.shape[1:] != (self.num_arms, self.keyframe_horizon, self.tcp_dim):
            raise ValueError(
                "keyframe_tcp must have shape "
                f"[B, {self.num_arms}, {self.keyframe_horizon}, {self.tcp_dim}], got {keyframe_tcp.shape}"
            )
        return trajectory, keyframe_node, keyframe_tcp, squeeze_trajectory_arm, squeeze_tcp_arm

    def _target_tokens(self, trajectory, keyframe_node, keyframe_tcp):
        batch_size = trajectory.shape[0]

        traj_tokens = self.trajectory_proj(trajectory)
        traj_tokens = traj_tokens + self.trajectory_pos_embed + self.arm_embed[:, : self.num_arms]
        traj_tokens = traj_tokens + self.target_type_embed[0]
        traj_xyz = trajectory[..., :3].reshape(batch_size, -1, 3)
        traj_tokens = traj_tokens.reshape(batch_size, -1, self.embed_dim)

        node_tokens = self.node_proj(keyframe_node)
        node_tokens = node_tokens + self.keyframe_time_embed + self.node_id_embed
        node_tokens = node_tokens + self.target_type_embed[1]
        node_xyz = keyframe_node[..., :3].reshape(batch_size, -1, 3)
        node_tokens = node_tokens.reshape(batch_size, -1, self.embed_dim)

        tcp_tokens = self.tcp_proj(keyframe_tcp)
        tcp_tokens = tcp_tokens + self.tcp_pos_embed + self.arm_embed[:, : self.num_arms]
        tcp_tokens = tcp_tokens + self.target_type_embed[2]
        tcp_xyz = keyframe_tcp[..., :3].reshape(batch_size, -1, 3)
        tcp_tokens = tcp_tokens.reshape(batch_size, -1, self.embed_dim)

        return traj_tokens, traj_xyz, node_tokens, node_xyz, tcp_tokens, tcp_xyz

    def _cross_context(self, module, query, query_xyz, context_token, context_xyz, cond):
        query_seq = self._to_seq(query)
        context_seq = self._to_seq(context_token)
        return self._to_batch(
            module(
                query=query_seq,
                value=context_seq,
                query_pos=self.relative_pe_layer(query_xyz),
                value_pos=self.relative_pe_layer(context_xyz),
                diff_ts=cond,
            )[-1]
        )

    def _self_attention(self, module, query, query_xyz, cond):
        query_seq = self._to_seq(query)
        return self._to_batch(
            module(query=query_seq, query_pos=self.relative_pe_layer(query_xyz), diff_ts=cond)[-1]
        )

    def forward(
        self,
        noisy_targets: Dict[str, torch.Tensor],
        timestep: Union[torch.Tensor, float, int],
        obs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        trajectory, keyframe_node, keyframe_tcp, squeeze_trajectory_arm, squeeze_tcp_arm = self._prepare_targets(
            noisy_targets
        )
        batch_size = trajectory.shape[0]
        device = trajectory.device
        context_xyz, context_token, global_cond = self._context_field(obs)
        timesteps = self._make_timestep(timestep, batch_size, device)
        cond = self.time_encoder(timesteps) + global_cond

        traj_tokens, traj_xyz, node_tokens, node_xyz, tcp_tokens, tcp_xyz = self._target_tokens(
            trajectory, keyframe_node, keyframe_tcp
        )

        node_feat = self._cross_context(
            self.node_context_attn, node_tokens, node_xyz, context_token, context_xyz, cond
        )
        node_feat = self._self_attention(self.node_self_attn, node_feat, node_xyz, cond)

        node_condition = node_feat.detach() if self.detach_stage_features else node_feat
        tcp_feat = self._cross_context(self.tcp_context_attn, tcp_tokens, tcp_xyz, context_token, context_xyz, cond)
        tcp_with_node = torch.cat([tcp_feat, node_condition], dim=1)
        tcp_with_node_xyz = torch.cat([tcp_xyz, node_xyz], dim=1)
        tcp_with_node = self._self_attention(self.tcp_self_attn, tcp_with_node, tcp_with_node_xyz, cond)
        tcp_feat = tcp_with_node[:, : tcp_tokens.shape[1]]

        tcp_condition = tcp_feat.detach() if self.detach_stage_features else tcp_feat
        action_feat = self._cross_context(
            self.action_context_attn, traj_tokens, traj_xyz, context_token, context_xyz, cond
        )
        action_with_plan = torch.cat([action_feat, node_condition, tcp_condition], dim=1)
        action_with_plan_xyz = torch.cat([traj_xyz, node_xyz, tcp_xyz], dim=1)
        action_with_plan = self._self_attention(self.action_self_attn, action_with_plan, action_with_plan_xyz, cond)
        traj_feat = action_with_plan[:, : traj_tokens.shape[1]]

        node_feat = self._finalize(node_feat, cond)
        tcp_feat = self._finalize(tcp_feat, cond)
        traj_feat = self._finalize(traj_feat, cond)

        pred_node = self.node_head(node_feat).reshape(
            batch_size, self.keyframe_horizon, self.num_target_nodes, self.node_dim
        )
        pred_tcp = self.tcp_head(tcp_feat).reshape(batch_size, self.num_arms, self.keyframe_horizon, self.tcp_dim)
        pred_trajectory = self.trajectory_head(traj_feat).reshape(
            batch_size, self.num_arms, self.action_horizon, self.trajectory_dim
        )
        pred_gripper = self.gripper_head(traj_feat).reshape(batch_size, self.num_arms, self.action_horizon, 1)

        if squeeze_trajectory_arm:
            pred_trajectory = pred_trajectory[:, 0]
            pred_gripper = pred_gripper[:, 0]
            traj_feat_out = traj_feat.reshape(batch_size, self.num_arms, self.action_horizon, self.embed_dim)[:, 0]
        else:
            traj_feat_out = traj_feat.reshape(batch_size, self.num_arms, self.action_horizon, self.embed_dim)
        if squeeze_tcp_arm:
            pred_tcp = pred_tcp[:, 0]

        return {
            "trajectory": pred_trajectory,
            "keyframe_node": pred_node,
            "keyframe_map4d": pred_node,
            "keyframe_object": pred_node,
            "keyframe_tcp": pred_tcp,
            "gripper_openness": pred_gripper,
            "trajectory_features": traj_feat_out,
            "keyframe_node_features": node_feat,
            "keyframe_tcp_features": tcp_feat,
        }
