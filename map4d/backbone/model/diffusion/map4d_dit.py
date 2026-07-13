"""Map4D staged denoiser for trajectory and keyframe latent planning."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from map4d.backbone.model.diffusion.diffuser_actor_utils.layers import (
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfAttentionModule,
)
from map4d.backbone.model.diffusion.diffuser_actor_utils.position_encodings import RotaryPositionEncoding3D
from map4d.backbone.model.diffusion.positional_embedding import SinusoidalPosEmb
from map4d.backbone.model.vision.observation_encoder import ObservationEncoder
from map4d.encoder.map_encoder import Map4DEncoder

MAPS4D_DIR = Path(__file__).resolve().parents[3] / "representation" / "maps4d"
if str(MAPS4D_DIR) not in sys.path:
    sys.path.insert(0, str(MAPS4D_DIR))
from maniskill_plugcharger import Map4d_PlugCharger  # noqa: E402
from maniskill_stackcube import Map4d_StackCube  # noqa: E402
from rlbench2_push_box import Map4d_RLBench2PushBox  # noqa: E402


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


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert canonical WXYZ quaternions to rotation matrices."""
    quat = normalize_quaternion(quat)
    w, x, y, z = quat.unbind(dim=-1)
    matrix = torch.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype, device=quat.device)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    return matrix_to_rot6d(quat_to_matrix(quat))


class Map4DDiT(nn.Module):
    """PPI-style staged denoiser for Map4D policy.

    The model keeps the PPI information flow:
      unified 3D context field -> keyframe node pose
      context + node hidden -> keyframe TCP pose
      context + node/TCP hidden -> trajectory/action

    Semantic Field points and map node features are treated as one read-only 3D
    context field for cross-attention.
    """

    node_dim = 7
    object_dim = 7

    def __init__(
        self,
        *,
        robot_state_dim: int,
        num_objects: int,
        action_horizon: int,
        keyframe_horizon: int,
        obs_horizon: int = 2,
        map4d_dim: int = 7,
        size_parameter_dim: int = 0,
        rgb_feature_dim: int = 0,
        relation_parameter_dim: int = 0,
        trajectory_dim: int = 7,
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
        pointcloud_encoder_cfg: Optional[Dict] = None,
        observation_encoder_cfg: Optional[Dict] = None,
        use_lang: bool = False,
        detach_stage_features: bool = True,
        cross_attn_layers: int = 2,
        self_attn_layers: Optional[int] = None,
        use_map_node_pos_embed: bool = True,
        separate_map_cross_attn: bool = False,
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
        self.node_dim = self.map4d_dim
        self.object_dim = self.map4d_dim
        self.size_parameter_dim = int(size_parameter_dim)
        self.relation_parameter_dim = int(relation_parameter_dim)
        self.trajectory_dim = int(trajectory_dim)
        self.tcp_dim = int(tcp_dim)
        if self.trajectory_dim not in {4, 7}:
            raise ValueError(f"Map4DDiT supports trajectory_dim 4 or 7, got {self.trajectory_dim}")
        self.embed_dim = int(embed_dim)
        self.max_context_tokens = int(max_context_tokens)
        self.point_dim = int(point_dim)
        self.semantic_feature_dim = int(semantic_feature_dim or rgb_feature_dim or 384)
        self.semantic_feature_mode = str(semantic_feature_mode)
        if self.semantic_feature_mode != "precomputed":
            raise ValueError(
                "Map4DDiT expects precomputed obs.dino_feature from ObservationEncoder input, "
                f"got {self.semantic_feature_mode!r}"
            )
        self.map_feature_dim = int(map_feature_dim)
        self.n_map_step = int(n_map_step or obs_horizon)
        self.num_arms = int(num_arms)
        self.num_map_nodes = None if num_map_nodes is None else int(num_map_nodes)
        self.detach_stage_features = bool(detach_stage_features)
        self.use_map_encoder = bool(use_map_encoder)
        self.map_name = map_name
        self.separate_map_cross_attn = bool(separate_map_cross_attn)
        if self.tcp_dim not in {4, 7}:
            raise ValueError(f"tcp_dim must be 4 or 7, got {self.tcp_dim}")
        if self.point_dim != 3:
            raise ValueError("Map4DDiT currently requires point_dim=3")
        if self.num_arms <= 0:
            raise ValueError("num_arms must be positive")
        if self.n_map_step <= 0:
            raise ValueError("n_map_step must be positive")
        self.trajectory_proj = nn.Linear(self.trajectory_dim, embed_dim)
        self.node_proj = nn.Linear(self.point_dim, embed_dim)
        self.object_proj = self.node_proj
        self.tcp_proj = nn.Linear(self.tcp_dim, embed_dim)
        if pointcloud_encoder_cfg is None:
            pointcloud_encoder_cfg = {
                "in_channels": self.point_dim + self.semantic_feature_dim,
                "out_channels": embed_dim,
                "use_bn": True,
                "npoint1": 1024,
                "npoint2": 512,
            }
        observation_encoder_cfg = dict(observation_encoder_cfg or {})
        self.obs_encoder = ObservationEncoder(
            state_shape=self.robot_state_dim,
            out_channel=embed_dim,
            dim_dino_feature=self.semantic_feature_dim,
            state_mlp_size=tuple(observation_encoder_cfg.pop("state_mlp_size", (embed_dim, embed_dim))),
            lang_mlp_size=tuple(observation_encoder_cfg.pop("lang_mlp_size", (embed_dim, embed_dim))),
            pcd_mlp_size=tuple(observation_encoder_cfg.pop("pcd_mlp_size", (embed_dim, embed_dim))),
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_lang=use_lang,
            state_key="robot_state",
            point_cloud_key="point_cloud",
            lang_key="lang",
            **observation_encoder_cfg,
        )
        self.map_feature_proj = nn.Sequential(
            nn.LayerNorm(self.map_feature_dim),
            nn.Linear(self.map_feature_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
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

        self.robot_encoder = None
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
        self.node_map_attn = None
        self.tcp_map_attn = None
        self.action_map_attn = None
        if self.separate_map_cross_attn:
            self.node_map_attn = FFWRelativeCrossAttentionModule(
                embed_dim, num_heads, num_layers=int(cross_attn_layers), use_adaln=True
            )
            self.tcp_map_attn = FFWRelativeCrossAttentionModule(
                embed_dim, num_heads, num_layers=int(cross_attn_layers), use_adaln=True
            )
            self.action_map_attn = FFWRelativeCrossAttentionModule(
                embed_dim, num_heads, num_layers=int(cross_attn_layers), use_adaln=True
            )
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
        self.node_head = nn.Linear(embed_dim, self.point_dim)
        self.object_head = self.node_head
        self.tcp_head = nn.Linear(embed_dim, self.tcp_dim)
        self.gripper_head = None if self.trajectory_dim == 4 else nn.Linear(embed_dim, 1)

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

    def _make_timestep(self, timestep: Union[torch.Tensor, float, int], batch_size: int, device):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device=device)
        else:
            timestep = timestep.to(device=device)
        return timestep.expand(batch_size).long()

    def _observation_context(self, obs: Dict[str, torch.Tensor]):
        point_cloud = obs.get("point_cloud")
        if point_cloud is None:
            raise ValueError("Map4DDiT requires obs.point_cloud")
        if point_cloud.ndim == 3:
            point_cloud = point_cloud[:, None]
        if point_cloud.ndim != 4:
            raise ValueError(f"obs.point_cloud must have shape [B, T, P, C], got {point_cloud.shape}")
        if point_cloud.shape[-1] < self.point_dim:
            raise ValueError(f"point_cloud last dim must be >= {self.point_dim}, got {point_cloud.shape[-1]}")

        dino_feature = obs.get("dino_feature")
        if dino_feature is None:
            raise ValueError("Map4DDiT requires precomputed obs.dino_feature")
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

        robot_state = obs["robot_state"]
        if robot_state.ndim != 3:
            raise ValueError(f"obs.robot_state must have shape [B, T, D], got {robot_state.shape}")
        if robot_state.shape[:2] != point_cloud.shape[:2]:
            raise ValueError(
                f"robot_state and point_cloud must share [B,T], got {robot_state.shape} and {point_cloud.shape}"
            )

        batch_size, steps, point_count = point_cloud.shape[:3]
        flat_obs = {
            "point_cloud": point_cloud[..., : self.point_dim].reshape(batch_size * steps, point_count, self.point_dim),
            "dino_feature": dino_feature.reshape(batch_size * steps, point_count, self.semantic_feature_dim),
            "robot_state": robot_state.reshape(batch_size * steps, robot_state.shape[-1]),
        }
        if "lang" in obs:
            lang = obs["lang"]
            if lang.ndim == 3:
                flat_obs["lang"] = lang.reshape(batch_size * steps, lang.shape[-1])
            elif lang.ndim == 2:
                flat_obs["lang"] = lang[:, None].expand(-1, steps, -1).reshape(batch_size * steps, lang.shape[-1])
            else:
                raise ValueError(f"obs.lang must have shape [B,D] or [B,T,D], got {lang.shape}")

        context_coord, context_feat, lang_feat, state_feat, sampled_coord, sampled_feat = self.obs_encoder(flat_obs)
        context_coord = context_coord.reshape(batch_size, steps * point_count, self.point_dim)
        context_feat = context_feat.reshape(batch_size, steps * point_count, self.embed_dim) + self.source_embed[0]
        sampled_coord = sampled_coord.reshape(batch_size, steps * sampled_coord.shape[1], self.point_dim)
        sampled_feat = sampled_feat.reshape(batch_size, steps * sampled_feat.shape[1], self.embed_dim) + self.source_embed[0]
        lang_feat = lang_feat.reshape(batch_size, steps, self.embed_dim)
        state_feat = state_feat.reshape(batch_size, steps, self.embed_dim)
        return context_coord, context_feat, lang_feat, state_feat, sampled_coord, sampled_feat

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

        node_position = obs.get("node_position")
        node_rotation = obs.get("node_rotation")
        if node_position is None or node_rotation is None:
            node_poses = obs.get("node_poses")
            if node_poses is None:
                raise ValueError("Map4DDiT requires obs.node_position and obs.node_rotation for online map encoding")
            node_position = node_poses[..., 0:3]
            node_rotation = node_poses[..., 3:]
        if node_position.ndim == 3:
            node_position = node_position[:, None]
        if node_rotation.ndim == 3:
            node_rotation = node_rotation[:, None]
        if node_position.ndim != 4 or node_position.shape[-1] != 3:
            raise ValueError(f"obs.node_position must have shape [B,T,N,3], got {node_position.shape}")
        if node_rotation.ndim != 4 or node_rotation.shape[-1] != 4:
            raise ValueError(f"obs.node_rotation must have shape [B,T,N,4], got {node_rotation.shape}")
        if node_position.shape[:3] != node_rotation.shape[:3]:
            raise ValueError(
                f"obs.node_position and obs.node_rotation must share [B,T,N], "
                f"got {node_position.shape} and {node_rotation.shape}"
            )
        if node_position.shape[1] != self.n_map_step:
            raise ValueError(f"Expected n_map_step={self.n_map_step}, got obs.node_position T={node_position.shape[1]}")
        if node_position.shape[2] != self.num_target_nodes:
            raise ValueError(
                f"Expected {self.num_target_nodes} target nodes in obs.node_position, got {node_position.shape[2]}"
            )

        batch_size, steps = node_position.shape[:2]
        positions = node_position.reshape(batch_size * steps, self.num_target_nodes, 3).reshape(batch_size * steps, -1)
        flat_rotation = node_rotation.reshape(batch_size * steps, self.num_target_nodes, node_rotation.shape[-1])
        rotations = flat_rotation.reshape(batch_size * steps, -1)

        size_parameters = obs.get("size_parameters")
        if size_parameters is None:
            raise ValueError("Map4DDiT requires obs.size_parameters for online map encoding")
        relation_parameters = obs.get("relation_parameters")
        if relation_parameters is None:
            raise ValueError("Map4DDiT requires obs.relation_parameters for online map encoding")
        sizes = self._expand_static_parameter(
            size_parameters.to(device=node_position.device, dtype=node_position.dtype),
            batch_size=batch_size,
            steps=steps,
            dim=self.size_parameter_dim,
            name="size_parameters",
        )
        relations = self._expand_static_parameter(
            relation_parameters.to(device=node_position.device, dtype=node_position.dtype),
            batch_size=batch_size,
            steps=steps,
            dim=self.relation_parameter_dim,
            name="relation_parameters",
        )

        if self.map_name == "StackCube-v1":
            return Map4d_StackCube(positions, rotations, sizes, relations, clip_model=None)
        if self.map_name == "PlugCharger-v1":
            return Map4d_PlugCharger(positions, rotations, sizes, relations, clip_model=None)
        if self.map_name == "rlbench2_push_box":
            return Map4d_RLBench2PushBox(positions, rotations, sizes, relations, clip_model=None)
        raise ValueError(f"Unsupported map_name={self.map_name!r}")

    def _encoded_map_nodes_from_gt(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_position = obs.get("node_position", obs.get("node_poses"))
        if node_position is None:
            raise ValueError("Map4DDiT requires obs.node_position")
        if node_position.ndim == 3:
            batch_size, steps = node_position.shape[0], 1
        else:
            batch_size, steps = node_position.shape[:2]
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
                f"remove {sorted(stale_keys)} and pass GT obs.node_position/node_rotation/size_parameters/relation_parameters."
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

    @staticmethod
    def _to_seq(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1)

    @staticmethod
    def _to_batch(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1)

    def _prepare_targets(self, noisy_targets: Dict[str, torch.Tensor]):
        trajectory = noisy_targets["trajectory"]
        keyframe_tcp = noisy_targets["keyframe_tcp"]

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
        if keyframe_tcp.shape[1:] != (self.num_arms, self.keyframe_horizon, self.tcp_dim):
            raise ValueError(
                "keyframe_tcp must have shape "
                f"[B, {self.num_arms}, {self.keyframe_horizon}, {self.tcp_dim}], got {keyframe_tcp.shape}"
            )
        return trajectory, keyframe_tcp, squeeze_trajectory_arm, squeeze_tcp_arm

    def _keyframe_node_queries(self, obs: Dict[str, torch.Tensor], batch_size: int, device, dtype):
        node_position = obs.get("node_position")
        if node_position is None:
            node_poses = obs.get("node_poses")
            if node_poses is None:
                raise ValueError("Map4DDiT requires obs.node_position for keyframe node queries")
            node_position = node_poses[..., 0:3]
        if node_position.ndim == 3:
            node_position = node_position[:, None]
        if node_position.ndim != 4 or node_position.shape[-1] != self.point_dim:
            raise ValueError(f"obs.node_position must have shape [B,T,N,3], got {node_position.shape}")
        if node_position.shape[0] != batch_size or node_position.shape[2] != self.num_target_nodes:
            raise ValueError(
                f"obs.node_position must have [B,N]=[{batch_size},{self.num_target_nodes}], "
                f"got {node_position.shape}"
            )
        current_node_position = node_position[:, -1].to(device=device, dtype=dtype)
        return current_node_position[:, None].expand(-1, self.keyframe_horizon, -1, -1)

    def _target_tokens(self, trajectory, keyframe_node_position, keyframe_tcp):
        batch_size = trajectory.shape[0]

        traj_tokens = self.trajectory_proj(trajectory)
        traj_tokens = traj_tokens + self.trajectory_pos_embed + self.arm_embed[:, : self.num_arms]
        traj_tokens = traj_tokens + self.target_type_embed[0]
        traj_xyz = trajectory[..., :3].reshape(batch_size, -1, 3)
        traj_tokens = traj_tokens.reshape(batch_size, -1, self.embed_dim)

        node_tokens = self.node_proj(keyframe_node_position)
        node_tokens = node_tokens + self.keyframe_time_embed + self.node_id_embed
        node_tokens = node_tokens + self.target_type_embed[1]
        node_xyz = keyframe_node_position.reshape(batch_size, -1, self.point_dim)
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

    def encode_denoising_timestep(self, timestep, state_feat):
        time_feats = self.time_encoder(timestep)
        state_feat = state_feat[:, -self.obs_horizon :]
        if state_feat.shape[1] < self.obs_horizon:
            pad = state_feat[:, :1].expand(-1, self.obs_horizon - state_feat.shape[1], -1)
            state_feat = torch.cat([pad, state_feat], dim=1)
        return time_feats + state_feat.mean(dim=1)

    def prediction_head(
        self,
        traj_tokens,
        traj_xyz,
        node_tokens,
        node_xyz,
        tcp_tokens,
        tcp_xyz,
        context_xyz,
        context_token,
        sampled_context_xyz,
        sampled_context_token,
        timestep,
        state_feat,
        semantic_xyz=None,
        semantic_token=None,
        map_xyz=None,
        map_token=None,
    ):
        cond = self.encode_denoising_timestep(timestep, state_feat)

        if self.separate_map_cross_attn:
            if semantic_xyz is None or semantic_token is None or map_xyz is None or map_token is None:
                raise ValueError("separate_map_cross_attn requires semantic and map contexts")
            node_tokens = self._cross_context(self.node_map_attn, node_tokens, node_xyz, map_token, map_xyz, cond)
            node_context_xyz = semantic_xyz
            node_context_token = semantic_token
        else:
            node_context_xyz = context_xyz
            node_context_token = context_token
        node_feat = self._cross_context(
            self.node_context_attn, node_tokens, node_xyz, node_context_token, node_context_xyz, cond
        )
        node_plan = torch.cat([node_feat, sampled_context_token], dim=1)
        node_plan_xyz = torch.cat([node_xyz, sampled_context_xyz], dim=1)
        node_plan = self._self_attention(self.node_self_attn, node_plan, node_plan_xyz, cond)
        node_feat = node_plan[:, : node_tokens.shape[1]]

        node_condition = node_plan.detach() if self.detach_stage_features else node_plan
        node_condition_xyz = node_plan_xyz.detach() if self.detach_stage_features else node_plan_xyz
        if self.separate_map_cross_attn:
            tcp_tokens = self._cross_context(self.tcp_map_attn, tcp_tokens, tcp_xyz, map_token, map_xyz, cond)
            tcp_context_xyz = semantic_xyz
            tcp_context_token = semantic_token
        else:
            tcp_context_xyz = context_xyz
            tcp_context_token = context_token
        tcp_feat = self._cross_context(self.tcp_context_attn, tcp_tokens, tcp_xyz, tcp_context_token, tcp_context_xyz, cond)
        tcp_plan = torch.cat([tcp_feat, node_condition], dim=1)
        tcp_plan_xyz = torch.cat([tcp_xyz, node_condition_xyz], dim=1)
        tcp_plan = self._self_attention(self.tcp_self_attn, tcp_plan, tcp_plan_xyz, cond)
        tcp_feat = tcp_plan[:, : tcp_tokens.shape[1]]

        tcp_condition = tcp_plan.detach() if self.detach_stage_features else tcp_plan
        tcp_condition_xyz = tcp_plan_xyz.detach() if self.detach_stage_features else tcp_plan_xyz
        if self.separate_map_cross_attn:
            traj_tokens = self._cross_context(self.action_map_attn, traj_tokens, traj_xyz, map_token, map_xyz, cond)
            action_context_xyz = semantic_xyz
            action_context_token = semantic_token
        else:
            action_context_xyz = context_xyz
            action_context_token = context_token
        traj_feat = self._cross_context(
            self.action_context_attn, traj_tokens, traj_xyz, action_context_token, action_context_xyz, cond
        )
        action_plan = torch.cat([traj_feat, tcp_condition], dim=1)
        action_plan_xyz = torch.cat([traj_xyz, tcp_condition_xyz], dim=1)
        action_plan = self._self_attention(self.action_self_attn, action_plan, action_plan_xyz, cond)
        traj_feat = action_plan[:, : traj_tokens.shape[1]]
        return traj_feat, node_feat, tcp_feat

    def forward(
        self,
        noisy_targets: Dict[str, torch.Tensor],
        timestep: Union[torch.Tensor, float, int],
        obs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        trajectory, keyframe_tcp, squeeze_trajectory_arm, squeeze_tcp_arm = self._prepare_targets(
            noisy_targets
        )
        batch_size = trajectory.shape[0]
        device = trajectory.device
        timesteps = self._make_timestep(timestep, batch_size, device)

        (
            semantic_xyz,
            semantic_token,
            _lang_feat,
            state_feat,
            sampled_semantic_xyz,
            sampled_semantic_token,
        ) = self._observation_context(obs)
        map_xyz, map_token = self._map_context(obs)
        context_xyz = torch.cat([semantic_xyz, map_xyz], dim=1)
        context_token = torch.cat([semantic_token, map_token], dim=1)
        sampled_context_xyz = torch.cat([sampled_semantic_xyz, map_xyz], dim=1)
        sampled_context_token = torch.cat([sampled_semantic_token, map_token], dim=1)

        keyframe_node_position = self._keyframe_node_queries(obs, batch_size, device, trajectory.dtype)

        traj_tokens, traj_xyz, node_tokens, node_xyz, tcp_tokens, tcp_xyz = self._target_tokens(
            trajectory, keyframe_node_position, keyframe_tcp
        )
        traj_feat, node_feat, tcp_feat = self.prediction_head(
            traj_tokens=traj_tokens,
            traj_xyz=traj_xyz,
            node_tokens=node_tokens,
            node_xyz=node_xyz,
            tcp_tokens=tcp_tokens,
            tcp_xyz=tcp_xyz,
            context_xyz=context_xyz,
            context_token=context_token,
            sampled_context_xyz=sampled_context_xyz,
            sampled_context_token=sampled_context_token,
            timestep=timesteps,
            state_feat=state_feat,
            semantic_xyz=semantic_xyz,
            semantic_token=semantic_token,
            map_xyz=map_xyz,
            map_token=map_token,
        )
        cond = self.encode_denoising_timestep(timesteps, state_feat)
        shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
        node_feat = self.final_norm(node_feat) * (1 + scale[:, None]) + shift[:, None]
        tcp_feat = self.final_norm(tcp_feat) * (1 + scale[:, None]) + shift[:, None]
        traj_feat = self.final_norm(traj_feat) * (1 + scale[:, None]) + shift[:, None]

        pred_node_position = self.node_head(node_feat).reshape(
            batch_size, self.keyframe_horizon, self.num_target_nodes, self.point_dim
        )
        pred_tcp = self.tcp_head(tcp_feat).reshape(batch_size, self.num_arms, self.keyframe_horizon, self.tcp_dim)
        pred_trajectory = self.trajectory_head(traj_feat).reshape(
            batch_size, self.num_arms, self.action_horizon, self.trajectory_dim
        )
        if self.gripper_head is None:
            pred_gripper = pred_trajectory[..., 3:4]
        else:
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
            "keyframe_node_position": pred_node_position,
            "keyframe_node": pred_node_position,
            "keyframe_map4d": pred_node_position,
            "keyframe_object": pred_node_position,
            "keyframe_tcp": pred_tcp,
            "gripper_openness": pred_gripper,
            "trajectory_features": traj_feat_out,
            "keyframe_node_features": node_feat,
            "keyframe_tcp_features": tcp_feat,
        }
