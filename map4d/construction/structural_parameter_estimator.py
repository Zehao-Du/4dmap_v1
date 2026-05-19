from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MAPS4D_DIR = _REPO_ROOT / "map4d" / "representation" / "maps4d"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_MAPS4D_DIR) not in sys.path:
    sys.path.insert(0, str(_MAPS4D_DIR))

try:
    from maniskill_stackcube import Map4d_StackCube
except Exception:
    Map4d_StackCube = None


@dataclass(frozen=True)
class StructuralTaskSpec:
    """Task-specific structural parameter layout and optional map builder."""

    task_name: str
    param_dim: int
    param_names: tuple[str, ...]
    map_class: Optional[type] = None
    num_objects: Optional[int] = None
    preprocess_map_parameters: bool = False
    default_size_value: float = 0.04

    def __post_init__(self):
        if self.param_dim <= 0:
            raise ValueError("param_dim must be positive.")
        if len(self.param_names) != self.param_dim:
            raise ValueError(
                f"Task {self.task_name!r} has param_dim={self.param_dim}, "
                f"but {len(self.param_names)} param_names."
            )


STRUCTURAL_TASK_SPECS: dict[str, StructuralTaskSpec] = {}


def register_structural_task(spec: StructuralTaskSpec, *, force: bool = False) -> None:
    if spec.task_name in STRUCTURAL_TASK_SPECS and not force:
        raise KeyError(f"Structural task {spec.task_name!r} is already registered.")
    STRUCTURAL_TASK_SPECS[spec.task_name] = spec
    if "STRUCTURAL_PARAM_DIM_VOCAB" in globals():
        STRUCTURAL_PARAM_DIM_VOCAB[spec.task_name] = spec.param_dim
    if "STRUCTURAL_MAP_CLASS_VOCAB" in globals():
        if spec.map_class is not None:
            STRUCTURAL_MAP_CLASS_VOCAB[spec.task_name] = spec.map_class
        else:
            STRUCTURAL_MAP_CLASS_VOCAB.pop(spec.task_name, None)


def get_structural_task_spec(task_name: str) -> StructuralTaskSpec:
    if task_name not in STRUCTURAL_TASK_SPECS:
        available = sorted(STRUCTURAL_TASK_SPECS)
        raise ValueError(f"Unknown structural task: {task_name!r}. Available: {available}")
    return STRUCTURAL_TASK_SPECS[task_name]


def _register_builtin_tasks() -> None:
    if Map4d_StackCube is None:
        return
    register_structural_task(
        StructuralTaskSpec(
            task_name="StackCube-v1",
            param_dim=9,
            param_names=(
                "red_cube_height",
                "red_cube_length",
                "red_cube_width",
                "green_cube_height",
                "green_cube_length",
                "green_cube_width",
                "desk_height",
                "desk_length",
                "desk_width",
            ),
            map_class=Map4d_StackCube,
            num_objects=3,
            preprocess_map_parameters=False,
        ),
        force=True,
    )


_register_builtin_tasks()

STRUCTURAL_PARAM_DIM_VOCAB = {
    task_name: spec.param_dim for task_name, spec in STRUCTURAL_TASK_SPECS.items()
}
STRUCTURAL_MAP_CLASS_VOCAB = {
    task_name: spec.map_class
    for task_name, spec in STRUCTURAL_TASK_SPECS.items()
    if spec.map_class is not None
}


@dataclass(frozen=True)
class StructuralParameterEstimatorConfig:
    """Configuration for PointNet + MLP structural parameter regression."""

    task_name: Optional[str] = None
    input_dim: int = 3
    pointnet_channels: tuple[int, ...] = (64, 128, 256)
    mlp_channels: tuple[int, ...] = (256, 128)
    param_dim: Optional[int] = None
    dropout: float = 0.0
    normalize_point_cloud: bool = True
    positive_output: bool = True
    min_positive_value: float = 1e-4
    param_names: Optional[tuple[str, ...]] = None
    build_map: bool = False
    map_preprocess: Optional[bool] = None


class SharedMLP(nn.Module):
    """Per-point MLP used by PointNet."""

    def __init__(
        self,
        channels: Sequence[int],
        *,
        use_batch_norm: bool = True,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must contain at least input and output dimensions.")

        layers = []
        for in_dim, out_dim in zip(channels[:-1], channels[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected points with shape [B, N, C], got {tuple(points.shape)}")

        batch_size, num_points, channels = points.shape
        flat = points.reshape(batch_size * num_points, channels)
        flat = self.net(flat)
        return flat.reshape(batch_size, num_points, -1)


class PointNetEncoder(nn.Module):
    """PointNet encoder for padded masked point clouds.

    Args:
        points: Tensor with shape [B, N, C].
        point_mask: Optional bool/0-1 tensor with shape [B, N]. True means valid.

    Returns:
        Global feature tensor with shape [B, pointnet_channels[-1]].
    """

    def __init__(
        self,
        *,
        input_dim: int = 3,
        channels: Sequence[int] = (64, 128, 256),
        normalize_point_cloud: bool = True,
    ):
        super().__init__()
        if input_dim < 3:
            raise ValueError("input_dim must be at least 3 because xyz coordinates are required.")
        if len(channels) == 0:
            raise ValueError("channels must be non-empty.")

        self.input_dim = int(input_dim)
        self.output_dim = int(channels[-1])
        self.normalize_point_cloud = bool(normalize_point_cloud)
        self.point_mlp = SharedMLP((input_dim, *channels))

    def forward(self, points: torch.Tensor, point_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        points, point_mask = self._validate_inputs(points, point_mask)
        points = self._normalize_points(points, point_mask) if self.normalize_point_cloud else points

        features = self.point_mlp(points)
        if point_mask is not None:
            features = features.masked_fill(~point_mask[..., None], torch.finfo(features.dtype).min)
        pooled = features.max(dim=1).values
        return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    def _validate_inputs(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if points.ndim != 3:
            raise ValueError(f"Expected points with shape [B, N, C], got {tuple(points.shape)}")
        if points.shape[-1] != self.input_dim:
            raise ValueError(f"Expected point channel dim {self.input_dim}, got {points.shape[-1]}")
        if points.shape[1] == 0:
            raise ValueError("Point cloud must contain at least one point.")

        points = points.float()
        if point_mask is None:
            return points, None

        if point_mask.shape != points.shape[:2]:
            raise ValueError(
                f"point_mask shape {tuple(point_mask.shape)} must match points shape {tuple(points.shape[:2])}"
            )
        point_mask = point_mask.to(device=points.device, dtype=torch.bool)
        if not point_mask.any(dim=1).all():
            raise ValueError("Every batch item must contain at least one valid point.")
        return points, point_mask

    @staticmethod
    def _normalize_points(points: torch.Tensor, point_mask: Optional[torch.Tensor]) -> torch.Tensor:
        xyz = points[..., :3]
        extra = points[..., 3:]

        if point_mask is None:
            center = xyz.mean(dim=1, keepdim=True)
            centered_xyz = xyz - center
            scale = centered_xyz.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            weights = point_mask[..., None].to(dtype=xyz.dtype)
            count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            center = (xyz * weights).sum(dim=1, keepdim=True) / count
            centered_xyz = (xyz - center) * weights
            scale = centered_xyz.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)

        normalized_xyz = centered_xyz / scale[..., None]
        if extra.numel() == 0:
            return normalized_xyz
        return torch.cat([normalized_xyz, extra], dim=-1)


class StructuralParameterEstimator(nn.Module):
    """Estimate object structural parameters from masked point clouds.

    The module follows the MapPolicy constructor style: task-specific output
    dimensions and map classes are resolved from a registry, while the network
    body stays generic. Register a new ``StructuralTaskSpec`` for each new map.
    """

    def __init__(self, config: StructuralParameterEstimatorConfig):
        super().__init__()
        self.config = config
        self.task_spec = get_structural_task_spec(config.task_name) if config.task_name is not None else None
        self.param_dim = self._resolve_param_dim(config)
        self.param_names = self._resolve_param_names(config)
        self.MapClass = self.task_spec.map_class if self.task_spec is not None else None
        self.encoder = PointNetEncoder(
            input_dim=config.input_dim,
            channels=config.pointnet_channels,
            normalize_point_cloud=config.normalize_point_cloud,
        )
        self.regressor = self._build_regressor(
            input_dim=self.encoder.output_dim,
            hidden_dims=config.mlp_channels,
            output_dim=self.param_dim,
            dropout=config.dropout,
        )

    @classmethod
    def for_cuboid(
        cls,
        *,
        input_dim: int = 3,
        param_names: Sequence[str] = ("height", "length", "width"),
        **kwargs,
    ) -> "StructuralParameterEstimator":
        return cls(
            StructuralParameterEstimatorConfig(
                input_dim=input_dim,
                param_dim=3,
                param_names=tuple(param_names),
                **kwargs,
            )
        )

    @classmethod
    def for_stackcube(
        cls,
        *,
        input_dim: int = 3,
        **kwargs,
    ) -> "StructuralParameterEstimator":
        return cls(
            StructuralParameterEstimatorConfig(
                task_name="StackCube-v1",
                input_dim=input_dim,
                **kwargs,
            )
        )

    @classmethod
    def for_task(
        cls,
        task_name: str,
        *,
        input_dim: int = 3,
        **kwargs,
    ) -> "StructuralParameterEstimator":
        return cls(
            StructuralParameterEstimatorConfig(
                task_name=task_name,
                input_dim=input_dim,
                **kwargs,
            )
        )

    def forward(
        self,
        masked_point_cloud: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        *,
        return_dict: bool = False,
        build_map: Optional[bool] = None,
    ):
        features = self.encoder(masked_point_cloud, point_mask)
        raw_params = self.regressor(features)
        params = self._transform_output(raw_params)

        should_build_map = self.config.build_map if build_map is None else bool(build_map)
        scene_map = self.build_map_from_params(params) if should_build_map else None

        if not return_dict:
            return scene_map if should_build_map else params

        return {
            "params": params,
            "raw_params": raw_params,
            "features": features,
            "param_names": self.param_names,
            "task_name": None if self.task_spec is None else self.task_spec.task_name,
            "scene_map": scene_map,
        }

    def build_map_from_params(
        self,
        structural_params: torch.Tensor,
        *,
        positions: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        clip_model=None,
    ):
        if self.task_spec is None or self.MapClass is None:
            raise ValueError("build_map_from_params requires a registered task with map_class.")
        if structural_params.ndim != 2 or structural_params.shape[1] != self.param_dim:
            raise ValueError(
                f"Expected structural_params shape [B, {self.param_dim}], got {tuple(structural_params.shape)}"
            )

        batch_size = structural_params.shape[0]
        device = structural_params.device
        dtype = structural_params.dtype
        num_objects = self.task_spec.num_objects or self._infer_num_objects(self.param_dim)
        pos_dim = num_objects * 3

        if positions is None:
            positions = torch.zeros((batch_size, pos_dim), dtype=dtype, device=device)
        if rotations is None:
            rotations = self._identity_rotations(batch_size, num_objects, dtype=dtype, device=device)

        preprocess = (
            self.task_spec.preprocess_map_parameters
            if self.config.map_preprocess is None
            else bool(self.config.map_preprocess)
        )
        return self.MapClass(structural_params, positions, rotations, clip_model, preprocess=preprocess)

    def _transform_output(self, raw_params: torch.Tensor) -> torch.Tensor:
        if not self.config.positive_output:
            return raw_params
        return F.softplus(raw_params) + float(self.config.min_positive_value)

    def _resolve_param_dim(self, config: StructuralParameterEstimatorConfig) -> int:
        if config.param_dim is not None:
            return int(config.param_dim)
        if self.task_spec is not None:
            return int(self.task_spec.param_dim)
        raise ValueError("param_dim must be provided when task_name is not set.")

    def _resolve_param_names(self, config: StructuralParameterEstimatorConfig) -> Optional[tuple[str, ...]]:
        if config.param_names is not None:
            if len(config.param_names) != self.param_dim:
                raise ValueError(
                    f"param_names length {len(config.param_names)} does not match param_dim {self.param_dim}."
                )
            return tuple(config.param_names)
        if self.task_spec is not None:
            return self.task_spec.param_names
        return None

    @staticmethod
    def _infer_num_objects(param_dim: int) -> int:
        if param_dim % 3 != 0:
            raise ValueError(f"Cannot infer number of objects from param_dim={param_dim}; expected multiple of 3.")
        return param_dim // 3

    @staticmethod
    def _identity_rotations(batch_size: int, num_objects: int, *, dtype, device) -> torch.Tensor:
        rotations = torch.zeros((batch_size, num_objects * 6), dtype=dtype, device=device)
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype, device=device)
        for object_index in range(num_objects):
            rotations[:, object_index * 6 : (object_index + 1) * 6] = identity_6d
        return rotations

    @staticmethod
    def _build_regressor(
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)


def estimate_structural_parameters(
    model: StructuralParameterEstimator,
    masked_point_cloud: torch.Tensor,
    point_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inference helper for structural parameter estimation."""

    was_training = model.training
    model.eval()
    with torch.no_grad():
        params = model(masked_point_cloud, point_mask)
    if was_training:
        model.train()
    return params


def structural_parameter_l1_loss(
    pred_params: torch.Tensor,
    target_params: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Default supervised loss for structural parameters."""

    if pred_params.shape != target_params.shape:
        raise ValueError(
            f"pred_params shape {tuple(pred_params.shape)} must match target_params shape {tuple(target_params.shape)}"
        )
    return F.l1_loss(pred_params, target_params.float(), reduction=reduction)


__all__ = [
    "PointNetEncoder",
    "StructuralParameterEstimator",
    "StructuralParameterEstimatorConfig",
    "StructuralTaskSpec",
    "STRUCTURAL_MAP_CLASS_VOCAB",
    "STRUCTURAL_PARAM_DIM_VOCAB",
    "STRUCTURAL_TASK_SPECS",
    "estimate_structural_parameters",
    "get_structural_task_spec",
    "register_structural_task",
    "structural_parameter_l1_loss",
]
