from __future__ import annotations

import importlib
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

from hydra.utils import instantiate
from omegaconf import OmegaConf


PathLike = Union[str, pathlib.Path]


DEFAULT_SAM2_VERSION = "sam2.1-hiera-large"
DEFAULT_SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CKPT_NAME = "sam2.1_hiera_large.pt"

HF_MODEL_ID_TO_LOCAL_FILES = {
    "facebook/sam2-hiera-tiny": ("configs/sam2/sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
    "facebook/sam2-hiera-small": ("configs/sam2/sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
    "facebook/sam2-hiera-base-plus": ("configs/sam2/sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
    "facebook/sam2-hiera-large": ("configs/sam2/sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    "facebook/sam2.1-hiera-tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "facebook/sam2.1-hiera-small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "facebook/sam2.1-hiera-base-plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "facebook/sam2.1-hiera-large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}

LOCAL_VERSION_ALIASES = {
    "sam2-hiera-tiny": "facebook/sam2-hiera-tiny",
    "sam2-hiera-small": "facebook/sam2-hiera-small",
    "sam2-hiera-base-plus": "facebook/sam2-hiera-base-plus",
    "sam2-hiera-large": "facebook/sam2-hiera-large",
    "sam2.1-hiera-tiny": "facebook/sam2.1-hiera-tiny",
    "sam2.1-hiera-small": "facebook/sam2.1-hiera-small",
    "sam2.1-hiera-base-plus": "facebook/sam2.1-hiera-base-plus",
    "sam2.1-hiera-large": "facebook/sam2.1-hiera-large",
    "sam2": "facebook/sam2-hiera-large",
    "sam2.1": "facebook/sam2.1-hiera-large",
}


@dataclass(frozen=True)
class SAM2CheckpointSpec:
    model_id: str
    checkpoint_path: pathlib.Path
    config_path: pathlib.Path


@dataclass(frozen=True)
class SAM2AssetsSpec:
    repo_root: pathlib.Path
    package_root: pathlib.Path
    checkpoints_root: pathlib.Path
    device: str
    checkpoint: SAM2CheckpointSpec


@dataclass(frozen=True)
class SAM2ValidationReport:
    ok: bool
    repo_root: pathlib.Path
    package_root: pathlib.Path
    checkpoints_root: pathlib.Path
    device: str
    checkpoint: SAM2CheckpointSpec
    checked_dependencies: tuple[str, ...] = field(default_factory=tuple)
    missing_dependencies: tuple[str, ...] = field(default_factory=tuple)
    checkpoint_load_checked: bool = False
    model_instantiation_checked: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


def _loader_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _default_repo_root() -> pathlib.Path:
    return _loader_dir() / "sam2"


def _default_package_root(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "sam2"


def default_sam2_checkpoints_root() -> pathlib.Path:
    return _loader_dir() / "weights"


def _resolve_optional_path(path: Optional[PathLike]) -> Optional[pathlib.Path]:
    if path is None:
        return None
    return pathlib.Path(path).expanduser().resolve()


def _require_existing_dir(path: pathlib.Path, desc: str) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{desc} is not a directory: {path}")
    return path


def _require_existing_file(path: pathlib.Path, desc: str) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{desc} is not a file: {path}")
    return path


def _ensure_import_paths(repo_root: pathlib.Path, package_root: pathlib.Path) -> None:
    for path in (repo_root, package_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _dependency_module_names() -> tuple[str, ...]:
    return (
        "torch",
        "hydra",
        "omegaconf",
        "numpy",
        "PIL",
    )


def _check_dependencies() -> None:
    missing = _find_missing_dependencies(_dependency_module_names())
    if missing:
        raise ImportError(
            "SAM2 dependencies are missing. Install or fix these modules first: "
            + ", ".join(missing)
        )


def _find_missing_dependencies(module_names: tuple[str, ...]) -> tuple[str, ...]:
    missing = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return tuple(missing)


def _normalize_device(device: Optional[str]) -> str:
    import torch

    requested = (device or ("cuda" if torch.cuda.is_available() else "cpu")).strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device {requested!r}, but torch.cuda.is_available() is False."
            )
        if ":" in requested:
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device string: {device!r}") from exc
            device_count = torch.cuda.device_count()
            if index < 0 or index >= device_count:
                raise ValueError(
                    f"CUDA device index out of range: {requested!r}; available device count={device_count}."
                )
        return requested
    if requested == "cpu":
        return requested
    raise ValueError(f"Unsupported SAM2 device: {device!r}. Expected 'cpu', 'cuda', or 'cuda:N'.")


def _normalize_model_id(version: str) -> str:
    normalized = str(version).strip().lower()
    normalized = LOCAL_VERSION_ALIASES.get(normalized, normalized)
    if normalized not in HF_MODEL_ID_TO_LOCAL_FILES:
        supported = ", ".join(sorted(LOCAL_VERSION_ALIASES))
        raise ValueError(f"Unsupported SAM2 version/model id {version!r}. Supported aliases: {supported}")
    return normalized


def _default_config_name(model_id: str) -> str:
    return HF_MODEL_ID_TO_LOCAL_FILES[model_id][0]


def _default_checkpoint_name(model_id: str) -> str:
    return HF_MODEL_ID_TO_LOCAL_FILES[model_id][1]


def _config_name_for_builder(repo_root: pathlib.Path, config_path: pathlib.Path) -> str:
    try:
        return str(config_path.relative_to(repo_root / "sam2"))
    except ValueError:
        return str(config_path)


def _resolve_checkpoint_spec(
    *,
    repo_root: pathlib.Path,
    checkpoints_root: pathlib.Path,
    version: str,
    checkpoint_path: Optional[pathlib.Path],
    config_path: Optional[pathlib.Path],
) -> SAM2CheckpointSpec:
    model_id = _normalize_model_id(version)
    resolved_config = _resolve_optional_path(config_path)
    resolved_checkpoint = _resolve_optional_path(checkpoint_path)

    if resolved_config is None:
        resolved_config = repo_root / "sam2" / _default_config_name(model_id)
    if resolved_checkpoint is None:
        resolved_checkpoint = checkpoints_root / _default_checkpoint_name(model_id)

    return SAM2CheckpointSpec(
        model_id=model_id,
        checkpoint_path=_require_existing_file(resolved_checkpoint, "SAM2 checkpoint"),
        config_path=_require_existing_file(resolved_config, "SAM2 config"),
    )


def _sam2_load_config(config_path: pathlib.Path):
    cfg = OmegaConf.load(str(config_path))
    OmegaConf.resolve(cfg)
    return cfg


def _omega_set(cfg, dotted_key: str, value):
    OmegaConf.update(cfg, dotted_key, value, merge=False)


def _build_sam2_model_private_hydra(
    *,
    config_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    device: str,
    mode: str,
    apply_postprocessing: bool,
):
    cfg = _sam2_load_config(config_path)
    if apply_postprocessing:
        _omega_set(cfg, 'model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability', True)
        _omega_set(cfg, 'model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta', 0.05)
        _omega_set(cfg, 'model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh', 0.98)
    model = instantiate(cfg.model, _recursive_=True)
    build_mod = importlib.import_module('sam2.build_sam')
    build_mod._load_checkpoint(model, str(checkpoint_path))
    model = model.to(device)
    if mode == 'eval':
        model.eval()
    return model


def _build_sam2_video_predictor_private_hydra(
    *,
    config_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    device: str,
    mode: str,
    apply_postprocessing: bool,
    vos_optimized: bool,
):
    cfg = _sam2_load_config(config_path)
    _omega_set(
        cfg,
        'model._target_',
        'sam2.sam2_video_predictor.SAM2VideoPredictorVOS' if vos_optimized else 'sam2.sam2_video_predictor.SAM2VideoPredictor',
    )
    if vos_optimized:
        _omega_set(cfg, 'model.compile_image_encoder', True)
    if apply_postprocessing:
        _omega_set(cfg, 'model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability', True)
        _omega_set(cfg, 'model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta', 0.05)
        _omega_set(cfg, 'model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh', 0.98)
        _omega_set(cfg, 'model.binarize_mask_from_pts_for_mem_enc', True)
        _omega_set(cfg, 'model.fill_hole_area', 8)
    model = instantiate(cfg.model, _recursive_=True)
    build_mod = importlib.import_module('sam2.build_sam')
    build_mod._load_checkpoint(model, str(checkpoint_path))
    model = model.to(device)
    if mode == 'eval':
        model.eval()
    return model


def _torch_load_checkpoint_header(checkpoint_path: pathlib.Path):
    import torch

    return torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)


class SAM2Loader:
    """Validated wrapper around the vendored SAM2 repo inside map4d.

    Default local layout:
      map4d/construction/SAM2/
        ├── sam2_loader.py
        ├── weights/
        │   └── sam2.1_hiera_large.pt
        └── sam2/
            └── sam2/
                └── configs/sam2.1/sam2.1_hiera_l.yaml
    """

    def __init__(
        self,
        repo_root: Optional[PathLike] = None,
        checkpoint_path: Optional[PathLike] = None,
        config_path: Optional[PathLike] = None,
        checkpoints_root: Optional[PathLike] = None,
        *,
        version: str = DEFAULT_SAM2_VERSION,
        device: Optional[str] = None,
    ):
        _check_dependencies()

        self.repo_root = _resolve_optional_path(repo_root) or _default_repo_root()
        self.package_root = _default_package_root(self.repo_root)
        self.checkpoints_root = _resolve_optional_path(checkpoints_root) or default_sam2_checkpoints_root()
        self.checkpoint_path = _resolve_optional_path(checkpoint_path)
        self.config_path = _resolve_optional_path(config_path)
        self.version = _normalize_model_id(version)
        self.device = _normalize_device(device)

        _require_existing_dir(self.repo_root, "SAM2 repo root")
        _require_existing_dir(self.package_root, "SAM2 package root")
        _require_existing_dir(self.checkpoints_root, "SAM2 checkpoints root")
        _ensure_import_paths(self.repo_root, self.package_root)

    def resolve_checkpoint(self) -> SAM2CheckpointSpec:
        return _resolve_checkpoint_spec(
            repo_root=self.repo_root,
            checkpoints_root=self.checkpoints_root,
            version=self.version,
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
        )

    def resolve_assets(self) -> SAM2AssetsSpec:
        return SAM2AssetsSpec(
            repo_root=self.repo_root,
            package_root=self.package_root,
            checkpoints_root=self.checkpoints_root,
            device=self.device,
            checkpoint=self.resolve_checkpoint(),
        )

    def _import_build_module(self):
        return importlib.import_module("sam2.build_sam")

    def load_model(self, *, mode: str = "eval", apply_postprocessing: bool = True, **kwargs):
        assets = self.resolve_assets()
        if kwargs:
            raise TypeError(f'Unsupported SAM2 load_model kwargs for private-hydra path: {list(kwargs.keys())}')
        return _build_sam2_model_private_hydra(
            config_path=assets.checkpoint.config_path,
            checkpoint_path=assets.checkpoint.checkpoint_path,
            device=assets.device,
            mode=mode,
            apply_postprocessing=bool(apply_postprocessing),
        )

    def load_image_predictor(self, *, mode: str = "eval", apply_postprocessing: bool = True, **kwargs):
        predictor_mod = importlib.import_module("sam2.sam2_image_predictor")
        model = self.load_model(mode=mode, apply_postprocessing=apply_postprocessing, **kwargs)
        return predictor_mod.SAM2ImagePredictor(model)

    def load_video_predictor(
        self,
        *,
        mode: str = "eval",
        apply_postprocessing: bool = True,
        vos_optimized: bool = False,
        **kwargs,
    ):
        assets = self.resolve_assets()
        if kwargs:
            raise TypeError(f'Unsupported SAM2 load_video_predictor kwargs for private-hydra path: {list(kwargs.keys())}')
        return _build_sam2_video_predictor_private_hydra(
            config_path=assets.checkpoint.config_path,
            checkpoint_path=assets.checkpoint.checkpoint_path,
            device=assets.device,
            mode=mode,
            apply_postprocessing=bool(apply_postprocessing),
            vos_optimized=bool(vos_optimized),
        )

    def validate(
        self,
        *,
        check_checkpoint_load: bool = False,
        check_model_instantiation: bool = False,
        instantiation_device: Optional[str] = None,
    ) -> SAM2ValidationReport:
        return validate_sam2_loader(
            repo_root=self.repo_root,
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            checkpoints_root=self.checkpoints_root,
            version=self.version,
            device=self.device,
            check_checkpoint_load=check_checkpoint_load,
            check_model_instantiation=check_model_instantiation,
            instantiation_device=instantiation_device,
        )


def validate_sam2_loader(
    *,
    repo_root: Optional[PathLike] = None,
    checkpoint_path: Optional[PathLike] = None,
    config_path: Optional[PathLike] = None,
    checkpoints_root: Optional[PathLike] = None,
    version: str = DEFAULT_SAM2_VERSION,
    device: Optional[str] = None,
    check_checkpoint_load: bool = False,
    check_model_instantiation: bool = False,
    instantiation_device: Optional[str] = None,
) -> SAM2ValidationReport:
    _check_dependencies()

    resolved_repo_root = _resolve_optional_path(repo_root) or _default_repo_root()
    resolved_repo_root = _require_existing_dir(resolved_repo_root, "SAM2 repo root")
    package_root = _require_existing_dir(_default_package_root(resolved_repo_root), "SAM2 package root")
    resolved_checkpoints_root = _resolve_optional_path(checkpoints_root) or default_sam2_checkpoints_root()
    resolved_checkpoints_root = _require_existing_dir(resolved_checkpoints_root, "SAM2 checkpoints root")
    normalized_version = _normalize_model_id(version)
    normalized_device = _normalize_device(device)

    _ensure_import_paths(resolved_repo_root, package_root)

    checkpoint = _resolve_checkpoint_spec(
        repo_root=resolved_repo_root,
        checkpoints_root=resolved_checkpoints_root,
        version=normalized_version,
        checkpoint_path=_resolve_optional_path(checkpoint_path),
        config_path=_resolve_optional_path(config_path),
    )

    checked_dependencies = _dependency_module_names()
    missing_dependencies = _find_missing_dependencies(checked_dependencies)
    notes = [
        f"Default checkpoints root is {default_sam2_checkpoints_root()}",
        f"Preferred local checkpoint path is {resolved_checkpoints_root / _default_checkpoint_name(normalized_version)}",
        f"Resolved builder config name is {_config_name_for_builder(resolved_repo_root, checkpoint.config_path)}",
        "Model instantiation smoke test builds the base SAM2 model only; it does not run predictor inference.",
    ]

    if check_checkpoint_load:
        payload = _torch_load_checkpoint_header(checkpoint.checkpoint_path)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected SAM2 checkpoint payload type: {type(payload).__name__}."
            )
        if "model" not in payload:
            raise RuntimeError(
                f"SAM2 checkpoint is missing expected top-level 'model' key: {checkpoint.checkpoint_path}"
            )
        notes.append("torch.load(map_location='cpu', weights_only=True) succeeded and top-level 'model' key is present.")

    if check_model_instantiation:
        requested_instantiation_device = _normalize_device(instantiation_device or normalized_device)
        temp_loader = SAM2Loader(
            repo_root=resolved_repo_root,
            checkpoint_path=checkpoint.checkpoint_path,
            config_path=checkpoint.config_path,
            checkpoints_root=resolved_checkpoints_root,
            version=normalized_version,
            device=requested_instantiation_device,
        )
        model = temp_loader.load_model()
        notes.append(
            f"SAM2 base model instantiation succeeded on device={requested_instantiation_device!r} with type={type(model).__name__}."
        )

    return SAM2ValidationReport(
        ok=True,
        repo_root=resolved_repo_root,
        package_root=package_root,
        checkpoints_root=resolved_checkpoints_root,
        device=normalized_device,
        checkpoint=checkpoint,
        checked_dependencies=checked_dependencies,
        missing_dependencies=missing_dependencies,
        checkpoint_load_checked=bool(check_checkpoint_load),
        model_instantiation_checked=bool(check_model_instantiation),
        notes=tuple(notes),
    )


def load_sam2_model(
    checkpoint_path: Optional[PathLike] = None,
    *,
    repo_root: Optional[PathLike] = None,
    config_path: Optional[PathLike] = None,
    checkpoints_root: Optional[PathLike] = None,
    version: str = DEFAULT_SAM2_VERSION,
    device: Optional[str] = None,
    mode: str = "eval",
    apply_postprocessing: bool = True,
    **kwargs,
):
    loader = SAM2Loader(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        checkpoints_root=checkpoints_root,
        version=version,
        device=device,
    )
    return loader.load_model(mode=mode, apply_postprocessing=apply_postprocessing, **kwargs)


def load_sam2_image_predictor(
    checkpoint_path: Optional[PathLike] = None,
    *,
    repo_root: Optional[PathLike] = None,
    config_path: Optional[PathLike] = None,
    checkpoints_root: Optional[PathLike] = None,
    version: str = DEFAULT_SAM2_VERSION,
    device: Optional[str] = None,
    mode: str = "eval",
    apply_postprocessing: bool = True,
    **kwargs,
):
    loader = SAM2Loader(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        checkpoints_root=checkpoints_root,
        version=version,
        device=device,
    )
    return loader.load_image_predictor(mode=mode, apply_postprocessing=apply_postprocessing, **kwargs)


def load_sam2_video_predictor(
    checkpoint_path: Optional[PathLike] = None,
    *,
    repo_root: Optional[PathLike] = None,
    config_path: Optional[PathLike] = None,
    checkpoints_root: Optional[PathLike] = None,
    version: str = DEFAULT_SAM2_VERSION,
    device: Optional[str] = None,
    mode: str = "eval",
    apply_postprocessing: bool = True,
    vos_optimized: bool = False,
    **kwargs,
):
    loader = SAM2Loader(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        checkpoints_root=checkpoints_root,
        version=version,
        device=device,
    )
    return loader.load_video_predictor(
        mode=mode,
        apply_postprocessing=apply_postprocessing,
        vos_optimized=vos_optimized,
        **kwargs,
    )


def sam2_validation_report_to_json(report: SAM2ValidationReport) -> str:
    payload = {
        "ok": report.ok,
        "repo_root": str(report.repo_root),
        "package_root": str(report.package_root),
        "checkpoints_root": str(report.checkpoints_root),
        "device": report.device,
        "checkpoint": {
            "model_id": report.checkpoint.model_id,
            "checkpoint_path": str(report.checkpoint.checkpoint_path),
            "config_path": str(report.checkpoint.config_path),
        },
        "checked_dependencies": list(report.checked_dependencies),
        "missing_dependencies": list(report.missing_dependencies),
        "checkpoint_load_checked": report.checkpoint_load_checked,
        "model_instantiation_checked": report.model_instantiation_checked,
        "notes": list(report.notes),
    }
    return json.dumps(payload, indent=2)
