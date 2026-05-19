from __future__ import annotations

import importlib
import logging
import os
import pathlib
import sys
import types
from dataclasses import dataclass, field
from typing import Optional, Union


PathLike = Union[str, pathlib.Path]


DEFAULT_SCORER_RUN_NAME = "2024-01-11-20-02-45"
DEFAULT_REFINER_RUN_NAME = "2023-10-28-18-33-37"


@dataclass(frozen=True)
class FoundationPoseCheckpointSpec:
    """Resolved config/checkpoint pair for one FoundationPose sub-model."""

    config_path: pathlib.Path
    ckpt_path: pathlib.Path


@dataclass(frozen=True)
class FoundationPoseRuntimeSpec:
    """Resolved runtime information for a FoundationPose loader instance."""

    repo_root: pathlib.Path
    device: str
    weights_root: pathlib.Path
    scorer: FoundationPoseCheckpointSpec
    refiner: FoundationPoseCheckpointSpec


@dataclass(frozen=True)
class FoundationPoseValidationReport:
    """Structured validation result for the vendored FoundationPose loader."""

    ok: bool
    repo_root: pathlib.Path
    default_weights_root: pathlib.Path
    resolved_weights_root: pathlib.Path
    device: str
    scorer: FoundationPoseCheckpointSpec
    refiner: FoundationPoseCheckpointSpec
    checked_dependencies: tuple[str, ...] = field(default_factory=tuple)
    missing_dependencies: tuple[str, ...] = field(default_factory=tuple)
    runtime_ready: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


def _foundationpose_loader_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _foundationpose_repo_root() -> pathlib.Path:
    return _foundationpose_loader_dir() / "FoundationPose"


def default_foundationpose_weights_root() -> pathlib.Path:
    """Default weights directory used by this loader.

    Convention:
      <directory containing foundationpose_loader.py>/weights/

    In this vendored MapPolicy layout, that resolves to:
      map4d/construction/foundationpose/weights/
    """

    return _foundationpose_loader_dir() / "weights"


def _install_transformations_compat() -> None:
    if 'transformations' in sys.modules:
        return
    from . import transformations_compat

    module = types.ModuleType('transformations')
    module.euler_matrix = transformations_compat.euler_matrix
    module.__all__ = ['euler_matrix']
    sys.modules['transformations'] = module


def _install_ruamel_yaml_compat() -> None:
    try:
        importlib.import_module("ruamel.yaml")
        return
    except Exception:
        pass

    import yaml as pyyaml

    class _CompatYAML:
        def load(self, stream):
            return pyyaml.safe_load(stream)

        def dump(self, data, stream=None, **kwargs):
            return pyyaml.safe_dump(data, stream, **kwargs)

        def safe_load(self, stream):
            return pyyaml.safe_load(stream)

        def safe_dump(self, data, stream=None, **kwargs):
            return pyyaml.safe_dump(data, stream, **kwargs)

    ruamel_pkg = sys.modules.get("ruamel")
    if ruamel_pkg is None:
        ruamel_pkg = types.ModuleType("ruamel")
        ruamel_pkg.__path__ = []

    yaml_module = types.ModuleType("ruamel.yaml")
    yaml_module.YAML = _CompatYAML
    ruamel_pkg.yaml = yaml_module
    sys.modules["ruamel"] = ruamel_pkg
    sys.modules["ruamel.yaml"] = yaml_module


def _install_mycpp_compat() -> None:
    try:
        mycpp_module = importlib.import_module("mycpp.build.mycpp")
        mycpp_pkg = importlib.import_module("mycpp")
        if hasattr(mycpp_module, "cluster_poses") and not hasattr(mycpp_pkg, "cluster_poses"):
            mycpp_pkg.cluster_poses = mycpp_module.cluster_poses
        return
    except Exception:
        pass

    def _cluster_poses(angle_bin, max_keep, poses, symmetry_tfs):
        logging.info("mycpp is unavailable; using Python fallback for cluster_poses (no-op)")
        return poses

    mycpp_pkg = sys.modules.get("mycpp")
    if mycpp_pkg is None:
        mycpp_pkg = types.ModuleType("mycpp")
        mycpp_pkg.__path__ = []

    build_pkg = types.ModuleType("mycpp.build")
    build_pkg.__path__ = []
    mycpp_module = types.ModuleType("mycpp.build.mycpp")
    mycpp_module.cluster_poses = _cluster_poses
    mycpp_pkg.cluster_poses = _cluster_poses
    build_pkg.mycpp = mycpp_module
    mycpp_pkg.build = build_pkg

    sys.modules["mycpp"] = mycpp_pkg
    sys.modules["mycpp.build"] = build_pkg
    sys.modules["mycpp.build.mycpp"] = mycpp_module


def _identity_depth_filter(depth, *, device: str):
    try:
        import numpy as np
        import torch
    except Exception:
        return depth

    if isinstance(depth, np.ndarray):
        return depth
    return torch.as_tensor(depth, dtype=torch.float, device=device)


def _patch_foundationpose_utils() -> None:
    utils_module = importlib.import_module("Utils")

    if not hasattr(utils_module, "bilateral_filter_depth"):
        def bilateral_filter_depth(depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device="cuda"):
            logging.info("warp is unavailable; using identity fallback for bilateral_filter_depth")
            return _identity_depth_filter(depth, device=device)

        utils_module.bilateral_filter_depth = bilateral_filter_depth

    if not hasattr(utils_module, "erode_depth"):
        def erode_depth(depth, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, zfar=100, device="cuda"):
            logging.info("warp is unavailable; using identity fallback for erode_depth")
            return _identity_depth_filter(depth, device=device)

        utils_module.erode_depth = erode_depth

    compute_crop_window_tf_batch = getattr(utils_module, "compute_crop_window_tf_batch", None)
    if compute_crop_window_tf_batch is not None and not getattr(compute_crop_window_tf_batch, "_map4d_compat", False):
        def _compute_crop_window_tf_batch_compat(*args, **kwargs):
            import torch

            bound = {}
            if args:
                param_names = ("pts", "H", "W", "poses", "K", "crop_ratio", "out_size", "rgb", "uvs", "method", "mesh_diameter")
                bound.update(zip(param_names, args))
            bound.update(kwargs)

            poses = bound.get("poses")
            K = bound.get("K")
            pts = bound.get("pts")
            crop_ratio = bound.get("crop_ratio")
            mesh_diameter = bound.get("mesh_diameter")
            if poses is not None:
                poses = torch.as_tensor(poses, dtype=torch.float32, device="cuda")
                bound["poses"] = poses
            if K is not None:
                device = poses.device if poses is not None else "cuda"
                bound["K"] = torch.as_tensor(K, dtype=torch.float32, device=device)
            if pts is not None:
                device = bound["K"].device if bound.get("K") is not None else (poses.device if poses is not None else "cuda")
                dtype = bound["K"].dtype if bound.get("K") is not None else torch.float32
                bound["pts"] = torch.as_tensor(pts, dtype=dtype, device=device)
            if crop_ratio is not None:
                bound["crop_ratio"] = float(crop_ratio)
            if mesh_diameter is not None:
                bound["mesh_diameter"] = float(mesh_diameter)

            return compute_crop_window_tf_batch(**bound)

        _compute_crop_window_tf_batch_compat._map4d_compat = True
        utils_module.compute_crop_window_tf_batch = _compute_crop_window_tf_batch_compat


def _prepare_foundationpose_imports() -> None:
    _install_transformations_compat()
    _install_ruamel_yaml_compat()
    _install_mycpp_compat()


def _prepare_foundationpose_runtime() -> None:
    _prepare_foundationpose_imports()
    _patch_foundationpose_utils()


def _ensure_repo_on_sys_path(repo_root: pathlib.Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    _prepare_foundationpose_imports()


def _resolve_optional_path(path: Optional[PathLike]) -> Optional[pathlib.Path]:
    if path is None:
        return None
    return pathlib.Path(path).expanduser().resolve()


def _require_existing_file(path: pathlib.Path, desc: str) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{desc} is not a file: {path}")
    return path


def _require_existing_dir(path: pathlib.Path, desc: str) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{desc} is not a directory: {path}")
    return path


def _resolve_weights_root(weights_root: Optional[PathLike]) -> pathlib.Path:
    resolved = _resolve_optional_path(weights_root)
    if resolved is not None:
        return resolved
    return default_foundationpose_weights_root()


def _resolve_checkpoint_spec(
    repo_root: pathlib.Path,
    *,
    weights_root: Optional[PathLike],
    run_name: str,
    ckpt_path: Optional[PathLike],
    config_path: Optional[PathLike],
) -> FoundationPoseCheckpointSpec:
    resolved_ckpt = _resolve_optional_path(ckpt_path)
    resolved_cfg = _resolve_optional_path(config_path)

    if resolved_ckpt is not None and resolved_cfg is not None:
        return FoundationPoseCheckpointSpec(
            config_path=_require_existing_file(resolved_cfg, "FoundationPose config"),
            ckpt_path=_require_existing_file(resolved_ckpt, "FoundationPose checkpoint"),
        )

    _require_existing_dir(repo_root, "FoundationPose repo root")
    weights_root_path = _require_existing_dir(
        _resolve_weights_root(weights_root),
        "FoundationPose weights root",
    )

    run_dir = _require_existing_dir(weights_root_path / run_name, "FoundationPose run directory")
    if resolved_cfg is None:
        resolved_cfg = run_dir / "config.yml"
    if resolved_ckpt is None:
        resolved_ckpt = run_dir / "model_best.pth"

    return FoundationPoseCheckpointSpec(
        config_path=_require_existing_file(resolved_cfg, "FoundationPose config"),
        ckpt_path=_require_existing_file(resolved_ckpt, "FoundationPose checkpoint"),
    )


def _check_runtime_dependencies() -> None:
    _prepare_foundationpose_imports()
    missing = []
    for module_name in (
        "torch",
        "omegaconf",
        "trimesh",
        "kornia",
        "nvdiffrast.torch",
        "transformations",
        "pytorch3d",
        "open3d",
        "cv2",
        "yaml",
    ):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        missing_str = ", ".join(missing)
        raise ImportError(
            "FoundationPose dependencies are missing. "
            f"Install or fix these modules first: {missing_str}"
        )


def _find_missing_dependencies(module_names: tuple[str, ...]) -> tuple[str, ...]:
    _prepare_foundationpose_imports()
    missing = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return tuple(missing)


def _normalize_cuda_device(device: str, *, require_available: bool = True) -> str:
    try:
        import torch
    except Exception as exc:
        raise ImportError("torch is required to validate the FoundationPose device.") from exc

    normalized = str(device).strip().lower()
    if normalized == "cuda":
        normalized = "cuda:0"

    if not normalized.startswith("cuda"):
        raise ValueError(
            "FoundationPose currently supports CUDA only; "
            f"received device={device!r}."
        )

    if require_available and not torch.cuda.is_available():
        raise RuntimeError(
            f"FoundationPose requires CUDA, but torch.cuda.is_available() is False (device={normalized!r})."
        )

    if ":" in normalized:
        try:
            index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device string: {device!r}") from exc
        device_count = torch.cuda.device_count()
        if require_available and (index < 0 or index >= device_count):
            raise ValueError(
                f"CUDA device index out of range: {normalized!r}; available device count={device_count}."
            )

    return normalized


class _ScorePredictorFromSpec:
    def __init__(self, checkpoint: FoundationPoseCheckpointSpec, device: str, amp: bool = True):
        _check_runtime_dependencies()
        _prepare_foundationpose_runtime()
        from omegaconf import OmegaConf
        import numpy as np
        import torch

        from learning.datasets.h5_dataset import ScoreMultiPairH5Dataset
        from learning.models.score_network import ScoreNetMultiPair

        self.amp = amp
        self.checkpoint = checkpoint
        self.device = device
        self.cfg = OmegaConf.load(str(checkpoint.config_path))
        self.cfg["ckpt_dir"] = str(checkpoint.ckpt_path)
        self.cfg["enable_amp"] = bool(amp)

        if "use_normal" not in self.cfg:
            self.cfg["use_normal"] = False
        if "use_BN" not in self.cfg:
            self.cfg["use_BN"] = False
        if "zfar" not in self.cfg:
            self.cfg["zfar"] = np.inf
        if "c_in" not in self.cfg:
            self.cfg["c_in"] = 4
        if "normalize_xyz" not in self.cfg:
            self.cfg["normalize_xyz"] = False
        if "crop_ratio" not in self.cfg or self.cfg["crop_ratio"] is None:
            self.cfg["crop_ratio"] = 1.2

        self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode="test", h5_file=None, max_num_key=1)
        self.model = ScoreNetMultiPair(cfg=self.cfg, c_in=self.cfg["c_in"]).to(self.device)

        ckpt = torch.load(str(checkpoint.ckpt_path), map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()

    def predict(self, *args, **kwargs):
        from learning.training.predict_score import ScorePredictor

        return ScorePredictor.predict(self, *args, **kwargs)


class _PoseRefinePredictorFromSpec:
    def __init__(self, checkpoint: FoundationPoseCheckpointSpec, device: str, amp: bool = True):
        _check_runtime_dependencies()
        _prepare_foundationpose_runtime()
        from omegaconf import OmegaConf
        import numpy as np
        import torch

        from learning.datasets.h5_dataset import PoseRefinePairH5Dataset
        from learning.models.refine_network import RefineNet

        self.amp = amp
        self.checkpoint = checkpoint
        self.device = device
        self.cfg = OmegaConf.load(str(checkpoint.config_path))
        self.cfg["ckpt_dir"] = str(checkpoint.ckpt_path)
        self.cfg["enable_amp"] = bool(amp)

        if "use_normal" not in self.cfg:
            self.cfg["use_normal"] = False
        if "use_mask" not in self.cfg:
            self.cfg["use_mask"] = False
        if "use_BN" not in self.cfg:
            self.cfg["use_BN"] = False
        if "c_in" not in self.cfg:
            self.cfg["c_in"] = 4
        if "crop_ratio" not in self.cfg or self.cfg["crop_ratio"] is None:
            self.cfg["crop_ratio"] = 1.2
        if "n_view" not in self.cfg:
            self.cfg["n_view"] = 1
        if "trans_rep" not in self.cfg:
            self.cfg["trans_rep"] = "tracknet"
        if "rot_rep" not in self.cfg:
            self.cfg["rot_rep"] = "axis_angle"
        if "zfar" not in self.cfg:
            self.cfg["zfar"] = 3
        if "normalize_xyz" not in self.cfg:
            self.cfg["normalize_xyz"] = False
        if isinstance(self.cfg["zfar"], str) and "inf" in self.cfg["zfar"].lower():
            self.cfg["zfar"] = np.inf
        if "normal_uint8" not in self.cfg:
            self.cfg["normal_uint8"] = False

        self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file="", mode="test")
        self.model = RefineNet(cfg=self.cfg, c_in=self.cfg["c_in"]).to(self.device)

        ckpt = torch.load(str(checkpoint.ckpt_path), map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()
        self.last_trans_update = None
        self.last_rot_update = None

    def predict(self, *args, **kwargs):
        from learning.training.predict_pose_refine import PoseRefinePredictor

        return PoseRefinePredictor.predict(self, *args, **kwargs)


class FoundationPoseLoader:
    """Small validated wrapper around the vendored FoundationPose codebase.

    Default weight layout (unless explicitly overridden via ``weights_root``):
      <loader package>/weights/
        ├── 2024-01-11-20-02-45/
        │   ├── config.yml
        │   └── model_best.pth
        └── 2023-10-28-18-33-37/
            ├── config.yml
            └── model_best.pth

    This loader intentionally keeps the public API narrow:
    - resolve_* methods only validate filesystem/runtime state
    - load_* methods instantiate scorer/refiner/estimator lazily
    - errors are raised early with concrete missing paths/devices
    """

    def __init__(
        self,
        foundationpose_root: Optional[PathLike] = None,
        weights_root: Optional[PathLike] = None,
        scorer_run_name: str = DEFAULT_SCORER_RUN_NAME,
        refiner_run_name: str = DEFAULT_REFINER_RUN_NAME,
        scorer_ckpt_path: Optional[PathLike] = None,
        scorer_config_path: Optional[PathLike] = None,
        refiner_ckpt_path: Optional[PathLike] = None,
        refiner_config_path: Optional[PathLike] = None,
        device: str = "cuda",
        amp: bool = True,
    ):
        self.foundationpose_root = _resolve_optional_path(foundationpose_root) or _foundationpose_repo_root()
        self.weights_root = _resolve_weights_root(weights_root)
        self.scorer_run_name = scorer_run_name
        self.refiner_run_name = refiner_run_name
        self.scorer_ckpt_path = _resolve_optional_path(scorer_ckpt_path)
        self.scorer_config_path = _resolve_optional_path(scorer_config_path)
        self.refiner_ckpt_path = _resolve_optional_path(refiner_ckpt_path)
        self.refiner_config_path = _resolve_optional_path(refiner_config_path)
        self.device = _normalize_cuda_device(device)
        self.amp = bool(amp)
        self._cached_scorer = None
        self._cached_refiner = None
        self._cached_glctx = None

        _require_existing_dir(self.foundationpose_root, "FoundationPose repo root")
        _ensure_repo_on_sys_path(self.foundationpose_root)

    def resolve_scorer_checkpoint(self) -> FoundationPoseCheckpointSpec:
        return _resolve_checkpoint_spec(
            self.foundationpose_root,
            weights_root=self.weights_root,
            run_name=self.scorer_run_name,
            ckpt_path=self.scorer_ckpt_path,
            config_path=self.scorer_config_path,
        )

    def resolve_refiner_checkpoint(self) -> FoundationPoseCheckpointSpec:
        return _resolve_checkpoint_spec(
            self.foundationpose_root,
            weights_root=self.weights_root,
            run_name=self.refiner_run_name,
            ckpt_path=self.refiner_ckpt_path,
            config_path=self.refiner_config_path,
        )

    def resolve_runtime(self) -> FoundationPoseRuntimeSpec:
        return FoundationPoseRuntimeSpec(
            repo_root=self.foundationpose_root,
            device=self.device,
            weights_root=self.weights_root,
            scorer=self.resolve_scorer_checkpoint(),
            refiner=self.resolve_refiner_checkpoint(),
        )

    def validate(self, *, check_runtime: bool = False) -> FoundationPoseValidationReport:
        return validate_foundationpose_loader(
            foundationpose_root=self.foundationpose_root,
            weights_root=self.weights_root,
            scorer_run_name=self.scorer_run_name,
            refiner_run_name=self.refiner_run_name,
            scorer_ckpt_path=self.scorer_ckpt_path,
            scorer_config_path=self.scorer_config_path,
            refiner_ckpt_path=self.refiner_ckpt_path,
            refiner_config_path=self.refiner_config_path,
            device=self.device,
            check_runtime=check_runtime,
        )

    def load_scorer(self):
        if self._cached_scorer is None:
            runtime = self.resolve_runtime()
            self._cached_scorer = _ScorePredictorFromSpec(
                runtime.scorer,
                device=runtime.device,
                amp=self.amp,
            )
        return self._cached_scorer

    def load_refiner(self):
        if self._cached_refiner is None:
            runtime = self.resolve_runtime()
            self._cached_refiner = _PoseRefinePredictorFromSpec(
                runtime.refiner,
                device=runtime.device,
                amp=self.amp,
            )
        return self._cached_refiner

    def create_glctx(self):
        _prepare_foundationpose_imports()
        if self._cached_glctx is None:
            _check_runtime_dependencies()
            dr = importlib.import_module("nvdiffrast.torch")
            self._cached_glctx = dr.RasterizeCudaContext(self.device)
        return self._cached_glctx

    def load_estimator(
        self,
        *,
        mesh,
        model_points=None,
        model_normals=None,
        symmetry_tfs=None,
        debug: int = 0,
        debug_dir: Optional[PathLike] = None,
        glctx=None,
    ):
        _check_runtime_dependencies()
        _prepare_foundationpose_runtime()
        import trimesh

        from estimater import FoundationPose

        if isinstance(mesh, (str, pathlib.Path)):
            mesh_path = pathlib.Path(mesh).expanduser().resolve()
            _require_existing_file(mesh_path, "FoundationPose mesh")
            mesh = trimesh.load(str(mesh_path))

        if model_points is None:
            model_points = mesh.vertices
        if model_normals is None:
            if getattr(mesh, "vertex_normals", None) is None or len(mesh.vertex_normals) == 0:
                raise ValueError("mesh.vertex_normals is empty; please provide model_normals explicitly.")
            model_normals = mesh.vertex_normals

        scorer = self.load_scorer()
        refiner = self.load_refiner()
        if glctx is None:
            glctx = self.create_glctx()

        resolved_debug_dir = _resolve_optional_path(debug_dir)
        if resolved_debug_dir is None:
            resolved_debug_dir = self.foundationpose_root / "debug"
        os.makedirs(resolved_debug_dir, exist_ok=True)

        estimator = FoundationPose(
            model_pts=model_points,
            model_normals=model_normals,
            symmetry_tfs=symmetry_tfs,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=debug,
            debug_dir=str(resolved_debug_dir),
        )
        estimator.to_device(self.device)
        return estimator


def validate_foundationpose_loader(
    *,
    foundationpose_root: Optional[PathLike] = None,
    weights_root: Optional[PathLike] = None,
    scorer_run_name: str = DEFAULT_SCORER_RUN_NAME,
    refiner_run_name: str = DEFAULT_REFINER_RUN_NAME,
    scorer_ckpt_path: Optional[PathLike] = None,
    scorer_config_path: Optional[PathLike] = None,
    refiner_ckpt_path: Optional[PathLike] = None,
    refiner_config_path: Optional[PathLike] = None,
    device: str = "cuda",
    check_runtime: bool = False,
) -> FoundationPoseValidationReport:
    """Validate FoundationPose loader configuration and readiness.

    ``check_runtime=False`` performs a filesystem/code-path smoke validation only:
    - repo root exists
    - default/overridden weights layout resolves correctly
    - expected config/checkpoint files exist
    - CUDA device string is syntactically acceptable

    ``check_runtime=True`` additionally validates Python runtime dependencies and live CUDA
    availability. Full estimator execution is intentionally not attempted here because that
    also requires usable CUDA kernels, nvdiffrast, and object-specific mesh/assets.
    """

    repo_root = _resolve_optional_path(foundationpose_root) or _foundationpose_repo_root()
    repo_root = _require_existing_dir(repo_root, "FoundationPose repo root")
    resolved_weights_root = _resolve_weights_root(weights_root)
    normalized_device = _normalize_cuda_device(device, require_available=check_runtime)

    scorer = _resolve_checkpoint_spec(
        repo_root,
        weights_root=resolved_weights_root,
        run_name=scorer_run_name,
        ckpt_path=scorer_ckpt_path,
        config_path=scorer_config_path,
    )
    refiner = _resolve_checkpoint_spec(
        repo_root,
        weights_root=resolved_weights_root,
        run_name=refiner_run_name,
        ckpt_path=refiner_ckpt_path,
        config_path=refiner_config_path,
    )

    checked_dependencies: tuple[str, ...] = ()
    missing_dependencies: tuple[str, ...] = ()
    notes = [
        f"Default weights root is {default_foundationpose_weights_root()}",
        "Smoke validation does not instantiate scorer/refiner networks or the estimator.",
        "Full runtime validation still needs importable FoundationPose dependencies, CUDA, and object mesh/assets.",
    ]

    if check_runtime:
        checked_dependencies = (
            "torch",
            "omegaconf",
            "trimesh",
            "kornia",
            "nvdiffrast.torch",
            "transformations",
            "pytorch3d",
            "open3d",
            "cv2",
            "yaml",
        )
        missing_dependencies = _find_missing_dependencies(checked_dependencies)
        if missing_dependencies:
            raise ImportError(
                "FoundationPose runtime validation failed; missing dependencies: "
                + ", ".join(missing_dependencies)
            )
        notes.append("Runtime validation confirms dependency imports and live CUDA visibility.")

    return FoundationPoseValidationReport(
        ok=True,
        repo_root=repo_root,
        default_weights_root=default_foundationpose_weights_root(),
        resolved_weights_root=resolved_weights_root,
        device=normalized_device,
        scorer=scorer,
        refiner=refiner,
        checked_dependencies=checked_dependencies,
        missing_dependencies=missing_dependencies,
        runtime_ready=check_runtime and not missing_dependencies,
        notes=tuple(notes),
    )


def load_foundationpose_estimator(
    *,
    mesh,
    foundationpose_root: Optional[PathLike] = None,
    weights_root: Optional[PathLike] = None,
    scorer_run_name: str = DEFAULT_SCORER_RUN_NAME,
    refiner_run_name: str = DEFAULT_REFINER_RUN_NAME,
    scorer_ckpt_path: Optional[PathLike] = None,
    scorer_config_path: Optional[PathLike] = None,
    refiner_ckpt_path: Optional[PathLike] = None,
    refiner_config_path: Optional[PathLike] = None,
    device: str = "cuda",
    amp: bool = True,
    symmetry_tfs=None,
    debug: int = 0,
    debug_dir: Optional[PathLike] = None,
    glctx=None,
):
    loader = FoundationPoseLoader(
        foundationpose_root=foundationpose_root,
        weights_root=weights_root,
        scorer_run_name=scorer_run_name,
        refiner_run_name=refiner_run_name,
        scorer_ckpt_path=scorer_ckpt_path,
        scorer_config_path=scorer_config_path,
        refiner_ckpt_path=refiner_ckpt_path,
        refiner_config_path=refiner_config_path,
        device=device,
        amp=amp,
    )
    return loader.load_estimator(
        mesh=mesh,
        symmetry_tfs=symmetry_tfs,
        debug=debug,
        debug_dir=debug_dir,
        glctx=glctx,
    )
