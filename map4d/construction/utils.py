from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MAPS4D_DIR = _REPO_ROOT / "map4d" / "representation" / "maps4d"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_MAPS4D_DIR) not in sys.path:
    sys.path.insert(0, str(_MAPS4D_DIR))


def load_map_metadata_for_task(task_name: str) -> dict[str, Any]:
    try:
        from map4d.representation.maps4d.metadata import TASK_METADATA_FILES
    except ImportError:
        from metadata import TASK_METADATA_FILES

    filename = TASK_METADATA_FILES.get(task_name)
    if filename is None:
        raise KeyError(f"No Map4D metadata JSON registered for task {task_name!r}")
    metadata_path = _MAPS4D_DIR / filename
    if not metadata_path.exists():
        raise FileNotFoundError(f"Map metadata JSON not found for task {task_name!r}: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if task_name not in metadata:
        raise KeyError(f"{metadata_path} does not contain top-level task key {task_name!r}")
    return metadata[task_name]


def map_dims_from_metadata(task_metadata: Mapping[str, Any]) -> dict[str, int]:
    size_dim = int(task_metadata["size_parameters"]["dim"])
    objects = task_metadata.get("objects", {})
    if not objects:
        actor_count = len(task_metadata.get("actor_names", []))
        pos_dim = actor_count * 3
        rot_dim = actor_count * 6
    else:
        pos_dim = max(int(obj_info["position_slice"][1]) for obj_info in objects.values())
        rot_dim = max(int(obj_info["rotation_slice"][1]) for obj_info in objects.values())
    return {
        "size": size_dim,
        "position": pos_dim,
        "rotation": rot_dim,
        "size_end": size_dim,
        "position_end": size_dim + pos_dim,
        "rotation_end": size_dim + pos_dim + rot_dim,
        "total": size_dim + pos_dim + rot_dim,
    }


def default_parameter_values_for_task(task_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    task_metadata = load_map_metadata_for_task(task_name)
    size = task_metadata["size_parameters"]
    relation = task_metadata["relation_parameters"]
    size_values = tuple(float(v) for v in size.get("default", ()))
    relation_values = tuple(float(v) for v in relation.get("default", ()))
    if len(size_values) != int(size["dim"]):
        raise ValueError(f"{task_name}: size parameter dim does not match default length.")
    if len(relation_values) != int(relation["dim"]):
        raise ValueError(f"{task_name}: relation parameter dim does not match default length.")
    return size_values, relation_values


def default_size_parameters_for_map_class(
    map_class: type,
    *,
    task_name: Optional[str] = None,
    device: str = "cpu",
) -> torch.Tensor:
    if task_name is None:
        raise ValueError("task_name must be provided.")
    size_values, _ = default_parameter_values_for_task(task_name)
    return torch.tensor([size_values], dtype=torch.float32, device=device)


def default_actor_names_for_map_class(map_class: type, *, task_name: Optional[str] = None) -> tuple[str, ...]:
    if task_name is None:
        raise ValueError("task_name must be provided.")
    task_metadata = load_map_metadata_for_task(task_name)
    actor_names = task_metadata.get("actor_names")
    if not actor_names:
        raise KeyError(f"actor_names is missing for {map_class.__name__}.")
    return tuple(str(name) for name in actor_names)


def feature_dim_from_encoder(point_cloud_encoder: nn.Module) -> int:
    for attr in ("feature_dim", "output_dim"):
        value = getattr(point_cloud_encoder, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("point_cloud_encoder must expose feature_dim or output_dim.")


def lookup_by_object_key(mapping: Mapping[Any, Any], obj, object_index: int, prompt: str):
    for key in (object_index, prompt, getattr(obj, "prompt", None), getattr(obj, "Object_Prompt", None), obj.__class__.__name__):
        if key is not None and key in mapping:
            return mapping[key]
    return None


def object_prompt(obj, object_index: int) -> str:
    for attr in ("prompt", "Object_Prompt", "semantic"):
        value = getattr(obj, attr, None)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return f"object_{object_index}"


def camera_intrinsics_for_frame(camera_intrinsics, frame_idx: int) -> np.ndarray:
    K = np.asarray(camera_intrinsics, dtype=np.float32)
    return K[frame_idx] if K.ndim == 3 else K


def as_rgb_uint8(rgb) -> np.ndarray:
    rgb_np = np.asarray(rgb)
    if rgb_np.ndim != 3 or rgb_np.shape[-1] not in (3, 4):
        raise ValueError(f"Expected rgb shape [H, W, 3/4], got {rgb_np.shape}")
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]
    if np.issubdtype(rgb_np.dtype, np.floating):
        if float(np.nanmax(rgb_np)) <= 1.0:
            rgb_np = rgb_np * 255.0
        rgb_np = np.clip(rgb_np, 0, 255)
    return rgb_np.astype(np.uint8)


def as_rgb_frames_uint8(rgb_frames) -> np.ndarray:
    arr = np.asarray(rgb_frames)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected rgb_frames shape [T, H, W, 3/4], got {arr.shape}")
    return np.stack([as_rgb_uint8(frame) for frame in arr], axis=0)


def as_depth_float32(depth) -> np.ndarray:
    depth_np = np.asarray(depth)
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    if depth_np.ndim != 2:
        raise ValueError(f"Expected depth shape [H, W] or [H, W, 1], got {depth_np.shape}")
    depth_np = depth_np.astype(np.float32)
    if np.nanmax(depth_np) > 100.0:
        depth_np = depth_np / 1000.0
    return depth_np


def as_depth_frames_float32(depth_frames) -> np.ndarray:
    arr = np.asarray(depth_frames)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 3 or (arr.ndim == 4 and arr.shape[-1] == 1):
        return np.stack([as_depth_float32(frame) for frame in arr], axis=0)
    raise ValueError(f"Expected depth_frames shape [T, H, W] or [T, H, W, 1], got {arr.shape}")


def as_mask_bool(mask, image_hw: tuple[int, int], prompt: str) -> np.ndarray:
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
        mask_np = mask_np[..., 0]
    if mask_np.shape != image_hw:
        raise ValueError(f"Mask for prompt={prompt!r} has shape {mask_np.shape}, expected {image_hw}")
    return mask_np.astype(bool)


def scalar_from_tensor_like(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr[0])


def masked_point_cloud_from_depth(depth: np.ndarray, mask: np.ndarray, camera_intrinsics: np.ndarray) -> np.ndarray:
    if camera_intrinsics.shape != (3, 3):
        raise ValueError(f"Expected camera_intrinsics shape [3, 3], got {camera_intrinsics.shape}")
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[v, u].astype(np.float32)
    fx, fy = float(camera_intrinsics[0, 0]), float(camera_intrinsics[1, 1])
    cx, cy = float(camera_intrinsics[0, 2]), float(camera_intrinsics[1, 2])
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_point_cloud(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape [N, 3], got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("Cannot sample an empty point cloud.")
    if num_points <= 0 or points.shape[0] == num_points:
        return points
    replace = points.shape[0] < num_points
    indices = rng.choice(points.shape[0], size=num_points, replace=replace)
    return points[indices].astype(np.float32)


def as_structural_params_numpy(params):
    if params is None:
        return None
    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()
    return np.asarray(params, dtype=np.float32)


def looks_like_map(obj) -> bool:
    return obj is not None and (hasattr(obj, "Objects") or hasattr(obj, "objects"))


def copy_map_attributes(dst, src) -> None:
    for attr in ("Objects", "objects", "Nodes", "Node", "Edges", "Edge", "object_node_slices", "Subgraph_Prompts"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def split_structural_estimator_output(estimator_output):
    if isinstance(estimator_output, Mapping):
        scene_map = estimator_output.get("scene_map")
        params = estimator_output.get("params")
        return scene_map, as_structural_params_numpy(params)
    if looks_like_map(estimator_output):
        params = None
        for attr in ("pred_structural_params", "pred_map_parameters", "structural_params"):
            if hasattr(estimator_output, attr):
                params = getattr(estimator_output, attr)
                break
        return estimator_output, as_structural_params_numpy(params)
    return None, as_structural_params_numpy(estimator_output)


def map_pose_tensors(map4d, *, dtype, device):
    positions = []
    rotations = []
    for obj in getattr(map4d, "Objects", []):
        nodes = getattr(obj, "Nodes", [])
        if len(nodes) == 0:
            continue
        node = nodes[0]
        positions.append(torch.as_tensor(node.position, dtype=dtype, device=device))
        rotations.append(torch.as_tensor(node.rotation, dtype=dtype, device=device))
    if len(positions) == 0:
        return None, None
    return torch.cat(positions, dim=1), torch.cat(rotations, dim=1)


def set_tensor_like_scalar(obj, attr: str, value: float) -> None:
    current = getattr(obj, attr)
    if hasattr(current, "new_full"):
        setattr(obj, attr, current.new_full(current.shape, float(value)))
    else:
        setattr(obj, attr, float(value))


def apply_structural_params_to_map(map4d, structural_params: np.ndarray) -> None:
    params = np.asarray(structural_params, dtype=np.float32)
    if params.ndim != 2 or params.shape[0] < 1:
        raise ValueError(f"Expected structural_params shape [B, D], got {params.shape}")

    objects = list(getattr(map4d, "Objects", []))
    expected_dim = len(objects) * 3
    if params.shape[1] < expected_dim:
        raise ValueError(f"structural_params dim {params.shape[1]} is too small for {len(objects)} objects.")

    for object_index, obj in enumerate(objects):
        nodes = getattr(obj, "Nodes", [])
        if len(nodes) == 0:
            continue
        node = nodes[0]
        height, length, width = params[0, object_index * 3 : object_index * 3 + 3]
        set_tensor_like_scalar(node, "height", height)
        set_tensor_like_scalar(node, "top_length", length)
        set_tensor_like_scalar(node, "top_width", width)
        if hasattr(node, "bottom_length"):
            set_tensor_like_scalar(node, "bottom_length", length)
        if hasattr(node, "bottom_width"):
            set_tensor_like_scalar(node, "bottom_width", width)
        if hasattr(node, "back_height"):
            set_tensor_like_scalar(node, "back_height", height)


def build_template_map_for_class(map_class: type, *, task_name: Optional[str] = None, device: str = "cpu"):
    if task_name is None:
        raise ValueError("task_name must be provided.")
    task_metadata = load_map_metadata_for_task(task_name)
    dims = map_dims_from_metadata(task_metadata)
    size_dim = dims["size"]
    pos_dim = dims["position"]
    rot_dim = dims["rotation"]
    sizes = default_size_parameters_for_map_class(map_class, task_name=task_name, device=device)
    if sizes.shape[1] != size_dim:
        raise ValueError(f"Map metadata size dim mismatch for {map_class.__name__}: {sizes.shape[1]} vs {size_dim}")
    positions = torch.zeros((1, pos_dim), dtype=torch.float32, device=device)
    rotations = torch.zeros((1, rot_dim), dtype=torch.float32, device=device)
    if rot_dim % 6 == 0:
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        for i in range(rot_dim // 6):
            rotations[:, i * 6 : (i + 1) * 6] = identity_6d
    return map_class(sizes, positions, rotations, None, preprocess=False)


def build_stackcube_template_map(*, sizes=None, positions=None, rotations=None, device: str = "cpu"):
    from maniskill_stackcube import Map4d_StackCube

    if sizes is None:
        sizes = default_size_parameters_for_map_class(Map4d_StackCube, task_name="StackCube-v1", device=device)
    positions = torch.zeros((1, 9), dtype=torch.float32, device=device) if positions is None else positions
    if rotations is None:
        rotations = torch.zeros((1, 18), dtype=torch.float32, device=device)
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        for i in range(3):
            rotations[:, i * 6 : (i + 1) * 6] = identity_6d
    return Map4d_StackCube(sizes, positions, rotations, clip_model=None, preprocess=False)


def instantiate_stackcube_map(
    *,
    rgb,
    depth,
    camera_intrinsics=None,
    object_masks: Optional[Mapping[Any, Any]] = None,
    object_meshes: Optional[Mapping[Any, Any]] = None,
    grounded_sam2_loader=None,
    foundationpose_loader=None,
    structural_parameter_estimator=None,
    structural_num_points: int = 2048,
    device: str = "cuda:0",
    foundationpose_debug: int = 0,
    foundationpose_debug_dir: Optional[pathlib.Path | str] = None,
):
    try:
        from .map_constructor import Map4dSingleFrameConstructor
    except ImportError:
        from map_constructor import Map4dSingleFrameConstructor

    constructor = Map4dSingleFrameConstructor(
        map_template=build_stackcube_template_map(device="cpu"),
        grounded_sam2_loader=grounded_sam2_loader,
        foundationpose_loader=foundationpose_loader,
        structural_parameter_estimator=structural_parameter_estimator,
        structural_num_points=structural_num_points,
        device=device,
        foundationpose_debug=foundationpose_debug,
        foundationpose_debug_dir=foundationpose_debug_dir,
    )
    return constructor.construct(
        rgb=rgb,
        depth=depth,
        camera_intrinsics=camera_intrinsics,
        object_masks=object_masks,
        object_meshes=object_meshes,
    )


def instantiate_stackcube_map_sequence(
    *,
    rgb_frames,
    depth_frames,
    camera_intrinsics=None,
    object_meshes: Optional[Mapping[Any, Any]] = None,
    grounded_sam2_loader=None,
    foundationpose_loader=None,
    structural_parameter_estimator=None,
    structural_num_points: int = 2048,
    device: str = "cuda:0",
    foundationpose_debug: int = 0,
    foundationpose_debug_dir: Optional[pathlib.Path | str] = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.3,
    select_by: str = "grounding_score",
    allow_empty: bool = False,
    start_frame_idx: int = 0,
    max_frame_num_to_track: Optional[int] = None,
    tracking_frames_dir: Optional[pathlib.Path | str] = None,
    foundationpose_refine_iter: int = 3,
):
    try:
        from .map_constructor import Map4dConstructor
    except ImportError:
        from map_constructor import Map4dConstructor

    constructor = Map4dConstructor(
        map_template=build_stackcube_template_map(device="cpu"),
        grounded_sam2_loader=grounded_sam2_loader,
        foundationpose_loader=foundationpose_loader,
        structural_parameter_estimator=structural_parameter_estimator,
        structural_num_points=structural_num_points,
        device=device,
        foundationpose_debug=foundationpose_debug,
        foundationpose_debug_dir=foundationpose_debug_dir,
    )
    return constructor.instantiate_sequence(
        rgb_frames=rgb_frames,
        depth_frames=depth_frames,
        camera_intrinsics=camera_intrinsics,
        object_meshes=object_meshes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        select_by=select_by,
        allow_empty=allow_empty,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        tracking_frames_dir=tracking_frames_dir,
        foundationpose_refine_iter=foundationpose_refine_iter,
    )


def quat_wxyz_to_rotation_6d(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape [T, 4], got {quat.shape}")
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


def normalize_actor_states(actor_states, actor_names, *, frame_indices=None) -> list[np.ndarray]:
    actor_states = [np.asarray(state, dtype=np.float32) for state in actor_states]
    if len(actor_states) != len(actor_names):
        raise ValueError(f"Expected {len(actor_names)} actor states, got {len(actor_states)}.")
    if frame_indices is not None:
        actor_states = [state[frame_indices] for state in actor_states]
    actor_states = [state[None] if state.ndim == 1 else state for state in actor_states]
    frame_count = actor_states[0].shape[0]
    if any(state.shape[0] != frame_count for state in actor_states):
        raise ValueError("All actor state arrays must have the same frame count.")
    if any(state.shape[1] < 7 for state in actor_states):
        shapes = [state.shape for state in actor_states]
        raise ValueError(f"Actor states must contain at least xyz + quaternion dims, got {shapes}.")
    return actor_states


def pose_parameters_from_actor_states(actor_states: list[np.ndarray], *, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.as_tensor(
        np.concatenate([state[:, 0:3] for state in actor_states], axis=1),
        dtype=torch.float32,
        device=device,
    )
    rotations = torch.as_tensor(
        np.concatenate([quat_wxyz_to_rotation_6d(state[:, 3:7]) for state in actor_states], axis=1),
        dtype=torch.float32,
        device=device,
    )
    return positions, rotations


def repeat_parameter_tensor(values, frame_count: int, *, device: str) -> torch.Tensor:
    values = tuple(float(v) for v in values)
    return torch.tensor([values], dtype=torch.float32, device=device).repeat(frame_count, 1)
