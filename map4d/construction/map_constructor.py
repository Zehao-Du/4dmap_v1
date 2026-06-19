from __future__ import annotations

import copy
import pathlib
import sys
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from .utils import (
        apply_structural_params_to_map,
        as_depth_float32,
        as_depth_frames_float32,
        as_mask_bool,
        as_rgb_frames_uint8,
        as_rgb_uint8,
        camera_intrinsics_for_frame,
        build_template_map_for_class,
        copy_map_attributes,
        default_parameter_values_for_task,
        feature_dim_from_encoder,
        load_map_metadata_for_task,
        lookup_by_object_key,
        map_pose_tensors,
        map_dims_from_metadata,
        masked_point_cloud_from_depth,
        normalize_actor_states,
        object_prompt,
        pose_parameters_from_actor_states,
        repeat_parameter_tensor,
        sample_point_cloud,
        scalar_from_tensor_like,
        split_structural_estimator_output,
    )
except ImportError:
    from utils import (
        apply_structural_params_to_map,
        as_depth_float32,
        as_depth_frames_float32,
        as_mask_bool,
        as_rgb_frames_uint8,
        as_rgb_uint8,
        camera_intrinsics_for_frame,
        build_template_map_for_class,
        copy_map_attributes,
        default_parameter_values_for_task,
        feature_dim_from_encoder,
        load_map_metadata_for_task,
        lookup_by_object_key,
        map_pose_tensors,
        map_dims_from_metadata,
        masked_point_cloud_from_depth,
        normalize_actor_states,
        object_prompt,
        pose_parameters_from_actor_states,
        repeat_parameter_tensor,
        sample_point_cloud,
        scalar_from_tensor_like,
        split_structural_estimator_output,
    )


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MAPS4D_DIR = _REPO_ROOT / "map4d" / "representation" / "maps4d"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_MAPS4D_DIR) not in sys.path:
    sys.path.insert(0, str(_MAPS4D_DIR))


from maniskill_stackcube import Map4d_StackCube
from maniskill_plugcharger import Map4d_PlugCharger

class ParameterEstimator_SingleFrame(nn.Module):
    """
    Single-frame structural map estimator.
    Input: single frame point cloud
    Output: 4d map
    """

    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        task_name: str,
        map_class: Optional[type],
        device: str = "cuda:0",
    ):
        super().__init__()
        if map_class is None:
            raise ValueError("map_class must be provided.")
        task_metadata = load_map_metadata_for_task(task_name)
        self.map_name = task_name
        self.dims = map_dims_from_metadata(task_metadata)
        self.device = device
        self.MapClass = map_class
        self.map_metadata = task_metadata
        self.clip_encoder = None
        self.point_cloud_encoder = point_cloud_encoder
        self.estimation_head = nn.Sequential(
            nn.Linear(feature_dim_from_encoder(point_cloud_encoder), self.dims["total"]),
        )

    def _build_scene_map(self, parameters: torch.Tensor, *, preprocess: bool):
        sizes = parameters[:, 0:self.dims["size_end"]]
        positions = parameters[:, self.dims["size_end"]:self.dims["position_end"]]
        rotations = parameters[:, self.dims["position_end"]:self.dims["rotation_end"]]
        return self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=preprocess)

    def forward(self, point_cloud):
        if not isinstance(point_cloud, torch.Tensor):
            point_cloud = torch.as_tensor(point_cloud, dtype=torch.float32, device=self.device)
        else:
            point_cloud = point_cloud.to(self.device, dtype=torch.float32)
        features = self.point_cloud_encoder(point_cloud)
        parameters = self.estimation_head(features)
        scene_map = self._build_scene_map(parameters, preprocess=False)
        scene_map.pred_map_parameters = parameters
        return scene_map


class ParameterEstimator_SingleFrame_Segmentation(ParameterEstimator_SingleFrame):
    """MapPolicy-style RGB-D estimator with text-prompt segmentation.

    The segmentation backend follows the local Grounded-SAM2 loader API. When
    camera intrinsics are unavailable, masked depth is projected into a simple
    normalized image-plane point cloud so the forward path remains usable for
    smoke tests.
    """

    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        task_name: str = "StackCube-v1",
        map_class: Optional[type] = None,
        grounded_sam2_loader=None,
        camera_intrinsics=None,
        num_points: int = 1024,
        device: str = "cuda:0",
    ):
        super().__init__(
            point_cloud_encoder=point_cloud_encoder,
            task_name=task_name,
            map_class=map_class,
            device=device,
        )
        self.grounded_sam2_loader = grounded_sam2_loader
        self.camera_intrinsics = camera_intrinsics
        self.num_points = int(num_points)
        self.subgraph_dict = self._build_subgraph_dict()

    def _build_subgraph_dict(self):
        template_map = build_template_map_for_class(self.MapClass, task_name=self.map_name, device=self.device)
        if hasattr(template_map, "Subgraph_Prompts") and template_map.Subgraph_Prompts is not None:
            subgraph_dict = OrderedDict()
            for i, item in enumerate(template_map.Subgraph_Prompts):
                subgraph_dict[f"subgraph_{i}"] = {
                    "text_prompt": item["text_prompt"],
                    "node_indices": item["node_indices"],
                }
            return subgraph_dict

        subgraph_dict = OrderedDict()
        for i, obj in enumerate(getattr(template_map, "Objects", [])):
            subgraph_dict[f"subgraph_{i}"] = {
                "text_prompt": object_prompt(obj, i),
                "node_indices": [i],
            }
        return subgraph_dict

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _prepare_segmentation_inputs(self, rgb, depth):
        rgb_np = self._to_numpy(rgb)
        depth_np = self._to_numpy(depth)

        if rgb_np.ndim == 3:
            if rgb_np.shape[0] in (1, 3) and rgb_np.shape[-1] not in (1, 3):
                rgb_np = np.moveaxis(rgb_np, 0, -1)
            rgb_np = rgb_np[None, ...]
        elif rgb_np.ndim == 4:
            if rgb_np.shape[1] in (1, 3) and rgb_np.shape[-1] not in (1, 3):
                rgb_np = np.moveaxis(rgb_np, 1, -1)
        else:
            raise ValueError(f"Unsupported rgb shape: {rgb_np.shape}")

        if depth_np.ndim == 2:
            depth_np = depth_np[None, ...]
        elif depth_np.ndim == 4:
            if depth_np.shape[1] == 1:
                depth_np = depth_np[:, 0]
            elif depth_np.shape[-1] == 1:
                depth_np = depth_np[..., 0]
            else:
                raise ValueError(f"Unsupported depth shape: {depth_np.shape}")
        elif depth_np.ndim != 3:
            raise ValueError(f"Unsupported depth shape: {depth_np.shape}")

        if rgb_np.shape[0] != depth_np.shape[0]:
            raise ValueError(f"RGB/depth batch mismatch: {rgb_np.shape} vs {depth_np.shape}")
        return rgb_np, depth_np

    def _segment_single_sample(self, rgb_sample, depth_sample, camera_intrinsics=None):
        if self.grounded_sam2_loader is None:
            raise ValueError("grounded_sam2_loader is required for segmentation forward.")

        rgb_sample = as_rgb_uint8(rgb_sample)
        depth_sample = as_depth_float32(depth_sample)
        subgraph_items = list(self.subgraph_dict.items())
        text_prompts = [item[1]["text_prompt"] for item in subgraph_items]
        segmented = self.grounded_sam2_loader.predict_prompts(rgb_sample, text_prompts, allow_empty=True)

        masks_dict = OrderedDict()
        partial_point_cloud_dict = OrderedDict()
        merged_partial_point_clouds = []
        for subgraph_index, (subgraph_name, subgraph_info) in enumerate(subgraph_items):
            mask = segmented.masks[subgraph_index].astype(bool)
            masks_dict[subgraph_name] = mask
            partial_pc = self._point_cloud_from_depth_mask(depth_sample, mask, camera_intrinsics)
            partial_point_cloud_dict[subgraph_name] = partial_pc
            if partial_pc.shape[0] > 0:
                merged_partial_point_clouds.append(partial_pc)

        if merged_partial_point_clouds:
            merged_pc = np.concatenate(merged_partial_point_clouds, axis=0)
        else:
            merged_pc = np.zeros((1, 3), dtype=np.float32)
        encoded_pc = self._prepare_point_cloud_for_encoder(merged_pc)
        return encoded_pc, masks_dict, partial_point_cloud_dict

    @staticmethod
    def _point_cloud_from_depth_mask(depth, mask, camera_intrinsics=None):
        if camera_intrinsics is not None:
            return masked_point_cloud_from_depth(
                depth,
                mask,
                np.asarray(camera_intrinsics, dtype=np.float32),
            )

        valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
        v, u = np.nonzero(valid)
        if len(u) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        h, w = depth.shape
        x = (u.astype(np.float32) / max(w - 1, 1)) * 2.0 - 1.0
        y = (v.astype(np.float32) / max(h - 1, 1)) * 2.0 - 1.0
        z = depth[v, u].astype(np.float32)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def _prepare_point_cloud_for_encoder(self, points_xyz: np.ndarray):
        points_xyz = np.asarray(points_xyz, dtype=np.float32)
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError(f"Expected point cloud shape [N, 3], got {points_xyz.shape}")
        if points_xyz.shape[0] == 0:
            points_xyz = np.zeros((1, 3), dtype=np.float32)

        replace = points_xyz.shape[0] < self.num_points
        idx = np.random.choice(points_xyz.shape[0], self.num_points, replace=replace)
        points_xyz = points_xyz[idx]
        return torch.from_numpy(points_xyz).unsqueeze(0).to(self.device)

    def extract_segmented_point_cloud_batch(self, rgb, depth, camera_intrinsics=None, keep_debug: bool = False):
        rgb_np, depth_np = self._prepare_segmentation_inputs(rgb, depth)
        K = self.camera_intrinsics if camera_intrinsics is None else camera_intrinsics

        batched_point_clouds = []
        all_masks_dict = []
        all_partial_point_cloud_dict = []
        for sample_idx in range(rgb_np.shape[0]):
            sample_K = camera_intrinsics_for_frame(K, sample_idx) if K is not None else None
            encoded_pc, masks_dict, partial_point_cloud_dict = self._segment_single_sample(
                rgb_np[sample_idx],
                depth_np[sample_idx],
                sample_K,
            )
            batched_point_clouds.append(encoded_pc)
            all_masks_dict.append(masks_dict)
            all_partial_point_cloud_dict.append(partial_point_cloud_dict)

        if keep_debug:
            self.latest_masks_dict = all_masks_dict[0] if len(all_masks_dict) == 1 else all_masks_dict
            self.latest_partial_point_cloud_dict = (
                all_partial_point_cloud_dict[0]
                if len(all_partial_point_cloud_dict) == 1
                else all_partial_point_cloud_dict
            )

        return torch.cat(batched_point_clouds, dim=0)

    def forward_precomputed_point_cloud(self, point_cloud):
        return super().forward(point_cloud)

    def forward(self, rgb, depth, camera_intrinsics=None):
        point_cloud = self.extract_segmented_point_cloud_batch(
            rgb,
            depth,
            camera_intrinsics=camera_intrinsics,
            keep_debug=True,
        )
        return self.forward_precomputed_point_cloud(point_cloud)


@dataclass
class ObjectConstructionResult:
    object_index: int
    prompt: str
    mask: np.ndarray
    pose_6d: Optional[np.ndarray]
    mesh: Any = None
    masks: Optional[np.ndarray] = None
    poses_6d: Optional[np.ndarray] = None
    box_xyxy: Optional[np.ndarray] = None
    grounding_score: Optional[float] = None
    sam_score: Optional[float] = None
    masked_point_cloud: Optional[np.ndarray] = None
    structural_params: Optional[np.ndarray] = None


class Map4dConstructor:
    """Instantiate a template 4D map from RGB-D observations.

    Flow:
      first frame: Grounded-SAM2 masks -> structural estimator sizes
        -> FoundationPose registration;
      later frames: Grounded-SAM2 tracking masks -> FoundationPose tracking.

    Precomputed object_masks can replace Grounded-SAM2 for smoke tests. The old
    direct SAM2 box-refinement path is intentionally removed.
    """

    def __init__(
        self,
        map_template=None,
        *,
        grounded_sam2_loader=None,
        foundationpose_loader=None,
        structural_parameter_estimator=None,
        structural_num_points: int = 2048,
        copy_template: bool = True,
        device: str = "cuda:0",
        foundationpose_debug: int = 0,
        foundationpose_debug_dir: Optional[pathlib.Path | str] = None,
        random_seed: int = 0,
    ):
        self.map_template = map_template
        self.copy_template = bool(copy_template)
        self.grounded_sam2_loader = grounded_sam2_loader
        self.foundationpose_loader = foundationpose_loader
        self.structural_parameter_estimator = structural_parameter_estimator
        self.structural_num_points = int(structural_num_points)
        self.device = device
        self.foundationpose_debug = int(foundationpose_debug)
        self.foundationpose_debug_dir = None if foundationpose_debug_dir is None else pathlib.Path(foundationpose_debug_dir)
        self.rng = np.random.default_rng(int(random_seed))
        self._glctx = None

    def construct(
        self,
        rgb,
        depth,
        *,
        camera_intrinsics=None,
        map_template=None,
        object_masks: Optional[Mapping[Any, Any]] = None,
        object_meshes: Optional[Mapping[Any, Any]] = None,
        foundationpose_refine_iter: int = 3,
    ):
        return self.instantiate(
            rgb=rgb,
            depth=depth,
            camera_intrinsics=camera_intrinsics,
            map_template=map_template,
            object_masks=object_masks,
            object_meshes=object_meshes,
            foundationpose_refine_iter=foundationpose_refine_iter,
        )

    def instantiate(
        self,
        *,
        rgb,
        depth,
        camera_intrinsics=None,
        map_template=None,
        object_masks: Optional[Mapping[Any, Any]] = None,
        object_meshes: Optional[Mapping[Any, Any]] = None,
        foundationpose_refine_iter: int = 3,
    ):
        map4d = self._resolve_map_template(map_template)
        rgb_np = as_rgb_uint8(rgb)
        depth_np = as_depth_float32(depth)
        if rgb_np.shape[:2] != depth_np.shape[:2]:
            raise ValueError(f"RGB/depth shape mismatch: rgb={rgb_np.shape}, depth={depth_np.shape}")

        objects = list(getattr(map4d, "Objects", []))
        prompts = [object_prompt(obj, idx) for idx, obj in enumerate(objects)]
        mask_info = self._segment_first_frame(rgb_np, objects, prompts, object_masks or {})
        point_clouds, structural_params = self._estimate_and_apply_structure(
            map4d=map4d,
            depth=depth_np,
            masks=[item["mask"] for item in mask_info],
            camera_intrinsics=camera_intrinsics,
        )
        objects = list(getattr(map4d, "Objects", []))
        prompts = [object_prompt(obj, idx) for idx, obj in enumerate(objects)]

        object_meshes = object_meshes or {}
        results = []
        for object_index, obj in enumerate(objects):
            prompt = prompts[object_index]
            mask = mask_info[object_index]["mask"]
            obj.mask = mask
            obj.segmentation_mask = mask
            self._write_grounded_sam2_metadata(obj, mask_info[object_index])

            mesh = self._resolve_object_mesh(obj, object_index, prompt, object_meshes)
            pose_6d = self._maybe_register_pose(
                rgb=rgb_np,
                depth=depth_np,
                mask=mask,
                camera_intrinsics=camera_intrinsics,
                mesh=mesh,
                object_index=object_index,
                prompt=prompt,
                refine_iter=foundationpose_refine_iter,
            )
            obj.pose_6d = pose_6d
            results.append(
                ObjectConstructionResult(
                    object_index=object_index,
                    prompt=prompt,
                    mask=mask,
                    pose_6d=pose_6d,
                    mesh=mesh,
                    box_xyxy=mask_info[object_index].get("box_xyxy"),
                    grounding_score=mask_info[object_index].get("grounding_score"),
                    sam_score=mask_info[object_index].get("sam_score"),
                    masked_point_cloud=point_clouds[object_index] if point_clouds is not None else None,
                    structural_params=structural_params,
                )
        )

        self._attach_common_outputs(map4d, rgb_np, depth_np, camera_intrinsics, results)
        return map4d

    def instantiate_sequence(
        self,
        *,
        rgb_frames,
        depth_frames,
        map_template=None,
        camera_intrinsics=None,
        object_meshes: Optional[Mapping[Any, Any]] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.3,
        select_by: str = "grounding_score",
        allow_empty: bool = False,
        start_frame_idx: int = 0,
        max_frame_num_to_track: Optional[int] = None,
        tracking_frames_dir: Optional[pathlib.Path | str] = None,
        foundationpose_refine_iter: int = 3,
    ):
        if self.grounded_sam2_loader is None:
            raise ValueError("grounded_sam2_loader is required for sequence construction.")

        map4d = self._resolve_map_template(map_template)
        rgb_np = as_rgb_frames_uint8(rgb_frames)
        depth_np = as_depth_frames_float32(depth_frames)
        if rgb_np.shape[:3] != depth_np.shape[:3]:
            raise ValueError(f"RGB/depth sequence shape mismatch: rgb={rgb_np.shape}, depth={depth_np.shape}")

        objects = list(getattr(map4d, "Objects", []))
        prompts = [object_prompt(obj, idx) for idx, obj in enumerate(objects)]
        tracked = self.grounded_sam2_loader.track_prompts(
            rgb_np,
            prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            select_by=select_by,
            allow_empty=allow_empty,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            frames_dir=tracking_frames_dir,
        )

        first_masks = [tracked.masks[start_frame_idx, idx] for idx in range(len(objects))]
        first_K = None if camera_intrinsics is None else camera_intrinsics_for_frame(camera_intrinsics, start_frame_idx)
        point_clouds, structural_params = self._estimate_and_apply_structure(
            map4d=map4d,
            depth=depth_np[start_frame_idx],
            masks=first_masks,
            camera_intrinsics=first_K,
        )
        objects = list(getattr(map4d, "Objects", []))
        prompts = [object_prompt(obj, idx) for idx, obj in enumerate(objects)]

        object_meshes = object_meshes or {}
        results = []
        for object_index, obj in enumerate(objects):
            prompt = prompts[object_index]
            masks = tracked.masks[:, object_index]
            mask = masks[start_frame_idx]
            obj.mask = mask
            obj.masks = masks
            obj.segmentation_mask = mask
            obj.segmentation_masks = masks
            obj.box_xyxy = tracked.boxes_xyxy[object_index]
            obj.grounding_score = float(tracked.grounding_scores[object_index])
            obj.sam_score = float(tracked.sam_scores[object_index])

            mesh = self._resolve_object_mesh(obj, object_index, prompt, object_meshes)
            poses_6d = self._maybe_track_pose_sequence(
                rgb_frames=rgb_np,
                depth_frames=depth_np,
                masks=masks,
                camera_intrinsics=camera_intrinsics,
                mesh=mesh,
                object_index=object_index,
                prompt=prompt,
                refine_iter=foundationpose_refine_iter,
                start_frame_idx=start_frame_idx,
            )
            pose_6d = None if poses_6d is None else poses_6d[start_frame_idx]
            obj.pose_6d = pose_6d
            obj.poses_6d = poses_6d
            results.append(
                ObjectConstructionResult(
                    object_index=object_index,
                    prompt=prompt,
                    mask=mask,
                    pose_6d=pose_6d,
                    mesh=mesh,
                    masks=masks,
                    poses_6d=poses_6d,
                    box_xyxy=tracked.boxes_xyxy[object_index],
                    grounding_score=float(tracked.grounding_scores[object_index]),
                    sam_score=float(tracked.sam_scores[object_index]),
                    masked_point_cloud=point_clouds[object_index] if point_clouds is not None else None,
                    structural_params=structural_params,
                )
            )

        self._attach_common_outputs(map4d, rgb_np, depth_np, camera_intrinsics, results)
        map4d.grounded_sam2_result = tracked
        map4d.structural_params = structural_params
        map4d.object_point_clouds = point_clouds
        return map4d

    def _resolve_map_template(self, map_template):
        template = self.map_template if map_template is None else map_template
        if template is None:
            raise ValueError("map_template must be provided at initialization or construct time.")
        return copy.deepcopy(template) if self.copy_template else template

    def _segment_first_frame(self, rgb: np.ndarray, objects: list[Any], prompts: list[str], object_masks: Mapping[Any, Any]):
        manual_masks = [
            lookup_by_object_key(object_masks, obj, object_index, prompts[object_index])
            for object_index, obj in enumerate(objects)
        ]
        if all(mask is not None for mask in manual_masks):
            return [
                {
                    "mask": as_mask_bool(mask, rgb.shape[:2], prompts[idx]),
                    "box_xyxy": None,
                    "grounding_score": None,
                    "sam_score": None,
                }
                for idx, mask in enumerate(manual_masks)
            ]
        if any(mask is not None for mask in manual_masks):
            raise ValueError("object_masks must provide either all objects or none; mixed manual/Grounded-SAM2 masks are ambiguous.")
        if self.grounded_sam2_loader is None:
            raise ValueError("grounded_sam2_loader is required when object_masks are not provided.")

        grounded = self.grounded_sam2_loader.predict_prompts(rgb, prompts, allow_empty=False)
        return [
            {
                "mask": grounded.masks[idx].astype(bool),
                "box_xyxy": grounded.boxes_xyxy[idx],
                "grounding_score": float(grounded.grounding_scores[idx]),
                "sam_score": float(grounded.sam_scores[idx]),
            }
            for idx in range(len(prompts))
        ]

    def _estimate_and_apply_structure(self, *, map4d, depth: np.ndarray, masks: list[np.ndarray], camera_intrinsics):
        if self.structural_parameter_estimator is None:
            return None, None
        if camera_intrinsics is None:
            raise ValueError("camera_intrinsics is required when structural_parameter_estimator is provided.")

        K = np.asarray(camera_intrinsics, dtype=np.float32)
        object_point_clouds = [masked_point_cloud_from_depth(depth, mask, K) for mask in masks]
        valid_clouds = [cloud for cloud in object_point_clouds if cloud.shape[0] > 0]
        if not valid_clouds:
            raise ValueError("No valid depth points inside any object mask; cannot estimate structural parameters.")

        merged = np.concatenate(valid_clouds, axis=0).astype(np.float32)
        sampled = sample_point_cloud(merged, self.structural_num_points, self.rng)
        estimator_output = self._run_structural_estimator(sampled)
        scene_map, structural_params = split_structural_estimator_output(estimator_output)
        if scene_map is not None:
            copy_map_attributes(map4d, scene_map)
        elif structural_params is not None:
            if not self._rebuild_map_from_structural_params(map4d, structural_params):
                apply_structural_params_to_map(map4d, structural_params)
        map4d.structural_params = structural_params
        map4d.object_point_clouds = object_point_clouds
        map4d.masked_point_cloud = sampled
        return object_point_clouds, structural_params

    def _run_structural_estimator(self, point_cloud: np.ndarray):
        model = self.structural_parameter_estimator
        try:
            param = next(model.parameters())
            device = param.device
        except StopIteration:
            device = torch.device(self.device if torch.cuda.is_available() and str(self.device).startswith("cuda") else "cpu")
        points = torch.as_tensor(point_cloud[None], dtype=torch.float32, device=device)
        was_training = getattr(model, "training", False)
        model.eval()
        with torch.no_grad():
            output = model(points)
        if was_training:
            model.train()
        return output

    def _rebuild_map_from_structural_params(self, map4d, structural_params: np.ndarray) -> bool:
        model = self.structural_parameter_estimator
        if model is None or not hasattr(model, "build_map_from_params"):
            return False
        try:
            import torch

            param = next(model.parameters())
            device = param.device
            dtype = param.dtype
            params = torch.as_tensor(structural_params, dtype=dtype, device=device)
            positions, rotations = map_pose_tensors(map4d, dtype=dtype, device=device)
            rebuilt = model.build_map_from_params(params, positions=positions, rotations=rotations, clip_model=None)
        except Exception:
            return False

        copy_map_attributes(map4d, rebuilt)
        return True

    def _maybe_register_pose(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        camera_intrinsics,
        mesh,
        object_index: int,
        prompt: str,
        refine_iter: int,
    ) -> Optional[np.ndarray]:
        if self.foundationpose_loader is None:
            return None
        if camera_intrinsics is None:
            raise ValueError("camera_intrinsics is required when foundationpose_loader is provided.")
        if mesh is None:
            raise ValueError(f"No mesh available for object prompt={prompt!r}; provide object_meshes or a box-like node.")
        return self._estimate_pose_with_foundationpose(
            rgb=rgb,
            depth=depth,
            mask=mask,
            camera_intrinsics=np.asarray(camera_intrinsics, dtype=np.float32),
            mesh=mesh,
            object_index=object_index,
            prompt=prompt,
            refine_iter=refine_iter,
        )

    def _maybe_track_pose_sequence(
        self,
        *,
        rgb_frames: np.ndarray,
        depth_frames: np.ndarray,
        masks: np.ndarray,
        camera_intrinsics,
        mesh,
        object_index: int,
        prompt: str,
        refine_iter: int,
        start_frame_idx: int,
    ) -> Optional[np.ndarray]:
        if self.foundationpose_loader is None:
            return None
        if camera_intrinsics is None:
            raise ValueError("camera_intrinsics is required when foundationpose_loader is provided.")
        if mesh is None:
            raise ValueError(f"No mesh available for object prompt={prompt!r}; provide object_meshes or a box-like node.")
        return self._estimate_pose_sequence_with_foundationpose(
            rgb_frames=rgb_frames,
            depth_frames=depth_frames,
            masks=masks,
            camera_intrinsics=np.asarray(camera_intrinsics, dtype=np.float32),
            mesh=mesh,
            object_index=object_index,
            prompt=prompt,
            refine_iter=refine_iter,
            start_frame_idx=start_frame_idx,
        )

    def _estimate_pose_with_foundationpose(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        camera_intrinsics: np.ndarray,
        mesh,
        object_index: int,
        prompt: str,
        refine_iter: int,
    ) -> np.ndarray:
        if self._glctx is None:
            self._glctx = self.foundationpose_loader.create_glctx()
        estimator = self.foundationpose_loader.load_estimator(
            mesh=mesh,
            debug=self.foundationpose_debug,
            debug_dir=self._foundationpose_object_debug_dir(object_index, prompt),
            glctx=self._glctx,
        )
        pose = estimator.register(
            K=camera_intrinsics,
            rgb=rgb,
            depth=depth,
            ob_mask=mask.astype(np.uint8),
            iteration=int(refine_iter),
        )
        return np.asarray(pose, dtype=np.float32)

    def _estimate_pose_sequence_with_foundationpose(
        self,
        *,
        rgb_frames: np.ndarray,
        depth_frames: np.ndarray,
        masks: np.ndarray,
        camera_intrinsics: np.ndarray,
        mesh,
        object_index: int,
        prompt: str,
        refine_iter: int,
        start_frame_idx: int,
    ) -> np.ndarray:
        if self._glctx is None:
            self._glctx = self.foundationpose_loader.create_glctx()
        estimator = self.foundationpose_loader.load_estimator(
            mesh=mesh,
            debug=self.foundationpose_debug,
            debug_dir=self._foundationpose_object_debug_dir(object_index, prompt),
            glctx=self._glctx,
        )
        poses = np.full((rgb_frames.shape[0], 4, 4), np.nan, dtype=np.float32)
        start_K = camera_intrinsics_for_frame(camera_intrinsics, start_frame_idx)
        start_pose = estimator.register(
            K=start_K,
            rgb=rgb_frames[start_frame_idx],
            depth=depth_frames[start_frame_idx],
            ob_mask=masks[start_frame_idx].astype(np.uint8),
            iteration=int(refine_iter),
        )
        poses[start_frame_idx] = np.asarray(start_pose, dtype=np.float32)
        for frame_idx in range(start_frame_idx + 1, rgb_frames.shape[0]):
            K = camera_intrinsics_for_frame(camera_intrinsics, frame_idx)
            pose = estimator.track_one(
                rgb=rgb_frames[frame_idx],
                depth=depth_frames[frame_idx],
                K=K,
                iteration=int(refine_iter),
            )
            poses[frame_idx] = np.asarray(pose, dtype=np.float32)
        return poses

    def _foundationpose_object_debug_dir(self, object_index: int, prompt: str):
        if self.foundationpose_debug_dir is None:
            return None
        safe_prompt = "".join(ch if ch.isalnum() else "_" for ch in prompt).strip("_") or f"object_{object_index}"
        return self.foundationpose_debug_dir / f"{object_index:02d}_{safe_prompt}"

    def _resolve_object_mesh(self, obj, object_index: int, prompt: str, object_meshes: Mapping[Any, Any]):
        mesh = lookup_by_object_key(object_meshes, obj, object_index, prompt)
        if mesh is not None:
            return mesh
        return self._mesh_from_first_box_node(obj)

    def _mesh_from_first_box_node(self, obj):
        nodes = getattr(obj, "Nodes", [])
        if len(nodes) == 0:
            return None
        node = nodes[0]
        if not all(hasattr(node, attr) for attr in ("height", "top_length", "top_width")):
            return None
        import trimesh

        height = scalar_from_tensor_like(node.height)
        length = scalar_from_tensor_like(node.top_length)
        width = scalar_from_tensor_like(node.top_width)
        mesh = trimesh.creation.box(extents=(length, height, width))
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        mesh.faces = np.asarray(mesh.faces, dtype=np.int64)
        _ = mesh.vertex_normals
        mesh.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        return mesh

    @staticmethod
    def _write_grounded_sam2_metadata(obj, info: Mapping[str, Any]) -> None:
        if info.get("box_xyxy") is not None:
            obj.box_xyxy = info["box_xyxy"]
        if info.get("grounding_score") is not None:
            obj.grounding_score = float(info["grounding_score"])
        if info.get("sam_score") is not None:
            obj.sam_score = float(info["sam_score"])

    @staticmethod
    def _attach_common_outputs(map4d, rgb, depth, camera_intrinsics, results) -> None:
        map4d.construction_results = results
        map4d.rgb = rgb
        map4d.depth = depth
        if camera_intrinsics is not None:
            map4d.camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float32)

class Map4dSingleFrameConstructor(Map4dConstructor):
    """Explicit single-frame constructor name."""

class ManiSkillGTMap4dConstructor:
    """Build ManiSkill 4D maps directly from actor GT states."""

    MAP_CLASSES = {
        "StackCube-v1": Map4d_StackCube,
        "PlugCharger-v1": Map4d_PlugCharger,
    }

    def __init__(
        self,
        *,
        task_name: str,
        device: str,
    ):
        # get map class, pose parameters functions from task name
        if task_name not in self.MAP_CLASSES:
            raise ValueError(f"Unsupported ManiSkill GT map task {task_name!r}. Available: {sorted(self.MAP_CLASSES)}")
        map_class = self.MAP_CLASSES[task_name]

        # get metadata from task name
        task_metadata = load_map_metadata_for_task(task_name)
        size_parameters, relation_parameters = default_parameter_values_for_task(task_name)
        actor_names = tuple(task_metadata.get("actor_names", ()))

        # check actor name dimensions
        if len(actor_names) == 0:
            raise ValueError(f"Task {task_name!r} metadata must define actor_names.")

        self.task_name = task_name
        self.map_class = map_class
        self.actor_names = tuple(actor_names)
        self.size_parameters = tuple(float(v) for v in size_parameters)
        self.relation_parameters = tuple(float(v) for v in relation_parameters)
        self.task_metadata = task_metadata
        self.device = device
        self.actor_states = None

    def build_map_train(self):
        pose_parameters = self._parameters_train()
        return self._build_map_from_parameters(pose_parameters)

    def build_map_test(self):
        pose_parameters = self._parameters_test()
        return self._build_map_from_parameters(pose_parameters)

    def parameters_train(self, actor_states=None, *, frame_indices=None):
        return self._parameters_train(actor_states=actor_states, frame_indices=frame_indices)

    def parameters_test(
        self,
        actor_states=None,
        *,
        frame_indices=None,
        sizes: Optional[tuple[float, ...]] = None,
        relation_parameters: Optional[tuple[float, ...]] = None,
    ):
        return self._parameters_test(
            actor_states=actor_states,
            frame_indices=frame_indices,
            sizes=sizes,
            relation_parameters=relation_parameters,
        )

    def _build_map_from_parameters(self, parameters):
        map4d = self.map_class(
            parameters["positions"],
            parameters["rotations"],
            parameters["size_parameters"],
            parameters["relation_parameters"],
        )
        return map4d

    ##################################### Parameters #####################################
    def _parameters_train(self, actor_states=None, *, frame_indices=None):
        """Use JSON size/relation parameters and H5 actor-state poses."""

        actor_states = self._resolve_actor_states(actor_states, frame_indices=frame_indices)
        positions, rotations = pose_parameters_from_actor_states(actor_states, device=self.device)
        frame_count = actor_states[0].shape[0]
        return {
            "size_parameters": repeat_parameter_tensor(self.size_parameters, frame_count, device=self.device),
            "relation_parameters": repeat_parameter_tensor(self.relation_parameters, frame_count, device=self.device),
            "positions": positions,
            "rotations": rotations,
        }

    def _parameters_test(
        self,
        actor_states=None,
        *,
        frame_indices=None,
        sizes: Optional[tuple[float, ...]] = None,
        relation_parameters: Optional[tuple[float, ...]] = None,
    ):
        """Build size/relation/pose parameters for test-time GT maps."""

        actor_states = self._resolve_actor_states(actor_states, frame_indices=frame_indices)
        positions, rotations = pose_parameters_from_actor_states(actor_states, device=self.device)
        frame_count = actor_states[0].shape[0]
        return {
            "size_parameters": repeat_parameter_tensor(
                self.size_parameters if sizes is None else sizes,
                frame_count,
                device=self.device,
            ),
            "relation_parameters": repeat_parameter_tensor(
                self.relation_parameters if relation_parameters is None else relation_parameters,
                frame_count,
                device=self.device,
            ),
            "positions": positions,
            "rotations": rotations,
        }

    ##################################### Helpers #####################################
    def _resolve_actor_states(self, actor_states=None, *, frame_indices=None) -> list[np.ndarray]:
        if actor_states is None:
            if self.actor_states is None:
                raise ValueError("actor_states must be set before building a ManiSkill GT map.")
            actor_states = self.actor_states
        return normalize_actor_states(actor_states, self.actor_names, frame_indices=frame_indices)
