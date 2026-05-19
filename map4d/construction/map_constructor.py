from __future__ import annotations

import copy
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MAPS4D_DIR = _REPO_ROOT / "map4d" / "representation" / "maps4d"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_MAPS4D_DIR) not in sys.path:
    sys.path.insert(0, str(_MAPS4D_DIR))


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
        rgb_np = self._as_rgb_uint8(rgb)
        depth_np = self._as_depth_float32(depth)
        if rgb_np.shape[:2] != depth_np.shape[:2]:
            raise ValueError(f"RGB/depth shape mismatch: rgb={rgb_np.shape}, depth={depth_np.shape}")

        objects = list(getattr(map4d, "Objects", []))
        prompts = [self._object_prompt(obj, idx) for idx, obj in enumerate(objects)]
        mask_info = self._segment_first_frame(rgb_np, objects, prompts, object_masks or {})
        point_clouds, structural_params = self._estimate_and_apply_structure(
            map4d=map4d,
            depth=depth_np,
            masks=[item["mask"] for item in mask_info],
            camera_intrinsics=camera_intrinsics,
        )
        objects = list(getattr(map4d, "Objects", []))
        prompts = [self._object_prompt(obj, idx) for idx, obj in enumerate(objects)]

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
        self._save_foundationpose_pose_visualizations(
            rgb=rgb_np,
            camera_intrinsics=camera_intrinsics,
            results=results,
            frame_tag="frame",
        )
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
        rgb_np = self._as_rgb_frames_uint8(rgb_frames)
        depth_np = self._as_depth_frames_float32(depth_frames)
        if rgb_np.shape[:3] != depth_np.shape[:3]:
            raise ValueError(f"RGB/depth sequence shape mismatch: rgb={rgb_np.shape}, depth={depth_np.shape}")

        objects = list(getattr(map4d, "Objects", []))
        prompts = [self._object_prompt(obj, idx) for idx, obj in enumerate(objects)]
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
        first_K = None if camera_intrinsics is None else self._camera_intrinsics_for_frame(camera_intrinsics, start_frame_idx)
        point_clouds, structural_params = self._estimate_and_apply_structure(
            map4d=map4d,
            depth=depth_np[start_frame_idx],
            masks=first_masks,
            camera_intrinsics=first_K,
        )
        objects = list(getattr(map4d, "Objects", []))
        prompts = [self._object_prompt(obj, idx) for idx, obj in enumerate(objects)]

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
        self._save_foundationpose_pose_visualizations(
            rgb=rgb_np[start_frame_idx],
            camera_intrinsics=first_K,
            results=results,
            frame_tag=f"frame_{start_frame_idx:06d}",
        )
        self._save_foundationpose_pose_videos(
            rgb_frames=rgb_np,
            camera_intrinsics=camera_intrinsics,
            results=results,
            fps=8,
        )
        return map4d

    def _resolve_map_template(self, map_template):
        template = self.map_template if map_template is None else map_template
        if template is None:
            raise ValueError("map_template must be provided at initialization or construct time.")
        return copy.deepcopy(template) if self.copy_template else template

    def _segment_first_frame(self, rgb: np.ndarray, objects: list[Any], prompts: list[str], object_masks: Mapping[Any, Any]):
        manual_masks = [
            self._lookup_by_object_key(object_masks, obj, object_index, prompts[object_index])
            for object_index, obj in enumerate(objects)
        ]
        if all(mask is not None for mask in manual_masks):
            return [
                {
                    "mask": self._as_mask_bool(mask, rgb.shape[:2], prompts[idx]),
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
        object_point_clouds = [self._masked_point_cloud_from_depth(depth, mask, K) for mask in masks]
        valid_clouds = [cloud for cloud in object_point_clouds if cloud.shape[0] > 0]
        if not valid_clouds:
            raise ValueError("No valid depth points inside any object mask; cannot estimate structural parameters.")

        merged = np.concatenate(valid_clouds, axis=0).astype(np.float32)
        sampled = self._sample_point_cloud(merged, self.structural_num_points)
        structural_params = self._run_structural_estimator(sampled)
        if not self._rebuild_map_from_structural_params(map4d, structural_params):
            self._apply_structural_params_to_map(map4d, structural_params)
        map4d.structural_params = structural_params
        map4d.object_point_clouds = object_point_clouds
        map4d.masked_point_cloud = sampled
        return object_point_clouds, structural_params

    def _run_structural_estimator(self, point_cloud: np.ndarray) -> np.ndarray:
        import torch
        try:
            from .structural_parameter_estimator import estimate_structural_parameters
        except ImportError:
            from structural_parameter_estimator import estimate_structural_parameters

        model = self.structural_parameter_estimator
        try:
            param = next(model.parameters())
            device = param.device
        except StopIteration:
            device = torch.device(self.device if torch.cuda.is_available() and str(self.device).startswith("cuda") else "cpu")
        points = torch.as_tensor(point_cloud[None], dtype=torch.float32, device=device)
        params = estimate_structural_parameters(model, points)
        return params.detach().cpu().numpy().astype(np.float32)

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
            positions, rotations = self._map_pose_tensors(map4d, dtype=dtype, device=device)
            rebuilt = model.build_map_from_params(params, positions=positions, rotations=rotations, clip_model=None)
        except Exception:
            return False

        for attr in ("Objects", "objects", "Nodes", "Node", "Edges", "Edge", "object_node_slices", "Subgraph_Prompts"):
            if hasattr(rebuilt, attr):
                setattr(map4d, attr, getattr(rebuilt, attr))
        return True

    @staticmethod
    def _map_pose_tensors(map4d, *, dtype, device):
        import torch

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

    def _apply_structural_params_to_map(self, map4d, structural_params: np.ndarray) -> None:
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
            self._set_tensor_like_scalar(node, "height", height)
            self._set_tensor_like_scalar(node, "top_length", length)
            self._set_tensor_like_scalar(node, "top_width", width)
            if hasattr(node, "bottom_length"):
                self._set_tensor_like_scalar(node, "bottom_length", length)
            if hasattr(node, "bottom_width"):
                self._set_tensor_like_scalar(node, "bottom_width", width)
            if hasattr(node, "back_height"):
                self._set_tensor_like_scalar(node, "back_height", height)

    @staticmethod
    def _set_tensor_like_scalar(obj, attr: str, value: float) -> None:
        current = getattr(obj, attr)
        if hasattr(current, "new_full"):
            setattr(obj, attr, current.new_full(current.shape, float(value)))
        else:
            setattr(obj, attr, float(value))

    @staticmethod
    def _masked_point_cloud_from_depth(depth: np.ndarray, mask: np.ndarray, camera_intrinsics: np.ndarray) -> np.ndarray:
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

    def _sample_point_cloud(self, points: np.ndarray, num_points: int) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected point cloud shape [N, 3], got {points.shape}")
        if points.shape[0] == 0:
            raise ValueError("Cannot sample an empty point cloud.")
        if num_points <= 0 or points.shape[0] == num_points:
            return points
        replace = points.shape[0] < num_points
        indices = self.rng.choice(points.shape[0], size=num_points, replace=replace)
        return points[indices].astype(np.float32)

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
        start_K = self._camera_intrinsics_for_frame(camera_intrinsics, start_frame_idx)
        start_pose = estimator.register(
            K=start_K,
            rgb=rgb_frames[start_frame_idx],
            depth=depth_frames[start_frame_idx],
            ob_mask=masks[start_frame_idx].astype(np.uint8),
            iteration=int(refine_iter),
        )
        poses[start_frame_idx] = np.asarray(start_pose, dtype=np.float32)
        for frame_idx in range(start_frame_idx + 1, rgb_frames.shape[0]):
            K = self._camera_intrinsics_for_frame(camera_intrinsics, frame_idx)
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
        mesh = self._lookup_by_object_key(object_meshes, obj, object_index, prompt)
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

        height = self._scalar_from_tensor_like(node.height)
        length = self._scalar_from_tensor_like(node.top_length)
        width = self._scalar_from_tensor_like(node.top_width)
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

    def _save_foundationpose_pose_visualizations(
        self,
        *,
        rgb: np.ndarray,
        camera_intrinsics,
        results: list[ObjectConstructionResult],
        frame_tag: str,
    ) -> None:
        if self.foundationpose_loader is None or self.foundationpose_debug_dir is None or camera_intrinsics is None:
            return
        drawable_results = [result for result in results if result.pose_6d is not None and result.mesh is not None]
        if not drawable_results:
            return
        try:
            draw_posed_3d_box, draw_xyz_axis = self._load_foundationpose_draw_utils()
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw
        except Exception as exc:
            warning_path = self.foundationpose_debug_dir / "pose_visualization_error.txt"
            warning_path.parent.mkdir(parents=True, exist_ok=True)
            warning_path.write_text(f"Failed to import FoundationPose drawing utilities: {exc}\n", encoding="utf-8")
            return

        output_dir = self.foundationpose_debug_dir / "pose_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        K = np.asarray(camera_intrinsics, dtype=np.float32)
        combined = rgb.copy()
        colors = self._pose_colors()
        for result in drawable_results:
            color = colors[result.object_index % len(colors)]
            object_overlay = rgb.copy()
            pose_for_box, bbox, axis_scale = self._pose_visualization_geometry(result.pose_6d, result.mesh)
            object_overlay = draw_posed_3d_box(K, img=object_overlay, ob_in_cam=pose_for_box, bbox=bbox, line_color=color, linewidth=2)
            object_overlay = draw_xyz_axis(
                object_overlay,
                ob_in_cam=pose_for_box,
                scale=axis_scale,
                K=K,
                thickness=2,
                transparency=0,
                is_input_rgb=True,
            )
            combined = draw_posed_3d_box(K, img=combined, ob_in_cam=pose_for_box, bbox=bbox, line_color=color, linewidth=2)
            combined = draw_xyz_axis(
                combined,
                ob_in_cam=pose_for_box,
                scale=axis_scale,
                K=K,
                thickness=2,
                transparency=0,
                is_input_rgb=True,
            )
            object_overlay = self._draw_pose_label(object_overlay, result.prompt, result.pose_6d, color)
            safe_prompt = "".join(ch if ch.isalnum() else "_" for ch in result.prompt).strip("_") or f"object_{result.object_index}"
            imageio.imwrite(output_dir / f"{frame_tag}_object_{result.object_index:02d}_{safe_prompt}_pose.png", object_overlay)
        combined = self._draw_pose_legend(combined, drawable_results, colors)
        imageio.imwrite(output_dir / f"{frame_tag}_foundationpose_poses.png", combined)

    def _save_foundationpose_pose_videos(
        self,
        *,
        rgb_frames: np.ndarray,
        camera_intrinsics,
        results: list[ObjectConstructionResult],
        fps: int = 8,
    ) -> None:
        if self.foundationpose_loader is None or self.foundationpose_debug_dir is None or camera_intrinsics is None:
            return
        drawable_results = [result for result in results if result.poses_6d is not None and result.mesh is not None]
        if not drawable_results:
            return
        try:
            draw_posed_3d_box, draw_xyz_axis = self._load_foundationpose_draw_utils()
            import imageio.v2 as imageio
        except Exception as exc:
            warning_path = self.foundationpose_debug_dir / "pose_video_error.txt"
            warning_path.parent.mkdir(parents=True, exist_ok=True)
            warning_path.write_text(f"Failed to import pose video utilities: {exc}\n", encoding="utf-8")
            return

        output_dir = self.foundationpose_debug_dir / "pose_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        colors = self._pose_colors()
        combined_frames = []
        per_object_frames = {result.object_index: [] for result in drawable_results}
        for frame_idx, rgb in enumerate(rgb_frames):
            K = self._camera_intrinsics_for_frame(camera_intrinsics, frame_idx)
            combined = rgb.copy()
            visible_results = []
            for result in drawable_results:
                pose = np.asarray(result.poses_6d[frame_idx], dtype=np.float32)
                if not np.isfinite(pose).all():
                    continue
                visible_results.append(result)
                color = colors[result.object_index % len(colors)]
                pose_for_box, bbox, axis_scale = self._pose_visualization_geometry(pose, result.mesh)
                object_overlay = rgb.copy()
                object_overlay = draw_posed_3d_box(K, img=object_overlay, ob_in_cam=pose_for_box, bbox=bbox, line_color=color, linewidth=2)
                object_overlay = draw_xyz_axis(
                    object_overlay,
                    ob_in_cam=pose_for_box,
                    scale=axis_scale,
                    K=K,
                    thickness=2,
                    transparency=0,
                    is_input_rgb=True,
                )
                object_overlay = self._draw_pose_label(object_overlay, result.prompt, pose, color)
                per_object_frames[result.object_index].append(object_overlay)
                combined = draw_posed_3d_box(K, img=combined, ob_in_cam=pose_for_box, bbox=bbox, line_color=color, linewidth=2)
                combined = draw_xyz_axis(
                    combined,
                    ob_in_cam=pose_for_box,
                    scale=axis_scale,
                    K=K,
                    thickness=2,
                    transparency=0,
                    is_input_rgb=True,
                )
            combined_frames.append(self._draw_pose_legend(combined, visible_results, colors))
        self._write_video(output_dir / "sequence_foundationpose_poses.mp4", combined_frames, fps=fps, imageio=imageio)
        for result in drawable_results:
            safe_prompt = "".join(ch if ch.isalnum() else "_" for ch in result.prompt).strip("_") or f"object_{result.object_index}"
            self._write_video(
                output_dir / f"sequence_object_{result.object_index:02d}_{safe_prompt}_pose.mp4",
                per_object_frames[result.object_index],
                fps=fps,
                imageio=imageio,
            )

    @staticmethod
    def _write_video(path: pathlib.Path, frames: list[np.ndarray], *, fps: int, imageio) -> None:
        if frames:
            imageio.mimsave(path, [np.asarray(frame, dtype=np.uint8) for frame in frames], fps=fps, macro_block_size=1)

    def _load_foundationpose_draw_utils(self):
        try:
            from Utils import draw_posed_3d_box, draw_xyz_axis
        except Exception:
            foundationpose_root = getattr(self.foundationpose_loader, "foundationpose_root", None)
            if foundationpose_root is not None and str(foundationpose_root) not in sys.path:
                sys.path.insert(0, str(foundationpose_root))
            from Utils import draw_posed_3d_box, draw_xyz_axis
        return draw_posed_3d_box, draw_xyz_axis

    @staticmethod
    def _pose_visualization_geometry(pose_6d: np.ndarray, mesh):
        import trimesh

        pose = np.asarray(pose_6d, dtype=np.float32)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).astype(np.float32)
        pose_for_box = pose @ np.linalg.inv(to_origin).astype(np.float32)
        axis_scale = float(np.clip(np.max(extents) * 0.35, 0.03, 0.12))
        return pose_for_box, bbox, axis_scale

    @staticmethod
    def _draw_pose_label(image: np.ndarray, prompt: str, pose_6d: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        from PIL import Image, ImageDraw

        xyz = np.asarray(pose_6d, dtype=np.float32)[:3, 3]
        text = f"{prompt}: t=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})m"
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([4, 4, min(pil.width - 1, 4 + 8 * len(text)), 24], fill=(0, 0, 0))
        draw.text((8, 7), text, fill=color)
        return np.asarray(pil)

    @staticmethod
    def _draw_pose_legend(image: np.ndarray, results: list[ObjectConstructionResult], colors: list[tuple[int, int, int]]) -> np.ndarray:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        row_h = 18
        width = min(pil.width - 1, 250)
        height = min(pil.height - 1, 6 + row_h * len(results))
        draw.rectangle([4, 4, width, height], fill=(0, 0, 0))
        for row, result in enumerate(results):
            color = colors[result.object_index % len(colors)]
            y = 7 + row * row_h
            draw.rectangle([8, y + 3, 18, y + 13], fill=color)
            draw.text((24, y), result.prompt, fill=color)
        return np.asarray(pil)

    @staticmethod
    def _pose_colors() -> list[tuple[int, int, int]]:
        return [
            (255, 64, 64),
            (64, 220, 64),
            (64, 144, 255),
            (255, 192, 64),
            (220, 64, 255),
        ]

    @staticmethod
    def _lookup_by_object_key(mapping: Mapping[Any, Any], obj, object_index: int, prompt: str):
        for key in (object_index, prompt, getattr(obj, "prompt", None), getattr(obj, "Object_Prompt", None), obj.__class__.__name__):
            if key is not None and key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _object_prompt(obj, object_index: int) -> str:
        for attr in ("prompt", "Object_Prompt", "semantic"):
            value = getattr(obj, attr, None)
            if value is not None and str(value).strip() != "":
                return str(value).strip()
        return f"object_{object_index}"

    @staticmethod
    def _camera_intrinsics_for_frame(camera_intrinsics, frame_idx: int) -> np.ndarray:
        K = np.asarray(camera_intrinsics, dtype=np.float32)
        return K[frame_idx] if K.ndim == 3 else K

    @staticmethod
    def _as_rgb_uint8(rgb) -> np.ndarray:
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

    @classmethod
    def _as_rgb_frames_uint8(cls, rgb_frames) -> np.ndarray:
        arr = np.asarray(rgb_frames)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Expected rgb_frames shape [T, H, W, 3/4], got {arr.shape}")
        return np.stack([cls._as_rgb_uint8(frame) for frame in arr], axis=0)

    @staticmethod
    def _as_depth_float32(depth) -> np.ndarray:
        depth_np = np.asarray(depth)
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
        if depth_np.ndim != 2:
            raise ValueError(f"Expected depth shape [H, W] or [H, W, 1], got {depth_np.shape}")
        depth_np = depth_np.astype(np.float32)
        if np.nanmax(depth_np) > 100.0:
            depth_np = depth_np / 1000.0
        return depth_np

    @classmethod
    def _as_depth_frames_float32(cls, depth_frames) -> np.ndarray:
        arr = np.asarray(depth_frames)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim == 3 or (arr.ndim == 4 and arr.shape[-1] == 1):
            return np.stack([cls._as_depth_float32(frame) for frame in arr], axis=0)
        raise ValueError(f"Expected depth_frames shape [T, H, W] or [T, H, W, 1], got {arr.shape}")

    @staticmethod
    def _as_mask_bool(mask, image_hw: tuple[int, int], prompt: str) -> np.ndarray:
        mask_np = np.asarray(mask)
        if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        if mask_np.shape != image_hw:
            raise ValueError(f"Mask for prompt={prompt!r} has shape {mask_np.shape}, expected {image_hw}")
        return mask_np.astype(bool)

    @staticmethod
    def _scalar_from_tensor_like(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return float(arr[0])


class Map4dSingleFrameConstructor(Map4dConstructor):
    """Explicit single-frame constructor name."""


def build_stackcube_template_map(*, sizes=None, positions=None, rotations=None, device: str = "cpu"):
    import torch
    from maniskill_stackcube import Map4d_StackCube

    if sizes is None:
        sizes = torch.tensor(
            [[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.02, 0.5, 0.5]],
            dtype=torch.float32,
            device=device,
        )
    positions = torch.zeros((1, 9), dtype=torch.float32, device=device) if positions is None else positions
    if rotations is None:
        rotations = torch.zeros((1, 18), dtype=torch.float32, device=device)
        for i in range(3):
            rotations[:, i * 6 : (i + 1) * 6] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                dtype=torch.float32,
                device=device,
            )
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
