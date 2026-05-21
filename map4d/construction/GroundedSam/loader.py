from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np


PathLike = Union[str, pathlib.Path]


@dataclass(frozen=True)
class GroundedSam2Assets:
    repo_root: pathlib.Path
    sam2_checkpoint: pathlib.Path
    sam2_config: str
    grounding_model_id: str
    device: str


@dataclass
class GroundedSam2Result:
    text_prompt: str
    boxes_xyxy: np.ndarray
    labels: list[str]
    grounding_scores: np.ndarray
    masks: np.ndarray
    sam_scores: np.ndarray


@dataclass
class GroundedSam2PromptResult:
    text_prompt: str
    mask: Optional[np.ndarray]
    box_xyxy: Optional[np.ndarray]
    label: Optional[str]
    grounding_score: float
    sam_score: float
    candidates: GroundedSam2Result


@dataclass
class GroundedSam2BatchResult:
    text_prompts: list[str]
    masks: np.ndarray
    boxes_xyxy: np.ndarray
    labels: list[Optional[str]]
    grounding_scores: np.ndarray
    sam_scores: np.ndarray
    per_prompt: list[GroundedSam2PromptResult]


@dataclass
class GroundedSam2TrackResult:
    text_prompts: list[str]
    masks: np.ndarray
    boxes_xyxy: np.ndarray
    labels: list[Optional[str]]
    grounding_scores: np.ndarray
    sam_scores: np.ndarray
    first_frame: GroundedSam2BatchResult


def _loader_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def default_grounded_sam2_root() -> pathlib.Path:
    return _loader_dir() / "Grounded-SAM-2"


def default_sam2_checkpoint(root: Optional[PathLike] = None) -> pathlib.Path:
    repo_root = pathlib.Path(root).expanduser().resolve() if root is not None else default_grounded_sam2_root()
    return repo_root / "checkpoints" / "sam2.1_hiera_large.pt"


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


def _normalize_device(device: Optional[str]) -> str:
    import torch

    if device is not None:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_repo_on_path(repo_root: pathlib.Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _normalize_text_prompt(text_prompt: str) -> str:
    prompt = str(text_prompt).strip().lower()
    if not prompt:
        raise ValueError("text_prompt must be non-empty.")
    if not prompt.endswith("."):
        prompt = prompt + "."
    return prompt


def _as_rgb_uint8(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected image shape [H, W, 3/4], got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        if float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


class GroundedSam2Loader:
    """Grounded-SAM-2 image wrapper.

    Pipeline:
      RGB image + text prompt
        -> GroundingDINO boxes
        -> SAM2 masks from boxes

    The GroundingDINO model is loaded through Hugging Face Transformers. The
    SAM2 model is loaded from the vendored Grounded-SAM-2 repo and local
    checkpoints directory.
    """

    def __init__(
        self,
        *,
        repo_root: Optional[PathLike] = None,
        sam2_checkpoint: Optional[PathLike] = None,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
    ):
        self.repo_root = pathlib.Path(repo_root).expanduser().resolve() if repo_root is not None else default_grounded_sam2_root()
        self.repo_root = _require_existing_dir(self.repo_root, "Grounded-SAM-2 repo root")
        self.sam2_checkpoint = (
            pathlib.Path(sam2_checkpoint).expanduser().resolve()
            if sam2_checkpoint is not None
            else default_sam2_checkpoint(self.repo_root)
        )
        self.sam2_checkpoint = _require_existing_file(self.sam2_checkpoint, "SAM2 checkpoint")
        self.sam2_config = sam2_config
        self.grounding_model_id = grounding_model_id
        self.device = _normalize_device(device)

        _ensure_repo_on_path(self.repo_root)
        self._processor = None
        self._grounding_model = None
        self._image_predictor = None
        self._video_predictor = None

    def resolve_assets(self) -> GroundedSam2Assets:
        return GroundedSam2Assets(
            repo_root=self.repo_root,
            sam2_checkpoint=self.sam2_checkpoint,
            sam2_config=self.sam2_config,
            grounding_model_id=self.grounding_model_id,
            device=self.device,
        )

    def load_grounding_model(self):
        if self._processor is None or self._grounding_model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.grounding_model_id)
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.grounding_model_id
            ).to(self.device)
            self._grounding_model.eval()
        return self._processor, self._grounding_model

    def load_sam2_image_predictor(self):
        if self._image_predictor is None:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if str(self.device).startswith("cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            sam2_model = build_sam2(
                self.sam2_config,
                str(self.sam2_checkpoint),
                device=self.device,
            )
            self._image_predictor = SAM2ImagePredictor(sam2_model)
        return self._image_predictor

    def load_sam2_video_predictor(self):
        if self._video_predictor is None:
            import torch
            from sam2.build_sam import build_sam2_video_predictor

            if str(self.device).startswith("cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            self._video_predictor = build_sam2_video_predictor(
                self.sam2_config,
                str(self.sam2_checkpoint),
                device=self.device,
            )
        return self._video_predictor

    def predict(
        self,
        image,
        text_prompt: str,
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.3,
        multimask_output: bool = False,
    ) -> GroundedSam2Result:
        import torch
        from PIL import Image

        rgb = _as_rgb_uint8(image)
        text = _normalize_text_prompt(text_prompt)
        pil_image = Image.fromarray(rgb)

        processor, grounding_model = self.load_grounding_model()
        inputs = processor(images=pil_image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[pil_image.size[::-1]],
        )[0]

        boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
        labels = [str(label) for label in results.get("labels", [])]
        grounding_scores = results["scores"].detach().cpu().numpy().astype(np.float32)

        if boxes.shape[0] == 0:
            h, w = rgb.shape[:2]
            return GroundedSam2Result(
                text_prompt=text,
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                labels=[],
                grounding_scores=np.zeros((0,), dtype=np.float32),
                masks=np.zeros((0, h, w), dtype=bool),
                sam_scores=np.zeros((0,), dtype=np.float32),
            )

        image_predictor = self.load_sam2_image_predictor()
        image_predictor.set_image(rgb)
        masks, sam_scores, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=bool(multimask_output),
        )

        masks, sam_scores = self._normalize_sam_outputs(masks, sam_scores, boxes.shape[0])
        return GroundedSam2Result(
            text_prompt=text,
            boxes_xyxy=boxes,
            labels=labels,
            grounding_scores=grounding_scores,
            masks=masks.astype(bool),
            sam_scores=sam_scores.astype(np.float32),
        )

    def predict_prompts(
        self,
        image,
        text_prompts: Sequence[str],
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.3,
        multimask_output: bool = False,
        select_by: str = "grounding_score",
        allow_empty: bool = False,
    ) -> GroundedSam2BatchResult:
        rgb = _as_rgb_uint8(image)
        prompts = [str(prompt) for prompt in text_prompts]
        if len(prompts) == 0:
            h, w = rgb.shape[:2]
            return GroundedSam2BatchResult(
                text_prompts=[],
                masks=np.zeros((0, h, w), dtype=bool),
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                labels=[],
                grounding_scores=np.zeros((0,), dtype=np.float32),
                sam_scores=np.zeros((0,), dtype=np.float32),
                per_prompt=[],
            )

        per_prompt = []
        selected_masks = []
        selected_boxes = []
        selected_labels = []
        selected_grounding_scores = []
        selected_sam_scores = []

        h, w = rgb.shape[:2]
        empty_mask = np.zeros((h, w), dtype=bool)
        empty_box = np.full((4,), np.nan, dtype=np.float32)

        for prompt in prompts:
            candidates = self.predict(
                rgb,
                prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                multimask_output=multimask_output,
            )
            best_idx = self._select_candidate_index(candidates, select_by)

            if best_idx is None:
                if not allow_empty:
                    raise RuntimeError(
                        f"Grounded-SAM2 found no mask for prompt={prompt!r}. "
                        "Try lowering thresholds or improving the Object.prompt."
                    )
                mask = empty_mask.copy()
                box = empty_box.copy()
                label = None
                grounding_score = 0.0
                sam_score = 0.0
            else:
                mask = candidates.masks[best_idx].astype(bool)
                box = candidates.boxes_xyxy[best_idx].astype(np.float32)
                label = candidates.labels[best_idx] if best_idx < len(candidates.labels) else None
                grounding_score = float(candidates.grounding_scores[best_idx])
                sam_score = float(candidates.sam_scores[best_idx])

            selected_masks.append(mask)
            selected_boxes.append(box)
            selected_labels.append(label)
            selected_grounding_scores.append(grounding_score)
            selected_sam_scores.append(sam_score)
            per_prompt.append(
                GroundedSam2PromptResult(
                    text_prompt=_normalize_text_prompt(prompt),
                    mask=None if best_idx is None else mask,
                    box_xyxy=None if best_idx is None else box,
                    label=label,
                    grounding_score=grounding_score,
                    sam_score=sam_score,
                    candidates=candidates,
                )
            )

        return GroundedSam2BatchResult(
            text_prompts=[item.text_prompt for item in per_prompt],
            masks=np.stack(selected_masks, axis=0),
            boxes_xyxy=np.stack(selected_boxes, axis=0).astype(np.float32),
            labels=selected_labels,
            grounding_scores=np.asarray(selected_grounding_scores, dtype=np.float32),
            sam_scores=np.asarray(selected_sam_scores, dtype=np.float32),
            per_prompt=per_prompt,
        )

    def track_prompts(
        self,
        frames,
        text_prompts: Sequence[str],
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.3,
        multimask_output: bool = False,
        select_by: str = "grounding_score",
        prompt_type: str = "mask",
        allow_empty: bool = False,
        start_frame_idx: int = 0,
        max_frame_num_to_track: Optional[int] = None,
        frames_dir: Optional[PathLike] = None,
    ) -> GroundedSam2TrackResult:
        frames_np = self._as_rgb_frames_uint8(frames)
        if start_frame_idx < 0 or start_frame_idx >= frames_np.shape[0]:
            raise ValueError(f"start_frame_idx={start_frame_idx} out of range for {frames_np.shape[0]} frames.")
        if prompt_type not in {"mask", "box"}:
            raise ValueError("prompt_type must be 'mask' or 'box'.")

        first_frame = self.predict_prompts(
            frames_np[start_frame_idx],
            text_prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            multimask_output=multimask_output,
            select_by=select_by,
            allow_empty=allow_empty,
        )

        video_predictor = self.load_sam2_video_predictor()
        resolved_frames_dir = self._prepare_tracking_frames(frames_np, frames_dir)
        inference_state = video_predictor.init_state(str(resolved_frames_dir))

        for object_idx, prompt in enumerate(first_frame.text_prompts, start=1):
            prompt_result = first_frame.per_prompt[object_idx - 1]
            if prompt_result.mask is None:
                continue
            if prompt_type == "mask":
                video_predictor.add_new_mask(
                    inference_state,
                    frame_idx=start_frame_idx,
                    obj_id=object_idx,
                    mask=prompt_result.mask.astype(np.uint8),
                )
            else:
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=start_frame_idx,
                    obj_id=object_idx,
                    box=prompt_result.box_xyxy,
                )

        frame_count = frames_np.shape[0]
        object_count = len(first_frame.text_prompts)
        tracked_masks = np.zeros((frame_count, object_count, frames_np.shape[1], frames_np.shape[2]), dtype=bool)

        for frame_idx, object_ids, mask_logits in video_predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
        ):
            for i, object_id in enumerate(object_ids):
                object_zero_idx = int(object_id) - 1
                if 0 <= object_zero_idx < object_count:
                    tracked_masks[int(frame_idx), object_zero_idx] = (
                        mask_logits[i] > 0.0
                    ).detach().cpu().numpy().squeeze().astype(bool)

        return GroundedSam2TrackResult(
            text_prompts=first_frame.text_prompts,
            masks=tracked_masks,
            boxes_xyxy=first_frame.boxes_xyxy,
            labels=first_frame.labels,
            grounding_scores=first_frame.grounding_scores,
            sam_scores=first_frame.sam_scores,
            first_frame=first_frame,
        )

    @staticmethod
    def _select_candidate_index(result: GroundedSam2Result, select_by: str) -> Optional[int]:
        if result.masks.shape[0] == 0:
            return None
        if select_by == "grounding_score":
            return int(np.argmax(result.grounding_scores))
        if select_by == "sam_score":
            return int(np.argmax(result.sam_scores))
        if select_by == "combined_score":
            return int(np.argmax(result.grounding_scores * result.sam_scores))
        raise ValueError(
            f"Unsupported select_by={select_by!r}; expected "
            "'grounding_score', 'sam_score', or 'combined_score'."
        )

    @staticmethod
    def _normalize_sam_outputs(masks: np.ndarray, scores: np.ndarray, num_boxes: int) -> tuple[np.ndarray, np.ndarray]:
        masks = np.asarray(masks)
        scores = np.asarray(scores)

        if masks.ndim == 3:
            masks = masks[None]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim == 4:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(num_boxes), best]
            scores = scores[np.arange(num_boxes), best]

        if scores.ndim == 2:
            scores = scores[:, 0]
        return masks, scores.reshape(-1)

    @staticmethod
    def _as_rgb_frames_uint8(frames) -> np.ndarray:
        arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[-1] not in (3, 4):
            raise ValueError(f"Expected frames shape [T, H, W, 3/4], got {arr.shape}")
        return np.stack([_as_rgb_uint8(frame) for frame in arr], axis=0)

    @staticmethod
    def _prepare_tracking_frames(frames: np.ndarray, frames_dir: Optional[PathLike]) -> pathlib.Path:
        from PIL import Image
        import shutil
        import tempfile

        if frames_dir is None:
            path = pathlib.Path(tempfile.mkdtemp(prefix="grounded_sam2_frames_"))
        else:
            path = pathlib.Path(frames_dir).expanduser().resolve()
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)

        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(path / f"{idx:05d}.jpg")
        return path


def load_grounded_sam2(**kwargs) -> GroundedSam2Loader:
    return GroundedSam2Loader(**kwargs)


__all__ = [
    "GroundedSam2Assets",
    "GroundedSam2BatchResult",
    "GroundedSam2Loader",
    "GroundedSam2PromptResult",
    "GroundedSam2Result",
    "GroundedSam2TrackResult",
    "default_grounded_sam2_root",
    "default_sam2_checkpoint",
    "load_grounded_sam2",
]
