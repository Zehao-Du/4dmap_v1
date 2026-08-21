"""Validate an RLBench2 Map4D DiT dataset and report per-key statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from map4d.backbone.dataset.rlbench2_map4d_dataset import RLBench2Map4DDataset


class ScalarStats:
    def __init__(self, key: str):
        self.key = key
        self.count = 0
        self.nonfinite = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.total = 0.0
        self.total_sq = 0.0
        self.dtype = None
        self.item_shape = None

    def update(self, value: np.ndarray, *, item_shape=None) -> None:
        array = np.asarray(value)
        if self.dtype is None:
            self.dtype = str(array.dtype)
            self.item_shape = list(array.shape if item_shape is None else item_shape)
        elif str(array.dtype) != self.dtype:
            raise TypeError(f"{self.key}: dtype changed from {self.dtype} to {array.dtype}")
        flat = array.reshape(-1)
        finite = np.isfinite(flat)
        self.nonfinite += int(flat.size - np.count_nonzero(finite))
        if not finite.all():
            raise ValueError(f"{self.key}: encountered {self.nonfinite} non-finite values")
        if flat.size == 0:
            return
        data = flat.astype(np.float64, copy=False)
        self.count += int(data.size)
        self.minimum = min(self.minimum, float(data.min()))
        self.maximum = max(self.maximum, float(data.max()))
        self.total += float(data.sum(dtype=np.float64))
        self.total_sq += float(np.square(data).sum(dtype=np.float64))

    def result(self) -> dict:
        if self.count == 0:
            return {
                "dtype": self.dtype,
                "item_shape": self.item_shape,
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "nonfinite": self.nonfinite,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "dtype": self.dtype,
            "item_shape": self.item_shape,
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
            "nonfinite": self.nonfinite,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--pcd-path", required=True)
    parser.add_argument("--dino-path", required=True)
    parser.add_argument("--lang-emb-path", required=True)
    parser.add_argument("--pose-path", required=True)
    parser.add_argument("--pcd-type", required=True)
    parser.add_argument("--prediction-type", default="continuous")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def add_array(stats: dict[str, ScalarStats], key: str, value: np.ndarray, item_shape) -> None:
    stats.setdefault(key, ScalarStats(key)).update(value, item_shape=item_shape)


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise ValueError(f"end={args.end} is smaller than start={args.start}")

    dataset = RLBench2Map4DDataset(
        data_path=args.data_path,
        pcd_path=args.pcd_path,
        dino_path=args.dino_path,
        lang_emb_path=args.lang_emb_path,
        pose_path=args.pose_path,
        start=args.start,
        end=args.end,
        val_ratio=0.0,
        prediction_type=args.prediction_type,
        pcd_type=args.pcd_type,
        robot_state_dim=16,
        action_type="bimanual_ee_pose",
        use_rgb=False,
    )
    raw_state = np.asarray(dataset.replay_buffer["state"], dtype=np.float32)
    raw_action = np.asarray(dataset.replay_buffer["action"], dtype=np.float32)
    dataset._validate_ppi_bimanual_layout(raw_state, field_name="robot_state")
    dataset._validate_ppi_bimanual_layout(raw_action, field_name="action")
    if args.prediction_type == "continuous" and not np.array_equal(raw_state, raw_action):
        raise ValueError("continuous PPI state and action arrays differ")

    stats: dict[str, ScalarStats] = {}
    point_refs = []
    dino_refs = []
    total_frames = 0
    for trajectory in tqdm(
        dataset.trajectories,
        desc="validating core episodes",
        unit="episode",
        dynamic_ncols=True,
        smoothing=0.0,
    ):
        episode = trajectory["episode_id"]
        state = trajectory["robot_state"]
        action = trajectory["actions"]
        tcp_pose = dataset._tcp_pose_from_robot_state(state)
        if action.shape[1:] != (2, 8):
            raise ValueError(f"episode {episode}: expected action [T,2,8], got {action.shape}")
        if tcp_pose.shape[1:] != (2, 7):
            raise ValueError(f"episode {episode}: expected TCP [T,2,7], got {tcp_pose.shape}")
        if not np.allclose(action[:, 0, 0:3], raw_action[total_frames:total_frames + len(state), 0:3]):
            raise ValueError(f"episode {episode}: left action position does not match PPI layout")
        if not np.allclose(action[:, 1, 0:3], raw_action[total_frames:total_frames + len(state), 7:10]):
            raise ValueError(f"episode {episode}: right action position does not match PPI layout")
        if not np.allclose(action[:, :, 7], raw_action[total_frames:total_frames + len(state), 14:16]):
            raise ValueError(f"episode {episode}: gripper action does not match PPI layout")

        add_array(stats, "obs.robot_state", state, state.shape[1:])
        add_array(stats, "obs.node_position", trajectory["map4d"][..., 0:3], trajectory["map4d"].shape[1:-1] + (3,))
        add_array(stats, "obs.node_rotation", trajectory["map4d"][..., 3:7], trajectory["map4d"].shape[1:-1] + (4,))
        add_array(stats, "obs.size_parameters", trajectory["size_parameters"], trajectory["size_parameters"].shape)
        add_array(stats, "obs.relation_parameters", trajectory["relation_parameters"], trajectory["relation_parameters"].shape)
        add_array(stats, "action.trajectory", action[..., 0:7], action.shape[1:-1] + (7,))
        add_array(stats, "action.gripper_openness", action[..., 7:8], action.shape[1:-1] + (1,))
        add_array(stats, "keyframe.map4d", trajectory["keyframe_object"], trajectory["keyframe_object"].shape[1:])
        add_array(stats, "keyframe.tcp", trajectory["keyframe_tcp"], trajectory["keyframe_tcp"].shape[1:])
        point_refs.extend(np.asarray(trajectory["point_refs"], dtype=np.int64))
        dino_refs.extend(np.asarray(trajectory["dino_refs"], dtype=np.int64))
        total_frames += len(state)

    if total_frames != len(raw_state):
        raise ValueError(f"trajectory frames={total_frames}, replay frames={len(raw_state)}")
    if len(point_refs) != total_frames or len(dino_refs) != total_frames:
        raise ValueError("visual reference count does not match total frames")

    visual_progress = tqdm(
        zip(point_refs, dino_refs),
        total=total_frames,
        desc="scanning point-cloud/DINO",
        unit="frame",
        dynamic_ncols=True,
        smoothing=0.0,
    )
    for index, (pc_ref, dino_ref) in enumerate(visual_progress):
        if not np.array_equal(pc_ref, dino_ref):
            raise ValueError(f"frame {index}: point/DINO refs differ: {pc_ref} vs {dino_ref}")
        episode, step = map(int, pc_ref)
        relative = Path(f"episode{episode}") / args.pcd_type / f"step{step:03d}.npy"
        pcd_file = Path(args.pcd_path) / relative
        dino_file = Path(args.dino_path) / relative
        if not pcd_file.is_file():
            raise FileNotFoundError(pcd_file)
        if not dino_file.is_file():
            raise FileNotFoundError(dino_file)
        pcd = np.load(pcd_file, mmap_mode="r")
        dino = np.load(dino_file, mmap_mode="r")
        if pcd.ndim != 2 or pcd.shape[1] != 6:
            raise ValueError(f"{pcd_file}: expected [P,6], got {pcd.shape}")
        if dino.ndim != 2 or dino.shape[0] != pcd.shape[0] or dino.shape[1] <= 3:
            raise ValueError(f"{dino_file}: incompatible DINO shape {dino.shape} for {pcd.shape}")
        add_array(stats, "obs.point_cloud", pcd, pcd.shape)
        add_array(stats, "obs.dino_feature", dino, dino.shape)

    report = {
        "scope": {"start_episode": args.start, "end_episode": args.end},
        "episodes": len(dataset.trajectories),
        "frames": total_frames,
        "keys": {key: value.result() for key, value in sorted(stats.items())},
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\nkey\tcount\tmin\tmax\tmean\tstd\tnonfinite")
    for key, value in report["keys"].items():
        print(
            f"{key}\t{value['count']}\t{value['min']}\t{value['max']}\t"
            f"{value['mean']}\t{value['std']}\t{value['nonfinite']}"
        )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
