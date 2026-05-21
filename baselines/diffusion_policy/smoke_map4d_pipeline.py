import argparse
import json
import pathlib
import shutil
import sys
from functools import partial

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from PIL import Image, ImageDraw


_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import train_rgbd as train_mod  # noqa: E402
from diffusion_policy.make_env import make_eval_envs  # noqa: E402
from train_rgbd import (  # noqa: E402
    Agent,
    Args,
    SmallDemoDataset_DiffusionPolicy,
    build_state_obs_extractor,
    convert_obs,
)


DEFAULT_DEMO_PATH = (
    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/"
    "dataset/ManiSkill/StackCube-v1/motionplanning/"
    "StackCube.rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.h5"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/map4d_pipeline_smoke"
)


def _shape_dtype(value):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    arr = np.asarray(value)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def _tensor_stats(value):
    tensor = value.detach().float().cpu() if torch.is_tensor(value) else torch.as_tensor(value).float()
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": 0}
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
    }


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rotation_6d_to_matrix_np(rotation_6d):
    rot = np.asarray(rotation_6d, dtype=np.float32)
    a1 = rot[..., 0:3]
    a2 = rot[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(min=1e-8)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(min=1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _cuboid_corners(size, position, rotation_6d):
    size = np.asarray(size, dtype=np.float32)
    position = np.asarray(position, dtype=np.float32)
    local = np.asarray(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    ) * size[None]
    rot = _rotation_6d_to_matrix_np(rotation_6d)
    return local @ rot.T + position[None]


def _project_points(points_world, intrinsic_cv, extrinsic_cv):
    points_world = np.asarray(points_world, dtype=np.float32)
    intrinsic_cv = np.asarray(intrinsic_cv, dtype=np.float32)
    extrinsic_cv = np.asarray(extrinsic_cv, dtype=np.float32)
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    points_cam = points_h @ extrinsic_cv.T
    z = points_cam[:, 2]
    pixels_h = points_cam @ intrinsic_cv.T
    pixels = pixels_h[:, :2] / z[:, None].clip(min=1e-8)
    return pixels, z


def _draw_map4d_projection(
    canvas,
    map4d_frame,
    intrinsic_cv,
    extrinsic_cv,
    object_indices=None,
    fill_faces=True,
):
    image = Image.fromarray(np.asarray(canvas).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    faces = [
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
    ]
    colors = [(255, 40, 40), (40, 220, 70), (50, 130, 255)]
    width, height = image.size
    map4d_frame = np.asarray(map4d_frame, dtype=np.float32)
    if object_indices is None:
        object_indices = range(map4d_frame.shape[0])
    for object_idx in object_indices:
        obj = map4d_frame[object_idx]
        corners = _cuboid_corners(obj[:3], obj[3:6], obj[6:12])
        pixels, depth = _project_points(corners, intrinsic_cv, extrinsic_cv)
        color = colors[object_idx % len(colors)]
        fill = tuple(int(0.35 * c + 0.65 * 255) for c in color)
        face_items = []
        for face in faces:
            face_depth = depth[list(face)]
            if np.any(face_depth <= 1e-6):
                continue
            poly = pixels[list(face)]
            if not np.isfinite(poly).all():
                continue
            face_items.append((float(face_depth.mean()), [tuple(p) for p in poly]))
        if fill_faces:
            for _, poly in sorted(face_items, reverse=True):
                draw.polygon(poly, fill=fill)
        for i, j in edges:
            if depth[i] <= 1e-6 or depth[j] <= 1e-6:
                continue
            p0 = pixels[i]
            p1 = pixels[j]
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                continue
            # Keep long off-screen table edges from dominating the image.
            if (
                (p0[0] < -width or p0[0] > 2 * width or p0[1] < -height or p0[1] > 2 * height)
                and (p1[0] < -width or p1[0] > 2 * width or p1[1] < -height or p1[1] > 2 * height)
            ):
                continue
            draw.line([tuple(p0), tuple(p1)], fill=(0, 0, 0), width=5)
            draw.line([tuple(p0), tuple(p1)], fill=color, width=3)
    return image


def _draw_map4d_overlay(rgb, map4d_frame, intrinsic_cv, extrinsic_cv, object_indices=None):
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    return _draw_map4d_projection(
        rgb,
        map4d_frame,
        intrinsic_cv,
        extrinsic_cv,
        object_indices=object_indices,
        fill_faces=False,
    )


def _draw_map4d_only(rgb, map4d_frame, intrinsic_cv, extrinsic_cv, object_indices=None):
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    canvas = np.full_like(rgb, 245, dtype=np.uint8)
    return _draw_map4d_projection(
        canvas,
        map4d_frame,
        intrinsic_cv,
        extrinsic_cv,
        object_indices=object_indices,
        fill_faces=True,
    )


def _save_image(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _save_camera_visualizations(prefix, output_dir, raw_obs, map4d_frame):
    paths = {}
    vis_root = output_dir / "visualizations" / prefix
    for camera_name, camera_data in raw_obs["sensor_data"].items():
        vis_dir = vis_root / camera_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        rgb = _to_numpy(camera_data["rgb"])
        params = raw_obs["sensor_param"][camera_name]
        intrinsic = _to_numpy(params["intrinsic_cv"])[0]
        extrinsic = _to_numpy(params["extrinsic_cv"])[0]
        original_path = vis_dir / "rgb.png"
        overlay_path = vis_dir / "overlay.png"
        map4d_path = vis_dir / "map4d.png"
        _save_image(original_path, rgb)
        overlay = _draw_map4d_overlay(rgb, map4d_frame, intrinsic, extrinsic, object_indices=[0, 1])
        overlay.save(overlay_path)
        map4d_only = _draw_map4d_only(rgb, map4d_frame, intrinsic, extrinsic, object_indices=[0, 1])
        map4d_only.save(map4d_path)
        paths[camera_name] = {
            "rgb": str(original_path),
            "map4d": str(map4d_path),
            "overlay": str(overlay_path),
        }
    return paths


def _load_train_raw_obs_frame(demo_path, traj_idx, frame_idx):
    import h5py

    with h5py.File(demo_path, "r") as f:
        traj_keys = [key for key in f.keys() if key.startswith("traj_")]
        traj_keys = sorted(traj_keys, key=lambda key: int(key.split("_")[-1]))
        obs_group = f[traj_keys[traj_idx]]["obs"]
        raw_obs = {"sensor_data": {}, "sensor_param": {}}
        for camera_name in obs_group["sensor_data"].keys():
            raw_obs["sensor_data"][camera_name] = {
                "rgb": obs_group["sensor_data"][camera_name]["rgb"][frame_idx],
            }
            raw_obs["sensor_param"][camera_name] = {
                "intrinsic_cv": obs_group["sensor_param"][camera_name]["intrinsic_cv"][frame_idx][None],
                "extrinsic_cv": obs_group["sensor_param"][camera_name]["extrinsic_cv"][frame_idx][None],
            }
    return raw_obs


def _make_raw_env_obs(args, env_kwargs):
    env = gym.make(args.env_id, reconfiguration_freq=1, **env_kwargs)
    raw_obs, _ = env.reset(seed=args.seed)
    env.close()
    return raw_obs


def _make_args(cli_args):
    return Args(
        env_id="StackCube-v1",
        demo_path=cli_args.demo_path,
        num_demos=cli_args.num_demos,
        obs_horizon=cli_args.obs_horizon,
        pred_horizon=cli_args.pred_horizon,
        act_horizon=cli_args.act_horizon,
        control_mode="pd_ee_delta_pos",
        use_map4d=True,
        map4d_future_horizon=cli_args.future_horizon,
        max_episode_steps=cli_args.max_episode_steps,
        batch_size=1,
        num_eval_envs=1,
        cuda=False,
        capture_video=False,
    )


def run_smoke(cli_args):
    output_dir = pathlib.Path(cli_args.output_dir)
    if output_dir.exists() and cli_args.clean_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    args = _make_args(cli_args)
    train_mod.args = args
    train_mod.device = device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
        human_render_camera_configs=dict(shader_pack="default"),
    )

    tmp_env = gym.make(args.env_id, **env_kwargs)
    obs_space = tmp_env.observation_space
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth=True,
    )

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=device,
        num_traj=args.num_demos,
        dataset_control_mode="pd_ee_delta_pose",
        target_control_mode=args.control_mode,
        use_map4d=True,
        map4d_source=args.map4d_source,
        map4d_task_name=args.map4d_task_name,
        map4d_future_horizon=args.map4d_future_horizon,
        map4d_strict=True,
    )

    eval_env = make_eval_envs(
        args.env_id,
        1,
        args.sim_backend,
        env_kwargs,
        {"obs_horizon": args.obs_horizon},
        wrappers=[FlattenRGBDObservationWrapper],
        map4d_source=args.map4d_source,
        map4d_task_name=args.map4d_task_name,
        map4d_strict=True,
    )
    agent = Agent(eval_env, args).to(device)

    item = dataset[cli_args.dataset_index]
    obs_batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in item["observations"].items()
    }
    action_batch = item["actions"].unsqueeze(0)

    agent.train()
    with torch.no_grad():
        map_feature, map_aux = agent.map4d_encoder.forward_with_aux(
            map4d_seq=obs_batch["map4d"],
            future_map4d_seq=obs_batch["future_map4d"],
        )
        train_loss = agent.compute_loss(obs_batch, action_batch)
        map_losses = agent.last_map_losses

    agent.eval()
    infer_obs, _ = eval_env.reset(seed=args.seed)
    infer_obs_tensor = common.to_tensor(infer_obs, device)
    with torch.no_grad():
        infer_map_feature = agent.map4d_encoder(map4d_seq=infer_obs_tensor["map4d"])
        infer_action = agent.get_action(infer_obs_tensor)
    eval_env.close()

    train_traj_idx, train_start, _ = dataset.slices[cli_args.dataset_index]
    train_frame_idx = max(0, train_start + args.obs_horizon - 1)
    train_raw_obs = _load_train_raw_obs_frame(args.demo_path, train_traj_idx, train_frame_idx)
    infer_raw_obs = _make_raw_env_obs(args, env_kwargs)
    train_visualization_paths = _save_camera_visualizations(
        "train",
        output_dir,
        train_raw_obs,
        _to_numpy(obs_batch["map4d"][0, -1]),
    )
    infer_visualization_paths = _save_camera_visualizations(
        "inference",
        output_dir,
        infer_raw_obs,
        _to_numpy(infer_obs_tensor["map4d"][0, -1]),
    )

    arrays_path = output_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        train_map4d=_to_numpy(obs_batch["map4d"]),
        train_future_map4d=_to_numpy(obs_batch["future_map4d"]),
        train_action_seq=_to_numpy(action_batch),
        train_map_feature=_to_numpy(map_feature),
        infer_obs_map4d=_to_numpy(infer_obs_tensor["map4d"]),
        infer_map_feature=_to_numpy(infer_map_feature),
        infer_action=_to_numpy(infer_action),
    )

    tensors_path = output_dir / "tensors.pt"
    torch.save(
        {
            "train_obs": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in obs_batch.items()},
            "train_actions": action_batch.detach().cpu(),
            "train_map_feature": map_feature.detach().cpu(),
            "train_map_aux": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in map_aux.items()},
            "infer_obs": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in infer_obs_tensor.items()},
            "infer_map_feature": infer_map_feature.detach().cpu(),
            "infer_action": infer_action.detach().cpu(),
        },
        tensors_path,
    )

    report = {
        "status": "ok",
        "demo_path": args.demo_path,
        "dataset_index": cli_args.dataset_index,
        "output_dir": str(output_dir),
        "files": {
            "report": str(output_dir / "report.json"),
            "arrays": str(arrays_path),
            "tensors": str(tensors_path),
            "visualizations": {
                "training": train_visualization_paths,
                "inference": infer_visualization_paths,
            },
        },
        "config": {
            "obs_horizon": args.obs_horizon,
            "future_horizon": args.map4d_future_horizon,
            "pred_horizon": args.pred_horizon,
            "act_horizon": args.act_horizon,
            "control_mode": args.control_mode,
            "map4d_source": args.map4d_source,
            "map4d_coordinate_system": "maniskill_world_xyz_z_up",
            "map4d_size_order": "x_extent, y_extent, z_extent",
            "map4d_pose_order": "x, y, z, rotation_6d(first two columns of ManiSkill actor rotation)",
        },
        "training": {
            "loss": float(train_loss.item()),
            "visualization_frame": {
                "traj_idx": int(train_traj_idx),
                "frame_idx": int(train_frame_idx),
            },
            "map_losses": {
                key: float(value.item())
                for key, value in (map_losses or {}).items()
            },
            "observations": {
                key: _shape_dtype(value)
                for key, value in obs_batch.items()
                if torch.is_tensor(value)
            },
            "actions": _shape_dtype(action_batch),
            "map_feature": _tensor_stats(map_feature),
            "map_aux": {
                key: _tensor_stats(value)
                for key, value in map_aux.items()
                if torch.is_tensor(value)
            },
            "first_observed_map4d_frame": _to_numpy(obs_batch["map4d"][0, 0]).tolist(),
            "first_future_map4d_frame": _to_numpy(obs_batch["future_map4d"][0, 0]).tolist(),
        },
        "inference": {
            "visualization_frame": {
                "seed": int(args.seed),
                "frame": "reset",
            },
            "observations": {
                key: _shape_dtype(value)
                for key, value in infer_obs_tensor.items()
                if torch.is_tensor(value)
            },
            "map_feature": _tensor_stats(infer_map_feature),
            "action": _tensor_stats(infer_action),
            "action_values": _to_numpy(infer_action[0]).tolist(),
            "first_map4d_frame": _to_numpy(infer_obs_tensor["map4d"][0, 0]).tolist(),
        },
    }

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["files"], indent=2))
    print(f"train_loss={report['training']['loss']:.6f}")
    print(f"infer_action_shape={report['inference']['action']['shape']}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-path", default=DEFAULT_DEMO_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-demos", type=int, default=1)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--future-horizon", type=int, default=3)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--act-horizon", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=5)
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=True)
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
