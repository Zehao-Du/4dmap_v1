"""Smoke checks for the standalone Map4D DiT backbone."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from helper.keyframe_targets import (
    MAP4D_DIT_TARGET_FORMAT,
    canonicalize_quaternion_np,
    gather_map4d_dit_keyframe_targets,
)
from map4d.backbone.dataset.maniskill_map4d_dataset import ManiSkillMap4DDataset
from map4d.backbone.model.diffusion.map4d_dit import rot6d_to_matrix
from map4d.backbone.policy.map4d_dit_policy import Map4DDiTPolicy


def _assert_geometry():
    quat = np.array([[-1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    canon = canonicalize_quaternion_np(quat)
    assert np.all(canon[:, 0] >= 0.0)
    assert np.allclose(np.linalg.norm(canon, axis=-1), 1.0)

    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    map4d = np.zeros((2, 1, 7), dtype=np.float32)
    map4d[..., 3:7] = identity_quat
    map4d[1, 0, 0:3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tcp = np.zeros((2, 7), dtype=np.float32)
    tcp[:, 3] = 1.0
    tcp[1, 0] = 1.0
    obj_target, tcp_target = gather_map4d_dit_keyframe_targets(
        map4d, tcp, np.array([[1], [1]], dtype=np.int64)
    )
    assert np.allclose(obj_target[0, 0, 0, 0:3], [1.0, 0.0, 0.0])
    assert np.allclose(tcp_target[0, 0, 0:3], [1.0, 0.0, 0.0])
    assert np.allclose(tcp_target[0, 0, 3:7], [1.0, 0.0, 0.0, 0.0])

    matrix = rot6d_to_matrix(torch.randn(8, 6))
    should_be_eye = matrix.transpose(-1, -2) @ matrix
    assert torch.allclose(should_be_eye, torch.eye(3).expand_as(should_be_eye), atol=1e-5, rtol=1e-5)


def _make_policy():
    cfg = OmegaConf.create(
        {
            "num_train_timesteps": 20,
            "prediction_type": "epsilon",
            "position_beta_schedule": "scaled_linear",
            "rotation_beta_schedule": "squaredcos_cap_v2",
        }
    )
    return Map4DDiTPolicy(
        action_horizon=8,
        keyframe_horizon=2,
        n_action_steps=4,
        n_obs_steps=2,
        num_inference_steps=2,
        noise_scheduler_cfg=cfg,
        model_cfg={
            "robot_state_dim": 16,
            "num_objects": 2,
            "num_map_nodes": 2,
            "map4d_dim": 7,
            "size_parameter_dim": 6,
            "relation_parameter_dim": 0,
            "rgb_feature_dim": 0,
            "semantic_feature_dim": 32,
            "pointcloud_encoder_cfg": {
                "in_channels": 35,
                "out_channels": 72,
                "use_bn": True,
                "npoint1": 64,
                "npoint2": 32,
            },
            "use_map_encoder": True,
            "map_name": "StackCube-v1",
            "embed_dim": 72,
            "depth": 2,
            "num_heads": 6,
            "diffusion_step_embed_dim": 72,
            "use_rgb": False,
        },
    )


def _assert_model_smoke():
    dataset = ManiSkillMap4DDataset(
        synthetic=True,
        synthetic_episodes=2,
        synthetic_length=12,
        horizon_action=8,
        horizon_keyframe=2,
        n_obs_steps=2,
        num_objects=2,
        robot_state_dim=16,
        map4d_dim=7,
        size_parameter_dim=6,
        relation_parameter_dim=0,
        use_rgb=True,
        rgb_feature_dim=32,
        semantic_feature_dim=32,
        seed=3,
    )
    batch_items = [dataset[i] for i in range(4)]
    batch = torch.utils.data.default_collate(batch_items)
    batch["obs"].pop("rgb_feature", None)
    policy = _make_policy()
    policy.set_normalizer(dataset.get_normalizer())
    loss, loss_dict = policy(batch)
    assert torch.isfinite(loss), loss_dict

    policy.eval()
    pred = policy.predict_action(batch["obs"])
    assert set(pred) >= {
        "action",
        "action_pred",
        "trajectory_pred",
        "gripper_openness_pred",
        "keyframe_node_position_pred",
        "keyframe_tcp_latent",
    }
    assert pred["action"].shape == (4, 4, 8), pred["action"].shape
    assert pred["action_pred"].shape == (4, 8, 8), pred["action_pred"].shape
    assert pred["trajectory_pred"].shape == (4, 8, 7), pred["trajectory_pred"].shape
    assert pred["gripper_openness_pred"].shape == (4, 8, 1), pred["gripper_openness_pred"].shape
    assert pred["keyframe_node_position_pred"].shape == (4, 2, 2, 3), pred["keyframe_node_position_pred"].shape
    assert pred["keyframe_tcp_latent"].shape == (4, 2, 7), pred["keyframe_tcp_latent"].shape
    for key, value in pred.items():
        assert torch.isfinite(value).all(), key


def _assert_sidecar_guard():
    if h5py is None:
        print("Skipping sidecar guard smoke: h5py is not installed.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        demo_path = os.path.join(tmp, "demo.h5")
        sidecar_path = os.path.join(tmp, "sidecar.h5")
        with h5py.File(demo_path, "w") as f:
            traj = f.create_group("traj_0")
            actions = np.zeros((2, 8), dtype=np.float32)
            actions[:, 3] = 1.0
            traj.create_dataset("actions", data=actions)
            obs = traj.create_group("obs")
            agent = obs.create_group("agent")
            agent.create_dataset("qpos", data=np.zeros((3, 4), dtype=np.float32))
            extra = obs.create_group("extra")
            tcp = np.zeros((3, 7), dtype=np.float32)
            tcp[:, 3] = 1.0
            extra.create_dataset("tcp_pose", data=tcp)
        with h5py.File(sidecar_path, "w") as f:
            f.attrs["target_format"] = "object_delta_pos_quat_plus_tcp_pose"
            group = f.create_group("traj_0")
            map4d = np.zeros((3, 3, 7), dtype=np.float32)
            map4d[..., 3] = 1.0
            group.create_dataset("map4d", data=map4d)
            group.create_dataset("future_keyframe_indices", data=np.zeros((3, 2), dtype=np.int64))
        try:
            ManiSkillMap4DDataset(
                demo_path=demo_path,
                keyframe_sidecar_path=sidecar_path,
                horizon_action=2,
                horizon_keyframe=2,
                num_objects=2,
                robot_state_dim=16,
            )
        except ValueError as exc:
            assert MAP4D_DIT_TARGET_FORMAT in str(exc)
        else:
            raise AssertionError("legacy sidecar format was accepted")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    _assert_geometry()
    _assert_model_smoke()
    _assert_sidecar_guard()
    print("Map4D DiT smoke checks passed.")


if __name__ == "__main__":
    main()
