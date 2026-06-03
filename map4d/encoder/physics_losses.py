import torch
import torch.nn as nn
import torch.nn.functional as F

from ..representation.maps4d.utils_torch import rotation_6d_to_matrix


class PhysicsLosses(nn.Module):
    """Compute physics-inspired losses for predicted object motion."""

    def __init__(
        self,
        num_objects: int,
        pose_weight: float = 1.0,
        penetration_weight: float = 0.1,
        kinematic_weight: float = 0.1,
        vel_limit: float = 0.5,
        acc_limit: float = 1.0,
        rot_vel_limit: float = 1.0,
        rot_acc_limit: float = 2.0,
        penetration_margin: float = 0.0,
    ):
        super().__init__()
        if num_objects <= 0:
            raise ValueError("num_objects must be positive")
        self.num_objects = num_objects
        self.pose_weight = pose_weight
        self.penetration_weight = penetration_weight
        self.kinematic_weight = kinematic_weight
        self.vel_limit = vel_limit
        self.acc_limit = acc_limit
        self.rot_vel_limit = rot_vel_limit
        self.rot_acc_limit = rot_acc_limit
        self.penetration_margin = penetration_margin
        self.register_buffer(
            "box_corners",
            torch.tensor(
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
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(self, pred, rep_seq=None):
        sizes = pred.get("sizes")
        positions = pred.get("positions")
        rotations = pred.get("rotations")
        pred_delta_pos = pred.get("pred_delta_pos")
        pred_delta_rot = pred.get("pred_delta_rot")
        pred_pos = pred.get("pred_pos")
        pred_rot = pred.get("pred_rot")

        if sizes is None or positions is None or rotations is None:
            raise ValueError("pred must include sizes/positions/rotations")

        if pred_delta_pos is None or pred_delta_rot is None:
            return _zero_losses(positions)

        if pred_delta_pos.shape[1] == 0:
            return _zero_losses(positions)

        gt_delta_pos = positions[:, 1:] - positions[:, :-1]

        R_curr = rotation_6d_to_matrix(rotations[:, :-1])
        R_next = rotation_6d_to_matrix(rotations[:, 1:])
        gt_delta_R = R_next @ R_curr.transpose(-1, -2)

        pred_delta_R = rotation_6d_to_matrix(pred_delta_rot)

        pose_loss = F.l1_loss(pred_delta_pos, gt_delta_pos) + F.mse_loss(
            pred_delta_R, gt_delta_R
        )

        kinematic_loss = _kinematic_loss(
            pred_delta_pos,
            pred_delta_rot,
            self.vel_limit,
            self.acc_limit,
            self.rot_vel_limit,
            self.rot_acc_limit,
        )

        penetration_loss = _penetration_loss(
            pred_pos,
            pred_rot,
            sizes[:, 1:],
            self.penetration_margin,
            self.box_corners,
        )

        total = (
            self.pose_weight * pose_loss
            + self.kinematic_weight * kinematic_loss
            + self.penetration_weight * penetration_loss
        )

        return {
            "pose": pose_loss,
            "kinematic": kinematic_loss,
            "penetration": penetration_loss,
            "total": total,
        }


def _zero_losses(ref):
    device = ref.device
    return {
        "pose": torch.tensor(0.0, device=device),
        "kinematic": torch.tensor(0.0, device=device),
        "penetration": torch.tensor(0.0, device=device),
        "total": torch.tensor(0.0, device=device),
    }


def _kinematic_loss(
    delta_pos,
    delta_rot,
    vel_limit,
    acc_limit,
    rot_vel_limit,
    rot_acc_limit,
):
    speed = torch.linalg.norm(delta_pos, dim=-1)
    vel_penalty = F.relu(speed - vel_limit).mean()

    rot_speed = torch.linalg.norm(delta_rot, dim=-1)
    rot_vel_penalty = F.relu(rot_speed - rot_vel_limit).mean()

    if delta_pos.shape[1] > 1:
        acc = delta_pos[:, 1:] - delta_pos[:, :-1]
        acc_mag = torch.linalg.norm(acc, dim=-1)
        acc_penalty = F.relu(acc_mag - acc_limit).mean()

        rot_acc = delta_rot[:, 1:] - delta_rot[:, :-1]
        rot_acc_mag = torch.linalg.norm(rot_acc, dim=-1)
        rot_acc_penalty = F.relu(rot_acc_mag - rot_acc_limit).mean()
    else:
        acc_penalty = torch.tensor(0.0, device=delta_pos.device)
        rot_acc_penalty = torch.tensor(0.0, device=delta_pos.device)

    return vel_penalty + acc_penalty + rot_vel_penalty + rot_acc_penalty


def _penetration_loss(positions, rotations, sizes, margin, corners):
    if positions is None or positions.shape[1] == 0:
        return torch.tensor(0.0, device=sizes.device)

    min_xyz, max_xyz = _aabb_from_pose(positions, rotations, sizes, corners)
    B, T, N, _ = min_xyz.shape
    if N < 2:
        return torch.tensor(0.0, device=positions.device)

    total = 0.0
    pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            overlap_min = torch.maximum(min_xyz[:, :, i], min_xyz[:, :, j])
            overlap_max = torch.minimum(max_xyz[:, :, i], max_xyz[:, :, j])
            overlap = (overlap_max - overlap_min - margin).clamp(min=0.0)
            vol = overlap.prod(dim=-1)
            total = total + vol.mean()
            pairs += 1

    return total / max(pairs, 1)


def _aabb_from_pose(positions, rotations, sizes, corners):
    verts = _box_vertices(positions, rotations, sizes, corners)
    min_xyz = verts.min(dim=-2).values
    max_xyz = verts.max(dim=-2).values
    return min_xyz, max_xyz


def _box_vertices(positions, rotations, sizes, corners):
    B, T, N, _ = positions.shape
    flat = B * T * N
    pos_flat = positions.reshape(flat, 3)
    rot_flat = rotations.reshape(flat, 6)
    size_flat = sizes.reshape(flat, 3)

    rot_mats = rotation_6d_to_matrix(rot_flat)
    corners = corners.to(device=positions.device, dtype=positions.dtype)
    verts = corners.unsqueeze(0) * size_flat.unsqueeze(1)
    verts = torch.matmul(verts, rot_mats.transpose(1, 2))
    verts = verts + pos_flat.unsqueeze(1)
    return verts.view(B, T, N, 8, 3)
