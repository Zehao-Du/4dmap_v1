import torch.nn as nn

import torch

from .construction import Map4dConstructor, build_stackcube_template_map
from .encoder.geometric_encoder import GeometricEncoder


class Map4d_Encoder(nn.Module):
    def __init__(
        self,
        map_constructor=None,
        map_template=None,
        encoder=None,
        num_objects=3,
        pre_horizon=None,
        future_horizon=1,
        node_dim=128,
        relation_dim=64,
        temporal_dim=128,
        feature_dim=128,
        **constructor_kwargs,
    ):
        super().__init__()
        self.pre_horizon = pre_horizon
        self.future_horizon = int(future_horizon)
        self.num_objects = int(num_objects)
        self.map_constructor = map_constructor or Map4dConstructor(
            map_template=map_template or build_stackcube_template_map(device="cpu"),
            **constructor_kwargs,
        )
        self.encoder = encoder or GeometricEncoder(
            num_objects=num_objects,
            node_dim=node_dim,
            relation_dim=relation_dim,
            temporal_dim=temporal_dim,
            feature_dim=feature_dim,
        )
        self.feature_dim = getattr(self.encoder, "feature_dim", feature_dim)
        self.future_head = nn.Linear(self.feature_dim, self.future_horizon * self.num_objects * 9)

    def forward(self, map4d_seq=None, rgb=None, depth=None, camera_intrinsics=None, **construction_kwargs):
        '''
        Input: 4d map sequence, or rgb + depth sequence for online construction
        Output: 4dmap feature
        '''
        map4d_seq = self._resolve_map4d_sequence(
            map4d_seq=map4d_seq,
            rgb=rgb,
            depth=depth,
            camera_intrinsics=camera_intrinsics,
            **construction_kwargs,
        )
        return self.encode(map4d_seq)

    def forward_with_aux(self, map4d_seq=None, future_map4d_seq=None, rgb=None, depth=None, camera_intrinsics=None, **construction_kwargs):
        map4d_seq = self._resolve_map4d_sequence(
            map4d_seq=map4d_seq,
            rgb=rgb,
            depth=depth,
            camera_intrinsics=camera_intrinsics,
            **construction_kwargs,
        )
        if future_map4d_seq is None:
            return self.encoder.forward_with_prediction(map4d_seq)
        map_feature, obj_feat, _, sizes, positions, rotations = self.encoder._encode_sequence_parts(map4d_seq)
        _, _, _, future_sizes, future_positions, future_rotations = self.encoder._encode_sequence_parts(future_map4d_seq)
        pred = self.predict_future_from_features(obj_feat, sizes, positions, rotations, future_sizes, future_positions, future_rotations)
        return map_feature, pred

    def predict_future_from_features(self, obj_feat, sizes, positions, rotations, future_sizes, future_positions, future_rotations):
        batch_size = obj_feat.shape[0]
        pred_delta = self.future_head(obj_feat[:, -1].mean(dim=1))
        pred_delta = pred_delta.view(batch_size, self.future_horizon, self.num_objects, 9)
        pred_delta_pos = pred_delta[..., :3]
        pred_delta_rot = pred_delta[..., 3:]
        pred_pos = positions[:, -1:] + torch.cumsum(pred_delta_pos, dim=1)
        pred_rot = rotations[:, -1:] + torch.cumsum(pred_delta_rot, dim=1)
        gt_sizes = torch.cat([sizes[:, -1:], future_sizes[:, : self.future_horizon]], dim=1)
        gt_positions = torch.cat([positions[:, -1:], future_positions[:, : self.future_horizon]], dim=1)
        gt_rotations = torch.cat([rotations[:, -1:], future_rotations[:, : self.future_horizon]], dim=1)
        return {
            "pred_delta_pos": pred_delta_pos,
            "pred_delta_rot": pred_delta_rot,
            "pred_pos": pred_pos,
            "pred_rot": pred_rot,
            "valid_mask": torch.ones((batch_size, pred_pos.shape[1], self.num_objects), device=pred_pos.device),
            "sizes": gt_sizes,
            "positions": gt_positions,
            "rotations": gt_rotations,
            "object_features": obj_feat,
        }

    def _resolve_map4d_sequence(self, map4d_seq=None, rgb=None, depth=None, camera_intrinsics=None, **construction_kwargs):
        if map4d_seq is not None:
            return map4d_seq
        if rgb is None or depth is None:
            raise ValueError("Either map4d_seq or rgb/depth must be provided.")
        if self.pre_horizon is not None:
            rgb = rgb[-self.pre_horizon :]
            depth = depth[-self.pre_horizon :]
        return self.construct_sequence(rgb, depth, camera_intrinsics=camera_intrinsics, **construction_kwargs)

    def construct_and_encode(self, rgb, depth, camera_intrinsics=None, **construction_kwargs):
        map_4d = self.construction(rgb, depth, camera_intrinsics=camera_intrinsics, **construction_kwargs)
        return self.encode(map_4d)

    def construction(self, rgb, depth, camera_intrinsics=None, **kwargs):
        '''
        Input: rgb + depth
        Output: 4d map
        '''
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()
        return self.map_constructor.construct(rgb, depth, camera_intrinsics=camera_intrinsics, **kwargs)

    def construct_sequence(self, rgb_frames, depth_frames, camera_intrinsics=None, **kwargs):
        '''
        Input: rgb/depth sequence
        Output: instantiated 4d map sequence
        '''
        if torch.is_tensor(rgb_frames):
            rgb_frames = rgb_frames.detach().cpu().numpy()
        if torch.is_tensor(depth_frames):
            depth_frames = depth_frames.detach().cpu().numpy()
        return self.map_constructor.instantiate_sequence(
            rgb_frames=rgb_frames,
            depth_frames=depth_frames,
            camera_intrinsics=camera_intrinsics,
            **kwargs,
        )

    def encode(self, map4d_seq):
        '''
        Input: 4d map sequence
        Output: 4d map feature
        '''
        return self.encoder(map4d_seq)
