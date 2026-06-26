from typing import Union, Dict
import copy

import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn
from map4d.backbone.common.pytorch_util import dict_apply
from map4d.backbone.model.common.dict_of_tensor_mixin import DictOfTensorMixin


# TODO(normalization-audit): Check PPI and Map4D normalization parity.
# - PPI stats generation uses mode="limits" for action, agent_pos, point_cloud,
#   dino_feature, lang, point_flow, and initial_point_flow.
# - PPI policy normalizes batch["obs"] as a dict, then normalizes action and
#   point_flow separately.
# - Map4D stats generation should use mode="limits" for every normalized field.
# - Map4D policy normalizes batch["obs"] as a dict; every possible obs key
#   returned by the dataset must have a matching normalizer entry.
# - Audit whether Map4D point_cloud, dino_feature, and rgb_feature should be
#   fitted in get_normalizer() when those keys are enabled in obs.
# - Audit target-field normalization names: trajectory_pos, gripper_openness,
#   keyframe_map4d_pos, keyframe_tcp_pos, and keyframe_tcp_gripper.
'''
ppi normalization
  action              -> 16 dims: left xyz, left quat, right xyz, right quat, left openess, right openess
  agent_pos (robot state) -> 16 dims: left xyz, left quat, right xyz, right quat, left openess, right openess
  point_cloud         -> 6 dims: xyz + rgb
  dino_feature        -> 384 dims: DINO channels
  lang                -> lang_dim dims: language embedding channels
  point_flow          -> 3 dims: xyz flow
  initial_point_flow  -> 3 dims: xyz flow
  
DiTmap4d normalization:
  action                -> 8 dims: left xyz, left quat, left openess
  robot state           -> 8 dims: left xyz, left quat, left openess
  4dmap_feature         -> map_dims: 4dmap embedding channels  (LayerNorm, not in this python file)
  point_cloud           -> 6 dims: xyz + rgb
  node_position         -> 3 dim: xyz, uses point_cloud xyz stats for graph space
  node_rotation         -> 4 dims: quaternion_wxyz, identity normalizer / canonicalization only
  dino_feature          -> 384 dims: DINO channels
  (optional) lang       -> lang_dim dims: language embedding channels
'''


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] =  _fit(value, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
        else:
            self.params_dict['_default'] = _fit(data, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
    
    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                if key not in self.params_dict and torch.is_tensor(value) and value.numel() == 0:
                    result[key] = value
                    continue
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def normalize_map4d_graph_data(
            self,
            data,
            point_cloud_key: str = "point_cloud",
            inplace: bool = False):
        return normalize_map4d_graph_data(
            data,
            self,
            point_cloud_key=point_cloud_key,
            inplace=inplace,
        )

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']
        
        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)
        
        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        self.params_dict = _fit(data, 
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset)
    
    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj
    
    @classmethod
    def create_manual(cls, 
            scale: Union[torch.Tensor, np.ndarray], 
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x
        
        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype
        
        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)



def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1,dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def _point_cloud_xyz_params(normalizer: LinearNormalizer, point_cloud_key: str = "point_cloud"):
    if point_cloud_key not in normalizer.params_dict:
        raise KeyError(
            f"LinearNormalizer is missing {point_cloud_key!r}; "
            "Map4D graph normalization must use point_cloud xyz stats."
        )
    params = normalizer.params_dict[point_cloud_key]
    scale = params["scale"]
    offset = params["offset"]
    if scale.numel() < 3 or offset.numel() < 3:
        raise ValueError(
            f"{point_cloud_key!r} normalizer must contain at least xyz channels, "
            f"got scale shape {tuple(scale.shape)} and offset shape {tuple(offset.shape)}."
        )
    return scale[:3], offset[:3]


def _normalize_absolute_xyz(value: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=value.device, dtype=value.dtype)
    offset = offset.to(device=value.device, dtype=value.dtype)
    return value * scale + offset


def _normalize_delta_xyz(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=value.device, dtype=value.dtype)
    return value * scale


def normalize_map4d_graph_data(
        data,
        normalizer: LinearNormalizer,
        point_cloud_key: str = "point_cloud",
        inplace: bool = False):
    """Normalize Map4D graph coordinates with the point-cloud xyz normalizer.

    Map4D representation is constructed in real metric space. Before feeding
    graph data to the map encoder, every absolute xyz coordinate must be moved
    into the same normalized coordinate system as obs.point_cloud[..., :3].

    This function intentionally does not normalize rotations, semantic features,
    category/index tensors, or anchor direction vectors. Relative xyz deltas use
    only the point-cloud scale, because affine offsets cancel in differences.
    """
    scale, offset = _point_cloud_xyz_params(normalizer, point_cloud_key=point_cloud_key)
    out = data if inplace else copy.copy(data)

    if hasattr(out, "node_pos") and out.node_pos is not None:
        out.node_pos = _normalize_absolute_xyz(out.node_pos, scale, offset)

    if hasattr(out, "x_pos") and out.x_pos is not None and out.x_pos.numel() > 0:
        out.x_pos = _normalize_absolute_xyz(out.x_pos, scale, offset)

    if hasattr(out, "x_aff") and out.x_aff is not None and out.x_aff.numel() > 0:
        out.x_aff = _normalize_absolute_xyz(out.x_aff, scale, offset)

    if hasattr(out, "edge_pose") and out.edge_pose is not None and out.edge_pose.numel() > 0:
        if out.edge_pose.shape[-1] < 3:
            raise ValueError(f"Map4D edge_pose must have at least 3 xyz dims, got {tuple(out.edge_pose.shape)}")
        edge_pose = out.edge_pose.clone()
        edge_pose[..., 0:3] = _normalize_delta_xyz(edge_pose[..., 0:3], scale)
        out.edge_pose = edge_pose

    if hasattr(out, "edge_anchor") and out.edge_anchor is not None and out.edge_anchor.numel() > 0:
        if out.edge_anchor.shape[-1] % 12 != 0:
            raise ValueError(
                "Map4D edge_anchor is expected to be packed as "
                "[p(3), primary(3), tangent(3), bitangent(3)] per anchor; "
                f"got last dim {out.edge_anchor.shape[-1]}."
            )
        edge_anchor = out.edge_anchor.clone()
        for start in range(0, edge_anchor.shape[-1], 12):
            edge_anchor[..., start:start + 3] = _normalize_absolute_xyz(
                edge_anchor[..., start:start + 3],
                scale,
                offset,
            )
        out.edge_anchor = edge_anchor

    return out


def test():
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1., atol=1e-3)
    assert np.allclose(datan.min(), 0., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100,10,9,2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='gaussian', last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0., atol=1e-3)
    assert np.allclose(datan.std(), 1., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


    # dict
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    data = {
        'obs': torch.zeros((1000,128,9,2)).uniform_() * 512,
        'action': torch.zeros((1000,128,2)).uniform_() * 512
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
    
    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
