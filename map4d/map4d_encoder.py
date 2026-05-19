import torch.nn as nn

import torch

from .construction import Map4dConstructor, build_stackcube_template_map


class Map4d_Encoder(nn.Module):
    def __init__(self, map_constructor=None, map_template=None, encoder=None, **constructor_kwargs):
        super().__init__()
        self.map_constructor = map_constructor or Map4dConstructor(
            map_template=map_template or build_stackcube_template_map(device="cpu"),
            **constructor_kwargs,
        )
        self.encoder = encoder

    def forward(self, rgb, depth, camera_intrinsics=None, **construction_kwargs):
        '''
        Input: rgb + depth
        Output: 4dmap feature
        '''
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

    def encode(self, map_4d):
        '''
        Input: 4d map
        Output: 4d map feature
        '''
        return map_4d if self.encoder is None else self.encoder(map_4d)
