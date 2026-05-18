import numpy as np
from utils_torch import relative_pose_6d
import torch
from torch_geometric.data import Data
import clip
import re
    
class Object:
    def __init__ (self, position=[0,0,0], rotation=[0,0,0], rotation_order="XYZ", semantic="object"):
        '''
        An object in 4dmap
        '''
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order
        self.semantic = semantic
    
class Map_4d:
    def __init__ (self, objects):
        self.objects = []
        for object in objects:
            self.objects.append(object)
    def constraint_functions (self):
        # TODO
        # Maybe many functions are needed, this function is just an example
        return NotImplementedError