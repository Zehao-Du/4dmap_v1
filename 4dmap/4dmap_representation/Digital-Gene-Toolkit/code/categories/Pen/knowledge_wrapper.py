import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

current_file_path = os.path.abspath(__file__)
sys.path.append(current_file_path)

from code.utils import *

from .concept_template import *

current_file_path = os.path.abspath(__file__)
sys.path.append(current_file_path)
sys.path.append(os.path.dirname(os.path.dirname(current_file_path)))  # categories
from code.base_knowledge import KnowledgeWrapper


class Single_Cap_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 2

        inner_size = obj.inner_size
        outer_size = obj.outer_size
        mid_radius = (outer_size[0] + outer_size[1]) / 2

        self.grasp_position = np.array(
            [
                0,
                outer_size[2] / 2 * (1 + trans_ratio * 0.7),
                -mid_radius - self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        self.grasp_position = Rot.from_euler("y", rot_angle).apply(self.grasp_position)
        self.grasp_rotation = Rot.from_euler("x", rot_angle).apply(self.grasp_rotation)

        self.force_direction = [0, -1, 0]

        bottom_mesh_position = [0, -inner_size[2] / 2, 0]
        self.geometry_position = np.array(bottom_mesh_position)


class Cylindrical_Barrel_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 2

        size = obj.size
        mid_radius = (size[0] + size[1]) / 2

        self.grasp_position = np.array(
            [0, size[2] / 2 * trans_ratio * 0.7, -mid_radius - self.gripper_length]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        self.grasp_position = Rot.from_euler("y", rot_angle).apply(self.grasp_position)
        self.grasp_rotation = Rot.from_euler("x", rot_angle).apply(self.grasp_rotation)

        self.force_direction = [0, 0, 1]


class Double_Layer_Barrel_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 2

        main_size = obj.main_size
        mid_radius = (main_size[0] + main_size[1]) / 2

        self.grasp_position = np.array(
            [0, main_size[2] / 2 * trans_ratio * 0.7, -mid_radius - self.gripper_length]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        self.grasp_position = Rot.from_euler("y", rot_angle).apply(self.grasp_position)
        self.grasp_rotation = Rot.from_euler("x", rot_angle).apply(self.grasp_rotation)

        self.force_direction = [0, 0, 1]


template2knowledge = {
    "Single_Cap": Single_Cap_Knowledge,
    "Cylindrical_Barrel": Cylindrical_Barrel_Knowledge,
    "Double_Layer_Barrel": Double_Layer_Barrel_Knowledge,
}
