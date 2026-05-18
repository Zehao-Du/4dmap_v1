import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

current_file_path = os.path.abspath(__file__)
sys.path.append(current_file_path)

from .concept_template import *

current_file_path = os.path.abspath(__file__)
sys.path.append(current_file_path)
sys.path.append(os.path.dirname(os.path.dirname(current_file_path)))  # categories
from code.base_knowledge import KnowledgeWrapper
from code.utils import *


class Cylindrical_body_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        x_z_ratio = obj.x_z_ratio
        num_of_part = obj.num_of_part
        all_sizes = obj.all_sizes
        top_radius, bottom_radius, height = (
            all_sizes[0][0],
            all_sizes[0][2],
            all_sizes[0][1],
        )

        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        self.grasp_position = np.array(
            [
                0,
                (trans_ratio + 1) * height,
                top_radius / x_z_ratio[0] + 2 * self.gripper_length,
            ]
        )

        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )


class Cuboidal_body_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio
        size = obj.size

        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.grasp_position = np.array(
            [
                0,
                (trans_ratio) * size[1] / 2,
                size[0] + 2 * self.gripper_length,
            ]
        )

        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )


class Toothpaste_body_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio
        radius = obj.radius
        height = obj.height
        bottom_length = obj.bottom_length

        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        self.grasp_position = np.array(
            [
                0,
                (trans_ratio) * height[0] * 0.4,
                (radius[0] * trans_ratio + bottom_length[0] / 2 * (1 - trans_ratio)) / 2
                + 2 * self.gripper_length,
            ]
        )

        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )

template2knowledge = {
    "Cylindrical_body": Cylindrical_body_Knowledge,
    "Cuboidal_body": Cuboidal_body_Knowledge,
    "Toothpaste_body": Toothpaste_body_Knowledge,
}
