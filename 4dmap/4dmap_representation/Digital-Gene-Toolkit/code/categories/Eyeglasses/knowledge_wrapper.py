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


class Regular_Leg_Knowledge(KnowledgeWrapper):
    """right leg"""
    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        grasp_pos_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 6

        size1 = obj.size1
        size2 = obj.size2
        offset_x = obj.offset_x
        glass_interval = obj.glass_interval
        rotation_1 = obj.rotation_1
        direction = -1
        if type(offset_x) is not list:
            offset_x = [offset_x]
        if type(glass_interval) is not list:
            glass_interval = [glass_interval]
        leg_interval = offset_x[0][0] * 2 + glass_interval[0]

        self.grasp_position = np.array(
            [0, size1[1] / 2 + self.gripper_length, size1[2] * grasp_pos_ratio * 0.3]
        )
        self.grasp_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )

        middle_mesh_position = [-direction * size1[0] / 2, -size1[1] / 2, -size1[2] / 2]
        self.geometry_position = np.array(middle_mesh_position)
        first_position = [
            direction * leg_interval / 2,
            0,
            0,
        ]
        first_rotation = [-rotation_1[0], -direction * rotation_1[1], 0]
        self.apply_transformation_for_geometry(
            self.geometry_position, [0, 0, 0], [first_position], [first_rotation], "YXZ"
        )


class Trifold_Leg_Knowledge(KnowledgeWrapper):
    """right leg"""
    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        grasp_pos_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 6

        size1 = obj.size1
        size2 = obj.size2
        connector_size = obj.connector_size
        rotation_1 = obj.rotation_1
        direction = -1

        self.grasp_position = np.array(
            [0, size1[1] / 2 + self.gripper_length, size1[2] * grasp_pos_ratio * 0.3]
        )
        self.grasp_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )

        middle_mesh_position = [-direction * size1[0] / 2, -size1[1] / 2, -size1[2] / 2]
        self.geometry_position = np.array(middle_mesh_position)
        first_position = [0, 0, 0]
        first_rotation = [-rotation_1[0], -direction * rotation_1[1], 0]
        self.apply_transformation_for_geometry(
            self.geometry_position, [0, 0, 0], [first_position], [first_rotation], "XYZ"
        )


template2knowledge = {
    "Regular_Leg": Regular_Leg_Knowledge,
    "Trifold_Leg": Trifold_Leg_Knowledge,
}
