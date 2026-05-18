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


class Roller_Door_Knowledge(KnowledgeWrapper):
    """open"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_1_ratio, rot_2_ratio = manip_args if manip_args else (0, 0)
        rot_1_angle = (rot_1_ratio + 0.75) * np.pi / 6 
        rot_2_angle = rot_2_ratio * np.pi / 6 
        circle_size = obj.circle_size

        # when door is open
        self.grasp_position = np.array([circle_size[0] + self.gripper_length, 0, 0])
        self.grasp_rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        self.grasp_rotation = (
            Rot.from_euler("z", rot_1_angle).as_matrix() @ self.grasp_rotation
        )
        self.grasp_position = (
            Rot.from_euler("y", rot_2_angle).as_matrix() @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("y", rot_2_angle).as_matrix() @ self.grasp_rotation
        )

        self.force_direction = np.array([0, 1, 0])

        circle_mesh_position = [circle_size[0], 0, circle_size[2] / 2]
        circle_mesh_rotation = [np.pi / 2, 0, 0]
        self.geometry_position = circle_mesh_position
        self.geometry_rotation = circle_mesh_rotation


class Cuboidal_Door_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_ratio, trans_ratio = manip_args if manip_args else (0, 0)
        rot_angle = (rot_ratio + 0.5) * np.pi / 6
        size = obj.size

        self.grasp_position = [
            trans_ratio * size[0] * 0.25,
            0,
            size[2] / 2 + self.gripper_length,
        ]
        self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            Rot.from_euler("x", -rot_angle).as_matrix() @ self.grasp_rotation
        )
        self.force_direction = np.array([0, 1, 0])  # outward

        mesh_position = [0, size[1] / 2, 0]
        self.geometry_position = mesh_position


template2knowledge = {
    "Roller_Door": Roller_Door_Knowledge,
    "Cuboidal_Door": Cuboidal_Door_Knowledge,
}
