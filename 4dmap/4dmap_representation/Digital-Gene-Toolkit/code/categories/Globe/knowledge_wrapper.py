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


class Semi_Ring_Bracket_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        grasp_pos_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 6

        bracket_offset = obj.bracket_offset
        bracket_exist_angle = obj.bracket_exist_angle
        bracket_rotation = obj.bracket_rotation
        bracket_size = obj.bracket_size

        self.grasp_position = np.array([bracket_size[0] + self.gripper_length, 0, 0])

        self.grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_position = (
            Rot.from_euler(
                "y", -bracket_exist_angle[0] * (1 / 2 + grasp_pos_ratio / 2)
            ).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler(
                "y", -bracket_exist_angle[0] * (1 / 2 + grasp_pos_ratio / 2)
            ).as_matrix()
            @ self.grasp_rotation
        )

        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )

        bracket_mesh_position = [0, bracket_offset[0], 0]
        bracket_mesh_rotation = [
            0,
            -np.pi / 2 + bracket_exist_angle[0] / 2 - bracket_rotation[0],
            np.pi / 2,
        ]
        self.geometry_position = bracket_mesh_position
        self.geometry_rotation = bracket_mesh_rotation


class Tilted_Bracket_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        grasp_pos_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 6

        bracket_size = obj.bracket_size
        circle_rotation = obj.circle_rotation

        self.grasp_position = np.array([bracket_size[1] + self.gripper_length, 0, 0])

        self.grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_position = (
            Rot.from_euler("y", np.pi * grasp_pos_ratio).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("y", np.pi * grasp_pos_ratio).as_matrix()
            @ self.grasp_rotation
        )

        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )

        circle_mesh_rotation = [0, 0, circle_rotation[0]]
        self.geometry_rotation = circle_mesh_rotation


class Enclosed_Bracket_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        grasp_pos_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 6

        bracket_size = obj.bracket_size
        circle_radius = obj.circle_radius
        circle_thickness = obj.circle_thickness
        outer_radius = circle_radius[0] + circle_thickness[0] / 2

        self.grasp_position = np.array([outer_radius + self.gripper_length, 0, 0])

        self.grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_position = (
            Rot.from_euler("y", np.pi * grasp_pos_ratio).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("y", np.pi * grasp_pos_ratio).as_matrix()
            @ self.grasp_rotation
        )

        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )


template2knowledge = {
    "Semi_Ring_Bracket": Semi_Ring_Bracket_Knowledge,
    "Tilted_Bracket": Tilted_Bracket_Knowledge,
    "Enclosed_Bracket": Enclosed_Bracket_Knowledge,
}
