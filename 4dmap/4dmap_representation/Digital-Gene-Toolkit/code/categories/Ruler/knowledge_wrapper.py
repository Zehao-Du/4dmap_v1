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


class Symmetrical_body_Knowledge(KnowledgeWrapper):
    """close the ruler"""

    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        length_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        body_rotation = obj.body_rotation
        separation = obj.separation
        left_right_offset = obj.left_right_offset
        size = obj.size

        self.grasp_position = np.array(
            [
                -(length_ratio + 1) * size[0] * 0.25,
                0,
                -size[2] / 2 - self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.grasp_position = (
            self.grasp_position
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", -rot_angle, degrees=False).as_matrix()
        )

        self.force_direction = np.array([0, -1, 0])

        # left, more rotation more down
        mesh_1_rotation = [0, 0, body_rotation[0]]
        mesh_1_position = [
            -separation[0]
            - (size[0] * np.cos(body_rotation[0]) + size[1] * np.sin(body_rotation[0]))
            / 2,
            (size[1] * np.cos(body_rotation[0]) - size[0] * np.sin(body_rotation[0]))
            / 2,
            -left_right_offset[0],
        ]
        self.geometry_position = np.array(mesh_1_position)
        self.geometry_rotation = np.array(mesh_1_rotation)


class Asymmetrical_body_Knowledge(KnowledgeWrapper):
    """close the ruler"""

    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        length_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        body_rotation = obj.body_rotation
        separation = obj.separation
        left_right_offset = obj.left_right_offset
        left_size = obj.left_size

        self.grasp_position = np.array(
            [
                -(length_ratio + 1) * left_size[0] * 0.25,
                0,
                -left_size[2] / 2 - self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.grasp_position = (
            self.grasp_position
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", -rot_angle, degrees=False).as_matrix()
        )

        self.force_direction = np.array([0, -1, 0])

        # left, more rotation more down
        mesh_1_rotation = [0, 0, body_rotation[0]]
        mesh_1_position = [
            -separation[0]
            - (
                left_size[0] * np.cos(body_rotation[0])
                + left_size[1] * np.sin(body_rotation[0])
            )
            / 2,
            (
                left_size[1] * np.cos(body_rotation[0])
                - left_size[0] * np.sin(body_rotation[0])
            )
            / 2,
            -left_right_offset[0],
        ]
        self.geometry_position = np.array(mesh_1_position)
        self.geometry_rotation = np.array(mesh_1_rotation)


template2knowledge = {
    "Symmetrical_body": Symmetrical_body_Knowledge,
    "Asymmetrical_body": Asymmetrical_body_Knowledge,
}
