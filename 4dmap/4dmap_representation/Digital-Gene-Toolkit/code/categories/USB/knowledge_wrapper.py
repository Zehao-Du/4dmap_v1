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


class Regular_Cap_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        size = obj.size
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        self.grasp_position = np.array(
            [trans_ratio * size[0] * 0.3, 0, size[2] + 2 * self.gripper_length]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )


class SquareEnded_Cap_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        size = obj.size
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        inclination = obj.inclination
        proximal_interval = obj.proximal_interval
        shaft_offset = obj.shaft_offset
        cap_rotation = obj.cap_rotation
        exist_angle = np.pi + inclination[0] * 2
        radius = proximal_interval[0] / 2 / np.cos(inclination[0]) + size[2] * np.tan(
            inclination[0]
        )
        back_position = [
            0,
            0,
            -radius * np.sin(inclination[0])
            - size[2] * np.cos(inclination[0])
            - shaft_offset[0],
        ]

        self.grasp_position = np.array(
            [0, trans_ratio * size[0] * 0.3, radius + size[1] + 2 * self.gripper_length]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.grasp_position = (
            self.grasp_position
            @ Rot.from_euler("x", -rot_angle, degrees=False).as_matrix()
        )
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )

        c_mesh_position = [
            back_position[0] * np.cos(cap_rotation[0])
            + back_position[2] * np.sin(cap_rotation[0]),
            back_position[1],
            back_position[0] * np.sin(cap_rotation[0])
            + back_position[2] * np.cos(cap_rotation[0])
            + shaft_offset[0],
        ]
        c_mesh_rotation = [0, np.pi, np.pi / 2]
        self.apply_transformation_for_geometry(
            [0, 0, 0], c_mesh_rotation, [c_mesh_position], [[0, cap_rotation[0], 0]]
        )


class RoundEnded_Cap_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        size = obj.size
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        inclination = obj.inclination
        proximal_interval = obj.proximal_interval
        shaft_offset = obj.shaft_offset
        cap_rotation = obj.cap_rotation
        exist_angle = np.pi + inclination[0] * 2
        radius = proximal_interval[0] / 2 / np.cos(inclination[0]) + size[2] * np.tan(
            inclination[0]
        )
        back_position = [
            0,
            0,
            -radius * np.sin(inclination[0])
            - size[2] * np.cos(inclination[0])
            - shaft_offset[0],
        ]

        self.grasp_position = np.array(
            [0, trans_ratio * size[0] * 0.3, radius + size[1] + 2 * self.gripper_length]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.grasp_position = (
            self.grasp_position
            @ Rot.from_euler("x", -rot_angle, degrees=False).as_matrix()
        )
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )

        c_mesh_position = [
            back_position[0] * np.cos(cap_rotation[0])
            + back_position[2] * np.sin(cap_rotation[0]),
            back_position[1],
            back_position[0] * np.sin(cap_rotation[0])
            + back_position[2] * np.cos(cap_rotation[0])
            + shaft_offset[0],
        ]
        c_mesh_rotation = [0, np.pi, np.pi / 2]
        self.apply_transformation_for_geometry(
            [0, 0, 0], c_mesh_rotation, [c_mesh_position], [[0, cap_rotation[0], 0]]
        )


template2knowledge = {
    "Regular_Cap": Regular_Cap_Knowledge,
    "SquareEnded_Cap": SquareEnded_Cap_Knowledge,
    "RoundEnded_Cap": RoundEnded_Cap_Knowledge,
}
