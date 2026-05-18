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


class Cuboidal_Tophandle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "ZXY"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 2

        size = obj.size

        self.grasp_position = np.array(
            [trans_ratio * size[0] / 2 * 0.8, size[1] / 2 + self.gripper_length, 0]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.grasp_position = (
            Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
            @ self.grasp_rotation
        )

        self.force_direction = np.array([0, 1, 0])  # up

        self.geometry_position = [0, size[1] / 2, 0]


class Trifold_Tophandle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "ZXY"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 2

        grip_size = obj.grip_size

        self.grasp_position = np.array(
            [
                trans_ratio * grip_size[0] / 2 * 0.8,
                grip_size[1] / 2 + self.gripper_length,
                0,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.grasp_position = (
            Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
            @ self.grasp_rotation
        )

        self.force_direction = np.array([0, 1, 0])  # up
        # refer to concept_template.Trifold_Handle
        self.geometry_position = [
            0,
            obj.mounting_size[1] * np.cos(obj.mounting_rotation[0]) + grip_size[1] / 2,
            0,
        ]
        self.geometry_rotation = [0, 0, 0]


class Semi_Ring_Tophandle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    geometry_rotation_order = "YXZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        angle_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 2
        angle = obj.curve_exist_angle[0] * (1 / 2 + angle_ratio / 4)

        self.grasp_position = np.array(
            [
                np.cos(angle) * (obj.curve_size[0] + self.gripper_length),
                0,
                np.sin(angle) * (obj.curve_size[0] + self.gripper_length),
            ]
        )
        self.force_direction = np.array(
            [
                np.cos(obj.curve_exist_angle[0] / 2),
                0,
                np.sin(obj.curve_exist_angle[0] / 2),
            ]
        )
        self.grasp_rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        self.grasp_rotation = (
            Rot.from_euler("Y", -angle, degrees=False).as_matrix() @ self.grasp_rotation
        )
        self.grasp_rotation = (
            Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
            @ self.grasp_rotation
        )

        # refer to concept_template.Semi_Ring_Tophandle
        self.geometry_position = [
            0,
            -obj.curve_size[0] * np.cos(obj.curve_exist_angle[0] / 2),
            0,
        ]
        self.geometry_rotation = [
            -np.pi / 2,
            -np.pi / 2 + obj.curve_exist_angle[0] / 2,
            0,
        ]


class Multilevel_Tophandle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_ratio, up_side_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi

        def get_level_size(idx):
            return getattr(self.obj, f"level_{idx}_size")

        manip_pos_num = (obj.num_levels[0] - 1) + 1  # exclude the first layer
        manip_pos_idx = -1
        up_side_ratio = up_side_ratio + 1
        for i in range(manip_pos_num):
            if i * 2 / manip_pos_num <= up_side_ratio <= (i + 1) * 2 / manip_pos_num:
                manip_pos_idx = i
        assert manip_pos_idx != -1, "No pos"

        if manip_pos_idx == 0:  # top
            level_idx = obj.num_levels[0] - 1
            mid_height = get_level_size(level_idx + 1)[2]

            self.grasp_position = np.array([0, mid_height + self.gripper_length, 0])
            self.grasp_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
        else:
            level_idx = (manip_pos_idx - 1) + 1  # exclude the first layer
            mid_radius = (
                get_level_size(level_idx + 1)[0] / 2
                + get_level_size(level_idx + 1)[1] / 2
            )
            mid_height = get_level_size(level_idx + 1)[2] / 2
            self.grasp_position = np.array(
                [0, mid_height, mid_radius + self.gripper_length]
            )
            self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            self.grasp_position = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_position
            )
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )

        # Refer to concept_template.Multilevel_Toplevel
        delta_height = 0
        mesh_position = [0, 0, 0]
        for i in range(level_idx):
            delta_height += get_level_size(i + 1)[2]
            mesh_position = [0, delta_height, 0]
            if i == level_idx:
                break

        self.force_direction = np.array([0, 1, 0])  # up

        self.geometry_position = mesh_position
        self.geometry_rotation = [0, 0, 0]


template2knowledge = {
    "Cuboidal_Tophandle": Cuboidal_Tophandle_Knowledge,
    "Trifold_Tophandle": Trifold_Tophandle_Knowledge,
    "Multilevel_Tophandle": Multilevel_Tophandle_Knowledge,
    "Semi_Ring_Tophandle": Semi_Ring_Tophandle_Knowledge,
}
