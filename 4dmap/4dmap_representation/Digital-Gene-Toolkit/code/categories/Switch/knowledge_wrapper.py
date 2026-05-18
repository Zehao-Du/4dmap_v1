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


class Round_Switch_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    offset_first = True
    manip_params_size = 2
    """push the button"""

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_ratio, idx_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 2
        idx_ratio = (idx_ratio + 1) / 2

        number_of_switch = obj.number_of_switch
        size = obj.size
        offset_1 = obj.offset_1
        offset_2 = obj.offset_2
        offset_3 = obj.offset_3
        offset_4 = obj.offset_4
        offset_Z = obj.offset_Z
        switch_rotation = obj.switch_rotation
        assert number_of_switch[0] >= 1, "Number of switch should be greater than 1"

        self.gripper_open_width = 0
        self.grasp_position = np.array(
            [0, size[1] / 2 + self.gripper_length + self.finger_length, 0]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.grasp_rotation = (
            Rot.from_euler("y", rot_angle).as_matrix() @ self.grasp_rotation
        )
        self.force_direction = np.array([0, -1, 0])  # push the button

        i = 0
        for j in range(number_of_switch[0]):
            if (
                1 / number_of_switch[0] * j
                < idx_ratio
                <= 1 / number_of_switch[0] * (j + 1)
            ):
                i = int(j)
                break

        base_mesh_position = [
            locals()["offset_%d" % (int(i + 1))][0],
            locals()["offset_%d" % (int(i + 1))][1],
            offset_Z[0],
        ]
        base_mesh_rotation = [np.pi / 2 + switch_rotation[0], 0, 0]
        self.geometry_position = base_mesh_position
        self.geometry_rotation = base_mesh_rotation


class FlipX_Switch_Knowledge(KnowledgeWrapper):
    "push the button"

    concept_rotation_order = "XYZ"
    offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_ratio, idx_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 2
        idx_ratio = (idx_ratio + 1) / 2

        number_of_switch = obj.number_of_switch
        switch_rotation = obj.switch_rotation
        separation = obj.separation
        size = obj.size
        assert number_of_switch[0] >= 1, "must be positive"

        self.gripper_open_width = 0
        is_upper = 2 * int(switch_rotation[0] > 0) - 1

        self.grasp_position = np.array(
            [
                0,
                is_upper * size[1] / 4,
                size[2] / 2 + self.gripper_length + self.finger_length,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            Rot.from_euler("z", rot_angle).as_matrix() @ self.grasp_rotation
        )
        self.force_direction = np.array([0, 0, -1])

        i = 0  # select the first switch
        for j in range(number_of_switch[0]):
            if (
                1 / number_of_switch[0] * j
                < idx_ratio
                <= 1 / number_of_switch[0] * (j + 1)
            ):
                i = int(j)
                break

        mesh_position = [(separation[0] + size[0]) * i, 0, 0]
        mesh_rotation = [switch_rotation[0], 0, 0]
        self.geometry_position = mesh_position
        self.geometry_rotation = mesh_rotation


class FlipY_Switch_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    offset_first = True
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio, idx_ratio = manip_args if manip_args else (0, 0, 0)
        rot_angle = rot_ratio * np.pi / 6
        idx_ratio = (idx_ratio + 1) / 2

        number_of_switch = obj.number_of_switch
        switch_rotation = obj.switch_rotation
        separation = obj.separation
        size = obj.size
        assert number_of_switch[0] >= 1, "must be positive"

        self.gripper_open_width = 0
        is_left = 2 * int(switch_rotation[0] < 0) - 1

        self.grasp_position = np.array(
            [
                is_left * size[0] / 4,
                trans_ratio * size[1] / 2 * 0.6,
                size[2] / 2 + self.gripper_length + self.finger_length,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            Rot.from_euler("z", rot_angle).as_matrix() @ self.grasp_rotation
        )
        self.force_direction = np.array([0, 0, -1])

        i = 0
        for j in range(number_of_switch[0]):
            if (
                1 / number_of_switch[0] * j
                < idx_ratio
                <= 1 / number_of_switch[0] * (j + 1)
            ):
                i = int(j)
                break
        mesh_position = [(separation[0] + size[0]) * i, 0, 0]
        mesh_rotation = [0, switch_rotation[0], 0]
        self.geometry_position = mesh_position
        self.geometry_rotation = mesh_rotation


class Lever_Switch_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    concept_offset_first = True
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_ratio_1, rot_ratio_2, idx_ratio = manip_args if manip_args else (0, 0, 0)
        rot_angle_1 = rot_ratio_1 * np.pi / 2
        rot_angle_2 = -(rot_ratio_2 + 1) / 2 * np.pi / 4

        number_of_switch = obj.number_of_switch
        inter_offset = obj.inter_offset
        base_size = obj.base_size
        main_size = obj.main_size
        switch_rotation = obj.switch_rotation
        separation = obj.separation
        main_size = obj.main_size

        self.grasp_position = np.array(
            [0, main_size[2] / 4, main_size[0] + self.gripper_length]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.grasp_position = (
            Rot.from_euler("xy", [rot_angle_2, rot_angle_1]).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("xy", [rot_angle_2, rot_angle_1]).as_matrix()
            @ self.grasp_rotation
        )

        self.force_direction = np.array([0, 0, 1])

        i = 0
        for j in range(number_of_switch[0]):
            if (
                1 / number_of_switch[0] * j
                < idx_ratio
                <= 1 / number_of_switch[0] * (j + 1)
            ):
                i = int(j)
                break
        main_mesh_position = [
            (separation[0] + main_size[0]) * i,
            0,
            main_size[2] / 2,
        ]
        main_mesh_rotation = [np.pi / 2, 0, 0]
        self.geometry_position = main_mesh_position
        self.geometry_rotation = [0, 0, 0]

        first_apply_position = [
            inter_offset[0],
            inter_offset[1],
            inter_offset[2] + base_size[1],
        ]
        first_apply_rotation = [switch_rotation[0], 0, 0]

        self.apply_transformation_for_geometry(
            main_mesh_position,
            main_mesh_rotation,
            [first_apply_position],
            [first_apply_rotation],
        )


template2knowledge = {
    "Round_Switch": Round_Switch_Knowledge,
    "Lever_Switch": Lever_Switch_Knowledge,
    "FlipX_Switch": FlipX_Switch_Knowledge,
    "FlipY_Switch": FlipY_Switch_Knowledge,
}
