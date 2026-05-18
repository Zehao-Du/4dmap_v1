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


class Straight_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        grasp_pos_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = (rot_ratio - 1) * np.pi / 6

        front_size = obj.front_size
        behind_size = obj.behind_size
        handle_separation = obj.handle_separation
        handle_rotation = obj.handle_rotation
        front_behind_offset = obj.front_behind_offset
        left_right_offset = obj.left_right_offset

        # behind_left
        behind_1_mesh_position_1 = [-behind_size[0] / 2, 0, -behind_size[2] / 2]
        behind_1_mesh_rotation = [0, -handle_rotation[1], 0]
        behind_1_mesh_position_1 = adjust_position_from_rotation(
            behind_1_mesh_position_1, behind_1_mesh_rotation
        )

        behind_1_mesh_position_2 = [
            handle_separation[0] / 2
            + front_size[0] / 2 * np.cos(handle_rotation[0])
            + front_size[2] * np.sin(handle_rotation[0])
            + front_behind_offset[0],
            front_behind_offset[1],
            -front_size[2] * np.cos(handle_rotation[0])
            + front_size[0] / 2 * np.sin(handle_rotation[0]),
        ]
        behind_1_mesh_position = list_add(
            behind_1_mesh_position_1, behind_1_mesh_position_2
        )

        # behind_right
        behind_2_mesh_position_1 = [behind_size[0] / 2, 0, -behind_size[2] / 2]
        behind_2_mesh_rotation = [0, handle_rotation[1], 0]
        behind_2_mesh_position_1 = adjust_position_from_rotation(
            behind_2_mesh_position_1, behind_2_mesh_rotation
        )

        behind_2_mesh_position_2 = [
            -handle_separation[0] / 2
            - front_size[0] / 2 * np.cos(handle_rotation[0])
            - front_size[2] * np.sin(handle_rotation[0])
            - front_behind_offset[0],
            front_behind_offset[1] + left_right_offset[0],
            -front_size[2] * np.cos(handle_rotation[0])
            + front_size[0] / 2 * np.sin(handle_rotation[0]),
        ]
        behind_2_mesh_position = list_add(
            behind_2_mesh_position_1, behind_2_mesh_position_2
        )

        self.grasp_position = np.array(
            [
                0,
                behind_size[1] / 2 + self.gripper_length,
                grasp_pos_ratio * behind_size[2] * 0.3,
            ]
        )
        self.grasp_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        # self.geometry_position = (np.array(behind_1_mesh_position) + np.array(behind_2_mesh_position))/2
        self.grasp_position = (
            Rot.from_euler("z", rot_angle).as_matrix() @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("z", rot_angle).as_matrix() @ self.grasp_rotation
        )
        self.geometry_position = behind_1_mesh_position
        self.geometry_rotation = behind_1_mesh_rotation


class Rear_Curved_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        grasp_pos_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = (rot_ratio + 1) * np.pi / 6

        front_size = obj.front_size
        behind_size = obj.behind_size
        exist_angle = obj.exist_angle
        handle_separation = obj.handle_separation
        handle_rotation = obj.handle_rotation
        front_behind_offset = obj.front_behind_offset
        left_right_offset = obj.left_right_offset

        # behind_left
        behind_1_mesh_rotation_1 = [0, np.pi, np.pi]
        behind_1_mesh_position_1 = [-behind_size[1], 0, 0]
        behind_1_mesh_rotation_2 = [0, -handle_rotation[1], 0]
        behind_1_mesh_rotation_2_reverse = [0, handle_rotation[1], 0]
        behind_1_mesh_position_1 = adjust_position_from_rotation(
            behind_1_mesh_position_1, behind_1_mesh_rotation_2
        )

        behind_1_mesh_position_2 = [
            handle_separation[0] / 2
            - front_size[0] / 2 * np.cos(handle_rotation[0])
            + front_size[2] * np.sin(handle_rotation[0])
            + front_behind_offset[0],
            front_behind_offset[1],
            -front_size[2] * np.cos(handle_rotation[0])
            + front_size[0] / 2 * np.sin(handle_rotation[0]),
        ]
        behind_1_mesh_position = list_add(
            behind_1_mesh_position_1, behind_1_mesh_position_2
        )
        behind_1_mesh_rotation = list_add(
            behind_1_mesh_rotation_1, behind_1_mesh_rotation_2_reverse
        )

        # behind_right
        behind_2_mesh_rotation_1 = [0, np.pi, 0]
        behind_2_mesh_position_1 = [behind_size[1], 0, 0]
        behind_2_mesh_rotation_2 = [0, handle_rotation[1], 0]
        behind_2_mesh_position_1 = adjust_position_from_rotation(
            behind_2_mesh_position_1, behind_2_mesh_rotation_2
        )

        behind_2_mesh_position_2 = [
            -handle_separation[0] / 2
            + front_size[0] / 2 * np.cos(handle_rotation[0])
            - front_size[2] * np.sin(handle_rotation[0])
            - front_behind_offset[0],
            front_behind_offset[1],
            -front_size[2] * np.cos(handle_rotation[0])
            + front_size[0] / 2 * np.sin(handle_rotation[0]),
        ]
        behind_2_mesh_position = list_add(
            behind_2_mesh_position_1, behind_2_mesh_position_2
        )
        behind_2_mesh_rotation = list_add(
            behind_2_mesh_rotation_1, behind_2_mesh_rotation_2
        )

        curve_length = 2 * behind_size[0] * np.sin(exist_angle[0] / 2)

        self.grasp_position = np.array(
            [
                0,
                -behind_size[2] / 2 - self.gripper_length,
                grasp_pos_ratio * curve_length * 0.2,
            ]
        )
        self.grasp_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        # self.geometry_position = (np.array(behind_1_mesh_position) + np.array(behind_2_mesh_position))/2

        self.grasp_position += np.array([behind_size[0], 0, 0])
        self.grasp_position = (
            Rot.from_euler("Y", -exist_angle[0] / 2, degrees=False).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("Y", -exist_angle[0] / 2, degrees=False).as_matrix()
            @ self.grasp_rotation
        )
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", -np.pi).as_matrix()
        )
        self.geometry_position = behind_1_mesh_position
        self.geometry_rotation = behind_1_mesh_rotation


class Middle_Curved_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        grasp_pos_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = (rot_ratio + 1) * np.pi / 6

        front_size = obj.front_size
        middle_size = obj.middle_size
        exist_angle = obj.exist_angle
        behind_size = obj.behind_size
        handle_separation = obj.handle_separation
        handle_rotation = obj.handle_rotation
        front_middle_offset = obj.front_middle_offset
        middle_behind_offset = obj.middle_behind_offset
        left_right_offset = obj.left_right_offset

        # middle_left
        middle_1_mesh_rotation_1 = [0, np.pi, np.pi]
        middle_1_mesh_position_1 = [-middle_size[1], 0, 0]
        middle_1_mesh_rotation_2 = [0, -handle_rotation[1], 0]
        middle_1_mesh_rotation_2_reverse = [0, handle_rotation[1], 0]
        middle_1_mesh_position_1 = adjust_position_from_rotation(
            middle_1_mesh_position_1, middle_1_mesh_rotation_2
        )

        middle_1_mesh_position_2 = [
            handle_separation[0] / 2
            - front_size[0] / 2 * np.cos(handle_rotation[0])
            + front_size[2] * np.sin(handle_rotation[0])
            + front_middle_offset[0],
            front_middle_offset[1],
            -front_size[2] * np.cos(handle_rotation[0])
            + front_size[0] / 2 * np.sin(handle_rotation[0]),
        ]
        middle_1_mesh_position = list_add(
            middle_1_mesh_position_1, middle_1_mesh_position_2
        )
        middle_1_mesh_rotation = list_add(
            middle_1_mesh_rotation_1, middle_1_mesh_rotation_2_reverse
        )

        # middle_right
        middle_2_mesh_rotation_1 = [0, np.pi, 0]
        middle_2_mesh_position_1 = [middle_size[1], 0, 0]
        middle_2_mesh_rotation_2 = [0, handle_rotation[1], 0]
        middle_2_mesh_position_1 = adjust_position_from_rotation(
            middle_2_mesh_position_1, middle_2_mesh_rotation_2
        )

        middle_2_mesh_position_2 = [
            -handle_separation[0] / 2
            + front_size[0] / 2 * np.cos(handle_rotation[0])
            - front_size[2] * np.sin(handle_rotation[0])
            - front_middle_offset[0],
            front_middle_offset[1],
            -front_size[2] * np.cos(handle_rotation[0])
            + front_size[0] / 2 * np.sin(handle_rotation[0]),
        ]
        middle_2_mesh_position = list_add(
            middle_2_mesh_position_1, middle_2_mesh_position_2
        )
        middle_2_mesh_rotation = list_add(
            middle_2_mesh_rotation_1, middle_2_mesh_rotation_2
        )

        curve_length = 2 * middle_size[0] * np.sin(exist_angle[0] / 2)

        # self.geometry_position = (np.array(middle_1_mesh_position) + np.array(middle_2_mesh_position))/2

        self.grasp_position = np.array(
            [
                0,
                -middle_size[2] / 2 - self.gripper_length,
                (grasp_pos_ratio + 0.5) * curve_length * 0.2,
            ]
        )
        self.grasp_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

        self.grasp_position += np.array([middle_size[0], 0, 0])
        self.grasp_position = (
            Rot.from_euler("Y", -exist_angle[0] / 2, degrees=False).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("Y", -exist_angle[0] / 2, degrees=False).as_matrix()
            @ self.grasp_rotation
        )
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", -np.pi).as_matrix()
        )
        self.geometry_position = middle_1_mesh_position
        self.geometry_rotation = middle_1_mesh_rotation


class Asymmetric_Straight_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        grasp_pos_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = (rot_ratio - 1) * np.pi / 6

        left_front_size = obj.left_front_size
        left_behind_size = obj.left_behind_size
        left_handle_rotation = obj.left_handle_rotation
        left_front_behind_offset = obj.left_front_behind_offset
        right_front_size = obj.right_front_size
        right_behind_size = obj.right_behind_size
        right_handle_rotation = obj.right_handle_rotation
        right_front_behind_offset = obj.right_front_behind_offset
        handle_separation = obj.handle_separation
        left_right_offset = obj.left_right_offset

        # behind_left
        behind_1_mesh_position_1 = [
            left_behind_size[0] / 2,
            0,
            -left_behind_size[2] / 2,
        ]
        behind_1_mesh_rotation = [0, left_handle_rotation[1], 0]
        behind_1_mesh_position_1 = adjust_position_from_rotation(
            behind_1_mesh_position_1, behind_1_mesh_rotation
        )

        behind_1_mesh_position_2 = [
            -handle_separation[0] / 2
            - left_front_size[0] / 2 * np.cos(left_handle_rotation[0])
            - left_front_size[2] * np.sin(left_handle_rotation[0])
            - left_front_behind_offset[0],
            left_front_behind_offset[1],
            -left_front_size[2] * np.cos(left_handle_rotation[0])
            + left_front_size[0] / 2 * np.sin(left_handle_rotation[0]),
        ]
        behind_1_mesh_position = list_add(
            behind_1_mesh_position_1, behind_1_mesh_position_2
        )

        # behind_right
        behind_2_mesh_position_1 = [
            -right_behind_size[0] / 2,
            0,
            -right_behind_size[2] / 2,
        ]
        behind_2_mesh_rotation = [0, -right_handle_rotation[1], 0]
        behind_2_mesh_position_1 = adjust_position_from_rotation(
            behind_2_mesh_position_1, behind_2_mesh_rotation
        )

        behind_2_mesh_position_2 = [
            handle_separation[0] / 2
            + right_front_size[0] / 2 * np.cos(right_handle_rotation[0])
            + right_front_size[2] * np.sin(right_handle_rotation[0])
            + right_front_behind_offset[0],
            right_front_behind_offset[1] + left_right_offset[0],
            -right_front_size[2] * np.cos(right_handle_rotation[0])
            + right_front_size[0] / 2 * np.sin(right_handle_rotation[0]),
        ]
        behind_2_mesh_position = list_add(
            behind_2_mesh_position_1, behind_2_mesh_position_2
        )

        self.grasp_position = np.array(
            [
                0,
                right_behind_size[1] / 2 + self.gripper_length,
                grasp_pos_ratio * right_behind_size[2] * 0.2,
            ]
        )
        self.grasp_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        # self.geometry_position = (np.array(behind_2_mesh_position) + np.array(behind_1_mesh_position))/2

        self.grasp_position = (
            Rot.from_euler("z", rot_angle).as_matrix() @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("z", rot_angle).as_matrix() @ self.grasp_rotation
        )
        self.geometry_position = behind_2_mesh_position
        self.geometry_rotation = behind_2_mesh_rotation


template2knowledge = {
    "Straight_Handle": Straight_Handle_Knowledge,
    "Rear_Curved_Handle": Rear_Curved_Handle_Knowledge,
    "Middle_Curved_Handle": Middle_Curved_Handle_Knowledge,
    "Asymmetric_Straight_Handle": Asymmetric_Straight_Handle_Knowledge,
}
