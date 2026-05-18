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


class Regular_drawer_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 4

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio, drawer_idx_ratio, handle_idx_ratio = (
            manip_args if manip_args else (0, 0, 0, 0)
        )
        rot_angle = rot_ratio * np.pi / 4
        drawer_idx_ratio, handle_idx_ratio = (
            (drawer_idx_ratio + 1) / 2,
            (handle_idx_ratio + 1) / 2,
        )

        front_size = obj.front_size
        drawer_size = obj.drawer_size
        handle_sizes = obj.handle_sizes
        handle_offset = obj.handle_offset
        handle_separation = obj.handle_separation
        drawer_offset = obj.drawer_offset
        number_of_drawer = obj.number_of_drawer
        number_of_handle = obj.number_of_handle

        drawer_idx = number_of_drawer[0] - 1
        handle_idx = number_of_handle[0] - 1
        for i in range(number_of_drawer[0]):
            if (
                i / number_of_drawer[0]
                <= drawer_idx_ratio
                < (i + 1) / number_of_drawer[0]
            ):
                drawer_idx = i
                break
        for i in range(number_of_handle[0]):
            if (
                i / number_of_handle[0]
                <= handle_idx_ratio
                < (i + 1) / number_of_handle[0]
            ):
                handle_idx = i
                break

        mesh_idx = 6 + handle_idx  # select the first handle mesh

        self.grasp_position = np.array(
            [
                trans_ratio * handle_sizes[drawer_idx][0] / 2 * 0.7,
                0,
                handle_sizes[drawer_idx][2] / 2 + self.gripper_length,
            ]
        )
        self.force_direction = np.array([0, 0, 1])  # forward
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_position = (
            Rot.from_euler("x", rot_angle).as_matrix() @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("x", rot_angle).as_matrix() @ self.grasp_rotation
        )

        # refer to concept_template.Regular_drawer
        if obj.number_of_handle[drawer_idx] == 2:
            position_sign = 1 if mesh_idx == 6 else -1
        else:
            position_sign = 0
        mesh_position = [
            drawer_offset[drawer_idx][0]
            + handle_offset[drawer_idx][0]
            + position_sign * handle_separation[drawer_idx] / 2,
            drawer_offset[drawer_idx][1] + handle_offset[drawer_idx][0],
            drawer_offset[drawer_idx][2]
            + drawer_size[drawer_idx][2] / 2
            + front_size[drawer_idx][2]
            + front_size[drawer_idx][2] / 2,
        ]
        self.geometry_position = mesh_position
        self.geometry_rotation = [0, 0, 0]


class Regular_door_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio, drawer_idx_ratio = (
            manip_args if manip_args else (0, 0, 0)
        )
        rot_angle = rot_ratio * np.pi / 4
        drawer_idx_ratio = (drawer_idx_ratio + 1) / 2

        handle_size = obj.handle_size
        handle_offset = obj.handle_offset
        door_rotation = obj.door_rotation
        door_offset = obj.door_offset
        number_of_door = obj.number_of_door

        door_idx = number_of_door[0] - 1
        for i in range(number_of_door[0]):
            if i / number_of_door[0] <= drawer_idx_ratio < (i + 1) / number_of_door[0]:
                door_idx = i
                break

        if handle_size[door_idx][1] > handle_size[door_idx][0]:
            self.grasp_position = [
                0,
                trans_ratio * handle_size[door_idx][1] / 2 * 0.7,
                handle_size[door_idx][2] / 2 + self.gripper_length,
            ]
            self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            self.grasp_position = (
                Rot.from_euler("y", rot_angle).as_matrix() @ self.grasp_position
            )
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle).as_matrix() @ self.grasp_rotation
            )
        else:
            self.grasp_position = [
                trans_ratio * handle_size[door_idx][0] / 2 * 0.7,
                0,
                handle_size[door_idx][2] / 2 + self.gripper_length,
            ]
            self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            self.grasp_position = (
                Rot.from_euler("x", rot_angle).as_matrix() @ self.grasp_position
            )
            self.grasp_rotation = (
                Rot.from_euler("x", rot_angle).as_matrix() @ self.grasp_rotation
            )
        self.force_direction = np.array([0, 0, 1])  # forward
        # Refer to knowledge.concept_template.Regular_Door

        mesh_rotation = [0, door_rotation[door_idx], 0]
        mesh_position = [
            door_offset[door_idx][0]
            + handle_offset[door_idx][0] * np.cos(door_rotation[door_idx])
            + handle_size[door_idx][2] / 2 * np.sin(door_rotation[door_idx]),
            door_offset[door_idx][1] + handle_offset[door_idx][1],
            door_offset[door_idx][2]
            - handle_offset[door_idx][0] * np.sin(door_rotation[door_idx])
            + handle_size[door_idx][2] / 2 * np.cos(door_rotation[door_idx]),
        ]

        self.geometry_position = mesh_position
        self.geometry_rotation = mesh_rotation


template2knowledge = {
    "Regular_door": Regular_door_Knowledge,
    "Regular_drawer": Regular_drawer_Knowledge,
}
