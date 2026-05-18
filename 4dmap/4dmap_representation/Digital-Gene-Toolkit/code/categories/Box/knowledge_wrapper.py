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


class Fourfold_Cover_Knowledge(KnowledgeWrapper):
    """open cover"""

    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio, cover_rot_ratio = (
            manip_args if manip_args else (0, 0, 0)
        )

        # Refer to concept_template.Fourfold_Cover
        has_cover = obj.has_cover
        cover_separation = obj.cover_separation
        cover_rotation = obj.cover_rotation
        front_behind_size = obj.front_behind_size
        left_right_size = obj.left_right_size

        cover_num = np.sum(has_cover)
        cover_index = [index for index, value in enumerate(has_cover) if value == 1]
        manip_cover_idx = -1
        cover_rot_ratio = (cover_rot_ratio + 1) / 2
        for i in range(cover_num):
            if i / cover_num <= cover_rot_ratio <= (i + 1) / cover_num:
                manip_cover_idx = cover_index[i]
        assert manip_cover_idx != -1, f"No cover: {has_cover} {cover_rot_ratio}"

        if manip_cover_idx == 0:
            self.grasp_position = np.array(
                [
                    trans_ratio * front_behind_size[0] / 2 * 0.8,
                    front_behind_size[1] / 2 + self.gripper_length,
                    0,
                ]
            )
            self.grasp_rotation = np.array(
                [
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0],
                ]
            )
            self.grasp_rotation = (
                Rot.from_euler(
                    "x",
                    rot_ratio * np.pi / 10,
                    degrees=False,
                ).as_matrix()
                @ self.grasp_rotation
            )
            self.force_direction = np.array([0, 0, 1])

            mesh_position = [
                0,
                front_behind_size[1] * np.cos(cover_rotation[0]) / 2,
                cover_separation[0] / 2
                + front_behind_size[1] * np.sin(cover_rotation[0]) / 2,
            ]
            mesh_rotation = [cover_rotation[0], 0, 0]
            self.geometry_position = mesh_position
            self.geometry_rotation = mesh_rotation

        elif manip_cover_idx == 1:
            self.grasp_position = np.array(
                [
                    trans_ratio * front_behind_size[0] / 2 * 0.8,
                    front_behind_size[1] / 2 + self.gripper_length,
                    0,
                ]
            )
            self.grasp_rotation = np.array(
                [
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0],
                ]
            )
            self.grasp_rotation = (
                Rot.from_euler(
                    "x",
                    rot_ratio * np.pi / 10,
                    degrees=False,
                ).as_matrix()
                @ self.grasp_rotation
            )
            self.force_direction = np.array([0, 0, -1])

            mesh_position = [
                0,
                front_behind_size[1] * np.cos(cover_rotation[1]) / 2,
                -cover_separation[0] / 2
                + front_behind_size[1] * np.sin(cover_rotation[1]) / 2,
            ]
            mesh_rotation = [cover_rotation[1], 0, 0]

            self.geometry_position = mesh_position
            self.geometry_rotation = mesh_rotation

        elif manip_cover_idx == 2:
            self.grasp_position = np.array(
                [
                    0,
                    left_right_size[1] / 2 + self.gripper_length,
                    trans_ratio * left_right_size[2] / 2 * 0.8,
                ]
            )
            self.grasp_rotation = np.array(
                [
                    [0, 1, 0],
                    [0, 0, -1],
                    [-1, 0, 0],
                ]
            )
            self.grasp_rotation = (
                Rot.from_euler(
                    "z",
                    rot_ratio * np.pi / 10,
                    degrees=False,
                ).as_matrix()
                @ self.grasp_rotation
            )
            self.force_direction = np.array([1, 0, 0])

            mesh_position = [
                -cover_separation[1] / 2
                - left_right_size[1] * np.sin(cover_rotation[2]) / 2,
                left_right_size[1] * np.cos(cover_rotation[2]) / 2,
                0,
            ]
            mesh_rotation = [0, 0, cover_rotation[2]]

            self.geometry_position = mesh_position
            self.geometry_rotation = mesh_rotation

        elif manip_cover_idx == 3:
            self.grasp_position = np.array(
                [
                    0,
                    left_right_size[1] / 2 + self.gripper_length,
                    trans_ratio * left_right_size[2] / 2 * 0.8,
                ]
            )
            self.grasp_rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
            self.grasp_rotation = (
                Rot.from_euler(
                    "z",
                    rot_ratio * np.pi / 10,
                    degrees=False,
                ).as_matrix()
                @ self.grasp_rotation
            )
            self.force_direction = np.array([-1, 0, 0])

            mesh_position = [
                cover_separation[1] / 2
                - left_right_size[1] * np.sin(cover_rotation[3]) / 2,
                left_right_size[1] * np.cos(cover_rotation[3]) / 2,
                0,
            ]
            mesh_rotation = [0, 0, cover_rotation[3]]

            self.geometry_position = mesh_position
            self.geometry_rotation = mesh_rotation
        else:
            raise ValueError("No cover")


class Regular_Cover_Knowledge(KnowledgeWrapper):
    """open cover"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)

        inner_size = obj.inner_size
        outer_size = obj.outer_size
        self.grasp_position = np.array(
            [
                trans_ratio * outer_size[0] / 2 * 0.8,
                0,
                outer_size[2] / 2 + self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            Rot.from_euler(
                "x",
                rot_ratio * np.pi / 10,
                degrees=False,
            ).as_matrix()
            @ self.grasp_rotation
        )
        self.force_direction = np.array([0, 1, 0])

        # Refer to concept_template.Regular_Cover
        bottom_mesh_position = [0, inner_size[1] / 2, outer_size[2] / 2]
        self.geometry_position = bottom_mesh_position


template2knowledge = {
    "Fourfold_Cover": Fourfold_Cover_Knowledge,
    "Regular_Cover": Regular_Cover_Knowledge,
}
