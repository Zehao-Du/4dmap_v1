import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

current_file_path = os.path.abspath(__file__)
sys.path.append(current_file_path)

from code.geometry_template import *
from code.utils import *

from .concept_template import *

current_file_path = os.path.abspath(__file__)
sys.path.append(current_file_path)
sys.path.append(os.path.dirname(os.path.dirname(current_file_path)))  # categories
from code.base_knowledge import KnowledgeWrapper


class Round_U_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2
    """we don't know info about the side of handle"""

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_angle_1, rot_angle_2 = manip_args if manip_args else (0, 0)
        rot_angle_1 = np.pi / 2 * rot_angle_1
        rot_angle_2 = np.pi / 4 * rot_angle_2
        self.grasp_position = np.array(
            [
                (obj.inner_radius[0] + obj.vertical_separation[0] / 2)
                * np.sin(rot_angle_1),
                0,
                (obj.inner_radius[0] + obj.vertical_separation[0] / 2)
                * np.cos(rot_angle_1)
                + self.gripper_length,
            ]
        )
        self.force_direction = np.array([0, 1, 0])  # up
        # self.gripper_open_width = obj.inner_radius[0]*2 * 1.5
        self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            Rot.from_euler("YX", [rot_angle_1, rot_angle_2], degrees=False).as_matrix()
            @ self.grasp_rotation
        )

        self.geometry_position = [0, obj.vertical_length[0], 0]
        self.geometry_rotation = [-np.pi / 2, 0, 0]


class Flat_U_Handle_Knowledge(KnowledgeWrapper):
    """we don't know info about the side of handle"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_angle_1, rot_angle_2 = manip_args if manip_args else (0, 0)
        rot_angle_1 = np.pi / 2 * rot_angle_1
        rot_angle_2 = np.pi / 4 * rot_angle_2
        self.grasp_position = np.array(
            [
                (obj.vertical_separation[0] + obj.vertical_size[0])
                / 2
                * np.sin(rot_angle_1),
                0,
                (obj.vertical_separation[0] + obj.vertical_size[0])
                / 2
                * np.cos(rot_angle_1)
                + self.gripper_length,
            ]
        )
        # self.gripper_open_width = obj.inner_radius[0]*2 * 1.5
        self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            Rot.from_euler("YX", [rot_angle_1, rot_angle_2], degrees=False).as_matrix()
            @ self.grasp_rotation
        )

        self.force_direction = np.array([0, 1, 0])  # up

        self.geometry_position = [0, obj.vertical_size[1], 0]
        self.geometry_rotation = [-np.pi / 2, 0, 0]


class Trifold_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 2

        vertical_separation = obj.vertical_separation
        vertical_length = obj.vertical_length
        vertical_thickness = obj.vertical_thickness
        horizontal_thickness = obj.horizontal_thickness
        vertical_rotation = obj.vertical_rotation

        # refer to concept_template.Regular_drawer
        delta_x = (
            vertical_separation[0]
            - vertical_length[0] * np.sin(vertical_rotation[0])
            + vertical_length[1] * np.sin(vertical_rotation[1])
        )
        delta_y = vertical_length[0] * np.cos(vertical_rotation[0]) - vertical_length[
            1
        ] * np.cos(vertical_rotation[1])
        horizontal_length = (
            np.sqrt(delta_y * delta_y + delta_x * delta_x) + vertical_thickness[0]
        )
        horizontal_rotation = np.arctan(delta_y / delta_x)
        vertical_x_offset = (
            -vertical_length[0] * np.sin(vertical_rotation[0])
            - vertical_length[1] * np.sin(vertical_rotation[1])
        ) / 2
        vertical_y_offset = (
            vertical_length[1] * np.cos(vertical_rotation[1])
            + vertical_length[0] * np.cos(vertical_rotation[0])
        ) / 2

        self.grasp_position = np.array(
            [
                horizontal_length / 2 * trans_ratio * 0.8,
                self.gripper_length + horizontal_thickness[0] / 2,
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
        self.force_direction = np.array([0, 1, 0])  # forward

        vertical_mesh_position = [
            vertical_x_offset,
            vertical_y_offset + horizontal_thickness[0] / 2,
            0,
        ]
        vertical_mesh_rotation = [0, 0, horizontal_rotation]
        self.geometry_position = vertical_mesh_position
        self.geometry_rotation = vertical_mesh_rotation


class Curved_Handle_Knowledge(KnowledgeWrapper):
    """
    Torus Handle,we don't know info about the side of handle
    """

    concept_rotation_order = "ZXY"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_angle_1, rot_angle_2 = manip_args if manip_args else (0, 0)
        rot_angle_1 = np.pi / 2 * rot_angle_1
        rot_angle_2 = np.pi / 4 * rot_angle_2
        angle = obj.exist_angle[0] / 2 + rot_angle_1

        self.grasp_position = np.array(
            [
                np.cos(angle)
                * (obj.radius[0] + obj.radius[1] / 2 + self.gripper_length),  #
                0,
                np.sin(angle)
                * (obj.radius[0] + obj.radius[1] / 2 + self.gripper_length),  #
            ]
        )
        self.force_direction = np.array([0, 1, 0])
        grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_rotation = (
            Rot.from_euler("Y", -angle, degrees=False).as_matrix() @ grasp_rotation
        )
        self.grasp_rotation = (
            Rot.from_euler("X", rot_angle_2, degrees=False).as_matrix()
            @ self.grasp_rotation
        )

        # refer to concept_template.Curve_Handle
        self.geometry_position = [0, 0, 0]
        self.geometry_rotation = [-np.pi / 2, 0, 0]


template2knowledge = {
    "Trifold_Handle": Trifold_Handle_Knowledge,
    "Curved_Handle": Curved_Handle_Knowledge,
    "Round_U_Handle": Round_U_Handle_Knowledge,
    "Flat_U_Handle": Flat_U_Handle_Knowledge,
}
