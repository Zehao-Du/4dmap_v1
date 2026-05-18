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


class Trifold_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio, up_side_ratio = manip_args if manip_args else (0, 0, 0)
        rot_angle = rot_ratio * np.pi / 2

        horizontal_thickness = obj.horizontal_thickness
        horizontal_length = obj.horizontal_length
        vertical_thickness = obj.vertical_thickness
        horizontal_rotation = obj.horizontal_rotation
        horizontal_separation = obj.horizontal_separation
        mounting_offset = obj.mounting_offset

        # refer to concept_template.Regular_drawer
        delta_y = (
            horizontal_separation[0]
            - horizontal_length[0] * np.sin(horizontal_rotation[0])
            + horizontal_length[1] * np.sin(horizontal_rotation[1])
        )
        delta_z = (
            mounting_offset[0]
            - horizontal_length[1] * np.cos(horizontal_rotation[1])
            + horizontal_length[0] * np.cos(horizontal_rotation[0])
        )
        vertical_length = (
            np.sqrt(delta_y * delta_y + delta_z * delta_z) + horizontal_thickness[1]
        )
        vertical_rotation = np.arctan(delta_z / delta_y)
        vertical_y_offset = (
            -horizontal_length[0] * np.sin(horizontal_rotation[0])
            - horizontal_length[1] * np.sin(horizontal_rotation[1])
        ) / 2
        vertical_z_offset = (
            horizontal_length[1] * np.cos(horizontal_rotation[1])
            + mounting_offset[0]
            + horizontal_length[0] * np.cos(horizontal_rotation[0])
        ) / 2

        if up_side_ratio >= 0:
            self.grasp_position = np.array(
                [
                    0,
                    vertical_length / 2 * trans_ratio * 0.7,
                    -vertical_thickness[1] / 2 - self.gripper_length,
                ]
            )
            self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            self.grasp_position = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_position
            )
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
            self.force_direction = np.array([0, 1, 0])  # forward

            self.geometry_position = [
                0,
                vertical_y_offset,
                -vertical_z_offset - vertical_thickness[1] / 2,
            ]
            self.geometry_rotation = [-vertical_rotation, 0, 0]
        else:
            self.grasp_position = np.array(
                [
                    0,
                    horizontal_thickness[1] / 2 + self.gripper_length,
                    horizontal_length[0] / 2 * trans_ratio * 0.7,
                ]
            )
            self.grasp_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            self.grasp_position = (
                Rot.from_euler("z", rot_angle, degrees=False).as_matrix()
                @ self.grasp_position
            )
            self.grasp_rotation = (
                Rot.from_euler("z", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
            self.force_direction = np.array([0, 1, 0])  # forward

            top_mesh_position = [
                0,
                horizontal_separation[0] / 2
                - horizontal_length[0] * np.sin(horizontal_rotation[0]) / 2,
                -mounting_offset[0]
                - horizontal_length[0] * np.cos(horizontal_rotation[0]) / 2,
            ]
            top_mesh_rotation = [-horizontal_rotation[0], 0, 0]
            self.geometry_position = top_mesh_position
            self.geometry_rotation = top_mesh_rotation


class Curved_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_ratio, angle_ratio = manip_args if manip_args else (0, 0)
        rot_angle = angle_ratio * np.pi / 2
        angle = obj.exist_angle[0] * (1 / 2 + rot_ratio / 3)

        self.grasp_position = np.array(
            [
                np.cos(angle)
                * (obj.radius[0] + obj.radius[1] / 2 + self.gripper_length),
                0,
                np.sin(angle)
                * (obj.radius[0] + obj.radius[1] / 2 + self.gripper_length),
            ]
        )
        self.force_direction = np.array([0, 1, 0])
        self.grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_rotation = (
            Rot.from_euler("zy", [rot_angle, -angle], degrees=False).as_matrix()
            @ self.grasp_rotation
        )
        # Refer to knowledge.concept_template.Curve_Handle
        self.geometry_position = [0, 0, 0]
        self.geometry_rotation = [np.pi, 0, -np.pi / 2]


class Ring_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        angle_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        angle_ratio = angle_ratio if angle_ratio > 0 else 0
        rot_angle = rot_ratio * np.pi / 2

        angle = obj.exist_angle[0] * (1 / 2 + angle_ratio / 3)
        self.grasp_position = np.array(
            [
                np.cos(angle) * (obj.size[0] + self.gripper_length),
                0,
                np.sin(angle) * (obj.size[0] + self.gripper_length),
            ]
        )
        self.grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

        self.grasp_rotation = (
            Rot.from_euler("z", rot_angle, degrees=False).as_matrix()
            @ self.grasp_rotation
        )
        # self.grasp_position = Rot.from_euler('x', rot_angle, degrees=False).as_matrix() @ self.grasp_position
        self.grasp_rotation = (
            Rot.from_euler("y", -angle, degrees=False).as_matrix() @ self.grasp_rotation
        )

        self.force_direction = np.array([-1, 0, 0])
        self.geometry_position = [0, 0, 0]
        self.geometry_rotation = [np.pi, 0, -np.pi / 2]


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

        vertical_length = obj.vertical_length
        vertical_separation = obj.vertical_separation

        self.grasp_position = np.array(
            [
                (vertical_separation[0] / 2) * np.sin(rot_angle_1),
                0,
                (vertical_separation[0] / 2 + self.gripper_length)
                * np.cos(rot_angle_1),
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            Rot.from_euler("xy", [rot_angle_2, rot_angle_1], degrees=False).as_matrix()
            @ self.grasp_rotation
        )
        self.force_direction = np.array([0, 0, 1])  # up

        curve_mesh_position = [0, vertical_length[0], 0]
        curve_mesh_rotation = [-np.pi / 2, np.pi / 2, 0]
        self.geometry_position = curve_mesh_position
        self.geometry_rotation = curve_mesh_rotation


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
                0,
                (obj.vertical_separation[0] + obj.vertical_size[0])
                / 2
                * np.sin(rot_angle_1),
                (obj.vertical_separation[0] + obj.vertical_size[0])
                / 2
                * np.cos(rot_angle_1)
                + self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.grasp_rotation = (
            Rot.from_euler("yx", [rot_angle_2, -rot_angle_1], degrees=False).as_matrix()
            @ self.grasp_rotation
        )

        self.force_direction = np.array([0, 0, 1])  # up

        self.geometry_position = [0, obj.vertical_size[1], 0]
        self.geometry_rotation = [-np.pi / 2, 0, 0]


class Cylindrical_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi * 2 / 3

        size = obj.size

        self.grasp_position = np.array(
            [
                0,
                size[2] * (1 / 6 + trans_ratio * 0.25),
                (size[1] + size[0]) / 2 + self.gripper_length,
            ]
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

        self.force_direction = np.array([0, 1, 0])  # upward

        self.geometry_position = [0, 0, -obj.size[2] / 2]
        self.geometry_rotation = [-np.pi / 2, 0, 0]


class Standard_Cover_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_ratio, up_side_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi

        def get_knob_size(idx):
            return getattr(self.obj, f"knob_{idx}_size")

        manip_pos_num = (int(obj.num_knobs[0]) - 1) + 1  # delete the last one
        manip_pos_idx = -1
        up_side_ratio = up_side_ratio + 1
        for i in range(manip_pos_num):
            if i * 2 / manip_pos_num <= up_side_ratio <= (i + 1) * 2 / manip_pos_num:
                manip_pos_idx = i
        assert manip_pos_idx != -1, "No pos"

        if manip_pos_idx == 0:  # top
            knob_idx = int(obj.num_knobs[0]) - 1
            mid_height = get_knob_size(knob_idx + 1)[2]
            self.grasp_position = np.array(
                [0, mid_height + self.gripper_length + self.finger_length / 2, 0]
            )
            self.grasp_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
        else:  # side
            knob_idx = (manip_pos_idx + 1) - 1  # delete the last one
            mid_radius = (
                get_knob_size(knob_idx + 1)[0] / 2 + get_knob_size(knob_idx + 1)[1] / 2
            )
            mid_height = get_knob_size(knob_idx + 1)[2] / 2
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

        # Refer to concept_template.Multiknob_Topknob
        delta_height = obj.outer_size[2]
        mesh_position = [0, 0, 0]
        for i in range(knob_idx):
            delta_height += get_knob_size(i + 1)[2]
            mesh_position = [0, delta_height, 0]
            if i == knob_idx:
                break

        self.force_direction = np.array([0, 1, 0])  # up

        self.geometry_position = mesh_position
        self.geometry_rotation = [0, 0, 0]


template2knowledge = {
    "Trifold_Handle": Trifold_Handle_Knowledge,
    "Curved_Handle": Curved_Handle_Knowledge,
    "Ring_Handle": Ring_Handle_Knowledge,
    "Flat_U_Handle": Flat_U_Handle_Knowledge,
    "Round_U_Handle": Round_U_Handle_Knowledge,
    "Cylindrical_Handle": Cylindrical_Handle_Knowledge,
    "Standard_Cover": Standard_Cover_Knowledge,
}
