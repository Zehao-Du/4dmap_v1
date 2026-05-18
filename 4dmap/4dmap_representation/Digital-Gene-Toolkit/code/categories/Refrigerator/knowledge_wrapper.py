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


class Trifold_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "ZXY"
    manip_params_size = 1

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        up_ratio = manip_args[0] if manip_args else 0

        self.grasp_position = np.array(
            [
                0,
                up_ratio * obj.grip_size[1] * 0.4,
                obj.grip_size[2] / 2 + self.gripper_length,
            ]
        )

        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        self.force_direction = np.array([0, 0, 1])
        if self.primact_type == "pushing":
            self.force_direction = -self.force_direction

        # refer to concept_template.Trifold_Handle
        self.geometry_position = [0, 0, obj.mounting_size[2] + obj.grip_size[2] / 2]
        self.geometry_rotation = [0, 0, 0]


class Curve_Handle_Knowledge(KnowledgeWrapper):
    """default pull, open the door"""

    concept_rotation_order = "ZXY"
    manip_params_size = 1

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        curve_angle_ratio = manip_args[0] if manip_args else 0
        angle = obj.curve_exist_angle[0] * (1 + curve_angle_ratio * 0.7) / 2

        self.grasp_position = np.array(
            [
                np.cos(angle) * (obj.curve_size[0] + self.gripper_length),
                0,
                np.sin(angle) * (obj.curve_size[0] + self.gripper_length),
            ]
        )
        grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_rotation = (
            Rot.from_euler("Y", -angle, degrees=False).as_matrix() @ grasp_rotation
        )

        self.force_direction = np.array(
            [
                np.cos(obj.curve_exist_angle[0] / 2),
                0,
                np.sin(obj.curve_exist_angle[0] / 2),
            ]
        )
        if self.primact_type == "pushing":
            self.force_direction = -self.force_direction

        # refer to concept_template.Curve_Handle
        self.geometry_position = [
            0,
            0,
            -obj.curve_size[0] * np.cos(obj.curve_exist_angle[0] / 2),
        ]
        self.geometry_rotation = [
            0,
            -np.pi / 2 + obj.curve_exist_angle[0] / 2,
            np.pi / 2,
        ]


class Trifold_Curve_Handle_Knowledge(KnowledgeWrapper):
    """default pull, open the door"""

    concept_rotation_order = "ZXY"
    manip_params_size = 1

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        curve_angle_ratio = manip_args[0] if manip_args else 0
        angle = obj.curve_exist_angle[0] * (1 + curve_angle_ratio * 0.7) / 2
        self.grasp_position = np.array(
            [
                np.cos(angle) * (obj.curve_size[0] + self.gripper_length),
                0,
                np.sin(angle) * (obj.curve_size[0] + self.gripper_length),
            ]
        )
        grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_rotation = (
            Rot.from_euler("Y", -angle, degrees=False).as_matrix() @ grasp_rotation
        )
        # the force direction is the same the grasp_position
        self.force_direction = np.array(
            [
                np.cos(obj.curve_exist_angle[0] / 2),
                0,
                np.sin(obj.curve_exist_angle[0] / 2),
            ]
        )
        if self.primact_type == "pushing":
            self.force_direction = -self.force_direction

        # refer to concept_template.Trifold_Curve_Handle
        curve_z_offset = obj.mounting_size[2] - np.sqrt(
            obj.curve_size[1] * obj.curve_size[1]
            - (obj.mounting_seperation[0] / 2) * (obj.mounting_seperation[0] / 2)
        )
        self.geometry_position = [0, 0, curve_z_offset]
        self.geometry_rotation = [
            0,
            -np.pi / 2 + obj.curve_exist_angle[0] / 2,
            np.pi / 2,
        ]


class Cuboidal_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "ZXY"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        up_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 6

        self.grasp_position = np.array(
            [0, up_ratio * obj.size[1] * 0.3, obj.size[2] / 2 + self.gripper_length]
        )
        # for two door
        self.gripper_open_width = obj.size[0] * 1.5
        self.force_direction = np.array([0, 0, 1])
        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.grasp_position = (
            Rot.from_euler("Y", rot_angle, degrees=False).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("Y", rot_angle, degrees=False).as_matrix()
            @ self.grasp_rotation
        )

        # refer to concept_template.Cuboidal_Handle
        self.geometry_position = [0, 0, obj.size[2] / 2]
        self.geometry_rotation = [0, 0, 0]


class Cuboidal_Door_Knowledge(KnowledgeWrapper):
    """open from the top or the right side"""

    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        if self.primact_type == "pulling":
            top_ratio, side_ratio, rot_side_ratio = (
                manip_args if manip_args else (0, 0, 0)
            )
            if rot_side_ratio > 0:  # up
                self.gripper_open_width = obj.size[2] * 1.5
                self.grasp_position = np.array(
                    [
                        obj.size[0] * (1 / 3 + side_ratio * 1 / 7),
                        obj.size[1] / 2 + self.gripper_length,
                        0,
                    ]
                )
                self.grasp_rotation = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            else:  # right
                self.grasp_position = np.array(
                    [
                        obj.size[0] / 2 + self.gripper_length,
                        top_ratio * obj.size[1] * 0.25,
                        obj.size[2] / 2,
                    ]
                )
                self.grasp_rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
            self.force_direction = np.array([0, 0, 1])
        else:  # self.primact_type == "pushing":
            up_ratio, horizontal_ratio, rot_ratio = (
                manip_args if manip_args else (0, 0, 0)
            )
            self.force_direction = np.array([0, 0, 1])
            self.gripper_open_width = 0
            self.grasp_position = np.array(
                [
                    obj.size[0] * (1 / 3 + horizontal_ratio * 1 / 7),
                    up_ratio * obj.size[1] * 0.25,
                    obj.size[2] / 2 + self.gripper_length + self.finger_length,
                ]
            )
            self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

            angle = np.pi / 2 * rot_ratio
            self.grasp_rotation = (
                Rot.from_euler("Z", angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
        self.geometry_position = [0, 0, obj.size[2] / 2]


class Sunken_Door_Knowledge(KnowledgeWrapper):
    """open from the top or the right side"""

    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        if self.primact_type == "pulling":
            top_ratio, side_ratio, rot_side_ratio = (
                manip_args if manip_args else (0, 0, 0)
            )
            if rot_side_ratio > 0:  # up
                self.gripper_open_width = obj.size[2] * 1.5
                self.grasp_position = np.array(
                    [
                        obj.size[0] * (1 / 3 + side_ratio * 1 / 7),
                        obj.size[1] / 2 + self.gripper_length,
                        0,
                    ]
                )
                self.grasp_rotation = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            else:  # right
                self.grasp_position = np.array(
                    [
                        obj.size[0] / 2 + self.gripper_length,
                        top_ratio * obj.size[1] * 0.25,
                        obj.size[2] / 2,
                    ]
                )
                self.grasp_rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
            self.force_direction = np.array([0, 0, 1])
        else:  # self.primact_type == "pushing":
            up_ratio, horizontal_ratio, rot_ratio = (
                manip_args if manip_args else (0, 0, 0)
            )
            self.force_direction = np.array([0, 0, 1])
            self.gripper_open_width = 0
            self.grasp_position = np.array(
                [
                    obj.size[0] * (1 / 3 + horizontal_ratio * 1 / 7),
                    up_ratio * obj.size[1] * 0.25,
                    obj.size[2] / 2 + self.gripper_length + self.finger_length,
                ]
            )
            self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

            angle = np.pi / 2 * rot_ratio
            self.grasp_rotation = (
                Rot.from_euler("Z", angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )

        bottom_mesh_position = [0, 0, -obj.sunken_size[2] / 2 + obj.size[2] / 2]
        self.geometry_position = bottom_mesh_position


template2knowledge = {
    "Curve_Handle": Curve_Handle_Knowledge,
    "Trifold_Handle": Trifold_Handle_Knowledge,
    "Trifold_Curve_Handle": Trifold_Curve_Handle_Knowledge,
    "Cuboidal_Handle": Cuboidal_Handle_Knowledge,
    "Cuboidal_Door": Cuboidal_Door_Knowledge,
    "Sunken_Door": Sunken_Door_Knowledge,
}
