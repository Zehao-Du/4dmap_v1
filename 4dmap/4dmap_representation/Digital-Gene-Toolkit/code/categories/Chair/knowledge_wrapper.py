import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

current_file_path = os.path.abspath(__file__)
sys.path.append(current_file_path)

from code.utils import *

from base_knowledge import KnowledgeWrapper

from .concept_template import *
from .geometry_template import *


class Trifold_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "ZXY"

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        self.grasp_position = np.array(
            [0, 0, obj.grip_size[2] / 2 + self.gripper_length]
        )
        self.force_direction = np.array([0, 0, 1])
        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        # refer to concept_template.Trifold_Handle
        self.geometry_position = [0, 0, obj.mounting_size[2] + obj.grip_size[2] / 2]
        self.geometry_rotation = [0, 0, 0]


class Curve_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "ZXY"

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj

        self.grasp_position = np.array(
            [
                np.cos(obj.curve_exist_angle[0] / 2)
                * (obj.curve_size[0] + self.gripper_length),  #
                0,
                np.sin(obj.curve_exist_angle[0] / 2)
                * (obj.curve_size[0] + self.gripper_length),  #
            ]
        )
        self.force_direction = np.array(
            [
                np.cos(obj.curve_exist_angle[0] / 2),
                0,
                np.sin(obj.curve_exist_angle[0] / 2),
            ]
        )
        grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_rotation = (
            Rot.from_euler(
                "Y", -obj.curve_exist_angle[0] / 2, degrees=False
            ).as_matrix()
            @ grasp_rotation
        )

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
    concept_rotation_order = "ZXY"

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj

        self.grasp_position = np.array(
            [
                [
                    np.cos(obj.curve_exist_angle[0] / 2)
                    * (obj.curve_size[0] + self.gripper_length),
                    0,
                    np.sin(obj.curve_exist_angle[0] / 2)
                    * (obj.curve_size[0] + self.gripper_length),
                ]
            ]
        )
        # the force direction is the opposite of the grasp_position
        self.force_direction = np.array(
            [
                np.cos(obj.curve_exist_angle[0] / 2),
                0,
                np.sin(obj.curve_exist_angle[0] / 2),
            ]
        )

        grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_rotation = (
            Rot.from_euler(
                "Y", -obj.curve_exist_angle[0] / 2, degrees=False
            ).as_matrix()
            @ grasp_rotation
        )
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

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        self.grasp_position = np.array([0, 0, obj.size[2] / 2 + self.gripper_length])
        self.force_direction = np.array([0, 0, 1])
        self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        # refer to concept_template.Cuboidal_Handle
        self.geometry_position = [0, 0, obj.size[2] / 2]
        self.geometry_rotation = [0, 0, 0]


class Cuboidal_Door_Knowledge(KnowledgeWrapper):
    """open from the top"""

    concept_rotation_order = "XYZ"

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        self.force_direction = np.array([0, 0, 1])
        self.gripper_open_width = obj.size[2] * 1.5
        self.grasp_position = np.array(
            [obj.size[0] / 4, obj.size[1] / 2 + self.gripper_length, 0]
        )
        self.grasp_rotation = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        self.geometry_position = [0, 0, obj.size[2] / 2]


class Sunken_Door_Knowledge(KnowledgeWrapper):
    """open from the top"""

    concept_rotation_order = "XYZ"

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        self.force_direction = np.array([0, 0, 1])
        self.gripper_open_width = obj.size[2] * 1.5
        self.grasp_position = np.array(
            [obj.size[0] / 4, obj.size[1] / 2 + self.gripper_length, 0]
        )
        self.grasp_rotation = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
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
