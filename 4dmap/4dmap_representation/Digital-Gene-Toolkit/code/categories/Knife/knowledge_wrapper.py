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


class Cuboidal_Handle_Knowledge(KnowledgeWrapper):
    """from front side or back"""

    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        size = obj.size
        up_ratio, rot_ratio, side_ratio = manip_args if manip_args else (0, 0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        if side_ratio >= 0:  # back
            self.grasp_position = np.array(
                [0, up_ratio * size[1] * 0.3, -obj.size[2] / 2 - self.gripper_length]
            )

            self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
        else:  # side
            self.grasp_position = np.array(
                [obj.size[0] / 2 + self.gripper_length, up_ratio * size[1] * 0.25, 0]
            )
            self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            self.grasp_rotation = (
                Rot.from_euler("y", -np.pi / 2, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )


class T_Shaped_Handle_Knowledge(KnowledgeWrapper):
    """from front side or back"""

    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        main_size = obj.main_size
        up_ratio, rot_ratio, side_ratio = manip_args if manip_args else (0, 0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        if side_ratio >= 0:  # back
            self.grasp_position = np.array(
                [
                    0,
                    up_ratio * main_size[1] * 0.3,
                    -obj.main_size[2] / 2 - self.gripper_length,
                ]
            )

            self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
        else:  # side
            self.grasp_position = np.array(
                [
                    obj.main_size[0] / 2 + self.gripper_length,
                    up_ratio * main_size[1] * 0.25,
                    0,
                ]
            )
            self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            self.grasp_rotation = (
                Rot.from_euler("y", -np.pi / 2, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )


class Cylindrical_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        up_ratio, rot_ratio = manip_args if manip_args else (0, 0)

        size = obj.size
        middle_size = (size[0] + size[1]) / 2
        rot_angle = np.pi / 2 * rot_ratio
        self.grasp_position = np.array(
            [0, up_ratio * size[2] * 0.3, -middle_size / 2 - self.gripper_length]
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


class Curve_Handle_Knowledge(KnowledgeWrapper):
    """TODO: add side knowledge"""

    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        curve_angle_ratio, rot_ratio, side_ratio = (
            manip_args if manip_args else (0, 0, 0)
        )
        rot_angle = np.pi / 6 * rot_ratio
        curve_angle = obj.exist_angle[0] * (1 + curve_angle_ratio * 0.7) / 2
        radius = obj.radius

        self.grasp_position = np.array(
            [
                np.cos(curve_angle) * (obj.radius[0] + self.gripper_length),
                0,
                np.sin(curve_angle) * (obj.radius[0] + self.gripper_length),
            ]
        )
        self.grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("Y", -curve_angle, degrees=False).as_matrix()
        )
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("X", rot_angle, degrees=False).as_matrix()
        )

        mesh_position = [0, 0, (radius[0] + radius[1]) / 2]
        mesh_rotation = [np.pi / 2, np.pi / 2, 0]
        self.geometry_position = np.array(mesh_position)
        self.geometry_rotation = np.array(mesh_rotation)


class Multideck_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        up_ratio, rot_ratio, side_ratio = manip_args if manip_args else (0, 0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        beside_offset = obj.beside_offset
        beside_size = obj.beside_size
        self.grasp_position = np.array(
            [
                beside_offset[0],
                up_ratio * beside_size[1] * 0.3,
                beside_offset[1] - beside_size[2] - self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.grasp_rotation = (
            Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
            @ self.grasp_rotation
        )


class Enveloping_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        up_ratio, rot_ratio, side_ratio = manip_args if manip_args else (0, 0, 0)
        rot_angle = np.pi / 6 * rot_ratio
        size = obj.size

        if side_ratio >= 0:  # back
            self.grasp_position = np.array(
                [0, up_ratio * size[1] * 0.3, -obj.size[2] / 2 - self.gripper_length]
            )

            self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
        else:  # side
            self.grasp_position = np.array(
                [obj.size[0] / 2 + self.gripper_length, up_ratio * size[1] * 0.25, 0]
            )
            self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            self.grasp_rotation = (
                Rot.from_euler("y", -np.pi / 2, degrees=False).as_matrix()
                @ self.grasp_rotation
            )
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )


class Regular_Button_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 3

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        size = obj.size
        up_ratio = manip_args[0] if manip_args else 0

        self.grasp_position = np.array(
            [obj.size[0] / 2 + self.finger_length, up_ratio * size[1] * 0.25, 0]
        )
        self.grasp_rotation = Rot.from_euler("y", -np.pi / 2, degrees=False).as_matrix()
        self.grasp_rotation = (
            Rot.from_euler("x", np.pi / 2, degrees=False).as_matrix()
            @ self.grasp_rotation
        )


template2knowledge = {
    "Cuboidal_Handle": Cuboidal_Handle_Knowledge,
    "T_Shaped_Handle": T_Shaped_Handle_Knowledge,
    "Cylindrical_Handle": Cylindrical_Handle_Knowledge,
    "Curve_Handle": Curve_Handle_Knowledge,
    "Multideck_Handle": Multideck_Handle_Knowledge,
    "Enveloping_Handle": Enveloping_Handle_Knowledge,
    "Regular_Button": Regular_Button_Knowledge,
}
