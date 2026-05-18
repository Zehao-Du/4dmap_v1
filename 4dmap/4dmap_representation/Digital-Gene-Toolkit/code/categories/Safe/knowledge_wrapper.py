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
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        up_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio
        self.grasp_position = np.array(
            [
                0,
                up_ratio * obj.top_size[1] * 0.4,
                obj.top_size[2] / 2 + self.gripper_length,
            ]
        )
        # don't touch the door
        if obj.top_size[2] + obj.bottom_size[2] < self.finger_length:
            self.grasp_position[2] += (
                self.finger_length - obj.top_size[2] - obj.bottom_size[2]
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

        self.force_direction = np.array([0, 0, 1])
        if self.primact_type == "pushing":
            self.force_direction = -self.force_direction

        # refer to concept_template.Trifold_Handle
        self.geometry_position = [0, 0, obj.bottom_size[2] + obj.top_size[2] / 2]
        self.geometry_rotation = [0, 0, 0]


class Claw_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "ZXY"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio
        self.grasp_position = np.array(
            [
                (trans_ratio + 0.5) * obj.fork_size[0] * 0.25,
                0,
                obj.fork_size[2] / 2 + self.gripper_length,  #
            ]
        )

        # self.gripper_open_width = obj.fork_size[1] * 2
        self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )
        self.force_direction = np.array([0, 0, 1])  # outward

        # refer to concept_template.Claw_Handle
        i = 0
        rotation_tmp = np.pi * 2 / obj.num_forks[0] * i
        rotate_length = obj.bottom_size[0] + obj.fork_size[0] / 2 * np.cos(
            obj.fork_tilt_rotation[0]
        )

        self.geometry_position = [
            rotate_length * np.cos(rotation_tmp),
            rotate_length * np.sin(rotation_tmp),
            -obj.fork_size[2] / 2
            + obj.bottom_size[1]
            + obj.fork_size[0] / 2 * np.sin(obj.fork_tilt_rotation[0])
            - obj.fork_offset[0],
        ]
        self.geometry_rotation = [0, -obj.fork_tilt_rotation[0], rotation_tmp]


class Round_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "ZXY"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_1_ratio, rot_2_ratio = manip_args if manip_args else (0, 0)
        rot_1_angle = np.pi * rot_1_ratio
        rot_2_angle = np.pi / 4 * (rot_2_ratio + 1) * 0.75
        # grasp from side
        self.grasp_position = np.array(
            [0, self.gripper_finger_width / 3, obj.circle_size[0] + self.gripper_length]
        )
        self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.grasp_position = (
            Rot.from_euler("y", rot_1_angle, degrees=False).as_matrix()
            @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("y", rot_1_angle, degrees=False).as_matrix()
            @ self.grasp_rotation
        )
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_2_angle, degrees=False).as_matrix()
        )
        self.force_direction = np.array([1, 0, 0])


        # refer to concept_template.Round_Handle
        self.geometry_position = [
            0,
            0,
            obj.bottom_size[1]
            - obj.fork_size[2] / 2
            + obj.fork_size[0] * np.sin(obj.fork_tilt_rotation[0])
            - obj.fork_offset[0],
        ]
        self.geometry_rotation = [np.pi / 2, 0, 0]  #


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


class Behind_Double_Layer_Door_Knowledge(KnowledgeWrapper):
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
                self.gripper_open_width = obj.main_size[2] * 1.5
                self.grasp_position = np.array(
                    [
                        obj.main_size[0] * (1 / 3 + side_ratio * 1 / 7),
                        obj.main_size[1] / 2 + self.gripper_length,
                        0,
                    ]
                )
                self.grasp_rotation = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            else:  # right
                self.grasp_position = np.array(
                    [
                        obj.main_size[0] / 2 + self.gripper_length,
                        top_ratio * obj.main_size[1] * 0.25,
                        obj.main_size[2] / 2,
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
                    obj.main_size[0] * (1 / 3 + horizontal_ratio * 1 / 7),
                    up_ratio * obj.main_size[1] * 0.25,
                    obj.main_size[2] / 2 + self.gripper_length + self.finger_length,
                ]
            )
            self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

            angle = np.pi / 2 * rot_ratio
            self.grasp_rotation = (
                Rot.from_euler("Z", angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )

        mesh_position = [0, 0, obj.main_size[2] / 2]
        self.geometry_position = mesh_position


class Front_Double_Layer_Door_Knowledge(KnowledgeWrapper):
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
                self.gripper_open_width = obj.main_size[2] * 1.5
                self.grasp_position = np.array(
                    [
                        obj.main_size[0] * (1 / 3 + side_ratio * 1 / 7),
                        obj.main_size[1] / 2 + self.gripper_length,
                        0,
                    ]
                )
                self.grasp_rotation = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            else:  # right
                self.grasp_position = np.array(
                    [
                        obj.main_size[0] / 2 + self.gripper_length,
                        top_ratio * obj.main_size[1] * 0.25,
                        obj.main_size[2] / 2,
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
                    obj.main_size[0] * (1 / 3 + horizontal_ratio * 1 / 7),
                    up_ratio * obj.main_size[1] * 0.25,
                    obj.main_size[2] / 2 + self.gripper_length + self.finger_length,
                ]
            )
            self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

            angle = np.pi / 2 * rot_ratio
            self.grasp_rotation = (
                Rot.from_euler("Z", angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )

        mesh_position = [0, 0, obj.main_size[2] / 2]
        self.geometry_position = mesh_position


template2knowledge = {
    "Claw_Handle": Claw_Handle_Knowledge,
    "Trifold_Handle": Trifold_Handle_Knowledge,
    "Round_Handle": Round_Handle_Knowledge,
    "Cuboidal_Door": Cuboidal_Door_Knowledge,
    "Sunken_Door": Sunken_Door_Knowledge,
    "Behind_Double_Layer_Door": Behind_Double_Layer_Door_Knowledge,
    "Front_Double_Layer_Door": Front_Double_Layer_Door_Knowledge,
}
