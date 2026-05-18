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


class LShaped_Handle_Knowledge(KnowledgeWrapper):
    """
    By default, push the handle to the opposite of the x-axis
    push the front handle
    """

    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        assert obj.num_of_handle[0] > 0 or obj.num_of_handle[1] > 0, (
            "The number of handle should be greater than 0"
        )
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio
        if obj.num_of_handle[0] > 0:  # front
            # by default, manipulate the first handle
            i = 0
            self.grasp_position = np.array(
                [
                    0,
                    trans_ratio * obj.size_top[1] * 0.3,
                    obj.size_top[2] / 2 + self.gripper_length,
                ]
            )
            self.force_direction = np.array([-2 * float(obj.offset_x[i] > 0) + 1, 0, 0])
            self.grasp_rotation = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
            self.grasp_rotation = (
                self.grasp_rotation
                @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
            )

            # refer to concept_template.LShaped_Handle
            position_z = 0
            z_layer_position = obj.handle_z_position[i]
            if obj.window_type == 0:
                if z_layer_position == -1:
                    position_z -= obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 0:
                    position_z += obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 1:
                    position_z += obj.windows_size["size_1"][1]
            elif obj.window_type == 1:
                if z_layer_position >= 0:
                    position_z += obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 1:
                    position_z += obj.windows_size["size_1"][1]
                if z_layer_position >= 2:
                    position_z += obj.windows_size["size_2"][1]
                if z_layer_position == 3:
                    position_z += obj.windows_size["size_3"][1]
            self.geometry_position = [
                obj.offset_x[i],
                obj.offset_middle_y[0] + obj.offset_top_y[0],
                position_z
                + obj.size_bottom[2]
                + obj.size_middle[2]
                + obj.size_top[2] / 2,
            ]
            self.geometry_rotation = [0, 0, 0]
        elif obj.num_of_handle[1] > 0:  # back
            # by default, manipulate the first handle
            i = 0
            self.grasp_position = np.array(
                [0, 0, -obj.size_top[2] / 2 - self.gripper_length]
            )
            self.force_direction = np.array([-2 * float(obj.offset_x[i] > 0) + 1, 0, 0])
            self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            position_z = 0
            z_layer_position = obj.handle_z_position[i]
            if obj.window_type == 0:
                if z_layer_position == -1:
                    position_z -= (
                        obj.windows_size["size_0"][1] / 2
                        + obj.windows_size["size_2"][1]
                    )
                if z_layer_position >= 0:
                    position_z += obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 1:
                    position_z += obj.windows_size["size_0"][1]
            elif obj.window_type == 1:
                if z_layer_position >= 0:
                    position_z -= obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 1:
                    position_z += obj.windows_size["size_0"][1]
                if z_layer_position >= 2:
                    position_z += obj.windows_size["size_1"][1]
                if z_layer_position == 3:
                    position_z += obj.windows_size["size_2"][1]

            self.geometry_position = [
                obj.offset_x[i + 2],
                obj.offset_middle_y[0] + obj.offset_top_y[0],
                position_z
                - obj.size_bottom[2]
                - obj.size_middle[2]
                - obj.size_top[2] / 2,
            ]
            self.geometry_rotation = [0, 0, 0]
        else:
            raise NotImplementedError("The number of handle should be greater than 0")


class Arched_Handle_Knowledge(KnowledgeWrapper):
    """only for the front side of window"""

    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_1_ratio, rot_2_ratio = manip_args if manip_args else (0, 0)
        rot_2_angle = np.pi / 6 * rot_2_ratio
        central_angle = (
            np.arcsin((obj.seperation[0] / 2 + obj.bottom_size[1]) / obj.outer_size[0])
            * 2
        )
        arch_offset_z = (
            obj.bottom_size[2] - np.cos(central_angle / 2) * obj.outer_size[0]
        )

        if obj.num_of_handle[0] > 0:  # front
            # by default, manipulate the first handle
            i = 0
            position_z = 0
            z_layer_position = obj.handle_z_position[i]
            if obj.window_type == 0:
                if z_layer_position == -1:
                    position_z -= obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 0:
                    position_z += obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 1:
                    position_z += obj.windows_size["size_1"][1]
            elif obj.window_type == 1:
                if z_layer_position >= 0:
                    position_z += obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 1:
                    position_z += obj.windows_size["size_1"][1]
                if z_layer_position >= 2:
                    position_z += obj.windows_size["size_2"][1]
                if z_layer_position == 3:
                    position_z += obj.windows_size["size_3"][1]

            rot_angle = central_angle * (1 + rot_1_ratio) / 2
            self.grasp_position = np.array(
                [
                    np.cos(rot_angle) * (obj.outer_size[0] + self.gripper_length),  #
                    0,
                    np.sin(rot_angle) * (obj.outer_size[0] + self.gripper_length),  #
                ]
            )
            self.force_direction = np.array(
                [
                    0,  # -np.cos(central_angle/2),
                    2 * float(obj.offset_x[i] > 0) - 1,
                    0,  # -np.sin(central_angle/2)
                ]
            )
            grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            self.grasp_rotation = (
                Rot.from_euler("Y", -rot_angle, degrees=False).as_matrix()
                @ grasp_rotation
            )

            self.grasp_rotation = (
                self.grasp_rotation
                @ Rot.from_euler("x", rot_2_angle, degrees=False).as_matrix()
            )

            # refer to concept_template.Arched_Handle
            self.geometry_position = [
                obj.offset_x[i],
                0,
                position_z + arch_offset_z,
            ]
            self.geometry_rotation = [0, -np.pi / 2 + central_angle / 2, np.pi / 2]
        elif obj.num_of_handle[1] > 0:  # back
            i = 0
            self.grasp_position = np.array(
                [
                    np.cos(central_angle / 2)
                    * (obj.outer_size[0] + self.gripper_length),  #
                    0,
                    np.sin(central_angle / 2)
                    * (obj.outer_size[0] + self.gripper_length),  #
                ]
            )
            self.force_direction = np.array(
                [
                    0,  # -np.cos(central_angle/2),
                    2 * float(obj.offset_x[i] > 0) - 1,
                    0,  # -np.sin(central_angle/2)
                ]
            )
            grasp_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            self.grasp_rotation = (
                Rot.from_euler("Y", -central_angle / 2, degrees=False).as_matrix()
                @ grasp_rotation
            )

            position_z = 0
            z_layer_position = obj.handle_z_position[i]
            if obj.window_type == 0:
                if z_layer_position == -1:
                    position_z -= (
                        obj.windows_size["size_0"][1] / 2
                        + obj.windows_size["size_2"][1]
                    )
                if z_layer_position >= 0:
                    position_z -= obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 1:
                    position_z += obj.windows_size["size_0"][1]
            elif obj.window_type == 1:
                if z_layer_position >= 0:
                    position_z -= obj.windows_size["size_0"][1] / 2
                if z_layer_position >= 1:
                    position_z += obj.windows_size["size_0"][1]
                if z_layer_position >= 2:
                    position_z += obj.windows_size["size_1"][1]
                if z_layer_position == 3:
                    position_z += obj.windows_size["size_2"][1]

            self.geometry_position = [
                obj.offset_x[i + 2],
                0,
                position_z - arch_offset_z,
            ]
            self.geometry_rotation = [0, np.pi / 2 + central_angle / 2, np.pi / 2]
        else:
            raise NotImplementedError("The number of handle should be greater than 0")


class Cuboidal_Handle_Knowledge(KnowledgeWrapper):
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        windows_size = obj.windows_size
        handle_z_position = obj.handle_z_position
        window_type = obj.window_type
        offset_x = obj.offset_x
        size = obj.size

        i = 0
        position_z = 0
        z_layer_position = handle_z_position[i]
        if window_type == 0:
            if z_layer_position == -1:
                position_z -= windows_size["size_0"][1] / 2
            if z_layer_position >= 0:
                position_z += windows_size["size_0"][1] / 2
            if z_layer_position >= 1:
                position_z += windows_size["size_1"][1]
        elif window_type == 1:
            if z_layer_position >= 0:
                position_z += windows_size["size_0"][1] / 2
            if z_layer_position >= 1:
                position_z += windows_size["size_1"][1]
            if z_layer_position >= 2:
                position_z += windows_size["size_2"][1]
            if z_layer_position == 3:
                position_z += windows_size["size_3"][1]

        self.grasp_position = np.array(
            [0, trans_ratio * size[1] * 0.3, size[2] / 2 + self.gripper_length]
        )
        self.grasp_rotation = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        )

        handle_mesh_position = [offset_x[i], 0, position_z + size[2] / 2]
        self.geometry_position = handle_mesh_position


class Symmetrical_Window_Knowledge(KnowledgeWrapper):
    # from back, push the left middle window to left
    concept_rotation_order = "XYZ"

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        window_configurations = obj.window_configurations

        self.force_direction = np.array([-1, 0, 0])
        self.gripper_open_width = 0
        i = 0
        configuration = window_configurations[i]
        self.grasp_position = np.array(
            [
                configuration["window_size"]["glass_size"][0] / 4,
                0,
                -configuration["window_size"]["glass_size"][2] / 2
                - self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        glass_mesh_position = []
        for j in range(3):
            glass_mesh_position.append(
                configuration["position"][j]
                + configuration["window_size"]["glass_offset"][j]
            )
        self.geometry_position = glass_mesh_position


class Asymmetrical_Window_Knowledge(KnowledgeWrapper):
    # from back, push the first window to left
    concept_rotation_order = "XYZ"

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        window_configurations = obj.window_configurations

        self.force_direction = np.array([-1, 0, 0])
        self.gripper_open_width = 0
        i = 0
        configuration = window_configurations[i]
        self.grasp_position = np.array(
            [
                configuration["window_size"]["glass_size"][0] / 4,
                0,
                -configuration["window_size"]["glass_size"][2] / 2
                - self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        glass_mesh_position = []
        for j in range(3):
            glass_mesh_position.append(
                configuration["position"][j]
                + configuration["window_size"]["glass_offset"][j]
            )
        self.geometry_position = glass_mesh_position


class VerticalSlid_Window_Knowledge(KnowledgeWrapper):
    # from back, push the first window upwards
    concept_rotation_order = "XYZ"

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        window_configurations = obj.window_configurations

        self.force_direction = np.array([0, 1, 0])
        self.gripper_open_width = 0
        i = 0
        configuration = window_configurations[i]
        self.grasp_position = np.array(
            [
                0,
                -configuration["window_size"]["glass_size"][1] / 4,
                -configuration["window_size"]["glass_size"][2] / 2
                - self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        glass_mesh_position = []
        for j in range(3):
            glass_mesh_position.append(
                configuration["position"][j]
                + configuration["window_size"]["glass_offset"][j]
            )
        self.geometry_position = glass_mesh_position


template2knowledge = {
    "LShaped_Handle": LShaped_Handle_Knowledge,
    "Arched_Handle": Arched_Handle_Knowledge,
    "Cuboidal_Handle": Cuboidal_Handle_Knowledge,
    "Symmetrical_Window": Symmetrical_Window_Knowledge,
    "Asymmetrical_Window": Asymmetrical_Window_Knowledge,
    "VerticalSlid_Window": VerticalSlid_Window_Knowledge,
}
