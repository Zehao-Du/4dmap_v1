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


class SimplifiedZ_Switch_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    offset_first = True

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        offsets = obj.offsets
        size = obj.size

        self.grasp_position = np.array([0, size[1] / 2 + self.gripper_length, 0])
        self.grasp_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.force_direction = np.array([0, -1, 0])
        tmp_mesh_rotation = [np.pi / 2, 0, 0]

        # select the first switch
        self.geometry_position = offsets[0]
        self.geometry_rotation = tmp_mesh_rotation


class Knob_Switch_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    offset_first = True

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        offsets = obj.offsets

        i = 0
        j = 0
        offset = offsets[i]
        size = obj.sizes[j]

        self.grasp_position = np.array([0, size[1] / 2 + self.gripper_length, 0])
        self.grasp_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.force_direction = np.array([0, -1, 0])

        # select the first switch
        self.geometry_position = offset[j]


class HandleY_Switch_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 6
        sizes = obj.sizes
        offsets = obj.offsets
        number_of_cube = obj.number_of_cube
        i = 0  # left
        j = number_of_cube[0] - 1  # handle
        offset = offsets[i]
        size = sizes[j]
        self.grasp_position = np.array(
            [-size[0] / 4 * (1 + trans_ratio), size[1] / 2 + self.gripper_length, 0]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.grasp_position = (
            Rot.from_euler("X", rot_angle).as_matrix() @ self.grasp_position
        )
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("X", rot_angle).as_matrix()
        )
        self.force_direction = np.array([0, 0, 1])

        self.geometry_position = offset[j]


class HandleZ_Switch_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    offset_first = True

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        sizes = obj.sizes
        offsets = obj.offsets
        number_of_cube = obj.number_of_cube
        i = 0  # left
        j = number_of_cube[0] - 1  # handle
        offset = offsets[i]
        # print(f"offsets = {offsets}")
        size = sizes[j]
        self.grasp_position = np.array(
            [-size[0] / 4, 0, size[2] / 2 + self.gripper_length]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.force_direction = np.array([0, 1, 0])

        self.geometry_position = offset[j]


class RegularY_Switch_Knowledge(KnowledgeWrapper):
    """push down"""

    concept_rotation_order = "YXZ"
    offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 6
        size = obj.size
        sub_size = obj.sub_size
        sub_offset = obj.sub_offset
        sub_rotation = obj.sub_rotation
        rotation_X = obj.rotation_X
        rotation_Y = obj.rotation_Y

        self.grasp_position = np.array(
            [
                0,
                sub_size[1] / 4 * (1 + trans_ratio),
                -sub_size[0] / 2 - self.gripper_length,
            ]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        self.grasp_position = (
            Rot.from_euler("Z", -rot_angle).as_matrix() @ self.grasp_position
        )
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("X", rot_angle).as_matrix()
        )
        self.force_direction = np.array([1, 0, 0])

        distance_to_center = (size[0] ** 2 - sub_size[0] ** 2) ** 0.5
        sub_mesh_position = [0, 0, sub_size[1] / 2 + sub_offset[1]]
        sub_mesh_rotation = [np.pi / 2, 0, 0]

        first_apply_position = [
            0,
            sub_offset[0],
            distance_to_center,
        ]
        first_apply_rotation = [sub_rotation[0], 0, 0]
        second_apply_position = [0, size[1] / 2, 0]
        second_apply_rotation = [rotation_X[0], rotation_Y[0], 0]

        self.apply_transformation_for_geometry(
            sub_mesh_position,
            sub_mesh_rotation,
            [first_apply_position, second_apply_position],
            [first_apply_rotation, second_apply_rotation],
        )


class RegularX_Switch_Knowledge(KnowledgeWrapper):
    """push down"""

    concept_rotation_order = "YXZ"
    offset_first = True

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        size = obj.size
        sub_size = obj.sub_size
        sub_offset = obj.sub_offset
        sub_rotation = obj.sub_rotation
        rotation_X = obj.rotation_X
        rotation_Z = obj.rotation_Z
        distance_to_center = (size[0] ** 2 - sub_size[0] ** 2) ** 0.5

        if sub_offset[0] < 0:
            self.grasp_position = np.array(
                [-sub_size[0] / 2 - self.gripper_length, sub_size[1] / 4, 0]
            )
            self.grasp_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        else:
            self.grasp_position = np.array(
                [sub_size[0] / 2 + self.gripper_length, sub_size[1] / 4, 0]
            )
            self.grasp_rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        self.force_direction = np.array([0, 0, 1])

        sub_mesh_position = [0, sub_size[1] / 2 + sub_offset[1], 0]
        sub_mesh_rotation = [0, 0, 0]
        first_apply_position = [
            sub_offset[0],
            distance_to_center,
            0,
        ]
        first_apply_rotation = [0, 0, sub_rotation[0]]
        second_apply_position = [size[1] / 2, 0, 0]
        second_apply_rotation = [rotation_X[0], 0, rotation_Z[0]]

        self.apply_transformation_for_geometry(
            sub_mesh_position,
            sub_mesh_rotation,
            [first_apply_position, second_apply_position],
            [first_apply_rotation, second_apply_rotation],
        )


class RegularZ_Switch_Knowledge(KnowledgeWrapper):
    """push right"""

    concept_rotation_order = "YXZ"
    offset_first = True

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        size = obj.size
        sub_size = obj.sub_size
        sub_offset = obj.sub_offset
        sub_rotation = obj.sub_rotation
        rotation_X = obj.rotation_X
        rotation_Z = obj.rotation_Z

        if sub_offset[0] < 0:
            self.grasp_position = np.array(
                [0, sub_size[1] / 4, -sub_size[0] / 2 - self.gripper_length]
            )
            self.grasp_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        else:
            self.grasp_position = np.array(
                [0, sub_size[1] / 4, sub_size[0] / 2 + self.gripper_length]
            )
            self.grasp_rotation = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

        self.force_direction = np.array([1, 0, 0])
        distance_to_center = (size[0] ** 2 - sub_size[0] ** 2) ** 0.5

        sub_mesh_position = [0, sub_size[1] / 2 + sub_offset[1], 0]
        sub_mesh_rotation = [0, 0, 0]
        first_apply_position = [
            0,
            distance_to_center,
            sub_offset[0],
        ]
        first_apply_rotation = [sub_rotation[0], 0, 0]
        second_apply_position = [0, 0, size[1] / 2]
        second_apply_rotation = [rotation_X[0], 0, rotation_Z[0]]

        self.apply_transformation_for_geometry(
            sub_mesh_position,
            sub_mesh_rotation,
            [first_apply_position, second_apply_position],
            [first_apply_rotation, second_apply_rotation],
        )


class Lever_Switch_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 6
        R = obj.R
        size = obj.size
        obj.positions = obj.positions
        position0, position1, position2 = obj.positions
        start_ends = [
            [position0, position1],
            [position1, position2],
        ]

        start_end = start_ends[1]  # select the second segment
        vector = np.array(start_end[1]) - np.array(start_end[0])
        for i in range(3):
            if vector[i] <= 0.00001:
                vector[i] += 0.00001
        length = np.linalg.norm(vector)

        self.grasp_position = np.array(
            [0, length / 4 * (1 + trans_ratio), -R[0] / 2 - self.gripper_length]
        )
        self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("X", rot_angle).as_matrix()
        )
        self.force_direction = np.array([1, 0, 0])

        tmp_mesh_position = [0, length / 2, 0]
        tmp_mesh_rotation = [0, 0, 0]

        first_apply_position = [
            start_end[0][0],
            start_end[0][1] + size[1],
            start_end[0][2] + size[0],
        ]
        first_apply_rotation = [
            np.arccos(vector[1] / length),
            np.arctan(vector[0] / vector[2]),
            0,
        ]

        self.apply_transformation_for_geometry(
            tmp_mesh_position,
            tmp_mesh_rotation,
            [first_apply_position],
            [first_apply_rotation],
        )


class Cuboidal_Switch_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "YXZ"
    offset_first = True

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        size = obj.size
        sub_size = obj.sub_size
        sub_rotation = obj.sub_rotation
        sub_offset = obj.sub_offset
        switch_rotation = obj.switch_rotation

        self.grasp_position = np.array(
            [0, sub_size[1] / 2 + self.gripper_length, sub_size[2] / 4]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        self.force_direction = np.array([1, 0, 0])

        offset_z = sub_size[2] / 2 + sub_offset[1]
        sub_mesh_position = [
            0,
            offset_z * np.sin(sub_rotation[0])
            + sub_offset[0]
            + size[1] / 2
            - sub_size[1] / 2 * np.cos(sub_rotation[0]),
            offset_z * np.cos(sub_rotation[0]),
        ]
        sub_mesh_rotation = [-sub_rotation[0], 0, 0]
        self.geometry_position = sub_mesh_position
        self.geometry_rotation = sub_mesh_rotation

        first_apply_position = [0, size[1] / 2, 0]
        first_apply_rotation = [0, -switch_rotation[0], 0]

        self.apply_transformation_for_geometry(
            sub_mesh_position,
            sub_mesh_rotation,
            [first_apply_position],
            [first_apply_rotation],
        )


class RotaryX_Switch_Knowledge(KnowledgeWrapper):
    """left switch first, pull to z-forward"""

    concept_rotation_order = "YXZ"
    offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_1_ratio, rot_2_ratio = manip_args if manip_args else (0, 0)
        rot_1_angle = rot_1_ratio * np.pi / 6
        rot_2_angle = rot_2_ratio * np.pi / 6
        tilt_angle = obj.tilt_angle
        sub_size = obj.sub_size
        existence_of_switch = obj.existence_of_switch
        sub_offset = obj.sub_offset
        offset_x = obj.offset_x
        interval = obj.interval
        main_size_1 = obj.main_size_1
        main_size_2 = obj.main_size_2
        number_of_sub = obj.number_of_sub
        rotation0 = obj.rotation0
        rotation1 = obj.rotation1

        rotations = [rotation0[0], rotation1[0]]
        existences = [existence_of_switch[0] * -1, existence_of_switch[1]]

        self.force_direction = np.array([0, 0, 1])

        # only manipulate one switch
        if existences[0] == 0:  # the second(right) switch
            j = 1
            existence = existences[1]
            self.grasp_position = np.array(
                [sub_size[2] / 2 + self.gripper_length, sub_size[1] / 4, 0]
            )
            self.grasp_rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
            self.grasp_rotation = (
                self.grasp_rotation
                @ Rot.from_euler("XY", [rot_1_angle, rot_2_angle]).as_matrix()
            )
        else:  # the first(left) switch
            j = 0
            existence = existences[0]
            self.grasp_position = np.array(
                [-sub_size[2] / 2 - self.gripper_length, sub_size[1] / 4, 0]
            )
            self.grasp_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            self.grasp_rotation = (
                self.grasp_rotation
                @ Rot.from_euler("XY", [rot_1_angle, rot_2_angle]).as_matrix()
            )

        i = 0
        tmp_mesh_position = [0, np.cos(tilt_angle[0]) * sub_size[1] / 2, 0]
        tmp_mesh_rotation = [0, 0, -tilt_angle[0]]

        first_apply_position = [
            offset_x[j]
            + existence
            * (
                main_size_1[1]
                + main_size_2[1]
                - sub_size[0] / 2
                + sub_offset[0]
                + interval / 2
            ),
            0,
            0,
        ]
        first_apply_rotation = [rotations[j] + np.pi * 2 * i / number_of_sub[0], 0, 0]

        self.apply_transformation_for_geometry(
            tmp_mesh_position,
            tmp_mesh_rotation,
            [first_apply_position],
            [first_apply_rotation],
        )


class RotaryY_Switch_Knowledge(KnowledgeWrapper):
    """left switch first"""

    concept_rotation_order = "YXZ"
    offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_1_ratio, rot_2_ratio = manip_args if manip_args else (0, 0)
        rot_1_angle = rot_1_ratio * np.pi / 6
        rot_2_angle = rot_2_ratio * np.pi / 6

        tilt_angle = obj.tilt_angle
        sub_size = obj.sub_size
        sub_offset = obj.sub_offset
        offset_x = obj.offset_x
        main_size_1 = obj.main_size_1
        main_size_2 = obj.main_size_2
        number_of_sub = obj.number_of_sub
        rotation0 = obj.rotation0
        rotation1 = obj.rotation1

        rotations = [rotation0[0], rotation1[0]]

        self.force_direction = np.array([1, 0, 0])
        # for the left switch
        self.grasp_position = np.array(
            [0, sub_size[1] / 2 + self.gripper_length, sub_size[2] / 4]
        )
        self.grasp_rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("XY", [rot_1_angle, rot_2_angle]).as_matrix()
        )

        i = 0
        j = 0
        tmp_mesh_position = [0, 0, np.cos(tilt_angle[0]) * sub_size[2] / 2]
        tmp_mesh_rotation = [-tilt_angle[0], 0, 0]

        first_apply_position = [
            offset_x[j],
            main_size_1[1] + main_size_2[1] - sub_size[1] / 2 + sub_offset[0],
            0,
        ]
        first_apply_rotation = [0, rotations[j] + np.pi * 2 * i / number_of_sub[0], 0]

        self.apply_transformation_for_geometry(
            tmp_mesh_position,
            tmp_mesh_rotation,
            [first_apply_position],
            [first_apply_rotation],
        )


class RotaryZ_Switch_Knowledge(KnowledgeWrapper):
    """push to the right"""

    concept_rotation_order = "YXZ"
    offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_1_ratio, rot_2_ratio = manip_args if manip_args else (0, 0)
        rot_1_angle = rot_1_ratio * np.pi / 6
        rot_2_angle = rot_2_ratio * np.pi / 6
        tilt_angle = obj.tilt_angle
        sub_size = obj.sub_size
        sub_offset = obj.sub_offset
        offset_x = obj.offset_x
        main_size_1 = obj.main_size_1
        main_size_2 = obj.main_size_2
        number_of_sub = obj.number_of_sub
        rotation0 = obj.rotation0
        rotation1 = obj.rotation1
        rotations = [rotation0[0], rotation1[0]]

        self.force_direction = np.array([0, -1, 0])
        self.grasp_position = np.array(
            [sub_size[0] / 4, 0, sub_size[2] / 4 + self.gripper_length]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            self.grasp_rotation
            @ Rot.from_euler("XY", [rot_1_angle, rot_2_angle]).as_matrix()
        )

        tmp_mesh_position = [np.cos(tilt_angle[0]) * sub_size[0] / 2, 0, 0]
        tmp_mesh_rotation = [0, -tilt_angle[0], 0]

        i = 0
        j = 0
        first_apply_position = (
            [
                offset_x[j],
                0,
                main_size_1[1] + main_size_2[1] - sub_size[2] / 2 + sub_offset[0],
            ],
        )
        first_apply_rotation = [0, 0, rotations[j] + np.pi * 2 * i / number_of_sub[0]]

        self.apply_transformation_for_geometry(
            tmp_mesh_position,
            tmp_mesh_rotation,
            [first_apply_position],
            [first_apply_rotation],
        )


template2knowledge = {
    "SimplifiedZ_Switch": SimplifiedZ_Switch_Knowledge,
    "Knob_Switch": Knob_Switch_Knowledge,
    "HandleY_Switch": HandleY_Switch_Knowledge,
    "HandleZ_Switch": HandleZ_Switch_Knowledge,
    "RegularY_Switch": RegularY_Switch_Knowledge,
    "RegularX_Switch": RegularX_Switch_Knowledge,
    "RegularZ_Switch": RegularZ_Switch_Knowledge,
    "Lever_Switch": Lever_Switch_Knowledge,
    "Cuboidal_Switch": Cuboidal_Switch_Knowledge,
    "RotaryX_Switch": RotaryX_Switch_Knowledge,
    "RotaryY_Switch": RotaryY_Switch_Knowledge,
    "RotaryZ_Switch": RotaryZ_Switch_Knowledge,
}
