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


class LShape_Handle_Knowledge(KnowledgeWrapper):
    """
    By default, pull the front handle
    """

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 4

        existence_of_door = obj.existence_of_door
        existence_of_handle = obj.existence_of_handle
        door_size = obj.door_size
        door_rotation = obj.door_rotation
        fixed_part_size = obj.fixed_part_size
        vertical_movable_size = obj.vertical_movable_size
        horizontal_movable_size = obj.horizontal_movable_size
        interpiece_offset = obj.interpiece_offset
        offset_x = obj.offset_x

        # Refer to concept_template.LShape_Handle
        direction_settings = []
        double_door = 0
        if existence_of_door[0] and existence_of_door[1]:
            double_door = 1

        # parameter calculate
        for door in [0, 1]:
            if not existence_of_door[door]:
                continue
            for handle in [0, 1]:
                if not existence_of_handle[handle]:
                    continue
                # right_door(x+)
                if door:
                    handle_y_rotation = door_rotation[1]
                    handle_x_direction = 1
                    handle_y_axis = door_size[0] / 2
                # left_door(x-)
                else:
                    handle_y_rotation = -door_rotation[0]
                    handle_x_direction = -1
                    handle_y_axis = -door_size[0] / 2
                # front_handle(z+)
                if handle:
                    handle_z_direction = 1
                # back_handle(z-)
                else:
                    handle_z_direction = -1
                if double_door:
                    handle_x_position = handle_y_axis
                    handle_y_axis *= 2
                else:
                    handle_x_position = 0
                direction_settings.append(
                    {
                        "handle_x_direction": handle_x_direction,
                        "handle_x_position": handle_x_position,
                        "handle_z_direction": handle_z_direction,
                        "handle_y_axis": handle_y_axis,
                        "handle_y_rotation": handle_y_rotation,
                    }
                )
                if handle == 0:
                    self.grasp_position = np.array(
                        [
                            -(trans_ratio * 0.4 + 0.5) * horizontal_movable_size[0] / 2,
                            0,
                            -horizontal_movable_size[2] / 2 - self.gripper_length,
                        ]
                    )
                    self.grasp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    self.grasp_position = (
                        Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
                        @ self.grasp_position
                    )
                    self.grasp_rotation = (
                        Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
                        @ self.grasp_rotation
                    )
                    self.force_direction = np.array([0, 0, -1])
                else:
                    self.grasp_position = np.array(
                        [
                            (trans_ratio * 0.4 + 0.5) * horizontal_movable_size[0] / 2,
                            0,
                            horizontal_movable_size[2] / 2 + self.gripper_length,
                        ]
                    )
                    self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                    self.force_direction = np.array([0, 0, 1])
                break
            break

        direction_setting = direction_settings[0]
        top_mesh_position = [
            direction_setting["handle_x_position"]
            + interpiece_offset[0]
            + direction_setting["handle_x_direction"]
            * (
                -offset_x[0]
                + (horizontal_movable_size[0] - vertical_movable_size[0]) / 2
            ),
            interpiece_offset[1],
            direction_setting["handle_z_direction"]
            * (
                door_size[2] / 2
                + fixed_part_size[2]
                + vertical_movable_size[2]
                + horizontal_movable_size[2] / 2
            ),
        ]
        top_mesh_rotation = [0, 0, 0]

        first_apply_position = [-direction_setting["handle_y_axis"], 0, 0]
        first_apply_rotation = [0, 0, 0]
        second_apply_position = [direction_setting["handle_y_axis"], 0, 0]
        second_apply_rotation = [0, direction_setting["handle_y_rotation"], 0]
        self.apply_transformation_for_geometry(
            top_mesh_position,
            top_mesh_rotation,
            [first_apply_position, second_apply_position],
            [first_apply_rotation, second_apply_rotation],
        )


class PiShape_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 1

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        height_ratio = manip_args[0] if manip_args else 0

        existence_of_door = obj.existence_of_door
        existence_of_handle = obj.existence_of_handle
        door_size = obj.door_size
        door_rotation = obj.door_rotation
        interpiece_offset = obj.interpiece_offset
        separation = obj.separation
        offset_x = obj.offset_x
        sub_size = obj.sub_size
        main_size = obj.main_size

        # Refer to concept_template.PiShape_Handle
        direction_settings = []
        double_door = 0
        if existence_of_door[0] and existence_of_door[1]:
            double_door = 1

        # parameter calculate
        for door in [0, 1]:
            if not existence_of_door[door]:
                continue
            for handle in [0, 1]:
                if not existence_of_handle[handle]:
                    continue
                # right_door(x+)
                if door:
                    handle_y_rotation = door_rotation[1]
                    handle_x_direction = 1
                    handle_y_axis = door_size[0] / 2
                # left_door(x-)
                else:
                    handle_y_rotation = -door_rotation[0]
                    handle_x_direction = -1
                    handle_y_axis = -door_size[0] / 2
                # front_handle(z+)
                if handle:
                    handle_z_direction = -1
                # back_handle(z-)
                else:
                    handle_z_direction = 1
                if double_door:
                    handle_x_position = handle_y_axis
                    handle_y_axis *= 2
                else:
                    handle_x_position = 0
                direction_settings.append(
                    {
                        "handle_x_direction": handle_x_direction,
                        "handle_x_position": handle_x_position,
                        "handle_z_direction": handle_z_direction,
                        "handle_y_axis": handle_y_axis,
                        "handle_y_rotation": handle_y_rotation,
                    }
                )
                if handle == 0:
                    self.grasp_position = np.array(
                        [
                            0,
                            height_ratio * sub_size[1] / 2 * 0.7,
                            main_size[2] / 2 + self.gripper_length,
                        ]
                    )

                    self.grasp_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

                    self.force_direction = np.array([0, 0, 1])
                else:
                    self.grasp_position = np.array(
                        [
                            0,
                            height_ratio * sub_size[1] / 2 * 0.7,
                            -main_size[2] / 2 - self.gripper_length,
                        ]
                    )

                    self.grasp_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

                    self.force_direction = np.array([0, 0, -1])
                break
            break

        direction_setting = direction_settings[0]
        main_mesh_position = [
            direction_setting["handle_x_direction"] * offset_x[0]
            + direction_setting["handle_x_position"],
            interpiece_offset[0] + separation[0] / 2 + sub_size[1] / 2,
            direction_setting["handle_z_direction"]
            * (door_size[2] / 2 + sub_size[2] + main_size[2] / 2),
        ]
        main_mesh_rotation = np.array([0, 0, 0])

        first_apply_position = [-direction_setting["handle_y_axis"], 0, 0]
        first_apply_rotation = [0, 0, 0]
        second_apply_position = [direction_setting["handle_y_axis"], 0, 0]
        second_apply_rotation = [0, direction_setting["handle_y_rotation"], 0]
        self.apply_transformation_for_geometry(
            main_mesh_position,
            main_mesh_rotation,
            [first_apply_position, second_apply_position],
            [first_apply_rotation, second_apply_rotation],
        )


class Cylindrical_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_angle_1, rot_angle_2 = manip_args if manip_args else (0, 0)
        rot_angle_1 = rot_angle_1 * np.pi / 2
        rot_angle_2 = rot_angle_2 * np.pi / 3

        existence_of_door = obj.existence_of_door
        existence_of_handle = obj.existence_of_handle
        door_size = obj.door_size
        door_rotation = obj.door_rotation
        fixed_part_size = obj.fixed_part_size
        offset_x = obj.offset_x
        sub_size = obj.sub_size
        main_size = obj.main_size

        # Refer to concept_template.PiShape_Handle
        direction_settings = []
        double_door = 0
        if existence_of_door[0] and existence_of_door[1]:
            double_door = 1

        # parameter calculate
        for door in [0, 1]:
            if not existence_of_door[door]:
                continue
            for handle in [0, 1]:
                if not existence_of_handle[handle]:
                    continue
                # right_door(x+)
                if door:
                    handle_y_rotation = door_rotation[1]
                    handle_x_direction = 1
                    handle_y_axis = door_size[0] / 2
                # left_door(x-)
                else:
                    handle_y_rotation = -door_rotation[0]
                    handle_x_direction = -1
                    handle_y_axis = -door_size[0] / 2
                # front_handle(z+)
                if handle:
                    handle_z_direction = 1
                # back_handle(z-)
                else:
                    handle_z_direction = -1
                if double_door:
                    handle_x_position = handle_y_axis
                    handle_y_axis *= 2
                else:
                    handle_x_position = 0
                direction_settings.append(
                    {
                        "handle_x_direction": handle_x_direction,
                        "handle_x_position": handle_x_position,
                        "handle_z_direction": handle_z_direction,
                        "handle_y_axis": handle_y_axis,
                        "handle_y_rotation": handle_y_rotation,
                    }
                )

                if handle == 0:  # from the back
                    self.grasp_position = np.array(
                        [
                            0,
                            -main_size[1] / 2 - self.gripper_length,
                            0,
                        ]
                    )
                    self.grasp_rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                    self.grasp_position = (
                        Rot.from_euler(
                            "yz", [rot_angle_1, rot_angle_2], degrees=False
                        ).as_matrix()
                        @ self.grasp_position
                    )
                    self.grasp_rotation = (
                        Rot.from_euler(
                            "yz", [rot_angle_1, rot_angle_2], degrees=False
                        ).as_matrix()
                        @ self.grasp_rotation
                    )

                    self.force_direction = np.array([0, -1, 0])
                else:  # from the front
                    self.grasp_position = np.array(
                        [
                            0,
                            main_size[1] / 2 + self.gripper_length,
                            0,
                        ]
                    )
                    self.grasp_rotation = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
                    self.grasp_position = (
                        Rot.from_euler(
                            "yz", [rot_angle_1, rot_angle_2], degrees=False
                        ).as_matrix()
                        @ self.grasp_position
                    )
                    self.grasp_rotation = (
                        Rot.from_euler(
                            "yz", [rot_angle_1, rot_angle_2], degrees=False
                        ).as_matrix()
                        @ self.grasp_rotation
                    )

                    self.force_direction = np.array([0, 1, 0])
                break
            break

        direction_setting = direction_settings[0]
        main_mesh_position = [
            -direction_setting["handle_x_direction"] * offset_x[0]
            + direction_setting["handle_x_position"],
            0,
            direction_setting["handle_z_direction"]
            * (door_size[2] / 2 + fixed_part_size[1] + sub_size[1] + main_size[1] / 2),
        ]
        main_mesh_rotation = np.array([np.pi / 2, 0, 0])

        first_apply_position = [-direction_setting["handle_y_axis"], 0, 0]
        first_apply_rotation = [0, 0, 0]
        second_apply_position = [direction_setting["handle_y_axis"], 0, 0]
        second_apply_rotation = [0, direction_setting["handle_y_rotation"], 0]
        self.apply_transformation_for_geometry(
            main_mesh_position,
            main_mesh_rotation,
            [first_apply_position, second_apply_position],
            [first_apply_rotation, second_apply_rotation],
        )


class Spherical_Handle_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        rot_angle_1, rot_angle_2 = manip_args if manip_args else (0, 0)
        rot_angle_1 = rot_angle_1 * np.pi / 2
        rot_angle_2 = rot_angle_2 * np.pi / 3

        existence_of_door = obj.existence_of_door
        existence_of_handle = obj.existence_of_handle
        door_size = obj.door_size
        door_rotation = obj.door_rotation
        fixed_part_size = obj.fixed_part_size
        offset_x = obj.offset_x
        sub_size = obj.sub_size
        main_size = obj.main_size

        # Refer to concept_template.Spherical_Handle
        direction_settings = []
        double_door = 0
        if existence_of_door[0] and existence_of_door[1]:
            double_door = 1

        # parameter calculate
        for door in [0, 1]:
            if not existence_of_door[door]:
                continue
            for handle in [0, 1]:
                if not existence_of_handle[handle]:
                    continue
                # right_door(x+)
                if door:
                    handle_y_rotation = door_rotation[1]
                    handle_x_direction = 1
                    handle_y_axis = door_size[0] / 2
                # left_door(x-)
                else:
                    handle_y_rotation = -door_rotation[0]
                    handle_x_direction = -1
                    handle_y_axis = -door_size[0] / 2
                # front_handle(z+)
                if handle:
                    handle_z_direction = 1
                # back_handle(z-)
                else:
                    handle_z_direction = -1
                if double_door:
                    handle_x_position = handle_y_axis
                    handle_y_axis *= 2
                else:
                    handle_x_position = 0
                direction_settings.append(
                    {
                        "handle_x_direction": handle_x_direction,
                        "handle_x_position": handle_x_position,
                        "handle_z_direction": handle_z_direction,
                        "handle_y_axis": handle_y_axis,
                        "handle_y_rotation": handle_y_rotation,
                    }
                )
                if handle == 0:
                    self.grasp_position = np.array(
                        [
                            0,
                            0,
                            -main_size[2] / 2 - self.gripper_length,
                        ]
                    )
                    self.grasp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    self.grasp_position = (
                        Rot.from_euler(
                            "yz", [rot_angle_1, rot_angle_2], degrees=False
                        ).as_matrix()
                        @ self.grasp_position
                    )
                    self.grasp_rotation = (
                        Rot.from_euler(
                            "yz", [rot_angle_1, rot_angle_2], degrees=False
                        ).as_matrix()
                        @ self.grasp_rotation
                    )

                    self.force_direction = np.array([0, 0, -1])
                else:
                    self.grasp_position = np.array(
                        [
                            0,
                            0,
                            main_size[2] / 2 + self.gripper_length,
                        ]
                    )
                    self.grasp_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                    self.grasp_position = (
                        Rot.from_euler(
                            "yz", [rot_angle_1, rot_angle_2], degrees=False
                        ).as_matrix()
                        @ self.grasp_position
                    )
                    self.grasp_rotation = (
                        Rot.from_euler(
                            "yz", [rot_angle_1, rot_angle_2], degrees=False
                        ).as_matrix()
                        @ self.grasp_rotation
                    )

                    self.force_direction = np.array([0, 0, 1])
                break
            break
        direction_setting = direction_settings[0]
        main_mesh_position = [
            -direction_setting["handle_x_direction"] * offset_x[0]
            + direction_setting["handle_x_position"],
            0,
            direction_setting["handle_z_direction"]
            * (door_size[2] / 2 + fixed_part_size[1] + sub_size[1] + main_size[1] / 2),
        ]
        main_mesh_rotation = [0, 0, 0]

        first_apply_position = [-direction_setting["handle_y_axis"], 0, 0]
        first_apply_rotation = [0, 0, 0]
        second_apply_position = [direction_setting["handle_y_axis"], 0, 0]
        second_apply_rotation = [0, direction_setting["handle_y_rotation"], 0]
        self.apply_transformation_for_geometry(
            main_mesh_position,
            main_mesh_rotation,
            [first_apply_position, second_apply_position],
            [first_apply_rotation, second_apply_rotation],
        )


class Standard_Door_Knowledge(KnowledgeWrapper):
    """open left from the right side"""

    concept_rotation_order = "YXZ"
    concept_offset_first = True
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        height_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = rot_ratio * np.pi / 4

        size = obj.size
        door_rotation = obj.door_rotation

        if obj.existence_of_door[0] and obj.existence_of_door[1]:
            self.grasp_position = np.array(
                [size[0] / 2 + self.gripper_length, height_ratio * size[1] / 2 * 0.7, 0]
            )
            self.grasp_rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )

            left_mesh_position = [size[0] / 2, 0, 0]
            rotation = [0, -door_rotation[0], 0]
            position = [-size[0], 0, 0]
            self.apply_transformation_for_geometry(
                left_mesh_position, [0, 0, 0], [position], [rotation]
            )

        elif obj.existence_of_door[0]:
            self.grasp_position = np.array(
                [
                    obj.size[0] / 2 + self.gripper_length,
                    height_ratio * size[1] / 2 * 0.7,
                    0,
                ]
            )
            self.grasp_rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )

            left_mesh_position = [size[0] / 2, 0, 0]
            rotation = [0, -door_rotation[0], 0]
            position = [-size[0] / 2, 0, 0]
            self.apply_transformation_for_geometry(
                left_mesh_position, [0, 0, 0], [position], [rotation]
            )
        else:
            self.grasp_position = np.array(
                [
                    -obj.size[0] / 2 - self.gripper_length,
                    height_ratio * size[1] / 2 * 0.7,
                    0,
                ]
            )
            self.grasp_rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            self.grasp_rotation = (
                Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
                @ self.grasp_rotation
            )

            right_mesh_position = [-size[0] / 2, 0, 0]
            rotation = [0, door_rotation[1], 0]
            position = [size[0] / 2, 0, 0]
            self.apply_transformation_for_geometry(
                right_mesh_position, [0, 0, 0], [position], [rotation]
            )
        self.force_direction = np.array([0, 0, 1])
        self.gripper_open_width = obj.size[2] * 1.5


template2knowledge = {
    "Cylindrical_Handle": Cylindrical_Handle_Knowledge,
    "LShape_Handle": LShape_Handle_Knowledge,
    "PiShape_Handle": PiShape_Handle_Knowledge,
    "Spherical_Handle": Spherical_Handle_Knowledge,
    "Standard_Door": Standard_Door_Knowledge,
}
