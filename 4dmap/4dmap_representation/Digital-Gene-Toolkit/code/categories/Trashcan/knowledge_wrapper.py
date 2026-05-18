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


class Cylindrical_Cover_Knowledge(KnowledgeWrapper):
    """open cover"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_params):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        radius_ratio, rot_ratio = manip_params if manip_params else (0, 0)
        radius_angle = radius_ratio * np.pi / 6
        rot_angle = rot_ratio * np.pi / 6
        size = obj.size
        self.grasp_position = np.array([0, 0, size[0] + self.gripper_length * 2])
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_position = (
            Rot.from_euler("y", radius_angle).as_matrix() @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("y", radius_angle).as_matrix() @ self.grasp_rotation
        )
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )

        self.force_direction = np.array([0, 1, 0])


class Cuboidal_Cover_Knowledge(KnowledgeWrapper):
    """open cover"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_params):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_params if manip_params else (0, 0)
        rot_angle = rot_ratio * np.pi / 6
        height = obj.height
        top_size = obj.top_size
        bottom_size = obj.bottom_size

        self.grasp_position = np.array(
            [
                trans_ratio * top_size[0] * 0.25,
                0,
                top_size[1] / 2 + self.gripper_length * 2,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )

        self.force_direction = np.array([0, 1, 0])
        # Refer to concept_template.Cuboidal_Cover
        mesh_position = [0, height[0] / 2, bottom_size[1] / 2]
        self.geometry_position = np.array(mesh_position)


class Double_Layer_Cuboidal_Cover_Knowledge(KnowledgeWrapper):
    """open cover"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_params):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_params if manip_params else (0, 0)
        rot_angle = rot_ratio * np.pi / 6
        top_size = obj.top_size
        bottom_size = obj.bottom_size

        self.grasp_position = np.array(
            [
                trans_ratio * top_size[0] * 0.25,
                0,
                bottom_size[2] / 2 + self.gripper_length * 2,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )

        self.force_direction = np.array([0, 1, 0])
        # Refer to concept_template.Double_Layer_Cuboidal_Cover
        bottom_mesh_position = [0, bottom_size[1] / 2, bottom_size[2] / 2]
        self.geometry_position = np.array(bottom_mesh_position)


class Cylindrical_Hollow_Cover_Knowledge(KnowledgeWrapper):
    """open cover"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_params):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        radius_ratio, rot_ratio = manip_params if manip_params else (0, 0)
        radius_angle = radius_ratio * np.pi / 6
        rot_angle = rot_ratio * np.pi / 6
        size = obj.size
        self.grasp_position = np.array([0, 0, -size[1] + self.gripper_length * 2])
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_position = (
            Rot.from_euler("y", radius_angle).as_matrix() @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("y", radius_angle).as_matrix() @ self.grasp_rotation
        )
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )

        self.force_direction = np.array([0, 1, 0])

        mesh_position = [0, size[2] / 2, 0]
        self.geometry_position = np.array(mesh_position)


class Cuboidal_Hollow_Cover_Knowledge(KnowledgeWrapper):
    """open cover"""

    # TODO add other sides
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_params):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_params if manip_params else (0, 0)
        rot_angle = rot_ratio * np.pi / 6
        outer_size = obj.outer_size
        inner_size = obj.inner_size

        self.grasp_position = np.array(
            [
                trans_ratio * inner_size[0] * 0.25,
                0,
                -inner_size[1] / 2 + self.gripper_length * 2,
            ]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )

        self.force_direction = np.array([0, 1, 0])

        mesh_position = [0, outer_size[1] / 2, 0]
        self.geometry_position = np.array(mesh_position)


class Holed_Cylindrical_Cover_Knowledge(KnowledgeWrapper):
    """open cover"""

    # TODO refine this
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_params):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        radius_ratio, rot_ratio = manip_params if manip_params else (0, 0)
        radius_angle = radius_ratio * np.pi
        rot_angle = rot_ratio * np.pi / 6
        radius = obj.radius
        height = obj.height

        self.grasp_position = np.array([0, 0, radius[0] + self.gripper_length * 2])
        self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.grasp_position = (
            Rot.from_euler("y", radius_angle).as_matrix() @ self.grasp_position
        )
        self.grasp_rotation = (
            Rot.from_euler("y", radius_angle).as_matrix() @ self.grasp_rotation
        )
        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )

        self.force_direction = np.array([0, 1, 0])

        top_mesh_position = [0, height[2] + height[1] + (height[0] - height[1]) / 2, 0]
        self.geometry_position = np.array(top_mesh_position)


class Holed_Cuboidal_Cover_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_params):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_params if manip_params else (0, 0)
        rot_angle = rot_ratio * np.pi / 6
        has_hole = obj.has_hole
        inner_size = obj.inner_size
        outer_size = obj.outer_size

        self.force_direction = np.array([0, 1, 0])  # up
        top_mesh_position = [0, outer_size[1] / 2 + inner_size[1] / 2, 0]
        if np.sum(has_hole) < 1:  # "Holed_Cuboidal_Cover should have at least 1 holes"
            # front
            # the same as concept_template.Cuboidal_Cover
            self.grasp_position = np.array(
                [
                    trans_ratio * outer_size[0] * 0.25,
                    0,
                    outer_size[2] / 2 + self.gripper_length * 2,
                ]
            )
            self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            self.force_direction = np.array([0, 1, 0])
            self.geometry_position = np.array(top_mesh_position)

        # Refer to concept_template.Holed_Cuboidal_Cover
        # front part
        elif has_hole[0] == 1:
            self.grasp_position = np.array(
                [
                    trans_ratio * outer_size[0] * 0.25,
                    0,
                    outer_size[2] / 2 + self.gripper_length * 2,
                ]
            )
            self.grasp_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            self.geometry_position = top_mesh_position

        # behind part
        elif has_hole[1] == 1:
            self.grasp_position = np.array(
                [
                    trans_ratio * outer_size[0] * 0.25,
                    0,
                    -outer_size[2] / 2 - self.gripper_length * 2,
                ]
            )
            self.grasp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.geometry_position = top_mesh_position

        # left part
        elif has_hole[2] == 1:
            self.grasp_position = np.array(
                [
                    -outer_size[0] / 2 - self.gripper_length * 2,
                    0,
                    trans_ratio * outer_size[2] * 0.25,
                ]
            )
            self.grasp_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

            self.geometry_position = top_mesh_position
        # right part
        else:  # has_hole[3] == 1:
            self.grasp_position = np.array(
                [
                    outer_size[0] / 2 + self.gripper_length * 2,
                    0,
                    trans_ratio * outer_size[2] * 0.25,
                ]
            )
            self.grasp_rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

            self.geometry_position = top_mesh_position

        self.grasp_rotation = (
            self.grasp_rotation @ Rot.from_euler("x", rot_angle).as_matrix()
        )


template2knowledge = {
    "Cylindrical_Cover": Cylindrical_Cover_Knowledge,
    "Cuboidal_Cover": Cuboidal_Cover_Knowledge,
    "Double_Layer_Cuboidal_Cover": Double_Layer_Cuboidal_Cover_Knowledge,
    "Cylindrical_Hollow_Cover": Cylindrical_Hollow_Cover_Knowledge,
    "Cuboidal_Hollow_Cover": Cuboidal_Hollow_Cover_Knowledge,
    "Holed_Cylindrical_Cover": Holed_Cylindrical_Cover_Knowledge,
    "Holed_Cuboidal_Cover": Holed_Cuboidal_Cover_Knowledge,
}
