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


class Simplified_Cover_Knowledge(KnowledgeWrapper):
    """lift the cover with a gripper from the top"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        size = obj.size
        length_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        size = obj.size

        self.grasp_position = np.array(
            [
                0,
                size[1] / 2 + 2 * self.gripper_length,
                length_ratio * size[2] * 0.3,
            ]
        )

        self.grasp_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.grasp_rotation = (
            Rot.from_euler("z", rot_angle, degrees=False).as_matrix()
            @ self.grasp_rotation
        )
        self.grasp_position = (
            Rot.from_euler("z", rot_angle, degrees=False).as_matrix()
            @ self.grasp_position
        )

        self.force_direction = np.array([0, 1, 0])

        mesh_position = [0, size[1] / 2, size[2] / 2]
        self.geometry_position = np.array(mesh_position)


class Carved_Cover_Knowledge(KnowledgeWrapper):
    """lift the cover with a gripper from the top"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        length_ratio, rot_ratio = manip_args if manip_args else (0, 0)
        rot_angle = np.pi / 6 * rot_ratio

        outer_size = obj.outer_size
        inner_size = obj.inner_size

        self.grasp_position = np.array(
            [
                0,
                (outer_size[1] - inner_size[1]) / 2 + 2 * self.gripper_length,
                length_ratio * outer_size[2] * 0.3,
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

        self.force_direction = np.array([0, 1, 0])

        top_mesh_position = [0, (outer_size[1] + inner_size[1]) / 2, outer_size[2] / 2]
        self.geometry_position = np.array(top_mesh_position)


template2knowledge = {
    "Simplified_Cover": Simplified_Cover_Knowledge,
    "Carved_Cover": Carved_Cover_Knowledge,
}
