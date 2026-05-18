import os
import sys
from code.utils import *

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .concept_template import *

current_file_path = os.path.abspath(__file__)
sys.path.append(current_file_path)
sys.path.append(os.path.dirname(os.path.dirname(current_file_path)))  # categories
from code.base_knowledge import KnowledgeWrapper


class Cylindrical_Lid_Knowledge(KnowledgeWrapper):
    """from side or top"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        x_rot_ratio, y_rot_ratio = manip_args if manip_args else (0, 0)

        outer_size = obj.outer_size
        inner_size = obj.inner_size

        if y_rot_ratio >= 0:  # side
            self.grasp_position = np.array(
                [0, -inner_size[2] / 2, -outer_size[0] - self.gripper_length]
            )
            self.grasp_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            self.force_direction = np.array([0, 1, 0])
            offset_rotation = Rot.from_euler("y", x_rot_ratio * np.pi).as_matrix()
            self.grasp_position = offset_rotation @ self.grasp_position
            self.grasp_rotation = offset_rotation @ self.grasp_rotation
        else:  # top
            self.grasp_position = np.array(
                [0, (outer_size[2] - inner_size[2]) / 2 + self.gripper_length, 0]
            )
            self.grasp_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            self.force_direction = np.array([0, 1, 0])
            offset_rotation = Rot.from_euler("y", x_rot_ratio * np.pi / 2).as_matrix()
            self.grasp_position = offset_rotation @ self.grasp_position
            self.grasp_rotation = offset_rotation @ self.grasp_rotation

        top_mesh_position = [0, inner_size[2] / 2, 0]
        self.geometry_position = np.array(top_mesh_position)


class Multilevel_Body_Knowledge(KnowledgeWrapper):
    concept_rotation_order = "XYZ"

    def __init__(self, concept, category, manipulation_knowledge):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj


template2knowledge = {
    "Cylindrical_Lid": Cylindrical_Lid_Knowledge,
    "Multilevel_Body": Multilevel_Body_Knowledge,
}
