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


class Regular_Screen_Knowledge(KnowledgeWrapper):
    """open screen"""

    concept_rotation_order = "XYZ"
    manip_params_size = 2

    def __init__(self, concept, category, manipulation_knowledge, *manip_args):
        super().__init__(concept, category, manipulation_knowledge)
        obj = self.obj
        trans_ratio, rot_ratio = manip_args if manip_args else (0, 0)

        size = obj.size
        offset = obj.offset
        screen_rotation = obj.screen_rotation

        self.force_direction = np.array([0, 0, -1])
        self.grasp_position = np.array(
            [trans_ratio * size[0] / 2 * 0.7, size[1] / 2 + self.gripper_length, 0]
        )
        self.grasp_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.grasp_rotation = (
            Rot.from_euler("x", rot_ratio * np.pi / 10).as_matrix()
            @ self.grasp_rotation
        )

        # refer to concept_template.Regular_Screen
        back_mesh_position = [
            0,
            offset[0] + size[1] * np.cos(screen_rotation[0]) / 2,
            offset[1],
        ]
        back_mesh_rotation = [screen_rotation[0], 0, 0]
        self.geometry_position = back_mesh_position
        self.geometry_rotation = back_mesh_rotation


template2knowledge = {"Regular_Screen": Regular_Screen_Knowledge}
