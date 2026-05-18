import copy
import importlib
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(current_file_path))  # code
sys.path.append(f"{os.path.dirname(current_file_path)}/categories")  # categories


def get_rodrigues_matrix(axis, angle):
    axis = np.array(axis)
    identity = np.eye(3)
    s1 = np.array(
        [
            [np.zeros([]), -axis[2], axis[1]],
            [axis[2], np.zeros([]), -axis[0]],
            [-axis[1], axis[0], np.zeros([])],
        ]
    )
    s2 = np.matmul(axis[:, None], axis[None])
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rodrigues_matrix = cos_angle * identity + sin_angle * s1 + (1 - cos_angle) * s2
    return rodrigues_matrix


def apply_transformation(
    vertices, position, rotation, rotation_order="XYZ", offset_first=False
):
    # process position first
    if offset_first:
        vertices = vertices + np.array(position)

    # process rotation
    rot_mat = {}

    rot_mat["X"] = get_rodrigues_matrix([1, 0, 0], rotation[0])
    rot_mat["Y"] = get_rodrigues_matrix([0, 1, 0], rotation[1])
    rot_mat["Z"] = get_rodrigues_matrix([0, 0, 1], rotation[2])

    for s in rotation_order:
        vertices = np.matmul(vertices, rot_mat[s].T)

    # process position second
    if not offset_first:
        vertices = vertices + np.array(position)

    return vertices


class KnowledgeWrapper:
    manip_params_size = 0
    # standard gripper
    gripper_length = 0.09  # Distance from the origin to the base of the gripper fingers
    finger_length = 0.236  # Length of the gripper fingers
    gripper_side_width = 0.0574  # Width along the x-axis (side dimension)
    gripper_finger_width = 0.279  # Maximum width of object that can be gripped; distance between the two fingers

    grasp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    grasp_position = np.array([0, 0, 0])
    force_direction = np.array([0, 0, 1])
    gripper_open_width = gripper_finger_width
    geometry_position = [0, 0, 0]
    geometry_rotation = [0, 0, 0]

    geometry_rotation_order = "XYZ"
    concept_rotation_order = "XYZ"
    concept_offset_first = False

    # for partnet-mobility
    cat_infos = {
        "Bottle": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["lid", "body"],
            "joint_type": ["prismatic"],
        },
        "Box": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["cover"],
            "joint_type": ["revolute", "continuous"],
        },
        "Bucket": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle"],
            "joint_type": ["revolute", "continuous"],
        },
        "Dishwasher": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle", "door"],
            "joint_type": ["revolute"],
        },
        "Door": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle", "door"],
            "joint_type": ["revolute", "prismatic"],
        },
        "Eyeglasses": {
            "scale": 1,
            "fix_root": 0,
            "crucial_parts": ["leg"],
            "joint_type": ["revolute"],
        },
        "Faucet": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["switch"],
            "joint_type": ["revolute", "continuous"],
        },
        "Globe": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["bracket"],
            "joint_type": ["revolute"],
        },
        "Kettle": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle", "cover"],
            "joint_type": ["prismatic"],
        },
        "Knife": {
            "scale": 1,
            "fix_root": 0,
            "crucial_parts": ["handle"],
            "joint_type": ["prismatic"],
        },
        "KitchenPot": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["tophandle"],
            "joint_type": ["prismatic"],
        },
        "Laptop": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["screen"],
            "joint_type": ["revolute"],
        },
        "Mug": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle", "cylinder"],
            "joint_type": ["revolute"],
        },
        "Microwave": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle", "door"],
            "joint_type": ["revolute"],
        },
        "Oven": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle", "door"],
            "joint_type": ["revolute"],
        },
        "Pen": {
            "scale": 1,
            "fix_root": 0,
            "crucial_parts": ["cap", "barrel"],
            "joint_type": ["prismatic"],
        },
        "Pliers": {
            "scale": 1,
            "fix_root": 0,
            "crucial_parts": ["handle"],
            "joint_type": ["revolute"],
        },
        "Refrigerator": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle", "door"],
            "joint_type": ["revolute"],
        },
        "Ruler": {
            "scale": 1,
            "fix_root": 0,
            "crucial_parts": ["body"],
            "joint_type": ["revolute"],
        },
        "Safe": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle", "door"],
            "joint_type": ["revolute"],
        },
        "Shampoo": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["cap", "nozzle", "body"],
            "joint_type": ["prismatic"],
        },
        "Stapler": {
            "scale": 1,
            "fix_root": 0,
            "crucial_parts": ["cover"],
            "joint_type": ["revolute"],
        },
        "StorageFurniture": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["Regular_door", "Regular_drawer"],
            "joint_type": ["revolute", "prismatic"],
        },
        "Switch": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["switch"],
            "joint_type": ["prismatic", "revolute", "continuous"],
        },
        "Table": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["drawer", "door"],
            "joint_type": ["prismatic", "revolute", "continuous"],
        },
        "Trashcan": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["cover"],
            "joint_type": ["revolute", "prismatic"],
        },
        "USB": {
            "scale": 1,
            "fix_root": 0,
            "crucial_parts": ["cap", "body"],
            "joint_type": ["revolute", "prismatic"],
        },
        "Washingmachine": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["door"],
            "joint_type": ["revolute"],
        },
        "Window": {
            "scale": 1,
            "fix_root": 1,
            "crucial_parts": ["handle", "window"],
            "joint_type": ["prismatic"],
        },
    }

    def __init__(self, concept, category, manipulation_knowledge):
        """category must follow the folder name in knowledge eg. concept_factory convention"""
        concept_template_module = importlib.import_module(
            f"{category}.concept_template"
        )
        module = getattr(concept_template_module, concept["template"])
        self.obj = module(**concept["parameters"])
        self.state, self.primact_type = (
            manipulation_knowledge["state"],
            manipulation_knowledge["primact_type"],
        )

    def compute_grasp_pose(self):
        """
        Note that this function not implement partnet2sapien transformation
        concept_rotation: radian. The concept has automaticlly converted to radian in the concept_template.py initialization
        """
        grasp_position = self.grasp_position
        grasp_rotation = self.grasp_rotation
        force_direction = self.force_direction

        geometry_position = self.geometry_position
        geometry_rotation = self.geometry_rotation
        geometry_rotation_order = self.geometry_rotation_order

        concept_position = self.obj.position
        concept_rotation = self.obj.rotation
        concept_rotation_order = self.concept_rotation_order
        concept_offset_first = self.concept_offset_first

        grasp_position = apply_transformation(
            grasp_position,
            geometry_position,
            geometry_rotation,
            rotation_order=geometry_rotation_order,
        )
        grasp_position = apply_transformation(
            grasp_position,
            concept_position,
            concept_rotation,
            rotation_order=concept_rotation_order,
            offset_first=concept_offset_first,
        )

        force_direction = apply_transformation(
            force_direction,
            [0, 0, 0],
            geometry_rotation,
            rotation_order=geometry_rotation_order,
        )
        force_direction = apply_transformation(
            force_direction,
            [0, 0, 0],
            concept_rotation,
            rotation_order=concept_rotation_order,
        )

        grasp_rotation = apply_transformation(
            grasp_rotation.T, [0, 0, 0], geometry_rotation, geometry_rotation_order
        ).T
        grasp_rotation = apply_transformation(
            grasp_rotation.T, [0, 0, 0], concept_rotation, concept_rotation_order
        ).T

        grasp_pose = np.eye(4)
        grasp_pose[:3, :3] = grasp_rotation
        grasp_pose[:3, 3] = grasp_position

        return grasp_pose, force_direction

    def apply_transformation_for_geometry(
        self,
        position2apply,
        rotation2apply,
        apply_position_list,
        apply_rotation_list,
        geometry_rotation_order="XYZ",
    ):
        geometry_position, geometry_rotation = position2apply, rotation2apply
        for apply_position, apply_rotation in zip(
            apply_position_list, apply_rotation_list
        ):
            geometry_position = apply_transformation(
                geometry_position,
                apply_position,
                apply_rotation,
                geometry_rotation_order,
            )

        # scipy order is the opposite of the order in apply_transformation
        geometry_rotation = Rot.from_euler(
            geometry_rotation_order[::-1], geometry_rotation[::-1], degrees=False
        ).as_matrix()
        for apply_position, apply_rotation in zip(
            apply_position_list, apply_rotation_list
        ):
            geometry_rotation = apply_transformation(
                geometry_rotation.T, [0, 0, 0], apply_rotation, geometry_rotation_order
            ).T
        # for apply_transformation, the rotation order is "XYZ"
        geometry_rotation = Rot.from_matrix(geometry_rotation).as_euler(
            geometry_rotation_order[::-1], degrees=False
        )[::-1]

        self.geometry_position = geometry_position
        self.geometry_rotation = geometry_rotation

    @staticmethod
    def select_concept(
        category, conceptualization, crucial_parts, manipulation_knowledge
    ):
        """
        select the most crucial concept to act, based on the order of crucial_parts
        crucial_parts: List[str]
        return:
            selected_concept: dict
            selected_template: str
        """
        selected_concept = None
        selected_idx = 100
        selected_template = None
        knowledge_wrapper_module = importlib.import_module(
            f"{category}.knowledge_wrapper"
        )
        template2knowledge = getattr(knowledge_wrapper_module, "template2knowledge")
        for concept in conceptualization:
            template_name = concept["template"]
            if template_name not in template2knowledge.keys():
                continue
            for idx, crucial_part in enumerate(crucial_parts):
                if crucial_part.lower() not in template_name.lower():
                    continue
                if selected_idx > idx:
                    selected_idx = idx
                    selected_concept = concept
                    selected_template = template_name

        if selected_concept is None:
            raise NotImplementedError("knowledge for this instance is not implemented")
        return copy.deepcopy(selected_concept), selected_template


def get_manip_knowledge(
    conceptualization, category, manipulation_knowledge, manipulation_params=None
):
    knowledge_wrapper_module = importlib.import_module(f"{category}.knowledge_wrapper")
    template2knowledge = getattr(knowledge_wrapper_module, "template2knowledge")
    cat_info = KnowledgeWrapper.cat_infos[category]
    primact_type, _ = (
        manipulation_knowledge["primact_type"],
        manipulation_knowledge["state"],
    )

    # select the most crucial concept to act
    selected_concept, template_name = KnowledgeWrapper.select_concept(
        category, conceptualization, cat_info["crucial_parts"], manipulation_knowledge
    )

    scale = cat_info["scale"]
    for name, param in selected_concept["parameters"].items():
        if not any(
            keyword in name
            for keyword in ("rotation", "angle", "num", "type", "existence", "has")
        ):
            if type(param) is dict:
                for key, value in param.items():
                    param[key] = np.array(value) * scale
            else:
                selected_concept["parameters"][name] = np.array(param) * scale
        else:
            selected_concept["parameters"][name] = np.array(param)

    manipulation_params = [] if manipulation_params is None else manipulation_params
    assert type(manipulation_params) is list, (
        f"manipulation_params should be a list, but got {type(manipulation_params), manipulation_params}"
    )
    crucial_part_knowledge = template2knowledge[template_name](
        selected_concept, category, manipulation_knowledge, *manipulation_params
    )
    grasp_pose, force_direction = crucial_part_knowledge.compute_grasp_pose()

    gripper_width_ratio = (
        crucial_part_knowledge.gripper_open_width
        / KnowledgeWrapper.gripper_finger_width
    )
    gripper_width_ratio = min(gripper_width_ratio, 1)

    if primact_type == "pushing":
        force_direction = -force_direction

    return (
        grasp_pose,
        force_direction,
        gripper_width_ratio,
        crucial_part_knowledge.manip_params_size,
    )
