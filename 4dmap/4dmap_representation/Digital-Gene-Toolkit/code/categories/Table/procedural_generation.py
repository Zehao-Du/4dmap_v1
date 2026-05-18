import argparse
import json
import os
import pickle
import random
from code.geometry_template import *
from code.utils import *

import numpy as np
import open3d as o3d
import trimesh

from .concept_template import *


def get_overall_model(concepts):
    vertices_list = []
    faces_list = []
    total_num_vertices = 0

    for c in concepts:
        module = eval(c["template"])
        component = module(**c["parameters"])
        vertices_list.append(component.vertices)
        faces_list.append(component.faces + total_num_vertices)
        total_num_vertices += len(component.vertices)

    final_vertices = np.concatenate(vertices_list)
    final_faces = np.concatenate(faces_list)

    return final_vertices, final_faces


def get_overall_pointcloud(vertices, faces, sample_num=20000):
    opt_trimesh = trimesh.Trimesh(vertices, faces)
    pointcloud = np.array(opt_trimesh.sample(sample_num))
    return pointcloud


def get_table_type():
    table_type = [
        "Regular_table",
        "table_special_leg",
        "single_level_drawer_table",
        "multi_level_drawer_table",
        "Cylindrical_table",
        "special_layer_table",
        "square_table",
    ]
    weights = [1, 1, 1, 1, 1, 1, 1]
    table_type = random.choices(table_type, weights=weights, k=1)[0]
    return table_type


def concept_template_existence(table_type):
    if table_type == "Regular_table":
        concept_template_variation = {
            "desktop": {"template": ["Regular_desktop"], "neccessity": True},
            "leg": {
                "template": ["Regular_leg", "Regular_with_splat_leg"],
                "neccessity": True,
            },
            "sublayer": {"template": ["Regular_sublayer"], "neccessity": False},
            "partition": {"template": ["Regular_partition"], "neccessity": False},
        }
    elif table_type == "table_special_leg":
        concept_template_variation = {
            "desktop": {
                "template": ["Regular_desktop", "Cylindrical_desktop"],
                "neccessity": True,
            },
            "leg": {
                "template": ["Star_leg", "Bar_cylindrical_leg", "Bar_cuboid_leg"],
                "neccessity": True,
            },
        }
    elif table_type == "single_level_drawer_table":
        concept_template_variation = {
            "desktop": {"template": ["Regular_desktop"], "neccessity": True},
            "leg": {
                "template": ["Regular_leg", "Regular_with_splat_leg"],
                "neccessity": True,
            },
            "sublayer": {"template": ["Regular_sublayer"], "neccessity": True},
            "drawer": {"template": ["Regular_drawer"], "neccessity": True},
            "partition": {"template": ["Regular_partition"], "neccessity": False},
        }
    elif table_type == "multi_level_drawer_table":
        concept_template_variation = {
            "desktop": {"template": ["Regular_desktop"], "neccessity": True},
            "leg": {"template": ["Regular_leg"], "neccessity": True},
            "sublayer": {"template": ["Regular_sublayer"], "neccessity": True},
            "door": {"template": ["Regular_door"], "neccessity": True},
            "drawer": {"template": ["Regular_drawer"], "neccessity": True},
            "partition": {"template": ["Regular_partition"], "neccessity": False},
        }
    elif table_type == "Cylindrical_table":
        concept_template_variation = {
            "desktop": {"template": ["Cylindrical_desktop"], "neccessity": True},
            "leg": {"template": ["Regular_leg"], "neccessity": True},
            "sublayer": {"template": ["Cylindrical_sublayer"], "neccessity": True},
        }
    elif table_type == "special_layer_table":
        concept_template_variation = {
            "desktop": {"template": ["Regular_desktop"], "neccessity": True},
            "leg": {
                "template": ["Regular_leg", "Regular_with_splat_leg"],
                "neccessity": True,
            },
            "sublayer": {"template": ["Regular_sublayer"], "neccessity": True},
            "backboard": {"template": ["Regular_backboard"], "neccessity": True},
            "partition": {"template": ["Regular_partition"], "neccessity": False},
        }
    elif table_type == "square_table":
        concept_template_variation = {
            "desktop": {"template": ["Regular_desktop"], "neccessity": True},
            "leg": {"template": ["Bar_cuboid_leg"], "neccessity": True},
            "partition": {"template": ["Regular_partition"], "neccessity": False},
        }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["neccessity"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, table_type):
    new_concepts = []
    leg_type = ""
    is_rotation_suitable_for_sublayer = ""
    square_list = []
    part_x_length = [0, 0, 0]
    square_existence = []
    radius_of_sublayer = 0
    sublayer_z_offset = 0
    total_front_angle = 0
    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if table_type == "Regular_table":
            if template == "Regular_desktop":
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                parameter["size"][2] = np.maximum(
                    parameter["size"][0] / 3 * randRange(1, 0.8, 1.2)[0],
                    parameter["size"][2],
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_leg":
                leg_type = "Regular"
                body_parameter = concepts[0]["parameters"]
                numbers = np.array([1, 2, 3, 4])
                weights = np.array([0.04, 0.37, 0.21, 0.38])
                num_of_legs = np.random.choice(numbers, p=weights)
                parameter["symmetry_mode"][0] = 1

                parameter["number_of_legs"] = np.array([int(num_of_legs)])
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]
                parameter["additional_legs_params"][0] = 0
                if num_of_legs == 1:
                    parameter["front_rotation"][0] = 0
                    parameter["front_rotation"][1] = 0
                    parameter["central_rotation"][0] = 0
                    parameter["front_legs_size"] *= randRange(
                        parameter["front_legs_size"].shape[0], 0.8, 1.2
                    )
                    parameter["front_legs_size"][1] = np.maximum(
                        parameter["front_legs_size"][1],
                        body_parameter["size"][0] * randRange(1, 1.0, 1.4)[0],
                    )
                elif num_of_legs == 2:
                    is_rotation_suitable_for_sublayer = "SUITABLE"
                    parameter["front_rotation"][0] = 0
                    parameter["front_rotation"][1] = np.random.uniform(-15, 5)
                    parameter["central_rotation"][0] = 0
                    parameter["front_legs_size"][0] = (
                        body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["front_legs_size"][1] *= randRange(1, 0.8, 1.2)[0]
                    parameter["front_legs_size"][2] = (
                        body_parameter["size"][2] * 2 / 3 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["legs_separation"][0] = (
                        body_parameter["size"][0] - parameter["front_legs_size"][0]
                    ) * randRange(1, 0.8, 0.9)[0]
                elif num_of_legs == 3:
                    leg_height = (
                        parameter["front_legs_size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["front_rotation"][0] = np.random.randint(-15, 0)
                    parameter["front_rotation"][1] = np.random.randint(-15, 0)
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = np.random.uniform(0, 10)
                    parameter["rear_rotation"][1] = 0

                    parameter["front_legs_size"][0] = (
                        body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                    )
                    projection_1_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][0] / 180 * np.pi
                    )
                    projection_2_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][1] / 180 * np.pi
                    )
                    parameter["front_legs_size"][1] = np.sqrt(
                        projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                    )
                    parameter["front_legs_size"][2] = (
                        body_parameter["size"][2] / 8 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][0] = (
                        parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][2] = (
                        parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][1] = (
                        leg_height / np.cos(parameter["rear_rotation"][0] / 180 * np.pi)
                    ) / np.cos(parameter["rear_rotation"][1] / 180 * np.pi)

                    parameter["legs_separation"][0] = (
                        body_parameter["size"][0]
                        - parameter["front_legs_size"][1]
                        * np.sin(parameter["front_rotation"][0] / 180 * np.pi)
                        * np.sin(parameter["front_rotation"][1] / 180 * np.pi)
                    ) * randRange(1, 0.7, 1.0)[0]
                    parameter["legs_separation"][2] = (
                        body_parameter["size"][2]
                        - parameter["rear_legs_size"][2]
                        - parameter["front_legs_size"][2]
                    ) * randRange(1, 0.7, 1.0)[0]
                elif num_of_legs == 4:
                    leg_height = (
                        parameter["front_legs_size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                    if random.random() < 0.5:
                        leg_height = (
                            body_parameter["size"][0] * randRange(1, 1.0, 1.2)[0]
                        )
                    if random.random() < 0.2:
                        is_rotation_suitable_for_sublayer = "NOT_SUITABLE"
                        rot_angle_0 = (
                            randRange(1, 0, 15)[0] if random.random() < 0.5 else 0
                        )
                        rot_angle_1 = (
                            randRange(1, 0, 15)[0] if random.random() < 0.5 else 0
                        )
                        rot_angle_2 = (
                            randRange(1, 0, 15)[0] if random.random() < 0.5 else 0
                        )
                        parameter["front_rotation"][0] = -rot_angle_0
                        parameter["front_rotation"][1] = -rot_angle_1
                        parameter["central_rotation"][0] = 0
                        parameter["rear_rotation"][0] = rot_angle_2
                        parameter["rear_rotation"][1] = 0
                    else:
                        is_rotation_suitable_for_sublayer = "SUITABLE"
                        rot_angle_0 = (
                            randRange(1, 0, 15)[0] if random.random() < 0.5 else 0
                        )
                        rot_angle_1 = (
                            randRange(1, 0, 15)[0] if random.random() < 0.5 else 0
                        )
                        parameter["front_rotation"][0] = -rot_angle_0
                        parameter["front_rotation"][1] = -rot_angle_1
                        parameter["central_rotation"][0] = 0
                        parameter["rear_rotation"][0] = -parameter["front_rotation"][0]
                        parameter["rear_rotation"][1] = 0

                    parameter["front_legs_size"][0] = (
                        body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                    )
                    projection_1_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][0] / 180 * np.pi
                    )
                    projection_2_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][1] / 180 * np.pi
                    )
                    parameter["front_legs_size"][1] = np.sqrt(
                        projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                    )
                    parameter["front_legs_size"][2] = (
                        body_parameter["size"][2] / 8 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][0] = (
                        parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][2] = (
                        parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                    )
                    projection_1_of_leg = leg_height * np.tan(
                        parameter["rear_rotation"][0] / 180 * np.pi
                    )
                    projection_2_of_leg = leg_height * np.tan(
                        parameter["rear_rotation"][1] / 180 * np.pi
                    )
                    parameter["rear_legs_size"][1] = np.sqrt(
                        projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                    )
                    parameter["legs_separation"][0] = (
                        body_parameter["size"][0]
                        - parameter["front_legs_size"][0]
                        - parameter["front_legs_size"][1]
                        * np.sin(parameter["front_rotation"][0] / 180 * np.pi)
                        * np.sin(parameter["front_rotation"][1] / 180 * np.pi)
                    ) * randRange(1, 0.7, 1.0)[0]
                    parameter["legs_separation"][1] = (
                        body_parameter["size"][0] - parameter["rear_legs_size"][0]
                    ) * randRange(1, 0.7, 1.0)[0]
                    parameter["legs_separation"][2] = (
                        body_parameter["size"][2]
                        - parameter["front_legs_size"][2] / 2
                        - parameter["rear_legs_size"][2] / 2
                        - parameter["rear_legs_size"][1]
                        * np.sin(parameter["rear_rotation"][0] / 180 * np.pi)
                        / 2
                    ) * randRange(1, 0.7, 1.0)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(
                    concept["parameters"]["additional_legs_params"][0]
                )

                new_concepts.append(concept)

            elif template == "Regular_with_splat_leg":
                leg_type = "Splat"
                body_parameter = concepts[0]["parameters"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                probabilities = [
                    0.5777777777777777,
                    0.5777777777777777,
                    0.9111111111111111,
                    0.9111111111111111,
                ]
                parameter["bridging_bars_existance"] = np.array(
                    [np.random.choice([0, 1], p=[1 - p, p]) for p in probabilities]
                )
                leg_height = parameter["front_legs_size"][1] * randRange(1, 0.8, 1.2)[0]
                if random.random() < 0.5:
                    leg_height = body_parameter["size"][0] * randRange(1, 1.0, 1.2)[0]
                if random.random() < 0.2:
                    is_rotation_suitable_for_sublayer = "NOT_SUITABLE"
                    parameter["front_rotation"][0] = -randRange(1, 0, 15)[0]
                    parameter["front_rotation"][1] = -randRange(1, 0, 15)[0]
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = randRange(1, 0, 15)[0]
                    parameter["rear_rotation"][1] = -randRange(1, 0, 15)[0]
                else:
                    is_rotation_suitable_for_sublayer = "SUITABLE"
                    parameter["front_rotation"][0] = -randRange(1, 0, 15)[0]
                    parameter["front_rotation"][1] = -randRange(1, 0, 15)[0]
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = -parameter["front_rotation"][0]
                    parameter["rear_rotation"][1] = parameter["front_rotation"][1]

                parameter["front_legs_size"][0] = (
                    body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                )
                projection_1_of_leg = leg_height * np.tan(
                    parameter["front_rotation"][0] / 180 * np.pi
                )
                projection_2_of_leg = leg_height * np.tan(
                    parameter["front_rotation"][1] / 180 * np.pi
                )
                parameter["front_legs_size"][1] = np.sqrt(
                    projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                )
                parameter["front_legs_size"][2] = (
                    body_parameter["size"][2] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][0] = (
                    parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][2] = (
                    parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                projection_1_of_leg = leg_height * np.tan(
                    parameter["rear_rotation"][0] / 180 * np.pi
                )
                projection_2_of_leg = leg_height * np.tan(
                    parameter["rear_rotation"][1] / 180 * np.pi
                )
                parameter["rear_legs_size"][1] = np.sqrt(
                    projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                )
                parameter["legs_separation"][0] = (
                    body_parameter["size"][0]
                    - parameter["front_legs_size"][1]
                    * np.sin(parameter["front_rotation"][0] / 180 * np.pi)
                    * np.sin(parameter["front_rotation"][1] / 180 * np.pi)
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][1] = (
                    body_parameter["size"][0] - parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][2] = (
                    body_parameter["size"][2]
                    - parameter["rear_legs_size"][1]
                    * np.sin(parameter["rear_rotation"][0] / 180 * np.pi)
                    / 2
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["front_rear_bridging_bars_sizes"][0] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["front_rear_bridging_bars_sizes"][1] = np.minimum(
                    parameter["front_legs_size"][2], parameter["rear_legs_size"][2]
                )
                parameter["left_right_bridging_bars_sizes"][1] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["left_right_bridging_bars_sizes"][0] = np.minimum(
                    parameter["front_legs_size"][0], parameter["rear_legs_size"][0]
                )

                parameter["front_rear_bridging_bars_offset"][0] = (
                    leg_height / 2 * randRange(1, 0.1, 0.8)[0]
                )
                parameter["left_right_bridging_bars_offset"][0] = (
                    leg_height / 2 * randRange(1, 0.1, 0.8)[0]
                )

                parameter["front_rear_bridging_bars_offset"][1] = (
                    parameter["front_rear_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_bridging_bars_offset"][1] = (
                    parameter["left_right_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )

                parameter["additional_legs_params"][0] = int(0)

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

            elif template == "Regular_sublayer":
                top_parameter = concepts[0]["parameters"]
                body_parameter = concepts[1]["parameters"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]
                parameter["number_of_subs"] = np.array(
                    [int(0) for _ in parameter["number_of_subs"]]
                )
                parameter["additional_sublayers_params"][0] = int(0)
                if leg_type == "Regular":
                    if body_parameter["number_of_legs"][0] % 2 == 0:
                        if is_rotation_suitable_for_sublayer == "SUITABLE":
                            parameter["number_of_subs"] = np.array(
                                [int(1) for _ in parameter["number_of_subs"]]
                            )
                            rotation_front_x = (
                                body_parameter["front_rotation"][0] / 180 * np.pi
                            )
                            rotation_front_z = (
                                body_parameter["front_rotation"][1] / 180 * np.pi
                            )
                            if body_parameter["number_of_legs"][0] == 2:
                                offset_y_of_layer = (
                                    body_parameter["front_legs_size"][1]
                                    / 2
                                    * np.cos(rotation_front_z)
                                    * randRange(1, -0.8, 0.8)[0]
                                )
                                parameter["subs_offset"][0] = (
                                    -top_parameter["size"][1] / 2
                                    - body_parameter["front_legs_size"][1]
                                    / 2
                                    * np.cos(rotation_front_z)
                                    - offset_y_of_layer
                                )
                                parameter["subs_size"][1] = (
                                    top_parameter["size"][1] * randRange(1, 0.8, 1.0)[0]
                                )
                                parameter["subs_size"][0] = body_parameter[
                                    "legs_separation"
                                ][0] - 2 * offset_y_of_layer / np.cos(
                                    rotation_front_z
                                ) * np.sin(rotation_front_z)
                                parameter["subs_size"][2] = (
                                    body_parameter["front_legs_size"][2]
                                    * randRange(1, 0.8, 1.0)[0]
                                )
                            elif body_parameter["number_of_legs"][0] == 4:
                                offset_y_of_layer = (
                                    body_parameter["front_legs_size"][1]
                                    / 2
                                    * np.cos(rotation_front_z)
                                    * np.cos(rotation_front_x)
                                    * randRange(1, -0.8, 0.8)[0]
                                )
                                parameter["subs_offset"][0] = (
                                    -top_parameter["size"][1] / 2
                                    - body_parameter["front_legs_size"][1]
                                    / 2
                                    * np.cos(rotation_front_z)
                                    * np.cos(rotation_front_x)
                                    - offset_y_of_layer
                                )
                                parameter["subs_size"][1] = (
                                    top_parameter["size"][1] * randRange(1, 0.8, 1.0)[0]
                                )
                                parameter["subs_size"][0] = np.maximum(
                                    body_parameter["legs_separation"][0],
                                    body_parameter["legs_separation"][1],
                                ) - 2 * offset_y_of_layer / np.cos(
                                    rotation_front_z
                                ) * np.sin(rotation_front_z)
                                parameter["subs_size"][2] = body_parameter[
                                    "legs_separation"
                                ][2] - 2 * offset_y_of_layer * np.tan(rotation_front_x)

                elif leg_type == "Splat":
                    if is_rotation_suitable_for_sublayer == "SUITABLE":
                        parameter["number_of_subs"] = np.array(
                            [int(1) for _ in parameter["number_of_subs"]]
                        )
                        rotation_front_x = (
                            body_parameter["front_rotation"][0] / 180 * np.pi
                        )
                        rotation_front_z = (
                            body_parameter["front_rotation"][1] / 180 * np.pi
                        )
                        offset_y_of_layer = (
                            body_parameter["front_legs_size"][1]
                            / 2
                            * np.cos(rotation_front_z)
                            * np.cos(rotation_front_x)
                            * randRange(1, -0.8, 0.8)[0]
                        )
                        parameter["subs_offset"][0] = (
                            -top_parameter["size"][1] / 2
                            - body_parameter["front_legs_size"][1]
                            / 2
                            * np.cos(rotation_front_z)
                            * np.cos(rotation_front_x)
                            - offset_y_of_layer
                        )
                        parameter["subs_size"][1] = (
                            top_parameter["size"][1] * randRange(1, 0.8, 1.0)[0]
                        )
                        parameter["subs_size"][0] = np.maximum(
                            body_parameter["legs_separation"][0],
                            body_parameter["legs_separation"][1],
                        ) - 2 * offset_y_of_layer / np.cos(rotation_front_z) * np.sin(
                            rotation_front_z
                        )
                        parameter["subs_size"][2] = body_parameter["legs_separation"][
                            2
                        ] - 2 * offset_y_of_layer / np.cos(rotation_front_x) * np.sin(
                            rotation_front_x
                        )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_sublayers_params"][0] = int(0)
                if is_rotation_suitable_for_sublayer == "SUITABLE":
                    new_concepts.append(concept)
                else:
                    pass
            elif template == "Regular_partition":
                top_parameter = concepts[0]["parameters"]
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] + top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]
                parameter["has_partition"] = np.array([0, 1, 0])
                parameter["rear_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["rear_size"][1] = (
                    top_parameter["size"][1] * 4 / 7 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_size"][0] = 0
                parameter["left_right_size"][2] = (
                    top_parameter["size"][2] - 2 * parameter["rear_size"][1]
                )
                parameter["left_right_separation"][0] = (
                    top_parameter["size"][0] * randRange(1, 0.8, 1.0)[0]
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif table_type == "single_level_drawer_table":
            if template == "Regular_desktop":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_leg":
                leg_type = "Regular"
                body_parameter = concepts[0]["parameters"]
                num_of_legs = 4
                parameter["number_of_legs"] = np.array([int(num_of_legs)])
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]
                parameter["additional_legs_params"][0] = 0
                parameter["symmetry_mode"][0] = 1
                if num_of_legs == 4:
                    leg_height = body_parameter["size"][0] * randRange(1, 1.0, 1.2)[0]
                    parameter["front_rotation"][0] = 0
                    parameter["front_rotation"][1] = 0
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = 0
                    parameter["rear_rotation"][1] = 0

                    parameter["front_legs_size"][0] = (
                        body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["front_legs_size"][1] = leg_height
                    parameter["front_legs_size"][2] = (
                        body_parameter["size"][2] / 8 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][0] = (
                        parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][2] = (
                        parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][1] = leg_height

                    parameter["legs_separation"][0] = (
                        (body_parameter["size"][0]) * randRange(1, 0.7, 0.9)[0]
                    )
                    parameter["legs_separation"][1] = parameter["legs_separation"][0]
                    parameter["legs_separation"][2] = (
                        (body_parameter["size"][2]) * randRange(1, 0.7, 0.9)[0]
                    )
                is_rotation_suitable_for_sublayer = "SUITABLE"
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

            elif template == "Regular_with_splat_leg":
                leg_type = "Splat"
                body_parameter = concepts[0]["parameters"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                probabilities = [
                    0.5777777777777777,
                    0.5777777777777777,
                    0.9111111111111111,
                    0.9111111111111111,
                ]
                parameter["bridging_bars_existance"] = np.array(
                    [np.random.choice([0, 1], p=[1 - p, p]) for p in probabilities]
                )
                leg_height = body_parameter["size"][0] * randRange(1, 1.0, 1.2)[0]
                parameter["front_rotation"][0] = 0
                parameter["front_rotation"][1] = 0
                parameter["central_rotation"][0] = 0
                parameter["rear_rotation"][0] = 0
                parameter["rear_rotation"][1] = 0

                parameter["front_legs_size"][0] = (
                    body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["front_legs_size"][1] = leg_height
                parameter["front_legs_size"][2] = (
                    body_parameter["size"][2] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][0] = (
                    parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][2] = (
                    parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][1] = leg_height

                parameter["legs_separation"][0] = (
                    (body_parameter["size"][0]) * randRange(1, 0.7, 0.9)[0]
                )
                parameter["legs_separation"][1] = parameter["legs_separation"][0]
                parameter["legs_separation"][2] = (
                    (body_parameter["size"][2]) * randRange(1, 0.7, 0.9)[0]
                )
                parameter["front_rear_bridging_bars_sizes"][0] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["front_rear_bridging_bars_sizes"][1] = np.minimum(
                    parameter["front_legs_size"][2], parameter["rear_legs_size"][2]
                )
                parameter["left_right_bridging_bars_sizes"][1] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["left_right_bridging_bars_sizes"][0] = np.minimum(
                    parameter["front_legs_size"][0], parameter["rear_legs_size"][0]
                )

                parameter["front_rear_bridging_bars_offset"][0] = (
                    -leg_height / 2 * randRange(1, 0.7, 0.8)[0]
                )
                parameter["left_right_bridging_bars_offset"][0] = (
                    -leg_height / 2 * randRange(1, 0.7, 0.8)[0]
                )

                parameter["front_rear_bridging_bars_offset"][1] = (
                    parameter["front_rear_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_bridging_bars_offset"][1] = (
                    parameter["left_right_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )

                parameter["additional_legs_params"][0] = int(0)
                is_rotation_suitable_for_sublayer = "SUITABLE"
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

            elif template == "Regular_sublayer":
                top_parameter = concepts[0]["parameters"]
                leg_parameter = concepts[1]["parameters"]
                parameter["position"][0] = leg_parameter["position"][0]
                parameter["position"][1] = leg_parameter["position"][1]
                parameter["position"][2] = leg_parameter["position"][2]
                parameter["rotation"][0] = leg_parameter["rotation"][0]
                parameter["rotation"][1] = leg_parameter["rotation"][1]
                parameter["rotation"][2] = leg_parameter["rotation"][2]

                parameter["number_of_subs"] = np.array(
                    [int(0) for _ in parameter["number_of_subs"]]
                )
                parameter["additional_sublayers_params"][0] = int(3)
                if leg_type == "Regular":
                    if leg_parameter["number_of_legs"][0] % 2 == 0:
                        if is_rotation_suitable_for_sublayer == "SUITABLE":
                            num_of_layers = np.random.randint(1, 3)
                            parameter["number_of_subs"] = np.array(
                                [
                                    int(num_of_layers)
                                    for _ in parameter["number_of_subs"]
                                ]
                            )
                            offset_y_of_layer = (
                                leg_parameter["front_legs_size"][1]
                                / 2
                                * randRange(1, -0.6, -0.5)[0]
                            )
                            parameter["subs_offset"][0] = (
                                -leg_parameter["front_legs_size"][1] / 2
                                - offset_y_of_layer
                            )
                            parameter["subs_size"][1] = (
                                top_parameter["size"][1] * randRange(1, 0.8, 1.0)[0]
                            )
                            parameter["subs_size"][0] = np.maximum(
                                leg_parameter["legs_separation"][0],
                                leg_parameter["legs_separation"][1],
                            )
                            parameter["subs_size"][2] = leg_parameter[
                                "legs_separation"
                            ][2]
                            if num_of_layers == 2:
                                offset_y_of_layer = (
                                    leg_parameter["front_legs_size"][1]
                                    / 2
                                    * randRange(1, 0.6, 0.8)[0]
                                )
                                parameter["interval_between_subs"][
                                    0
                                ] = -offset_y_of_layer
                                parameter["subs_size"][1] = (
                                    top_parameter["size"][1] * randRange(1, 0.8, 1.0)[0]
                                )
                                parameter["subs_size"][0] = np.maximum(
                                    leg_parameter["legs_separation"][0],
                                    leg_parameter["legs_separation"][1],
                                )
                                parameter["subs_size"][2] = leg_parameter[
                                    "legs_separation"
                                ][2]
                            parameter["subs_offset"][0] -= parameter["subs_size"][1] / 2
                            drawer_height = -parameter["subs_offset"][0]
                            back_size_x = parameter["subs_size"][0]
                            back_size_z = (
                                leg_parameter["rear_legs_size"][2]
                                * randRange(1, 0.8, 1.0)[0]
                            )
                            side_size_x = (
                                np.minimum(
                                    leg_parameter["front_legs_size"][0],
                                    leg_parameter["rear_legs_size"][0],
                                )
                                * randRange(1, 0.8, 1.0)[0]
                            )
                            side_size_z = parameter["subs_size"][2]
                            cur = 0
                            for i in range(3):
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 3
                                ] = parameter["position"][0]
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 4
                                ] = parameter["position"][1]
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 5
                                ] = parameter["position"][2]
                                if i == 0:
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 0
                                    ] = back_size_x
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 1
                                    ] = drawer_height
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 2
                                    ] = back_size_z
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 3
                                    ] += 0
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 4
                                    ] += -drawer_height / 2
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 5
                                    ] += -parameter["subs_size"][2] / 2
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 6
                                    ] = 0
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 7
                                    ] = 0
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 8
                                    ] = 0
                                    cur += 1
                                elif i == 1:
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 0
                                    ] = side_size_x
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 1
                                    ] = drawer_height
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 2
                                    ] = side_size_z
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 3
                                    ] += -parameter["subs_size"][0] / 2
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 4
                                    ] += -drawer_height / 2
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 5
                                    ] += 0
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 6
                                    ] = 0
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 7
                                    ] = 0
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 8
                                    ] = 0
                                    cur += 1
                                elif i == 2:
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 0
                                    ] = side_size_x
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 1
                                    ] = drawer_height
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 2
                                    ] = side_size_z
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 3
                                    ] += parameter["subs_size"][0] / 2
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 4
                                    ] += -drawer_height / 2
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 5
                                    ] += 0
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 6
                                    ] = 0
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 7
                                    ] = 0
                                    parameter["additional_sublayers_params"][
                                        1 + cur * 9 + 8
                                    ] = 0
                                    cur += 1

                elif leg_type == "Splat":
                    if is_rotation_suitable_for_sublayer == "SUITABLE":
                        num_of_layers = np.random.randint(1, 3)
                        parameter["number_of_subs"] = np.array(
                            [int(num_of_layers) for _ in parameter["number_of_subs"]]
                        )
                        offset_y_of_layer = (
                            leg_parameter["front_legs_size"][1]
                            / 2
                            * randRange(1, -0.6, -0.5)[0]
                        )
                        parameter["subs_offset"][0] = (
                            -leg_parameter["front_legs_size"][1] / 2 - offset_y_of_layer
                        )
                        parameter["subs_size"][1] = (
                            top_parameter["size"][1] * randRange(1, 0.8, 1.0)[0]
                        )
                        parameter["subs_size"][0] = np.maximum(
                            leg_parameter["legs_separation"][0],
                            leg_parameter["legs_separation"][1],
                        )
                        parameter["subs_size"][2] = leg_parameter["legs_separation"][2]
                        if num_of_layers == 2:
                            offset_y_of_layer = (
                                leg_parameter["front_legs_size"][1]
                                / 2
                                * randRange(1, 0.6, 0.8)[0]
                            )
                            parameter["interval_between_subs"][0] = -offset_y_of_layer
                            parameter["subs_size"][1] = (
                                top_parameter["size"][1] * randRange(1, 0.8, 1.0)[0]
                            )
                            parameter["subs_size"][0] = np.maximum(
                                leg_parameter["legs_separation"][0],
                                leg_parameter["legs_separation"][1],
                            )
                            parameter["subs_size"][2] = leg_parameter[
                                "legs_separation"
                            ][2]
                        parameter["subs_offset"][0] -= parameter["subs_size"][1] / 2
                        drawer_height = -parameter["subs_offset"][0]
                        back_size_x = parameter["subs_size"][0]
                        back_size_z = (
                            leg_parameter["rear_legs_size"][2]
                            * randRange(1, 0.8, 1.0)[0]
                        )
                        side_size_x = (
                            np.minimum(
                                leg_parameter["front_legs_size"][0],
                                leg_parameter["rear_legs_size"][0],
                            )
                            * randRange(1, 0.8, 1.0)[0]
                        )
                        side_size_z = parameter["subs_size"][2]
                        cur = 0
                        for i in range(3):
                            parameter["additional_sublayers_params"][
                                1 + cur * 9 + 3
                            ] = parameter["position"][0]
                            parameter["additional_sublayers_params"][
                                1 + cur * 9 + 4
                            ] = parameter["position"][1]
                            parameter["additional_sublayers_params"][
                                1 + cur * 9 + 5
                            ] = parameter["position"][2]
                            if i == 0:
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 0
                                ] = back_size_x
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 1
                                ] = drawer_height
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 2
                                ] = back_size_z
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 3
                                ] += 0
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 4
                                ] += -drawer_height / 2
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 5
                                ] += -parameter["subs_size"][2] / 2
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 6
                                ] = 0
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 7
                                ] = 0
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 8
                                ] = 0
                                cur += 1
                            elif i == 1:
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 0
                                ] = side_size_x
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 1
                                ] = drawer_height
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 2
                                ] = side_size_z
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 3
                                ] += -parameter["subs_size"][0] / 2
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 4
                                ] += -drawer_height / 2
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 5
                                ] += 0
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 6
                                ] = 0
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 7
                                ] = 0
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 8
                                ] = 0
                                cur += 1
                            elif i == 2:
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 0
                                ] = side_size_x
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 1
                                ] = drawer_height
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 2
                                ] = side_size_z
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 3
                                ] += parameter["subs_size"][0] / 2
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 4
                                ] += -drawer_height / 2
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 5
                                ] += 0
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 6
                                ] = 0
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 7
                                ] = 0
                                parameter["additional_sublayers_params"][
                                    1 + cur * 9 + 8
                                ] = 0
                                cur += 1

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_sublayers_params"][0] = int(3)
                new_concepts.append(concept)

            elif template == "Regular_drawer":
                top_parameter = concepts[0]["parameters"]
                leg_parameter = concepts[1]["parameters"]
                layer_parameter = concepts[2]["parameters"]
                parameter["number_of_drawer"] = np.array(
                    [int(1) for _ in parameter["number_of_drawer"]]
                )
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] - top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]

                outer_size_x = (
                    layer_parameter["subs_size"][0]
                    - leg_parameter["front_legs_size"][0]
                )
                outer_size_z = (
                    layer_parameter["subs_size"][2]
                    - leg_parameter["front_legs_size"][2] / 2
                    - leg_parameter["rear_legs_size"][2] / 2
                )
                outer_size_y = -layer_parameter["subs_offset"][0]

                parameter["drawers_params"][0 * 21 + 0] = (
                    outer_size_x * randRange(1, 0.7, 0.8)[0]
                )
                parameter["drawers_params"][0 * 21 + 1] = (
                    outer_size_y * randRange(1, 0.7, 0.8)[0]
                )
                parameter["drawers_params"][0 * 21 + 2] = (
                    outer_size_z * randRange(1, 0.7, 0.8)[0]
                )
                parameter["drawers_params"][0 * 21 + 3] = (
                    outer_size_y - parameter["drawers_params"][0 * 21 + 1]
                ) / 4
                parameter["drawers_params"][0 * 21 + 4] = outer_size_x
                parameter["drawers_params"][0 * 21 + 5] = (
                    outer_size_y - layer_parameter["subs_size"][1] / 2
                )
                parameter["drawers_params"][0 * 21 + 6] *= randRange(1, 0.8, 1.2)[0]
                parameter["drawers_params"][0 * 21 + 7] = (
                    -parameter["drawers_params"][0 * 21 + 3] / 2
                )
                parameter["drawers_params"][0 * 21 + 8] = (
                    leg_parameter["front_legs_size"][0] / 3 * randRange(1, 0.7, 0.9)[0]
                )
                parameter["drawers_params"][0 * 21 + 9] = (
                    leg_parameter["rear_legs_size"][2] / 3 * randRange(1, 0.7, 0.9)[0]
                )
                parameter["drawers_params"][0 * 21 + 10] = int(1)
                parameter["drawers_params"][0 * 21 + 11] = (
                    parameter["drawers_params"][0 * 21 + 4]
                    / 3
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["drawers_params"][0 * 21 + 12] = (
                    parameter["drawers_params"][0 * 21 + 5]
                    / 20
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["drawers_params"][0 * 21 + 13] *= randRange(1, 0.9, 1.2)[0]
                parameter["drawers_params"][0 * 21 + 14] = 0
                parameter["drawers_params"][0 * 21 + 15] = 0
                parameter["drawers_params"][0 * 21 + 16] = (
                    parameter["drawers_params"][0 * 21 + 5]
                    / 4
                    * randRange(1, 0.5, 0.9)[0]
                )
                parameter["drawers_params"][0 * 21 + 17] = 0
                parameter["drawers_params"][0 * 21 + 18] = 0
                parameter["drawers_params"][0 * 21 + 19] = (
                    layer_parameter["subs_offset"][0] / 2
                    + parameter["drawers_params"][0 * 21 + 1] / 2
                    + parameter["drawers_params"][0 * 21 + 3]
                )
                parameter["drawers_params"][0 * 21 + 20] = (
                    layer_parameter["subs_size"][2]
                    - parameter["drawers_params"][21 * 0 + 2]
                ) / 2 + parameter["drawers_params"][20 * 0 + 2] * randRange(
                    1, 0.2, 0.6
                )[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["drawers_params"][0 * 21 + 10] = int(
                    parameter["drawers_params"][0 * 21 + 10]
                )
                new_concepts.append(concept)

            elif template == "Regular_partition":
                top_parameter = concepts[0]["parameters"]
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] + top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]
                parameter["has_partition"] = np.array([0, 1, 0])
                parameter["rear_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["rear_size"][1] = (
                    top_parameter["size"][1] * 4 / 7 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_size"][0] = 0
                parameter["left_right_size"][2] = (
                    top_parameter["size"][2] - 2 * parameter["rear_size"][1]
                )
                parameter["left_right_separation"][0] = (
                    top_parameter["size"][0] * randRange(1, 0.8, 1.0)[0]
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif table_type == "table_special_leg":
            if template == "Regular_desktop":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cylindrical_desktop":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Star_leg":
                top_parameter = concepts[0]["parameters"]
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] - top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]
                parameter["number_of_sub_legs"] = np.array(
                    [int(np.random.randint(3, 5))]
                )
                parameter["horizontal_rotation"][0] = 0
                parameter["vertical_size"] *= randRange(
                    parameter["vertical_size"].shape[0], 0.8, 1.2
                )
                parameter["tilt_angle"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["sub_size"][2] = parameter["vertical_size"][1] * 4 / 5
                parameter["sub_size"] *= randRange(
                    parameter["sub_size"].shape[0], 0.8, 1.2
                )
                parameter["central_rotation"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["additional_legs_params"][0] = 0

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

            elif template == "Bar_cylindrical_leg":
                top_parameter = concepts[0]["parameters"]
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] - top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]
                parameter["vertical_size"] *= randRange(
                    parameter["vertical_size"].shape[0], 0.8, 1.2
                )
                parameter["vertical_size"][0] *= randRange(1, 0.5, 1.2)[0]
                parameter["vertical_size"][1] *= randRange(1, 0.8, 1.4)[0]
                parameter["bottom_size"] *= randRange(
                    parameter["bottom_size"].shape[0], 0.5, 1.2
                )
                parameter["horizontal_rotation"][0] = 0
                parameter["additional_legs_params"][0] = 0

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

            elif template == "Bar_cuboid_leg":
                top_parameter = concepts[0]["parameters"]
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] - top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]
                parameter["vertical_size"] *= randRange(
                    parameter["vertical_size"].shape[0], 0.8, 1.2
                )
                parameter["vertical_size"][0] *= randRange(1, 0.6, 1.2)[0]
                parameter["vertical_size"][1] *= randRange(1, 0.8, 1.4)[0]
                parameter["vertical_size"][2] *= randRange(1, 0.6, 1.2)[0]
                parameter["bottom_size"] *= randRange(
                    parameter["bottom_size"].shape[0], 0.8, 1.2
                )
                parameter["bottom_size"][0] = np.minimum(
                    parameter["vertical_size"][0] * randRange(1, 1.2, 1.4)[0],
                    top_parameter["size"][0] * randRange(1, 0.5, 0.8)[0],
                )
                parameter["bottom_size"][2] = (
                    parameter["vertical_size"][2] * randRange(1, 1.2, 1.4)[0]
                )
                parameter["horizontal_rotation"][0] = 0
                parameter["additional_legs_params"][0] = 0

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

        elif table_type == "multi_level_drawer_table":
            if template == "Regular_desktop":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_leg":
                leg_type = "Regular"
                body_parameter = concepts[0]["parameters"]
                num_of_legs = 2
                parameter["number_of_legs"] = np.array([int(num_of_legs)])
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]
                parameter["additional_legs_params"][0] = 0
                parameter["symmetry_mode"][0] = 1
                if num_of_legs == 2:
                    is_rotation_suitable_for_sublayer = "SUITABLE"
                    parameter["front_rotation"][0] = 0
                    parameter["front_rotation"][1] = 0
                    parameter["central_rotation"][0] = 0
                    parameter["front_legs_size"][0] = (
                        body_parameter["size"][1] / 2 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["front_legs_size"][1] *= randRange(1, 0.8, 1.2)[0]
                    parameter["front_legs_size"][2] = (
                        body_parameter["size"][2] * randRange(1, 0.6, 1.0)[0]
                    )
                    parameter["legs_separation"][0] = (
                        body_parameter["size"][0] - 2 * parameter["front_legs_size"][0]
                    ) * randRange(1, 0.85, 1.0)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

            elif template == "Regular_sublayer":
                top_parameter = concepts[0]["parameters"]
                leg_parameter = concepts[1]["parameters"]
                parameter["position"][0] = leg_parameter["position"][0]
                parameter["position"][1] = leg_parameter["position"][1]
                parameter["position"][2] = leg_parameter["position"][2]
                parameter["rotation"][0] = leg_parameter["rotation"][0]
                parameter["rotation"][1] = leg_parameter["rotation"][1]
                parameter["rotation"][2] = leg_parameter["rotation"][2]
                parameter["number_of_subs"] = np.array([int(0)])
                part_x_length[0] = (
                    (
                        leg_parameter["legs_separation"][0]
                        - leg_parameter["front_legs_size"][0]
                    )
                    / 3
                    * randRange(1, 0.8, 1.2)[0]
                )
                part_x_length[1] = (
                    (
                        leg_parameter["legs_separation"][0]
                        - leg_parameter["front_legs_size"][0]
                    )
                    / 3
                    * randRange(1, 0.8, 1.2)[0]
                )
                part_x_length[2] = (
                    (
                        leg_parameter["legs_separation"][0]
                        - leg_parameter["front_legs_size"][0]
                    )
                    - part_x_length[0]
                    - part_x_length[1]
                )
                total_height = leg_parameter["front_legs_size"][1]
                while True:
                    square_existence = np.random.choice([0, 1], size=3)
                    if np.any(square_existence == 0) and np.any(square_existence == 1):
                        break
                cur = 0
                current_x_position = (
                    -leg_parameter["legs_separation"][0] / 2
                    + leg_parameter["front_legs_size"][0] / 2
                )
                WHOLE_layers_offset_x = (
                    -leg_parameter["legs_separation"][0] / 2
                    + leg_parameter["front_legs_size"][0] / 2
                    + part_x_length[0]
                )
                WHOLE_layers_size_x = (
                    leg_parameter["front_legs_size"][0] * randRange(1, 0.7, 0.9)[0]
                )
                if square_existence[0] or square_existence[1]:
                    parameter["additional_sublayers_params"][1 + cur * 9 + 0] = (
                        WHOLE_layers_size_x
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 1] = (
                        total_height
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 2] = (
                        leg_parameter["front_legs_size"][2]
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 3] = (
                        WHOLE_layers_offset_x + parameter["position"][0]
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 4] = (
                        -parameter["additional_sublayers_params"][1 + cur * 9 + 1] / 2
                        + parameter["position"][1]
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 5] = (
                        parameter["position"][2]
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 6] = 0
                    parameter["additional_sublayers_params"][1 + cur * 9 + 7] = 0
                    parameter["additional_sublayers_params"][1 + cur * 9 + 8] = 0
                    cur += 1
                    if square_existence[0]:
                        parameter["additional_sublayers_params"][1 + cur * 9 + 0] = (
                            part_x_length[0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 1] = (
                            total_height
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 2] = (
                            leg_parameter["front_legs_size"][0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 3] = (
                            -leg_parameter["legs_separation"][0] / 2
                            + leg_parameter["front_legs_size"][0] / 2
                            + part_x_length[0] / 2
                            + parameter["position"][0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 4] = (
                            -parameter["additional_sublayers_params"][1 + cur * 9 + 1]
                            / 2
                            + parameter["position"][1]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 5] = (
                            -leg_parameter["front_legs_size"][2] / 2
                            + parameter["position"][2]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 8] = 0
                        cur += 1
                    if square_existence[1]:
                        parameter["additional_sublayers_params"][1 + cur * 9 + 0] = (
                            part_x_length[1]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 1] = (
                            total_height
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 2] = (
                            leg_parameter["front_legs_size"][0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 3] = (
                            -leg_parameter["legs_separation"][0] / 2
                            + leg_parameter["front_legs_size"][0] / 2
                            + part_x_length[0]
                            + part_x_length[1] / 2
                            + parameter["position"][0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 4] = (
                            -parameter["additional_sublayers_params"][1 + cur * 9 + 1]
                            / 2
                            + parameter["position"][1]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 5] = (
                            -leg_parameter["front_legs_size"][2] / 2
                            + parameter["position"][2]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 8] = 0
                        cur += 1
                WHOLE_layers_offset_x += part_x_length[1]
                if square_existence[1] or square_existence[2]:
                    parameter["additional_sublayers_params"][1 + cur * 9 + 0] = (
                        WHOLE_layers_size_x
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 1] = (
                        total_height
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 2] = (
                        leg_parameter["front_legs_size"][2]
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 3] = (
                        WHOLE_layers_offset_x + parameter["position"][0]
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 4] = (
                        -parameter["additional_sublayers_params"][1 + cur * 9 + 1] / 2
                        + parameter["position"][1]
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 5] = (
                        parameter["position"][2]
                    )
                    parameter["additional_sublayers_params"][1 + cur * 9 + 6] = 0
                    parameter["additional_sublayers_params"][1 + cur * 9 + 7] = 0
                    parameter["additional_sublayers_params"][1 + cur * 9 + 8] = 0
                    cur += 1
                    if square_existence[2]:
                        parameter["additional_sublayers_params"][1 + cur * 9 + 0] = (
                            part_x_length[2]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 1] = (
                            total_height
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 2] = (
                            leg_parameter["front_legs_size"][0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 3] = (
                            -leg_parameter["legs_separation"][0] / 2
                            + leg_parameter["front_legs_size"][0] / 2
                            + part_x_length[0]
                            + part_x_length[1]
                            + part_x_length[2] / 2
                            + parameter["position"][0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 4] = (
                            -parameter["additional_sublayers_params"][1 + cur * 9 + 1]
                            / 2
                            + parameter["position"][1]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 5] = (
                            -leg_parameter["front_legs_size"][2] / 2
                            + parameter["position"][2]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 8] = 0
                        cur += 1
                for i in range(3):
                    if square_existence[i] == 0:
                        square_list.append([-1])
                        current_x_position += part_x_length[i]
                        continue
                    num_of_squares = np.random.randint(1, 5)
                    square_list.append([0 for _ in range(num_of_squares)])
                    average_height = total_height / num_of_squares
                    current_height_offset = (
                        -average_height + top_parameter["size"][1] / 4
                    )
                    for j in range(num_of_squares):
                        size_x = part_x_length[i]
                        size_y = top_parameter["size"][1] / 2
                        size_z = leg_parameter["front_legs_size"][2]
                        parameter["additional_sublayers_params"][1 + cur * 9 + 0] = (
                            size_x
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 1] = (
                            size_y
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 2] = (
                            size_z
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 3] = (
                            current_x_position + size_x / 2 + parameter["position"][0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 4] = (
                            current_height_offset + parameter["position"][1]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 5] = (
                            parameter["position"][2]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 8] = 0
                        current_height_offset -= average_height
                        cur += 1
                    current_x_position += part_x_length[i]

                parameter["additional_sublayers_params"][0] = cur
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_sublayers_params"][0] = int(cur)
                new_concepts.append(concept)

            elif template == "Regular_door":
                top_parameter = concepts[0]["parameters"]
                leg_parameter = concepts[1]["parameters"]
                parameter["position"][0] = leg_parameter["position"][0]
                parameter["position"][1] = leg_parameter["position"][1]
                parameter["position"][2] = leg_parameter["position"][2]
                parameter["rotation"][0] = leg_parameter["rotation"][0]
                parameter["rotation"][1] = leg_parameter["rotation"][1]
                parameter["rotation"][2] = leg_parameter["rotation"][2]
                total_height = leg_parameter["front_legs_size"][1]
                cur = 0
                door_x_offset = 0
                for i in range(3):
                    if square_existence[i] == 0:
                        door_x_offset += part_x_length[i]
                        continue
                    lengths_in_square_list = [len(sub_arr) for sub_arr in square_list]
                    if lengths_in_square_list[i] == 1:
                        probability = 1
                        height_of_door = total_height - top_parameter["size"][1] / 2
                        square_list[i][0] = 1
                    elif lengths_in_square_list[i] == 2:
                        num_of_vertical = np.random.randint(0, 3)
                        average_height = (
                            total_height - top_parameter["size"][1] / 2
                        ) / lengths_in_square_list[i]
                        height_of_door = average_height * num_of_vertical
                        if num_of_vertical == 0:
                            probability = 0
                        else:
                            probability = 1
                        for j in range(num_of_vertical):
                            square_list[i][1 - j] = 1
                    elif lengths_in_square_list[i] == 3:
                        num_of_vertical = np.random.randint(0, 4)
                        average_height = (
                            total_height - top_parameter["size"][1] / 2
                        ) / lengths_in_square_list[i]
                        height_of_door = average_height * num_of_vertical
                        if num_of_vertical == 0:
                            probability = 0
                        else:
                            probability = 1
                        for j in range(num_of_vertical):
                            square_list[i][2 - j] = 1
                    else:
                        num_of_vertical = np.random.randint(0, 5)
                        average_height = (
                            total_height - top_parameter["size"][1] / 2
                        ) / lengths_in_square_list[i]
                        height_of_door = average_height * num_of_vertical
                        if num_of_vertical == 0:
                            probability = 0
                        else:
                            probability = 1
                        for j in range(num_of_vertical):
                            square_list[i][3 - j] = 1
                    if probability == 0:
                        door_x_offset += part_x_length[i]
                        continue
                    parameter["doors_params"][cur * 13 + 0] = part_x_length[i]
                    parameter["doors_params"][cur * 13 + 1] = height_of_door
                    parameter["doors_params"][cur * 13 + 2] = (
                        leg_parameter["front_legs_size"][0]
                        / 2
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][cur * 13 + 3] = (
                        parameter["doors_params"][13 * cur + 0]
                        / 50
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][cur * 13 + 4] = (
                        parameter["doors_params"][13 * cur + 1]
                        / 4
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][cur * 13 + 5] = parameter["doors_params"][
                        13 * cur + 2
                    ]
                    parameter["doors_params"][cur * 13 + 6] = 0
                    parameter["doors_params"][cur * 13 + 10] = (
                        -leg_parameter["legs_separation"][0] / 2
                        + leg_parameter["front_legs_size"][0] / 2
                        + door_x_offset
                        + parameter["doors_params"][cur * 13 + 0] / 2
                    )
                    door_x_offset += parameter["doors_params"][cur * 13 + 0]
                    parameter["doors_params"][cur * 13 + 11] = (
                        -total_height
                        + top_parameter["size"][1] / 2
                        + parameter["doors_params"][cur * 13 + 1]
                    )
                    door_open_type = np.random.randint(0, 2)

                    if door_open_type == 0:
                        parameter["doors_params"][13 * cur + 9] = -randRange(1, 0, 90)[
                            0
                        ]
                        if i == 0:
                            parameter["doors_params"][13 * cur + 9] = -randRange(
                                1, 0, 180
                            )[0]
                        parameter["doors_params"][13 * cur + 7] = (
                            parameter["doors_params"][13 * cur + 0]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][cur * 13 + 10] -= (
                            parameter["doors_params"][cur * 13 + 0]
                            / 2
                            * (
                                1
                                - np.cos(
                                    parameter["doors_params"][13 * cur + 9]
                                    / 180
                                    * np.pi
                                )
                            )
                        )
                    elif door_open_type == 1:
                        parameter["doors_params"][13 * cur + 9] = randRange(1, 0, 90)[0]
                        if i == 2:
                            parameter["doors_params"][13 * cur + 9] = randRange(
                                1, 0, 180
                            )[0]
                        parameter["doors_params"][13 * cur + 7] = (
                            -parameter["doors_params"][13 * cur + 0]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][cur * 13 + 10] += (
                            parameter["doors_params"][cur * 13 + 0]
                            / 2
                            * (
                                1
                                - np.cos(
                                    parameter["doors_params"][13 * cur + 9]
                                    / 180
                                    * np.pi
                                )
                            )
                        )
                    parameter["doors_params"][cur * 13 + 12] = leg_parameter[
                        "front_legs_size"
                    ][2] / 2 + parameter["doors_params"][cur * 13 + 0] / 2 * np.sin(
                        np.abs(parameter["doors_params"][13 * cur + 9]) / 180 * np.pi
                    )
                    cur += 1
                parameter["number_of_door"] = np.array([int(cur)])

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                if cur != 0:
                    new_concepts.append(concept)

            elif template == "Regular_drawer":
                top_parameter = concepts[0]["parameters"]
                leg_parameter = concepts[1]["parameters"]
                sublayer_parameter = concepts[2]["parameters"]
                parameter["position"][0] = leg_parameter["position"][0]
                parameter["position"][1] = leg_parameter["position"][1]
                parameter["position"][2] = leg_parameter["position"][2]
                parameter["rotation"][0] = leg_parameter["rotation"][0]
                parameter["rotation"][1] = leg_parameter["rotation"][1]
                parameter["rotation"][2] = leg_parameter["rotation"][2]
                total_height = leg_parameter["front_legs_size"][1]
                cur = 0
                drawer_x_offset = (
                    -leg_parameter["legs_separation"][0] / 2
                    + leg_parameter["front_legs_size"][0] / 2
                )
                for i in range(3):
                    if square_existence[i] == 0:
                        drawer_x_offset += part_x_length[i]
                        continue
                    num_of_squares = len(square_list[i])
                    for j in range(num_of_squares):
                        if square_list[i][j] != 0:
                            break
                        average_height = (
                            total_height - top_parameter["size"][1] / 4
                        ) / num_of_squares
                        if j == 0:
                            height_of_drawer = (
                                average_height - top_parameter["size"][1] / 4
                            )
                            drawer_y_offset = 0
                            drawer_expected_height = (
                                height_of_drawer - top_parameter["size"][1] / 4
                            ) * randRange(1, 0.7, 0.8)[0]
                        else:
                            height_of_drawer = average_height
                            drawer_y_offset = (
                                -(j - 1) * average_height
                                - average_height
                                + top_parameter["size"][1] / 4
                            )
                            drawer_expected_height = (
                                height_of_drawer - top_parameter["size"][1] / 2
                            ) * randRange(1, 0.7, 0.8)[0]

                        parameter["drawers_params"][cur * 21 + 0] = (
                            part_x_length[i]
                            - sublayer_parameter["additional_sublayers_params"][1 + 0]
                        ) * randRange(1, 0.7, 0.8)[0]
                        parameter["drawers_params"][cur * 21 + 2] = (
                            leg_parameter["front_legs_size"][2]
                            - leg_parameter["front_legs_size"][0] / 2
                        ) * randRange(1, 0.7, 0.8)[0]
                        parameter["drawers_params"][cur * 21 + 3] = (
                            (top_parameter["size"][1]) / 2 * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][cur * 21 + 1] = (
                            drawer_expected_height
                            - parameter["drawers_params"][cur * 21 + 3]
                        )
                        if j == 0:
                            drawer_y_offset -= (
                                height_of_drawer - drawer_expected_height
                            ) / 2
                        else:
                            drawer_y_offset -= (
                                height_of_drawer - drawer_expected_height
                            ) / 2
                        parameter["drawers_params"][cur * 21 + 4] = part_x_length[i]
                        parameter["drawers_params"][cur * 21 + 5] = (
                            height_of_drawer * randRange(1, 0.9, 1.0)[0]
                        )
                        parameter["drawers_params"][cur * 21 + 6] = (
                            parameter["drawers_params"][6] * randRange(1, 0.8, 1.2)[0]
                        )

                        parameter["drawers_params"][cur * 21 + 7] = (
                            -parameter["drawers_params"][0 * 21 + 3] / 2
                        )

                        parameter["drawers_params"][cur * 21 + 8] = (
                            leg_parameter["front_legs_size"][0]
                            / 3
                            * randRange(1, 0.7, 0.9)[0]
                        )

                        parameter["drawers_params"][cur * 21 + 9] = (
                            leg_parameter["front_legs_size"][0]
                            / 3
                            * randRange(1, 0.7, 0.9)[0]
                        )

                        parameter["drawers_params"][cur * 21 + 10] = int(1)

                        parameter["drawers_params"][cur * 21 + 11] = (
                            parameter["drawers_params"][cur * 21 + 4]
                            / 3
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][cur * 21 + 12] = (
                            parameter["drawers_params"][cur * 21 + 5]
                            / 20
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][cur * 21 + 13] *= randRange(
                            1, 0.8, 1.2
                        )[0]

                        parameter["drawers_params"][cur * 21 + 14] = 0

                        parameter["drawers_params"][cur * 21 + 15] = 0
                        parameter["drawers_params"][cur * 21 + 16] = (
                            parameter["drawers_params"][0 * 21 + 5]
                            / 4
                            * randRange(1, 0.5, 0.9)[0]
                        )

                        parameter["drawers_params"][cur * 21 + 17] = 0

                        parameter["drawers_params"][cur * 21 + 18] = (
                            drawer_x_offset
                            + parameter["drawers_params"][cur * 21 + 4] / 2
                        )
                        parameter["drawers_params"][cur * 21 + 19] = drawer_y_offset
                        parameter["drawers_params"][cur * 21 + 20] = (
                            leg_parameter["front_legs_size"][2]
                            - parameter["drawers_params"][21 * cur + 2]
                        ) / 2 + parameter["drawers_params"][21 * cur + 2] * randRange(
                            1, 0.2, 0.6
                        )[0]
                        cur += 1
                    drawer_x_offset += part_x_length[i]
                parameter["number_of_drawer"] = np.array([int(cur)])
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                for k in range(cur):
                    concept["parameters"]["drawers_params"][k * 21 + 10] = int(1)
                if cur != 0:
                    new_concepts.append(concept)
                else:
                    pass

            elif template == "Regular_partition":
                top_parameter = concepts[0]["parameters"]
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] + top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]
                parameter["has_partition"] = np.array([0, 1, 0])
                parameter["rear_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["rear_size"][1] = (
                    top_parameter["size"][1] * 4 / 7 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_size"][0] = 0
                parameter["left_right_size"][2] = (
                    top_parameter["size"][2] - 2 * parameter["rear_size"][1]
                )
                parameter["left_right_separation"][0] = (
                    top_parameter["size"][0] * randRange(1, 0.8, 1.0)[0]
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif table_type == "Cylindrical_table":
            if template == "Cylindrical_desktop":
                parameter["size"] *= randRange(1, 0.8, 1.2)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)
            elif template == "Regular_leg":
                leg_type = "Regular"
                body_parameter = concepts[0]["parameters"]
                num_of_legs = np.random.randint(3, 5)
                parameter["symmetry_mode"][0] = 1
                parameter["number_of_legs"] = np.array([int(num_of_legs)])
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]
                parameter["additional_legs_params"][0] = 0
                if num_of_legs == 3:
                    leg_type = "3"
                    leg_height = (
                        parameter["front_legs_size"][1] * randRange(1, 0.8, 1.5)[0]
                    )
                    if random.random() < 0.5:
                        leg_height = (
                            body_parameter["size"][0] * 2 * randRange(1, 1.0, 1.2)[0]
                        )
                    parameter["front_rotation"][0] = np.random.randint(-15, 0)
                    parameter["front_rotation"][1] = np.random.randint(-15, 0)
                    parameter["central_rotation"][0] = 0
                    front_rotation_0 = parameter["front_rotation"][0] / 180 * np.pi
                    front_rotation_1 = parameter["front_rotation"][1] / 180 * np.pi
                    parameter["rear_rotation"][0] = (
                        np.arctan(
                            np.sqrt(
                                (np.tan(front_rotation_0)) ** 2
                                + (np.tan(front_rotation_1)) ** 2
                            )
                        )
                        / np.pi
                        * 180
                    )
                    total_front_angle = np.arctan(
                        np.sqrt(
                            (np.tan(front_rotation_0)) ** 2
                            + (np.tan(front_rotation_1)) ** 2
                        )
                    )
                    parameter["rear_rotation"][1] = 0

                    parameter["front_legs_size"][0] = (
                        body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                    )
                    projection_1_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][0] / 180 * np.pi
                    )
                    projection_2_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][1] / 180 * np.pi
                    )
                    parameter["front_legs_size"][1] = np.sqrt(
                        projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                    )
                    parameter["front_legs_size"][2] = (
                        body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][0] = (
                        parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][2] = (
                        parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][1] = (
                        leg_height / np.cos(parameter["rear_rotation"][0] / 180 * np.pi)
                    ) / np.cos(parameter["rear_rotation"][1] / 180 * np.pi)
                    separation_angle = 60 * randRange(1, 0.75, 1.5)[0] / 180 * np.pi
                    separation_radius = (
                        body_parameter["size"][0] * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["legs_separation"][0] = (
                        separation_radius * np.sin(separation_angle / 2) * 2
                    )
                    parameter["legs_separation"][2] = (
                        separation_radius * np.cos(separation_angle / 2) * 2
                    )
                    half_leg_separation_0 = parameter["legs_separation"][0] / 2
                    half_leg_separation_1 = parameter["legs_separation"][2] / 2
                    radius_of_sublayer = (
                        half_leg_separation_0**2 + (4 * half_leg_separation_1**2)
                    ) / (4 * half_leg_separation_1)
                    sublayer_z_offset = -half_leg_separation_1 + radius_of_sublayer
                elif num_of_legs == 4:
                    leg_type = "4"
                    leg_height = (
                        parameter["front_legs_size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                    if random.random() < 0.5:
                        leg_height = (
                            body_parameter["size"][0] * 2 * randRange(1, 1.0, 1.2)[0]
                        )
                    is_rotation_suitable_for_sublayer = "SUITABLE"
                    parameter["front_rotation"][0] = -randRange(1, 0, 15)[0]
                    parameter["front_rotation"][1] = -randRange(1, 0, 15)[0]
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = -parameter["front_rotation"][0]
                    parameter["rear_rotation"][1] = 0
                    front_rotation_0 = parameter["front_rotation"][0] / 180 * np.pi
                    front_rotation_1 = parameter["front_rotation"][1] / 180 * np.pi
                    total_front_angle = np.arctan(
                        np.sqrt(
                            (np.tan(front_rotation_0)) ** 2
                            + (np.tan(front_rotation_1)) ** 2
                        )
                    )
                    parameter["front_legs_size"][0] = (
                        body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                    )
                    projection_1_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][0] / 180 * np.pi
                    )
                    projection_2_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][1] / 180 * np.pi
                    )
                    parameter["front_legs_size"][1] = np.sqrt(
                        projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                    )
                    parameter["front_legs_size"][2] = (
                        body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][0] = (
                        parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][2] = (
                        parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                    )
                    projection_1_of_leg = leg_height * np.tan(
                        parameter["rear_rotation"][0] / 180 * np.pi
                    )
                    projection_2_of_leg = leg_height * np.tan(
                        parameter["rear_rotation"][1] / 180 * np.pi
                    )
                    parameter["rear_legs_size"][1] = np.sqrt(
                        projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                    )
                    separation_radius = (
                        body_parameter["size"][0] * randRange(1, 0.8, 1.0)[0]
                    )
                    legs_separation_position_angle = (
                        45 * randRange(1, 0.8, 1.2)[0] / 180 * np.pi
                    )
                    parameter["legs_separation"][0] = (
                        separation_radius * 2 * np.sin(legs_separation_position_angle)
                    )
                    parameter["legs_separation"][2] = (
                        separation_radius * 2 * np.cos(legs_separation_position_angle)
                    )
                    parameter["legs_separation"][1] = (
                        separation_radius * 2 * np.sin(legs_separation_position_angle)
                    )
                    radius_of_sublayer = separation_radius
                    sublayer_z_offset = 0
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

            elif template == "Cylindrical_sublayer":
                top_parameter = concepts[0]["parameters"]
                leg_parameter = concepts[1]["parameters"]
                parameter["position"][0] = leg_parameter["position"][0]
                parameter["position"][1] = leg_parameter["position"][1]
                parameter["position"][2] = leg_parameter["position"][2]
                parameter["rotation"][0] = leg_parameter["rotation"][0]
                parameter["rotation"][1] = leg_parameter["rotation"][1]
                parameter["rotation"][2] = leg_parameter["rotation"][2]

                if leg_type == "3":
                    parameter["number_of_subs"] = np.array([int(1)])
                    leg_basic_height = (
                        leg_parameter["front_legs_size"][1]
                        * np.cos(total_front_angle)
                        / 2
                    ) * randRange(1, 0.8, 1.2)[0]
                    size_mul = leg_basic_height / (
                        leg_parameter["front_legs_size"][1]
                        * np.cos(total_front_angle)
                        / 2
                    )
                    leg_projection_xz = leg_parameter["front_legs_size"][1] * np.tan(
                        total_front_angle
                    )
                    cur_radius = (
                        radius_of_sublayer
                        - leg_projection_xz
                        + leg_projection_xz * size_mul
                    )
                    parameter["additional_sublayers_params"][0] = 0
                    parameter["subs_size"][0] = cur_radius
                    parameter["position"][2] += sublayer_z_offset
                    parameter["subs_size"][1] = (
                        top_parameter["size"][1] / 2 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["subs_offset"][0] = -leg_basic_height
                elif leg_type == "4":
                    parameter["number_of_subs"] = np.array([int(1)])
                    leg_basic_height = (
                        leg_parameter["front_legs_size"][1]
                        * np.cos(total_front_angle)
                        / 2
                    ) * randRange(1, 0.8, 1.2)[0]
                    size_mul = leg_basic_height / (
                        leg_parameter["front_legs_size"][1]
                        * np.cos(total_front_angle)
                        / 2
                    )
                    leg_projection_xz = leg_parameter["front_legs_size"][1] * np.tan(
                        total_front_angle
                    )
                    cur_radius = (
                        radius_of_sublayer
                        - leg_projection_xz
                        + leg_projection_xz * size_mul
                    )
                    parameter["additional_sublayers_params"][0] = 0
                    parameter["subs_size"][0] = cur_radius
                    parameter["subs_size"][1] = (
                        top_parameter["size"][1] / 2 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["subs_offset"][0] = -leg_basic_height

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_sublayers_params"][0] = int(0)
                new_concepts.append(concept)

        elif table_type == "special_layer_table":
            if template == "Regular_desktop":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_leg":
                leg_type = "Regular"
                body_parameter = concepts[0]["parameters"]
                num_of_legs = 4
                parameter["symmetry_mode"][0] = 1
                parameter["number_of_legs"] = np.array([int(num_of_legs)])
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]
                parameter["additional_legs_params"][0] = 0
                if num_of_legs == 4:
                    leg_height = (
                        parameter["front_legs_size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                    if random.random() < 0.5:
                        leg_height = (
                            body_parameter["size"][0] * randRange(1, 1.0, 1.2)[0]
                        )
                    is_rotation_suitable_for_sublayer = "SUITABLE"
                    parameter["front_rotation"][0] = 0
                    parameter["front_rotation"][1] = 0
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = -parameter["front_rotation"][0]
                    parameter["rear_rotation"][1] = parameter["front_rotation"][1]

                    parameter["front_legs_size"][0] = (
                        body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                    )
                    projection_1_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][0] / 180 * np.pi
                    )
                    projection_2_of_leg = leg_height * np.tan(
                        parameter["front_rotation"][1] / 180 * np.pi
                    )
                    parameter["front_legs_size"][1] = np.sqrt(
                        projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                    )
                    parameter["front_legs_size"][2] = (
                        body_parameter["size"][2] / 10 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][0] = (
                        parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rear_legs_size"][2] = (
                        parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                    )
                    projection_1_of_leg = leg_height * np.tan(
                        parameter["rear_rotation"][0] / 180 * np.pi
                    )
                    projection_2_of_leg = leg_height * np.tan(
                        parameter["rear_rotation"][1] / 180 * np.pi
                    )
                    parameter["rear_legs_size"][1] = np.sqrt(
                        projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                    )
                    parameter["legs_separation"][0] = (
                        body_parameter["size"][0] - parameter["front_legs_size"][0]
                    ) * randRange(1, 0.7, 1.0)[0]
                    parameter["legs_separation"][1] = (
                        body_parameter["size"][0] - parameter["front_legs_size"][0]
                    ) * randRange(1, 0.7, 1.0)[0]
                    parameter["legs_separation"][2] = (
                        body_parameter["size"][2]
                        - parameter["front_legs_size"][2]
                        - parameter["rear_legs_size"][1]
                        * np.sin(parameter["rear_rotation"][0] / 180 * np.pi)
                        / 2
                    ) * randRange(1, 0.7, 1.0)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

            elif template == "Regular_with_splat_leg":
                leg_type = "Splat"
                body_parameter = concepts[0]["parameters"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]
                probabilities = [
                    0.5777777777777777,
                    0.5777777777777777,
                    0.9111111111111111,
                    0.9111111111111111,
                ]
                parameter["bridging_bars_existance"] = np.array(
                    [np.random.choice([0, 1], p=[1 - p, p]) for p in probabilities]
                )
                leg_height = parameter["front_legs_size"][1] * randRange(1, 0.8, 1.2)[0]
                if random.random() < 0.5:
                    leg_height = body_parameter["size"][0] * randRange(1, 1.0, 1.2)[0]

                is_rotation_suitable_for_sublayer = "SUITABLE"
                parameter["front_rotation"][0] = 0
                parameter["front_rotation"][1] = 0
                parameter["central_rotation"][0] = 0
                parameter["rear_rotation"][0] = -parameter["front_rotation"][0]
                parameter["rear_rotation"][1] = parameter["front_rotation"][1]
                parameter["front_legs_size"][0] = (
                    body_parameter["size"][0] / 14 * randRange(1, 0.8, 1.2)[0]
                )
                projection_1_of_leg = leg_height * np.tan(
                    parameter["front_rotation"][0] / 180 * np.pi
                )
                projection_2_of_leg = leg_height * np.tan(
                    parameter["front_rotation"][1] / 180 * np.pi
                )
                parameter["front_legs_size"][1] = np.sqrt(
                    projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                )
                parameter["front_legs_size"][2] = (
                    body_parameter["size"][2] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][0] = (
                    parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][2] = (
                    parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                projection_1_of_leg = leg_height * np.tan(
                    parameter["rear_rotation"][0] / 180 * np.pi
                )
                projection_2_of_leg = leg_height * np.tan(
                    parameter["rear_rotation"][1] / 180 * np.pi
                )
                parameter["rear_legs_size"][1] = np.sqrt(
                    projection_1_of_leg**2 + projection_2_of_leg**2 + leg_height**2
                )
                parameter["legs_separation"][0] = (
                    body_parameter["size"][0] - parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][1] = (
                    body_parameter["size"][0] - parameter["rear_legs_size"][0]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][2] = (
                    body_parameter["size"][2]
                    - parameter["rear_legs_size"][2]
                    - parameter["rear_legs_size"][1]
                    * np.sin(parameter["rear_rotation"][0] / 180 * np.pi)
                    / 2
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["front_rear_bridging_bars_sizes"][0] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["front_rear_bridging_bars_sizes"][1] = np.minimum(
                    parameter["front_legs_size"][2], parameter["rear_legs_size"][2]
                )
                parameter["left_right_bridging_bars_sizes"][1] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["left_right_bridging_bars_sizes"][0] = np.minimum(
                    parameter["front_legs_size"][0], parameter["rear_legs_size"][0]
                )

                parameter["front_rear_bridging_bars_offset"][0] = (
                    -leg_height / 2 * randRange(1, 0.1, 0.8)[0]
                )
                parameter["left_right_bridging_bars_offset"][0] = (
                    -leg_height / 2 * randRange(1, 0.1, 0.8)[0]
                )

                parameter["front_rear_bridging_bars_offset"][1] = (
                    parameter["front_rear_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_bridging_bars_offset"][1] = (
                    parameter["left_right_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["additional_legs_params"][0] = int(0)

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(0)
                new_concepts.append(concept)

            elif template == "Regular_sublayer":
                top_parameter = concepts[0]["parameters"]
                body_parameter = concepts[1]["parameters"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]
                parameter["number_of_subs"] = np.array(
                    [int(0) for _ in parameter["number_of_subs"]]
                )
                parameter["additional_sublayers_params"][0] = int(0)
                cur = 0
                if leg_type == "Regular":
                    if body_parameter["number_of_legs"][0] % 2 == 0:
                        if is_rotation_suitable_for_sublayer == "SUITABLE":
                            parameter["number_of_subs"] = np.array(
                                [int(1) for _ in parameter["number_of_subs"]]
                            )
                            rotation_front_x = (
                                body_parameter["front_rotation"][0] / 180 * np.pi
                            )
                            rotation_front_z = (
                                body_parameter["front_rotation"][1] / 180 * np.pi
                            )
                            if body_parameter["number_of_legs"][0] == 4:
                                offset_y_of_layer = (
                                    body_parameter["front_legs_size"][1]
                                    / 2
                                    * np.cos(rotation_front_z)
                                    * np.cos(rotation_front_x)
                                    * np.random.uniform(-0.8, 0)
                                )
                                parameter["subs_offset"][0] = (
                                    -top_parameter["size"][1] / 2
                                    - body_parameter["front_legs_size"][1]
                                    / 2
                                    * np.cos(rotation_front_z)
                                    * np.cos(rotation_front_x)
                                    - offset_y_of_layer
                                )
                                parameter["subs_size"][1] = (
                                    top_parameter["size"][1]
                                    / 2
                                    * randRange(1, 0.8, 1.0)[0]
                                )
                                parameter["subs_size"][0] = np.maximum(
                                    body_parameter["legs_separation"][0],
                                    body_parameter["legs_separation"][1],
                                ) - 2 * offset_y_of_layer / np.cos(
                                    rotation_front_z
                                ) * np.sin(rotation_front_z)
                                parameter["subs_size"][2] = body_parameter[
                                    "legs_separation"
                                ][2] + 2 * offset_y_of_layer / np.cos(
                                    rotation_front_x
                                ) * np.sin(rotation_front_z)

                elif leg_type == "Splat":
                    if is_rotation_suitable_for_sublayer == "SUITABLE":
                        parameter["number_of_subs"] = np.array(
                            [int(1) for _ in parameter["number_of_subs"]]
                        )
                        rotation_front_x = (
                            body_parameter["front_rotation"][0] / 180 * np.pi
                        )
                        rotation_front_z = (
                            body_parameter["front_rotation"][1] / 180 * np.pi
                        )
                        offset_y_of_layer = (
                            body_parameter["front_legs_size"][1]
                            / 2
                            * np.cos(rotation_front_z)
                            * np.cos(rotation_front_x)
                            * np.random.uniform(-0.8, 0)
                        )
                        parameter["subs_offset"][0] = (
                            -top_parameter["size"][1] / 2
                            - body_parameter["front_legs_size"][1]
                            / 2
                            * np.cos(rotation_front_z)
                            * np.cos(rotation_front_x)
                            - offset_y_of_layer
                        )
                        parameter["subs_size"][1] = (
                            top_parameter["size"][1] / 2 * randRange(1, 0.8, 1.0)[0]
                        )
                        parameter["subs_size"][0] = np.maximum(
                            body_parameter["legs_separation"][0],
                            body_parameter["legs_separation"][1],
                        ) - 2 * offset_y_of_layer / np.cos(rotation_front_z) * np.sin(
                            rotation_front_z
                        )
                        parameter["subs_size"][2] = body_parameter["legs_separation"][
                            2
                        ] + 2 * offset_y_of_layer / np.cos(rotation_front_x) * np.sin(
                            rotation_front_z
                        )
                parameter["additional_sublayers_params"][0] = 4
                front_size = (
                    body_parameter["legs_separation"][0]
                    - body_parameter["front_legs_size"][0]
                ) * randRange(1, 0.5, 0.8)[0]
                side_size = (
                    body_parameter["legs_separation"][2]
                    - body_parameter["front_legs_size"][2]
                ) * randRange(1, 0.5, 0.8)[0]
                layer_height = -parameter["subs_offset"][0] * randRange(1, 0.5, 0.7)[0]
                for j in range(4):
                    if j == 0 or j == 2:
                        parameter["additional_sublayers_params"][1 + cur * 9 + 0] = (
                            parameter["subs_size"][1] * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 1] = (
                            layer_height
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 2] = (
                            side_size
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 3] = (
                            parameter["position"][0]
                            + front_size / 2 * (-1 if j == 0 else 1)
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 4] = (
                            parameter["position"][1]
                            + parameter["additional_sublayers_params"][1 + cur * 9 + 1]
                            / 2
                            + parameter["subs_offset"][0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 5] = (
                            parameter["position"][2] + 0
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 8] = 0
                        cur += 1
                    if j == 1 or j == 3:
                        parameter["additional_sublayers_params"][1 + cur * 9 + 0] = (
                            front_size
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 1] = (
                            layer_height
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 2] = (
                            parameter["subs_size"][1] * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 3] = (
                            parameter["position"][0] + 0
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 4] = (
                            parameter["position"][1]
                            + parameter["additional_sublayers_params"][1 + cur * 9 + 1]
                            / 2
                            + parameter["subs_offset"][0]
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 5] = (
                            parameter["position"][2]
                            + side_size / 2 * (-1 if j == 1 else 1)
                        )
                        parameter["additional_sublayers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_sublayers_params"][1 + cur * 9 + 8] = 0
                        cur += 1

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_sublayers_params"][0] = int(4)
                new_concepts.append(concept)

            elif template == "Regular_backboard":
                leg_parameter = concepts[1]["parameters"]
                parameter["position"][0] = leg_parameter["position"][0]
                parameter["position"][1] = leg_parameter["position"][1]
                parameter["position"][2] = leg_parameter["position"][2]
                parameter["rotation"][0] = leg_parameter["rotation"][0]
                parameter["rotation"][1] = leg_parameter["rotation"][1]
                parameter["rotation"][2] = leg_parameter["rotation"][2]
                parameter["size"][0] = (
                    leg_parameter["legs_separation"][1]
                    - leg_parameter["rear_legs_size"][0]
                )
                parameter["size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][2] = (
                    leg_parameter["rear_legs_size"][2] * randRange(1, 0.7, 0.9)[0]
                )
                parameter["position"][2] -= leg_parameter["legs_separation"][2] / 2
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_partition":
                top_parameter = concepts[0]["parameters"]
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] + top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]
                parameter["has_partition"] = np.array([0, 1, 0])
                parameter["rear_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["rear_size"][1] = (
                    top_parameter["size"][1] * 4 / 7 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_size"][0] = 0
                parameter["left_right_size"][2] = (
                    top_parameter["size"][2] - 2 * parameter["rear_size"][1]
                )
                parameter["left_right_separation"][0] = (
                    top_parameter["size"][0] * randRange(1, 0.8, 1.0)[0]
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif table_type == "square_table":
            if template == "Regular_desktop":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)
            elif template == "Bar_cuboid_leg":
                top_parameter = concepts[0]["parameters"]
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] - top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]
                have_additional_legs = np.random.randint(0, 2)
                if have_additional_legs == 0:
                    parameter["vertical_size"] *= randRange(
                        parameter["vertical_size"].shape[0], 0.8, 1.2
                    )
                    parameter["vertical_size"][1] = (
                        top_parameter["size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["bottom_size"][0] = parameter["vertical_size"][0]
                    parameter["bottom_size"][1] *= randRange(1, 0.8, 1.2)[0]
                    parameter["bottom_size"][2] = parameter["vertical_size"][2]
                    parameter["horizontal_rotation"][0] = 0
                    parameter["additional_legs_params"][0] = 0
                elif have_additional_legs == 1:
                    leg_height = top_parameter["size"][0] * randRange(1, 0.8, 1.2)[0]
                    leg_size_x = top_parameter["size"][1] * randRange(1, 0.8, 1.2)[0]
                    leg_size_z = top_parameter["size"][1] * randRange(1, 0.8, 1.2)[0]
                    length_offset_x = (
                        top_parameter["size"][0] - leg_size_x
                    ) * randRange(1, 0.7, 1.0)[0]
                    length_offset_z = (
                        top_parameter["size"][2] - leg_size_z
                    ) * randRange(1, 0.7, 1.0)[0]
                    parameter["additional_legs_params"][0] = 4
                    cur = 0
                    for i in range(4):
                        parameter["additional_legs_params"][1 + 9 * cur + 0] = (
                            leg_size_x
                        )
                        parameter["additional_legs_params"][1 + 9 * cur + 1] = (
                            leg_height
                        )
                        parameter["additional_legs_params"][1 + 9 * cur + 2] = (
                            leg_size_z
                        )
                        parameter["additional_legs_params"][1 + 9 * cur + 6] = 0
                        parameter["additional_legs_params"][1 + 9 * cur + 7] = 0
                        parameter["additional_legs_params"][1 + 9 * cur + 8] = 0
                        if i == 0 or i == 1:
                            parameter["additional_legs_params"][1 + 9 * cur + 3] = (
                                length_offset_x / 2 * (-1 if i == 0 else 1)
                                + parameter["position"][0]
                            )
                            parameter["additional_legs_params"][1 + 9 * cur + 5] = (
                                length_offset_z / 2 + parameter["position"][2]
                            )
                            parameter["additional_legs_params"][1 + 9 * cur + 4] = (
                                leg_height
                            )
                        elif i == 2 or i == 3:
                            parameter["additional_legs_params"][1 + 9 * cur + 3] = (
                                length_offset_x / 2 * (-1 if i == 2 else 1)
                                + parameter["position"][0]
                            )
                            parameter["additional_legs_params"][1 + 9 * cur + 5] = (
                                -length_offset_z / 2 + parameter["position"][2]
                            )
                            parameter["additional_legs_params"][1 + 9 * cur + 4] = (
                                leg_height
                            )
                        cur += 1
                    parameter["vertical_size"][0] = (
                        length_offset_x * randRange(1, 1.0, 1.2)[0]
                    )
                    parameter["vertical_size"][2] = (
                        length_offset_z * randRange(1, 1.0, 1.2)[0]
                    )
                    parameter["vertical_size"][1] = (
                        top_parameter["size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["bottom_size"][0] = (
                        parameter["vertical_size"][0] * randRange(1, 0.5, 0.8)[0]
                    )
                    parameter["bottom_size"][1] *= randRange(1, 0.8, 1.2)[0]
                    parameter["bottom_size"][2] = (
                        parameter["vertical_size"][2] * randRange(1, 0.5, 0.8)[0]
                    )
                    parameter["horizontal_rotation"][0] = 0
                    parameter["position"][1] -= leg_height
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["additional_legs_params"][0] = int(
                    parameter["additional_legs_params"][0]
                )
                new_concepts.append(concept)

            elif template == "Regular_partition":
                top_parameter = concepts[0]["parameters"]
                parameter["position"][0] = top_parameter["position"][0]
                parameter["position"][1] = (
                    top_parameter["position"][1] + top_parameter["size"][1] / 2
                )
                parameter["position"][2] = top_parameter["position"][2]
                parameter["rotation"][0] = top_parameter["rotation"][0]
                parameter["rotation"][1] = top_parameter["rotation"][1]
                parameter["rotation"][2] = top_parameter["rotation"][2]
                parameter["has_partition"] = np.array([0, 1, 0])
                parameter["rear_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["rear_size"][1] = (
                    top_parameter["size"][1] * 4 / 7 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_size"][0] = 0
                parameter["left_right_size"][2] = (
                    top_parameter["size"][2] - 2 * parameter["rear_size"][1]
                )
                parameter["left_right_separation"][0] = (
                    top_parameter["size"][0] * randRange(1, 0.8, 1.0)[0]
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

    return new_concepts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", default=False, action="store_true")
    parser.add_argument(
        "--gen_num", default=10, type=int, help="number of objects to generate"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.debug:
        table_type = get_table_type()
        existing_concept_templates = concept_template_existence(table_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, table_type)

        vertices, faces = get_overall_model(new_concepts)
        pointcloud = get_overall_pointcloud(vertices, faces)
        opt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))
        coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        vertices = o3d.utility.Vector3dVector(vertices)
        faces = o3d.utility.Vector3iVector(faces)
        opt_mesh = o3d.geometry.TriangleMesh(vertices, faces)
        opt_mesh.compute_vertex_normals()
        coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        o3d.visualization.draw_geometries([opt_mesh])

    else:
        concept_list = []
        for obj_idx in range(args.gen_num):
            table_type = get_table_type()
            existing_concept_templates = concept_template_existence(table_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, table_type)
            concept_list.append(new_concepts)

            vertices, faces = get_overall_model(new_concepts)
            pointcloud = get_overall_pointcloud(vertices, faces)
            opt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))
            coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            vertices = o3d.utility.Vector3dVector(vertices)
            faces = o3d.utility.Vector3iVector(faces)
            opt_mesh = o3d.geometry.TriangleMesh(vertices, faces)
            opt_mesh.compute_vertex_normals()
            coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

            o3d.visualization.draw_geometries([opt_mesh])

        save_path = (
            f"{os.path.dirname(os.path.abspath(__file__))}/conceptualization.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(concept_list, f)
