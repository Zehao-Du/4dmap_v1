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


def get_storagefurniture_type():
    storagefurniture_type = [
        "Fridge",
        "Cabinet_transverse",
        "Cabinet_vertical",
        "wardrobe",
        "panel_Cabinet",
        "multi_drawer_Cabinet",
        "Multi_layer_wardrobe",
        "bookcase",
        "Drinking_machine",
    ]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    storagefurniture_type = random.choices(storagefurniture_type, weights=weights, k=1)[
        0
    ]

    return storagefurniture_type


def concept_template_existence(storagefurniture_type):
    if storagefurniture_type == "Fridge":
        concept_template_variation = {
            "body": {"template": ["Storagefurniture_body"], "neccessity": True},
            "leg": {"template": ["Enclosed_leg"], "neccessity": True},
            "door": {"template": ["Regular_door"], "neccessity": True},
        }
    elif storagefurniture_type == "Cabinet_transverse":
        concept_template_variation = {
            "body": {"template": ["Storagefurniture_body"], "neccessity": True},
            "door": {"template": ["Regular_door"], "neccessity": True},
            "drawer": {"template": ["Regular_drawer"], "neccessity": True},
        }
    elif storagefurniture_type == "Cabinet_vertical":
        concept_template_variation = {
            "body": {"template": ["Storagefurniture_body"], "neccessity": True},
            "door": {"template": ["Regular_door"], "neccessity": True},
            "drawer": {"template": ["Regular_drawer"], "neccessity": True},
        }
    elif storagefurniture_type == "wardrobe":
        concept_template_variation = {
            "body": {"template": ["Storagefurniture_body"], "neccessity": True},
            "door": {"template": ["Regular_door"], "neccessity": True},
        }
    elif storagefurniture_type == "panel_Cabinet":
        concept_template_variation = {
            "body": {"template": ["Storagefurniture_body"], "neccessity": True},
            "door": {"template": ["Regular_door"], "neccessity": True},
            "panel": {"template": ["Regular_front_panel"], "neccessity": True},
        }
    elif storagefurniture_type == "multi_drawer_Cabinet":
        concept_template_variation = {
            "body": {"template": ["Storagefurniture_body"], "neccessity": True},
            "drawer": {"template": ["Regular_drawer"], "neccessity": True},
        }
    elif storagefurniture_type == "Multi_layer_wardrobe":
        concept_template_variation = {
            "body": {"template": ["Storagefurniture_body"], "neccessity": True},
            "door": {"template": ["Regular_door"], "neccessity": True},
        }
    elif storagefurniture_type == "bookcase":
        concept_template_variation = {
            "body": {"template": ["Storagefurniture_body"], "neccessity": True},
            "door": {"template": ["Regular_door"], "neccessity": True},
            "panel": {"template": ["Regular_front_panel"], "neccessity": True},
        }
    elif storagefurniture_type == "Drinking_machine":
        concept_template_variation = {
            "body": {"template": ["Storagefurniture_body"], "neccessity": True},
            "door": {"template": ["Regular_door"], "neccessity": True},
        }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["neccessity"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, storagefurniture_type):
    DOOR_PARAMS_PER_DOOR = 13
    DOOR_0_ACTION = 0
    DOOR_1_ACTION = 1
    DOOR_2_ACTION = 2

    def process_enclosed_leg(item):
        parameters = item.get("parameters", {})
        parameters.pop("type", None)

    def process_door_params(door_params, door_type, index):
        theta = door_params[index + 9] * np.pi / 180
        sin_theta = np.sin(np.abs(theta))
        cos_theta = np.cos(theta)

        # 公共参数
        width = door_params[index + 1]
        height = door_params[index + 2]

        if door_type == DOOR_0_ACTION:
            door_params[index + 12] += width / 2 * np.abs(sin_theta)
            door_params[index + 10] -= width / 2 * (1 - cos_theta)

        elif door_type == DOOR_1_ACTION:
            door_params[index + 12] += width / 2 * np.abs(sin_theta)
            door_params[index + 10] += width / 2 * (1 - cos_theta)

        elif door_type == DOOR_2_ACTION:
            door_params[index + 12] += height / 2 * sin_theta
            door_params[index + 11] -= height / 2 * (1 - cos_theta)

    def process_regular_door(item):
        parameters = item.get("parameters")
        if not parameters:
            return
        door_params = parameters.get("doors_params")
        if not door_params:
            return

        # 逆向处理避免索引错位
        for door_index in reversed(range(8)):
            # for door_index in reversed(range(parameters['number_of_door'][0])):
            start_index = DOOR_PARAMS_PER_DOOR * door_index
            if len(door_params) < start_index + DOOR_PARAMS_PER_DOOR:
                continue

            door_type = int(door_params[start_index])
            process_door_params(door_params, door_type, start_index)
            # 移除已处理的类型字段
            door_params.pop(start_index)

    def convert_door_params(data):
        for item in data:
            if not isinstance(item, dict):
                continue

            template_type = item.get("template")
            if template_type == "Enclosed_leg":
                process_enclosed_leg(item)
            elif template_type == "Regular_door":
                process_regular_door(item)

        return data

    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    occupancy_of_each_compartment = []

    num_of_additional_layers = []

    for concept in concepts:
        template = concept["template"]
        parameter = {
            k: np.array(v)
            if k
            not in [
                "storagefurniture_layers_params",
                "additional_layers_params",
                "drawers_params",
                "additional_legs_params",
            ]
            else v
            for k, v in concept["parameters"].items()
        }

        if template == "Storagefurniture_body":
            parameter["size"] *= randRange(parameter["size"].shape[0], 0.7, 1.3)
            parameter["back_size"] *= randRange(
                parameter["back_size"].shape[0], 0.7, 1.3
            )
            parameter["left_right_inner_size"] *= randRange(
                parameter["left_right_inner_size"].shape[0], 0.7, 1.3
            )
            parameter["base_size"] *= randRange(
                parameter["base_size"].shape[0], 0.7, 1.3
            )
            parameter["has_lid"][0] = np.random.randint(0, 2)
            parameter["lid_size"] *= randRange(parameter["lid_size"].shape[0], 0.7, 1.3)
            parameter["lid_offset"] *= randRange(
                parameter["lid_offset"].shape[0], 0.7, 1.3
            )

            if storagefurniture_type == "Fridge":
                parameter["has_lid"][0] = 1
                parameter["lid_size"][0] = parameter["size"][0]
                parameter["lid_size"][2] = parameter["size"][2]
                parameter["WHOLE_number_of_layer"][0] = np.random.randint(0, 5)
                parameter["WHOLE_number_of_layer"] = np.array(
                    [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                )
                num_of_levels = int(parameter["WHOLE_number_of_layer"][0])

                parameter["WHOLE_layer_offset"][0] *= randRange(1, 0.8, 1.2)[0]

                if num_of_levels == 0:
                    num_of_levels = 1

                average_level_interval = (
                    parameter["size"][1]
                    - parameter["lid_offset"][1]
                    - parameter["base_size"][0]
                    - parameter["WHOLE_layer_offset"][0]
                ) / num_of_levels
                for i in range(num_of_levels - 1):
                    parameter["WHOLE_interval_between_layers"][i] = (
                        average_level_interval * randRange(1, 0.8, 1.2)[0]
                    )

                parameter["WHOLE_layer_sizes"][0] *= randRange(1, 0.8, 1.2)[0]

                for i in range(num_of_levels + 1):
                    parameter["storagefurniture_layers_params"][i * 5] = (
                        np.random.randint(0, 5)
                    )
                    num_of_vertical = parameter["storagefurniture_layers_params"][i * 5]
                    parameter["storagefurniture_layers_params"][i * 5 + 1] *= randRange(
                        1, 0.8, 1.2
                    )[0]
                    parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                        parameter["size"][2] - parameter["back_size"][0]
                    )
                    average_vertical_interval = (
                        parameter["size"][0] - 2 * parameter["left_right_inner_size"][0]
                    ) / (num_of_vertical + 1)

                    parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                        average_vertical_interval * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                        average_vertical_interval * randRange(1, 0.8, 1.2)[0]
                    )

                parameter["additional_layers_params"][0] = 0

            elif storagefurniture_type == "Cabinet_transverse":
                parameter["size"][0] = np.maximum(
                    4 * parameter["size"][1] / 3, parameter["size"][0]
                )
                parameter["lid_size"][0] = parameter["size"][0]
                parameter["lid_size"][2] = parameter["size"][2]
                parameter["WHOLE_number_of_layer"][0] = 1
                parameter["WHOLE_number_of_layer"] = np.array(
                    [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                )
                num_of_levels = int(parameter["WHOLE_number_of_layer"][0])
                average_level_interval = (
                    (
                        parameter["size"][1]
                        - parameter["lid_offset"][1]
                        - parameter["base_size"][0]
                        - parameter["WHOLE_layer_offset"][0]
                    )
                    * 2
                    / 7
                )
                parameter["WHOLE_layer_offset"][0] = (
                    average_level_interval * randRange(1, 0.8, 1.2)[0]
                )

                height_remained = (
                    parameter["size"][1]
                    - parameter["base_size"][0]
                    - parameter["WHOLE_layer_offset"][0]
                )
                average_remained_height = height_remained / num_of_levels

                for i in range(num_of_levels - 1):
                    parameter["WHOLE_interval_between_layers"][i] = (
                        average_remained_height * randRange(1, 0.8, 1.2)[0]
                    )

                parameter["WHOLE_layer_sizes"][0] *= randRange(1, 0.8, 1.2)[0]
                num_of_vertical_on_second_layer = 0
                for i in range(num_of_levels + 1):
                    if i == 0:
                        parameter["storagefurniture_layers_params"][i * 5] = (
                            np.random.randint(0, 3)
                        )
                        num_of_vertical = parameter["storagefurniture_layers_params"][
                            i * 5
                        ]
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2] - parameter["back_size"][0]
                        )
                        average_vertical_interval = (
                            parameter["size"][0]
                            - 2 * parameter["left_right_inner_size"][0]
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            average_vertical_interval * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            average_vertical_interval * randRange(1, 0.8, 1.2)[0]
                        )
                    else:
                        parameter["storagefurniture_layers_params"][i * 5] = (
                            np.random.randint(1, 3)
                        )
                        num_of_vertical = parameter["storagefurniture_layers_params"][
                            i * 5
                        ]
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2] - parameter["back_size"][0]
                        )
                        average_vertical_interval = (
                            parameter["size"][0]
                            - 2 * parameter["left_right_inner_size"][0]
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            average_vertical_interval * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            average_vertical_interval * randRange(1, 0.8, 1.2)[0]
                        )
                        num_of_vertical_on_second_layer = int(num_of_vertical)

                num_of_layers_in_each_vertical_lattice = [
                    np.random.randint(0, 3)
                    for _ in range(int(num_of_vertical_on_second_layer + 1))
                ]

                parameter["additional_layers_params"][0] = (
                    sum(num_of_layers_in_each_vertical_lattice) * 1.0
                )

                cur = 0
                for i in range(num_of_vertical_on_second_layer + 1):
                    if num_of_layers_in_each_vertical_lattice[i] == 0:
                        occupancy_of_each_compartment.append([0])
                        continue
                    else:
                        if i == 0:
                            cur_compartment_y_offset = 0
                            cur_number_of_layers = (
                                num_of_layers_in_each_vertical_lattice[i]
                            )
                            total_remained_height = (
                                parameter["size"][1]
                                - parameter["base_size"][0]
                                - parameter["WHOLE_layer_offset"][0]
                            )

                            average_total_remained_height = (
                                total_remained_height
                                - (cur_number_of_layers + 1)
                                * parameter["WHOLE_layer_sizes"][0]
                            ) / (cur_number_of_layers + 1)
                            for j in range(num_of_layers_in_each_vertical_lattice[i]):
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 6
                                ] = 0
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 7
                                ] = 0
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 8
                                ] = 0

                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 0
                                ] = parameter["storagefurniture_layers_params"][5 + 3]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 2
                                ] = parameter["size"][2]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 1
                                ] = parameter["WHOLE_layer_sizes"][0]

                                cur_compartment_y_offset += (
                                    average_total_remained_height
                                )
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 3
                                ] = (
                                    -parameter["size"][0] / 2
                                    + parameter["left_right_inner_size"][0]
                                    + parameter["additional_layers_params"][
                                        1 + cur * 9 + 0
                                    ]
                                    / 2
                                )
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 4
                                ] = (
                                    -parameter["size"][1] / 2
                                    + parameter["base_size"][0]
                                    + parameter["additional_layers_params"][
                                        1 + cur * 9 + 1
                                    ]
                                    / 2
                                    + cur_compartment_y_offset
                                )
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 5
                                ] = 0
                                cur_compartment_y_offset += parameter[
                                    "WHOLE_layer_sizes"
                                ][0]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 3
                                ] += parameter["position"][0]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 4
                                ] += parameter["position"][1]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 5
                                ] += parameter["position"][2]
                                cur += 1
                            occupancy_of_each_compartment.append(
                                [
                                    0
                                    for _ in range(
                                        num_of_layers_in_each_vertical_lattice[i] + 1
                                    )
                                ]
                            )

                        elif i == num_of_vertical_on_second_layer:
                            if i == 2:
                                cur_total_offset_x = (
                                    parameter["storagefurniture_layers_params"][5 + 3]
                                    + parameter["storagefurniture_layers_params"][5 + 4]
                                )
                            elif i == 1:
                                cur_total_offset_x = parameter[
                                    "storagefurniture_layers_params"
                                ][5 + 3]
                            cur_compartment_y_offset = 0
                            cur_number_of_layers = (
                                num_of_layers_in_each_vertical_lattice[i]
                            )
                            total_remained_height = (
                                parameter["size"][1]
                                - parameter["base_size"][0]
                                - parameter["WHOLE_layer_offset"][0]
                            )

                            average_total_remained_height = (
                                total_remained_height
                                - (cur_number_of_layers + 1)
                                * parameter["WHOLE_layer_sizes"][0]
                            ) / (cur_number_of_layers + 1)
                            for j in range(num_of_layers_in_each_vertical_lattice[i]):
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 6
                                ] = 0
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 7
                                ] = 0
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 8
                                ] = 0

                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 0
                                ] = (
                                    parameter["size"][0]
                                    - 2 * parameter["left_right_inner_size"][0]
                                    - cur_total_offset_x
                                )

                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 2
                                ] = parameter["size"][2]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 1
                                ] = parameter["WHOLE_layer_sizes"][0]

                                cur_compartment_y_offset += (
                                    average_total_remained_height
                                )
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 3
                                ] = (
                                    -parameter["size"][0] / 2
                                    + parameter["left_right_inner_size"][0]
                                    + parameter["additional_layers_params"][
                                        1 + cur * 9 + 0
                                    ]
                                    / 2
                                    + cur_total_offset_x
                                )
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 4
                                ] = (
                                    -parameter["size"][1] / 2
                                    + parameter["base_size"][0]
                                    + parameter["additional_layers_params"][
                                        1 + cur * 9 + 1
                                    ]
                                    / 2
                                    + cur_compartment_y_offset
                                )
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 5
                                ] = 0
                                cur_compartment_y_offset += parameter[
                                    "WHOLE_layer_sizes"
                                ][0]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 3
                                ] += parameter["position"][0]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 4
                                ] += parameter["position"][1]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 5
                                ] += parameter["position"][2]
                                cur += 1
                            occupancy_of_each_compartment.append(
                                [
                                    0
                                    for _ in range(
                                        num_of_layers_in_each_vertical_lattice[i] + 1
                                    )
                                ]
                            )

                        elif i == 1:
                            cur_compartment_y_offset = 0
                            cur_number_of_layers = (
                                num_of_layers_in_each_vertical_lattice[i]
                            )
                            total_remained_height = (
                                parameter["size"][1]
                                - parameter["base_size"][0]
                                - parameter["WHOLE_layer_offset"][0]
                            )
                            average_total_remained_height = (
                                total_remained_height
                                - (cur_number_of_layers + 1)
                                * parameter["WHOLE_layer_sizes"][0]
                            ) / (cur_number_of_layers + 1)
                            for j in range(num_of_layers_in_each_vertical_lattice[i]):
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 6
                                ] = 0
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 7
                                ] = 0
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 8
                                ] = 0

                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 0
                                ] = parameter["storagefurniture_layers_params"][5 + 4]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 2
                                ] = parameter["size"][2]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 1
                                ] = parameter["WHOLE_layer_sizes"][0]

                                cur_compartment_y_offset += (
                                    average_total_remained_height
                                )
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 3
                                ] = (
                                    -parameter["size"][0] / 2
                                    + parameter["left_right_inner_size"][0]
                                    + parameter["additional_layers_params"][
                                        1 + cur * 9 + 0
                                    ]
                                    / 2
                                    + parameter["storagefurniture_layers_params"][5 + 3]
                                )
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 4
                                ] = (
                                    -parameter["size"][1] / 2
                                    + parameter["base_size"][0]
                                    + parameter["additional_layers_params"][
                                        1 + cur * 9 + 1
                                    ]
                                    / 2
                                    + cur_compartment_y_offset
                                )
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 5
                                ] = 0
                                cur_compartment_y_offset += parameter[
                                    "WHOLE_layer_sizes"
                                ][0]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 3
                                ] += parameter["position"][0]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 4
                                ] += parameter["position"][1]
                                parameter["additional_layers_params"][
                                    1 + cur * 9 + 5
                                ] += parameter["position"][2]
                                cur += 1
                            occupancy_of_each_compartment.append(
                                [
                                    0
                                    for _ in range(
                                        num_of_layers_in_each_vertical_lattice[i] + 1
                                    )
                                ]
                            )

            elif storagefurniture_type == "wardrobe":
                parameter["has_lid"][0] = 1
                parameter["WHOLE_number_of_layer"][0] = 1
                parameter["WHOLE_number_of_layer"] = np.array(
                    [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                )

                parameter["size"][1] = (
                    parameter["size"][0] * 1.7 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = (
                    parameter["size"][0] * 0.47 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["lid_size"][0] = parameter["size"][0]
                parameter["lid_size"][2] = parameter["size"][2]
                num_of_levels = int(parameter["WHOLE_number_of_layer"][0])

                parameter["WHOLE_layer_offset"][0] = (
                    (parameter["size"][1] - parameter["base_size"][0])
                    / 2
                    * randRange(1, 0.8, 1.2)[0]
                )

                parameter["WHOLE_layer_sizes"][0] *= randRange(1, 0.8, 1.2)[0]

                for i in range(num_of_levels + 1):
                    parameter["storagefurniture_layers_params"][i * 5] = 1
                    num_of_vertical = int(
                        parameter["storagefurniture_layers_params"][i * 5]
                    )
                    if i == 0:
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                    else:
                        parameter["storagefurniture_layers_params"][i * 5 + 1] = (
                            parameter["storagefurniture_layers_params"][1]
                        )
                    parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                        parameter["size"][2] - parameter["back_size"][0]
                    )
                    average_vertical_interval = (
                        parameter["size"][0] - 2 * parameter["left_right_inner_size"][0]
                    ) / (num_of_vertical + 1)
                    parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                        average_vertical_interval
                    )
                parameter["additional_layers_params"][0] = 0

            elif storagefurniture_type == "panel_Cabinet":
                parameter["has_lid"][0] = 1
                parameter["WHOLE_number_of_layer"][0] = 1
                parameter["WHOLE_number_of_layer"] = np.array(
                    [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                )
                parameter["size"][1] = parameter["size"][0] * randRange(1, 0.8, 1.2)[0]
                parameter["size"][0] = (
                    parameter["size"][1] * 1.5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = (
                    parameter["size"][1] * 0.47 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["lid_size"][0] = parameter["size"][0]
                parameter["lid_size"][2] = parameter["size"][2]
                num_of_levels = int(parameter["WHOLE_number_of_layer"][0])

                parameter["WHOLE_layer_offset"][0] = (
                    (parameter["size"][1] - parameter["base_size"][0])
                    / 2
                    * randRange(1, 0.8, 1.2)[0]
                )

                parameter["WHOLE_layer_sizes"][0] *= randRange(1, 0.8, 1.2)[0]

                for i in range(num_of_levels + 1):
                    if i == 0:
                        parameter["storagefurniture_layers_params"][i * 5] = (
                            np.random.randint(0, 4)
                        )
                        num_of_vertical = int(
                            parameter["storagefurniture_layers_params"][i * 5]
                        )
                        interval_of_vertical_compartment = (
                            parameter["size"][0]
                            - parameter["left_right_inner_size"][0] * 2
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2] - parameter["back_size"][0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            interval_of_vertical_compartment
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            interval_of_vertical_compartment
                        )
                    else:
                        parameter["storagefurniture_layers_params"][i * 5] = (
                            np.random.randint(0, 4)
                        )
                        num_of_vertical = int(
                            parameter["storagefurniture_layers_params"][i * 5]
                        )
                        interval_of_vertical_compartment = (
                            parameter["size"][0]
                            - parameter["left_right_inner_size"][0] * 2
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2] - parameter["back_size"][0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            interval_of_vertical_compartment
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            interval_of_vertical_compartment
                        )
                parameter["additional_layers_params"][0] = 0

            elif storagefurniture_type == "multi_drawer_Cabinet":
                parameter["has_lid"][0] = 1
                parameter["lid_size"][0] = parameter["size"][0]
                parameter["lid_size"][2] = parameter["size"][2]
                parameter["WHOLE_number_of_layer"][0] = np.random.randint(2, 5)
                parameter["WHOLE_number_of_layer"] = np.array(
                    [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                )
                num_of_levels = int(parameter["WHOLE_number_of_layer"][0]) + 1

                parameter["size"][1] = np.maximum(
                    parameter["size"][1],
                    parameter["size"][0]
                    * 0.38
                    * num_of_levels
                    * randRange(1, 0.8, 1.2)[0],
                )
                if random.random() < 0.2:
                    parameter["WHOLE_number_of_layer"][0] = np.random.randint(6, 11)
                    parameter["WHOLE_number_of_layer"] = np.array(
                        [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                    )
                    num_of_levels = int(parameter["WHOLE_number_of_layer"][0]) + 1
                    parameter["size"][1] = np.maximum(
                        parameter["size"][1],
                        parameter["size"][0]
                        * 0.25
                        * num_of_levels
                        * randRange(1, 0.8, 1.2)[0],
                    )

                parameter["WHOLE_layer_sizes"][0] *= randRange(1, 0.8, 1.2)[0]
                average_level_interval = (
                    parameter["size"][1]
                    - parameter["lid_offset"][1]
                    - parameter["base_size"][0]
                ) / num_of_levels
                parameter["WHOLE_layer_offset"][0] = average_level_interval
                for i in range(num_of_levels - 1):
                    parameter["WHOLE_interval_between_layers"][i] = (
                        average_level_interval
                    )

                for i in range(num_of_levels + 1):
                    parameter["storagefurniture_layers_params"][i * 5] = 0
                parameter["additional_layers_params"][0] = 0

            elif storagefurniture_type == "Multi_layer_wardrobe":
                parameter["has_lid"][0] = 1
                num_of_drawer_level = np.random.randint(0, 3)
                parameter["WHOLE_number_of_layer"][0] = 2 + num_of_drawer_level
                parameter["WHOLE_number_of_layer"] = np.array(
                    [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                )

                parameter["size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][1] = (
                    parameter["size"][0] * 3 / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["lid_size"][0] = parameter["size"][0]
                parameter["lid_size"][2] = parameter["size"][2]

                part_1_height = parameter["size"][1] / 4 * 1 * randRange(1, 0.8, 1.2)[0]
                part_2_height = parameter["size"][1] / 4 * 2 * randRange(1, 0.8, 1.2)[0]
                part_3_height = parameter["size"][1] - part_1_height - part_2_height
                part_3_inner_height = part_3_height / (num_of_drawer_level + 1)

                num_of_levels = int(parameter["WHOLE_number_of_layer"][0])

                parameter["WHOLE_layer_offset"][0] = part_1_height

                parameter["WHOLE_interval_between_layers"][0] = part_2_height
                for i in range(num_of_drawer_level):
                    parameter["WHOLE_interval_between_layers"][1 + i] = (
                        part_3_inner_height * (i + 1) + part_2_height
                    ) / (i + 2)

                parameter["WHOLE_layer_sizes"][0] *= randRange(1, 0.8, 1.2)[0]

                num_of_center_division = 0
                for i in range(num_of_levels + 1):
                    if i == 0:
                        parameter["storagefurniture_layers_params"][i * 5] = (
                            np.random.randint(0, 4)
                        )
                        num_of_vertical = int(
                            parameter["storagefurniture_layers_params"][i * 5]
                        )
                        interval_of_vertical_compartment = (
                            parameter["size"][0]
                            - parameter["left_right_inner_size"][0] * 2
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            interval_of_vertical_compartment
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            interval_of_vertical_compartment
                        )
                    elif i == 1:
                        parameter["storagefurniture_layers_params"][i * 5] = (
                            np.random.randint(1, 4)
                        )
                        num_of_vertical = int(
                            parameter["storagefurniture_layers_params"][i * 5]
                        )
                        num_of_center_division = num_of_vertical + 1
                        interval_of_vertical_compartment = (
                            parameter["size"][0]
                            - parameter["left_right_inner_size"][0] * 2
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            interval_of_vertical_compartment
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            interval_of_vertical_compartment
                        )
                    else:
                        parameter["storagefurniture_layers_params"][i * 5] = (
                            np.random.randint(0, 3)
                        )
                        num_of_vertical = int(
                            parameter["storagefurniture_layers_params"][i * 5]
                        )
                        interval_of_vertical_compartment = (
                            parameter["size"][0]
                            - parameter["left_right_inner_size"][0] * 2
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            interval_of_vertical_compartment
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            interval_of_vertical_compartment
                        )

                num_of_additional_layers = np.random.randint(
                    0, 4, size=num_of_center_division
                )
                total_number_of_layers = 0
                for i in range(num_of_center_division):
                    total_number_of_layers += num_of_additional_layers[i]
                top_offset_y_of_middle_level = (
                    part_1_height + parameter["WHOLE_layer_sizes"][0] / 2
                )
                cur = 0
                parameter["additional_layers_params"][0] = total_number_of_layers * 1.0
                for i in range(num_of_center_division):
                    number = num_of_additional_layers[i]
                    y_interval_of_additional_layer = part_2_height / (number + 1)
                    for j in range(number):
                        if i == 0:
                            x_offset = 0
                            x_width = parameter["storagefurniture_layers_params"][
                                1 * 5 + 3
                            ]
                            z_width = parameter["size"][2] - parameter["back_size"][0]
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                -parameter["size"][0] / 2
                                + parameter["left_right_inner_size"][0]
                                + parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                                + x_offset
                            )
                        elif i == num_of_center_division - 1:
                            x_width = (
                                parameter["size"][0]
                                - parameter["left_right_inner_size"][0] * 2
                                - parameter["storagefurniture_layers_params"][1 * 5 + 3]
                                - (num_of_center_division - 2)
                                * parameter["storagefurniture_layers_params"][1 * 5 + 4]
                            )
                            z_width = parameter["size"][2] - parameter["back_size"][0]
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                parameter["size"][0] / 2
                                - parameter["left_right_inner_size"][0]
                                - parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                            )
                        else:
                            x_offset = (
                                parameter["storagefurniture_layers_params"][1 * 5 + 3]
                                + (i - 1)
                                * parameter["storagefurniture_layers_params"][1 * 5 + 4]
                            )
                            x_width = parameter["storagefurniture_layers_params"][
                                1 * 5 + 4
                            ]
                            z_width = parameter["size"][2] - parameter["back_size"][0]
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                -parameter["size"][0] / 2
                                + parameter["left_right_inner_size"][0]
                                + parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                                + x_offset
                            )
                        parameter["additional_layers_params"][1 + cur * 9 + 4] = (
                            parameter["size"][1] / 2
                            - top_offset_y_of_middle_level
                            - y_interval_of_additional_layer * (j + 1)
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 3] += (
                            parameter["position"][0]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 4] += (
                            parameter["position"][1]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 5] += (
                            parameter["position"][2]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_layers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_layers_params"][1 + cur * 9 + 8] = 0
                        cur += 1

            elif storagefurniture_type == "bookcase":
                parameter["has_lid"][0] = 1
                parameter["WHOLE_number_of_layer"][0] = 2
                parameter["WHOLE_number_of_layer"] = np.array(
                    [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                )

                parameter["size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][1] = (
                    parameter["size"][0] * 3 / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = (
                    parameter["size"][0] * 2 / 5 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["lid_size"][0] = parameter["size"][0]
                parameter["lid_size"][2] = parameter["size"][2]

                parameter["WHOLE_layer_sizes"][0] *= randRange(1, 0.8, 1.2)[0]

                height_1 = parameter["size"][1] * 3 / 7 * randRange(1, 0.8, 1.2)[0]
                height_2 = parameter["size"][1] / 7 * randRange(1, 0.8, 1.2)[0]
                height_3 = (
                    parameter["size"][1]
                    - height_1
                    - height_2
                    - parameter["base_size"][0]
                )
                parameter["WHOLE_layer_offset"][0] = height_1
                parameter["WHOLE_interval_between_layers"][0] = height_2

                num_of_levels = 3

                for i in range(num_of_levels):
                    if i == 0:
                        parameter["storagefurniture_layers_params"][i * 5] = int(1)
                        num_of_vertical = int(
                            parameter["storagefurniture_layers_params"][i * 5]
                        )
                        interval_of_vertical_compartment = (
                            parameter["size"][0]
                            - parameter["left_right_inner_size"][0] * 2
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2]
                            - parameter["back_size"][0]
                            - parameter["storagefurniture_layers_params"][i * 5 + 1]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            interval_of_vertical_compartment
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            interval_of_vertical_compartment
                        )
                        num_of_transverse_1 = num_of_vertical + 1
                    elif i == 1:
                        parameter["storagefurniture_layers_params"][i * 5] = int(
                            np.random.randint(0, 4)
                        )
                        num_of_vertical = int(
                            parameter["storagefurniture_layers_params"][i * 5]
                        )
                        interval_of_vertical_compartment = (
                            parameter["size"][0]
                            - parameter["left_right_inner_size"][0] * 2
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2]
                            - parameter["back_size"][0]
                            - parameter["storagefurniture_layers_params"][i * 5 + 1]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            interval_of_vertical_compartment
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            interval_of_vertical_compartment
                        )
                    else:
                        parameter["storagefurniture_layers_params"][i * 5] = (
                            int(1) if random.random() < 0.5 else int(3)
                        )
                        num_of_vertical = int(
                            parameter["storagefurniture_layers_params"][i * 5]
                        )
                        interval_of_vertical_compartment = (
                            parameter["size"][0]
                            - parameter["left_right_inner_size"][0] * 2
                        ) / (num_of_vertical + 1)
                        parameter["storagefurniture_layers_params"][i * 5 + 1] *= (
                            randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 2] = (
                            parameter["size"][2]
                            - parameter["back_size"][0]
                            - parameter["storagefurniture_layers_params"][i * 5 + 1]
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 3] = (
                            interval_of_vertical_compartment
                        )
                        parameter["storagefurniture_layers_params"][i * 5 + 4] = (
                            interval_of_vertical_compartment
                        )
                        num_of_transverse_2 = num_of_vertical + 1

                num_of_additional_layers_1 = np.random.randint(
                    0, 4, size=num_of_transverse_1
                )
                num_of_additional_layers_2 = np.random.randint(
                    0, 4, size=num_of_transverse_2
                )
                total_number_of_layers = 0
                for i in range(num_of_transverse_1):
                    total_number_of_layers += num_of_additional_layers_1[i]
                for i in range(num_of_transverse_2):
                    total_number_of_layers += num_of_additional_layers_2[i]

                center_top_offset_y_1 = np.maximum(0, -parameter["lid_offset"][1])
                center_top_offset_y_2 = (
                    center_top_offset_y_1
                    + parameter["WHOLE_layer_offset"][0]
                    + parameter["WHOLE_interval_between_layers"][0]
                )
                cur = 0
                parameter["additional_layers_params"][0] = total_number_of_layers * 1.0
                for i in range(num_of_transverse_1):
                    number = num_of_additional_layers_1[i]
                    interval_of_layers_y = height_1 / (number + 1)
                    for j in range(number):
                        if i == 0:
                            x_offset = 0
                            x_width = parameter["storagefurniture_layers_params"][
                                0 * 5 + 3
                            ]
                            z_width = (
                                parameter["size"][2]
                                - parameter["back_size"][0]
                                - parameter["storagefurniture_layers_params"][0 * 5 + 1]
                            )
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                -parameter["size"][0] / 2
                                + parameter["left_right_inner_size"][0]
                                + parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                                + x_offset
                            )
                        elif i == num_of_transverse_1 - 1:
                            x_width = (
                                parameter["size"][0]
                                - parameter["left_right_inner_size"][0] * 2
                                - parameter["storagefurniture_layers_params"][0 * 5 + 3]
                                - (num_of_transverse_1 - 2)
                                * parameter["storagefurniture_layers_params"][0 * 5 + 4]
                            )
                            z_width = (
                                parameter["size"][2]
                                - parameter["back_size"][0]
                                - parameter["storagefurniture_layers_params"][0 * 5 + 1]
                            )
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                parameter["size"][0] / 2
                                - parameter["left_right_inner_size"][0]
                                - parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                            )

                        parameter["additional_layers_params"][1 + cur * 9 + 4] = (
                            parameter["size"][1] / 2
                            - center_top_offset_y_1
                            - interval_of_layers_y * (j + 1)
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 3] += (
                            parameter["position"][0]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 4] += (
                            parameter["position"][1]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 5] += (
                            parameter["position"][2]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_layers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_layers_params"][1 + cur * 9 + 8] = 0
                        cur += 1

                for i in range(num_of_transverse_2):
                    number = num_of_additional_layers_2[i]
                    interval_of_layers_y = height_3 / (number + 1)
                    for j in range(number):
                        if i == 0:
                            x_offset = 0
                            x_width = parameter["storagefurniture_layers_params"][
                                2 * 5 + 3
                            ]
                            z_width = (
                                parameter["size"][2]
                                - parameter["back_size"][0]
                                - parameter["storagefurniture_layers_params"][0 * 5 + 1]
                            )
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                -parameter["size"][0] / 2
                                + parameter["left_right_inner_size"][0]
                                + parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                                + x_offset
                            )
                        elif i == num_of_transverse_2 - 1:
                            x_width = (
                                parameter["size"][0]
                                - parameter["left_right_inner_size"][0] * 2
                                - parameter["storagefurniture_layers_params"][2 * 5 + 3]
                                - (num_of_transverse_2 - 2)
                                * parameter["storagefurniture_layers_params"][2 * 5 + 4]
                            )
                            z_width = (
                                parameter["size"][2]
                                - parameter["back_size"][0]
                                - parameter["storagefurniture_layers_params"][0 * 5 + 1]
                            )
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                parameter["size"][0] / 2
                                - parameter["left_right_inner_size"][0]
                                - parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                            )
                        else:
                            x_offset = (
                                parameter["storagefurniture_layers_params"][2 * 5 + 3]
                                + (i - 1)
                                * parameter["storagefurniture_layers_params"][2 * 5 + 4]
                            )
                            x_width = parameter["storagefurniture_layers_params"][
                                2 * 5 + 4
                            ]
                            z_width = (
                                parameter["size"][2]
                                - parameter["back_size"][0]
                                - parameter["storagefurniture_layers_params"][0 * 5 + 1]
                            )
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                -parameter["size"][0] / 2
                                + parameter["left_right_inner_size"][0]
                                + parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                                + x_offset
                            )

                        parameter["additional_layers_params"][1 + cur * 9 + 4] = (
                            parameter["size"][1] / 2
                            - center_top_offset_y_2
                            - interval_of_layers_y * (j + 1)
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 3] += (
                            parameter["position"][0]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 4] += (
                            parameter["position"][1]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 5] += (
                            parameter["position"][2]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_layers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_layers_params"][1 + cur * 9 + 8] = 0
                        cur += 1

            elif storagefurniture_type == "Drinking_machine":
                parameter["has_lid"][0] = 1
                parameter["WHOLE_number_of_layer"][0] = 2
                parameter["WHOLE_number_of_layer"] = np.array(
                    [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                )
                num_of_levels = int(parameter["WHOLE_number_of_layer"][0]) + 1

                parameter["size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][1] = (
                    5 * parameter["size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = parameter["size"][0] * randRange(1, 0.8, 1.2)[0]
                parameter["lid_size"][0] = parameter["size"][0]
                parameter["lid_size"][2] = parameter["size"][2]

                parameter["WHOLE_layer_sizes"][0] *= randRange(1, 0.8, 1.2)[0]

                height_1 = parameter["size"][1] / 5 * 2 * randRange(1, 0.8, 1.2)[0]
                height_2 = parameter["size"][1] / 5 * 1 * randRange(1, 0.8, 1.2)[0]

                parameter["WHOLE_layer_offset"][0] = height_1
                parameter["WHOLE_interval_between_layers"][0] = height_2

                for i in range(num_of_levels + 1):
                    parameter["storagefurniture_layers_params"][i * 5] = int(0)
                parameter["additional_layers_params"][0] = 0

            elif storagefurniture_type == "Cabinet_vertical":
                parameter["has_lid"][0] = 1
                parameter["WHOLE_number_of_layer"][0] = 0
                parameter["WHOLE_number_of_layer"] = np.array(
                    [int(layer) for layer in parameter["WHOLE_number_of_layer"]]
                )
                num_of_vertical_partition = np.random.randint(2, 5)
                parameter["size"][0] /= 2
                parameter["size"][0] *= (
                    num_of_vertical_partition * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][1] = (
                    4
                    * parameter["size"][0]
                    / (2.5 * num_of_vertical_partition)
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = (
                    2
                    * parameter["size"][0]
                    / (2.5 * num_of_vertical_partition)
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["lid_size"][0] = parameter["size"][0]
                parameter["lid_size"][2] = parameter["size"][2]

                parameter["WHOLE_layer_sizes"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["storagefurniture_layers_params"][0 * 5] = (
                    num_of_vertical_partition - 1
                )
                parameter["storagefurniture_layers_params"][0 * 5 + 1] *= randRange(
                    1, 0.8, 1.2
                )[0]
                parameter["storagefurniture_layers_params"][0 * 5 + 2] = parameter[
                    "size"
                ][2]
                interval_of_vertical_compartment = (
                    parameter["size"][0] - parameter["left_right_inner_size"][0] * 2
                ) / num_of_vertical_partition
                parameter["storagefurniture_layers_params"][0 * 5 + 3] = (
                    interval_of_vertical_compartment
                )
                parameter["storagefurniture_layers_params"][0 * 5 + 4] = (
                    interval_of_vertical_compartment
                )

                num_of_additional_layers = np.random.randint(
                    0, 4, size=num_of_vertical_partition
                )
                total_number_of_layers = 0
                for i in range(num_of_vertical_partition):
                    total_number_of_layers += num_of_additional_layers[i]
                cur = 0
                parameter["additional_layers_params"][0] = total_number_of_layers * 1.0
                for i in range(num_of_vertical_partition):
                    number = num_of_additional_layers[i]
                    y_interval_of_additional_layer = (
                        parameter["size"][1]
                        - parameter["base_size"][0]
                        + parameter["lid_offset"][1]
                    ) / (number + 1)
                    for j in range(number):
                        if i == 0:
                            x_offset = 0
                            x_width = parameter["storagefurniture_layers_params"][
                                0 * 5 + 3
                            ]
                            z_width = parameter["size"][2] - parameter["back_size"][0]
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                -parameter["size"][0] / 2
                                + parameter["left_right_inner_size"][0]
                                + parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                                + x_offset
                            )
                        elif i == num_of_vertical_partition - 1:
                            x_width = (
                                parameter["size"][0]
                                - parameter["left_right_inner_size"][0] * 2
                                - parameter["storagefurniture_layers_params"][0 * 5 + 3]
                                - (num_of_vertical_partition - 2)
                                * parameter["storagefurniture_layers_params"][0 * 5 + 4]
                            )
                            z_width = parameter["size"][2] - parameter["back_size"][0]
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                parameter["size"][0] / 2
                                - parameter["left_right_inner_size"][0]
                                - parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                            )
                        else:
                            x_offset = (
                                parameter["storagefurniture_layers_params"][0 * 5 + 3]
                                + (i - 1)
                                * parameter["storagefurniture_layers_params"][0 * 5 + 4]
                            )
                            x_width = parameter["storagefurniture_layers_params"][
                                0 * 5 + 4
                            ]
                            z_width = parameter["size"][2] - parameter["back_size"][0]
                            y_height = parameter["WHOLE_layer_sizes"][0]
                            parameter["additional_layers_params"][1 + cur * 9 + 0] = (
                                x_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 2] = (
                                z_width
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 1] = (
                                y_height
                            )
                            parameter["additional_layers_params"][1 + cur * 9 + 5] = 0
                            parameter["additional_layers_params"][1 + cur * 9 + 3] = (
                                -parameter["size"][0] / 2
                                + parameter["left_right_inner_size"][0]
                                + parameter["additional_layers_params"][1 + cur * 9 + 0]
                                / 2
                                + x_offset
                            )
                        parameter["additional_layers_params"][1 + cur * 9 + 4] = (
                            parameter["size"][1] / 2
                            + parameter["lid_offset"][1]
                            - y_interval_of_additional_layer * (j + 1)
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 3] += (
                            parameter["position"][0]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 4] += (
                            parameter["position"][1]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 5] += (
                            parameter["position"][2]
                        )
                        parameter["additional_layers_params"][1 + cur * 9 + 6] = 0
                        parameter["additional_layers_params"][1 + cur * 9 + 7] = 0
                        parameter["additional_layers_params"][1 + cur * 9 + 8] = 0
                        cur += 1

            concept["parameters"] = {
                k: v.tolist()
                if k
                not in [
                    "storagefurniture_layers_params",
                    "additional_layers_params",
                    "drawers_params",
                    "additional_legs_params",
                ]
                else v
                for k, v in parameter.items()
            }
            new_concepts.append(concept)

        elif template == "Enclosed_leg":
            body_parameter = concepts[0]["parameters"]
            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            leg_type = 1
            if leg_type == 0:
                parameter["size"][0] = (
                    body_parameter["size"][0] * randRange(1, 0.7, 1.0)[0]
                )
                parameter["size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][2] = (
                    body_parameter["size"][2] * randRange(1, 0.7, 1.0)[0]
                )
                parameter["inner_sizes"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["inner_sizes"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["position"][1] -= (
                    body_parameter["size"][1] / 2 + parameter["size"][1] / 2
                )
            elif leg_type == 1:
                parameter["size"][0] = (
                    body_parameter["size"][0] * randRange(1, 0.7, 1.0)[0]
                )
                parameter["size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][2] = (
                    body_parameter["size"][2] * randRange(1, 0.7, 1.0)[0]
                )
                parameter["inner_sizes"][0] = (
                    parameter["size"][0] / 7 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["inner_sizes"][1] = (
                    parameter["size"][2] / 7 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position"][1] -= (
                    body_parameter["size"][1] / 2 + parameter["size"][1] / 2
                )

            concept["parameters"] = {
                k: v.tolist()
                if k
                not in [
                    "storagefurniture_layers_params",
                    "additional_layers_params",
                    "drawers_params",
                    "additional_legs_params",
                ]
                else v
                for k, v in parameter.items()
            }
            new_concepts.append(concept)

        elif template == "Regular_door":
            body_parameter = concepts[0]["parameters"]

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            if storagefurniture_type == "Fridge":
                parameter["number_of_door"][0] = 1
                parameter["number_of_door"] = np.array(
                    [int(num) for num in parameter["number_of_door"]]
                )

                for i in range(parameter["number_of_door"][0]):
                    parameter["doors_params"][1] = body_parameter["size"][0] - (
                        body_parameter["left_right_inner_size"][0]
                        * 2
                        * randRange(1, 0, 1.0)[0]
                    )
                    parameter["doors_params"][2] = (
                        body_parameter["size"][1]
                        - body_parameter["base_size"][0] * randRange(1, 0.6, 1.0)[0]
                    )
                    parameter["doors_params"][3] *= randRange(1, 0.8, 1.2)[0]
                    parameter["doors_params"][9] = np.random.uniform(0, 180)
                    parameter["doors_params"][10] = 0
                    parameter["doors_params"][11] = 0
                    parameter["doors_params"][12] = body_parameter["size"][2] / 2

                    handle_type = np.random.randint(0, 2)
                    door_handle_move_dir = -1 if np.random.uniform(0, 2) == 0 else 1
                    if handle_type == 0:
                        parameter["doors_params"][0] = 0
                        parameter["doors_params"][4] *= randRange(1, 0.5, 0.8)[0]
                        parameter["doors_params"][5] *= randRange(1, 0.8, 1.2)[0]
                        parameter["doors_params"][6] *= randRange(1, 0.8, 1.2)[0]
                        parameter["doors_params"][9] = -parameter["doors_params"][9]
                        parameter["doors_params"][7] = (
                            parameter["doors_params"][1] / 2 * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][8] = (
                            parameter["doors_params"][2] / 8 * randRange(1, 0.8, 1.2)[0]
                        )
                    elif handle_type == 1:
                        parameter["doors_params"][0] = 1
                        parameter["doors_params"][4] *= randRange(1, 0.5, 0.8)[0]
                        parameter["doors_params"][5] *= randRange(1, 0.8, 1.2)[0]
                        parameter["doors_params"][6] *= randRange(1, 0.8, 1.2)[0]
                        parameter["doors_params"][7] = (
                            -parameter["doors_params"][1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][8] = (
                            -parameter["doors_params"][2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                    # elif handle_type == 2:
                    #     parameter['doors_params'][0] = 2
                    #     parameter['doors_params'][4] *= randRange(1, 0.5, 0.8)[0]
                    #     parameter['doors_params'][5] *= randRange(1, 0.8, 1.2)[0]
                    #     tmp = parameter['doors_params'][4]
                    #     parameter['doors_params'][4] = parameter['doors_params'][5]
                    #     parameter['doors_params'][5] = tmp
                    #     parameter['doors_params'][6] *= randRange(1, 0.8, 1.2)[0]
                    #     parameter['doors_params'][9] = np.random.uniform(0, 90)
                    #     parameter['doors_params'][7] = door_handle_move_dir * parameter['doors_params'][1] / 8 * randRange(1, 0.8, 1.2)[0]
                    #     parameter['doors_params'][8] = parameter['doors_params'][2] / 2 * randRange(1, 0.8, 0.9)[0]

            elif storagefurniture_type == "Cabinet_transverse":
                parameter["number_of_door"][0] = (
                    body_parameter["storagefurniture_layers_params"][5] + 1
                )
                parameter["number_of_door"] = np.array(
                    [int(num) for num in parameter["number_of_door"]]
                )
                num_of_vertical_on_second_layer = int(parameter["number_of_door"][0])
                num_of_compartment_this_level = [
                    len(row) for row in occupancy_of_each_compartment
                ]
                door_open_direction = [
                    -1 for _ in range(num_of_vertical_on_second_layer)
                ]
                door_num_record = 0
                cur = 0

                for i in range(num_of_vertical_on_second_layer):
                    height_remained = (
                        body_parameter["size"][1]
                        - body_parameter["base_size"][0]
                        - body_parameter["WHOLE_layer_offset"][0]
                    )
                    parameter["doors_params"][13 * cur + 3] = (
                        parameter["doors_params"][3] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 9] = 0
                    parameter["doors_params"][13 * cur + 4] *= randRange(1, 0.5, 0.6)[0]
                    parameter["doors_params"][13 * cur + 5] *= randRange(1, 0.8, 1.0)[0]
                    parameter["doors_params"][13 * cur + 6] = parameter["doors_params"][
                        13 * cur + 3
                    ]

                    cur_door_occupied_num = np.random.randint(
                        0, num_of_compartment_this_level[i] + 1
                    )
                    door_num_record += cur_door_occupied_num
                    if (
                        door_num_record == 0
                        and i == num_of_vertical_on_second_layer - 1
                    ):
                        cur_door_occupied_num = np.random.randint(
                            1, num_of_compartment_this_level[i] + 1
                        )
                    door_size_y = 0
                    average_total_remained_height = (
                        height_remained
                        - (num_of_compartment_this_level[i] - 1)
                        * body_parameter["WHOLE_layer_sizes"][0]
                    ) / (num_of_compartment_this_level[i])
                    if cur_door_occupied_num == num_of_compartment_this_level[i]:
                        door_size_y = height_remained
                        for j in range(num_of_compartment_this_level[i]):
                            occupancy_of_each_compartment[i][j] = 1
                    elif cur_door_occupied_num == 0:
                        parameter["number_of_door"][0] -= 1
                        continue
                    elif cur_door_occupied_num == num_of_compartment_this_level[i] - 1:
                        door_size_y = (
                            height_remained
                            - average_total_remained_height
                            - body_parameter["WHOLE_layer_sizes"][0] / 2
                        )
                        for j in range(num_of_compartment_this_level[i]):
                            occupancy_of_each_compartment[i][j] = 1
                        occupancy_of_each_compartment[i][0] = 0
                    elif cur_door_occupied_num == num_of_compartment_this_level[i] - 2:
                        door_size_y = (
                            height_remained
                            - average_total_remained_height * 2
                            - body_parameter["WHOLE_layer_sizes"][0]
                        )
                        for j in range(num_of_compartment_this_level[i]):
                            occupancy_of_each_compartment[i][j] = 1
                        occupancy_of_each_compartment[i][0] = 0
                        occupancy_of_each_compartment[i][1] = 0

                    if i == 0:
                        parameter["doors_params"][13 * cur + 2] = door_size_y
                        parameter["doors_params"][13 * cur + 1] = body_parameter[
                            "storagefurniture_layers_params"
                        ][5 + 3]
                        parameter["doors_params"][13 * cur + 4] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 5] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 6] = parameter[
                            "doors_params"
                        ][13 * cur + 3]
                        parameter["doors_params"][13 * cur + 10] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + parameter["doors_params"][13 * cur + 1] / 2
                        )
                        parameter["doors_params"][13 * cur + 11] = (
                            -body_parameter["size"][1] / 2
                            + body_parameter["base_size"][0]
                            + parameter["doors_params"][13 * cur + 2] / 2
                        )
                        parameter["doors_params"][13 * cur + 12] = (
                            body_parameter["size"][2] / 2
                        )
                        door_open_direction[i] = np.random.randint(0, 2)

                        if door_open_direction[i] == 0:
                            parameter["doors_params"][
                                13 * cur + 9
                            ] = -np.random.uniform(0, 30)
                            parameter["doors_params"][13 * cur + 0] = 0
                            parameter["doors_params"][13 * cur + 7] = (
                                parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        elif door_open_direction[i] == 1:
                            parameter["doors_params"][13 * cur + 9] = np.random.uniform(
                                0, 30
                            )
                            parameter["doors_params"][13 * cur + 0] = 1
                            parameter["doors_params"][13 * cur + 7] = (
                                -parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        # another_type = np.random.randint(0, 3)
                        # if another_type == 2:
                        #     parameter['doors_params'][13 * cur + 9] = np.random.uniform(0, 30)
                        #     parameter['doors_params'][13 * cur + 0] = 2
                        #     tmp = parameter['doors_params'][13 * cur + 4]
                        #     parameter['doors_params'][13 * cur + 4] = parameter['doors_params'][13 * cur + 5]
                        #     parameter['doors_params'][13 * cur + 5] = tmp
                        #     parameter['doors_params'][13 * cur + 7] = parameter['doors_params'][13 * cur + 1] / 8 * randRange(1,0.8,1.2)[0]
                        #     parameter['doors_params'][13 * cur + 8] = parameter['doors_params'][13 * cur + 2] / 2 * randRange(1, 0.8, 0.9)[0]
                        cur += 1

                    elif i == num_of_vertical_on_second_layer - 1:
                        if i == 2:
                            cur_total_offset_x = (
                                body_parameter["storagefurniture_layers_params"][5 + 3]
                                + body_parameter["storagefurniture_layers_params"][
                                    5 + 4
                                ]
                                + 2
                                * body_parameter["storagefurniture_layers_params"][
                                    5 + 1
                                ]
                            )
                        elif i == 1:
                            cur_total_offset_x = (
                                body_parameter["storagefurniture_layers_params"][5 + 3]
                                + body_parameter["storagefurniture_layers_params"][
                                    5 + 1
                                ]
                            )

                        parameter["doors_params"][13 * cur + 2] = door_size_y
                        parameter["doors_params"][13 * cur + 1] = (
                            body_parameter["size"][0]
                            - 2 * body_parameter["left_right_inner_size"][0]
                            - cur_total_offset_x
                        )

                        parameter["doors_params"][13 * cur + 4] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 5] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 6] = parameter[
                            "doors_params"
                        ][13 * cur + 3]

                        parameter["doors_params"][13 * cur + 10] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + parameter["doors_params"][13 * cur + 1] / 2
                            + cur_total_offset_x
                        )
                        parameter["doors_params"][13 * cur + 11] = (
                            -body_parameter["size"][1] / 2
                            + body_parameter["base_size"][0]
                            + parameter["doors_params"][13 * cur + 2] / 2
                        )
                        parameter["doors_params"][13 * cur + 12] = (
                            body_parameter["size"][2] / 2
                        )

                        door_open_direction[i] = np.random.randint(0, 2)

                        if door_open_direction[i] == 0:
                            parameter["doors_params"][
                                13 * cur + 9
                            ] = -np.random.uniform(0, 30)
                            parameter["doors_params"][13 * cur + 0] = 0
                            parameter["doors_params"][13 * cur + 7] = (
                                parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        elif door_open_direction[i] == 1:
                            parameter["doors_params"][13 * cur + 9] = np.random.uniform(
                                0, 30
                            )
                            parameter["doors_params"][13 * cur + 0] = 1
                            parameter["doors_params"][13 * cur + 7] = (
                                -parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )

                        # another_type = np.random.randint(0, 3)
                        # if another_type == 2:
                        #     parameter['doors_params'][13 * cur + 9] = np.random.uniform(0, 30)
                        #     parameter['doors_params'][13 * cur + 0] = 2
                        #     tmp = parameter['doors_params'][13 * cur + 4]
                        #     parameter['doors_params'][13 * cur + 4] = parameter['doors_params'][13 * cur + 5]
                        #     parameter['doors_params'][13 * cur + 5] = tmp
                        #     parameter['doors_params'][13 * cur + 7] = parameter['doors_params'][13 * cur + 1] / 8 * randRange(1, 0.8, 1.2)[0]
                        #     parameter['doors_params'][13 * cur + 8] = parameter['doors_params'][13 * cur + 2] / 2 * randRange(1, 0.8, 0.9)[0]
                        cur += 1

                    elif i == 1:
                        parameter["doors_params"][13 * cur + 2] = door_size_y
                        cur_total_offset_x = (
                            body_parameter["storagefurniture_layers_params"][5 + 3]
                            + body_parameter["storagefurniture_layers_params"][5 + 1]
                        )
                        parameter["doors_params"][13 * cur + 1] = body_parameter[
                            "storagefurniture_layers_params"
                        ][5 + 4]

                        parameter["doors_params"][13 * cur + 10] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + parameter["doors_params"][13 * cur + 1] / 2
                            + cur_total_offset_x
                        )
                        parameter["doors_params"][13 * cur + 11] = (
                            -body_parameter["size"][1] / 2
                            + body_parameter["base_size"][0]
                            + parameter["doors_params"][13 * cur + 2] / 2
                        )
                        parameter["doors_params"][13 * cur + 12] = (
                            body_parameter["size"][2] / 2
                        )

                        parameter["doors_params"][13 * cur + 4] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 5] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 6] = parameter[
                            "doors_params"
                        ][13 * cur + 3]

                        door_open_direction[i] = np.random.randint(0, 2)

                        if door_open_direction[i] == 0:
                            parameter["doors_params"][
                                13 * cur + 9
                            ] = -np.random.uniform(0, 30)
                            parameter["doors_params"][13 * cur + 0] = 0
                            parameter["doors_params"][13 * cur + 7] = (
                                parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )

                        elif door_open_direction[i] == 1:
                            parameter["doors_params"][13 * cur + 9] = np.random.uniform(
                                0, 30
                            )
                            parameter["doors_params"][13 * cur + 0] = 1
                            parameter["doors_params"][13 * cur + 7] = (
                                -parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )

                        # another_type = np.random.randint(0, 3)
                        # if another_type == 2:
                        #     parameter['doors_params'][13 * cur + 9] = np.random.uniform(0, 30)
                        #     parameter['doors_params'][13 * cur + 0] = 2
                        #     tmp = parameter['doors_params'][13 * cur + 4]
                        #     parameter['doors_params'][13 * cur + 4] = parameter['doors_params'][5]
                        #     parameter['doors_params'][13 * cur + 5] = tmp
                        #     parameter['doors_params'][13 * cur + 7] = parameter['doors_params'][13 * cur + 1] / 8 * randRange(1, 0.8, 1.2)[0]
                        #     parameter['doors_params'][13 * cur + 8] = parameter['doors_params'][13 * cur + 2] / 2 * randRange(1, 0.8, 0.9)[0]
                        cur += 1

            elif storagefurniture_type == "wardrobe":
                wardrobe_type = np.random.randint(0, 2)
                if wardrobe_type == 0:
                    door_size_y = (
                        body_parameter["size"][1] - body_parameter["base_size"][0]
                    )
                    door_width_1 = body_parameter["storagefurniture_layers_params"][3]
                    door_width_2 = (
                        body_parameter["size"][0]
                        - body_parameter["left_right_inner_size"][0] * 2
                        - door_width_1
                        - body_parameter["storagefurniture_layers_params"][1]
                    )
                    parameter["number_of_door"] = np.array(
                        [int(2) for _ in parameter["number_of_door"]]
                    )
                    for i in range(2):
                        parameter["doors_params"][13 * i + 0] = int(0)
                        parameter["doors_params"][13 * i + 1] = (
                            door_width_1 if i == 0 else door_width_2
                        )
                        parameter["doors_params"][13 * i + 2] = door_size_y
                        parameter["doors_params"][13 * i + 3] = (
                            parameter["doors_params"][3] * randRange(1, 0.8, 1.2)[0]
                        )

                        parameter["doors_params"][13 * i + 4] = (
                            parameter["doors_params"][13 * i + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * i + 5] = (
                            parameter["doors_params"][13 * i + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * i + 6] = parameter[
                            "doors_params"
                        ][13 * i + 3]
                        if i == 1:
                            parameter["doors_params"][13 * i + 4] = parameter[
                                "doors_params"
                            ][4]
                            parameter["doors_params"][13 * i + 5] = parameter[
                                "doors_params"
                            ][5]
                            parameter["doors_params"][13 * i + 6] = parameter[
                                "doors_params"
                            ][6]

                        if i == 0:
                            parameter["doors_params"][13 * i + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][13 * i + 1] / 2
                            )
                            parameter["doors_params"][13 * i + 11] = (
                                -body_parameter["size"][1] / 2
                                + body_parameter["base_size"][0]
                                + parameter["doors_params"][13 * i + 2] / 2
                            )
                            parameter["doors_params"][13 * i + 12] = (
                                body_parameter["size"][2] / 2
                            )

                        elif i == 1:
                            cur_total_offset_x = (
                                body_parameter["storagefurniture_layers_params"][3]
                                + body_parameter["storagefurniture_layers_params"][1]
                            )
                            parameter["doors_params"][13 * i + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][13 * i + 1] / 2
                                + cur_total_offset_x
                            )
                            parameter["doors_params"][13 * i + 11] = (
                                -body_parameter["size"][1] / 2
                                + body_parameter["base_size"][0]
                                + parameter["doors_params"][13 * i + 2] / 2
                            )
                            parameter["doors_params"][13 * i + 12] = (
                                body_parameter["size"][2] / 2
                            )

                        if i == 0:
                            parameter["doors_params"][13 * i + 9] = -np.random.uniform(
                                0, 180
                            )
                            parameter["doors_params"][13 * i + 0] = 0
                            parameter["doors_params"][13 * i + 7] = (
                                parameter["doors_params"][13 * i + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * i + 8] = (
                                parameter["doors_params"][13 * i + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )

                        elif i == 1:
                            parameter["doors_params"][13 * i + 9] = np.random.uniform(
                                0, 180
                            )
                            parameter["doors_params"][13 * i + 0] = 1
                            parameter["doors_params"][13 * i + 7] = (
                                -parameter["doors_params"][13 * i + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * i + 8] = (
                                parameter["doors_params"][13 * i + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )

                elif wardrobe_type == 1:
                    door_height_1 = body_parameter["WHOLE_layer_offset"][0]
                    door_height_2 = (
                        body_parameter["size"][1]
                        - body_parameter["WHOLE_layer_offset"][0]
                        - body_parameter["base_size"][0]
                    )
                    door_width_1 = body_parameter["storagefurniture_layers_params"][3]
                    door_width_2 = (
                        body_parameter["size"][0]
                        - body_parameter["left_right_inner_size"][0] * 2
                        - door_width_1
                        - body_parameter["storagefurniture_layers_params"][1]
                    )
                    parameter["number_of_door"] = np.array(
                        [int(4) for _ in parameter["number_of_door"]]
                    )

                    for i in range(2):
                        parameter["doors_params"][13 * i + 0] = 0
                        parameter["doors_params"][13 * i + 1] = (
                            door_width_1 if i == 0 else door_width_2
                        )
                        parameter["doors_params"][13 * i + 2] = door_height_1
                        parameter["doors_params"][13 * i + 3] = (
                            parameter["doors_params"][3] * randRange(1, 0.8, 1.2)[0]
                        )

                        parameter["doors_params"][13 * i + 4] = (
                            parameter["doors_params"][13 * i + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * i + 5] = (
                            parameter["doors_params"][13 * i + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * i + 6] = parameter[
                            "doors_params"
                        ][13 * i + 3]
                        if i == 1:
                            parameter["doors_params"][13 * i + 4] = parameter[
                                "doors_params"
                            ][4]
                            parameter["doors_params"][13 * i + 5] = parameter[
                                "doors_params"
                            ][5]
                            parameter["doors_params"][13 * i + 6] = parameter[
                                "doors_params"
                            ][6]

                        if i == 0:
                            parameter["doors_params"][13 * i + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][13 * i + 1] / 2
                            )
                            parameter["doors_params"][13 * i + 11] = (
                                body_parameter["size"][1] / 2
                                - parameter["doors_params"][13 * i + 2] / 2
                            )
                            parameter["doors_params"][13 * i + 12] = (
                                body_parameter["size"][2] / 2
                            )

                        elif i == 1:
                            cur_total_offset_x = (
                                body_parameter["storagefurniture_layers_params"][3]
                                + body_parameter["storagefurniture_layers_params"][1]
                            )
                            parameter["doors_params"][13 * i + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][13 * i + 1] / 2
                                + cur_total_offset_x
                            )
                            parameter["doors_params"][13 * i + 11] = (
                                body_parameter["size"][1] / 2
                                - parameter["doors_params"][13 * i + 2] / 2
                            )
                            parameter["doors_params"][13 * i + 12] = (
                                body_parameter["size"][2] / 2
                            )

                        if i == 0:
                            parameter["doors_params"][13 * i + 9] = -np.random.uniform(
                                0, 30
                            )
                            parameter["doors_params"][13 * i + 0] = 0
                            parameter["doors_params"][13 * i + 7] = (
                                parameter["doors_params"][13 * i + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * i + 8] = (
                                -parameter["doors_params"][13 * i + 2]
                                / 4
                                * randRange(1, 0.8, 1.2)[0]
                            )

                        elif i == 1:
                            parameter["doors_params"][13 * i + 9] = np.random.uniform(
                                0, 30
                            )
                            parameter["doors_params"][13 * i + 0] = 1
                            parameter["doors_params"][13 * i + 7] = (
                                -parameter["doors_params"][13 * i + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * i + 8] = (
                                -parameter["doors_params"][13 * i + 2]
                                / 4
                                * randRange(1, 0.8, 1.2)[0]
                            )
                    for j in range(2):
                        parameter["doors_params"][13 * (j + 2) + 0] = 0
                        parameter["doors_params"][13 * (j + 2) + 1] = (
                            door_width_1 if j == 0 else door_width_2
                        )
                        parameter["doors_params"][13 * (j + 2) + 2] = door_height_2
                        parameter["doors_params"][13 * (j + 2) + 3] = (
                            parameter["doors_params"][3] * randRange(1, 0.8, 1.2)[0]
                        )

                        parameter["doors_params"][13 * (j + 2) + 4] = (
                            parameter["doors_params"][13 * (j + 2) + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * (j + 2) + 5] = (
                            parameter["doors_params"][13 * (j + 2) + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * (j + 2) + 6] = parameter[
                            "doors_params"
                        ][13 * (j + 2) + 3]
                        if j == 1:
                            parameter["doors_params"][13 * (j + 2) + 4] = parameter[
                                "doors_params"
                            ][13 * (j + 1) + 4]
                            parameter["doors_params"][13 * (j + 2) + 5] = parameter[
                                "doors_params"
                            ][13 * (j + 1) + 5]
                            parameter["doors_params"][13 * (j + 2) + 6] = parameter[
                                "doors_params"
                            ][13 * (j + 1) + 6]

                        if j == 0:
                            parameter["doors_params"][13 * (j + 2) + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][13 * (j + 2) + 1] / 2
                            )
                            parameter["doors_params"][13 * (j + 2) + 11] = (
                                -body_parameter["size"][1] / 2
                                + body_parameter["base_size"][0]
                                + parameter["doors_params"][13 * (j + 2) + 2] / 2
                            )
                            parameter["doors_params"][13 * (j + 2) + 12] = (
                                body_parameter["size"][2] / 2
                            )

                        elif j == 1:
                            cur_total_offset_x = (
                                body_parameter["storagefurniture_layers_params"][3]
                                + body_parameter["storagefurniture_layers_params"][1]
                            )
                            parameter["doors_params"][13 * (j + 2) + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][13 * (j + 2) + 1] / 2
                                + cur_total_offset_x
                            )
                            parameter["doors_params"][13 * (j + 2) + 11] = (
                                -body_parameter["size"][1] / 2
                                + body_parameter["base_size"][0]
                                + parameter["doors_params"][13 * (j + 2) + 2] / 2
                            )
                            parameter["doors_params"][13 * (j + 2) + 12] = (
                                body_parameter["size"][2] / 2
                            )

                        if j == 0:
                            parameter["doors_params"][
                                13 * (j + 2) + 9
                            ] = -np.random.uniform(0, 30)
                            parameter["doors_params"][13 * (j + 2) + 0] = 0
                            parameter["doors_params"][13 * (j + 2) + 7] = (
                                parameter["doors_params"][13 * (j + 2) + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * (j + 2) + 8] = (
                                parameter["doors_params"][13 * (j + 2) + 2]
                                / 4
                                * randRange(1, 0.8, 1.2)[0]
                            )

                        elif j == 1:
                            parameter["doors_params"][13 * (j + 2) + 9] = (
                                np.random.uniform(0, 30)
                            )
                            parameter["doors_params"][13 * (j + 2) + 0] = 1
                            parameter["doors_params"][13 * (j + 2) + 7] = (
                                -parameter["doors_params"][13 * (j + 2) + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * (j + 2) + 8] = (
                                parameter["doors_params"][13 * (j + 2) + 2]
                                / 4
                                * randRange(1, 0.8, 1.2)[0]
                            )

            elif storagefurniture_type == "panel_Cabinet":
                num_of_door = (
                    int(body_parameter["storagefurniture_layers_params"][5 + 0]) + 1
                )
                parameter["number_of_door"][0] = num_of_door
                parameter["number_of_door"] = np.array(
                    [int(num) for num in parameter["number_of_door"]]
                )
                cur = 0
                current_door_offset_x_from_left = body_parameter[
                    "left_right_inner_size"
                ][0]
                for i in range(num_of_door):
                    parameter["doors_params"][cur * 13 + 1] = (
                        body_parameter["size"][0]
                        - 2 * body_parameter["left_right_inner_size"][0]
                    ) / num_of_door
                    parameter["doors_params"][cur * 13 + 2] = (
                        body_parameter["size"][1]
                        - body_parameter["base_size"][0]
                        - body_parameter["WHOLE_layer_offset"][0]
                        - body_parameter["WHOLE_layer_sizes"][0]
                    )
                    parameter["doors_params"][cur * 13 + 3] = (
                        parameter["doors_params"][3] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 4] = (
                        parameter["doors_params"][13 * cur + 1]
                        / 50
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 5] = (
                        parameter["doors_params"][13 * cur + 2]
                        / 4
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 6] = parameter["doors_params"][
                        13 * cur + 3
                    ]
                    parameter["doors_params"][13 * cur + 9] = np.random.uniform(0, 30)
                    if i % 2 == 0:
                        parameter["doors_params"][cur * 13 + 0] = 0
                        parameter["doors_params"][13 * cur + 9] = -parameter[
                            "doors_params"
                        ][13 * cur + 9]
                        parameter["doors_params"][13 * cur + 7] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                    else:
                        parameter["doors_params"][cur * 13 + 0] = 1
                        parameter["doors_params"][13 * cur + 7] = (
                            -parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )

                    # if np.random.randint(0,2) == 0:
                    #     parameter['doors_params'][cur * 13 + 0] = 2
                    #     parameter['doors_params'][13 * cur + 9] = np.random.uniform(0, 30)
                    #     tmp = parameter['doors_params'][13 * cur + 4]
                    #     parameter['doors_params'][13 * cur + 4] = parameter['doors_params'][13 * cur + 5]
                    #     parameter['doors_params'][13 * cur + 5] = tmp
                    #     parameter['doors_params'][13 * cur + 7] = parameter['doors_params'][13 * cur + 1] / 8 * randRange(1, 0.8, 1.2)[0]
                    #     parameter['doors_params'][13 * cur + 8] = parameter['doors_params'][13 * cur + 2] / 2 * randRange(1, 0.8, 0.9)[0]

                    parameter["doors_params"][13 * cur + 10] = (
                        -body_parameter["size"][0] / 2
                        + current_door_offset_x_from_left
                        + parameter["doors_params"][cur * 13 + 1] / 2
                    )
                    parameter["doors_params"][13 * cur + 11] = (
                        -body_parameter["size"][1] / 2
                        + body_parameter["base_size"][0]
                        + parameter["doors_params"][cur * 13 + 2] / 2
                    )
                    parameter["doors_params"][13 * cur + 12] = (
                        body_parameter["size"][2] / 2
                        + parameter["doors_params"][cur * 13 + 3] / 2
                    )
                    current_door_offset_x_from_left += (
                        parameter["doors_params"][cur * 13 + 1]
                        + body_parameter["storagefurniture_layers_params"][5 + 1]
                    )
                    cur += 1

            elif storagefurniture_type == "Multi_layer_wardrobe":
                num_of_door_up = int(
                    body_parameter["storagefurniture_layers_params"][0 * 5] + 1
                )
                num_of_door_bottom = int(
                    body_parameter["storagefurniture_layers_params"][1 * 5] + 1
                )
                parameter["number_of_door"][0] = num_of_door_up + num_of_door_bottom
                parameter["number_of_door"] = np.array(
                    [int(num) for num in parameter["number_of_door"]]
                )
                cur = 0

                for i in range(num_of_door_up):
                    parameter["doors_params"][13 * cur + 2] = (
                        body_parameter["WHOLE_layer_offset"][0]
                        + body_parameter["lid_offset"][1]
                    )
                    parameter["doors_params"][13 * cur + 3] = (
                        np.minimum(
                            parameter["doors_params"][3],
                            body_parameter["storagefurniture_layers_params"][0 * 5 + 1],
                        )
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 11] = (
                        body_parameter["size"][1] / 2
                        - parameter["doors_params"][13 * cur + 2] / 2
                        + body_parameter["lid_offset"][1]
                    )
                    parameter["doors_params"][13 * cur + 6] = parameter["doors_params"][
                        13 * cur + 3
                    ]
                    if i == 0:
                        parameter["doors_params"][13 * cur + 1] = body_parameter[
                            "storagefurniture_layers_params"
                        ][0 * 5 + 3]
                        parameter["doors_params"][13 * cur + 10] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + parameter["doors_params"][13 * cur + 1] / 2
                        )
                    elif i == num_of_door_up - 1:
                        parameter["doors_params"][13 * cur + 1] = (
                            body_parameter["size"][0]
                            - 2 * body_parameter["left_right_inner_size"][0]
                            - body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 3
                            ]
                            - body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 4
                            ]
                            * (num_of_door_up - 2)
                        )
                        parameter["doors_params"][13 * cur + 10] = (
                            body_parameter["size"][0] / 2
                            - body_parameter["left_right_inner_size"][0]
                            - parameter["doors_params"][13 * cur + 1] / 2
                        )
                    else:
                        parameter["doors_params"][13 * cur + 1] = body_parameter[
                            "storagefurniture_layers_params"
                        ][0 * 5 + 4]
                        parameter["doors_params"][13 * cur + 10] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 3
                            ]
                            + body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 4
                            ]
                            * (i - 1)
                            + parameter["doors_params"][13 * cur + 1] / 2
                        )

                    parameter["doors_params"][13 * cur + 4] = (
                        parameter["doors_params"][13 * cur + 1]
                        / 50
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 5] = (
                        parameter["doors_params"][13 * cur + 2]
                        / 4
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 6] = parameter["doors_params"][
                        13 * cur + 3
                    ]
                    door_open_type = np.random.randint(0, 2)
                    if door_open_type == 0:
                        if i == 0:
                            parameter["doors_params"][
                                13 * cur + 9
                            ] = -np.random.uniform(0, 180)
                        else:
                            parameter["doors_params"][
                                13 * cur + 9
                            ] = -np.random.uniform(0, 90)
                        parameter["doors_params"][13 * cur + 0] = 0
                        parameter["doors_params"][13 * cur + 7] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 12] = (
                            body_parameter["size"][2] / 2
                            + parameter["doors_params"][cur * 13 + 3] / 2
                        )
                    elif door_open_type == 1:
                        if i == num_of_door_up - 1:
                            parameter["doors_params"][13 * cur + 9] = np.random.uniform(
                                0, 180
                            )
                        else:
                            parameter["doors_params"][13 * cur + 9] = np.random.uniform(
                                0, 90
                            )
                        parameter["doors_params"][13 * cur + 0] = 1
                        parameter["doors_params"][13 * cur + 7] = (
                            -parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 12] = (
                            body_parameter["size"][2] / 2
                            + parameter["doors_params"][cur * 13 + 3] / 2
                        )
                    # elif door_open_type == 2:
                    #     parameter['doors_params'][13 * cur + 9] = np.random.uniform(0, 90)
                    #     parameter['doors_params'][13 * cur + 0] = 2
                    #     tmp = parameter['doors_params'][13 * cur + 4]
                    #     parameter['doors_params'][13 * cur + 4] = parameter['doors_params'][5]
                    #     parameter['doors_params'][13 * cur + 5] = tmp
                    #     parameter['doors_params'][13 * cur + 7] = parameter['doors_params'][13 * cur + 1] / 8 * randRange(1, 0.8, 1.2)[0]
                    #     parameter['doors_params'][13 * cur + 8] = parameter['doors_params'][13 * cur + 2] / 2 * randRange(1, 0.8, 0.9)[0]
                    #     parameter['doors_params'][13 * cur + 12] = body_parameter['size'][2] / 2 + parameter['doors_params'][cur * 13 + 3] / 2
                    cur += 1

                for i in range(num_of_door_bottom):
                    parameter["doors_params"][13 * cur + 2] = body_parameter[
                        "WHOLE_interval_between_layers"
                    ][0]
                    parameter["doors_params"][13 * cur + 3] = (
                        np.minimum(
                            parameter["doors_params"][3],
                            body_parameter["storagefurniture_layers_params"][1 * 5 + 1],
                        )
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 11] = (
                        body_parameter["size"][1] / 2
                        - body_parameter["WHOLE_layer_offset"][0]
                        - parameter["doors_params"][13 * cur + 2] / 2
                    )
                    parameter["doors_params"][13 * cur + 6] = parameter["doors_params"][
                        13 * cur + 3
                    ]
                    if i == 0:
                        parameter["doors_params"][13 * cur + 1] = body_parameter[
                            "storagefurniture_layers_params"
                        ][1 * 5 + 3]
                        parameter["doors_params"][13 * cur + 10] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + parameter["doors_params"][13 * cur + 1] / 2
                        )
                    elif i == num_of_door_bottom - 1:
                        parameter["doors_params"][13 * cur + 1] = (
                            body_parameter["size"][0]
                            - 2 * body_parameter["left_right_inner_size"][0]
                            - body_parameter["storagefurniture_layers_params"][
                                1 * 5 + 3
                            ]
                            - body_parameter["storagefurniture_layers_params"][
                                1 * 5 + 4
                            ]
                            * (num_of_door_bottom - 2)
                        )
                        parameter["doors_params"][13 * cur + 10] = (
                            body_parameter["size"][0] / 2
                            - body_parameter["left_right_inner_size"][0]
                            - parameter["doors_params"][13 * cur + 1] / 2
                        )
                    else:
                        parameter["doors_params"][13 * cur + 1] = body_parameter[
                            "storagefurniture_layers_params"
                        ][1 * 5 + 4]
                        parameter["doors_params"][13 * cur + 10] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + body_parameter["storagefurniture_layers_params"][
                                1 * 5 + 3
                            ]
                            + body_parameter["storagefurniture_layers_params"][
                                1 * 5 + 4
                            ]
                            * (i - 1)
                            + parameter["doors_params"][13 * cur + 1] / 2
                        )

                    parameter["doors_params"][13 * cur + 4] = (
                        parameter["doors_params"][13 * cur + 1]
                        / 50
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 5] = (
                        parameter["doors_params"][13 * cur + 2]
                        / 4
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 6] = parameter["doors_params"][
                        13 * cur + 3
                    ]
                    door_open_type = np.random.randint(0, 2)
                    if door_open_type == 0:
                        if i == 0:
                            parameter["doors_params"][
                                13 * cur + 9
                            ] = -np.random.uniform(0, 180)
                        else:
                            parameter["doors_params"][
                                13 * cur + 9
                            ] = -np.random.uniform(0, 90)
                        parameter["doors_params"][13 * cur + 9] = -np.random.uniform(
                            0, 90
                        )
                        parameter["doors_params"][13 * cur + 0] = 0
                        parameter["doors_params"][13 * cur + 7] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 12] = (
                            body_parameter["size"][2] / 2
                        )
                    elif door_open_type == 1:
                        if i == num_of_door_bottom - 1:
                            parameter["doors_params"][13 * cur + 9] = np.random.uniform(
                                0, 180
                            )
                        else:
                            parameter["doors_params"][13 * cur + 9] = np.random.uniform(
                                0, 90
                            )
                        parameter["doors_params"][13 * cur + 0] = 1
                        parameter["doors_params"][13 * cur + 7] = (
                            -parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 12] = (
                            body_parameter["size"][2] / 2
                        )
                    cur += 1

            elif storagefurniture_type == "bookcase":
                num_of_door_up = 2 if random.random() < 0.5 else 4
                num_of_door_bottom = 2 if random.random() < 0.5 else 4
                num_of_door = num_of_door_up + num_of_door_bottom
                parameter["number_of_door"][0] = num_of_door
                parameter["number_of_door"] = np.array(
                    [int(num) for num in parameter["number_of_door"]]
                )
                cur = 0
                for i in range(num_of_door_up):
                    if num_of_door_up == 2:
                        if i == 0:
                            parameter["doors_params"][cur * 13 + 1] = body_parameter[
                                "storagefurniture_layers_params"
                            ][0 * 5 + 3]
                            parameter["doors_params"][cur * 13 + 2] = (
                                body_parameter["WHOLE_layer_offset"][0]
                                + body_parameter["lid_offset"][1]
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                            )
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        0 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][cur * 13 + 1] / 2
                                + parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2]
                                - body_parameter["back_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 1
                                ]
                            ) / 2 + parameter["doors_params"][cur * 13 + 3] / 2
                        elif i == 1:
                            parameter["doors_params"][cur * 13 + 1] = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 3
                                ]
                            )
                            parameter["doors_params"][cur * 13 + 2] = (
                                body_parameter["WHOLE_layer_offset"][0]
                                + body_parameter["lid_offset"][1]
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                            )
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        0 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                body_parameter["size"][0] / 2
                                - body_parameter["left_right_inner_size"][0]
                                - parameter["doors_params"][cur * 13 + 1] / 2
                                - parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2] / 2
                                + parameter["doors_params"][cur * 13 + 3] / 2
                            )

                        parameter["doors_params"][13 * cur + 4] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 5] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 6] = parameter[
                            "doors_params"
                        ][13 * cur + 3]
                        parameter["doors_params"][13 * cur + 9] = 0
                        if i % 2 == 0:
                            parameter["doors_params"][cur * 13 + 0] = 1
                            parameter["doors_params"][13 * cur + 7] = (
                                -parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        else:
                            parameter["doors_params"][cur * 13 + 0] = 0
                            parameter["doors_params"][13 * cur + 7] = (
                                parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )

                        parameter["doors_params"][13 * cur + 11] = (
                            body_parameter["size"][1] / 2
                            + body_parameter["lid_offset"][1]
                            - parameter["doors_params"][cur * 13 + 2] / 2
                        )
                        cur += 1
                    elif num_of_door_up == 4:
                        if i == 0:
                            parameter["doors_params"][cur * 13 + 1] = body_parameter[
                                "storagefurniture_layers_params"
                            ][0 * 5 + 3]
                            parameter["doors_params"][cur * 13 + 2] = (
                                body_parameter["WHOLE_layer_offset"][0]
                                + body_parameter["lid_offset"][1]
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                            ) / 2
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        0 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][cur * 13 + 1] / 2
                                + parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 11] = (
                                body_parameter["size"][1] / 2
                                + body_parameter["lid_offset"][1]
                                - parameter["doors_params"][cur * 13 + 2] / 2
                            )

                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2]
                                - body_parameter["back_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 1
                                ]
                            ) / 2 + parameter["doors_params"][cur * 13 + 3] / 2
                        elif i == 1:
                            parameter["doors_params"][cur * 13 + 1] = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 3
                                ]
                            )
                            parameter["doors_params"][cur * 13 + 2] = (
                                body_parameter["WHOLE_layer_offset"][0]
                                + body_parameter["lid_offset"][1]
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                            ) / 2
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        0 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                body_parameter["size"][0] / 2
                                - body_parameter["left_right_inner_size"][0]
                                - parameter["doors_params"][cur * 13 + 1] / 2
                                - parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 11] = (
                                body_parameter["size"][1] / 2
                                + body_parameter["lid_offset"][1]
                                - parameter["doors_params"][cur * 13 + 2] / 2
                            )

                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2] / 2
                                + parameter["doors_params"][cur * 13 + 3] / 2
                            )

                        elif i == 2:
                            parameter["doors_params"][cur * 13 + 1] = body_parameter[
                                "storagefurniture_layers_params"
                            ][0 * 5 + 3]
                            parameter["doors_params"][cur * 13 + 2] = (
                                body_parameter["WHOLE_layer_offset"][0]
                                + body_parameter["lid_offset"][1]
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                            ) / 2
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        0 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][cur * 13 + 1] / 2
                                + parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 11] = (
                                body_parameter["size"][1] / 2
                                + body_parameter["lid_offset"][1]
                                - parameter["doors_params"][cur * 13 + 2]
                                - parameter["doors_params"][cur * 13 + 2] / 2
                            )
                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2]
                                - body_parameter["back_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 1
                                ]
                            ) / 2 + parameter["doors_params"][cur * 13 + 3] / 2
                        elif i == 3:
                            parameter["doors_params"][cur * 13 + 1] = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 3
                                ]
                            )
                            parameter["doors_params"][cur * 13 + 2] = (
                                body_parameter["WHOLE_layer_offset"][0]
                                + body_parameter["lid_offset"][1]
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                            ) / 2
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        0 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                body_parameter["size"][0] / 2
                                - body_parameter["left_right_inner_size"][0]
                                - parameter["doors_params"][cur * 13 + 1] / 2
                                - parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 11] = (
                                body_parameter["size"][1] / 2
                                + body_parameter["lid_offset"][1]
                                - parameter["doors_params"][cur * 13 + 2]
                                - parameter["doors_params"][cur * 13 + 2] / 2
                            )
                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2] / 2
                                + parameter["doors_params"][cur * 13 + 3] / 2
                            )

                        parameter["doors_params"][13 * cur + 4] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 5] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 6] = parameter[
                            "doors_params"
                        ][13 * cur + 3]
                        parameter["doors_params"][13 * cur + 9] = 0
                        if i % 2 == 0:
                            parameter["doors_params"][cur * 13 + 0] = 1
                            parameter["doors_params"][13 * cur + 7] = (
                                -parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        else:
                            parameter["doors_params"][cur * 13 + 0] = 0
                            parameter["doors_params"][13 * cur + 7] = (
                                parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        cur += 1

                for i in range(num_of_door_bottom):
                    if num_of_door_bottom == 2:
                        if i == 0:
                            parameter["doors_params"][cur * 13 + 1] = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                            ) / 2
                            parameter["doors_params"][cur * 13 + 2] = (
                                (
                                    body_parameter["size"][1]
                                    + body_parameter["lid_offset"][1]
                                    - body_parameter["WHOLE_layer_offset"][0]
                                    - body_parameter["WHOLE_interval_between_layers"][0]
                                )
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                                - body_parameter["base_size"][0] / 2
                            )
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        2 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][cur * 13 + 1] / 2
                                + parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )

                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2]
                                - body_parameter["back_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    2 * 5 + 1
                                ]
                            ) / 2 + parameter["doors_params"][cur * 13 + 3] / 2
                        elif i == 1:
                            parameter["doors_params"][cur * 13 + 1] = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                            ) / 2
                            parameter["doors_params"][cur * 13 + 2] = (
                                (
                                    body_parameter["size"][1]
                                    + body_parameter["lid_offset"][1]
                                    - body_parameter["WHOLE_layer_offset"][0]
                                    - body_parameter["WHOLE_interval_between_layers"][0]
                                )
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                                - body_parameter["base_size"][0] / 2
                            )
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        2 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                body_parameter["size"][0] / 2
                                - body_parameter["left_right_inner_size"][0]
                                - parameter["doors_params"][cur * 13 + 1] / 2
                                - parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2] / 2
                                + parameter["doors_params"][cur * 13 + 3] / 2
                            )

                        parameter["doors_params"][13 * cur + 4] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 5] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 6] = parameter[
                            "doors_params"
                        ][13 * cur + 3]
                        parameter["doors_params"][13 * cur + 9] = 0
                        if i % 2 == 0:
                            parameter["doors_params"][cur * 13 + 0] = 1
                            parameter["doors_params"][13 * cur + 7] = (
                                -parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        else:
                            parameter["doors_params"][cur * 13 + 0] = 0
                            parameter["doors_params"][13 * cur + 7] = (
                                parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )

                        parameter["doors_params"][13 * cur + 11] = (
                            -body_parameter["size"][1] / 2
                            + body_parameter["base_size"][0]
                            + parameter["doors_params"][cur * 13 + 2] / 2
                        )
                        cur += 1
                    elif num_of_door_bottom == 4:
                        if i == 0:
                            parameter["doors_params"][cur * 13 + 1] = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                            ) / 2
                            parameter["doors_params"][cur * 13 + 2] = (
                                (
                                    body_parameter["size"][1]
                                    + body_parameter["lid_offset"][1]
                                    - body_parameter["WHOLE_layer_offset"][0]
                                    - body_parameter["WHOLE_interval_between_layers"][0]
                                )
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                                - body_parameter["base_size"][0] / 2
                            ) / 2
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        2 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][cur * 13 + 1] / 2
                                + parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 11] = (
                                -body_parameter["size"][1] / 2
                                + body_parameter["base_size"][0]
                                + parameter["doors_params"][cur * 13 + 2] / 2
                                + parameter["doors_params"][cur * 13 + 2]
                            )

                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2]
                                - body_parameter["back_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    2 * 5 + 1
                                ]
                            ) / 2 + parameter["doors_params"][cur * 13 + 3] / 2
                        elif i == 1:
                            parameter["doors_params"][cur * 13 + 1] = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                            ) / 2

                            parameter["doors_params"][cur * 13 + 2] = (
                                (
                                    body_parameter["size"][1]
                                    + body_parameter["lid_offset"][1]
                                    - body_parameter["WHOLE_layer_offset"][0]
                                    - body_parameter["WHOLE_interval_between_layers"][0]
                                )
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                                - body_parameter["base_size"][0] / 2
                            ) / 2
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        2 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                body_parameter["size"][0] / 2
                                - body_parameter["left_right_inner_size"][0]
                                - parameter["doors_params"][cur * 13 + 1] / 2
                                - parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 11] = (
                                -body_parameter["size"][1] / 2
                                + body_parameter["base_size"][0]
                                + parameter["doors_params"][cur * 13 + 2] / 2
                                + parameter["doors_params"][cur * 13 + 2]
                            )

                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2] / 2
                                + parameter["doors_params"][cur * 13 + 3] / 2
                            )

                        elif i == 2:
                            parameter["doors_params"][cur * 13 + 1] = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                            ) / 2
                            parameter["doors_params"][cur * 13 + 2] = (
                                (
                                    body_parameter["size"][1]
                                    + body_parameter["lid_offset"][1]
                                    - body_parameter["WHOLE_layer_offset"][0]
                                    - body_parameter["WHOLE_interval_between_layers"][0]
                                )
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                                - body_parameter["base_size"][0] / 2
                            ) / 2
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        2 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + parameter["doors_params"][cur * 13 + 1] / 2
                                + parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 11] = (
                                -body_parameter["size"][1] / 2
                                + body_parameter["base_size"][0]
                                + parameter["doors_params"][cur * 13 + 2] / 2
                            )

                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2]
                                - body_parameter["back_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    2 * 5 + 1
                                ]
                            ) / 2 + parameter["doors_params"][cur * 13 + 3] / 2
                        elif i == 3:
                            parameter["doors_params"][cur * 13 + 1] = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                            ) / 2

                            parameter["doors_params"][cur * 13 + 2] = (
                                (
                                    body_parameter["size"][1]
                                    + body_parameter["lid_offset"][1]
                                    - body_parameter["WHOLE_layer_offset"][0]
                                    - body_parameter["WHOLE_interval_between_layers"][0]
                                )
                                - body_parameter["WHOLE_layer_sizes"][0] / 2
                                - body_parameter["base_size"][0] / 2
                            ) / 2
                            parameter["doors_params"][cur * 13 + 3] = (
                                np.minimum(
                                    parameter["doors_params"][3],
                                    body_parameter["storagefurniture_layers_params"][
                                        2 * 5 + 1
                                    ],
                                )
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["doors_params"][13 * cur + 10] = (
                                body_parameter["size"][0] / 2
                                - body_parameter["left_right_inner_size"][0]
                                - parameter["doors_params"][cur * 13 + 1] / 2
                                - parameter["doors_params"][cur * 13 + 1]
                                / 2
                                * randRange(1, 0.2, 0.7)[0]
                            )
                            parameter["doors_params"][13 * cur + 11] = (
                                -body_parameter["size"][1] / 2
                                + body_parameter["base_size"][0]
                                + parameter["doors_params"][cur * 13 + 2] / 2
                            )

                            parameter["doors_params"][13 * cur + 12] = (
                                body_parameter["size"][2] / 2
                                + parameter["doors_params"][cur * 13 + 3] / 2
                            )

                        parameter["doors_params"][13 * cur + 4] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 50
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 5] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 4
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 6] = parameter[
                            "doors_params"
                        ][13 * cur + 3]
                        parameter["doors_params"][13 * cur + 9] = 0
                        if i % 2 == 0:
                            parameter["doors_params"][cur * 13 + 0] = 1
                            parameter["doors_params"][13 * cur + 7] = (
                                -parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        else:
                            parameter["doors_params"][cur * 13 + 0] = 0
                            parameter["doors_params"][13 * cur + 7] = (
                                parameter["doors_params"][13 * cur + 1]
                                / 2
                                * randRange(1, 0.8, 0.9)[0]
                            )
                            parameter["doors_params"][13 * cur + 8] = (
                                parameter["doors_params"][13 * cur + 2]
                                / 8
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        cur += 1

            elif storagefurniture_type == "Drinking_machine":
                num_of_door = 2
                parameter["number_of_door"][0] = num_of_door
                parameter["number_of_door"] = np.array(
                    [int(num) for num in parameter["number_of_door"]]
                )
                height_1 = body_parameter["WHOLE_layer_offset"][0]
                height_3 = (
                    body_parameter["size"][1]
                    - body_parameter["WHOLE_interval_between_layers"][0]
                    - body_parameter["WHOLE_layer_offset"][0]
                )
                cur = 0
                for i in range(num_of_door):
                    parameter["doors_params"][cur * 13 + 1] = (
                        body_parameter["size"][0]
                        - body_parameter["left_right_inner_size"][0] * 2
                    )
                    parameter["doors_params"][cur * 13 + 2] = (
                        height_1 if i == 0 else height_3
                    )
                    parameter["doors_params"][cur * 13 + 3] = (
                        np.minimum(
                            parameter["doors_params"][3],
                            body_parameter["storagefurniture_layers_params"][2 * 5 + 1],
                        )
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 4] = (
                        parameter["doors_params"][13 * cur + 1]
                        / 50
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 5] = (
                        parameter["doors_params"][13 * cur + 2]
                        / 4
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 6] = (
                        2
                        * parameter["doors_params"][13 * cur + 3]
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 9] = np.random.uniform(0, 180)
                    if i % 2 == 0:
                        parameter["doors_params"][cur * 13 + 0] = 0
                        parameter["doors_params"][13 * cur + 9] = -parameter[
                            "doors_params"
                        ][13 * cur + 9]
                        parameter["doors_params"][13 * cur + 7] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                    else:
                        parameter["doors_params"][cur * 13 + 0] = 1
                        parameter["doors_params"][13 * cur + 7] = (
                            -parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )

                    parameter["doors_params"][13 * cur + 10] = (
                        -body_parameter["size"][0] / 2
                        + body_parameter["left_right_inner_size"][0]
                        + parameter["doors_params"][cur * 13 + 1] / 2
                    )
                    parameter["doors_params"][13 * cur + 11] = (
                        (
                            -body_parameter["size"][1] / 2
                            + body_parameter["base_size"][0]
                            + parameter["doors_params"][cur * 13 + 2] / 2
                        )
                        if i == 1
                        else (
                            body_parameter["size"][1] / 2
                            + body_parameter["lid_offset"][1]
                            - parameter["doors_params"][cur * 13 + 2] / 2
                        )
                    )
                    parameter["doors_params"][13 * cur + 12] = (
                        body_parameter["size"][2] / 2
                        + parameter["doors_params"][cur * 13 + 3] / 2
                    )
                    cur += 1

            elif storagefurniture_type == "Cabinet_vertical":
                parameter["number_of_door"][0] = (
                    body_parameter["storagefurniture_layers_params"][0 * 5] + 1
                )
                parameter["number_of_door"] = np.array(
                    [int(num) for num in parameter["number_of_door"]]
                )
                num_of_vertical_on_second_layer = int(parameter["number_of_door"][0])
                cur = 0

                for i in range(num_of_vertical_on_second_layer):
                    if i == 0:
                        parameter["doors_params"][13 * cur + 1] = body_parameter[
                            "storagefurniture_layers_params"
                        ][0 * 5 + 3]
                        parameter["doors_params"][13 * cur + 10] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + parameter["doors_params"][13 * cur + 1] / 2
                        )
                    elif i == body_parameter["storagefurniture_layers_params"][0 * 5]:
                        parameter["doors_params"][13 * cur + 1] = (
                            body_parameter["size"][0]
                            - 2 * body_parameter["left_right_inner_size"][0]
                            - body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 3
                            ]
                            - body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 4
                            ]
                            * (
                                body_parameter["storagefurniture_layers_params"][0 * 5]
                                - 1
                            )
                        )
                        parameter["doors_params"][13 * cur + 10] = (
                            body_parameter["size"][0] / 2
                            - body_parameter["left_right_inner_size"][0]
                            - parameter["doors_params"][13 * cur + 1] / 2
                        )
                    else:
                        parameter["doors_params"][13 * cur + 1] = body_parameter[
                            "storagefurniture_layers_params"
                        ][0 * 5 + 4]
                        parameter["doors_params"][13 * cur + 10] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 3
                            ]
                            + body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 4
                            ]
                            * (i - 1)
                            + parameter["doors_params"][13 * cur + 1] / 2
                        )

                    if (
                        num_of_additional_layers[i] == 2
                        or num_of_additional_layers[i] == 3
                    ):
                        total_height = (
                            body_parameter["size"][1]
                            + body_parameter["lid_offset"][1]
                            - body_parameter["base_size"][0]
                        )
                        cur_doors_height = (
                            total_height
                            / (num_of_additional_layers[i] + 1)
                            * (num_of_additional_layers[i])
                        )
                    else:
                        cur_doors_height = (
                            body_parameter["size"][1]
                            + body_parameter["lid_offset"][1]
                            - body_parameter["base_size"][0]
                        )

                    parameter["doors_params"][13 * cur + 4] = (
                        parameter["doors_params"][13 * cur + 1]
                        / 50
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 5] = (
                        parameter["doors_params"][13 * cur + 2]
                        / 4
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 2] = cur_doors_height
                    parameter["doors_params"][13 * cur + 3] = (
                        np.minimum(
                            parameter["doors_params"][3],
                            body_parameter["storagefurniture_layers_params"][0 * 5 + 1],
                        )
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["doors_params"][13 * cur + 11] = (
                        -body_parameter["size"][1] / 2
                        + parameter["doors_params"][13 * cur + 2] / 2
                        + body_parameter["base_size"][0]
                    )
                    parameter["doors_params"][13 * cur + 6] = parameter["doors_params"][
                        13 * cur + 3
                    ]
                    door_open_type = np.random.randint(0, 2)
                    if door_open_type == 0:
                        if i == 0:
                            parameter["doors_params"][
                                13 * cur + 9
                            ] = -np.random.uniform(0, 30)
                        else:
                            parameter["doors_params"][
                                13 * cur + 9
                            ] = -np.random.uniform(0, 30)
                        parameter["doors_params"][13 * cur + 0] = 0
                        parameter["doors_params"][13 * cur + 7] = (
                            parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 12] = (
                            body_parameter["size"][2] / 2
                            + parameter["doors_params"][cur * 13 + 3] / 2
                        )
                    elif door_open_type == 1:
                        if i == body_parameter["storagefurniture_layers_params"][0 * 5]:
                            parameter["doors_params"][13 * cur + 9] = np.random.uniform(
                                0, 30
                            )
                        else:
                            parameter["doors_params"][13 * cur + 9] = np.random.uniform(
                                0, 30
                            )
                        parameter["doors_params"][13 * cur + 0] = 1
                        parameter["doors_params"][13 * cur + 7] = (
                            -parameter["doors_params"][13 * cur + 1]
                            / 2
                            * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["doors_params"][13 * cur + 8] = (
                            parameter["doors_params"][13 * cur + 2]
                            / 8
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["doors_params"][13 * cur + 12] = (
                            body_parameter["size"][2] / 2
                            + parameter["doors_params"][cur * 13 + 3] / 2
                        )
                    # elif door_open_type == 2:
                    #     parameter['doors_params'][13 * cur + 9] = np.random.uniform(0, 30)
                    #     parameter['doors_params'][13 * cur + 0] = 2
                    #     tmp = parameter['doors_params'][13 * cur + 4]
                    #     parameter['doors_params'][13 * cur + 4] = parameter['doors_params'][13 * cur + 5]
                    #     parameter['doors_params'][13 * cur + 5] = tmp
                    #     parameter['doors_params'][13 * cur + 7] = parameter['doors_params'][13 * cur + 1] / 8 * randRange(1, 0.8, 1.2)[0]
                    #     parameter['doors_params'][13 * cur + 8] = parameter['doors_params'][13 * cur + 2] / 2 * randRange(1, 0.8, 0.9)[0]
                    #     parameter['doors_params'][13 * cur + 12] = body_parameter['size'][2] / 2 + parameter['doors_params'][cur * 13 + 3] / 2
                    cur += 1
            concept["parameters"] = {
                k: v.tolist()
                if k
                not in [
                    "storagefurniture_layers_params",
                    "additional_layers_params",
                    "drawers_params",
                    "additional_legs_params",
                ]
                else v
                for k, v in parameter.items()
            }
            new_concepts.append(concept)

        elif template == "Regular_drawer":
            body_parameter = concepts[0]["parameters"]
            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]
            num_of_compartment_this_level = [
                len(row) for row in occupancy_of_each_compartment
            ]
            if storagefurniture_type == "Cabinet_transverse":
                upper_num = int(body_parameter["storagefurniture_layers_params"][0] + 1)
                record_cur_distance = body_parameter["left_right_inner_size"][0]
                cur = 0
                parameter["number_of_drawer"][0] = upper_num
                parameter["number_of_drawer"] = np.array(
                    [int(num) for num in parameter["number_of_drawer"]]
                )
                for i in range(upper_num):
                    if i == upper_num - 1:
                        parameter["drawers_params"][20 * cur + 0] = (
                            body_parameter["size"][0]
                            - body_parameter["left_right_inner_size"][0]
                            - record_cur_distance
                        ) * randRange(1, 0.7, 0.95)[0]
                        parameter["drawers_params"][20 * cur + 2] = (
                            body_parameter["size"][2] - body_parameter["back_size"][0]
                        ) * randRange(1, 0.7, 0.9)[0]
                        parameter["drawers_params"][20 * cur + 1] = (
                            body_parameter["WHOLE_layer_offset"][0]
                            - body_parameter["WHOLE_layer_sizes"][0]
                        ) * randRange(1, 0.8, 0.85)[0]
                        parameter["drawers_params"][20 * cur + 3] = (
                            body_parameter["WHOLE_layer_offset"][0]
                            - parameter["drawers_params"][20 * cur + 1]
                        ) / 3
                        parameter["drawers_params"][20 * cur + 4] = (
                            body_parameter["size"][0]
                            - body_parameter["left_right_inner_size"][0]
                            - record_cur_distance
                        )
                        parameter["drawers_params"][20 * cur + 5] = body_parameter[
                            "WHOLE_layer_offset"
                        ][0]
                        parameter["drawers_params"][20 * cur + 6] *= randRange(
                            1, 0.8, 1.2
                        )[0]
                        parameter["drawers_params"][20 * cur + 7] = 0
                        parameter["drawers_params"][20 * cur + 8] = (
                            body_parameter["left_right_inner_size"][0]
                            * randRange(1, 0.7, 0.8)[0]
                        )
                        parameter["drawers_params"][20 * cur + 9] = (
                            body_parameter["back_size"][0] * randRange(1, 0.7, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * cur + 10] = int(1)
                        parameter["drawers_params"][20 * cur + 11] = (
                            parameter["drawers_params"][20 * cur + 4]
                            / 3
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 12] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 20
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 13] *= randRange(
                            1, 0.9, 1.2
                        )[0]
                        parameter["drawers_params"][20 * cur + 14] = 0
                        parameter["drawers_params"][20 * cur + 15] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 4
                            * randRange(1, 0.5, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * cur + 16] = 0
                        parameter["drawers_params"][20 * cur + 17] = (
                            -body_parameter["size"][0] / 2
                            + record_cur_distance
                            + parameter["drawers_params"][20 * cur + 4] / 2
                        )
                        parameter["drawers_params"][20 * cur + 18] = (
                            body_parameter["size"][1] / 2
                            - parameter["drawers_params"][20 * cur + 5] / 2
                        )
                        parameter["drawers_params"][20 * cur + 19] = (
                            body_parameter["size"][2]
                            - parameter["drawers_params"][20 * cur + 2]
                        ) / 2 + parameter["drawers_params"][20 * cur + 2] * randRange(
                            1, 0.2, 0.6
                        )[0]

                        record_cur_distance += (
                            parameter["drawers_params"][20 * cur + 4]
                            + body_parameter["storagefurniture_layers_params"][1]
                        )
                        cur += 1

                    elif i == 0:
                        parameter["drawers_params"][20 * cur + 0] = (
                            body_parameter["storagefurniture_layers_params"][3]
                            * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][20 * cur + 2] = (
                            body_parameter["size"][2] - body_parameter["back_size"][0]
                        ) * randRange(1, 0.7, 0.9)[0]
                        parameter["drawers_params"][20 * cur + 1] = (
                            body_parameter["WHOLE_layer_offset"][0]
                            - body_parameter["WHOLE_layer_sizes"][0]
                        ) * randRange(1, 0.8, 0.85)[0]
                        parameter["drawers_params"][20 * cur + 3] = (
                            body_parameter["WHOLE_layer_offset"][0]
                            - parameter["drawers_params"][20 * cur + 1]
                        ) / 3
                        parameter["drawers_params"][20 * cur + 4] = body_parameter[
                            "storagefurniture_layers_params"
                        ][3]
                        parameter["drawers_params"][20 * cur + 5] = body_parameter[
                            "WHOLE_layer_offset"
                        ][0]
                        parameter["drawers_params"][20 * cur + 6] *= randRange(
                            1, 0.8, 1.2
                        )[0]
                        parameter["drawers_params"][20 * cur + 7] = 0
                        parameter["drawers_params"][20 * cur + 8] = (
                            body_parameter["left_right_inner_size"][0]
                            * randRange(1, 0.7, 0.8)[0]
                        )
                        parameter["drawers_params"][20 * cur + 9] = (
                            body_parameter["back_size"][0] * randRange(1, 0.7, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * cur + 10] = int(1)
                        parameter["drawers_params"][20 * cur + 11] = (
                            parameter["drawers_params"][20 * cur + 4]
                            / 3
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 12] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 20
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 13] *= randRange(
                            1, 0.9, 1.2
                        )[0]
                        parameter["drawers_params"][20 * cur + 14] = 0
                        parameter["drawers_params"][20 * cur + 15] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 4
                            * randRange(1, 0.5, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * cur + 16] = 0
                        parameter["drawers_params"][20 * cur + 17] = (
                            -body_parameter["size"][0] / 2
                            + record_cur_distance
                            + parameter["drawers_params"][20 * cur + 4] / 2
                        )
                        parameter["drawers_params"][20 * cur + 18] = (
                            body_parameter["size"][1] / 2
                            - parameter["drawers_params"][20 * cur + 5] / 2
                        )
                        parameter["drawers_params"][20 * cur + 19] = (
                            body_parameter["size"][2]
                            - parameter["drawers_params"][20 * cur + 2]
                        ) / 2 + parameter["drawers_params"][20 * cur + 2] * randRange(
                            1, 0.2, 0.6
                        )[0]

                        record_cur_distance += (
                            parameter["drawers_params"][20 * cur + 4]
                            + body_parameter["storagefurniture_layers_params"][1]
                        )
                        cur += 1
                    elif i == 1:
                        parameter["drawers_params"][20 * cur + 0] = (
                            body_parameter["storagefurniture_layers_params"][4]
                            * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][20 * cur + 2] = (
                            body_parameter["size"][2] - body_parameter["back_size"][0]
                        ) * randRange(1, 0.7, 0.9)[0]
                        parameter["drawers_params"][20 * cur + 1] = (
                            body_parameter["WHOLE_layer_offset"][0]
                            - -body_parameter["WHOLE_layer_sizes"][0]
                        ) * randRange(1, 0.8, 0.85)[0]
                        parameter["drawers_params"][20 * cur + 3] = (
                            body_parameter["WHOLE_layer_offset"][0]
                            - parameter["drawers_params"][20 * cur + 1]
                        ) / 3
                        parameter["drawers_params"][20 * cur + 4] = body_parameter[
                            "storagefurniture_layers_params"
                        ][4]
                        parameter["drawers_params"][20 * cur + 5] = body_parameter[
                            "WHOLE_layer_offset"
                        ][0]
                        parameter["drawers_params"][20 * cur + 6] *= randRange(
                            1, 0.8, 1.2
                        )[0]
                        parameter["drawers_params"][20 * cur + 7] = 0
                        parameter["drawers_params"][20 * cur + 8] = (
                            body_parameter["left_right_inner_size"][0]
                            * randRange(1, 0.7, 0.8)[0]
                        )
                        parameter["drawers_params"][20 * cur + 9] = (
                            body_parameter["back_size"][0] * randRange(1, 0.7, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * cur + 10] = int(1)
                        parameter["drawers_params"][20 * cur + 11] = (
                            parameter["drawers_params"][20 * cur + 4]
                            / 3
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 12] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 20
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 13] *= randRange(
                            1, 0.9, 1.2
                        )[0]
                        parameter["drawers_params"][20 * cur + 14] = 0
                        parameter["drawers_params"][20 * cur + 15] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 4
                            * randRange(1, 0.5, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * cur + 16] = 0
                        parameter["drawers_params"][20 * cur + 17] = (
                            -body_parameter["size"][0] / 2
                            + record_cur_distance
                            + parameter["drawers_params"][20 * cur + 4] / 2
                        )
                        parameter["drawers_params"][20 * cur + 18] = (
                            body_parameter["size"][1] / 2
                            - parameter["drawers_params"][20 * cur + 5] / 2
                        )
                        parameter["drawers_params"][20 * cur + 19] = (
                            body_parameter["size"][2]
                            - parameter["drawers_params"][20 * cur + 2]
                        ) / 2 + parameter["drawers_params"][20 * cur + 2] * randRange(
                            1, 0.2, 0.6
                        )[0]

                        record_cur_distance += (
                            parameter["drawers_params"][20 * cur + 4]
                            + body_parameter["storagefurniture_layers_params"][1]
                        )
                        cur += 1

                for j in range(len(num_of_compartment_this_level)):
                    total_remained_height = (
                        body_parameter["size"][1]
                        - body_parameter["base_size"][0]
                        - body_parameter["WHOLE_layer_offset"][0]
                    )
                    average_remained_height = (
                        total_remained_height
                        - (num_of_compartment_this_level[j] - 1)
                        * body_parameter["WHOLE_layer_sizes"][0]
                    ) / num_of_compartment_this_level[j]
                    record_height = body_parameter["WHOLE_layer_offset"][0]
                    x_width = 0
                    x_offset = 0

                    if j == 0:
                        x_width = body_parameter["storagefurniture_layers_params"][
                            5 + 3
                        ]
                        x_offset = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + body_parameter["storagefurniture_layers_params"][5 + 3]
                            / 2
                        )
                    elif j == len(num_of_compartment_this_level) - 1:
                        tmp_drawer_position_x = 0
                        if j == 1:
                            tmp_drawer_position_x = (
                                body_parameter["storagefurniture_layers_params"][5 + 3]
                                + body_parameter["storagefurniture_layers_params"][
                                    5 + 1
                                ]
                            )
                        elif j == 2:
                            tmp_drawer_position_x = (
                                body_parameter["storagefurniture_layers_params"][5 + 3]
                                + body_parameter["storagefurniture_layers_params"][
                                    5 + 4
                                ]
                                + 2
                                * body_parameter["storagefurniture_layers_params"][
                                    5 + 1
                                ]
                            )
                        x_width = (
                            body_parameter["size"][0]
                            - 2 * body_parameter["left_right_inner_size"][0]
                            - tmp_drawer_position_x
                        )
                        x_offset = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + tmp_drawer_position_x
                            + x_width / 2
                        )
                    elif j == 1:
                        x_width = body_parameter["storagefurniture_layers_params"][
                            5 + 4
                        ]
                        x_offset = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + body_parameter["storagefurniture_layers_params"][5 + 3]
                            + body_parameter["storagefurniture_layers_params"][5 + 1]
                            + body_parameter["storagefurniture_layers_params"][5 + 4]
                            / 2
                        )

                    for k in range(num_of_compartment_this_level[j]):
                        if occupancy_of_each_compartment[j][k] == 0:
                            parameter["number_of_drawer"][0] += 1
                            parameter["drawers_params"][20 * cur + 0] = (
                                x_width * randRange(1, 0.7, 0.95)[0]
                            )
                            parameter["drawers_params"][20 * cur + 2] = (
                                body_parameter["size"][2]
                                - body_parameter["back_size"][0]
                            ) * randRange(1, 0.7, 0.9)[0]
                            parameter["drawers_params"][20 * cur + 1] = (
                                average_remained_height * randRange(1, 0.8, 0.85)[0]
                            )
                            parameter["drawers_params"][20 * cur + 3] = (
                                average_remained_height
                                - parameter["drawers_params"][20 * cur + 1]
                            ) / 3
                            parameter["drawers_params"][20 * cur + 4] = x_width
                            parameter["drawers_params"][20 * cur + 5] = (
                                average_remained_height
                            )
                            parameter["drawers_params"][20 * cur + 6] = (
                                parameter["drawers_params"][6]
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["drawers_params"][20 * cur + 7] = 0
                            parameter["drawers_params"][20 * cur + 8] = (
                                body_parameter["left_right_inner_size"][0]
                                * randRange(1, 0.7, 0.8)[0]
                            )
                            parameter["drawers_params"][20 * cur + 9] = (
                                body_parameter["back_size"][0]
                                * randRange(1, 0.7, 0.9)[0]
                            )
                            parameter["drawers_params"][20 * cur + 10] = int(1)
                            parameter["drawers_params"][20 * cur + 11] = (
                                parameter["drawers_params"][20 * cur + 4]
                                / 3
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["drawers_params"][20 * cur + 12] = (
                                parameter["drawers_params"][20 * cur + 5]
                                / 20
                                * randRange(1, 0.8, 1.2)[0]
                            )
                            parameter["drawers_params"][20 * cur + 13] = (
                                parameter["drawers_params"][20 * cur + 6]
                                * randRange(1, 0.9, 1.1)[0]
                            )
                            parameter["drawers_params"][20 * cur + 14] = 0
                            parameter["drawers_params"][20 * cur + 15] = (
                                parameter["drawers_params"][20 * cur + 5]
                                / 4
                                * randRange(1, 0.5, 0.9)[0]
                            )
                            parameter["drawers_params"][20 * cur + 16] = 0
                            parameter["drawers_params"][20 * cur + 17] = x_offset
                            parameter["drawers_params"][20 * cur + 18] = (
                                body_parameter["size"][1] / 2
                                - parameter["drawers_params"][20 * cur + 5] / 2
                                - record_height
                            )
                            parameter["drawers_params"][20 * cur + 19] = (
                                body_parameter["size"][2]
                                - parameter["drawers_params"][20 * cur + 2]
                            ) / 2 + parameter["drawers_params"][
                                20 * cur + 2
                            ] * randRange(1, 0.2, 0.6)[0]

                            record_height += (
                                average_remained_height
                                + body_parameter["WHOLE_layer_sizes"][0]
                            )
                            cur += 1

            elif storagefurniture_type == "multi_drawer_Cabinet":
                parameter["number_of_drawer"][0] = (
                    body_parameter["WHOLE_number_of_layer"][0] + 1
                )
                parameter["number_of_drawer"] = np.array(
                    [int(num) for num in parameter["number_of_drawer"]]
                )
                num_of_drawer = parameter["number_of_drawer"][0]
                x_size = (
                    body_parameter["size"][0]
                    - body_parameter["left_right_inner_size"][0] * 2
                )
                z_size = body_parameter["size"][2] - body_parameter["back_size"][0]
                for i in range(num_of_drawer):
                    if i == 0:
                        y_size = (
                            body_parameter["WHOLE_layer_offset"][0]
                            - body_parameter["WHOLE_layer_sizes"][0]
                        )
                        parameter["drawers_params"][i * 20 + 0] = (
                            x_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][i * 20 + 1] = (
                            y_size * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["drawers_params"][i * 20 + 2] = (
                            z_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][i * 20 + 3] = (
                            y_size - parameter["drawers_params"][20 * i + 1]
                        ) / 3
                        parameter["drawers_params"][i * 20 + 4] = np.maximum(
                            body_parameter["size"][0] * randRange(1, 0.8, 1.0)[0],
                            x_size,
                        )
                        parameter["drawers_params"][i * 20 + 5] = y_size
                        parameter["drawers_params"][i * 20 + 6] = (
                            body_parameter["back_size"][0] * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][i * 20 + 7] = 0
                        parameter["drawers_params"][i * 20 + 8] = (
                            body_parameter["left_right_inner_size"][0]
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][i * 20 + 9] = (
                            body_parameter["back_size"][0] * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][i * 20 + 10] = int(1)
                        parameter["drawers_params"][20 * i + 11] = (
                            parameter["drawers_params"][20 * i + 4]
                            / 3
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * i + 12] = (
                            parameter["drawers_params"][20 * i + 5]
                            / 20
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * i + 13] = (
                            parameter["drawers_params"][20 * i + 6]
                            * randRange(1, 0.9, 1.1)[0]
                        )
                        parameter["drawers_params"][20 * i + 14] = 0
                        parameter["drawers_params"][20 * i + 15] = (
                            parameter["drawers_params"][20 * i + 5]
                            / 4
                            * randRange(1, 0.5, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * i + 16] = 0
                        parameter["drawers_params"][20 * i + 17] = 0
                        parameter["drawers_params"][20 * i + 18] = (
                            body_parameter["size"][1] / 2
                            - parameter["drawers_params"][20 * i + 5] / 2
                        )
                        parameter["drawers_params"][20 * i + 19] = (
                            body_parameter["size"][2]
                            - parameter["drawers_params"][20 * i + 2]
                        ) / 2 + parameter["drawers_params"][20 * i + 2] * randRange(
                            1, 0.2, 0.6
                        )[0]
                    elif i == num_of_drawer - 1:
                        total_occupied_height = 0
                        for j in range(num_of_drawer - 2):
                            total_occupied_height += body_parameter[
                                "WHOLE_interval_between_layers"
                            ][j]
                        y_size = (
                            body_parameter["size"][1]
                            - body_parameter["WHOLE_layer_offset"][0]
                            - total_occupied_height
                            - body_parameter["base_size"][0]
                            - body_parameter["WHOLE_layer_sizes"][0]
                        )
                        parameter["drawers_params"][i * 20 + 0] = (
                            x_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][i * 20 + 1] = (
                            y_size * randRange(1, 0.8, 0.9)[0]
                        )
                        parameter["drawers_params"][i * 20 + 2] = (
                            z_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][i * 20 + 3] = (
                            y_size - parameter["drawers_params"][20 * i + 1]
                        ) / 3
                        parameter["drawers_params"][i * 20 + 4] = np.maximum(
                            body_parameter["size"][0] * randRange(1, 0.8, 1.0)[0],
                            x_size,
                        )
                        parameter["drawers_params"][i * 20 + 5] = y_size
                        parameter["drawers_params"][i * 20 + 6] = (
                            body_parameter["back_size"][0] * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][i * 20 + 7] = 0
                        parameter["drawers_params"][i * 20 + 8] = (
                            body_parameter["left_right_inner_size"][0]
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][i * 20 + 9] = (
                            body_parameter["back_size"][0] * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][i * 20 + 10] = int(1)
                        parameter["drawers_params"][20 * i + 11] = (
                            parameter["drawers_params"][20 * i + 4]
                            / 3
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * i + 12] = (
                            parameter["drawers_params"][20 * i + 5]
                            / 20
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * i + 13] = (
                            parameter["drawers_params"][20 * i + 6]
                            * randRange(1, 0.9, 1.1)[0]
                        )
                        parameter["drawers_params"][20 * i + 14] = 0
                        parameter["drawers_params"][20 * i + 15] = (
                            parameter["drawers_params"][20 * i + 5]
                            / 4
                            * randRange(1, 0.5, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * i + 16] = 0
                        parameter["drawers_params"][20 * i + 17] = 0
                        parameter["drawers_params"][20 * i + 18] = (
                            -body_parameter["size"][1] / 2
                            + body_parameter["base_size"][0]
                            + parameter["drawers_params"][20 * i + 5] / 2
                        )
                        parameter["drawers_params"][20 * i + 19] = (
                            body_parameter["size"][2]
                            - parameter["drawers_params"][20 * i + 2]
                        ) / 2 + parameter["drawers_params"][20 * i + 2] * randRange(
                            1, 0.2, 0.6
                        )[0]
                    else:
                        total_occupied_height = 0
                        for j in range(i):
                            if j == 0:
                                total_occupied_height += body_parameter[
                                    "WHOLE_layer_offset"
                                ][0]
                            else:
                                total_occupied_height += body_parameter[
                                    "WHOLE_interval_between_layers"
                                ][j - 1]
                        y_size = (
                            body_parameter["WHOLE_interval_between_layers"][i - 1]
                            - body_parameter["WHOLE_layer_sizes"][0]
                        )
                        parameter["drawers_params"][i * 20 + 0] = (
                            x_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][i * 20 + 1] = (
                            y_size * randRange(1, 0.8, 0.85)[0]
                        )
                        parameter["drawers_params"][i * 20 + 2] = (
                            z_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][i * 20 + 3] = (
                            y_size - parameter["drawers_params"][20 * i + 1]
                        ) / 3
                        parameter["drawers_params"][i * 20 + 4] = np.maximum(
                            body_parameter["size"][0] * randRange(1, 0.8, 1.0)[0],
                            x_size,
                        )
                        parameter["drawers_params"][i * 20 + 5] = y_size
                        parameter["drawers_params"][i * 20 + 6] = (
                            body_parameter["back_size"][0] * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][i * 20 + 7] = 0
                        parameter["drawers_params"][i * 20 + 8] = (
                            body_parameter["left_right_inner_size"][0]
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][i * 20 + 9] = (
                            body_parameter["back_size"][0] * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][i * 20 + 10] = int(1)
                        parameter["drawers_params"][20 * i + 11] = (
                            parameter["drawers_params"][20 * i + 4]
                            / 3
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * i + 12] = (
                            parameter["drawers_params"][20 * i + 5]
                            / 20
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * i + 13] = (
                            parameter["drawers_params"][20 * i + 6]
                            * randRange(1, 0.9, 1.1)[0]
                        )
                        parameter["drawers_params"][20 * i + 14] = 0
                        parameter["drawers_params"][20 * i + 15] = (
                            parameter["drawers_params"][20 * i + 5]
                            / 4
                            * randRange(1, 0.5, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * i + 16] = 0
                        parameter["drawers_params"][20 * i + 17] = 0
                        parameter["drawers_params"][20 * i + 18] = (
                            body_parameter["size"][1] / 2
                            - total_occupied_height
                            - parameter["drawers_params"][20 * i + 5] / 2
                        )
                        parameter["drawers_params"][20 * i + 19] = (
                            body_parameter["size"][2]
                            - parameter["drawers_params"][20 * i + 2]
                        ) / 2 + parameter["drawers_params"][20 * i + 2] * randRange(
                            1, 0.2, 0.6
                        )[0]

            elif storagefurniture_type == "Multi_layer_wardrobe":
                num_of_layer = body_parameter["WHOLE_number_of_layer"][0] - 2
                num_of_drawer_level = num_of_layer + 1
                num_of_drawer_each_level = [
                    int(
                        body_parameter["storagefurniture_layers_params"][(i + 2) * 5]
                        + 1
                    )
                    for i in range(num_of_drawer_level)
                ]
                parameter["number_of_drawer"][0] = 0
                cur = 0
                for i in range(num_of_drawer_level):
                    if i == num_of_drawer_level - 1:
                        y_size = (
                            body_parameter["size"][1]
                            - body_parameter["base_size"][0]
                            - body_parameter["WHOLE_layer_offset"][0]
                            - body_parameter["WHOLE_interval_between_layers"][i]
                            * (i + 1)
                        ) - body_parameter["WHOLE_layer_sizes"][0]
                    else:
                        y_size = (
                            body_parameter["WHOLE_interval_between_layers"][i + 1]
                            * (i + 2)
                            - body_parameter["WHOLE_interval_between_layers"][0]
                        ) / (i + 1) - body_parameter["WHOLE_layer_sizes"][0]
                    z_size = body_parameter["size"][2] - body_parameter["back_size"][0]
                    total_occupied_height = (
                        body_parameter["WHOLE_layer_offset"][0]
                        + (i + 1) * body_parameter["WHOLE_interval_between_layers"][i]
                        + body_parameter["WHOLE_layer_sizes"][0] / 2
                    )
                    for j in range(num_of_drawer_each_level[i]):
                        if j == 0:
                            x_size = body_parameter["storagefurniture_layers_params"][
                                (i + 2) * 5 + 3
                            ]
                            x_offset = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                            )
                        elif j == num_of_drawer_each_level[i] - 1:
                            x_size = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    (i + 2) * 5 + 3
                                ]
                                - (num_of_drawer_each_level[i] - 2)
                                * body_parameter["storagefurniture_layers_params"][
                                    (i + 2) * 5 + 4
                                ]
                            )
                            x_offset = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + body_parameter["storagefurniture_layers_params"][
                                    (i + 2) * 5 + 3
                                ]
                                + (num_of_drawer_each_level[i] - 2)
                                * body_parameter["storagefurniture_layers_params"][
                                    (i + 2) * 5 + 4
                                ]
                            )
                        else:
                            x_size = body_parameter["storagefurniture_layers_params"][
                                (i + 2) * 5 + 4
                            ]
                            x_offset = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + body_parameter["storagefurniture_layers_params"][
                                    (i + 2) * 5 + 3
                                ]
                                + (j - 1)
                                * body_parameter["storagefurniture_layers_params"][
                                    (i + 2) * 5 + 4
                                ]
                            )
                        parameter["drawers_params"][cur * 20 + 0] = (
                            x_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 1] = (
                            y_size * randRange(1, 0.8, 0.85)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 2] = (
                            z_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 3] = (
                            y_size - parameter["drawers_params"][20 * cur + 1]
                        ) / 3
                        parameter["drawers_params"][cur * 20 + 4] = x_size
                        parameter["drawers_params"][cur * 20 + 5] = y_size
                        parameter["drawers_params"][cur * 20 + 6] = (
                            body_parameter["back_size"][0] * randRange(1, 0.5, 0.8)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 7] = 0
                        parameter["drawers_params"][cur * 20 + 8] = (
                            x_size / 10 * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 9] = (
                            body_parameter["back_size"][0] * randRange(1, 0.5, 0.8)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 10] = int(1)
                        parameter["drawers_params"][20 * cur + 11] = (
                            parameter["drawers_params"][20 * cur + 4]
                            / 3
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 12] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 20
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 13] = (
                            parameter["drawers_params"][20 * cur + 6]
                            * randRange(1, 0.9, 1.1)[0]
                        )
                        parameter["drawers_params"][20 * cur + 14] = 0
                        parameter["drawers_params"][20 * cur + 15] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 4
                            * randRange(1, 0.5, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * cur + 16] = 0
                        parameter["drawers_params"][20 * cur + 17] = (
                            x_offset + parameter["drawers_params"][cur * 20 + 4] / 2
                        )
                        parameter["drawers_params"][20 * cur + 18] = (
                            body_parameter["size"][1] / 2
                            - total_occupied_height
                            - parameter["drawers_params"][20 * cur + 5] / 2
                        )
                        parameter["drawers_params"][20 * cur + 19] = (
                            body_parameter["size"][2]
                            - parameter["drawers_params"][20 * cur + 2]
                        ) / 2 + parameter["drawers_params"][20 * cur + 2] * randRange(
                            1, 0.2, 0.6
                        )[0]
                        cur += 1
                        parameter["number_of_drawer"][0] += 1
                parameter["number_of_drawer"] = np.array(
                    [int(num) for num in parameter["number_of_drawer"]]
                )

            elif storagefurniture_type == "Cabinet_vertical":
                number_of_drawer = 0
                length_of_compartment_list = len(num_of_additional_layers)
                drawer_existence = [0 for _ in range(length_of_compartment_list)]
                for j in range(length_of_compartment_list):
                    if (
                        num_of_additional_layers[j] == 3
                        or num_of_additional_layers[j] == 2
                    ):
                        number_of_drawer += 1
                        drawer_existence[j] = 1
                parameter["number_of_drawer"][0] = number_of_drawer
                parameter["number_of_drawer"] = np.array(
                    [int(num) for num in parameter["number_of_drawer"]]
                )
                cur = 0
                for i in range(length_of_compartment_list):
                    if drawer_existence[i] == 1:
                        y_size = (
                            body_parameter["size"][1]
                            + body_parameter["lid_offset"][1]
                            - body_parameter["base_size"][0]
                            - body_parameter["WHOLE_layer_sizes"][0] / 2
                        ) / (num_of_additional_layers[i] + 1)
                        z_size = (
                            body_parameter["size"][2] - body_parameter["back_size"][0]
                        )
                        if i == 0:
                            x_size = body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 3
                            ]
                            x_offset = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                            )
                        elif i == length_of_compartment_list - 1:
                            x_size = (
                                body_parameter["size"][0]
                                - 2 * body_parameter["left_right_inner_size"][0]
                                - body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 3
                                ]
                                - (length_of_compartment_list - 2)
                                * body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 4
                                ]
                            )
                            x_offset = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 3
                                ]
                                + (length_of_compartment_list - 2)
                                * body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 4
                                ]
                            )
                        else:
                            x_size = body_parameter["storagefurniture_layers_params"][
                                0 * 5 + 4
                            ]
                            x_offset = (
                                -body_parameter["size"][0] / 2
                                + body_parameter["left_right_inner_size"][0]
                                + body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 3
                                ]
                                + (i - 1)
                                * body_parameter["storagefurniture_layers_params"][
                                    0 * 5 + 4
                                ]
                            )

                        parameter["drawers_params"][cur * 20 + 0] = (
                            x_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 1] = (
                            y_size * randRange(1, 0.8, 0.85)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 2] = (
                            z_size * randRange(1, 0.7, 0.95)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 3] = (
                            y_size - parameter["drawers_params"][20 * cur + 1]
                        ) / 3
                        parameter["drawers_params"][cur * 20 + 4] = x_size
                        parameter["drawers_params"][cur * 20 + 5] = y_size
                        parameter["drawers_params"][cur * 20 + 6] = (
                            body_parameter["back_size"][0] * randRange(1, 0.5, 0.8)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 7] = 0
                        parameter["drawers_params"][cur * 20 + 8] = (
                            x_size / 10 * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 9] = (
                            body_parameter["back_size"][0] * randRange(1, 0.5, 0.8)[0]
                        )
                        parameter["drawers_params"][cur * 20 + 10] = int(1)
                        parameter["drawers_params"][20 * cur + 11] = (
                            parameter["drawers_params"][20 * cur + 4]
                            / 3
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 12] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 20
                            * randRange(1, 0.8, 1.2)[0]
                        )
                        parameter["drawers_params"][20 * cur + 13] = (
                            parameter["drawers_params"][20 * cur + 6]
                            * randRange(1, 0.9, 1.1)[0]
                        )
                        parameter["drawers_params"][20 * cur + 14] = 0
                        parameter["drawers_params"][20 * cur + 15] = (
                            parameter["drawers_params"][20 * cur + 5]
                            / 4
                            * randRange(1, 0.5, 0.9)[0]
                        )
                        parameter["drawers_params"][20 * cur + 16] = 0
                        parameter["drawers_params"][20 * cur + 17] = (
                            x_offset + parameter["drawers_params"][cur * 20 + 4] / 2
                        )
                        parameter["drawers_params"][20 * cur + 18] = (
                            body_parameter["size"][1] / 2
                            + body_parameter["lid_offset"][1] / 2
                            - parameter["drawers_params"][20 * cur + 5] / 2
                        )
                        parameter["drawers_params"][20 * cur + 19] = (
                            body_parameter["size"][2]
                            - parameter["drawers_params"][20 * cur + 2]
                        ) / 2 + parameter["drawers_params"][20 * cur + 2] * randRange(
                            1, 0.2, 0.6
                        )[0]
                        cur += 1

            concept["parameters"] = {
                k: v.tolist()
                if k
                not in [
                    "storagefurniture_layers_params",
                    "additional_layers_params",
                    "drawers_params",
                    "additional_legs_params",
                ]
                else v
                for k, v in parameter.items()
            }
            if concept["parameters"]["number_of_drawer"][0] > 0:
                new_concepts.append(concept)

        elif template == "Regular_front_panel":
            body_parameter = concepts[0]["parameters"]
            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            if storagefurniture_type == "panel_Cabinet":
                parameter["number_of_frontPanel"][0] = 1
                parameter["number_of_frontPanel"] = np.array(
                    [int(num) for num in parameter["number_of_frontPanel"]]
                )
                parameter["frontPanel_params"][0] = (
                    body_parameter["size"][0]
                    - body_parameter["left_right_inner_size"][0] * 2
                )
                parameter["frontPanel_params"][1] = (
                    body_parameter["WHOLE_layer_offset"][0]
                    / 2
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["frontPanel_params"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["frontPanel_params"][3] = 0
                parameter["frontPanel_params"][4] = (
                    parameter["frontPanel_params"][1] / 2
                    + body_parameter["size"][1] / 2
                    - body_parameter["WHOLE_layer_offset"][0]
                    - body_parameter["WHOLE_layer_sizes"][0]
                )
                parameter["frontPanel_params"][5] = (
                    body_parameter["size"][2] / 2
                    + parameter["frontPanel_params"][2] / 2
                )

            elif storagefurniture_type == "bookcase":
                number_of_panel = int(
                    body_parameter["storagefurniture_layers_params"][1 * 5] + 1
                )
                parameter["number_of_frontPanel"][0] = number_of_panel
                parameter["number_of_frontPanel"] = np.array(
                    [int(num) for num in parameter["number_of_frontPanel"]]
                )
                for i in range(number_of_panel):
                    if i == 0:
                        if i == number_of_panel - 1:
                            parameter["frontPanel_params"][6 * i + 0] = body_parameter[
                                "storagefurniture_layers_params"
                            ][1 * 5 + 3]
                        else:
                            parameter["frontPanel_params"][6 * i + 0] = (
                                body_parameter["storagefurniture_layers_params"][
                                    1 * 5 + 3
                                ]
                                - body_parameter["storagefurniture_layers_params"][
                                    1 * 5 + 1
                                ]
                            )
                        parameter["frontPanel_params"][6 * i + 1] = body_parameter[
                            "WHOLE_interval_between_layers"
                        ][0]
                        parameter["frontPanel_params"][6 * i + 2] *= randRange(
                            1, 0.7, 0.9
                        )[0]
                        parameter["frontPanel_params"][6 * i + 3] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + parameter["frontPanel_params"][6 * i + 0] / 2
                        )
                        parameter["frontPanel_params"][6 * i + 4] = (
                            parameter["frontPanel_params"][1] / 2
                            + body_parameter["size"][1] / 2
                            - body_parameter["WHOLE_layer_offset"][0]
                            - body_parameter["WHOLE_interval_between_layers"][0]
                        )
                        parameter["frontPanel_params"][6 * i + 5] = (
                            body_parameter["size"][2] / 2
                            + parameter["frontPanel_params"][6 * i + 2] / 2
                        )
                    elif i == number_of_panel - 1:
                        parameter["frontPanel_params"][6 * i + 0] = (
                            body_parameter["storagefurniture_layers_params"][1 * 5 + 4]
                            - body_parameter["storagefurniture_layers_params"][
                                1 * 5 + 1
                            ]
                        )
                        parameter["frontPanel_params"][6 * i + 1] = body_parameter[
                            "WHOLE_interval_between_layers"
                        ][0]
                        parameter["frontPanel_params"][6 * i + 2] *= randRange(
                            1, 0.7, 0.9
                        )[0]
                        parameter["frontPanel_params"][6 * i + 3] = (
                            body_parameter["size"][0] / 2
                            - body_parameter["left_right_inner_size"][0]
                            - parameter["frontPanel_params"][6 * i + 0] / 2
                        )
                        parameter["frontPanel_params"][6 * i + 4] = (
                            parameter["frontPanel_params"][1] / 2
                            + body_parameter["size"][1] / 2
                            - body_parameter["WHOLE_layer_offset"][0]
                            - body_parameter["WHOLE_interval_between_layers"][0]
                        )
                        parameter["frontPanel_params"][6 * i + 5] = (
                            body_parameter["size"][2] / 2
                            + parameter["frontPanel_params"][6 * i + 2] / 2
                        )
                    else:
                        parameter["frontPanel_params"][6 * i + 0] = (
                            body_parameter["storagefurniture_layers_params"][1 * 5 + 4]
                            - body_parameter["storagefurniture_layers_params"][
                                1 * 5 + 1
                            ]
                        )
                        parameter["frontPanel_params"][6 * i + 1] = body_parameter[
                            "WHOLE_interval_between_layers"
                        ][0]
                        parameter["frontPanel_params"][6 * i + 2] *= randRange(
                            1, 0.7, 0.9
                        )[0]
                        parameter["frontPanel_params"][6 * i + 3] = (
                            -body_parameter["size"][0] / 2
                            + body_parameter["left_right_inner_size"][0]
                            + body_parameter["storagefurniture_layers_params"][
                                1 * 5 + 3
                            ]
                            + body_parameter["storagefurniture_layers_params"][
                                1 * 5 + 4
                            ]
                            * (i - 1)
                            + parameter["frontPanel_params"][6 * i + 0] / 2
                        )
                        parameter["frontPanel_params"][6 * i + 4] = (
                            parameter["frontPanel_params"][1] / 2
                            + body_parameter["size"][1] / 2
                            - body_parameter["WHOLE_layer_offset"][0]
                            - body_parameter["WHOLE_interval_between_layers"][0]
                        )
                        parameter["frontPanel_params"][6 * i + 5] = (
                            body_parameter["size"][2] / 2
                            + parameter["frontPanel_params"][6 * i + 2] / 2
                        )

            concept["parameters"] = {
                k: v.tolist()
                if k
                not in [
                    "storagefurniture_layers_params",
                    "additional_layers_params",
                    "drawers_params",
                    "additional_legs_params",
                ]
                else v
                for k, v in parameter.items()
            }
            new_concepts.append(concept)

    new_concepts = convert_door_params(new_concepts)

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
        storagefurniture_type = get_storagefurniture_type()
        existing_concept_templates = concept_template_existence(storagefurniture_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, storagefurniture_type)

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
            storagefurniture_type = get_storagefurniture_type()
            existing_concept_templates = concept_template_existence(
                storagefurniture_type
            )
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, storagefurniture_type)
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
