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


def get_laptop_type():
    total_type = ["regular"]
    weights = [1]
    laptop_type = random.choices(total_type, weights=weights, k=1)[0]
    return laptop_type


def concept_template_existence(laptop_type):
    concept_template_variation = {
        "base": {"template": ["Regular_Base"], "necessary": True},
        "screen": {"template": ["Regular_Screen"], "necessary": True},
        "connection": {
            "template": ["Cuboidal_Connector", "Cylindrical_Connector"],
            "necessary": False,
        },
    }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        elif random.random() < 0.5:
            templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, laptop_type):
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    if "Cuboidal_Connector" in templates:
        connector_type = "Cuboidal_Connector"
    elif "Cylindrical_Connector" in templates:
        connector_type = "Cylindrical_Connector"
    else:
        connector_type = "invisible_Connector"

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if connector_type == "invisible_Connector":
            if template == "Regular_Base":
                base_size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
                new_base_size = parameter["size"] * base_size_mul
                new_base_size[2] = new_base_size[0] * 0.6825 * randRange(1, 0.8, 1.2)[0]
                parameter["size"] = new_base_size
                if parameter["size"][1] > 0.1 * parameter["size"][0]:
                    parameter["size"][1] = (
                        0.04 * parameter["size"][0] * randRange(1, 0.8, 1.2)[0]
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_Screen":
                base_parameter = concepts[0]["parameters"]
                parameter["rotation"] = base_parameter["rotation"]
                parameter["position"] = base_parameter["position"]

                screen_size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
                new_screen_size = parameter["size"] * screen_size_mul
                new_screen_size[1] = (
                    base_parameter["size"][2] * randRange(1, 0.9, 1.0)[0]
                )
                new_screen_size[2] = (
                    base_parameter["size"][1] * randRange(1, 0.6, 0.8)[0]
                )
                if np.abs(new_screen_size[0] - base_parameter["size"][0]) > 0.02:
                    if new_screen_size[0] > base_parameter["size"][0]:
                        new_screen_size[0] = (
                            base_parameter["size"][0] + 0.01 * randRange(1, 0.5, 1.2)[0]
                        )
                    else:
                        new_screen_size[0] = (
                            base_parameter["size"][0] - 0.01 * randRange(1, 0.5, 1.2)[0]
                        )

                parameter["size"] = new_screen_size
                screen_type = np.random.randint(1, 3)
                if screen_type == 1:
                    parameter["offset"][0] = base_parameter["size"][1] / 2
                    parameter["screen_rotation"][0] = randRange(1, -90, 90)[0]
                    screen_rot_angles = [
                        x / 180 * np.pi for x in parameter["screen_rotation"]
                    ]
                    parameter["offset"][1] = (
                        new_screen_size[1] / 2 * np.sin(screen_rot_angles[0])
                        - (base_parameter["size"][2] / 2) * randRange(1, 0.8, 1.0)[0]
                    )
                elif screen_type == 2:
                    parameter["offset"][0] = (
                        base_parameter["size"][1] / 2 * randRange(1, 0, 0.7)[0]
                    )
                    parameter["screen_rotation"][0] = randRange(1, -90, 90)[0]
                    screen_rot_angles = [
                        x / 180 * np.pi for x in parameter["screen_rotation"]
                    ]
                    parameter["offset"][1] = (
                        new_screen_size[1] / 2 * np.sin(screen_rot_angles[0])
                        - (base_parameter["size"][2] / 2)
                    ) - (
                        np.cos(screen_rot_angles[0]) * new_screen_size[2] / 2
                    ) * randRange(1, 0.4, 0.6)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif connector_type == "Cuboidal_Connector":
            if template == "Regular_Base":
                base_size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
                new_base_size = parameter["size"] * base_size_mul
                new_base_size[2] = new_base_size[0] * 0.6825 * randRange(1, 0.8, 1.2)[0]
                parameter["size"] = new_base_size
                if parameter["size"][1] > 0.1 * parameter["size"][0]:
                    parameter["size"][1] = (
                        0.04 * parameter["size"][0] * randRange(1, 0.8, 1.2)[0]
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_Screen":
                base_parameter = concepts[0]["parameters"]
                parameter["rotation"] = base_parameter["rotation"]
                parameter["position"] = base_parameter["position"]

                screen_size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
                new_screen_size = parameter["size"] * screen_size_mul
                new_screen_size[1] = (
                    base_parameter["size"][2] * randRange(1, 0.9, 1.0)[0]
                )
                new_screen_size[2] = (
                    base_parameter["size"][1] * randRange(1, 0.6, 0.8)[0]
                )
                if np.abs(new_screen_size[0] - base_parameter["size"][0]) > 0.02:
                    if new_screen_size[0] > base_parameter["size"][0]:
                        new_screen_size[0] = (
                            base_parameter["size"][0] + 0.01 * randRange(1, 0.5, 1.2)[0]
                        )
                    else:
                        new_screen_size[0] = (
                            base_parameter["size"][0] - 0.01 * randRange(1, 0.5, 1.2)[0]
                        )

                parameter["size"] = new_screen_size
                screen_type = np.random.randint(1, 3)

                if screen_type == 1:
                    parameter["screen_rotation"][0] = randRange(1, -90, 90)[0]
                    screen_rot_angles = [
                        x / 180 * np.pi for x in parameter["screen_rotation"]
                    ]
                    parameter["offset"][0] = (
                        base_parameter["size"][1]
                        - base_parameter["size"][1] / 2
                        - np.abs(
                            new_screen_size[2]
                            / 2
                            * np.sin(screen_rot_angles[0])
                            * randRange(1, 1, 1.2)[0]
                        )
                    )
                    if parameter["screen_rotation"][0] > 0:
                        parameter["offset"][0] = (
                            base_parameter["size"][1]
                            - base_parameter["size"][1] / 2
                            + np.abs(
                                new_screen_size[2]
                                / 2
                                * np.sin(screen_rot_angles[0])
                                * randRange(1, 0.5, 1)[0]
                            )
                        )
                    parameter["offset"][1] = (
                        new_screen_size[1] / 2 * np.sin(screen_rot_angles[0])
                        - (base_parameter["size"][2] / 2) * randRange(1, 0.8, 1.0)[0]
                    )
                elif screen_type == 2:
                    parameter["screen_rotation"][0] = randRange(1, -90, 90)[0]
                    screen_rot_angles = [
                        x / 180 * np.pi for x in parameter["screen_rotation"]
                    ]
                    parameter["offset"][0] = (
                        base_parameter["size"][1]
                        - base_parameter["size"][1] / 2
                        + new_screen_size[2]
                        / 2
                        * np.sin(screen_rot_angles[0])
                        * randRange(1, 1.0, 2.0)[0]
                    )
                    parameter["offset"][1] = (
                        new_screen_size[1] / 2 * np.sin(screen_rot_angles[0])
                        - (base_parameter["size"][2] / 2)
                    ) - (
                        np.cos(screen_rot_angles[0]) * new_screen_size[2] / 2
                    ) * randRange(1, 1.2, 2)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cuboidal_Connector":
                parameter["number_of_connector"] = np.array(
                    parameter["number_of_connector"], dtype=int
                )
                base_parameter = concepts[0]["parameters"]
                screen_parameter = concepts[1]["parameters"]
                number_of_connector = np.random.randint(1, 4)

                connector_length = (
                    base_parameter["size"][0]
                    / 2
                    * randRange(1, 0.5, 2)[0]
                    / number_of_connector
                )
                connector_separate_dis = (
                    base_parameter["size"][0] - number_of_connector * connector_length
                )
                if number_of_connector > 1:
                    connector_separate_dis /= number_of_connector - 1
                for i in range(number_of_connector):
                    parameter["separation"][i] = connector_separate_dis

                if number_of_connector == 1:
                    parameter["offset"][0] = -connector_length / 2
                elif number_of_connector > 1:
                    parameter["offset"][0] = (
                        -(
                            connector_length * number_of_connector
                            + connector_separate_dis * (number_of_connector - 1)
                        )
                        / 2
                    )

                parameter["size"][0] = connector_length
                rotation_sign = np.random.choice([-1, 1])
                parameter["connector_rotation"][0] = (
                    screen_parameter["screen_rotation"][0]
                    + 1.5 * randRange(1, 0.5, 2)[0] * rotation_sign
                )
                if (
                    parameter["connector_rotation"][0] <= -90
                    or parameter["connector_rotation"][0] >= 90
                ):
                    parameter["connector_rotation"][0] = screen_parameter[
                        "screen_rotation"
                    ][0]

                cur_screen_rotation = [
                    x / 180 * np.pi for x in screen_parameter["screen_rotation"]
                ]

                half_length = 0
                half_width = 0

                if screen_parameter["offset"][1] < screen_parameter["size"][
                    1
                ] / 2 * np.sin(cur_screen_rotation[0]) - (
                    base_parameter["size"][2] / 2
                ) - screen_parameter["size"][2] / 2 * np.cos(cur_screen_rotation[0]):
                    connector_height_from_base = (
                        base_parameter["size"][1] - base_parameter["size"][1] / 2
                    )
                    connector_height_from_screen = screen_parameter["offset"][
                        0
                    ] - screen_parameter["size"][2] / 2 * np.sin(cur_screen_rotation[0])
                    lower_bound = min(
                        connector_height_from_base, connector_height_from_screen
                    )
                    upper_bound = max(
                        connector_height_from_base, connector_height_from_screen
                    )
                    connector_length_by_base = base_parameter["size"][2] / 2
                    connector_length_by_screen = (
                        -screen_parameter["offset"][1]
                        + screen_parameter["size"][1]
                        / 2
                        * np.sin(cur_screen_rotation[0])
                        - screen_parameter["size"][2]
                        / 2
                        * np.cos(cur_screen_rotation[0])
                    )

                    parameter["offset"][1] = np.random.uniform(lower_bound, upper_bound)
                    parameter["offset"][2] = -np.random.uniform(
                        connector_length_by_base, connector_length_by_screen
                    )

                    radius_limit_by_length = (
                        connector_length_by_screen - connector_length_by_base
                    )
                    radius_limit_by_height = upper_bound - lower_bound

                    half_width = np.sqrt(
                        radius_limit_by_length**2 + radius_limit_by_height**2
                    )

                    half_length = (
                        (upper_bound - lower_bound)
                        * np.cos(cur_screen_rotation[0])
                        * randRange(1, 1.0, 1.2)[0]
                    )
                    if half_length < 0.005:
                        half_length = (
                            half_width
                            * np.cos(cur_screen_rotation[0])
                            * randRange(1, 1.0, 1.2)[0]
                        )

                else:
                    if screen_parameter["offset"][0] <= base_parameter["size"][
                        1
                    ] - base_parameter["size"][1] / 2 + screen_parameter["size"][
                        2
                    ] / 2 * np.sin(cur_screen_rotation[0]):
                        if screen_parameter["screen_rotation"][0] < 0:
                            screen_height_position = (
                                base_parameter["size"][1]
                                - base_parameter["size"][1] / 2
                                - screen_parameter["offset"][0]
                            )
                            height_diff = (
                                screen_height_position
                                + screen_parameter["size"][2]
                                * np.sin(cur_screen_rotation[0])
                                / 2
                            )
                            connector_offset_x = (
                                height_diff * np.tan(cur_screen_rotation[0])
                                + screen_parameter["size"][2]
                                * np.cos(cur_screen_rotation[0])
                                / 2
                            )
                            half_width = (
                                screen_parameter["size"][2]
                                / 4
                                * randRange(1, 0.5, 0.8)[0]
                            )
                            half_length = (
                                screen_parameter["size"][1]
                                / 10
                                * randRange(1, 0.9, 1.5)[0]
                            )
                            parameter["offset"][2] = (
                                screen_parameter["offset"][1]
                                - screen_parameter["size"][1]
                                / 2
                                * np.sin(cur_screen_rotation[0])
                                + connector_offset_x
                            )
                            parameter["offset"][2] = np.maximum(
                                parameter["offset"][2], -base_parameter["size"][2] / 2
                            )
                            parameter["offset"][1] = (
                                base_parameter["size"][1]
                                - base_parameter["size"][1] / 2
                            )

                            parameter["offset"][1] += (
                                half_length / 2 * np.cos(cur_screen_rotation[0])
                            )
                            parameter["offset"][2] += (
                                half_length / 2 * np.sin(cur_screen_rotation[0])
                            )
                            half_width /= 2
                            half_length /= 2
                            parameter["connector_rotation"][0] = screen_parameter[
                                "screen_rotation"
                            ][0]
                        else:
                            parameter["offset"][2] = (
                                screen_parameter["offset"][1]
                                - screen_parameter["size"][1]
                                / 2
                                * np.sin(cur_screen_rotation[0])
                                + screen_parameter["size"][2]
                                * np.cos(cur_screen_rotation[0])
                                / 2
                            )
                            parameter["offset"][1] = (
                                base_parameter["size"][1]
                                - base_parameter["size"][1] / 2
                            )
                            half_length = (
                                screen_parameter["size"][2]
                                * np.sin(cur_screen_rotation[0])
                                * np.cos(cur_screen_rotation[0])
                            )

                            half_width = (
                                screen_parameter["size"][2]
                                / 3
                                * randRange(1, 0.6, 0.8)[0]
                            )
                            parameter["offset"][1] = (
                                base_parameter["size"][1]
                                - base_parameter["size"][1] / 2
                            )

                parameter["size"][1] = half_length * 2
                parameter["size"][2] = half_width * 2

                parameter["number_of_connector"][0] = int(number_of_connector)

                parameter["rotation"] = base_parameter["rotation"]
                parameter["position"] = base_parameter["position"]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif connector_type == "Cylindrical_Connector":
            if template == "Regular_Base":
                base_size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
                new_base_size = parameter["size"] * base_size_mul
                new_base_size[2] = new_base_size[0] * 0.6825 * randRange(1, 0.8, 1.2)[0]
                parameter["size"] = new_base_size
                if parameter["size"][1] > 0.1 * parameter["size"][0]:
                    parameter["size"][1] = (
                        0.04 * parameter["size"][0] * randRange(1, 0.8, 1.2)[0]
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_Screen":
                base_parameter = concepts[0]["parameters"]
                parameter["rotation"] = base_parameter["rotation"]
                parameter["position"] = base_parameter["position"]

                screen_size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
                new_screen_size = parameter["size"] * screen_size_mul
                new_screen_size[1] = (
                    base_parameter["size"][2] * randRange(1, 0.9, 1.0)[0]
                )
                new_screen_size[2] = (
                    base_parameter["size"][1] * randRange(1, 0.6, 0.8)[0]
                )
                if np.abs(new_screen_size[0] - base_parameter["size"][0]) > 0.02:
                    if new_screen_size[0] > base_parameter["size"][0]:
                        new_screen_size[0] = (
                            base_parameter["size"][0] + 0.01 * randRange(1, 0.5, 1.2)[0]
                        )
                    else:
                        new_screen_size[0] = (
                            base_parameter["size"][0] - 0.01 * randRange(1, 0.5, 1.2)[0]
                        )

                parameter["size"] = new_screen_size
                screen_type = np.random.randint(1, 3)

                if screen_type == 1:
                    parameter["screen_rotation"][0] = randRange(1, -90, 90)[0]
                    screen_rot_angles = [
                        x / 180 * np.pi for x in parameter["screen_rotation"]
                    ]
                    parameter["offset"][0] = (
                        base_parameter["size"][1] - base_parameter["size"][1] / 2
                    ) - np.abs(
                        new_screen_size[2]
                        / 2
                        * np.sin(screen_rot_angles[0])
                        * randRange(1, 0, 1.2)[0]
                    )
                    if parameter["screen_rotation"][0] > 0:
                        parameter["offset"][0] = (
                            base_parameter["size"][1] - base_parameter["size"][1] / 2
                        ) + np.abs(
                            new_screen_size[2]
                            / 2
                            * np.sin(screen_rot_angles[0])
                            * randRange(1, 0.5, 1)[0]
                        )
                    parameter["offset"][1] = (
                        new_screen_size[1] / 2 * np.sin(screen_rot_angles[0])
                        - (base_parameter["size"][2] / 2) * randRange(1, 0.8, 1.0)[0]
                    )

                elif screen_type == 2:
                    parameter["screen_rotation"][0] = randRange(1, -90, 90)[0]
                    screen_rot_angles = [
                        x / 180 * np.pi for x in parameter["screen_rotation"]
                    ]
                    parameter["offset"][0] = (
                        base_parameter["size"][1] / 2
                        + new_screen_size[2]
                        / 2
                        * np.sin(screen_rot_angles[0])
                        * randRange(1, 1.0, 1.2)[0]
                    )
                    parameter["offset"][1] = (
                        new_screen_size[1] / 2 * np.sin(screen_rot_angles[0])
                        - (base_parameter["size"][2] / 2)
                    ) - (
                        np.cos(screen_rot_angles[0]) * new_screen_size[2] / 2
                    ) * randRange(1, 1, 2)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cylindrical_Connector":
                only_one_connector = 0
                parameter["number_of_connector"] = np.array(
                    parameter["number_of_connector"], dtype=int
                )
                base_parameter = concepts[0]["parameters"]
                screen_parameter = concepts[1]["parameters"]
                number_of_connector = np.random.randint(1, 4)

                connector_length = (
                    base_parameter["size"][0]
                    / 2
                    * randRange(1, 0.5, 2)[0]
                    / number_of_connector
                )
                connector_separate_dis = (
                    base_parameter["size"][0] - number_of_connector * connector_length
                )
                if number_of_connector > 1:
                    connector_separate_dis /= number_of_connector - 1
                for i in range(number_of_connector):
                    parameter["separation"][i] = connector_separate_dis

                if number_of_connector == 1:
                    parameter["offset"][0] = -connector_length / 2
                elif number_of_connector > 1:
                    parameter["offset"][0] = (
                        -(
                            connector_length * number_of_connector
                            + connector_separate_dis * (number_of_connector - 1)
                        )
                        / 2
                    )

                parameter["size"][1] = connector_length

                screen_offset = screen_parameter["offset"]
                screen_rot_angles = [
                    x / 180 * np.pi for x in screen_parameter["screen_rotation"]
                ]
                radius = 0

                if screen_offset[1] < screen_parameter["size"][1] / 2 * np.sin(
                    screen_rot_angles[0]
                ) - (base_parameter["size"][2] / 2):
                    connector_height_from_base = (
                        base_parameter["size"][1] - base_parameter["size"][1] / 2
                    )
                    connector_height_from_screen = screen_parameter["offset"][
                        0
                    ] - screen_parameter["size"][2] / 2 * np.sin(screen_rot_angles[0])
                    lower_bound = min(
                        connector_height_from_base, connector_height_from_screen
                    )
                    upper_bound = max(
                        connector_height_from_base, connector_height_from_screen
                    )
                    connector_length_by_base = base_parameter["size"][2] / 2
                    connector_length_by_screen = (
                        -screen_parameter["offset"][1]
                        + screen_parameter["size"][1] / 2 * np.sin(screen_rot_angles[0])
                        - screen_parameter["size"][2] / 2 * np.cos(screen_rot_angles[0])
                    )

                    parameter["offset"][1] = np.random.uniform(lower_bound, upper_bound)
                    parameter["offset"][2] = -np.random.uniform(
                        connector_length_by_base, connector_length_by_screen
                    )
                    lower_limit_by_length = abs(lower_bound - parameter["offset"][1])
                    upper_limit_by_length = abs(upper_bound - parameter["offset"][1])
                    base_limit_by_height = abs(
                        -parameter["offset"][2] - connector_length_by_base
                    )
                    screen_limit_by_height = abs(
                        -parameter["offset"][2] - connector_length_by_screen
                    )
                    radius_limit_by_length = np.maximum(
                        lower_limit_by_length, upper_limit_by_length
                    )
                    radius_limit_by_height = np.maximum(
                        base_limit_by_height, screen_limit_by_height
                    )

                    radius = np.sqrt(
                        radius_limit_by_length * radius_limit_by_length
                        + radius_limit_by_height * radius_limit_by_height
                    )
                    radius *= randRange(1, 1.1, 1.2)[0]

                    if radius < 0.0005 and screen_parameter["screen_rotation"][0] <= 0:
                        parameter["offset"][1] = (
                            base_parameter["size"][1] / 2 * randRange(1, 0.05, 0.1)[0]
                        )
                        parameter["offset"][2] = screen_offset[1] - screen_parameter[
                            "size"
                        ][1] / 2 * np.sin(screen_rot_angles[0])
                        radius_limit_by_length = (
                            -base_parameter["size"][2] / 2 - parameter["offset"][2]
                        )
                        radius_limit_by_height = base_parameter["size"][1] / 2
                        radius = np.sqrt(
                            radius_limit_by_length * radius_limit_by_length
                            + radius_limit_by_height * radius_limit_by_height
                        )
                        parameter["size"][1] = screen_parameter["size"][0]
                        only_one_connector = 1
                        parameter["number_of_connector"][0] = 1

                else:
                    if screen_offset[0] <= (
                        base_parameter["size"][1]
                        - base_parameter["size"][1] / 2
                        - np.abs(
                            screen_parameter["size"][2]
                            / 2
                            * np.sin(screen_rot_angles[0])
                        )
                    ):
                        if screen_parameter["screen_rotation"][0] < 0:
                            screen_height_position = (
                                base_parameter["size"][1]
                                - base_parameter["size"][1] / 2
                            ) - screen_offset[0]
                            height_diff = (
                                screen_height_position
                                + screen_parameter["size"][2]
                                * np.sin(screen_rot_angles[0])
                                / 2
                            )
                            connector_offset_x = (
                                -height_diff * np.tan(screen_rot_angles[0])
                                - screen_parameter["size"][2]
                                * np.cos(screen_rot_angles[0])
                                / 2
                            )
                            parameter["offset"][2] = (
                                screen_offset[1]
                                - screen_parameter["size"][1]
                                / 2
                                * np.sin(screen_rot_angles[0])
                                - connector_offset_x
                            )
                            parameter["offset"][1] = (
                                base_parameter["size"][1]
                                - base_parameter["size"][1] / 2
                            )
                            radius = (
                                screen_parameter["size"][2]
                                / 3
                                * randRange(1, 0.8, 1.2)[0]
                            )
                        else:
                            parameter["offset"][2] = (
                                screen_offset[1]
                                - screen_parameter["size"][1]
                                / 2
                                * np.sin(screen_rot_angles[0])
                                + screen_parameter["size"][2]
                                * np.cos(screen_rot_angles[0])
                                / 2
                            )
                            parameter["offset"][1] = (
                                base_parameter["size"][1]
                                - base_parameter["size"][1] / 2
                            )
                            radius = (
                                screen_parameter["size"][2]
                                / 2
                                * randRange(1, 0.8, 1.2)[0]
                            )

                    elif screen_offset[0] > (
                        base_parameter["size"][1] - base_parameter["size"][1] / 2
                    ) - np.abs(
                        screen_parameter["size"][2] / 2 * np.sin(screen_rot_angles[0])
                    ) and screen_offset[0] <= (
                        base_parameter["size"][1] - base_parameter["size"][1] / 2
                    ) + np.abs(
                        screen_parameter["size"][2] / 2 * np.sin(screen_rot_angles[0])
                    ):
                        parameter["offset"][2] = (
                            screen_offset[1]
                            - screen_parameter["size"][1]
                            / 2
                            * np.sin(screen_rot_angles[0])
                            + screen_parameter["size"][2]
                            * np.cos(screen_rot_angles[0])
                            / 2
                        )
                        if screen_parameter["screen_rotation"][0] > 0:
                            parameter["offset"][2] = (
                                screen_offset[1]
                                - screen_parameter["size"][1]
                                / 2
                                * np.sin(screen_rot_angles[0])
                                - screen_parameter["size"][2]
                                * np.cos(screen_rot_angles[0])
                                / 2
                            )
                        parameter["offset"][1] = (
                            base_parameter["size"][1] - base_parameter["size"][1] / 2
                        )
                        radius = (
                            screen_parameter["size"][2] / 2 * randRange(1, 0.8, 1.2)[0]
                        )

                parameter["size"][0] = radius

                if only_one_connector == 0:
                    parameter["number_of_connector"][0] = int(number_of_connector)

                parameter["rotation"] = base_parameter["rotation"]
                parameter["position"] = base_parameter["position"]
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
        laptop_type = get_laptop_type()
        existing_concept_templates = concept_template_existence(laptop_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, laptop_type)

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
            laptop_type = get_laptop_type()
            existing_concept_templates = concept_template_existence(laptop_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, laptop_type)
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
