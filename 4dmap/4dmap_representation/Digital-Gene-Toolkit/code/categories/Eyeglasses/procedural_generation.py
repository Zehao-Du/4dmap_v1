import argparse
import copy
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


def get_eyeglasses_type():
    total_type = ["no_connector_square", "Round+F", "TrapezoidalFrame+F"]
    weights = [1, 1, 1]
    eyeglasses_type = random.choices(total_type, weights=weights, k=1)[0]
    return eyeglasses_type


def concept_template_existence(eyeglasses_type):
    if eyeglasses_type == "no_connector_square":
        glass_template = ["Trapezoidal_Glasses", "TrapezoidalFrame_Glasses"]
        leg_template = ["Regular_Leg", "Trifold_Leg"]
        connector_template = ["Standard_Connector", "Dual_Connector"]
        support_template = ["Standard_Support"]
        necessary = [1, 1, 0, 0.5]
    elif eyeglasses_type == "Round+F":
        glass_template = ["Round_Glasses", "RoundFrame_Glasses"]
        leg_template = ["Regular_Leg", "Trifold_Leg"]
        connector_template = ["Standard_Connector", "Dual_Connector"]
        support_template = ["Standard_Support"]
        necessary = [1, 1, 1, 0.5]
    else:
        glass_template = ["Trapezoidal_Glasses", "TrapezoidalFrame_Glasses"]
        leg_template = ["Regular_Leg", "Trifold_Leg"]
        connector_template = ["Standard_Connector", "Dual_Connector"]
        support_template = ["Standard_Support"]
        necessary = [1, 1, 1, 0.5]

    concept_template_variation = {
        "glass": {"template": glass_template, "necessary": necessary[0]},
        "leg": {"template": leg_template, "necessary": necessary[1]},
        "connector": {"template": connector_template, "necessary": necessary[2]},
        "support": {"template": support_template, "necessary": necessary[3]},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if not part["necessary"]:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))
        else:
            templates.append(random.choice(part["template"]))

    return templates


def get_elliptical_radius(a, b, angle):
    angle = np.abs(angle / 180 * np.pi)
    tmp1 = (np.sin(angle) ** 2) / (b**2)
    tmp2 = (np.cos(angle) ** 2) / (a**2)
    ans_radius = 1 / np.sqrt(tmp1 + tmp2)
    return ans_radius


def interval_reduction(a, b, angle):
    angle = np.abs(angle / 180 * np.pi)
    tmp_c = -np.sqrt((a**2) + ((b**2) * (np.tan(angle)) ** 2))
    tmp_lower = np.sqrt(np.tan(angle) ** 2 + 1)
    tmp_upper = np.abs(a + tmp_c)
    return tmp_upper / tmp_lower


def calculate_suitable_leg_angle(leg_length_1, leg_length_2, leg_angle_2):
    leg_angle_2 = leg_angle_2 / 180 * np.pi
    required_angle = (
        np.arctan(
            leg_length_2
            * np.sin(leg_angle_2)
            / (leg_length_1 + leg_length_2 * np.cos(leg_angle_2))
        )
        * 180
        / np.pi
    )
    return required_angle


def modify_templates_parameter_index(data, template_name, key, idx, new_value):
    for item in data:
        if item["template"] == template_name:
            item["parameters"][key][idx] = new_value
    # return data


def generate_random_angles(min_angle, max_angle, count):
    return [random.uniform(min_angle, max_angle) for _ in range(count)]


def generate_random_lengths(min_length, max_length, count):
    return [random.uniform(min_length, max_length) for _ in range(count)]


def get_new_parameters(data):
    # example:
    # templates_config = [
    #     {
    #         "template": "Regular_Leg",
    #         "modifications": [
    #             {"key": "rotation_1", "index": 0, "type": "angle", "bounds": [0, 90]},
    #             {"key": "rotation_1", "index": 1, "type": "length", "bounds": [0.1, 0.3]},
    #             {"key": "rotation_2", "index": 0, "type": "angle", "bounds": [0, 180]},
    #             {"key": "rotation_2", "index": 1, "type": "angle", "bounds": [-30, 30]},
    #         ]
    #     },
    #     {
    #         "template": "Trifold_Leg",
    #         "modifications": [
    #             {"key": "rotation_1", "index": 0, "type": "length", "bounds": [0.05, 0.2]},
    #             {"key": "rotation_2", "index": 2, "type": "angle", "bounds": [0, 90]},
    #         ]
    #     }
    # ]
    templates_config = [
        {
            "template": "Regular_Leg",
            "modifications": [
                {"key": "rotation_1", "index": 1, "type": "length", "bounds": [-90, 0]},
            ],
        },
        {
            "template": "Trifold_Leg",
            "modifications": [
                {"key": "rotation_1", "index": 1, "type": "length", "bounds": [-90, 0]},
            ],
        },
    ]

    for template_info in templates_config:
        temp_name = template_info["template"]
        for mod in template_info["modifications"]:
            key = mod["key"]
            idx = mod["index"]
            value_type = mod["type"]
            bounds = mod["bounds"]

            if value_type == "angle":
                new_value = generate_random_angles(bounds[0], bounds[1], 1)[0]
            elif value_type == "length":
                new_value = generate_random_lengths(bounds[0], bounds[1], 1)[0]
            else:
                raise ValueError(f"Unknown type: {value_type}")

            modify_templates_parameter_index(data, temp_name, key, idx, new_value)

    return data


def run_multiple_modifications(original_data, num_trials=10):
    results = []
    for _ in range(num_trials):
        data_copy = copy.deepcopy(original_data)
        modified_data = get_new_parameters(data_copy)
        results.append(modified_data)
    return results


def jitter_parameters(concepts, eyeglasses_type):
    new_concepts = []
    glass_type = ""

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if eyeglasses_type == "Round+F":
            if template == "Round_Glasses":
                glass_type = "Round_Glasses"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size"][0] *= randRange(1, 0.8, 1.2)
                parameter["size"][1] *= randRange(1, 0.8, 1.2)
                parameter["size"][2] *= randRange(1, 0.8, 1.2)
                parameter["interval"][0] = (
                    parameter["size"][0] * 2 / 3 * randRange(1, 0.8, 1.2)
                )
                parameter["glass_rotation"][0] = np.random.uniform(0, 15) * randRange(
                    1, 0.8, 1.2
                )
                if random.random() < 0.1:
                    parameter["glass_rotation"][0] = -12 * randRange(1, 0.8, 1.2)
                parameter["glass_rotation"][1] = np.random.uniform(-6, 15) * randRange(
                    1, 0.8, 1.2
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "RoundFrame_Glasses":
                glass_type = "RoundFrame_Glasses"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size"][0] *= randRange(1, 0.8, 1.2)
                parameter["size"][1] *= randRange(1, 0.8, 1.2)
                parameter["size"][2] *= randRange(1, 0.8, 1.2)
                parameter["interval"][0] = (
                    parameter["size"][0] / 2 * randRange(1, 0.8, 1.2)
                )
                parameter["glass_rotation"][0] = np.random.uniform(0, 15) * randRange(
                    1, 0.8, 1.2
                )
                parameter["width"][0] *= randRange(1, 0.8, 1.2)
                if random.random() < 0.1:
                    parameter["glass_rotation"][0] = -12 * randRange(1, 0.8, 1.2)
                parameter["glass_rotation"][1] = np.random.uniform(-6, 15) * randRange(
                    1, 0.8, 1.2
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_Leg":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                parameter["size1"][0] = (
                    glass_parameter["size"][0] / 6 * randRange(1, 0.8, 1.2)
                )
                parameter["size1"][1] = (
                    parameter["size1"][0] * 5 / 3 * randRange(1, 0.8, 1.2)
                )
                parameter["size1"][2] = (
                    glass_parameter["size"][0] * 2.5 * randRange(1, 0.8, 1.5)
                )

                parameter["size2"][0] = (
                    parameter["size1"][0]
                    * (7 / 6 if random.random() < 0.5 else 5 / 6)
                    * randRange(1, 0.8, 1.2)
                )
                parameter["size2"][1] = (
                    parameter["size1"][1]
                    * (3 / 5 if random.random() < 0.5 else 7 / 6)
                    * randRange(1, 0.8, 1.2)
                )
                parameter["size2"][2] = (
                    parameter["size1"][2] / 2 * randRange(1, 0.8, 1.2)
                )

                parameter["rotation_1"][0] *= randRange(1, 0.5, 1.5)
                parameter["rotation_1"][1] *= randRange(1, 0.5, 1.5)

                parameter["rotation_2"][0] *= randRange(1, 0.8, 1.2)
                parameter["rotation_2"][1] *= randRange(1, 0.8, 1.2)

                if random.random() < 0.5:
                    unreachable_angle = calculate_suitable_leg_angle(
                        parameter["size1"][0],
                        parameter["size2"][0],
                        -parameter["rotation_2"][1],
                    )
                    parameter["rotation_1"][1] = -(85 - unreachable_angle)
                    if glass_parameter["glass_rotation"][0] < 0:
                        parameter["rotation_1"][1] = -(
                            85 + glass_parameter["glass_rotation"][0]
                        )

                x_z_ratio = glass_parameter["size"][0] / glass_parameter["size"][1]
                if glass_type == "RoundFrame_Glasses":
                    radius_semi_minor = glass_parameter["size"][0]
                    radius_semi_major = radius_semi_minor * x_z_ratio
                else:
                    radius_semi_major = glass_parameter["size"][0]
                    radius_semi_minor = glass_parameter["size"][1]

                current_radius = get_elliptical_radius(
                    radius_semi_major,
                    radius_semi_minor,
                    glass_parameter["glass_rotation"][1],
                )
                current_interval_to_delete = interval_reduction(
                    radius_semi_major,
                    radius_semi_minor,
                    glass_parameter["glass_rotation"][1],
                )
                parameter["offset_x"][0] = (
                    glass_parameter["interval"][0] / 2 - current_interval_to_delete
                )
                parameter["glass_interval"][0] = current_radius * 4
                parameter["glass_interval"][0] *= np.cos(
                    glass_parameter["glass_rotation"][0] / 180 * np.pi
                )
                parameter["glass_interval"][0] += parameter["size1"][0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Trifold_Leg":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                parameter["size1"][0] = (
                    glass_parameter["size"][0] / 6 * randRange(1, 0.8, 1.2)
                )
                parameter["size1"][1] = (
                    parameter["size1"][0] * 5 / 3 * randRange(1, 0.8, 1.2)
                )
                parameter["size1"][2] = (
                    glass_parameter["size"][0] * 2.5 * randRange(1, 0.8, 1.5)
                )

                parameter["size2"][0] = (
                    parameter["size1"][0]
                    * (7 / 6 if random.random() < 0.5 else 5 / 6)
                    * randRange(1, 0.8, 1.2)
                )
                parameter["size2"][1] = (
                    parameter["size1"][1]
                    * (3 / 5 if random.random() < 0.5 else 7 / 6)
                    * randRange(1, 0.8, 1.2)
                )
                parameter["size2"][2] = (
                    parameter["size1"][2] / 2 * randRange(1, 0.8, 1.2)
                )

                parameter["rotation_1"][0] *= randRange(1, 0.5, 1.5)
                parameter["rotation_1"][1] *= randRange(1, 0.5, 1.5)

                parameter["rotation_2"][0] *= randRange(1, 0.8, 1.2)
                parameter["rotation_2"][1] *= randRange(1, 0.8, 1.2)

                if random.random() < 0.5:
                    unreachable_angle = calculate_suitable_leg_angle(
                        parameter["size1"][0],
                        parameter["size2"][0],
                        -parameter["rotation_2"][1],
                    )
                    parameter["rotation_1"][1] = -(90 - unreachable_angle)
                    if glass_parameter["glass_rotation"][0] < 0:
                        parameter["rotation_1"][1] = -(
                            85 + glass_parameter["glass_rotation"][0]
                        )

                parameter["connector_size"][0] = (
                    parameter["size1"][0] * 2 * randRange(1, 0.8, 1.2)
                )
                parameter["connector_size"][1] = parameter["size1"][1] * randRange(
                    1, 0.8, 1.2
                )
                parameter["connector_size"][2] = glass_parameter["size"][2] * randRange(
                    1, 0.8, 2.0
                )
                parameter["offset_x"][0] = 0
                x_z_ratio = glass_parameter["size"][0] / glass_parameter["size"][1]
                if glass_type == "RoundFrame_Glasses":
                    radius_semi_minor = glass_parameter["size"][0]
                    radius_semi_major = radius_semi_minor * x_z_ratio
                else:
                    radius_semi_major = glass_parameter["size"][0]
                    radius_semi_minor = glass_parameter["size"][1]

                current_radius = get_elliptical_radius(
                    radius_semi_major,
                    radius_semi_minor,
                    glass_parameter["glass_rotation"][1],
                )
                current_interval_to_delete = interval_reduction(
                    radius_semi_major,
                    radius_semi_minor,
                    glass_parameter["glass_rotation"][1],
                )
                parameter["glass_interval"][0] = current_radius * 4
                parameter["glass_interval"][0] *= np.cos(
                    glass_parameter["glass_rotation"][0] / 180 * np.pi
                )
                parameter["glass_interval"][0] += (
                    glass_parameter["interval"][0] - current_interval_to_delete * 2
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Standard_Connector":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                height_offset = glass_parameter["size"][1] / 3 * randRange(1, 0, 1.2)
                offset_angle = np.arcsin(height_offset / glass_parameter["size"][1])
                additional_dis = (1 - np.cos(offset_angle)) * glass_parameter["size"][1]
                total_interval = glass_parameter["interval"][0] + additional_dis
                parameter["position"][1] += height_offset

                parameter["size"][0] = total_interval * randRange(1, 1.0, 1.1)
                parameter["size"][1] = (
                    glass_parameter["size"][1] / 6 * randRange(1, 0.8, 1.2)
                )
                parameter["size"][2] = glass_parameter["size"][2] * randRange(
                    1, 1.0, 1.2
                )
                parameter["position"][2] += (
                    parameter["size"][2]
                    + np.sin(glass_parameter["glass_rotation"][0] * np.pi / 180)
                    * 2
                    * glass_parameter["size"][0]
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Dual_Connector":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                height_offset_1 = (
                    glass_parameter["size"][1] / 3 * randRange(1, 0.7, 1.0)
                )
                radius_1 = (glass_parameter["size"][1] + glass_parameter["size"][0]) / 2
                offset_angle_1 = np.arcsin(
                    height_offset_1
                    / ((glass_parameter["size"][1] + glass_parameter["size"][0]) / 2)
                )
                additional_dis = (1 - np.cos(offset_angle_1)) * radius_1 * 2
                total_interval_1 = glass_parameter["interval"][0] + additional_dis
                parameter["offset_1"][1] = height_offset_1
                parameter["offset_1"][0] = 0
                parameter["offset_1"][2] = glass_parameter["size"][2] * randRange(
                    1, -0.3, 0.3
                )

                height_offset_2 = (
                    glass_parameter["size"][1] / 3 * randRange(1, -1.0, 0.4)
                )
                radius_2 = (glass_parameter["size"][1] + glass_parameter["size"][0]) / 2
                offset_angle_2 = np.arcsin(
                    height_offset_2
                    / ((glass_parameter["size"][1] + glass_parameter["size"][0]) / 2)
                )
                additional_dis = (1 - np.cos(offset_angle_2)) * radius_2 * 2
                total_interval_2 = glass_parameter["interval"][0] + additional_dis
                parameter["offset_2"][1] = height_offset_2
                parameter["offset_2"][0] = 0
                parameter["offset_2"][2] = glass_parameter["size"][2] * randRange(
                    1, -0.3, 0.3
                )

                parameter["size_1"][0] = total_interval_1 * randRange(1, 1.0, 1.2)
                parameter["size_1"][1] = (
                    glass_parameter["size"][1] / 7 * randRange(1, 0.8, 1.2)
                )
                parameter["size_1"][2] = glass_parameter["size"][2] * randRange(
                    1, 0.8, 1.2
                )
                parameter["offset_1"][2] += (
                    parameter["size_1"][2]
                    + np.sin(glass_parameter["glass_rotation"][0] * np.pi / 180)
                    * 2
                    * glass_parameter["size"][0]
                )

                parameter["size_2"][0] = total_interval_2 * randRange(1, 1.0, 1.2)
                parameter["size_2"][1] = (
                    glass_parameter["size"][1] / 7 * randRange(1, 0.8, 1.2)
                )
                parameter["size_2"][2] = glass_parameter["size"][2] * randRange(
                    1, 0.8, 1.2
                )
                parameter["offset_2"][2] += (
                    parameter["size_2"][2]
                    + np.sin(glass_parameter["glass_rotation"][0] * np.pi / 180)
                    * 2
                    * glass_parameter["size"][0]
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Standard_Support":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                parameter["support_rotation"][0] = -32 * randRange(1, 0.8, 1.2)
                parameter["support_rotation"][1] = 50 * randRange(1, 0.8, 1.2)
                parameter["support_rotation"][2] = 5 * randRange(1, 0.8, 1.2)
                parameter["size"][0] = (
                    glass_parameter["size"][0] / 3 * randRange(1, 0.8, 1.2)
                )
                parameter["size"][1] = (
                    glass_parameter["size"][1] / 2.5 * randRange(1, 0.8, 1.2)
                )
                parameter["size"][2] = (
                    glass_parameter["size"][2] * 2 / 3 * randRange(1, 0.5, 1.2)
                )

                parameter["position"][2] += (
                    np.sin(glass_parameter["glass_rotation"][0] * np.pi / 180)
                    * 2
                    * glass_parameter["size"][0]
                )

                support_offset_0 = np.abs(
                    parameter["size"][0]
                    / 2
                    * np.sin(parameter["support_rotation"][1] / 180 * np.pi)
                )
                support_offset_1 = np.abs(
                    parameter["size"][1]
                    / 2
                    * np.sin(parameter["support_rotation"][0] / 180 * np.pi)
                )

                parameter["position"][2] -= support_offset_0 + support_offset_1

                parameter["position"][2] += glass_parameter["size"][2] * randRange(
                    1, 0.8, 0.9
                )

                parameter["offset_x"][0] = glass_parameter["interval"][0] / 2

                parameter["position"][1] -= (
                    glass_parameter["size"][1] / 3 * randRange(1, 0.5, 1.0)
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif (
            eyeglasses_type == "TrapezoidalFrame+F"
            or eyeglasses_type == "no_connector_square"
        ):
            if template == "Trapezoidal_Glasses":
                glass_type = "Trapezoidal_Glasses"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size"][0] *= randRange(1, 0.8, 1.2)
                parameter["size"][1] = (
                    parameter["size"][0] * 5 / 6 * randRange(1, 0.8, 1.2)
                )
                parameter["size"][2] = (
                    parameter["size"][1] * 17 / 30 * randRange(1, 0.8, 1.2)
                )
                parameter["size"][3] = (
                    parameter["size"][0] / 20 * randRange(1, 0.8, 1.2)
                )
                if eyeglasses_type == "no_connector_square":
                    parameter["size"][1] = (
                        parameter["size"][0] * 2 / 3 * randRange(1, 0.8, 1.2)
                    )
                parameter["interval"][0] = (
                    parameter["size"][0] / 5 * randRange(1, 0.8, 1.2)
                )
                parameter["glass_rotation"][0] = 9 * randRange(1, 0.5, 1.5)
                parameter["glass_rotation"][1] = 3 * randRange(1, 0.5, 2.0)
                parameter["top_offset"][0] = (
                    0
                    if random.random() < 0.5
                    else parameter["size"][0] / 22 * randRange(1, 0.8, 1.2)
                )

                if eyeglasses_type == "no_connector_square":
                    parameter["interval"][0] = 0

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "TrapezoidalFrame_Glasses":
                glass_type = "TrapezoidalFrame_Glasses"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size"][0] *= randRange(1, 0.8, 1.2)
                parameter["size"][1] = (
                    parameter["size"][0] * 25 / 33 * randRange(1, 0.8, 1.2)
                )
                parameter["size"][2] = (
                    parameter["size"][1] * 4 / 5 * randRange(1, 0.8, 1.3)
                )
                parameter["size"][3] *= randRange(1, 0.8, 1.2)
                if eyeglasses_type == "no_connector_square":
                    parameter["size"][1] = (
                        parameter["size"][0] * 2 / 3 * randRange(1, 0.8, 1.2)
                    )
                parameter["interval"][0] = (
                    parameter["size"][1] / 8 * randRange(1, 0.8, 1.2)
                )
                parameter["width"][0] = (
                    parameter["size"][1] / 12 * randRange(1, 0.8, 1.2)
                )
                parameter["glass_rotation"][0] = 8 * randRange(1, 0.8, 1.5)
                parameter["glass_rotation"][1] = 3 * randRange(1, 0.5, 2.0)
                parameter["top_offset"][0] = 0

                if eyeglasses_type == "no_connector_square":
                    parameter["interval"][0] = 0

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_Leg":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                parameter["size1"][0] = (
                    glass_parameter["size"][0] / 12 * randRange(1, 0.6, 1.2)
                )
                parameter["size1"][1] = (
                    parameter["size1"][0] * 5 / 3 * randRange(1, 0.5, 1.2)
                )
                parameter["size1"][2] = (
                    glass_parameter["size"][0] * 1.5 * randRange(1, 0.8, 1.2)
                )

                parameter["size2"][0] = (
                    parameter["size1"][0]
                    * (7 / 6 if random.random() < 0.5 else 5 / 6)
                    * randRange(1, 0.6, 1.2)
                )
                parameter["size2"][1] = (
                    parameter["size1"][1]
                    * (3 / 5 if random.random() < 0.5 else 7 / 6)
                    * randRange(1, 0.6, 1.2)
                )
                parameter["size2"][2] = (
                    parameter["size1"][2] / 2 * randRange(1, 0.8, 1.2)
                )

                parameter["rotation_1"][0] *= randRange(1, 0.5, 1.5)
                parameter["rotation_1"][1] *= randRange(1, 0.5, 1.5)

                parameter["rotation_2"][0] *= randRange(1, 0.8, 1.2)
                parameter["rotation_2"][1] *= randRange(1, 0.8, 1.2)

                current_leg_height = np.random.uniform(
                    0, glass_parameter["size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                )
                height_ratio_1 = (
                    glass_parameter["size"][2] / 2 - current_leg_height
                ) / glass_parameter["size"][2]
                height_ratio_2 = (
                    glass_parameter["size"][2] / 2 + current_leg_height
                ) / glass_parameter["size"][2]
                current_offset = (
                    glass_parameter["size"][0] * height_ratio_2
                    + glass_parameter["size"][1] * height_ratio_1
                )
                parameter["position"][1] += current_leg_height

                glass_half_interval = (
                    current_offset + glass_parameter["top_offset"][0] * height_ratio_2
                ) / 2 + glass_parameter["size"][0] / 2

                parameter["offset_x"][0] = glass_parameter["interval"][0] / 2
                parameter["glass_interval"][0] = glass_half_interval * 2
                parameter["glass_interval"][0] *= np.cos(
                    glass_parameter["glass_rotation"][0] / 180 * np.pi
                )

                if random.random() < 0.5:
                    unreachable_angle = calculate_suitable_leg_angle(
                        parameter["size1"][0],
                        parameter["size2"][0],
                        -parameter["rotation_2"][1],
                    )
                    parameter["rotation_1"][1] = -(85 - unreachable_angle)
                    if glass_parameter["glass_rotation"][0] < 0:
                        parameter["rotation_1"][1] = -(
                            85 + glass_parameter["glass_rotation"][0]
                        )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Trifold_Leg":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                parameter["size1"][0] = (
                    glass_parameter["size"][0] / 12 * randRange(1, 0.6, 1.2)
                )
                parameter["size1"][1] = (
                    parameter["size1"][0] * 4 / 3 * randRange(1, 0.5, 1.2)
                )
                parameter["size1"][2] = (
                    glass_parameter["size"][0] * 1.5 * randRange(1, 0.8, 1.2)
                )

                parameter["size2"][0] = (
                    parameter["size1"][0]
                    * (7 / 6 if random.random() < 0.5 else 5 / 6)
                    * randRange(1, 0.6, 1.2)
                )
                parameter["size2"][1] = (
                    parameter["size1"][1]
                    * (3 / 5 if random.random() < 0.5 else 7 / 6)
                    * randRange(1, 0.6, 1.2)
                )
                parameter["size2"][2] = (
                    parameter["size1"][2] / 2 * randRange(1, 0.8, 1.2)
                )

                parameter["rotation_1"][0] *= randRange(1, 0.5, 1.5)
                parameter["rotation_1"][1] *= randRange(1, 0.5, 1.5)

                parameter["rotation_2"][0] *= randRange(1, 0.8, 1.2)
                parameter["rotation_2"][1] *= randRange(1, 0.8, 1.2)

                parameter["connector_size"][0] = (
                    parameter["size1"][0] * 2 * randRange(1, 0.8, 1.1)
                )
                parameter["connector_size"][1] = parameter["size1"][1] * randRange(
                    1, 0.8, 1.2
                )
                parameter["connector_size"][2] = (
                    glass_parameter["size"][3] / 1.5 * randRange(1, 0.8, 2.0)
                )

                parameter["offset_x"][0] = 0

                current_leg_height = np.random.uniform(
                    0, glass_parameter["size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                )
                height_ratio_1 = (
                    glass_parameter["size"][2] / 2 - current_leg_height
                ) / glass_parameter["size"][2]
                height_ratio_2 = (
                    glass_parameter["size"][2] / 2 + current_leg_height
                ) / glass_parameter["size"][2]
                current_offset = (
                    glass_parameter["size"][0] * height_ratio_2
                    + glass_parameter["size"][1] * height_ratio_1
                )
                parameter["position"][1] += current_leg_height

                glass_half_interval = (
                    current_offset + glass_parameter["top_offset"][0] * height_ratio_2
                ) / 2 + glass_parameter["size"][0] / 2

                parameter["glass_interval"][0] = glass_half_interval * 2
                parameter["glass_interval"][0] *= np.cos(
                    glass_parameter["glass_rotation"][0] / 180 * np.pi
                )
                parameter["glass_interval"][0] += (
                    glass_parameter["interval"][0]
                    - parameter["connector_size"][0] * randRange(1, 0.8, 1.0)[0]
                )

                if random.random() < 0.5:
                    unreachable_angle = calculate_suitable_leg_angle(
                        parameter["size1"][0],
                        parameter["size2"][0],
                        -parameter["rotation_2"][1],
                    )
                    parameter["rotation_1"][1] = -(90 - unreachable_angle)
                    if glass_parameter["glass_rotation"][0] < 0:
                        parameter["rotation_1"][1] = -(
                            85 + glass_parameter["glass_rotation"][0]
                        )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Standard_Connector":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                upper_interval = glass_parameter["interval"][0]
                lower_interval = upper_interval + (
                    glass_parameter["size"][0] - glass_parameter["size"][1]
                )
                height_offset = glass_parameter["size"][2] * randRange(1, 0, 0.6)[0]
                height_ratio_1 = height_offset / glass_parameter["size"][2]
                height_ratio_2 = 1 - height_ratio_1
                total_interval = (
                    lower_interval * height_ratio_1 + upper_interval * height_ratio_2
                )
                total_glass_length = (
                    glass_parameter["size"][0] * height_ratio_2
                    + glass_parameter["size"][1] * height_ratio_1
                )
                parameter["position"][1] += (
                    glass_parameter["size"][2] / 2 - height_offset
                )

                parameter["size"][0] = total_interval * randRange(1, 1.0, 1.1)[0]
                parameter["size"][1] = (
                    glass_parameter["size"][2] / 6 * randRange(1, 0.6, 1.1)[0]
                )
                parameter["size"][2] = (
                    glass_parameter["size"][3] * randRange(1, 1.0, 1.2)[0]
                )
                parameter["position"][2] += (
                    parameter["size"][2]
                    + np.sin(glass_parameter["glass_rotation"][0] * np.pi / 180)
                    * total_glass_length
                )
                parameter["position"][1] -= parameter["size"][1] / 2

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Dual_Connector":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                upper_interval = glass_parameter["interval"][0]
                lower_interval = upper_interval + (
                    glass_parameter["size"][0] - glass_parameter["size"][1]
                )
                height_offset = glass_parameter["size"][2] * randRange(1, 0, 0.5)[0]
                height_ratio_1 = height_offset / glass_parameter["size"][2]
                height_ratio_2 = 1 - height_ratio_1
                total_interval = (
                    lower_interval * height_ratio_1 + upper_interval * height_ratio_2
                )
                total_glass_length = (
                    glass_parameter["size"][0] * height_ratio_2
                    + glass_parameter["size"][1] * height_ratio_1
                )
                parameter["offset_1"][1] = (
                    glass_parameter["size"][2] / 2 - height_offset
                )
                parameter["offset_1"][0] = 0
                parameter["size_1"][0] = total_interval * randRange(1, 1.0, 1.1)[0]
                parameter["size_1"][1] = (
                    glass_parameter["size"][2] / 6 * randRange(1, 0.6, 1.1)[0]
                )
                parameter["size_1"][2] = (
                    glass_parameter["size"][3] * randRange(1, 1.0, 1.2)[0]
                )
                parameter["offset_1"][2] = (
                    parameter["size_1"][2]
                    + np.sin(glass_parameter["glass_rotation"][0] * np.pi / 180)
                    * total_glass_length
                )
                parameter["offset_1"][1] -= parameter["size_1"][1] / 2

                upper_interval = glass_parameter["interval"][0]
                lower_interval = upper_interval + (
                    glass_parameter["size"][0] - glass_parameter["size"][1]
                )
                height_offset = glass_parameter["size"][2] * randRange(1, 0.6, 0.8)[0]
                height_ratio_1 = height_offset / glass_parameter["size"][2]
                height_ratio_2 = 1 - height_ratio_1
                total_interval = (
                    lower_interval * height_ratio_1 + upper_interval * height_ratio_2
                )
                total_glass_length = (
                    glass_parameter["size"][0] * height_ratio_2
                    + glass_parameter["size"][1] * height_ratio_1
                )
                parameter["offset_2"][1] = (
                    glass_parameter["size"][2] / 2 - height_offset
                )
                parameter["offset_2"][0] = 0
                parameter["size_2"][0] = total_interval * randRange(1, 1.0, 1.1)[0]
                parameter["size_2"][1] = (
                    glass_parameter["size"][2] / 6 * randRange(1, 0.6, 1.1)[0]
                )
                parameter["size_2"][2] = (
                    glass_parameter["size"][3] * randRange(1, 1.0, 1.2)[0]
                )
                parameter["offset_2"][2] = (
                    parameter["size_2"][2]
                    + np.sin(glass_parameter["glass_rotation"][0] * np.pi / 180)
                    * total_glass_length
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Standard_Support":
                glass_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = glass_parameter["position"][0]
                parameter["position"][1] = glass_parameter["position"][1]
                parameter["position"][2] = glass_parameter["position"][2]
                parameter["rotation"][0] = glass_parameter["rotation"][0]
                parameter["rotation"][1] = glass_parameter["rotation"][1]
                parameter["rotation"][2] = glass_parameter["rotation"][2]

                upper_interval = glass_parameter["interval"][0]
                lower_interval = upper_interval + (
                    glass_parameter["size"][0] - glass_parameter["size"][1]
                )
                height_offset = glass_parameter["size"][2] * randRange(1, 0.7, 0.8)[0]
                height_ratio_1 = height_offset / glass_parameter["size"][2]
                height_ratio_2 = 1 - height_ratio_1
                total_interval = (
                    lower_interval * height_ratio_1 + upper_interval * height_ratio_2
                )
                total_glass_length = (
                    glass_parameter["size"][0] * height_ratio_2
                    + glass_parameter["size"][1] * height_ratio_1
                )
                parameter["position"][1] += (
                    glass_parameter["size"][2] / 2 - height_offset
                )
                parameter["position"][2] += (
                    np.sin(glass_parameter["glass_rotation"][0] * np.pi / 180)
                    * total_glass_length
                )

                parameter["support_rotation"][0] = -32 * randRange(1, 0.8, 1.2)[0]
                parameter["support_rotation"][1] = 50 * randRange(1, 0.8, 1.2)[0]
                parameter["support_rotation"][2] = 5 * randRange(1, 0.8, 1.2)[0]
                parameter["size"][0] = (
                    glass_parameter["size"][0] / 7 * randRange(1, 0.5, 1.2)[0]
                )
                parameter["size"][1] = (
                    glass_parameter["size"][2] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = (
                    glass_parameter["size"][3] * 2 / 3 * randRange(1, 0.5, 1.2)[0]
                )

                support_offset_0 = np.abs(
                    parameter["size"][0]
                    / 2
                    * np.sin(parameter["support_rotation"][1] / 180 * np.pi)
                )
                support_offset_1 = np.abs(
                    parameter["size"][1]
                    / 2
                    * np.sin(parameter["support_rotation"][0] / 180 * np.pi)
                )

                parameter["position"][2] -= support_offset_0 + support_offset_1
                parameter["position"][2] += glass_parameter["size"][3]
                parameter["offset_x"][0] = total_interval / 2

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
        eyeglasses_type = get_eyeglasses_type()
        existing_concept_templates = concept_template_existence(eyeglasses_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, eyeglasses_type)

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
            eyeglasses_type = get_eyeglasses_type()
            existing_concept_templates = concept_template_existence(eyeglasses_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, eyeglasses_type)
            # print(new_concepts)

            new_list = run_multiple_modifications(new_concepts)

            for cur_concept in new_list:
                vertices, faces = get_overall_model(cur_concept)
                pointcloud = get_overall_pointcloud(vertices, faces)
                opt_pcd = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(pointcloud)
                )
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
