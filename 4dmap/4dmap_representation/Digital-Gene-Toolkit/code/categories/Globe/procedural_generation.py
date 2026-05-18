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


def get_globe_type():
    globe_type = ["Regular_Globe", "High_Globe", "Table_Globe", "Strange_Globe"]
    weights = [1, 1, 1, 1]
    globe_type = random.choices(globe_type, weights=weights, k=1)[0]
    return globe_type


def concept_template_existence(globe_type):
    sphere_template = ["Standard_Sphere"]

    if globe_type == "Regular_Globe":
        bracket_template = ["Semi_Ring_Bracket", "Tilted_Bracket", "Enclosed_Bracket"]
        base_template = ["Cylindrical_Base", "Cuboidal_Base"]

    elif globe_type == "High_Globe":
        bracket_template = ["Tilted_Bracket", "Enclosed_Bracket"]
        base_template = ["Star_Shaped_Base"]

    elif globe_type == "Table_Globe":
        bracket_template = ["Semi_Ring_Bracket", "Tilted_Bracket", "Enclosed_Bracket"]
        base_template = ["Table_Like_Base"]

    elif globe_type == "Strange_Globe":
        bracket_template = ["Semi_Ring_Bracket"]
        base_template = ["Special_Base"]

    necessary = [1, 1, 1]

    concept_template_variation = {
        "sphere": {"template": sphere_template, "necessary": necessary[0]},
        "bracket": {"template": bracket_template, "necessary": necessary[1]},
        "base": {"template": base_template, "necessary": necessary[2]},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"] == 0.5:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))
        elif part["necessary"] == 1:
            templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, globe_type):
    new_concepts = []
    bottom_half_ball_height = 0
    total_offset_distance = 0
    offset_x = 0

    rot_sign = 0
    lean_angle_tmp = 0

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if globe_type == "Regular_Globe":
            if template == "Standard_Sphere":
                parameter["radius"][0] *= randRange(1, 0.8, 1.2)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Semi_Ring_Bracket":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                is_tilt = 0 if random.random() < 0.4 else 1
                is_connected = 0 if random.random() < 0.5 else 1

                parameter["pivot_size"][0] = (
                    sphere_parameter["radius"][0] / 20 * randRange(1, 0.8, 1.2)[0]
                )

                sphere_bracket_distance = (
                    sphere_parameter["radius"][0] / 50 * randRange(1, 0.9, 1.3)[0]
                    if random.random() < 0.5
                    else sphere_parameter["radius"][0]
                    * 2
                    / 15
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][1] = (
                    sphere_bracket_distance + sphere_parameter["radius"][0]
                )
                parameter["bracket_size"][0] = (
                    parameter["bracket_size"][1]
                    + sphere_parameter["radius"][0] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][2] = (
                    sphere_parameter["radius"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_exist_angle"][0] = 180 + np.random.uniform(10, 50)
                if random.random() < 0.2:
                    parameter["bracket_exist_angle"][0] = 360

                parameter["has_top_endpoint"][0] = 0
                parameter["has_bottom_endpoint"][0] = 0
                if random.random() < 0.3:
                    parameter["has_top_endpoint"][0] = 1
                    if random.random() < 0.5:
                        parameter["has_bottom_endpoint"][0] = 1

                parameter["endpoint_radius"][0] = 2.9 * parameter["pivot_size"][0]
                movable_angle = (parameter["bracket_exist_angle"][0] - 180) / 2
                rotation_sign = 0 if random.random() < 0.7 else 1
                parameter["bracket_rotation"][0] = (
                    movable_angle * randRange(1, 0.5, 1.0)[0]
                    if rotation_sign == 0
                    else (180 - movable_angle * randRange(1, 0.5, 1.0)[0])
                )

                if is_tilt == 1:
                    tilt_angle = (
                        np.random.uniform(10, 15)
                        / 180
                        * np.pi
                        * (1 if rotation_sign == 0 else -1)
                    )
                else:
                    tilt_angle = 0

                if is_connected == 0:
                    parameter["pivot_continuity"][0] = 1
                    parameter["pivot_size"][1] = (
                        parameter["bracket_size"][0] * 2 * randRange(1, 1.0, 1.3)[0]
                    )
                    pivot_height = parameter["pivot_size"][1] / 2 * np.cos(tilt_angle)
                    if parameter["has_bottom_endpoint"][0] == 1:
                        pivot_height = (
                            parameter["pivot_size"][1] / 2
                            + parameter["endpoint_radius"][0]
                        ) * np.cos(tilt_angle)
                    parameter["pivot_seperation"][0] = 0

                else:
                    parameter["pivot_continuity"][0] = 0
                    parameter["pivot_size"][1] = (
                        parameter["bracket_size"][0] - sphere_parameter["radius"][0]
                    ) * randRange(1, 1.0, 1.3)[0]
                    parameter["pivot_seperation"][0] = sphere_parameter["radius"][0] * 2
                    pivot_height = (
                        parameter["pivot_size"][1] + sphere_parameter["radius"][0]
                    ) * np.cos(tilt_angle)
                    if parameter["has_bottom_endpoint"][0] == 1:
                        pivot_height = (
                            parameter["pivot_size"][1]
                            + sphere_parameter["radius"][0]
                            + parameter["endpoint_radius"][0]
                        ) * np.cos(tilt_angle)

                bottom_half_ball_height = np.maximum(
                    pivot_height, parameter["bracket_size"][0]
                )
                parameter["bracket_offset"][0] = 0
                if pivot_height > parameter["bracket_size"][0]:
                    total_offset_distance = pivot_height * np.tan(tilt_angle)
                    offset_x = total_offset_distance * np.sin(
                        parameter["rotation"][1] / 180 * np.pi
                    )

                parameter["rotation"][0] += tilt_angle / np.pi * 180
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Tilted_Bracket":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]
                original_outer_ring_thickness = (
                    parameter["bracket_size"][0] - parameter["bracket_size"][1]
                )
                parameter["circle_thickness"][0] = (
                    sphere_parameter["radius"][0] * 0.1 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["circle_thickness"][1] = (
                    sphere_parameter["radius"][0] / 10 * randRange(1, 0.8, 1.0)[0]
                )
                parameter["bracket_size"][1] = (
                    np.maximum(
                        (
                            sphere_parameter["radius"][0]
                            + parameter["circle_thickness"][0]
                            * randRange(1, 1.0, 1.2)[0]
                        ),
                        sphere_parameter["radius"][0] * 1.1,
                    )
                    * randRange(1, 1.0, 1.2)[0]
                )
                parameter["bracket_size"][0] = (
                    parameter["bracket_size"][1]
                    + original_outer_ring_thickness * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][2] *= randRange(1, 0.8, 1.2)[0]

                if random.random() < 0.2:
                    parameter["pivot_size"][0] = (
                        parameter["circle_thickness"][1] * randRange(1, 0.8, 1.0)[0]
                    )
                else:
                    parameter["pivot_size"][0] = 0
                parameter["pivot_size"][1] = (
                    parameter["bracket_size"][0] + parameter["bracket_size"][1]
                )

                parameter["circle_rotation"][0] = np.random.uniform(-45, 45)
                parameter["rotation"][0] += np.random.uniform(-45, 45)
                parameter["rotation"][1] += np.random.uniform(-90, 90)

                bottom_half_ball_height = parameter["bracket_size"][0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Enclosed_Bracket":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                parameter["circle_radius"][0] = (
                    sphere_parameter["radius"][0] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["circle_thickness"][0] = (
                    sphere_parameter["radius"][0] / 6 * randRange(1, 0.5, 1.2)[0]
                )
                parameter["circle_thickness"][1] = (
                    sphere_parameter["radius"][0] * 2 / 17 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["half_circle_number"][0] = np.random.randint(1, 3)
                bracket_thickness = (
                    parameter["circle_thickness"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][1] = (
                    parameter["circle_radius"][0] - parameter["circle_thickness"][0] / 2
                )
                parameter["bracket_size"][0] = (
                    parameter["bracket_size"][1] + bracket_thickness
                )
                parameter["bracket_size"][2] = (
                    sphere_parameter["radius"][0] / 8.5 * randRange(1, 0.5, 1.2)[0]
                )

                bottom_half_ball_height = parameter["bracket_size"][0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cylindrical_Base":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                parameter["bottom_size"][1] = (
                    sphere_parameter["radius"][0] * randRange(1, 0.5, 1.0)[0]
                )
                parameter["bottom_size"][0] = (
                    parameter["bottom_size"][1] * randRange(1, 0.8, 1.0)[0]
                )
                parameter["bottom_size"][2] = (
                    sphere_parameter["radius"][0] * 2 / 15 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["top_size"][1] = (
                    (parameter["bottom_size"][0] + 0.01 * randRange(1, 0.8, 1.8)[0])
                    if random.random() < 0.5
                    else 0
                )
                parameter["top_size"][0] = (
                    parameter["top_size"][1] * randRange(1, 0.5, 0.8)[0]
                )
                parameter["top_size"][2] = (
                    parameter["bottom_size"][2] * randRange(1, 0.8, 1.0)[0]
                )

                if random.random() < 0.5:
                    parameter["top_size"][1] = (
                        parameter["bottom_size"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["top_size"][0] = 0.01
                    parameter["top_size"][2] = (
                        parameter["bottom_size"][2] * randRange(1, 1.0, 1.5)[0]
                    )

                parameter["position"][0] -= offset_x
                parameter["position"][1] -= bottom_half_ball_height
                parameter["position"][2] -= total_offset_distance
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cuboidal_Base":
                sphere_parameter = concepts[0]["parameters"]
                bracket_parameter = concepts[1]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                parameter["bottom_size"][0] = (
                    sphere_parameter["radius"][0] * 3 / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bottom_size"][2] = (
                    parameter["bottom_size"][0] * randRange(1, 1.0, 1.3)[0]
                )
                if random.random() < 1:
                    parameter["top_size"][2] = parameter["bottom_size"][0]
                    parameter["top_size"][0] = (
                        parameter["top_size"][2] * randRange(1, 0.5, 0.8)[0]
                    )
                else:
                    parameter["top_size"][0] = (
                        bracket_parameter["bracket_size"][2]
                        * 2
                        * randRange(1, 1.1, 1.2)[0]
                    )
                    parameter["top_size"][2] = (
                        parameter["top_size"][0] * 1.5 * randRange(1, 0.8, 1.2)[0]
                    )
                parameter["top_size"][1] = (
                    sphere_parameter["radius"][0] * 2 / 15 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bottom_size"][1] = (
                    parameter["top_size"][1] * randRange(1, 0.8, 1.2)[0]
                )

                parameter["position"][0] -= offset_x
                parameter["position"][1] -= bottom_half_ball_height
                parameter["position"][2] -= total_offset_distance
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Star_Shaped_Base":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                parameter["top_size"][0] = (
                    sphere_parameter["radius"][0] / 6 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["top_size"][1] = (
                    sphere_parameter["radius"][0] * 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][2] = (
                    parameter["top_size"][1] * 2 / 5 * randRange(1, 0.8, 1.2)[0]
                )

                if random.random() < 0.9:
                    parameter["top_size"][1] = (
                        sphere_parameter["radius"][0] * 1.5 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["sub_size"][2] = (
                        parameter["top_size"][1] / 2 * randRange(1, 0.8, 1.2)[0]
                    )

                parameter["sub_size"][0] = (
                    sphere_parameter["radius"][0] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][1] = (
                    sphere_parameter["radius"][0] / 7 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["tilt_angle"][0] = randRange(1, 0, 48)[0]

                num_of_legs = np.random.randint(3, 5)
                parameter["num_legs"] = np.array([int(num_of_legs)])

                parameter["position"][0] -= offset_x
                parameter["position"][1] -= bottom_half_ball_height
                parameter["position"][2] -= total_offset_distance
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif globe_type == "High_Globe":
            if template == "Standard_Sphere":
                parameter["radius"][0] *= randRange(1, 0.8, 1.2)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)
            elif template == "Tilted_Bracket":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]
                original_outer_ring_thickness = (
                    parameter["bracket_size"][0] - parameter["bracket_size"][1]
                )
                parameter["circle_thickness"][0] = (
                    sphere_parameter["radius"][0] * 0.1 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["circle_thickness"][1] = (
                    sphere_parameter["radius"][0] / 10 * randRange(1, 0.8, 1.0)[0]
                )
                parameter["bracket_size"][1] = (
                    np.maximum(
                        (
                            sphere_parameter["radius"][0]
                            + parameter["circle_thickness"][0]
                            * randRange(1, 1.0, 1.2)[0]
                        ),
                        sphere_parameter["radius"][0] * 1.1,
                    )
                    * randRange(1, 1.0, 1.2)[0]
                )
                parameter["bracket_size"][0] = (
                    parameter["bracket_size"][1]
                    + original_outer_ring_thickness * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][2] *= randRange(1, 0.8, 1.2)[0]

                if random.random() < 0.2:
                    parameter["pivot_size"][0] = (
                        parameter["circle_thickness"][1] * randRange(1, 0.8, 1.0)[0]
                    )
                else:
                    parameter["pivot_size"][0] = 0
                parameter["pivot_size"][1] = (
                    parameter["bracket_size"][0] + parameter["bracket_size"][1]
                )

                parameter["circle_rotation"][0] = randRange(1, -45, 45)[0]
                parameter["rotation"][0] += randRange(1, -45, -45)[0]
                parameter["rotation"][1] += randRange(1, -90, 90)[0]

                bottom_half_ball_height = parameter["bracket_size"][0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Enclosed_Bracket":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                parameter["circle_radius"][0] = (
                    sphere_parameter["radius"][0] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["circle_thickness"][0] = (
                    sphere_parameter["radius"][0] / 6 * randRange(1, 0.5, 1.2)[0]
                )
                parameter["circle_thickness"][1] = (
                    sphere_parameter["radius"][0] * 2 / 17 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["half_circle_number"][0] = np.random.randint(1, 3)
                bracket_thickness = (
                    parameter["circle_thickness"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][1] = (
                    parameter["circle_radius"][0] - parameter["circle_thickness"][0] / 2
                )
                parameter["bracket_size"][0] = (
                    parameter["bracket_size"][1] + bracket_thickness
                )
                parameter["bracket_size"][2] = (
                    sphere_parameter["radius"][0] / 8.5 * randRange(1, 0.5, 1.2)[0]
                )

                bottom_half_ball_height = parameter["bracket_size"][0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Star_Shaped_Base":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                parameter["top_size"][0] = (
                    sphere_parameter["radius"][0] / 6 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["top_size"][1] = (
                    sphere_parameter["radius"][0] * 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][2] = (
                    parameter["top_size"][1] * 2 / 5 * randRange(1, 0.8, 1.2)[0]
                )

                if random.random() < 0.7:
                    parameter["top_size"][1] = (
                        sphere_parameter["radius"][0] * 1.5 * randRange(1, 0.3, 1.2)[0]
                    )
                    parameter["sub_size"][2] = (
                        sphere_parameter["radius"][0] * randRange(1, 0.8, 1.2)[0]
                    )

                parameter["sub_size"][0] = (
                    sphere_parameter["radius"][0] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][1] = (
                    sphere_parameter["radius"][0] / 7 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["tilt_angle"][0] = randRange(1, 0, 40)[0]

                num_of_legs = np.random.randint(3, 5)
                parameter["num_legs"] = np.array([int(num_of_legs)])

                parameter["position"][0] -= offset_x
                parameter["position"][1] -= bottom_half_ball_height
                parameter["position"][2] -= total_offset_distance
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif globe_type == "Table_Globe":
            if template == "Standard_Sphere":
                parameter["radius"][0] *= randRange(1, 0.8, 1.2)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Semi_Ring_Bracket":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                is_tilt = 0 if random.random() < 0.4 else 1
                is_connected = 0 if random.random() < 0.5 else 1

                parameter["pivot_size"][0] = (
                    sphere_parameter["radius"][0] / 20 * randRange(1, 0.8, 1.2)[0]
                )

                sphere_bracket_distance = (
                    sphere_parameter["radius"][0] / 50 * randRange(1, 0.9, 1.3)[0]
                    if random.random() < 0.5
                    else sphere_parameter["radius"][0]
                    * 2
                    / 15
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][1] = (
                    sphere_bracket_distance + sphere_parameter["radius"][0]
                )
                parameter["bracket_size"][0] = (
                    parameter["bracket_size"][1]
                    + sphere_parameter["radius"][0] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][2] = (
                    sphere_parameter["radius"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_exist_angle"][0] = 180 + np.random.uniform(10, 50)
                if random.random() < 0.2:
                    parameter["bracket_exist_angle"][0] = 360

                parameter["has_top_endpoint"][0] = 0
                parameter["has_bottom_endpoint"][0] = 0
                if random.random() < 0.3:
                    parameter["has_top_endpoint"][0] = 1
                    if random.random() < 0.5:
                        parameter["has_bottom_endpoint"][0] = 1

                parameter["endpoint_radius"][0] = 2.9 * parameter["pivot_size"][0]
                movable_angle = (parameter["bracket_exist_angle"][0] - 180) / 2
                rotation_sign = 0 if random.random() < 0.7 else 1
                parameter["bracket_rotation"][0] = (
                    movable_angle * randRange(1, 0.5, 1.0)[0]
                    if rotation_sign == 0
                    else (180 - movable_angle * randRange(1, 0.5, 1.0)[0])
                )

                if is_tilt == 1:
                    sign = 1 if rotation_sign == 0 else -1
                    tilt_angle = sign * np.random.uniform(10, 15) / 180 * np.pi
                else:
                    tilt_angle = 0

                if is_connected == 0:
                    parameter["pivot_continuity"][0] = 1
                    parameter["pivot_size"][1] = (
                        parameter["bracket_size"][0] * 2 * randRange(1, 1.0, 1.3)[0]
                    )
                    pivot_height = parameter["pivot_size"][1] / 2 * np.cos(tilt_angle)
                    if parameter["has_bottom_endpoint"][0] == 1:
                        pivot_height = (
                            parameter["pivot_size"][1] / 2
                            + parameter["endpoint_radius"][0]
                        ) * np.cos(tilt_angle)
                    parameter["pivot_seperation"][0] = 0

                else:
                    parameter["pivot_continuity"][0] = 0
                    parameter["pivot_size"][1] = (
                        parameter["bracket_size"][0] - sphere_parameter["radius"][0]
                    ) * randRange(1, 1.0, 1.3)[0]
                    parameter["pivot_seperation"][0] = sphere_parameter["radius"][0] * 2
                    pivot_height = (
                        parameter["pivot_size"][1] + sphere_parameter["radius"][0]
                    ) * np.cos(tilt_angle)
                    if parameter["has_bottom_endpoint"][0] == 1:
                        pivot_height = (
                            parameter["pivot_size"][1]
                            + sphere_parameter["radius"][0]
                            + parameter["endpoint_radius"][0]
                        ) * np.cos(tilt_angle)

                bottom_half_ball_height = np.maximum(
                    pivot_height, parameter["bracket_size"][0]
                )
                parameter["bracket_offset"][0] = 0
                if pivot_height > parameter["bracket_size"][0]:
                    total_offset_distance = pivot_height * np.tan(tilt_angle)
                    offset_x = total_offset_distance * np.sin(
                        parameter["rotation"][1] / 180 * np.pi
                    )

                parameter["rotation"][0] += tilt_angle / np.pi * 180
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Tilted_Bracket":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]
                original_outer_ring_thickness = (
                    parameter["bracket_size"][0] - parameter["bracket_size"][1]
                )
                parameter["circle_thickness"][0] = (
                    sphere_parameter["radius"][0] * 0.1 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["circle_thickness"][1] = (
                    sphere_parameter["radius"][0] / 10 * randRange(1, 0.8, 1.0)[0]
                )
                parameter["bracket_size"][1] = (
                    np.maximum(
                        (
                            sphere_parameter["radius"][0]
                            + parameter["circle_thickness"][0]
                            * randRange(1, 1.0, 1.2)[0]
                        ),
                        sphere_parameter["radius"][0] * 1.1,
                    )
                    * randRange(1, 1.0, 1.2)[0]
                )
                parameter["bracket_size"][0] = (
                    parameter["bracket_size"][1]
                    + original_outer_ring_thickness * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][2] *= randRange(1, 0.8, 1.2)[0]

                if random.random() < 0.2:
                    parameter["pivot_size"][0] = (
                        parameter["circle_thickness"][1] * randRange(1, 0.8, 1.0)[0]
                    )
                else:
                    parameter["pivot_size"][0] = 0
                parameter["pivot_size"][1] = (
                    parameter["bracket_size"][0] + parameter["bracket_size"][1]
                )

                parameter["circle_rotation"][0] = randRange(1, -45, 45)[0]
                parameter["rotation"][1] += randRange(1, -90, 90)[0]

                bottom_half_ball_height = parameter["bracket_size"][0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Enclosed_Bracket":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                parameter["circle_radius"][0] = (
                    sphere_parameter["radius"][0] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["circle_thickness"][0] = (
                    sphere_parameter["radius"][0] / 6 * randRange(1, 0.5, 1.2)[0]
                )
                parameter["circle_thickness"][1] = (
                    sphere_parameter["radius"][0] * 2 / 17 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["half_circle_number"][0] = np.random.randint(1, 3)
                bracket_thickness = (
                    parameter["circle_thickness"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][1] = (
                    parameter["circle_radius"][0] - parameter["circle_thickness"][0] / 2
                )
                parameter["bracket_size"][0] = (
                    parameter["bracket_size"][1] + bracket_thickness
                )
                parameter["bracket_size"][2] = (
                    sphere_parameter["radius"][0] / 8.5 * randRange(1, 0.5, 1.2)[0]
                )

                bottom_half_ball_height = parameter["bracket_size"][0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Table_Like_Base":
                sphere_parameter = concepts[0]["parameters"]
                bracket_parameter = concepts[1]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                number_of_legs = np.random.randint(3, 5)
                parameter["num_legs"] = np.array([int(number_of_legs)])
                parameter["circle_size"][2] = (
                    sphere_parameter["radius"][0] * 2 / 11 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["circle_size"][1] = bracket_parameter["bracket_size"][0]
                parameter["circle_size"][0] = (
                    parameter["circle_size"][1]
                    + bracket_parameter["bracket_size"][0]
                    / 4
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["leg_size"][1] = (
                    parameter["circle_size"][1] * 2 * 0.8125 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["leg_size"][0] = (
                    (parameter["circle_size"][0] - parameter["circle_size"][1])
                    / 2
                    * randRange(1, 0.8, 0.9)[0]
                )
                parameter["leg_seperation"][0] = (
                    parameter["circle_size"][0] + parameter["circle_size"][1]
                ) / 2
                parameter["has_bottom_part"][0] = np.random.randint(0, 2)
                parameter["bottom_size"][0] = (
                    parameter["circle_size"][0] * randRange(1, 1.0, 1.2)[0]
                )
                parameter["bottom_size"][1] = (
                    parameter["circle_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bottom_size"][2] = (
                    parameter["leg_size"][0] * 2 * 2 * randRange(1, 0.9, 1.5)[0]
                )
                parameter["bottom_offset"][0] = (
                    parameter["leg_size"][1] / 4 * randRange(1, 0.8, 1.2)[0]
                    if random.random() < 0.4
                    else 0
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif globe_type == "Strange_Globe":
            if template == "Standard_Sphere":
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["radius"][0] *= randRange(1, 0.8, 1.2)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Semi_Ring_Bracket":
                sphere_parameter = concepts[0]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = sphere_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                is_connected = 0 if random.random() < 0.5 else 1

                parameter["pivot_size"][0] = (
                    sphere_parameter["radius"][0] / 20 * randRange(1, 0.8, 1.2)[0]
                )

                sphere_bracket_distance = (
                    sphere_parameter["radius"][0] / 50 * randRange(1, 0.9, 1.3)[0]
                    if random.random() < 0.5
                    else sphere_parameter["radius"][0]
                    * 2
                    / 15
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][1] = (
                    sphere_bracket_distance + sphere_parameter["radius"][0]
                )
                parameter["bracket_size"][0] = (
                    parameter["bracket_size"][1]
                    + sphere_parameter["radius"][0] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_size"][2] = (
                    sphere_parameter["radius"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bracket_exist_angle"][0] = 180 + np.random.uniform(10, 50)
                if random.random() < 0.2:
                    parameter["bracket_exist_angle"][0] = 360

                parameter["has_top_endpoint"][0] = 0
                parameter["has_bottom_endpoint"][0] = 0
                if random.random() < 0.3:
                    parameter["has_top_endpoint"][0] = 1
                    if random.random() < 0.5:
                        parameter["has_bottom_endpoint"][0] = 1

                parameter["endpoint_radius"][0] = 2.9 * parameter["pivot_size"][0]
                movable_angle = (parameter["bracket_exist_angle"][0] - 180) / 2
                rotation_sign = 0 if random.random() < 0.7 else 1
                parameter["bracket_rotation"][0] = (
                    movable_angle * randRange(1, 0.5, 1.0)[0]
                    if rotation_sign == 0
                    else (180 - movable_angle * randRange(1, 0.5, 1.0)[0])
                )

                tilt_angle = (
                    (1 if rotation_sign == 0 else -1)
                    * np.random.uniform(10, 15)
                    / 180
                    * np.pi
                )
                rot_sign = 0 if rotation_sign == 0 else 1

                lean_angle_tmp = tilt_angle / np.pi * 180

                if is_connected == 0:
                    parameter["pivot_continuity"][0] = 1
                    parameter["pivot_size"][1] = (
                        parameter["bracket_size"][0] * 2 * randRange(1, 1.0, 1.3)[0]
                    )
                    pivot_height = parameter["pivot_size"][1] / 2 * np.cos(tilt_angle)
                    if parameter["has_bottom_endpoint"][0] == 1:
                        pivot_height = (
                            parameter["pivot_size"][1] / 2
                            + parameter["endpoint_radius"][0]
                        ) * np.cos(tilt_angle)
                    parameter["pivot_seperation"][0] = 0

                else:
                    parameter["pivot_continuity"][0] = 0
                    parameter["pivot_size"][1] = (
                        parameter["bracket_size"][0] - sphere_parameter["radius"][0]
                    ) * randRange(1, 1.0, 1.3)[0]
                    parameter["pivot_seperation"][0] = sphere_parameter["radius"][0] * 2
                    pivot_height = (
                        parameter["pivot_size"][1] + sphere_parameter["radius"][0]
                    ) * np.cos(tilt_angle)
                    if parameter["has_bottom_endpoint"][0] == 1:
                        pivot_height = (
                            parameter["pivot_size"][1]
                            + sphere_parameter["radius"][0]
                            + parameter["endpoint_radius"][0]
                        ) * np.cos(tilt_angle)

                bottom_half_ball_height = pivot_height
                parameter["bracket_offset"][0] = 0
                total_offset_distance = pivot_height * np.tan(tilt_angle)
                offset_x = total_offset_distance * np.sin(
                    parameter["rotation"][1] / 180 * np.pi
                )

                parameter["rotation"][0] += tilt_angle / np.pi * 180
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Special_Base":
                sphere_parameter = concepts[0]["parameters"]
                bracket_parameter = concepts[1]["parameters"]
                parameter["position"][0] = sphere_parameter["position"][0]
                parameter["position"][1] = sphere_parameter["position"][1]
                parameter["position"][2] = sphere_parameter["position"][2]
                parameter["rotation"][0] = sphere_parameter["rotation"][0]
                parameter["rotation"][1] = bracket_parameter["rotation"][1]
                parameter["rotation"][2] = sphere_parameter["rotation"][2]

                parameter["top_size"][0] = (
                    sphere_parameter["radius"][0] / 20 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["top_size"][1] = (
                    parameter["top_size"][0] * 11 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["radius"][0] = (
                    sphere_parameter["radius"][0] * 2 / 3 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["radius"][1] = np.maximum(
                    parameter["top_size"][0],
                    parameter["radius"][0] / 25 * randRange(1, 0.8, 1.2)[0],
                )
                parameter["top_rotation"][0] = np.abs(lean_angle_tmp)

                parameter["position"][1] += bracket_parameter["bracket_size"][0] * (
                    1 - np.cos(parameter["top_rotation"][0] / 180 * np.pi)
                )
                if rot_sign == 0:
                    parameter["rotation"][1] += 0
                elif rot_sign == 1:
                    parameter["rotation"][1] += 180

                parameter["position"][0] -= offset_x
                parameter["position"][1] -= bottom_half_ball_height
                parameter["position"][2] -= total_offset_distance

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
        globe_type = get_globe_type()
        existing_concept_templates = concept_template_existence(globe_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, globe_type)

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
            globe_type = get_globe_type()
            existing_concept_templates = concept_template_existence(globe_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, globe_type)
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
