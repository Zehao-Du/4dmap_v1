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


def get_safe_type():
    total_type = ["dialing", "keyboard"]
    weights = [1, 1]
    safe_type = random.choices(total_type, weights=weights, k=1)[0]
    return safe_type


def concept_template_existence(safe_type):
    if safe_type == "dialing":
        concept_template_variation = {
            "body": {"template": ["Mutiple_Layer_Body"], "neccessity": True},
            "door": {
                "template": [
                    "Cuboidal_Door",
                    "Behind_Double_Layer_Door",
                    "Front_Double_Layer_Door",
                    "Sunken_Door",
                ],
                "neccessity": True,
            },
            "connector": {
                "template": ["Cylindrical_Connecter", "T_Shaped_Connecter"],
                "neccessity": True,
            },
            "handle": {
                "template": ["Trifold_Handle", "Claw_Handle", "Round_Handle"],
                "neccessity": True,
            },
            "dial": {"template": ["Cylindrical_Dial"], "neccessity": True},
            "leg": {"template": ["Cuboidal_Leg"], "neccessity": False},
        }
    elif safe_type == "keyboard":
        concept_template_variation = {
            "body": {
                "template": ["Cuboidal_Body", "Mutiple_Layer_Body"],
                "neccessity": True,
            },
            "door": {
                "template": [
                    "Cuboidal_Door",
                    "Behind_Double_Layer_Door",
                    "Front_Double_Layer_Door",
                    "Sunken_Door",
                ],
                "neccessity": True,
            },
            "connector": {
                "template": ["Cylindrical_Connecter", "T_Shaped_Connecter"],
                "neccessity": True,
            },
            "handle": {
                "template": ["Trifold_Handle", "Claw_Handle", "Round_Handle"],
                "neccessity": True,
            },
            "controller": {"template": ["Regular_Controller"], "neccessity": True},
            "leg": {"template": ["Cuboidal_Leg"], "neccessity": False},
        }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["neccessity"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, safe_type):
    new_concepts = []

    body_type = ""
    door_type = ""
    handle_type = ""
    rot_direction = 0
    door_rot_angle = 0
    number_of_connector = 0
    handle_offset_x = 0
    handle_offset_y = 0
    connector_offset_x = 0

    handle_size_x = 0
    handle_size_y = 0

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if safe_type == "dialing":
            if template == "Cuboidal_Body":
                body_type = "Cuboidal_Body"
                parameter["size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][1] = parameter["thickness"][0]
                parameter["thickness"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][3] = parameter["thickness"][2]
                parameter["thickness"][4] *= randRange(1, 0.8, 1.2)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Mutiple_Layer_Body":
                body_type = "Mutiple_Layer_Body"
                parameter["size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][1] = parameter["thickness"][0]
                parameter["thickness"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][3] = parameter["thickness"][2]
                parameter["thickness"][4] *= randRange(1, 0.8, 1.2)[0]

                parameter["main_clapboard_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["main_clapboard_size"][1] = (
                    parameter["size"][2] * randRange(1, 0.7, 0.8)[0]
                )

                sub_layer_num = 1
                sublayer_height = [0, 0, 0]

                if random.random() < (17 / 22):
                    number_of_main_layers = 1
                    main_clapboard_offset = (
                        parameter["size"][1] / 4 * randRange(1, 0.5, 1.0)[0]
                    )
                    sign = 1 if random.random() < 0.5 else -1
                    parameter["main_clapboard_offset"][0] = sign * main_clapboard_offset
                    sublayer_height[0] = (
                        parameter["size"][1] / 2
                        - parameter["thickness"][0]
                        - parameter["main_clapboard_offset"][0]
                    )
                    sublayer_height[1] = (
                        parameter["size"][1]
                        - parameter["thickness"][0]
                        - parameter["thickness"][1]
                        - sublayer_height[0]
                    )
                else:
                    number_of_main_layers = 2
                    main_clapboard_offset = (
                        parameter["size"][1] / 6 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["main_clapboard_offset"][0] = main_clapboard_offset
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][0] = (
                        parameter["size"][0]
                    )
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][1] = (
                        parameter["main_clapboard_size"][0]
                    )
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][2] = (
                        parameter["main_clapboard_size"][1]
                    )
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_offset"][0] = 0
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_offset"][
                        1
                    ] = -main_clapboard_offset
                    sub_layer_num += 1
                    sublayer_height[0] = (
                        parameter["size"][1] / 2
                        - parameter["thickness"][0]
                        - main_clapboard_offset
                    )
                    sublayer_height[1] = 2 * main_clapboard_offset
                    sublayer_height[2] = (
                        parameter["size"][1] / 2
                        - parameter["thickness"][1]
                        - main_clapboard_offset
                    )

                cur_offset_from_top = 0
                for j in range(number_of_main_layers + 1):
                    if sub_layer_num > 10:
                        break
                    total_num = [0, 1, 2, 3]
                    weights = [0.8, 0.16, 0.03, 0.01]
                    num_of_vertical = random.choices(total_num, weights=weights, k=1)[0]
                    for k in range(num_of_vertical):
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][
                            0
                        ] = parameter["main_clapboard_size"][0]
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][
                            1
                        ] = sublayer_height[j]
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][
                            2
                        ] = parameter["main_clapboard_size"][1]
                        x_interval = (
                            parameter["size"][0] - 2 * parameter["thickness"][2]
                        ) / (num_of_vertical + 1)
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_offset"][
                            0
                        ] = (
                            -parameter["size"][0] / 2
                            + parameter["thickness"][2]
                            + x_interval * (k + 1)
                        )
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_offset"][
                            1
                        ] = (
                            parameter["size"][1] / 2
                            - parameter["thickness"][0]
                            - cur_offset_from_top
                            - sublayer_height[j] / 2
                        )
                        sub_layer_num += 1
                        if sub_layer_num > 10:
                            break
                    cur_offset_from_top += sublayer_height[j]

                parameter["num_of_sub_clapboards"] = np.array([int(sub_layer_num - 1)])

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cuboidal_Door":
                door_type = "Cuboidal_Door"
                body_parameter = concepts[0]["parameters"]
                total_free_space_for_door_thickness = 0
                if body_type == "Cuboidal_Body":
                    total_free_space_for_door_thickness = body_parameter["size"][2] / 16
                elif body_type == "Mutiple_Layer_Body":
                    total_free_space_for_door_thickness = (
                        body_parameter["size"][2]
                        - body_parameter["main_clapboard_size"][1]
                        - body_parameter["thickness"][4]
                    )
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["size"][0] = (
                    body_parameter["size"][0] - body_parameter["thickness"][2] * 2
                )
                parameter["size"][1] = (
                    body_parameter["size"][1] - body_parameter["thickness"][0] * 2
                )
                parameter["size"][2] = (
                    total_free_space_for_door_thickness * randRange(1, 0.8, 0.9)[0]
                )
                parameter["position"][2] -= parameter["size"][2]

                door_rot_angle = np.random.uniform(0, 180)
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                rot_direction = 0 if random.random() < 0.5 else 1

                if rot_direction == 0:
                    parameter["rotation"][1] -= door_rot_angle
                    parameter["position"][0] -= (
                        parameter["size"][0] / 2 * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] += parameter["size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )
                elif rot_direction == 1:
                    parameter["rotation"][1] += door_rot_angle
                    parameter["position"][0] += (
                        parameter["size"][0] / 2 * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] -= parameter["size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Behind_Double_Layer_Door":
                door_type = "Behind_Double_Layer_Door"
                body_parameter = concepts[0]["parameters"]
                total_free_space_for_door_thickness = 0
                door_thickness = 0
                if body_type == "Cuboidal_Body":
                    total_free_space_for_door_thickness = body_parameter["size"][2] / 16
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Mutiple_Layer_Body":
                    total_free_space_for_door_thickness = (
                        body_parameter["size"][2]
                        - body_parameter["main_clapboard_size"][1]
                        - body_parameter["thickness"][4]
                    )
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.7, 0.9)[0]
                    )
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["main_size"][0] = (
                    body_parameter["size"][0] - body_parameter["thickness"][2] * 2
                )
                parameter["main_size"][1] = (
                    body_parameter["size"][1] - body_parameter["thickness"][0] * 2
                )
                parameter["main_size"][2] = door_thickness * randRange(1, 0.8, 0.9)[0]
                parameter["behind_size"][0] = (
                    parameter["main_size"][0] * randRange(1, 0.5, 0.7)[0]
                )
                parameter["behind_size"][1] = (
                    parameter["main_size"][1] * randRange(1, 0.5, 0.7)[0]
                )
                parameter["position"][2] -= parameter["main_size"][2]
                if body_type == "Mutiple_Layer_Body":
                    parameter["behind_size"][2] = (
                        total_free_space_for_door_thickness - parameter["main_size"][2]
                    )
                elif body_type == "Cuboidal_Body":
                    parameter["behind_size"][2] = (
                        total_free_space_for_door_thickness - parameter["main_size"][2]
                    ) * randRange(1, 0.8, 1.2)[0]

                sign = -1 if random.random() < 0.5 else 1
                parameter["behind_offset"][0] = (
                    sign
                    * (parameter["main_size"][0] - parameter["behind_size"][0])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )
                parameter["behind_offset"][1] = (
                    sign
                    * (parameter["main_size"][1] - parameter["behind_size"][1])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )

                door_rot_angle = np.random.uniform(0, 180)
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                rot_direction = 0 if random.random() < 0.5 else 1

                if rot_direction == 0:
                    parameter["rotation"][1] -= door_rot_angle
                    parameter["position"][0] -= (
                        parameter["main_size"][0]
                        / 2
                        * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["main_size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] += parameter["main_size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["main_size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )
                elif rot_direction == 1:
                    parameter["rotation"][1] += door_rot_angle
                    parameter["position"][0] += (
                        parameter["main_size"][0]
                        / 2
                        * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["main_size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] -= parameter["main_size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["main_size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Front_Double_Layer_Door":
                door_type = "Front_Double_Layer_Door"
                body_parameter = concepts[0]["parameters"]
                door_thickness = 0
                if body_type == "Cuboidal_Body":
                    total_free_space_for_door_thickness = body_parameter["size"][2] / 14
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Mutiple_Layer_Body":
                    total_free_space_for_door_thickness = (
                        body_parameter["size"][2]
                        - body_parameter["main_clapboard_size"][1]
                        - body_parameter["thickness"][4]
                    )
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.7, 0.9)[0]
                    )
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["main_size"][0] = (
                    body_parameter["size"][0] - body_parameter["thickness"][2] * 2
                )
                parameter["main_size"][1] = (
                    body_parameter["size"][1] - body_parameter["thickness"][0] * 2
                )
                parameter["main_size"][2] = door_thickness * randRange(1, 0.8, 0.9)[0]
                parameter["front_size"][0] = (
                    parameter["main_size"][0] * randRange(1, 0.8, 0.9)[0]
                )
                parameter["front_size"][1] = (
                    parameter["main_size"][1] * randRange(1, 0.8, 0.9)[0]
                )
                parameter["position"][2] -= parameter["main_size"][2]
                if body_type == "Mutiple_Layer_Body":
                    parameter["front_size"][2] = (
                        parameter["main_size"][2] * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Cuboidal_Body":
                    parameter["front_size"][2] = (
                        parameter["main_size"][2] * randRange(1, 0.8, 1.0)[0]
                    )

                sign = -1 if random.random() < 0.5 else 1
                parameter["front_offset"][0] = (
                    sign
                    * (parameter["main_size"][0] - parameter["front_size"][0])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )
                parameter["front_offset"][1] = (
                    sign
                    * (parameter["main_size"][1] - parameter["front_size"][1])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )

                door_rot_angle = np.random.uniform(0, 180)
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                rot_direction = 0 if random.random() < 0.5 else 1

                if rot_direction == 0:
                    parameter["rotation"][1] -= door_rot_angle
                    parameter["position"][0] -= (
                        parameter["main_size"][0]
                        / 2
                        * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["main_size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] += parameter["main_size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["main_size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )
                elif rot_direction == 1:
                    parameter["rotation"][1] += door_rot_angle
                    parameter["position"][0] += (
                        parameter["main_size"][0]
                        / 2
                        * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["main_size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] -= parameter["main_size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["main_size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Sunken_Door":
                door_type = "Sunken_Door"
                body_parameter = concepts[0]["parameters"]
                door_thickness = 0
                if body_type == "Cuboidal_Body":
                    total_free_space_for_door_thickness = body_parameter["size"][2] / 14
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Mutiple_Layer_Body":
                    total_free_space_for_door_thickness = (
                        body_parameter["size"][2]
                        - body_parameter["main_clapboard_size"][1]
                        - body_parameter["thickness"][4]
                    )
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.7, 0.9)[0]
                    )
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["size"][0] = (
                    body_parameter["size"][0] - body_parameter["thickness"][2] * 2
                )
                parameter["size"][1] = (
                    body_parameter["size"][1] - body_parameter["thickness"][0] * 2
                )
                parameter["size"][2] = door_thickness
                sunken_mul = randRange(1, 0.8, 0.9)[0]
                parameter["sunken_size"][0] = parameter["size"][0] * sunken_mul
                parameter["sunken_size"][1] = parameter["size"][1] * sunken_mul
                parameter["position"][2] -= parameter["size"][2]
                if body_type == "Mutiple_Layer_Body":
                    parameter["sunken_size"][2] = (
                        parameter["size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Cuboidal_Body":
                    parameter["sunken_size"][2] = (
                        parameter["size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                    )

                sign = -1 if random.random() < 0.5 else 1
                parameter["sunken_offset"][0] = (
                    sign
                    * (parameter["size"][0] - parameter["sunken_size"][0])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )
                parameter["sunken_offset"][1] = (
                    sign
                    * (parameter["size"][1] - parameter["sunken_size"][1])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )

                door_rot_angle = np.random.uniform(0, 180)
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                rot_direction = 0 if random.random() < 0.5 else 1

                if rot_direction == 0:
                    parameter["rotation"][1] -= door_rot_angle
                    parameter["position"][0] -= (
                        parameter["size"][0] / 2 * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] += parameter["size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )
                elif rot_direction == 1:
                    parameter["rotation"][1] += door_rot_angle
                    parameter["position"][0] += (
                        parameter["size"][0] / 2 * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] -= parameter["size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cylindrical_Connecter":
                body_parameter = concepts[0]["parameters"]
                door_parameter = concepts[1]["parameters"]
                if door_type == "Sunken_Door" or door_type == "Cuboidal_Door":
                    inner_door_size = door_parameter["size"]
                else:
                    inner_door_size = door_parameter["main_size"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                number_of_connector = np.random.randint(1, 4)

                height_interval_between_connectors = body_parameter["size"][1] / (
                    number_of_connector + 1
                )
                cur_height_offset = height_interval_between_connectors
                parameter["size"][1] = (
                    body_parameter["size"][1]
                    / (number_of_connector + 1)
                    * randRange(1, 0.5, 0.7)[0]
                )
                if number_of_connector == 1:
                    parameter["size"][1] = (
                        body_parameter["size"][1] * randRange(1, 0.5, 0.7)[0]
                    )

                if door_type == "Sunken_Door" or door_type == "Cuboidal_Door":
                    parameter["size"][0] = (
                        door_parameter["size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                    )
                else:
                    parameter["size"][0] = (
                        door_parameter["main_size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                    )

                if rot_direction == 0:
                    parameter["position"][0] -= (
                        inner_door_size[0] / 2
                        + parameter["size"][0] * randRange(1, 0, 0.5)[0]
                    )
                else:
                    parameter["position"][0] += (
                        inner_door_size[0] / 2
                        + parameter["size"][0] * randRange(1, 0, 0.5)[0]
                    )

                connector_offset_x = parameter["size"][0]

                for i in range(int(number_of_connector)):
                    parameter["position"][1] = (
                        body_parameter["size"][1] / 2 - cur_height_offset
                    )
                    for k, v in parameter.items():
                        if isinstance(v, list):
                            parameter[k] = np.array(v)
                    concept_copy = copy.deepcopy(concept)
                    concept_copy["parameters"] = {
                        k: v.tolist() for k, v in parameter.items()
                    }
                    new_concepts.append(concept_copy)
                    cur_height_offset += height_interval_between_connectors

            elif template == "T_Shaped_Connecter":
                body_parameter = concepts[0]["parameters"]
                door_parameter = concepts[1]["parameters"]
                if door_type == "Sunken_Door" or door_type == "Cuboidal_Door":
                    inner_door_size = door_parameter["size"]
                else:
                    inner_door_size = door_parameter["main_size"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                number_of_connector = np.random.randint(2, 4)

                parameter["cylinder_size"][0] = (
                    inner_door_size[2] / 2 * randRange(1, 0.8, 1.0)[0]
                )

                if number_of_connector == 2:
                    height_interval_between_connectors = body_parameter["size"][1] / 4
                    mid_height = (
                        inner_door_size[1] - 2 * height_interval_between_connectors
                    )
                    cur_height_offset = [
                        height_interval_between_connectors,
                        mid_height,
                        height_interval_between_connectors,
                    ]
                    parameter["cylinder_size"][1] = (
                        inner_door_size[1] / 4 * randRange(1, 0.5, 0.7)[0]
                    )
                else:
                    height_interval_between_connectors = body_parameter["size"][1] / 4
                    cur_height_offset = height_interval_between_connectors
                    cur_height_offset = [
                        height_interval_between_connectors,
                        height_interval_between_connectors,
                        height_interval_between_connectors,
                    ]
                    parameter["cylinder_size"][1] = (
                        inner_door_size[1] / 4 * randRange(1, 0.5, 0.7)[0]
                    )

                parameter["lateral_cuboid_size"][0] = (
                    parameter["cylinder_size"][1] * randRange(1, 0.8, 1.1)[0]
                )
                parameter["lateral_cuboid_size"][1] = (
                    parameter["cylinder_size"][1] * 2 / 3 * randRange(1, 0.8, 1.0)[0]
                )
                parameter["lateral_cuboid_size"][2] = (
                    parameter["cylinder_size"][0] * 2 * randRange(1, 0.8, 0.9)[0]
                )
                parameter["lateral_cuboid_offset"][0] = (
                    parameter["lateral_cuboid_size"][1]
                    / 4
                    * (-1 if random.random() < 0.5 else 1)
                )

                connector_offset_x = (
                    parameter["lateral_cuboid_size"][0] + parameter["cylinder_size"][0]
                )

                if rot_direction == 0:
                    parameter["position"][0] -= inner_door_size[0] / 2
                else:
                    parameter["position"][0] += inner_door_size[0] / 2
                    parameter["rotation"][1] += 180

                total_height_offset = height_interval_between_connectors

                for i in range(number_of_connector):
                    parameter["position"][1] = (
                        inner_door_size[1] / 2 - total_height_offset
                    )
                    for k, v in parameter.items():
                        if isinstance(v, list):
                            parameter[k] = np.array(v)
                    concept_copy = copy.deepcopy(concept)
                    concept_copy["parameters"] = {
                        k: v.tolist() for k, v in parameter.items()
                    }
                    new_concepts.append(concept_copy)
                    total_height_offset += cur_height_offset[i]

            elif template == "Trifold_Handle":
                handle_type = "Trifold_Handle"
                door_parameter = concepts[1]["parameters"]
                if door_type == "Cuboidal_Door":
                    inner_door_size = copy.deepcopy(door_parameter["size"])
                elif door_type == "Behind_Double_Layer_Door":
                    inner_door_size = copy.deepcopy(door_parameter["main_size"])
                elif door_type == "Sunken_Door":
                    inner_door_size = copy.deepcopy(door_parameter["sunken_size"])
                    inner_door_size[2] = door_parameter["size"][2]
                else:
                    inner_door_size = copy.deepcopy(door_parameter["front_size"])
                    inner_door_size[2] = (
                        door_parameter["front_size"][2] + door_parameter["main_size"][2]
                    )

                parameter["position"][0] = door_parameter["position"][0]
                parameter["position"][1] = door_parameter["position"][1]
                parameter["position"][2] = door_parameter["position"][2]
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                parameter["bottom_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["bottom_size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["bottom_size"][2] *= randRange(1, 0.8, 1.2)[0]

                parameter["top_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["top_size"][1] *= randRange(1, 0.5, 1.2)[0]
                parameter["top_size"][2] *= randRange(1, 0.8, 1.2)[0]

                parameter["bottom_seperation"][0] = (
                    parameter["top_size"][1] - parameter["bottom_size"][1]
                )
                cur_x_offset = inner_door_size[0] / 2 * randRange(1, 0.3, 0.8)[0]
                if rot_direction == 1:
                    cur_x_offset = -inner_door_size[0] / 2 * randRange(1, 0.3, 0.8)[0]
                if door_type == "Sunken_Door":
                    cur_x_offset = (
                        inner_door_size[0] / 2
                        + (
                            door_parameter["size"][0] / 2
                            - inner_door_size[0] / 2
                            - door_parameter["sunken_offset"][0]
                        )
                        / 2
                    )
                    if rot_direction == 1:
                        cur_x_offset = -(
                            inner_door_size[0] / 2
                            + (
                                door_parameter["size"][0] / 2
                                - inner_door_size[0] / 2
                                + door_parameter["sunken_offset"][0]
                            )
                            / 2
                        )
                door_rot_angle_radian = door_rot_angle / 180 * np.pi
                cur_y_offset = (
                    (inner_door_size[1] - parameter["top_size"][1])
                    / 2
                    * randRange(1, 0.0, 0.5)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )

                handle_offset_x = cur_x_offset
                handle_offset_y = cur_y_offset

                handle_size_x = np.maximum(
                    parameter["bottom_size"][0], parameter["top_size"][0]
                )
                handle_size_y = parameter["top_size"][1]

                if rot_direction == 0:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] -= inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                elif rot_direction == 1:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] -= cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] += inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Claw_Handle":
                handle_type = "Claw_Handle"
                door_parameter = new_concepts[1]["parameters"]
                if door_type == "Cuboidal_Door":
                    inner_door_size = copy.deepcopy(door_parameter["size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Behind_Double_Layer_Door":
                    inner_door_size = copy.deepcopy(door_parameter["main_size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Sunken_Door":
                    inner_door_size = copy.deepcopy(door_parameter["sunken_size"])
                    inner_door_offset = [
                        door_parameter["sunken_offset"][0],
                        door_parameter["sunken_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["size"])
                else:
                    inner_door_size = copy.deepcopy(door_parameter["front_size"])
                    inner_door_size[2] = (
                        door_parameter["front_size"][2] + door_parameter["main_size"][2]
                    )
                    inner_door_offset = [
                        door_parameter["front_offset"][0],
                        door_parameter["front_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["main_size"])

                parameter["position"][0] = door_parameter["position"][0]
                parameter["position"][1] = door_parameter["position"][1]
                parameter["position"][2] = door_parameter["position"][2]
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                parameter["bottom_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["bottom_size"][1] *= randRange(1, 0.8, 1.2)[0]

                parameter["fork_size"][0] = (
                    inner_door_size[0] / 2 * randRange(1, 0.15, 0.5)[0]
                )
                parameter["fork_size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["fork_size"][2] = (
                    parameter["fork_size"][1] * randRange(1, 0.8, 0.9)[0]
                )
                parameter["fork_offset"][0] = (
                    parameter["bottom_size"][1] * randRange(1, 0.0, 0.3)[0]
                )
                parameter["fork_tilt_rotation"][0] = np.random.uniform(0, 10)
                num_of_fork = np.random.randint(1, 5)
                parameter["num_forks"] = np.array([int(num_of_fork)])

                if rot_direction == 0:
                    x_offset_lower_bound = np.minimum(
                        inner_door_size[0] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                        - inner_door_offset[0],
                        outer_door_size[0] / 2
                        - connector_offset_x
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0]),
                    )
                    x_offset_upper_bound = (
                        inner_door_size[0] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                        + inner_door_offset[0]
                    )
                    cur_x_offset = np.random.uniform(
                        -x_offset_lower_bound, x_offset_upper_bound
                    )
                else:
                    x_offset_lower_bound = (
                        inner_door_size[0] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                        - inner_door_offset[0]
                    )
                    x_offset_upper_bound = np.minimum(
                        inner_door_size[0] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                        + inner_door_offset[0],
                        outer_door_size[0] / 2
                        - connector_offset_x
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0]),
                    )
                    cur_x_offset = np.random.uniform(
                        -x_offset_lower_bound, x_offset_upper_bound
                    )

                door_rot_angle_radian = door_rot_angle / 180 * np.pi
                cur_y_offset = (
                    (
                        inner_door_size[1] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                    )
                    * randRange(1, 0.0, 0.5)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["position"][1] += cur_y_offset
                handle_offset_x = cur_x_offset
                handle_offset_y = cur_y_offset

                handle_size_x = (
                    parameter["fork_size"][0] + parameter["bottom_size"][0]
                ) * 2
                handle_size_y = (
                    parameter["fork_size"][0] + parameter["bottom_size"][0]
                ) * 2

                parameter["rotation"][2] += np.random.uniform(0, 360)

                if rot_direction == 0:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] -= inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                elif rot_direction == 1:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] -= cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] += inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Round_Handle":
                handle_type = "Claw_Handle"
                door_parameter = new_concepts[1]["parameters"]
                if door_type == "Cuboidal_Door":
                    inner_door_size = copy.deepcopy(door_parameter["size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Behind_Double_Layer_Door":
                    inner_door_size = copy.deepcopy(door_parameter["main_size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Sunken_Door":
                    inner_door_size = copy.deepcopy(door_parameter["sunken_size"])
                    inner_door_offset = [
                        door_parameter["sunken_offset"][0],
                        door_parameter["sunken_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["size"])
                else:
                    inner_door_size = copy.deepcopy(door_parameter["front_size"])
                    inner_door_size[2] = (
                        door_parameter["front_size"][2] + door_parameter["main_size"][2]
                    )
                    inner_door_offset = [
                        door_parameter["front_offset"][0],
                        door_parameter["front_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["main_size"])

                parameter["position"][0] = door_parameter["position"][0]
                parameter["position"][1] = door_parameter["position"][1]
                parameter["position"][2] = door_parameter["position"][2]
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                parameter["bottom_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["bottom_size"][1] *= randRange(1, 0.8, 1.2)[0]

                parameter["fork_size"][0] = (
                    inner_door_size[0] / 2 * randRange(1, 0.1, 0.5)[0]
                )
                parameter["fork_size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["fork_size"][2] = (
                    parameter["fork_size"][1] * randRange(1, 0.8, 0.9)[0]
                )
                parameter["fork_offset"][0] = (
                    parameter["bottom_size"][1] * randRange(1, 0.0, 0.3)[0]
                )
                parameter["fork_tilt_rotation"][0] = np.random.uniform(0, 10)
                num_of_fork = np.random.randint(3, 5)
                parameter["num_forks"] = np.array([int(num_of_fork)])

                parameter["circle_size"][0] = (
                    parameter["fork_size"][0] + parameter["bottom_size"][0]
                ) * randRange(1, 1.15, 1.2)[0]
                parameter["circle_size"][1] = parameter["fork_size"][2]

                if rot_direction == 0:
                    x_offset_lower_bound = np.minimum(
                        inner_door_size[0] / 2
                        - parameter["circle_size"][0]
                        - inner_door_offset[0],
                        outer_door_size[0] / 2
                        - connector_offset_x
                        - parameter["circle_size"][0],
                    )
                    x_offset_upper_bound = (
                        inner_door_size[0] / 2
                        - parameter["circle_size"][0]
                        + inner_door_offset[0]
                    )
                    cur_x_offset = np.random.uniform(
                        -x_offset_lower_bound, x_offset_upper_bound
                    )
                else:
                    x_offset_lower_bound = (
                        inner_door_size[0] / 2
                        - parameter["circle_size"][0]
                        - inner_door_offset[0]
                    )
                    x_offset_upper_bound = np.minimum(
                        inner_door_size[0] / 2
                        - parameter["circle_size"][0]
                        + inner_door_offset[0],
                        outer_door_size[0] / 2
                        - connector_offset_x
                        - parameter["circle_size"][0],
                    )
                    cur_x_offset = np.random.uniform(
                        -x_offset_lower_bound, x_offset_upper_bound
                    )

                door_rot_angle_radian = door_rot_angle / 180 * np.pi
                cur_y_offset = (
                    (inner_door_size[1] / 2 - parameter["circle_size"][0])
                    * randRange(1, 0.0, 0.5)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["position"][1] += cur_y_offset
                handle_offset_x = cur_x_offset
                handle_offset_y = cur_y_offset

                handle_size_x = parameter["circle_size"][0] * 2
                handle_size_y = parameter["circle_size"][0] * 2

                parameter["rotation"][2] += np.random.uniform(0, 360)

                if rot_direction == 0:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] -= inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                elif rot_direction == 1:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] -= cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] += inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cylindrical_Dial":
                door_parameter = new_concepts[1]["parameters"]
                if door_type == "Cuboidal_Door":
                    inner_door_size = copy.deepcopy(door_parameter["size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Behind_Double_Layer_Door":
                    inner_door_size = copy.deepcopy(door_parameter["main_size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Sunken_Door":
                    inner_door_size = copy.deepcopy(door_parameter["sunken_size"])
                    inner_door_offset = [
                        door_parameter["sunken_offset"][0],
                        door_parameter["sunken_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["size"])
                else:
                    inner_door_size = copy.deepcopy(door_parameter["front_size"])
                    inner_door_size[2] = (
                        door_parameter["front_size"][2] + door_parameter["main_size"][2]
                    )
                    inner_door_offset = [
                        door_parameter["front_offset"][0],
                        door_parameter["front_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["main_size"])

                parameter["position"][0] = door_parameter["position"][0]
                parameter["position"][1] = door_parameter["position"][1]
                parameter["position"][2] = door_parameter["position"][2]
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                bottom_size_mul = (
                    2 * randRange(1, 0.4, 1.0)[0] if random.random() < 0.5 else 0.8
                )

                parameter["bottom_size"][1] = (
                    inner_door_size[0]
                    / 20
                    * randRange(1, 0.8, 1.2)[0]
                    * bottom_size_mul
                )
                parameter["bottom_size"][0] = (
                    parameter["bottom_size"][1] * 3 / 2 * randRange(1, 1.0, 1.2)[0]
                )
                parameter["bottom_size"][2] *= (
                    randRange(1, 0.8, 1.2)[0] * bottom_size_mul
                )
                parameter["top_size"][0] = (
                    parameter["bottom_size"][1] * randRange(1, 0.95, 1.05)[0]
                )
                parameter["top_size"][1] = (
                    parameter["top_size"][0] / 2 * randRange(1, 1.0, 1.2)[0]
                )

                outer_x_lower_bound = -inner_door_size[0] / 2 + inner_door_offset[0]
                other_x_lower_bound = -outer_door_size[0] / 2 + +(
                    connector_offset_x if rot_direction == 0 else 0
                )
                outer_x_lower_bound = np.maximum(
                    outer_x_lower_bound, other_x_lower_bound
                )
                outer_x_upper_bound = inner_door_size[0] / 2 + inner_door_offset[0]
                other_x_upper_bound = outer_door_size[0] / 2 - (
                    connector_offset_x if rot_direction == 1 else 0
                )
                outer_x_upper_bound = np.minimum(
                    outer_x_upper_bound, other_x_upper_bound
                )
                inner_x_lower_bound = np.maximum(
                    handle_offset_x - handle_size_x / 2, outer_x_lower_bound
                )
                inner_x_upper_bound = np.minimum(
                    handle_offset_x + handle_size_x / 2, outer_x_upper_bound
                )

                radius_x_1 = inner_x_lower_bound - outer_x_lower_bound
                radius_x_2 = outer_x_upper_bound - inner_x_upper_bound
                radius_x_max = np.maximum(radius_x_1, radius_x_2)

                if radius_x_max <= parameter["bottom_size"][0] * 2:
                    bottom_size_mul = randRange(1, 0.6, 1.0)[0]
                    parameter["bottom_size"][1] = (
                        inner_door_size[0]
                        / 20
                        * randRange(1, 0.8, 1.2)[0]
                        * bottom_size_mul
                    )
                    parameter["bottom_size"][0] = (
                        parameter["bottom_size"][1] * 3 / 2 * randRange(1, 1.0, 1.2)[0]
                    )
                    parameter["bottom_size"][2] *= (
                        randRange(1, 0.8, 1.2)[0] * bottom_size_mul
                    )
                    parameter["top_size"][0] = (
                        parameter["bottom_size"][1] * randRange(1, 0.95, 1.05)[0]
                    )
                    parameter["top_size"][1] = (
                        parameter["top_size"][0] / 2 * randRange(1, 1.0, 1.2)[0]
                    )
                radius = parameter["bottom_size"][0]

                x_lower = outer_x_lower_bound + radius
                x_upper = outer_x_upper_bound - radius
                y_lower = -inner_door_size[1] / 2 + inner_door_offset[1] + radius
                y_upper = inner_door_size[1] / 2 + inner_door_offset[1] - radius

                if door_type != "Sunken_Door" or handle_type != "Trifold_Handle":
                    exp_x_lower = np.maximum(
                        handle_offset_x - handle_size_x / 2 - radius, x_lower
                    )
                    exp_x_upper = np.minimum(
                        handle_offset_x + handle_size_x / 2 + radius, x_upper
                    )
                    exp_y_lower = np.maximum(
                        -handle_size_y / 2 + handle_offset_y - radius, y_lower
                    )
                    exp_y_upper = np.minimum(
                        handle_size_y / 2 + handle_offset_y + radius, y_upper
                    )
                else:
                    exp_x_lower = x_upper + 1
                    exp_x_upper = x_upper + 1
                    exp_y_lower = y_upper + 1
                    exp_y_upper = y_upper + 1

                points = []

                count = 0
                if (
                    x_lower >= exp_x_lower
                    and x_upper <= exp_x_upper
                    and y_lower >= exp_y_lower
                    and y_upper <= exp_y_upper
                ):
                    continue
                while len(points) < 1:
                    if count > 50:
                        break
                    x = np.random.uniform(x_lower, x_upper)
                    y = np.random.uniform(y_lower, y_upper)
                    if not (
                        exp_x_lower <= x <= exp_x_upper
                        and exp_y_lower <= y <= exp_y_upper
                    ):
                        points.append([x, y])
                    else:
                        count += 1
                if count > 50:
                    continue

                cur_x_offset = points[0][0]
                cur_y_offset = points[0][1]

                parameter["position"][1] += cur_y_offset
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                if rot_direction == 0:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] -= inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                elif rot_direction == 1:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] -= cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] += inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cuboidal_Leg":
                body_parameter = concepts[0]["parameters"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["front_legs_size"][0] = (
                    body_parameter["size"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["front_legs_size"][1] = (
                    body_parameter["size"][1] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["front_legs_size"][2] = (
                    body_parameter["size"][2] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][0] = (
                    parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][1] = parameter["front_legs_size"][1]
                parameter["rear_legs_size"][2] = (
                    parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["legs_separation"][0] = (
                    body_parameter["size"][0] - 2 * parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 0.9)[0]
                parameter["legs_separation"][1] = (
                    body_parameter["size"][0] - 2 * parameter["rear_legs_size"][0]
                ) * randRange(1, 0.7, 0.9)[0]
                parameter["legs_separation"][2] = (
                    body_parameter["size"][2]
                    - parameter["front_legs_size"][2]
                    - parameter["rear_legs_size"][2]
                ) * randRange(1, 0.7, 0.9)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif safe_type == "keyboard":
            if template == "Cuboidal_Body":
                body_type = "Cuboidal_Body"
                parameter["size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][1] = parameter["thickness"][0]
                parameter["thickness"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][3] = parameter["thickness"][2]
                parameter["thickness"][4] *= randRange(1, 0.8, 1.2)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Mutiple_Layer_Body":
                body_type = "Mutiple_Layer_Body"
                parameter["size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][1] = parameter["thickness"][0]
                parameter["thickness"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["thickness"][3] = parameter["thickness"][2]
                parameter["thickness"][4] *= randRange(1, 0.8, 1.2)[0]

                parameter["main_clapboard_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["main_clapboard_size"][1] = (
                    parameter["size"][2] * randRange(1, 0.7, 0.8)[0]
                )

                sub_layer_num = 1
                sublayer_height = [0, 0, 0]

                if random.random() < (17 / 22):
                    number_of_main_layers = 1
                    main_clapboard_offset = (
                        parameter["size"][1] / 4 * randRange(1, 0.5, 1.0)[0]
                    )
                    sign = 1 if random.random() < 0.5 else -1
                    parameter["main_clapboard_offset"][0] = sign * main_clapboard_offset
                    sublayer_height[0] = (
                        parameter["size"][1] / 2
                        - parameter["thickness"][0]
                        - parameter["main_clapboard_offset"][0]
                    )
                    sublayer_height[1] = (
                        parameter["size"][1]
                        - parameter["thickness"][0]
                        - parameter["thickness"][1]
                        - sublayer_height[0]
                    )
                else:
                    number_of_main_layers = 2
                    main_clapboard_offset = (
                        parameter["size"][1] / 6 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["main_clapboard_offset"][0] = main_clapboard_offset
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][0] = (
                        parameter["size"][0]
                    )
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][1] = (
                        parameter["main_clapboard_size"][0]
                    )
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][2] = (
                        parameter["main_clapboard_size"][1]
                    )
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_offset"][0] = 0
                    parameter["sub_clapboard_" + str(sub_layer_num) + "_offset"][
                        1
                    ] = -main_clapboard_offset
                    sub_layer_num += 1
                    sublayer_height[0] = (
                        parameter["size"][1] / 2
                        - parameter["thickness"][0]
                        - main_clapboard_offset
                    )
                    sublayer_height[1] = 2 * main_clapboard_offset
                    sublayer_height[2] = (
                        parameter["size"][1] / 2
                        - parameter["thickness"][1]
                        - main_clapboard_offset
                    )

                cur_offset_from_top = 0
                for j in range(number_of_main_layers + 1):
                    if sub_layer_num > 10:
                        break
                    total_num = [0, 1, 2, 3]
                    weights = [0.8, 0.16, 0.03, 0.01]
                    num_of_vertical = random.choices(total_num, weights=weights, k=1)[0]
                    for k in range(num_of_vertical):
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][
                            0
                        ] = parameter["main_clapboard_size"][0]
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][
                            1
                        ] = sublayer_height[j]
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_size"][
                            2
                        ] = parameter["main_clapboard_size"][1]
                        x_interval = (
                            parameter["size"][0] - 2 * parameter["thickness"][2]
                        ) / (num_of_vertical + 1)
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_offset"][
                            0
                        ] = (
                            -parameter["size"][0] / 2
                            + parameter["thickness"][2]
                            + x_interval * (k + 1)
                        )
                        parameter["sub_clapboard_" + str(sub_layer_num) + "_offset"][
                            1
                        ] = (
                            parameter["size"][1] / 2
                            - parameter["thickness"][0]
                            - cur_offset_from_top
                            - sublayer_height[j] / 2
                        )
                        sub_layer_num += 1
                        if sub_layer_num > 10:
                            break
                    cur_offset_from_top += sublayer_height[j]

                parameter["num_of_sub_clapboards"] = np.array([int(sub_layer_num - 1)])

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cuboidal_Door":
                door_type = "Cuboidal_Door"
                body_parameter = concepts[0]["parameters"]
                total_free_space_for_door_thickness = 0
                if body_type == "Cuboidal_Body":
                    total_free_space_for_door_thickness = body_parameter["size"][2] / 16
                elif body_type == "Mutiple_Layer_Body":
                    total_free_space_for_door_thickness = (
                        body_parameter["size"][2]
                        - body_parameter["main_clapboard_size"][1]
                        - body_parameter["thickness"][4]
                    )
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["size"][0] = (
                    body_parameter["size"][0] - body_parameter["thickness"][2] * 2
                )
                parameter["size"][1] = (
                    body_parameter["size"][1] - body_parameter["thickness"][0] * 2
                )
                parameter["size"][2] = (
                    total_free_space_for_door_thickness * randRange(1, 0.8, 0.9)[0]
                )
                parameter["position"][2] -= parameter["size"][2]

                door_rot_angle = np.random.uniform(0, 180)
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                rot_direction = 0 if random.random() < 0.5 else 1

                if rot_direction == 0:
                    parameter["rotation"][1] -= door_rot_angle
                    parameter["position"][0] -= (
                        parameter["size"][0] / 2 * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] += parameter["size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )
                elif rot_direction == 1:
                    parameter["rotation"][1] += door_rot_angle
                    parameter["position"][0] += (
                        parameter["size"][0] / 2 * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] -= parameter["size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Behind_Double_Layer_Door":
                door_type = "Behind_Double_Layer_Door"
                body_parameter = concepts[0]["parameters"]
                total_free_space_for_door_thickness = 0
                door_thickness = 0
                if body_type == "Cuboidal_Body":
                    total_free_space_for_door_thickness = body_parameter["size"][2] / 16
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Mutiple_Layer_Body":
                    total_free_space_for_door_thickness = (
                        body_parameter["size"][2]
                        - body_parameter["main_clapboard_size"][1]
                        - body_parameter["thickness"][4]
                    )
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.7, 0.9)[0]
                    )
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["main_size"][0] = (
                    body_parameter["size"][0] - body_parameter["thickness"][2] * 2
                )
                parameter["main_size"][1] = (
                    body_parameter["size"][1] - body_parameter["thickness"][0] * 2
                )
                parameter["main_size"][2] = door_thickness * randRange(1, 0.8, 0.9)[0]
                parameter["behind_size"][0] = (
                    parameter["main_size"][0] * randRange(1, 0.5, 0.7)[0]
                )
                parameter["behind_size"][1] = (
                    parameter["main_size"][1] * randRange(1, 0.5, 0.7)[0]
                )
                parameter["position"][2] -= parameter["main_size"][2]
                if body_type == "Mutiple_Layer_Body":
                    parameter["behind_size"][2] = (
                        total_free_space_for_door_thickness - parameter["main_size"][2]
                    )
                elif body_type == "Cuboidal_Body":
                    parameter["behind_size"][2] = (
                        total_free_space_for_door_thickness - parameter["main_size"][2]
                    ) * randRange(1, 0.8, 1.2)[0]

                sign = -1 if random.random() < 0.5 else 1
                parameter["behind_offset"][0] = (
                    sign
                    * (parameter["main_size"][0] - parameter["behind_size"][0])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )
                parameter["behind_offset"][1] = (
                    sign
                    * (parameter["main_size"][1] - parameter["behind_size"][1])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )

                door_rot_angle = np.random.uniform(0, 180)
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                rot_direction = 0 if random.random() < 0.5 else 1

                if rot_direction == 0:
                    parameter["rotation"][1] -= door_rot_angle
                    parameter["position"][0] -= (
                        parameter["main_size"][0]
                        / 2
                        * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["main_size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] += parameter["main_size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["main_size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )
                elif rot_direction == 1:
                    parameter["rotation"][1] += door_rot_angle
                    parameter["position"][0] += (
                        parameter["main_size"][0]
                        / 2
                        * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["main_size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] -= parameter["main_size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["main_size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Front_Double_Layer_Door":
                door_type = "Front_Double_Layer_Door"
                body_parameter = concepts[0]["parameters"]
                door_thickness = 0
                if body_type == "Cuboidal_Body":
                    total_free_space_for_door_thickness = body_parameter["size"][2] / 14
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Mutiple_Layer_Body":
                    total_free_space_for_door_thickness = (
                        body_parameter["size"][2]
                        - body_parameter["main_clapboard_size"][1]
                        - body_parameter["thickness"][4]
                    )
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.7, 0.9)[0]
                    )
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["main_size"][0] = (
                    body_parameter["size"][0] - body_parameter["thickness"][2] * 2
                )
                parameter["main_size"][1] = (
                    body_parameter["size"][1] - body_parameter["thickness"][0] * 2
                )
                parameter["main_size"][2] = door_thickness * randRange(1, 0.8, 0.9)[0]
                parameter["front_size"][0] = (
                    parameter["main_size"][0] * randRange(1, 0.8, 0.9)[0]
                )
                parameter["front_size"][1] = (
                    parameter["main_size"][1] * randRange(1, 0.8, 0.9)[0]
                )
                parameter["position"][2] -= parameter["main_size"][2]
                if body_type == "Mutiple_Layer_Body":
                    parameter["front_size"][2] = (
                        parameter["main_size"][2] * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Cuboidal_Body":
                    parameter["front_size"][2] = (
                        parameter["main_size"][2] * randRange(1, 0.8, 1.0)[0]
                    )

                sign = -1 if random.random() < 0.5 else 1
                parameter["front_offset"][0] = (
                    sign
                    * (parameter["main_size"][0] - parameter["front_size"][0])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )
                parameter["front_offset"][1] = (
                    sign
                    * (parameter["main_size"][1] - parameter["front_size"][1])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )

                door_rot_angle = np.random.uniform(0, 180)
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                rot_direction = 0 if random.random() < 0.5 else 1

                if rot_direction == 0:
                    parameter["rotation"][1] -= door_rot_angle
                    parameter["position"][0] -= (
                        parameter["main_size"][0]
                        / 2
                        * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["main_size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] += parameter["main_size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["main_size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )
                elif rot_direction == 1:
                    parameter["rotation"][1] += door_rot_angle
                    parameter["position"][0] += (
                        parameter["main_size"][0]
                        / 2
                        * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["main_size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] -= parameter["main_size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["main_size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Sunken_Door":
                door_type = "Sunken_Door"
                body_parameter = concepts[0]["parameters"]
                door_thickness = 0
                if body_type == "Cuboidal_Body":
                    total_free_space_for_door_thickness = body_parameter["size"][2] / 14
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Mutiple_Layer_Body":
                    total_free_space_for_door_thickness = (
                        body_parameter["size"][2]
                        - body_parameter["main_clapboard_size"][1]
                        - body_parameter["thickness"][4]
                    )
                    door_thickness = (
                        total_free_space_for_door_thickness * randRange(1, 0.7, 0.9)[0]
                    )
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["size"][0] = (
                    body_parameter["size"][0] - body_parameter["thickness"][2] * 2
                )
                parameter["size"][1] = (
                    body_parameter["size"][1] - body_parameter["thickness"][0] * 2
                )
                parameter["size"][2] = door_thickness
                sunken_mul = randRange(1, 0.8, 0.9)[0]
                parameter["sunken_size"][0] = parameter["size"][0] * sunken_mul
                parameter["sunken_size"][1] = parameter["size"][1] * sunken_mul
                parameter["position"][2] -= parameter["size"][2]
                if body_type == "Mutiple_Layer_Body":
                    parameter["sunken_size"][2] = (
                        parameter["size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                    )
                elif body_type == "Cuboidal_Body":
                    parameter["sunken_size"][2] = (
                        parameter["size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                    )

                sign = -1 if random.random() < 0.5 else 1
                parameter["sunken_offset"][0] = (
                    sign
                    * (parameter["size"][0] - parameter["sunken_size"][0])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )
                parameter["sunken_offset"][1] = (
                    sign
                    * (parameter["size"][1] - parameter["sunken_size"][1])
                    / 2
                    * randRange(1, 0, 0.5)[0]
                )

                door_rot_angle = np.random.uniform(0, 180)
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                rot_direction = 0 if random.random() < 0.5 else 1

                if rot_direction == 0:
                    parameter["rotation"][1] -= door_rot_angle
                    parameter["position"][0] -= (
                        parameter["size"][0] / 2 * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] += parameter["size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )
                elif rot_direction == 1:
                    parameter["rotation"][1] += door_rot_angle
                    parameter["position"][0] += (
                        parameter["size"][0] / 2 * (1 - np.cos(door_rot_angle_radian))
                    )
                    parameter["position"][2] += (
                        parameter["size"][0] / 2 * np.sin(door_rot_angle_radian)
                    )
                    parameter["position"][0] -= parameter["size"][2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += parameter["size"][2] * (
                        1 - np.cos(door_rot_angle_radian)
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cylindrical_Connecter":
                body_parameter = concepts[0]["parameters"]
                door_parameter = concepts[1]["parameters"]
                if door_type == "Sunken_Door" or door_type == "Cuboidal_Door":
                    inner_door_size = door_parameter["size"]
                else:
                    inner_door_size = door_parameter["main_size"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                number_of_connector = np.random.randint(1, 4)

                height_interval_between_connectors = body_parameter["size"][1] / (
                    number_of_connector + 1
                )
                cur_height_offset = height_interval_between_connectors
                parameter["size"][1] = (
                    body_parameter["size"][1]
                    / (number_of_connector + 1)
                    * randRange(1, 0.5, 0.7)[0]
                )
                if number_of_connector == 1:
                    parameter["size"][1] = (
                        body_parameter["size"][1] * randRange(1, 0.5, 0.7)[0]
                    )

                if door_type == "Sunken_Door" or door_type == "Cuboidal_Door":
                    parameter["size"][0] = (
                        door_parameter["size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                    )
                else:
                    parameter["size"][0] = (
                        door_parameter["main_size"][2] / 2 * randRange(1, 0.8, 1.0)[0]
                    )

                if rot_direction == 0:
                    parameter["position"][0] -= (
                        inner_door_size[0] / 2
                        + parameter["size"][0] * randRange(1, 0, 0.5)[0]
                    )

                else:
                    parameter["position"][0] += (
                        inner_door_size[0] / 2
                        + parameter["size"][0] * randRange(1, 0, 0.5)[0]
                    )

                connector_offset_x = parameter["size"][0]

                for i in range(int(number_of_connector)):
                    parameter["position"][1] = (
                        body_parameter["size"][1] / 2 - cur_height_offset
                    )
                    for k, v in parameter.items():
                        if isinstance(v, list):
                            parameter[k] = np.array(v)
                    concept_copy = copy.deepcopy(concept)
                    concept_copy["parameters"] = {
                        k: v.tolist() for k, v in parameter.items()
                    }
                    new_concepts.append(concept_copy)
                    cur_height_offset += height_interval_between_connectors

            elif template == "T_Shaped_Connecter":
                body_parameter = concepts[0]["parameters"]
                door_parameter = concepts[1]["parameters"]
                if door_type == "Sunken_Door" or door_type == "Cuboidal_Door":
                    inner_door_size = door_parameter["size"]
                else:
                    inner_door_size = door_parameter["main_size"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = body_parameter["position"][1]
                parameter["position"][2] = (
                    body_parameter["position"][2] + body_parameter["size"][2] / 2
                )
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                number_of_connector = np.random.randint(2, 4)

                parameter["cylinder_size"][0] = (
                    inner_door_size[2] / 2 * randRange(1, 0.8, 1.0)[0]
                )

                if number_of_connector == 2:
                    height_interval_between_connectors = body_parameter["size"][1] / 4
                    mid_height = (
                        inner_door_size[1] - 2 * height_interval_between_connectors
                    )
                    cur_height_offset = [
                        height_interval_between_connectors,
                        mid_height,
                        height_interval_between_connectors,
                    ]
                    parameter["cylinder_size"][1] = (
                        inner_door_size[1] / 4 * randRange(1, 0.5, 0.7)[0]
                    )
                else:
                    height_interval_between_connectors = body_parameter["size"][1] / 4
                    cur_height_offset = [
                        height_interval_between_connectors,
                        height_interval_between_connectors,
                        height_interval_between_connectors,
                    ]
                    parameter["cylinder_size"][1] = (
                        inner_door_size[1] / 4 * randRange(1, 0.5, 0.7)[0]
                    )

                parameter["lateral_cuboid_size"][0] = (
                    parameter["cylinder_size"][1] * randRange(1, 0.8, 1.1)[0]
                )
                parameter["lateral_cuboid_size"][1] = (
                    parameter["cylinder_size"][1] * 2 / 3 * randRange(1, 0.8, 1.0)[0]
                )
                parameter["lateral_cuboid_size"][2] = (
                    parameter["cylinder_size"][0] * 2 * randRange(1, 0.8, 0.9)[0]
                )
                parameter["lateral_cuboid_offset"][0] = (
                    parameter["lateral_cuboid_size"][1]
                    / 4
                    * (-1 if random.random() < 0.5 else 1)
                )

                connector_offset_x = (
                    parameter["lateral_cuboid_size"][0] + parameter["cylinder_size"][0]
                )

                if rot_direction == 0:
                    parameter["position"][0] -= inner_door_size[0] / 2
                else:
                    parameter["position"][0] += inner_door_size[0] / 2
                    parameter["rotation"][1] += 180

                total_height_offset = height_interval_between_connectors

                for i in range(number_of_connector):
                    parameter["position"][1] = (
                        inner_door_size[1] / 2 - total_height_offset
                    )
                    for k, v in parameter.items():
                        if isinstance(v, list):
                            parameter[k] = np.array(v)
                    concept_copy = copy.deepcopy(concept)
                    concept_copy["parameters"] = {
                        k: v.tolist() for k, v in parameter.items()
                    }
                    new_concepts.append(concept_copy)

                    total_height_offset += cur_height_offset[i]

            elif template == "Trifold_Handle":
                handle_type = "Trifold_Handle"
                door_parameter = concepts[1]["parameters"]
                if door_type == "Cuboidal_Door":
                    inner_door_size = copy.deepcopy(door_parameter["size"])
                elif door_type == "Behind_Double_Layer_Door":
                    inner_door_size = copy.deepcopy(door_parameter["main_size"])
                elif door_type == "Sunken_Door":
                    inner_door_size = copy.deepcopy(door_parameter["sunken_size"])
                    inner_door_size[2] = door_parameter["size"][2]
                else:
                    inner_door_size = copy.deepcopy(door_parameter["front_size"])
                    inner_door_size[2] = (
                        door_parameter["front_size"][2] + door_parameter["main_size"][2]
                    )
                parameter["position"][0] = door_parameter["position"][0]
                parameter["position"][1] = door_parameter["position"][1]
                parameter["position"][2] = door_parameter["position"][2]
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                parameter["bottom_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["bottom_size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["bottom_size"][2] *= randRange(1, 0.8, 1.2)[0]

                parameter["top_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["top_size"][1] *= randRange(1, 0.5, 1.2)[0]
                parameter["top_size"][2] *= randRange(1, 0.8, 1.2)[0]

                parameter["bottom_seperation"][0] = (
                    parameter["top_size"][1] - parameter["bottom_size"][1]
                )
                cur_x_offset = inner_door_size[0] / 2 * randRange(1, 0.3, 0.8)[0]
                if rot_direction == 1:
                    cur_x_offset = -inner_door_size[0] / 2 * randRange(1, 0.3, 0.8)[0]
                if door_type == "Sunken_Door":
                    cur_x_offset = (
                        inner_door_size[0] / 2
                        + (
                            door_parameter["size"][0] / 2
                            - inner_door_size[0] / 2
                            - door_parameter["sunken_offset"][0]
                        )
                        / 2
                    )
                    if rot_direction == 1:
                        cur_x_offset = -(
                            inner_door_size[0] / 2
                            + (
                                door_parameter["size"][0] / 2
                                - inner_door_size[0] / 2
                                + door_parameter["sunken_offset"][0]
                            )
                            / 2
                        )
                door_rot_angle_radian = door_rot_angle / 180 * np.pi
                cur_y_offset = (
                    (inner_door_size[1] - parameter["top_size"][1])
                    / 2
                    * randRange(1, 0.0, 0.5)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )

                handle_offset_x = cur_x_offset
                handle_offset_y = cur_y_offset

                handle_size_x = parameter["bottom_size"][0]
                handle_size_y = parameter["top_size"][1]

                if rot_direction == 0:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] -= inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                elif rot_direction == 1:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] -= cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] += inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Claw_Handle":
                handle_type = "Claw_Handle"
                door_parameter = new_concepts[1]["parameters"]
                if door_type == "Cuboidal_Door":
                    inner_door_size = copy.deepcopy(door_parameter["size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Behind_Double_Layer_Door":
                    inner_door_size = copy.deepcopy(door_parameter["main_size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Sunken_Door":
                    inner_door_size = copy.deepcopy(door_parameter["sunken_size"])
                    inner_door_offset = [
                        door_parameter["sunken_offset"][0],
                        door_parameter["sunken_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["size"])
                else:
                    inner_door_size = copy.deepcopy(door_parameter["front_size"])
                    inner_door_size[2] = (
                        door_parameter["front_size"][2] + door_parameter["main_size"][2]
                    )
                    inner_door_offset = [
                        door_parameter["front_offset"][0],
                        door_parameter["front_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["main_size"])

                parameter["position"][0] = door_parameter["position"][0]
                parameter["position"][1] = door_parameter["position"][1]
                parameter["position"][2] = door_parameter["position"][2]
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                parameter["bottom_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["bottom_size"][1] *= randRange(1, 0.8, 1.2)[0]

                parameter["fork_size"][0] = (
                    inner_door_size[0] / 2 * randRange(1, 0.15, 0.5)[0]
                )
                parameter["fork_size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["fork_size"][2] = (
                    parameter["fork_size"][1] * randRange(1, 0.8, 0.9)[0]
                )
                parameter["fork_offset"][0] = (
                    parameter["bottom_size"][1] * randRange(1, 0.0, 0.3)[0]
                )
                parameter["fork_tilt_rotation"][0] = np.random.uniform(0, 10)
                num_of_fork = np.random.randint(1, 5)
                parameter["num_forks"] = np.array([int(num_of_fork)])

                if rot_direction == 0:
                    x_offset_lower_bound = np.minimum(
                        inner_door_size[0] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                        - inner_door_offset[0],
                        outer_door_size[0] / 2
                        - connector_offset_x
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0]),
                    )
                    x_offset_upper_bound = (
                        inner_door_size[0] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                        + inner_door_offset[0]
                    )
                    cur_x_offset = np.random.uniform(
                        -x_offset_lower_bound, x_offset_upper_bound
                    )
                else:
                    x_offset_lower_bound = (
                        inner_door_size[0] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                        - inner_door_offset[0]
                    )
                    x_offset_upper_bound = np.minimum(
                        inner_door_size[0] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                        + inner_door_offset[0],
                        outer_door_size[0] / 2
                        - connector_offset_x
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0]),
                    )
                    cur_x_offset = np.random.uniform(
                        -x_offset_lower_bound, x_offset_upper_bound
                    )

                door_rot_angle_radian = door_rot_angle / 180 * np.pi
                cur_y_offset = (
                    (
                        inner_door_size[1] / 2
                        - (parameter["bottom_size"][0] + parameter["fork_size"][0])
                    )
                    * randRange(1, 0.0, 0.5)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["position"][1] += cur_y_offset
                handle_offset_x = cur_x_offset
                handle_offset_y = cur_y_offset

                handle_size_x = (
                    parameter["fork_size"][0] + parameter["bottom_size"][0]
                ) * 2
                handle_size_y = (
                    parameter["fork_size"][0] + parameter["bottom_size"][0]
                ) * 2

                parameter["rotation"][2] += np.random.uniform(0, 360)

                if rot_direction == 0:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] -= inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                elif rot_direction == 1:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] -= cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] += inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Round_Handle":
                handle_type = "Claw_Handle"
                door_parameter = new_concepts[1]["parameters"]
                if door_type == "Cuboidal_Door":
                    inner_door_size = copy.deepcopy(door_parameter["size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Behind_Double_Layer_Door":
                    inner_door_size = copy.deepcopy(door_parameter["main_size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Sunken_Door":
                    inner_door_size = copy.deepcopy(door_parameter["sunken_size"])
                    inner_door_offset = [
                        door_parameter["sunken_offset"][0],
                        door_parameter["sunken_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["size"])
                else:
                    inner_door_size = copy.deepcopy(door_parameter["front_size"])
                    inner_door_size[2] = (
                        door_parameter["front_size"][2] + door_parameter["main_size"][2]
                    )
                    inner_door_offset = [
                        door_parameter["front_offset"][0],
                        door_parameter["front_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["main_size"])

                parameter["position"][0] = door_parameter["position"][0]
                parameter["position"][1] = door_parameter["position"][1]
                parameter["position"][2] = door_parameter["position"][2]
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                parameter["bottom_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["bottom_size"][1] *= randRange(1, 0.8, 1.2)[0]

                parameter["fork_size"][0] = (
                    inner_door_size[0] / 2 * randRange(1, 0.1, 0.5)[0]
                )
                parameter["fork_size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["fork_size"][2] = (
                    parameter["fork_size"][1] * randRange(1, 0.8, 0.9)[0]
                )
                parameter["fork_offset"][0] = (
                    parameter["bottom_size"][1] * randRange(1, 0.0, 0.3)[0]
                )
                parameter["fork_tilt_rotation"][0] = np.random.uniform(0, 10)
                num_of_fork = np.random.randint(3, 5)
                parameter["num_forks"] = np.array([int(num_of_fork)])

                parameter["circle_size"][0] = (
                    parameter["fork_size"][0] + parameter["bottom_size"][0]
                ) * randRange(1, 1.15, 1.2)[0]
                parameter["circle_size"][1] = parameter["fork_size"][2]

                if rot_direction == 0:
                    x_offset_lower_bound = np.minimum(
                        inner_door_size[0] / 2
                        - parameter["circle_size"][0]
                        - inner_door_offset[0],
                        outer_door_size[0] / 2
                        - connector_offset_x
                        - parameter["circle_size"][0],
                    )
                    x_offset_upper_bound = (
                        inner_door_size[0] / 2
                        - parameter["circle_size"][0]
                        + inner_door_offset[0]
                    )
                    cur_x_offset = np.random.uniform(
                        -x_offset_lower_bound, x_offset_upper_bound
                    )
                else:
                    x_offset_lower_bound = (
                        inner_door_size[0] / 2
                        - parameter["circle_size"][0]
                        - inner_door_offset[0]
                    )
                    x_offset_upper_bound = np.minimum(
                        inner_door_size[0] / 2
                        - parameter["circle_size"][0]
                        + inner_door_offset[0],
                        outer_door_size[0] / 2
                        - connector_offset_x
                        - parameter["circle_size"][0],
                    )
                    cur_x_offset = np.random.uniform(
                        -x_offset_lower_bound, x_offset_upper_bound
                    )
                door_rot_angle_radian = door_rot_angle / 180 * np.pi
                cur_y_offset = (
                    (inner_door_size[1] / 2 - parameter["circle_size"][0])
                    * randRange(1, 0.0, 0.5)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["position"][1] += cur_y_offset
                handle_offset_x = cur_x_offset
                handle_offset_y = cur_y_offset
                handle_size_x = parameter["circle_size"][0] * 2
                handle_size_y = parameter["circle_size"][0] * 2

                parameter["rotation"][2] += np.random.uniform(0, 360)

                if rot_direction == 0:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] -= inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                elif rot_direction == 1:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] -= cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] += inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_Controller":
                door_parameter = new_concepts[1]["parameters"]
                if door_type == "Cuboidal_Door":
                    inner_door_size = copy.deepcopy(door_parameter["size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Behind_Double_Layer_Door":
                    inner_door_size = copy.deepcopy(door_parameter["main_size"])
                    inner_door_offset = [0, 0]
                    outer_door_size = copy.deepcopy(inner_door_size)
                elif door_type == "Sunken_Door":
                    inner_door_size = copy.deepcopy(door_parameter["sunken_size"])
                    inner_door_offset = [
                        door_parameter["sunken_offset"][0],
                        door_parameter["sunken_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["size"])
                else:
                    inner_door_size = copy.deepcopy(door_parameter["front_size"])
                    inner_door_size[2] = (
                        door_parameter["front_size"][2] + door_parameter["main_size"][2]
                    )
                    inner_door_offset = [
                        door_parameter["front_offset"][0],
                        door_parameter["front_offset"][1],
                    ]
                    outer_door_size = copy.deepcopy(door_parameter["main_size"])

                parameter["position"][0] = door_parameter["position"][0]
                parameter["position"][1] = door_parameter["position"][1]
                parameter["position"][2] = door_parameter["position"][2]
                parameter["rotation"][0] = door_parameter["rotation"][0]
                parameter["rotation"][1] = door_parameter["rotation"][1]
                parameter["rotation"][2] = door_parameter["rotation"][2]

                bottom_size_mul = (
                    2 * randRange(1, 0.4, 1.0)[0] if random.random() < 0.5 else 1
                )

                parameter["size"][2] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][0] = (
                    inner_door_size[0] / 4 * randRange(1, 0.5, 0.7)[0] * bottom_size_mul
                )
                parameter["size"][1] = (
                    inner_door_size[1] / 5 * randRange(1, 0.8, 1.2)[0] * bottom_size_mul
                )

                outer_x_lower_bound = -inner_door_size[0] / 2 + inner_door_offset[0]
                other_x_lower_bound = -outer_door_size[0] / 2 + (
                    connector_offset_x if rot_direction == 0 else 0
                )
                outer_x_lower_bound = np.maximum(
                    outer_x_lower_bound, other_x_lower_bound
                )
                outer_x_upper_bound = (
                    inner_door_size[0] / 2
                    + inner_door_offset[0]
                    - (connector_offset_x if rot_direction == 1 else 0)
                )
                other_x_upper_bound = outer_door_size[0] / 2 - (
                    connector_offset_x if rot_direction == 1 else 0
                )
                outer_x_upper_bound = np.minimum(
                    outer_x_upper_bound, other_x_upper_bound
                )
                inner_x_lower_bound = np.maximum(
                    handle_offset_x - handle_size_x / 2, outer_x_lower_bound
                )
                inner_x_upper_bound = np.minimum(
                    handle_offset_x + handle_size_x / 2, outer_x_upper_bound
                )

                radius_x_1 = inner_x_lower_bound - outer_x_lower_bound
                radius_x_2 = outer_x_upper_bound - inner_x_upper_bound
                radius_x_max = np.maximum(radius_x_1, radius_x_2)

                size_mul_radius = bottom_size_mul
                while radius_x_max <= parameter["size"][0]:
                    size_mul_radius *= 0.7
                    parameter["size"][0] = (
                        inner_door_size[0]
                        / 4
                        * randRange(1, 0.5, 0.7)[0]
                        * size_mul_radius
                    )

                radius_x = parameter["size"][0] / 2
                radius_y = parameter["size"][1] / 2
                x_lower = outer_x_lower_bound + radius_x
                x_upper = outer_x_upper_bound - radius_x
                y_lower = -inner_door_size[1] / 2 + inner_door_offset[1] + radius_y
                y_upper = inner_door_size[1] / 2 + inner_door_offset[1] - radius_y

                if door_type != "Sunken_Door" or handle_type != "Trifold_Handle":
                    exp_x_lower = np.maximum(
                        handle_offset_x - handle_size_x / 2 - radius_x, x_lower
                    )
                    exp_x_upper = np.minimum(
                        handle_offset_x + handle_size_x / 2 + radius_x, x_upper
                    )
                    exp_y_lower = np.maximum(
                        -handle_size_y / 2 + handle_offset_y - radius_y, y_lower
                    )
                    exp_y_upper = np.minimum(
                        handle_size_y / 2 + handle_offset_y + radius_y, y_upper
                    )
                else:
                    exp_x_lower = x_upper + 1
                    exp_x_upper = x_upper + 1
                    exp_y_lower = y_upper + 1
                    exp_y_upper = y_upper + 1

                points = []

                count = 0
                if (
                    x_lower >= exp_x_lower
                    and x_upper <= exp_x_upper
                    and y_lower >= exp_y_lower
                    and y_upper <= exp_y_upper
                ):
                    continue
                while len(points) < 1:
                    if count > 50:
                        break
                    x = np.random.uniform(x_lower, x_upper)
                    y = np.random.uniform(y_lower, y_upper)
                    if not (
                        exp_x_lower <= x <= exp_x_upper
                        and exp_y_lower <= y <= exp_y_upper
                    ):
                        points.append([x, y])
                    else:
                        count += 1
                if count > 50:
                    continue

                cur_x_offset = points[0][0]
                cur_y_offset = points[0][1]

                parameter["position"][1] += cur_y_offset
                door_rot_angle_radian = door_rot_angle / 180 * np.pi

                if rot_direction == 0:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] -= inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                elif rot_direction == 1:
                    parameter["position"][0] += cur_x_offset * np.cos(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] -= cur_x_offset * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][0] += inner_door_size[2] * np.sin(
                        door_rot_angle_radian
                    )
                    parameter["position"][2] += inner_door_size[2] * np.cos(
                        door_rot_angle_radian
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cuboidal_Leg":
                body_parameter = concepts[0]["parameters"]
                parameter["position"][0] = body_parameter["position"][0]
                parameter["position"][1] = (
                    body_parameter["position"][1] - body_parameter["size"][1] / 2
                )
                parameter["position"][2] = body_parameter["position"][2]
                parameter["rotation"][0] = body_parameter["rotation"][0]
                parameter["rotation"][1] = body_parameter["rotation"][1]
                parameter["rotation"][2] = body_parameter["rotation"][2]

                parameter["front_legs_size"][0] = (
                    body_parameter["size"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["front_legs_size"][1] = (
                    body_parameter["size"][1] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["front_legs_size"][2] = (
                    body_parameter["size"][2] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][0] = (
                    parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][1] = parameter["front_legs_size"][1]
                parameter["rear_legs_size"][2] = (
                    parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["legs_separation"][0] = (
                    body_parameter["size"][0] - 2 * parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 0.9)[0]
                parameter["legs_separation"][1] = (
                    body_parameter["size"][0] - 2 * parameter["rear_legs_size"][0]
                ) * randRange(1, 0.7, 0.9)[0]
                parameter["legs_separation"][2] = (
                    body_parameter["size"][2]
                    - parameter["front_legs_size"][2]
                    - parameter["rear_legs_size"][2]
                ) * randRange(1, 0.7, 0.9)[0]

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
        safe_type = get_safe_type()
        existing_concept_templates = concept_template_existence(safe_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, safe_type)

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
            safe_type = get_safe_type()
            existing_concept_templates = concept_template_existence(safe_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, safe_type)
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
