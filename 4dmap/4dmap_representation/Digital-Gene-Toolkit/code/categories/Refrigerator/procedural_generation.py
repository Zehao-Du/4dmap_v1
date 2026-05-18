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


def get_refrigerator_type():
    refrigerator_type = "regular"
    return refrigerator_type


def concept_template_existence(refrigerator_type):
    concept_template_variation = {
        "body": {
            "template": [
                "Cuboidal_Body",
                "Double_Layer_Body",
                "Left_Right_Double_Layer_Body",
            ],
            "necessary": True,
        },
        "door": {"template": ["Cuboidal_Door", "Sunken_Door"], "necessary": True},
        "handle": {
            "template": [
                "Cuboidal_Handle",
                "Trifold_Handle",
                "Trifold_Curve_Handle",
                "Curve_Handle",
            ],
            "necessary": True,
        },
        "vessel": {"template": ["Cuboidal_Vessel"], "necessary": False},
        "tray": {"template": ["Flat_Tray", "Drawer_Like_Tray"], "necessary": False},
        "leg": {"template": ["Multilevel_Leg"], "necessary": False},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        elif random.random() < 0.5:
            templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, refrigerator_type):
    new_concepts = []

    door_psuedo_type = ""

    body_type = ""

    door_parameters = []

    for concept in concepts:
        template = concept["template"]
        parameters = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Cuboidal_Body":
            init_param_size = copy.deepcopy(parameters["size"])
            parameters["size"] = randRange(3, 0.8, 1.25) * init_param_size
            while parameters["size"][1] / parameters["size"][0] < 1.8:
                parameters["size"] = randRange(3, 0.8, 1.25) * init_param_size
            parameters["thickness"] = randRange(5, 0.8, 1.5) * parameters["thickness"]
            parameters["thickness"][1] = parameters["thickness"][0]
            parameters["thickness"][3] = parameters["thickness"][2]

            parameters["position"] = np.array([0, 0, 0])
            parameters["rotation"] = np.array([0, 0, 0])

            door_psuedo_type = "single"
            body_type = "Cuboidal_Body"

            concept["parameters"] = {k: v.tolist() for k, v in parameters.items()}
            new_concepts.append(concept)
            body_params = concept["parameters"]

        elif template == "Double_Layer_Body":
            init_param_size = copy.deepcopy(parameters["size"])
            parameters["size"] = randRange(3, 0.8, 1.25) * init_param_size
            while parameters["size"][1] / parameters["size"][0] < 1.8:
                parameters["size"] = randRange(3, 0.8, 1.25) * init_param_size
            parameters["thickness"] = randRange(5, 0.8, 1.5) * parameters["thickness"]
            parameters["thickness"][1] = parameters["thickness"][0]
            parameters["thickness"][3] = parameters["thickness"][2]

            parameters["clapboard_size"] = np.array(
                [
                    randRange(1, 0.7, 1.3)[0] * parameters["clapboard_size"][0],
                    (parameters["size"][2] - 2 * parameters["thickness"][4])
                    * randRange(1, 0.7, 1.0)[0],
                ]
            )
            parameters["clapboard_offset"] = np.array(
                [
                    randRange(1, -0.15, 0.15)[0]
                    * (
                        parameters["size"][1]
                        - 2 * parameters["thickness"][0]
                        - 0.5 * parameters["clapboard_size"][0]
                    )
                ]
            )

            parameters["position"] = np.array([0, 0, 0])
            parameters["rotation"] = np.array([0, 0, 0])

            door_num = random.choice([2, 3])
            if door_num == 3:
                door_psuedo_type = "upper2_lower1"
            else:
                door_psuedo_type = "upper1_lower1"
            body_type = "Double_Layer_Body"

            concept["parameters"] = {k: v.tolist() for k, v in parameters.items()}
            new_concepts.append(concept)
            body_params = concept["parameters"]

        elif template == "Left_Right_Double_Layer_Body":
            init_param_size = copy.deepcopy(parameters["size"])
            parameters["size"] = randRange(3, 0.8, 1.25) * init_param_size
            while parameters["size"][1] / parameters["size"][0] < 1.8:
                parameters["size"] = randRange(3, 0.8, 1.25) * init_param_size
            parameters["thickness"] = randRange(5, 0.8, 1.5) * parameters["thickness"]
            parameters["thickness"][1] = parameters["thickness"][0]
            parameters["thickness"][3] = parameters["thickness"][2]

            parameters["size"][0] *= 1.2
            parameters["clapboard_size"] = np.array(
                [
                    randRange(1, 0.7, 1.3)[0] * parameters["clapboard_size"][0],
                    (parameters["size"][2] - 2 * parameters["thickness"][4])
                    * randRange(1, 0.7, 1.0)[0],
                ]
            )
            parameters["clapboard_offset"] = np.array(
                [
                    randRange(1, -0.1, 0.1)[0]
                    * (
                        parameters["size"][0]
                        - 2 * parameters["thickness"][2]
                        - 0.5 * parameters["clapboard_size"][0]
                    )
                ]
            )

            parameters["position"] = np.array([0, 0, 0])
            parameters["rotation"] = np.array([0, 0, 0])

            door_psuedo_type = "left1_right1"
            body_type = "Left_Right_Double_Layer_Body"

            concept["parameters"] = {k: v.tolist() for k, v in parameters.items()}
            new_concepts.append(concept)
            body_params = concept["parameters"]

        elif template == "Cuboidal_Door":
            if door_psuedo_type == "single":
                parameters["size"] = np.array(
                    [
                        body_params["size"][0],
                        body_params["size"][1],
                        body_params["thickness"][4],
                    ]
                )
                rotation = randRange(1, 0, 90)[0]
                door_psuedo_rotation = random.choice(["left", "right"])
                if door_psuedo_rotation == "right":
                    parameters["rotation"] = np.array([0, rotation, 0])
                    parameters["position"] = np.array(
                        [
                            -0.5
                            * parameters["size"][0]
                            * np.cos(rotation * np.pi / 180)
                            + 0.5 * parameters["size"][0],
                            0,
                            0.5 * parameters["size"][0] * np.sin(rotation * np.pi / 180)
                            + 0.5 * body_params["size"][2],
                        ]
                    )
                elif door_psuedo_rotation == "left":
                    parameters["rotation"] = np.array([0, -rotation, 0])
                    parameters["position"] = np.array(
                        [
                            0.5 * parameters["size"][0] * np.cos(rotation * np.pi / 180)
                            - 0.5 * parameters["size"][0],
                            0,
                            0.5 * parameters["size"][0] * np.sin(rotation * np.pi / 180)
                            + 0.5 * body_params["size"][2],
                        ]
                    )
                concept["parameters"] = {k: v.tolist() for k, v in parameters.items()}
                new_concepts.append(concept)
                door_parameters.append(concept["parameters"])

            elif door_psuedo_type == "upper2_lower1":
                basic_upper_size = np.array(
                    [
                        body_params["size"][0] - 1.4 * body_params["thickness"][2],
                        body_params["size"][1] / 2
                        - body_params["clapboard_offset"][0]
                        - 0.5 * body_params["clapboard_size"][0]
                        - 0.01,
                        parameters["size"][2] * randRange(1, 0.7, 1.2)[0],
                    ]
                )

                left_upper_door_param = copy.deepcopy(parameters)
                left_upper_door_concept = copy.deepcopy(concept)
                left_upper_door_param["size"] = np.array(
                    [
                        basic_upper_size[0] * randRange(1, 0.3, 0.7)[0],
                        basic_upper_size[1],
                        basic_upper_size[2],
                    ]
                )
                rotation = randRange(1, 0, 90)[0]
                left_upper_door_param["rotation"] = np.array([0, -rotation, 0])
                left_upper_door_param["position"] = np.array(
                    [
                        -0.5 * body_params["size"][0]
                        + 0.7 * body_params["thickness"][2]
                        + 0.5
                        * left_upper_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180),
                        0.5 * body_params["size"][1]
                        - 0.5 * left_upper_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * left_upper_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                left_upper_door_concept["parameters"] = {
                    k: v.tolist() for k, v in left_upper_door_param.items()
                }
                new_concepts.append(left_upper_door_concept)
                door_parameters.append(left_upper_door_concept["parameters"])

                right_upper_door_param = copy.deepcopy(parameters)
                right_upper_door_concept = copy.deepcopy(concept)
                right_upper_door_param["size"] = np.array(
                    [
                        basic_upper_size[0] - left_upper_door_param["size"][0],
                        basic_upper_size[1],
                        basic_upper_size[2],
                    ]
                )
                rotation = randRange(1, 0, 90)[0]
                right_upper_door_param["rotation"] = np.array([0, rotation, 0])
                right_upper_door_param["position"] = np.array(
                    [
                        0.5 * body_params["size"][0]
                        - 0.7 * body_params["thickness"][2]
                        - 0.5
                        * right_upper_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180),
                        0.5 * body_params["size"][1]
                        - 0.5 * right_upper_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * right_upper_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                right_upper_door_concept["parameters"] = {
                    k: v.tolist() for k, v in right_upper_door_param.items()
                }
                new_concepts.append(right_upper_door_concept)
                door_parameters.append(right_upper_door_concept["parameters"])

                lower_door_param = copy.deepcopy(parameters)
                lower_door_concept = copy.deepcopy(concept)
                lower_door_param["size"] = np.array(
                    [
                        basic_upper_size[0],
                        body_params["size"][1]
                        - basic_upper_size[1]
                        - 0.02
                        - body_params["clapboard_size"][0],
                        basic_upper_size[2],
                    ]
                )
                rotation = randRange(1, 0, 90)[0]
                lower_door_param["rotation"] = np.array([0, -rotation, 0])
                lower_door_param["position"] = np.array(
                    [
                        0.5
                        * lower_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180)
                        - 0.5 * lower_door_param["size"][0],
                        -0.5 * body_params["size"][1]
                        + 0.5 * lower_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * lower_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                lower_door_concept["parameters"] = {
                    k: v.tolist() for k, v in lower_door_param.items()
                }
                new_concepts.append(lower_door_concept)
                door_parameters.append(lower_door_concept["parameters"])

            elif door_psuedo_type == "upper1_lower1":
                upper_door_param = copy.deepcopy(parameters)
                upper_door_concept = copy.deepcopy(concept)
                upper_door_param["size"] = np.array(
                    [
                        body_params["size"][0] - 1.4 * body_params["thickness"][2],
                        body_params["size"][1] / 2
                        - body_params["clapboard_offset"][0]
                        - 0.5 * body_params["clapboard_size"][0]
                        - 0.01,
                        parameters["size"][2] * randRange(1, 0.7, 1.2)[0],
                    ]
                )
                rotation = randRange(1, 0, 90)[0]
                upper_door_param["rotation"] = np.array([0, -rotation, 0])
                upper_door_param["position"] = np.array(
                    [
                        0.5
                        * upper_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180)
                        - 0.5 * upper_door_param["size"][0],
                        0.5 * body_params["size"][1]
                        - 0.5 * upper_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * upper_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                upper_door_concept["parameters"] = {
                    k: v.tolist() for k, v in upper_door_param.items()
                }
                new_concepts.append(upper_door_concept)
                door_parameters.append(upper_door_concept["parameters"])

                lower_door_param = copy.deepcopy(parameters)
                lower_door_concept = copy.deepcopy(concept)
                lower_door_param["size"] = np.array(
                    [
                        upper_door_param["size"][0],
                        body_params["size"][1]
                        - upper_door_param["size"][1]
                        - 0.02
                        - body_params["clapboard_size"][0],
                        upper_door_param["size"][2],
                    ]
                )
                rotation = randRange(1, 0, 90)[0]
                lower_door_param["rotation"] = np.array([0, -rotation, 0])
                lower_door_param["position"] = np.array(
                    [
                        0.5
                        * lower_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180)
                        - 0.5 * lower_door_param["size"][0],
                        -0.5 * body_params["size"][1]
                        + 0.5 * lower_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * lower_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                lower_door_concept["parameters"] = {
                    k: v.tolist() for k, v in lower_door_param.items()
                }
                new_concepts.append(lower_door_concept)
                door_parameters.append(lower_door_concept["parameters"])

            elif door_psuedo_type == "left1_right1":
                left_door_param = copy.deepcopy(parameters)
                left_door_concept = copy.deepcopy(concept)
                left_door_param["size"] = np.array(
                    [
                        body_params["size"][0] / 2
                        + body_params["clapboard_offset"][0]
                        - 0.5 * body_params["clapboard_size"][0]
                        - 0.01,
                        body_params["size"][1] - body_params["thickness"][0],
                        parameters["size"][2] * randRange(1, 0.7, 1.2)[0],
                    ]
                )
                rotation = randRange(1, 0, 90)[0]
                left_door_param["rotation"] = np.array([0, -rotation, 0])
                left_door_param["position"] = np.array(
                    [
                        -0.5 * body_params["size"][0]
                        + 0.7 * body_params["thickness"][2]
                        + 0.5
                        * left_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180),
                        0.5 * body_params["size"][1] - 0.5 * left_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * left_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                left_door_concept["parameters"] = {
                    k: v.tolist() for k, v in left_door_param.items()
                }
                new_concepts.append(left_door_concept)
                door_parameters.append(left_door_concept["parameters"])

                right_door_param = copy.deepcopy(parameters)
                right_door_concept = copy.deepcopy(concept)
                right_door_param["size"] = np.array(
                    [
                        body_params["size"][0]
                        - left_door_param["size"][0]
                        - 0.02
                        - body_params["clapboard_size"][0],
                        left_door_param["size"][1],
                        left_door_param["size"][2],
                    ]
                )
                rotation = randRange(1, 0, 90)[0]
                right_door_param["rotation"] = np.array([0, rotation, 0])
                right_door_param["position"] = np.array(
                    [
                        0.5 * body_params["size"][0]
                        - 0.7 * body_params["thickness"][2]
                        - 0.5
                        * right_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180),
                        -0.5 * body_params["size"][1]
                        + 0.5 * right_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * right_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                right_door_concept["parameters"] = {
                    k: v.tolist() for k, v in right_door_param.items()
                }
                new_concepts.append(right_door_concept)
                door_parameters.append(right_door_concept["parameters"])

        elif template == "Sunken_Door":
            if door_psuedo_type == "single":
                parameters["size"] = np.array(
                    [
                        body_params["size"][0],
                        body_params["size"][1],
                        body_params["thickness"][-1],
                    ]
                )
                parameters["sunken_offset"] = np.zeros(2)
                parameters["sunken_size"] = randRange(3, 0.6, 0.8) * parameters["size"]
                rotation = randRange(1, 0, 90)[0]
                door_psuedo_rotation = random.choice(["left", "right"])
                if door_psuedo_rotation[0] == "right":
                    parameters["rotation"] = np.array([0, rotation, 0])
                    parameters["position"] = np.array(
                        [
                            -0.5
                            * parameters["size"][0]
                            * np.cos(rotation * np.pi / 180)
                            + 0.5 * parameters["size"][0],
                            0,
                            0.5 * parameters["size"][0] * np.sin(rotation * np.pi / 180)
                            + 0.5 * body_params["size"][2],
                        ]
                    )
                elif door_psuedo_rotation[0] == "left":
                    parameters["rotation"] = np.array([0, -rotation, 0])
                    parameters["position"] = np.array(
                        [
                            0.5 * parameters["size"][0] * np.cos(rotation * np.pi / 180)
                            - 0.5 * parameters["size"][0],
                            0,
                            0.5 * parameters["size"][0] * np.sin(rotation * np.pi / 180)
                            + 0.5 * body_params["size"][2],
                        ]
                    )

                concept["parameters"] = {k: v.tolist() for k, v in parameters.items()}
                new_concepts.append(concept)
                door_parameters.append(concept["parameters"])

            elif door_psuedo_type == "upper2_lower1":
                basic_upper_size = np.array(
                    [
                        body_params["size"][0] - 1.4 * body_params["thickness"][2],
                        body_params["size"][1] / 2
                        - body_params["clapboard_offset"][0]
                        - 0.5 * body_params["clapboard_size"][0]
                        - 0.01,
                        parameters["size"][2] * randRange(1, 0.7, 1.2)[0],
                    ]
                )

                left_upper_door_param = copy.deepcopy(parameters)
                left_upper_door_concept = copy.deepcopy(concept)
                left_upper_door_param["size"] = np.array(
                    [
                        basic_upper_size[0] * randRange(1, 0.3, 0.7)[0],
                        basic_upper_size[1],
                        basic_upper_size[2],
                    ]
                )
                left_upper_door_param["sunken_offset"] = np.zeros(2)
                left_upper_door_param["sunken_size"] = (
                    randRange(3, 0.6, 0.8) * left_upper_door_param["size"]
                )
                rotation = randRange(1, 0, 90)[0]
                left_upper_door_param["rotation"] = np.array([0, -rotation, 0])
                left_upper_door_param["position"] = np.array(
                    [
                        -0.5 * body_params["size"][0]
                        + 0.7 * body_params["thickness"][2]
                        + 0.5
                        * left_upper_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180),
                        0.5 * body_params["size"][1]
                        - 0.5 * left_upper_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * left_upper_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                left_upper_door_concept["parameters"] = {
                    k: v.tolist() for k, v in left_upper_door_param.items()
                }
                new_concepts.append(left_upper_door_concept)
                door_parameters.append(left_upper_door_concept["parameters"])

                right_upper_door_param = copy.deepcopy(parameters)
                right_upper_door_concept = copy.deepcopy(concept)
                right_upper_door_param["size"] = np.array(
                    [
                        basic_upper_size[0] - left_upper_door_param["size"][0],
                        basic_upper_size[1],
                        basic_upper_size[2],
                    ]
                )
                right_upper_door_param["sunken_offset"] = np.zeros(2)
                right_upper_door_param["sunken_size"] = (
                    randRange(3, 0.6, 0.8) * right_upper_door_param["size"]
                )
                rotation = randRange(1, 0, 90)[0]
                right_upper_door_param["rotation"] = np.array([0, rotation, 0])
                right_upper_door_param["position"] = np.array(
                    [
                        0.5 * body_params["size"][0]
                        - 0.7 * body_params["thickness"][2]
                        - 0.5
                        * right_upper_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180),
                        0.5 * body_params["size"][1]
                        - 0.5 * right_upper_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * right_upper_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                right_upper_door_concept["parameters"] = {
                    k: v.tolist() for k, v in right_upper_door_param.items()
                }
                new_concepts.append(right_upper_door_concept)
                door_parameters.append(right_upper_door_concept["parameters"])

                lower_door_param = copy.deepcopy(parameters)
                lower_door_concept = copy.deepcopy(concept)
                lower_door_param["size"] = np.array(
                    [
                        basic_upper_size[0],
                        body_params["size"][1]
                        - basic_upper_size[1]
                        - 0.02
                        - body_params["clapboard_size"][0],
                        basic_upper_size[2],
                    ]
                )
                lower_door_param["sunken_offset"] = np.zeros(2)
                lower_door_param["sunken_size"] = (
                    randRange(3, 0.6, 0.8) * lower_door_param["size"]
                )
                rotation = randRange(1, 0, 90)[0]
                lower_door_param["rotation"] = np.array([0, -rotation, 0])
                lower_door_param["position"] = np.array(
                    [
                        0.5
                        * lower_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180)
                        - 0.5 * lower_door_param["size"][0],
                        -0.5 * body_params["size"][1]
                        + 0.5 * lower_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * lower_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                lower_door_concept["parameters"] = {
                    k: v.tolist() for k, v in lower_door_param.items()
                }
                new_concepts.append(lower_door_concept)
                door_parameters.append(lower_door_concept["parameters"])

            elif door_psuedo_type == "upper1_lower1":
                upper_door_param = copy.deepcopy(parameters)
                upper_door_concept = copy.deepcopy(concept)
                upper_door_param["size"] = np.array(
                    [
                        body_params["size"][0] - 1.4 * body_params["thickness"][2],
                        body_params["size"][1] / 2
                        - body_params["clapboard_offset"][0]
                        - 0.5 * body_params["clapboard_size"][0]
                        - 0.01,
                        parameters["size"][2] * randRange(1, 0.7, 1.2)[0],
                    ]
                )
                upper_door_param["sunken_offset"] = np.zeros(2)
                upper_door_param["sunken_size"] = (
                    randRange(3, 0.6, 0.8) * upper_door_param["size"]
                )
                rotation = randRange(1, 0, 90)[0]
                upper_door_param["rotation"] = np.array([0, -rotation, 0])
                upper_door_param["position"] = np.array(
                    [
                        0.5
                        * upper_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180)
                        - 0.5 * upper_door_param["size"][0],
                        0.5 * body_params["size"][1]
                        - 0.5 * upper_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * upper_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                upper_door_concept["parameters"] = {
                    k: v.tolist() for k, v in upper_door_param.items()
                }
                new_concepts.append(upper_door_concept)
                door_parameters.append(upper_door_concept["parameters"])

                lower_door_param = copy.deepcopy(parameters)
                lower_door_concept = copy.deepcopy(concept)
                lower_door_param["size"] = np.array(
                    [
                        upper_door_param["size"][0],
                        body_params["size"][1]
                        - upper_door_param["size"][1]
                        - 0.02
                        - body_params["clapboard_size"][0],
                        upper_door_param["size"][2],
                    ]
                )
                lower_door_param["sunken_offset"] = np.zeros(2)
                lower_door_param["sunken_size"] = (
                    randRange(3, 0.6, 0.8) * lower_door_param["size"]
                )
                rotation = randRange(1, 0, 90)[0]
                lower_door_param["rotation"] = np.array([0, -rotation, 0])
                lower_door_param["position"] = np.array(
                    [
                        0.5
                        * lower_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180)
                        - 0.5 * lower_door_param["size"][0],
                        -0.5 * body_params["size"][1]
                        + 0.5 * lower_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * lower_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                lower_door_concept["parameters"] = {
                    k: v.tolist() for k, v in lower_door_param.items()
                }
                new_concepts.append(lower_door_concept)
                door_parameters.append(lower_door_concept["parameters"])

            elif door_psuedo_type == "left1_right1":
                left_door_param = copy.deepcopy(parameters)
                left_door_concept = copy.deepcopy(concept)
                left_door_param["size"] = np.array(
                    [
                        body_params["size"][0] / 2
                        + body_params["clapboard_offset"][0]
                        - 0.5 * body_params["clapboard_size"][0]
                        - 0.01,
                        body_params["size"][1] - body_params["thickness"][0],
                        parameters["size"][2] * randRange(1, 0.7, 1.2)[0],
                    ]
                )
                left_door_param["sunken_offset"] = np.zeros(2)
                left_door_param["sunken_size"] = (
                    randRange(3, 0.6, 0.8) * left_door_param["size"]
                )
                rotation = randRange(1, 0, 90)[0]
                left_door_param["rotation"] = np.array([0, -rotation, 0])
                left_door_param["position"] = np.array(
                    [
                        -0.5 * body_params["size"][0]
                        + 0.7 * body_params["thickness"][2]
                        + 0.5
                        * left_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180),
                        0.5 * body_params["size"][1] - 0.5 * left_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * left_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                left_door_concept["parameters"] = {
                    k: v.tolist() for k, v in left_door_param.items()
                }
                new_concepts.append(left_door_concept)
                door_parameters.append(left_door_concept["parameters"])

                right_door_param = copy.deepcopy(parameters)
                right_door_concept = copy.deepcopy(concept)
                right_door_param["size"] = np.array(
                    [
                        body_params["size"][0]
                        - left_door_param["size"][0]
                        - 0.02
                        - body_params["clapboard_size"][0],
                        left_door_param["size"][1],
                        left_door_param["size"][2],
                    ]
                )
                right_door_param["sunken_offset"] = np.zeros(2)
                right_door_param["sunken_size"] = (
                    randRange(3, 0.6, 0.8) * right_door_param["size"]
                )
                rotation = randRange(1, 0, 90)[0]
                right_door_param["rotation"] = np.array([0, rotation, 0])
                right_door_param["position"] = np.array(
                    [
                        0.5 * body_params["size"][0]
                        - 0.7 * body_params["thickness"][2]
                        - 0.5
                        * right_door_param["size"][0]
                        * np.cos(rotation * np.pi / 180),
                        -0.5 * body_params["size"][1]
                        + 0.5 * right_door_param["size"][1],
                        0.5 * body_params["size"][2]
                        + 0.5
                        * right_door_param["size"][0]
                        * np.sin(rotation * np.pi / 180),
                    ]
                )
                right_door_concept["parameters"] = {
                    k: v.tolist() for k, v in right_door_param.items()
                }
                new_concepts.append(right_door_concept)
                door_parameters.append(right_door_concept["parameters"])

        elif template == "Cuboidal_Handle":
            y_position_on_door_ratio = randRange(1, 0.7, 1.2)[0]
            handle_size = np.array(
                [
                    max(
                        [
                            door_parameter["size"][0]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.01, 0.03)[0],
                    min(
                        [
                            door_parameter["size"][1]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.5, 0.9)[0],
                    door_parameters[0]["size"][2] * randRange(1, 0.8, 0.9)[0],
                ]
            )

            for door_parameter in door_parameters:
                handle_concept = copy.deepcopy(concept)
                handle_parameters = copy.deepcopy(parameters)
                handle_parameters["size"] = handle_size
                if door_parameter["rotation"][1] >= 0:
                    handle_parameters["rotation"] = np.array(
                        [0, door_parameter["rotation"][1], 0]
                    )
                    handle_parameters["position"] = np.array(
                        [
                            door_parameter["position"][0]
                            + 0.5
                            * (handle_parameters["size"][2] + door_parameter["size"][2])
                            * np.sin(door_parameter["rotation"][1] * np.pi / 180)
                            - 0.42
                            * door_parameter["size"][0]
                            * np.cos(door_parameter["rotation"][1] * np.pi / 180),
                            door_parameter["position"][1]
                            + y_position_on_door_ratio
                            * (door_parameter["size"][1] - handle_parameters["size"][1])
                            / 3.5,
                            door_parameter["position"][2]
                            + 0.5
                            * (handle_parameters["size"][2] + door_parameter["size"][2])
                            * np.cos(door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.sin(door_parameter["rotation"][1] * np.pi / 180),
                        ]
                    )
                elif door_parameter["rotation"][1] < 0:
                    handle_parameters["rotation"] = np.array(
                        [0, door_parameter["rotation"][1], 0]
                    )
                    handle_parameters["position"] = np.array(
                        [
                            door_parameter["position"][0]
                            - 0.5
                            * (handle_parameters["size"][2] + door_parameter["size"][2])
                            * np.sin(-door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.cos(-door_parameter["rotation"][1] * np.pi / 180),
                            door_parameter["position"][1]
                            + y_position_on_door_ratio
                            * (door_parameter["size"][1] - handle_parameters["size"][1])
                            / 3.5,
                            door_parameter["position"][2]
                            + 0.5
                            * (handle_parameters["size"][2] + door_parameter["size"][2])
                            * np.cos(-door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.sin(-door_parameter["rotation"][1] * np.pi / 180),
                        ]
                    )

                handle_concept["parameters"] = {
                    k: v.tolist() for k, v in handle_parameters.items()
                }
                new_concepts.append(handle_concept)

        elif template == "Trifold_Handle":
            y_position_on_door_ratio = randRange(1, 0.7, 1.2)[0]
            grip_size = np.array(
                [
                    max(
                        [
                            door_parameter["size"][0]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.01, 0.03)[0],
                    min(
                        [
                            door_parameter["size"][1]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.5, 0.9)[0],
                    door_parameters[0]["size"][2] * randRange(1, 0.8, 0.9)[0],
                ]
            )
            mounting_size = np.array(
                [
                    grip_size[0],
                    grip_size[1] * randRange(1, 0.03, 0.05)[0],
                    grip_size[2] * randRange(1, 0.8, 1.5)[0],
                ]
            )
            mounting_separation = randRange(1, 0.9, 1.0)[0] * (
                grip_size[1] - mounting_size[1]
            )

            for door_parameter in door_parameters:
                handle_concept = copy.deepcopy(concept)
                handle_parameters = copy.deepcopy(parameters)
                handle_parameters["grip_size"] = grip_size
                handle_parameters["mounting_size"] = mounting_size
                handle_parameters["mounting_seperation"][0] = mounting_separation
                if door_parameter["rotation"][1] >= 0:
                    handle_parameters["rotation"] = np.array(
                        [0, door_parameter["rotation"][1], 0]
                    )
                    handle_parameters["position"] = np.array(
                        [
                            door_parameter["position"][0]
                            + 0.5
                            * (
                                handle_parameters["grip_size"][2]
                                + door_parameter["size"][2]
                            )
                            * np.sin(door_parameter["rotation"][1] * np.pi / 180)
                            - 0.42
                            * door_parameter["size"][0]
                            * np.cos(door_parameter["rotation"][1] * np.pi / 180),
                            door_parameter["position"][1]
                            + y_position_on_door_ratio
                            * (
                                door_parameter["size"][1]
                                - handle_parameters["grip_size"][1]
                            )
                            / 3.5,
                            door_parameter["position"][2]
                            + 0.5
                            * (
                                handle_parameters["grip_size"][2]
                                + door_parameter["size"][2]
                            )
                            * np.cos(door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.sin(door_parameter["rotation"][1] * np.pi / 180),
                        ]
                    )
                elif door_parameter["rotation"][1] < 0:
                    handle_parameters["rotation"] = np.array(
                        [0, door_parameter["rotation"][1], 0]
                    )
                    handle_parameters["position"] = np.array(
                        [
                            door_parameter["position"][0]
                            - 0.5
                            * (
                                handle_parameters["grip_size"][2]
                                + door_parameter["size"][2]
                            )
                            * np.sin(-door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.cos(-door_parameter["rotation"][1] * np.pi / 180),
                            door_parameter["position"][1]
                            + y_position_on_door_ratio
                            * (
                                door_parameter["size"][1]
                                - handle_parameters["grip_size"][1]
                            )
                            / 3.5,
                            door_parameter["position"][2]
                            + 0.5
                            * (
                                handle_parameters["grip_size"][2]
                                + door_parameter["size"][2]
                            )
                            * np.cos(-door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.sin(-door_parameter["rotation"][1] * np.pi / 180),
                        ]
                    )

                handle_concept["parameters"] = {
                    k: v.tolist() for k, v in handle_parameters.items()
                }
                new_concepts.append(handle_concept)

        elif template == "Trifold_Curve_Handle":
            y_position_on_door_ratio = randRange(1, 0.7, 1.2)[0]
            psuedo_grip_size = np.array(
                [
                    max(
                        [
                            door_parameter["size"][0]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.01, 0.03)[0],
                    min(
                        [
                            door_parameter["size"][1]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.5, 0.9)[0],
                    door_parameters[0]["size"][2] * randRange(1, 0.8, 0.9)[0],
                ]
            )
            curve_exist_angle = randRange(1, 0.8, 1.25)[0] * 60
            curve_size = np.array(
                [
                    psuedo_grip_size[1] / 2 / np.sin(curve_exist_angle * np.pi / 180),
                    psuedo_grip_size[1]
                    / 2
                    / np.sin(curve_exist_angle * np.pi / 180)
                    * randRange(1, 0.9, 0.95)[0],
                    door_parameters[0]["size"][2] * randRange(1, 0.8, 0.9)[0],
                ]
            )
            mounting_size = np.array(
                [
                    psuedo_grip_size[0],
                    psuedo_grip_size[1] * randRange(1, 0.03, 0.05)[0],
                    psuedo_grip_size[2] * randRange(1, 0.8, 1.5)[0],
                ]
            )
            mounting_seperation = (
                curve_size[1]
                / (curve_exist_angle * np.pi / 180)
                * np.sin(curve_exist_angle * np.pi / 180 / 2)
                * randRange(1, 1, 1.2)[0]
            )

            for door_parameter in door_parameters:
                handle_concept = copy.deepcopy(concept)
                handle_parameters = copy.deepcopy(parameters)
                handle_parameters["curve_size"] = curve_size
                handle_parameters["mounting_size"] = mounting_size
                handle_parameters["curve_exist_angle"][0] = curve_exist_angle
                handle_parameters["mounting_seperation"][0] = mounting_seperation
                if door_parameter["rotation"][1] >= 0:
                    handle_parameters["rotation"] = np.array(
                        [0, door_parameter["rotation"][1], 0]
                    )
                    handle_parameters["position"] = np.array(
                        [
                            door_parameter["position"][0]
                            + 0.5
                            * (mounting_size[2] + door_parameter["size"][2])
                            * np.sin(door_parameter["rotation"][1] * np.pi / 180)
                            - 0.42
                            * door_parameter["size"][0]
                            * np.cos(door_parameter["rotation"][1] * np.pi / 180),
                            door_parameter["position"][1]
                            + y_position_on_door_ratio
                            * (door_parameter["size"][1] - psuedo_grip_size[1])
                            / 3.5,
                            door_parameter["position"][2]
                            + 0.5
                            * (mounting_size[2] + door_parameter["size"][2])
                            * np.cos(door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.sin(door_parameter["rotation"][1] * np.pi / 180),
                        ]
                    )
                elif door_parameter["rotation"][1] < 0:
                    handle_parameters["rotation"] = np.array(
                        [0, door_parameter["rotation"][1], 0]
                    )
                    handle_parameters["position"] = np.array(
                        [
                            door_parameter["position"][0]
                            - 0.5
                            * (mounting_size[2] + door_parameter["size"][2])
                            * np.sin(-door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.cos(-door_parameter["rotation"][1] * np.pi / 180),
                            door_parameter["position"][1]
                            + y_position_on_door_ratio
                            * (door_parameter["size"][1] - psuedo_grip_size[1])
                            / 3.5,
                            door_parameter["position"][2]
                            + 0.5
                            * (mounting_size[2] + door_parameter["size"][2])
                            * np.cos(-door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.sin(-door_parameter["rotation"][1] * np.pi / 180),
                        ]
                    )

                handle_concept["parameters"] = {
                    k: v.tolist() for k, v in handle_parameters.items()
                }
                new_concepts.append(handle_concept)

        elif template == "Curve_Handle":
            y_position_on_door_ratio = randRange(1, 0.7, 1.2)[0]
            psuedo_grip_size = np.array(
                [
                    max(
                        [
                            door_parameter["size"][0]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.01, 0.03)[0],
                    min(
                        [
                            door_parameter["size"][1]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.5, 0.9)[0],
                    door_parameters[0]["size"][2] * randRange(1, 0.8, 0.9)[0],
                ]
            )
            curve_exist_angle = randRange(1, 0.8, 1.25)[0] * 60
            curve_size = np.array(
                [
                    psuedo_grip_size[1] / 2 / np.sin(curve_exist_angle * np.pi / 180),
                    psuedo_grip_size[1]
                    / 2
                    / np.sin(curve_exist_angle * np.pi / 180)
                    * randRange(1, 0.9, 0.95)[0],
                    door_parameters[0]["size"][2] * randRange(1, 0.8, 0.9)[0],
                ]
            )

            for door_parameter in door_parameters:
                handle_concept = copy.deepcopy(concept)
                handle_parameters = copy.deepcopy(parameters)
                handle_parameters["curve_size"] = curve_size
                handle_parameters["curve_exist_angle"][0] = curve_exist_angle
                if door_parameter["rotation"][1] >= 0:
                    handle_parameters["rotation"] = np.array(
                        [0, door_parameter["rotation"][1], 0]
                    )
                    handle_parameters["position"] = np.array(
                        [
                            door_parameter["position"][0]
                            + 0.5
                            * (curve_size[2] + door_parameter["size"][2])
                            * np.sin(door_parameter["rotation"][1] * np.pi / 180)
                            - 0.42
                            * door_parameter["size"][0]
                            * np.cos(door_parameter["rotation"][1] * np.pi / 180),
                            door_parameter["position"][1]
                            + y_position_on_door_ratio
                            * (door_parameter["size"][1] - psuedo_grip_size[1])
                            / 3.5,
                            door_parameter["position"][2]
                            + 0.5
                            * (curve_size[2] + door_parameter["size"][2])
                            * np.cos(door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.sin(door_parameter["rotation"][1] * np.pi / 180),
                        ]
                    )
                elif door_parameter["rotation"][1] < 0:
                    handle_parameters["rotation"] = np.array(
                        [0, door_parameter["rotation"][1], 0]
                    )
                    handle_parameters["position"] = np.array(
                        [
                            door_parameter["position"][0]
                            - 0.5
                            * (curve_size[2] + door_parameter["size"][2])
                            * np.sin(-door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.cos(-door_parameter["rotation"][1] * np.pi / 180),
                            door_parameter["position"][1]
                            + y_position_on_door_ratio
                            * (door_parameter["size"][1] - psuedo_grip_size[1])
                            / 3.5,
                            door_parameter["position"][2]
                            + 0.5
                            * (curve_size[2] + door_parameter["size"][2])
                            * np.cos(-door_parameter["rotation"][1] * np.pi / 180)
                            + 0.42
                            * door_parameter["size"][0]
                            * np.sin(-door_parameter["rotation"][1] * np.pi / 180),
                        ]
                    )

                handle_concept["parameters"] = {
                    k: v.tolist() for k, v in handle_parameters.items()
                }
                new_concepts.append(handle_concept)

        elif template == "Cuboidal_Vessel":
            vessel_num = np.random.randint(1, 4)
            outer_size = np.array(
                [
                    min(
                        [
                            door_parameter["size"][0]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.8, 0.9)[0],
                    min(
                        [
                            door_parameter["size"][1]
                            for door_parameter in door_parameters
                        ]
                    )
                    * randRange(1, 0.02, 0.05)[0],
                    door_parameters[0]["size"][2] * randRange(1, 0.8, 1.2)[0],
                ]
            )
            inner_size = outer_size * randRange(outer_size.shape[0], 0.8, 0.95)

            for door_parameter in door_parameters:
                for vessel_idx in range(vessel_num):
                    vessel_concept = copy.deepcopy(concept)
                    vessel_parameters = copy.deepcopy(parameters)
                    vessel_parameters["outer_size"] = outer_size
                    vessel_parameters["inner_size"] = inner_size
                    vessel_parameters["rotation"] = np.array(
                        [0, door_parameter["rotation"][1], 0]
                    )
                    vessel_parameters["position"] = np.array(
                        [
                            door_parameter["position"][0],
                            door_parameter["position"][1]
                            + (
                                door_parameter["size"][1]
                                / (vessel_num + 1)
                                * (vessel_idx + 1)
                            )
                            - door_parameter["size"][1] / 2,
                            door_parameter["position"][2],
                        ]
                    )
                    vessel_concept["parameters"] = {
                        k: v.tolist() for k, v in vessel_parameters.items()
                    }
                    new_concepts.append(vessel_concept)

        elif template == "Flat_Tray":
            tray_num = np.random.randint(2, 5)
            size = np.array(
                [
                    body_params["size"][0]
                    - body_params["thickness"][2]
                    - body_params["thickness"][3],
                    body_params["size"][1] * randRange(1, 0.02, 0.03)[0],
                    (body_params["size"][2] - body_params["thickness"][4])
                    * randRange(1, 0.9, 0.95)[0],
                ]
            )

            if body_type == "Cuboidal_Body":
                for tray_idx in range(tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["size"] = size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            -body_params["size"][1] / 2
                            + (
                                body_params["size"][1] / (tray_num + 1) * (tray_idx + 1)
                            ),
                            size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

            elif body_type == "Double_Layer_Body":
                if body_params["clapboard_offset"][0] > 0:
                    lower_tray_num = tray_num // 2 + 1
                    upper_tray_num = tray_num - lower_tray_num
                else:
                    upper_tray_num = tray_num // 2 + 1
                    lower_tray_num = tray_num - upper_tray_num

                for tray_idx in range(upper_tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["size"] = size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            body_params["clapboard_offset"][0]
                            + (
                                (
                                    body_params["size"][1] / 2
                                    - body_params["clapboard_offset"][0]
                                )
                                / (upper_tray_num + 1)
                                * (tray_idx + 1)
                            ),
                            size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

                for tray_idx in range(lower_tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["size"] = size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            -body_params["size"][1] / 2
                            + (
                                (
                                    body_params["size"][1] / 2
                                    + body_params["clapboard_offset"][0]
                                )
                                / (lower_tray_num + 1)
                                * (tray_idx + 1)
                            ),
                            size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

            elif body_type == "Left_Right_Double_Layer_Body":
                if body_params["clapboard_offset"][0] > 0:
                    lower_tray_num = tray_num // 2 + 1
                    upper_tray_num = tray_num - lower_tray_num
                else:
                    upper_tray_num = tray_num // 2 + 1
                    lower_tray_num = tray_num - upper_tray_num

                for tray_idx in range(upper_tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["size"] = size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            body_params["clapboard_offset"][0]
                            + (
                                (
                                    body_params["size"][1] / 2
                                    - body_params["clapboard_offset"][0]
                                )
                                / (upper_tray_num + 1)
                                * (tray_idx + 1)
                            ),
                            size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

                for tray_idx in range(lower_tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["size"] = size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            -body_params["size"][1] / 2
                            + (
                                (
                                    body_params["size"][1] / 2
                                    + body_params["clapboard_offset"][0]
                                )
                                / (lower_tray_num + 1)
                                * (tray_idx + 1)
                            ),
                            size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

        elif template == "Drawer_Like_Tray":
            tray_num = np.random.randint(2, 6)
            outer_size = np.array(
                [
                    body_params["size"][0]
                    - body_params["thickness"][2]
                    - body_params["thickness"][3],
                    body_params["size"][1] * randRange(1, 0.02, 0.03)[0],
                    (body_params["size"][2] - body_params["thickness"][4])
                    * randRange(1, 0.9, 0.95)[0],
                ]
            )
            inner_size = outer_size * randRange(3, 0.9, 0.95)

            if body_type == "Cuboidal_Body":
                for tray_idx in range(tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["outer_size"] = outer_size
                    tray_parameters["inner_size"] = inner_size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            -body_params["size"][1] / 2
                            + (
                                body_params["size"][1] / (tray_num + 1) * (tray_idx + 1)
                            ),
                            outer_size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

            elif body_type == "Double_Layer_Body":
                if body_params["clapboard_offset"][0] > 0:
                    lower_tray_num = tray_num // 2 + 1
                    upper_tray_num = tray_num - lower_tray_num
                else:
                    upper_tray_num = tray_num // 2 + 1
                    lower_tray_num = tray_num - upper_tray_num

                for tray_idx in range(upper_tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["outer_size"] = outer_size
                    tray_parameters["inner_size"] = inner_size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            body_params["clapboard_offset"][0]
                            + (
                                (
                                    body_params["size"][1] / 2
                                    - body_params["clapboard_offset"][0]
                                )
                                / (upper_tray_num + 1)
                                * (tray_idx + 1)
                            ),
                            outer_size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

                for tray_idx in range(lower_tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["outer_size"] = outer_size
                    tray_parameters["inner_size"] = inner_size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            -body_params["size"][1] / 2
                            + (
                                (
                                    body_params["size"][1] / 2
                                    + body_params["clapboard_offset"][0]
                                )
                                / (lower_tray_num + 1)
                                * (tray_idx + 1)
                            ),
                            outer_size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

            elif body_type == "Left_Right_Double_Layer_Body":
                if body_params["clapboard_offset"][0] > 0:
                    lower_tray_num = tray_num // 2 + 1
                    upper_tray_num = tray_num - lower_tray_num
                else:
                    upper_tray_num = tray_num // 2 + 1
                    lower_tray_num = tray_num - upper_tray_num

                for tray_idx in range(upper_tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["outer_size"] = outer_size
                    tray_parameters["inner_size"] = inner_size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            body_params["clapboard_offset"][0]
                            + (
                                (
                                    body_params["size"][1] / 2
                                    - body_params["clapboard_offset"][0]
                                )
                                / (upper_tray_num + 1)
                                * (tray_idx + 1)
                            ),
                            outer_size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

                for tray_idx in range(lower_tray_num):
                    tray_concept = copy.deepcopy(concept)
                    tray_parameters = copy.deepcopy(parameters)
                    tray_parameters["outer_size"] = outer_size
                    tray_parameters["inner_size"] = inner_size
                    tray_parameters["position"] = np.array(
                        [
                            0,
                            -body_params["size"][1] / 2
                            + (
                                (
                                    body_params["size"][1] / 2
                                    + body_params["clapboard_offset"][0]
                                )
                                / (lower_tray_num + 1)
                                * (tray_idx + 1)
                            ),
                            outer_size[2] / 2
                            - body_params["size"][2] / 2
                            + body_params["thickness"][4],
                        ]
                    )
                    tray_parameters["rotation"] = np.array([0, 0, 0])
                    tray_concept["parameters"] = {
                        k: v.tolist() for k, v in tray_parameters.items()
                    }
                    new_concepts.append(tray_concept)

        elif template == "Multilevel_Leg":
            parameters["num_legs"][0] = int(4)
            parameters["front_legs_size"] = np.array(
                [
                    body_params["size"][0] * randRange(1, 0.05, 0.08)[0],
                    body_params["size"][1] * randRange(1, 0.01, 0.025)[0],
                    body_params["size"][2] * randRange(1, 0.05, 0.08)[0],
                ]
            )
            parameters["rear_legs_size"] = parameters["front_legs_size"]
            parameters["legs_separation"][0] = (
                body_params["size"][0] - parameters["front_legs_size"][0]
            )
            parameters["legs_separation"][1] = (
                body_params["size"][0] - parameters["front_legs_size"][0]
            )
            parameters["legs_separation"][2] = (
                body_params["size"][2] - parameters["front_legs_size"][2]
            )
            parameters["position"] = np.array([0, -body_params["size"][1] / 2, 0])
            parameters["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameters.items()}
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
        refrigerator_type = get_refrigerator_type()
        existing_concept_templates = concept_template_existence(refrigerator_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, refrigerator_type)

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
            refrigerator_type = get_refrigerator_type()
            existing_concept_templates = concept_template_existence(refrigerator_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, refrigerator_type)
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
