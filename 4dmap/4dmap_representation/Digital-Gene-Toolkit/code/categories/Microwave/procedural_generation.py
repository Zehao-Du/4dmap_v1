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


def get_microwave_type():
    total_type = ["regular"]
    weights = [1]
    microwave_type = random.choices(total_type, weights=weights, k=1)[0]
    return microwave_type


def concept_template_existence(microwave_type):
    concept_template_variation = {
        "body": {"template": ["Cuboidal_Body"], "necessary": True},
        "door": {"template": ["Cuboidal_Door", "Sunken_Door"], "necessary": True},
        "controller": {"template": ["Controller_With_Button"], "necessary": True},
        "handle": {
            "template": [
                "Cuboidal_Handle",
                "Trifold_Handle",
                "Trifold_Curve_Handle",
                "Curve_Handle",
            ],
            "necessary": True,
        },
        "tray": {"template": ["Cylindrical_Tray"], "necessary": False},
    }

    templates = []

    has_cover = False
    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        elif random.random() < 0.5:
            templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, microwave_type):
    new_concepts = []

    size_x: float = 0
    size_y: float = 0
    size_z: float = 0

    back_thick: float = 0
    side_thick: float = 0

    door_x: float = 0
    door_y: float = 0
    door_z: float = 0

    door_px: float = 0
    door_py: float = 0
    door_pz: float = 0
    door_rx: float = 0
    door_ry: float = 0
    door_rz: float = 0

    drd: bool = True

    if_controller: bool = random.choice([True, False])

    if_x_rotation: bool = random.choice([True, False])

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Cuboidal_Body":
            parameter["size"][0] = randRange(1, 1.74, 3.06)[0]

            parameter["size"][1] = parameter["size"][0] * randRange(1, 0.6, 0.7)[0]

            parameter["size"][2] = parameter["size"][0] * randRange(1, 0.514, 0.892)[0]

            parameter["thickness"][0] = randRange(1, 0.08, 0.12)[0]
            parameter["thickness"][2] = randRange(1, 0.06, 0.09)[0]
            parameter["thickness"][1] = parameter["thickness"][0]
            parameter["thickness"][3] = parameter["thickness"][2]
            parameter["thickness"][4] = parameter["thickness"][0]

            size_x = parameter["size"][0] + parameter["thickness"][2]
            size_y = parameter["size"][1]
            size_z = parameter["size"][2]
            back_thick = parameter["thickness"][4]
            side_thick = parameter["thickness"][2]
            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cuboidal_Door":
            if if_x_rotation:
                if if_controller:
                    parameter["size"] = np.array(
                        [
                            size_x - side_thick,
                            size_y * randRange(1, 0.7, 0.8)[0],
                            randRange(1, 0.06, 0.08)[0],
                        ]
                    )
                else:
                    parameter["size"] = np.array(
                        [size_x - side_thick, size_y, randRange(1, 0.06, 0.08)[0]]
                    )

                if random.choice([0, 1]):
                    parameter["rotation"] = np.array([randRange(1, 50, 90)[0], 0, 0])
                    parameter["position"] = np.array(
                        [
                            0,
                            -0.5
                            * (
                                size_y
                                - parameter["size"][1]
                                * np.cos(parameter["rotation"][0] * np.pi / 180)
                            ),
                            0.5
                            * (
                                size_z
                                + parameter["size"][1]
                                * np.sin(parameter["rotation"][0] * np.pi / 180)
                            ),
                        ]
                    )
                    drd = False
                else:
                    parameter["rotation"] = np.array([-randRange(1, 50, 90)[0], 0, 0])
                    parameter["position"] = np.array(
                        [
                            0,
                            0.5
                            * (
                                size_y
                                - parameter["size"][1]
                                * np.cos(parameter["rotation"][0] * np.pi / 180)
                            ),
                            0.5
                            * (
                                size_z
                                - parameter["size"][1]
                                * np.sin(parameter["rotation"][0] * np.pi / 180)
                            ),
                        ]
                    )
            else:
                if if_controller:
                    parameter["size"] = np.array(
                        [
                            (size_x - side_thick) * randRange(1, 0.7, 0.8)[0],
                            size_y,
                            randRange(1, 0.06, 0.08)[0],
                        ]
                    )
                else:
                    parameter["size"] = np.array(
                        [size_x - side_thick, size_y, randRange(1, 0.06, 0.08)[0]]
                    )
                if random.choice([0, 1]):
                    parameter["rotation"] = np.array([0, randRange(1, 30, 90)[0], 0])
                    parameter["position"] = np.array(
                        [
                            0.5
                            * (
                                size_x
                                - parameter["size"][0]
                                * np.cos(parameter["rotation"][1] * np.pi / 180)
                                - parameter["size"][2]
                                * np.sin(parameter["rotation"][1] * np.pi / 180)
                            ),
                            0,
                            0.5
                            * (
                                size_z
                                + parameter["size"][0]
                                * np.sin(parameter["rotation"][1] * np.pi / 180)
                            ),
                        ]
                    )
                else:
                    parameter["rotation"] = np.array([0, -randRange(1, 30, 90)[0], 0])
                    parameter["position"] = np.array(
                        [
                            -0.5
                            * (
                                size_x
                                - parameter["size"][0]
                                * np.cos(parameter["rotation"][1] * np.pi / 180)
                                + parameter["size"][2]
                                * np.sin(parameter["rotation"][1] * np.pi / 180)
                            ),
                            0,
                            0.5
                            * (
                                size_z
                                - parameter["size"][0]
                                * np.sin(parameter["rotation"][1] * np.pi / 180)
                            ),
                        ]
                    )
                    drd = False

            door_x = parameter["size"][0]
            door_y = parameter["size"][1]
            door_z = parameter["size"][2]
            door_px = parameter["position"][0]
            door_py = parameter["position"][1]
            door_pz = parameter["position"][2]
            door_rx = parameter["rotation"][0]
            door_ry = parameter["rotation"][1]
            door_rz = parameter["rotation"][2]

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Sunken_Door":
            if if_x_rotation:
                if if_controller:
                    parameter["size"] = np.array(
                        [
                            size_x - side_thick,
                            size_y * randRange(1, 0.7, 0.8)[0],
                            randRange(1, 0.06, 0.08)[0],
                        ]
                    )
                else:
                    parameter["size"] = np.array(
                        [size_x - side_thick, size_y, randRange(1, 0.06, 0.08)[0]]
                    )

                if random.choice([0, 1]):
                    parameter["rotation"] = np.array([randRange(1, 50, 90)[0], 0, 0])
                    parameter["position"] = np.array(
                        [
                            0,
                            -0.5
                            * (
                                size_y
                                - parameter["size"][1]
                                * np.cos(parameter["rotation"][0] * np.pi / 180)
                            ),
                            0.5
                            * (
                                size_z
                                + parameter["size"][1]
                                * np.sin(parameter["rotation"][0] * np.pi / 180)
                            ),
                        ]
                    )
                    drd = False
                else:
                    parameter["rotation"] = np.array([-randRange(1, 50, 90)[0], 0, 0])
                    parameter["position"] = np.array(
                        [
                            0,
                            0.5
                            * (
                                size_y
                                - parameter["size"][1]
                                * np.cos(parameter["rotation"][0] * np.pi / 180)
                            ),
                            0.5
                            * (
                                size_z
                                - parameter["size"][1]
                                * np.sin(parameter["rotation"][0] * np.pi / 180)
                            ),
                        ]
                    )
            else:
                if if_controller:
                    parameter["size"] = np.array(
                        [
                            (size_x - side_thick) * randRange(1, 0.7, 0.8)[0],
                            size_y,
                            randRange(1, 0.06, 0.08)[0],
                        ]
                    )
                else:
                    parameter["size"] = np.array(
                        [size_x - side_thick, size_y, randRange(1, 0.06, 0.08)[0]]
                    )

                if random.choice([0, 1]):
                    parameter["rotation"] = np.array([0, randRange(1, 30, 90)[0], 0])
                    parameter["position"] = np.array(
                        [
                            0.5
                            * (
                                size_x
                                - parameter["size"][0]
                                * np.cos(parameter["rotation"][1] * np.pi / 180)
                                - parameter["size"][2]
                                * np.sin(parameter["rotation"][1] * np.pi / 180)
                            ),
                            0,
                            0.5
                            * (
                                size_z
                                + parameter["size"][0]
                                * np.sin(parameter["rotation"][1] * np.pi / 180)
                            ),
                        ]
                    )
                else:
                    parameter["rotation"] = np.array([0, -randRange(1, 30, 90)[0], 0])
                    parameter["position"] = np.array(
                        [
                            -0.5
                            * (
                                size_x
                                - parameter["size"][0]
                                * np.cos(parameter["rotation"][1] * np.pi / 180)
                                + parameter["size"][2]
                                * np.sin(parameter["rotation"][1] * np.pi / 180)
                            ),
                            0,
                            0.5
                            * (
                                size_z
                                - parameter["size"][0]
                                * np.sin(parameter["rotation"][1] * np.pi / 180)
                            ),
                        ]
                    )
                    drd = False

            parameter["sunken_size"][0] = (
                parameter["size"][0] * randRange(1, 0.8, 0.9)[0]
            )
            parameter["sunken_size"][1] = (
                parameter["size"][1] * parameter["sunken_size"][0] / size_x
            )
            parameter["sunken_size"][2] = (
                parameter["size"][2] * randRange(1, 0.4, 0.6)[0]
            )
            parameter["sunken_offset"] = np.array([0, 0])

            door_x = parameter["size"][0]
            door_y = parameter["size"][1]
            door_z = parameter["size"][2]

            door_px = parameter["position"][0]
            door_py = parameter["position"][1]
            door_pz = parameter["position"][2]
            door_rx = parameter["rotation"][0]
            door_ry = parameter["rotation"][1]
            door_rz = parameter["rotation"][2]

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Controller_With_Button":
            parameter["button_1_size"] = np.array([0, 0, 0])
            parameter["button_2_size"] = np.array([0, 0, 0])
            parameter["button_3_size"] = np.array([0, 0, 0])
            parameter["button_4_size"] = np.array([0, 0, 0])

            parameter["bottom_size"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            if if_controller:
                if if_x_rotation:
                    parameter["bottom_size"] = np.array(
                        [size_x - side_thick, size_y - door_y, 1.5 * door_z]
                    )
                    if drd:
                        parameter["position"] = np.array(
                            [0, -0.5 * door_y, 0.5 * size_z]
                        )
                    else:
                        parameter["position"] = np.array(
                            [0, 0.5 * door_y, 0.5 * size_z]
                        )

                else:
                    parameter["bottom_size"] = np.array(
                        [size_x - side_thick - door_x, size_y, door_z]
                    )
                    if drd:
                        parameter["position"] = np.array(
                            [-0.5 * door_x, 0, 0.5 * size_z]
                        )
                    else:
                        parameter["position"] = np.array(
                            [0.5 * door_x, 0, 0.5 * size_z]
                        )

                parameter["num_buttons"][0] = random.choice(range(1, 5))

                if parameter["num_buttons"][0] == 1:
                    parameter["button_1_size"] = np.array([0.23, 0.23, 0.12])
                    parameter["button_1_offset"] = np.array([0, 0, 0])

                elif parameter["num_buttons"][0] == 2:
                    parameter["button_1_size"] = np.array([0.2, 0.2, 0.12])
                    parameter["button_2_size"] = parameter["button_1_size"]
                    if if_x_rotation:
                        parameter["button_1_offset"] = np.array([-door_x / 4, 0, 0])
                        parameter["button_2_offset"] = np.array([door_x / 4, 0, 0])
                    else:
                        parameter["button_1_offset"] = np.array([0, -0.36, 0])
                        parameter["button_2_offset"] = np.array([0, 0.36, 0])
                elif parameter["num_buttons"][0] == 3:
                    parameter["button_1_size"] = np.array(
                        [randRange(1, 0.1, 0.14)[0], randRange(1, 0.08, 0.1)[0], 0.06]
                    )
                    parameter["button_2_size"] = np.array([0.2, 0.2, 0.12])
                    parameter["button_3_size"] = parameter["button_2_size"]
                    if if_x_rotation:
                        parameter["button_1_offset"] = np.array([0, 0, 0])
                        parameter["button_2_offset"] = np.array([-0.72, 0, 0])
                        parameter["button_3_offset"] = np.array([0.72, 0, 0])
                    else:
                        parameter["button_1_offset"] = np.array([0, -door_y / 4, 0])
                        parameter["button_2_offset"] = np.array([0, 0, 0])
                        parameter["button_3_offset"] = np.array([0, door_y / 4, 0])

                elif parameter["num_buttons"][0] == 4:
                    parameter["button_1_size"] = np.array(
                        [
                            randRange(1, 0.16, 0.2)[0],
                            randRange(1, 0.12, 0.18)[0],
                            randRange(1, 0.11, 0.13)[0],
                        ]
                    )
                    parameter["button_2_size"] = parameter["button_1_size"]

                    if if_x_rotation:
                        parameter["button_3_size"] = parameter["button_1_size"]
                        parameter["button_4_size"] = parameter["button_1_size"]

                        parameter["button_1_offset"] = np.array([-0.72, 0, 0])
                        parameter["button_2_offset"] = np.array([-0.24, 0, 0])
                        parameter["button_3_offset"] = np.array([0.24, 0, 0])
                        parameter["button_4_offset"] = np.array([0.72, 0, 0])
                    else:
                        parameter["button_3_size"] = np.array([0.18, 0.18, 0.08])
                        parameter["button_4_size"] = parameter["button_3_size"]

                        parameter["button_1_offset"] = np.array([0, -0.24, 0])
                        parameter["button_2_offset"] = np.array([0, -0.12, 0])
                        parameter["button_3_offset"] = np.array([0, 0.24, 0])
                        parameter["button_4_offset"] = np.array([0, 0.36, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Cuboidal_Handle":
            parameter["size"][1] = randRange(1, 0.1, 0.14)[0]
            parameter["size"][2] = randRange(1, 0.1, 0.14)[0]

            parameter["rotation"][0] = door_rx
            parameter["rotation"][1] = door_ry

            if if_x_rotation:
                parameter["size"][0] = door_x * randRange(1, 0.8, 1)[0]
                parameter["position"][0] = door_px
                parameter["rotation"][2] = door_rz
                if drd:
                    parameter["position"][1] = door_py - (0.5 * door_y - 0.24) * np.cos(
                        parameter["rotation"][0] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        - (0.5 * door_y - 0.24)
                        * np.sin(parameter["rotation"][0] * np.pi / 180)
                        + door_z
                    )
                else:
                    parameter["position"][1] = door_py + (0.5 * door_y - 0.24) * np.cos(
                        parameter["rotation"][0] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        + (0.5 * door_y - 0.24)
                        * np.sin(parameter["rotation"][0] * np.pi / 180)
                        + door_z
                    )

            else:
                parameter["size"][0] = door_y * randRange(1, 0.8, 1)[0]
                parameter["position"][1] = door_py
                parameter["rotation"][2] = door_rz + 90
                if drd:
                    parameter["position"][0] = door_px - (0.5 * door_x - 0.24) * np.cos(
                        parameter["rotation"][1] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        + (0.5 * door_x - 0.24)
                        * np.sin(parameter["rotation"][1] * np.pi / 180)
                        + door_z
                    )
                else:
                    parameter["position"][0] = door_px + (0.5 * door_x - 0.24) * np.cos(
                        parameter["rotation"][1] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        - (0.5 * door_x - 0.24)
                        * np.sin(parameter["rotation"][1] * np.pi / 180)
                        + door_z
                    )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Trifold_Handle":
            parameter["mounting_size"][0] = randRange(1, 0.06, 0.09)[0]
            parameter["mounting_size"][1] = randRange(1, 0.06, 0.09)[0]
            parameter["mounting_size"][2] = randRange(1, 0.15, 0.17)[0]
            parameter["grip_size"][0] = (
                parameter["mounting_size"][0] * randRange(1, 1, 1.1)[0]
            )
            parameter["grip_size"][2] = randRange(1, 0.06, 0.09)[0]
            parameter["rotation"][0] = door_rx
            parameter["rotation"][1] = door_ry

            if if_x_rotation:
                parameter["position"][0] = door_px
                parameter["rotation"][2] = door_rz + 90
                parameter["grip_size"][1] = door_x * randRange(1, 0.8, 1)[0]
                if drd:
                    parameter["position"][1] = door_py - (0.5 * door_y - 0.24) * np.cos(
                        parameter["rotation"][0] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        - (0.5 * door_y - 0.24)
                        * np.sin(parameter["rotation"][0] * np.pi / 180)
                        + door_z
                    )
                else:
                    parameter["position"][1] = door_py + (0.5 * door_y - 0.24) * np.cos(
                        parameter["rotation"][0] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        + (0.5 * door_y - 0.24)
                        * np.sin(parameter["rotation"][0] * np.pi / 180)
                        + door_z
                    )

            else:
                parameter["position"][1] = door_py
                parameter["rotation"][2] = door_rz
                parameter["grip_size"][1] = door_y * randRange(1, 0.8, 1)[0]
                parameter["mounting_seperation"] = np.array(
                    [parameter["grip_size"][1] * randRange(1, 0.7, 0.8)[0]]
                )
                if drd:
                    parameter["position"][0] = door_px - (0.5 * door_x - 0.24) * np.cos(
                        parameter["rotation"][1] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        + (0.5 * door_x - 0.24)
                        * np.sin(parameter["rotation"][1] * np.pi / 180)
                        + door_z
                    )
                else:
                    parameter["position"][0] = door_px + (0.5 * door_x - 0.24) * np.cos(
                        parameter["rotation"][1] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        - (0.5 * door_x - 0.24)
                        * np.sin(parameter["rotation"][1] * np.pi / 180)
                        + door_z
                    )
            parameter["mounting_seperation"][0] = (
                parameter["grip_size"][1] * randRange(1, 0.7, 0.8)[0]
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Trifold_Curve_Handle":
            parameter["mounting_size"][0] = randRange(1, 0.06, 0.09)[0]
            parameter["mounting_size"][1] = randRange(1, 0.06, 0.09)[0]
            parameter["mounting_size"][2] = randRange(1, 0.15, 0.17)[0]

            parameter["curve_exist_angle"][0] = randRange(1, 70, 80)[0]
            parameter["rotation"][0] = door_rx
            parameter["rotation"][1] = door_ry

            if if_x_rotation:
                parameter["mounting_seperation"] = np.array(
                    [door_x * randRange(1, 0.6, 0.7)[0]]
                )
                parameter["curve_size"][0] = (
                    door_x
                    * randRange(1, 0.8, 1)[0]
                    / (
                        2
                        * np.sin(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )
                )
                parameter["curve_size"][1] = parameter["curve_size"][0] - 0.06
                parameter["curve_size"][2] = randRange(1, 0.06, 0.1)[0]
                parameter["position"][0] = door_px
                parameter["rotation"][2] = door_rz + 90
                if drd:
                    parameter["position"][1] = door_py - (0.5 * door_y - 0.24) * np.cos(
                        parameter["rotation"][0] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        - (0.5 * door_y - 0.24)
                        * np.sin(parameter["rotation"][0] * np.pi / 180)
                        + door_z
                        + 0.06
                        * np.cos(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )
                else:
                    parameter["position"][1] = door_py + (0.5 * door_y - 0.24) * np.cos(
                        parameter["rotation"][0] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        + (0.5 * door_y - 0.24)
                        * np.sin(parameter["rotation"][0] * np.pi / 180)
                        + door_z
                        + 0.06
                        * np.cos(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )

            else:
                parameter["mounting_seperation"] = np.array(
                    [door_y * randRange(1, 0.6, 0.7)[0]]
                )
                parameter["curve_size"][0] = (
                    door_y
                    * randRange(1, 0.8, 1)[0]
                    / (
                        2
                        * np.sin(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )
                )
                parameter["curve_size"][1] = parameter["curve_size"][0] - 0.06
                parameter["curve_size"][2] = randRange(1, 0.06, 0.1)[0]
                parameter["position"][1] = door_py
                parameter["rotation"][2] = door_rz
                if drd:
                    parameter["position"][0] = door_px - (0.5 * door_x - 0.24) * np.cos(
                        parameter["rotation"][1] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        + (0.5 * door_x - 0.24)
                        * np.sin(parameter["rotation"][1] * np.pi / 180)
                        + door_z
                        + 0.06
                        * np.cos(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )
                else:
                    parameter["position"][0] = door_px + (0.5 * door_x - 0.24) * np.cos(
                        parameter["rotation"][1] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        - (0.5 * door_x - 0.24)
                        * np.sin(parameter["rotation"][1] * np.pi / 180)
                        + door_z
                        + 0.06
                        * np.cos(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Curve_Handle":
            parameter["curve_exist_angle"][0] = randRange(1, 70, 80)[0]
            parameter["rotation"][0] = door_rx
            parameter["rotation"][1] = door_ry

            if if_x_rotation:
                parameter["curve_size"][0] = (
                    door_x
                    * randRange(1, 0.8, 1)[0]
                    / (
                        2
                        * np.sin(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )
                )
                parameter["curve_size"][1] = parameter["curve_size"][0] - 0.06
                parameter["curve_size"][2] = randRange(1, 0.06, 0.1)[0]
                parameter["position"][0] = door_px
                parameter["rotation"][2] = door_rz + 90
                if drd:
                    parameter["position"][1] = door_py - (0.5 * door_y - 0.24) * np.cos(
                        parameter["rotation"][0] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        - (0.5 * door_y - 0.24)
                        * np.sin(parameter["rotation"][0] * np.pi / 180)
                        + door_z
                        + 0.06
                        * np.cos(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )
                else:
                    parameter["position"][1] = door_py + (0.5 * door_y - 0.24) * np.cos(
                        parameter["rotation"][0] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        + (0.5 * door_y - 0.24)
                        * np.sin(parameter["rotation"][0] * np.pi / 180)
                        + door_z
                        + 0.06
                        * np.cos(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )

            else:
                parameter["curve_size"][0] = (
                    door_y
                    * randRange(1, 0.8, 1)[0]
                    / (
                        2
                        * np.sin(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )
                )
                parameter["curve_size"][1] = parameter["curve_size"][0] - 0.06
                parameter["curve_size"][2] = randRange(1, 0.06, 0.1)[0]
                parameter["position"][1] = door_py
                parameter["rotation"][2] = door_rz
                if drd:
                    parameter["position"][0] = door_px - (0.5 * door_x - 0.24) * np.cos(
                        parameter["rotation"][1] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        + (0.5 * door_x - 0.24)
                        * np.sin(parameter["rotation"][1] * np.pi / 180)
                        + door_z
                        + 0.06
                        * np.cos(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )
                else:
                    parameter["position"][0] = door_px + (0.5 * door_x - 0.24) * np.cos(
                        parameter["rotation"][1] * np.pi / 180
                    )
                    parameter["position"][2] = (
                        door_pz
                        - (0.5 * door_x - 0.24)
                        * np.sin(parameter["rotation"][1] * np.pi / 180)
                        + door_z
                        + 0.06
                        * np.cos(0.5 * parameter["curve_exist_angle"][0] * np.pi / 180)
                    )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

    if template == "Cylindrical_Tray":
        parameter["size"] = np.array(
            [size_z * randRange(1, 0.35, 0.4)[0], randRange(1, 0.075, 0.08)[0]]
        )
        parameter["position"] = np.array(
            [0, -0.5 * size_y + parameter["size"][1], 0.5 * back_thick]
        )
        parameter["rotation"] = np.array([0, 0, 0])
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
        microwave_type = get_microwave_type()
        existing_concept_templates = concept_template_existence(microwave_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, microwave_type)

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
            microwave_type = get_microwave_type()
            existing_concept_templates = concept_template_existence(microwave_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, microwave_type)
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
