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


def get_oven_type():
    oven_type = [
        "no_cover_or_coaster",
        "has_cover",
        "has_coaster",
        "has_cover_and_coaster",
    ]
    weights = [1, 1, 1, 1]
    oven_type = random.choices(oven_type, weights=weights, k=1)[0]
    return oven_type


def concept_template_existence(oven_type):
    concept_template_variation = {
        "body": {"template": ["Cuboidal_Body", "Double_Layer_Body"], "necessary": True},
        "door": {"template": ["Cuboidal_Door", "Sunken_Door"], "necessary": True},
        "handle": {
            "template": ["Trifold_Handle", "Trifold_Curve_Handle", "Curve_Handle"],
            "necessary": True,
        },
        "tray": {"template": ["Flat_Tray", "Drawer_Like_Tray"], "necessary": False},
        "baffle": {"template": ["Cuboidal_Baffle"], "necessary": False},
        "controller": {"template": ["Controller_With_Button"], "necessary": False},
        "top": {"template": ["Top_With_Burner"], "necessary": False},
        "leg": {"template": ["Multilevel_Leg"], "necessary": False},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        elif random.random() < 0.5:
            templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, oven_type):
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Cuboidal_Body":
            size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
            parameter["size"] *= size_mul

            if "Controller_With_Button" in templates:
                parameter["thickness"][0] = (
                    randRange(1, 0.15, 0.2)[0] * parameter["size"][1]
                )
            else:
                parameter["thickness"][0] = (
                    randRange(1, 0.02, 0.05)[0] * parameter["size"][1]
                )
            parameter["thickness"][1] = (
                randRange(1, 0.02, 0.05)[0] * parameter["size"][1]
            )
            parameter["thickness"][2] = (
                randRange(1, 0.02, 0.05)[0] * parameter["size"][0]
            )
            parameter["thickness"][3] = parameter["thickness"][2]
            parameter["thickness"][4] = (
                randRange(1, 0.05, 0.1)[0] * parameter["size"][2]
            )

            parameter["position"][:] = 0
            parameter["rotation"][:] = 0

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
            body_parameter = concept["parameters"]

        elif template == "Double_Layer_Body":
            size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
            parameter["size"] *= size_mul

            if "Controller_With_Button" in templates:
                parameter["thickness"][0] = (
                    randRange(1, 0.15, 0.2)[0] * parameter["size"][1]
                )
            else:
                parameter["thickness"][0] = (
                    randRange(1, 0.02, 0.05)[0] * parameter["size"][1]
                )
            parameter["thickness"][1] = (
                randRange(1, 0.02, 0.05)[0] * parameter["size"][1]
            )
            parameter["thickness"][2] = (
                randRange(1, 0.02, 0.05)[0] * parameter["size"][0]
            )
            parameter["thickness"][3] = parameter["thickness"][2]
            parameter["thickness"][4] = (
                randRange(1, 0.05, 0.1)[0] * parameter["size"][2]
            )

            parameter["clapboard_size"][0] = (
                randRange(1, 0.1, 0.15)[0] * parameter["size"][1]
            )
            parameter["clapboard_size"][1] = (
                parameter["size"][2] - parameter["thickness"][4]
            )

            parameter["clapboard_offset"][0] = randRange(
                1,
                -parameter["size"][1] / 2
                + parameter["thickness"][1]
                + parameter["clapboard_size"][0] / 2,
                parameter["size"][1] / 2
                - parameter["thickness"][0]
                - parameter["clapboard_size"][0] / 2,
            )[0]

            parameter["position"][:] = 0
            parameter["rotation"][:] = 0

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
            body_parameter = concept["parameters"]

        elif template == "Cuboidal_Door":
            parameter["size"][0], parameter["size"][1] = (
                body_parameter["size"][0],
                body_parameter["size"][1],
            )
            if "Controller_With_Button" in templates:
                parameter["size"][1] -= body_parameter["thickness"][0]
            parameter["size"][2] = randRange(
                1, 0.5 * body_parameter["thickness"][4], 0.1 * body_parameter["size"][2]
            )[0]

            parameter["rotation"][0] = randRange(1, 0, 90)[0]
            parameter["rotation"][[1, 2]] = 0

            parameter["position"][0] = 0
            parameter["position"][1] = -body_parameter["size"][1] / 2 + parameter[
                "size"
            ][1] / 2 * np.cos(parameter["rotation"][0] / 180 * np.pi)
            parameter["position"][2] = body_parameter["size"][2] / 2 + parameter[
                "size"
            ][1] / 2 * np.sin(parameter["rotation"][0] / 180 * np.pi)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
            door_parameter = concept["parameters"]

        elif template == "Sunken_Door":
            parameter["size"][0], parameter["size"][1] = (
                body_parameter["size"][0],
                body_parameter["size"][1],
            )
            if "Controller_With_Button" in templates:
                parameter["size"][1] -= body_parameter["thickness"][0]
            parameter["size"][2] = randRange(
                1, 0.5 * body_parameter["thickness"][2], 0.1 * body_parameter["size"][0]
            )[0]

            parameter["sunken_size"][[0, 1]] = randRange(
                2, 0.8 * parameter["size"][[0, 1]], 0.95 * parameter["size"][[0, 1]]
            )
            parameter["sunken_size"][2] = randRange(
                1, 0.2 * parameter["size"][2], 0.95 * parameter["size"][2]
            )[0]

            parameter["sunken_offset"][1] = randRange(
                1, -parameter["size"][1] / 2 + parameter["sunken_size"][1] / 2, 0
            )[0]
            parameter["sunken_offset"][0] = 0

            parameter["rotation"][0] = randRange(1, 0, 90)[0]
            parameter["rotation"][[1, 2]] = 0

            parameter["position"][0] = 0
            parameter["position"][1] = -body_parameter["size"][1] / 2 + parameter[
                "size"
            ][1] / 2 * np.cos(parameter["rotation"][0] / 180 * np.pi)
            parameter["position"][2] = body_parameter["size"][2] / 2 + parameter[
                "size"
            ][1] / 2 * np.sin(parameter["rotation"][0] / 180 * np.pi)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
            door_parameter = concept["parameters"]

        elif template == "Trifold_Handle":
            if concepts[1]["template"] == "Sunken_Door":
                parameter["mounting_size"][[0, 2]] *= randRange(2, 0.7, 1.3)
                sunken_face_margin = door_parameter["size"][1] / 2 - (
                    door_parameter["sunken_size"][1] / 2
                    + door_parameter["sunken_offset"][1]
                )
                parameter["mounting_size"][1] = randRange(
                    1,
                    min(0.7 * parameter["mounting_size"][1], sunken_face_margin * 0.7),
                    min(1.3 * parameter["mounting_size"][1], sunken_face_margin),
                )[0]
                vertical_move_up = randRange(
                    1,
                    door_parameter["sunken_size"][1] / 2
                    + door_parameter["sunken_offset"][1]
                    + parameter["mounting_size"][1] / 2,
                    door_parameter["size"][1] / 2 - parameter["mounting_size"][1] / 2,
                )[0]

            else:
                parameter["mounting_size"] *= randRange(3, 0.7, 1.3)
                vertical_move_up = randRange(
                    1,
                    0.7 * door_parameter["size"][1] / 2,
                    door_parameter["size"][1] / 2 - parameter["mounting_size"][1] / 2,
                )[0]

            parameter["mounting_seperation"][0] = randRange(
                1,
                0.5 * door_parameter["size"][0],
                door_parameter["size"][0] - parameter["mounting_size"][0],
            )[0]
            parameter["grip_size"][0] = parameter["mounting_size"][0]
            parameter["grip_size"][1] = (
                parameter["mounting_size"][1] + parameter["mounting_seperation"][0]
            )
            parameter["grip_size"][2] *= randRange(1, 0.7, 1.3)[0]

            parameter["rotation"] = np.array(door_parameter["rotation"])
            parameter["rotation"][2] = 90
            parameter["position"] = np.array(door_parameter["position"])
            cos_door_rot = np.cos(door_parameter["rotation"][0] / 180 * np.pi)
            sin_door_rot = np.sin(door_parameter["rotation"][0] / 180 * np.pi)
            parameter["position"][1] += (
                vertical_move_up * cos_door_rot
                - door_parameter["size"][2] * sin_door_rot
            )
            parameter["position"][2] += (
                door_parameter["size"][2] * cos_door_rot
                + vertical_move_up * sin_door_rot
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Trifold_Curve_Handle":
            if concepts[1]["template"] == "Sunken_Door":
                parameter["mounting_size"][[0, 2]] *= randRange(2, 0.7, 1.3)
                sunken_face_margin = door_parameter["size"][1] / 2 - (
                    door_parameter["sunken_size"][1] / 2
                    + door_parameter["sunken_offset"][1]
                )
                parameter["mounting_size"][1] = randRange(
                    1,
                    min(0.7 * parameter["mounting_size"][1], sunken_face_margin * 0.7),
                    min(1.3 * parameter["mounting_size"][1], sunken_face_margin),
                )[0]
                vertical_move_up = randRange(
                    1,
                    door_parameter["sunken_size"][1] / 2
                    + door_parameter["sunken_offset"][1]
                    + parameter["mounting_size"][1] / 2,
                    door_parameter["size"][1] / 2 - parameter["mounting_size"][1] / 2,
                )[0]

            else:
                parameter["mounting_size"] *= randRange(3, 0.7, 1.3)
                vertical_move_up = randRange(
                    1,
                    0.7 * door_parameter["size"][1] / 2,
                    door_parameter["size"][1] / 2 - parameter["mounting_size"][1] / 2,
                )[0]

            parameter["mounting_seperation"][0] = randRange(
                1,
                0.5 * door_parameter["size"][0],
                door_parameter["size"][0] - parameter["mounting_size"][0],
            )[0]
            parameter["curve_size"][2] = parameter["mounting_size"][0]
            parameter["curve_size"][0] = randRange(
                1,
                parameter["mounting_seperation"][0],
                2 * parameter["mounting_seperation"][0],
            )[0]
            parameter["curve_size"][1] = (
                parameter["curve_size"][0] - parameter["mounting_size"][0]
            )

            parameter["curve_exist_angle"][0] = (
                2
                * np.arcsin(
                    parameter["mounting_seperation"] / 2 / parameter["curve_size"][1]
                )
                / np.pi
                * 180
            )

            parameter["rotation"] = np.array(door_parameter["rotation"])
            parameter["rotation"][2] = 90
            parameter["position"] = np.array(door_parameter["position"])
            cos_door_rot = np.cos(door_parameter["rotation"][0] / 180 * np.pi)
            sin_door_rot = np.sin(door_parameter["rotation"][0] / 180 * np.pi)
            parameter["position"][1] += (
                vertical_move_up * cos_door_rot
                - door_parameter["size"][2] * sin_door_rot
            )
            parameter["position"][2] += (
                door_parameter["size"][2] * cos_door_rot
                + vertical_move_up * sin_door_rot
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Curve_Handle":
            if concepts[1]["template"] == "Sunken_Door":
                sunken_face_margin = door_parameter["size"][1] / 2 - (
                    door_parameter["sunken_size"][1] / 2
                    + door_parameter["sunken_offset"][1]
                )
                parameter["curve_size"][2] = randRange(
                    1,
                    min(0.7 * parameter["curve_size"][2], sunken_face_margin * 0.7),
                    min(1.3 * parameter["curve_size"][2], sunken_face_margin),
                )[0]
                vertical_move_up = randRange(
                    1,
                    door_parameter["sunken_size"][1] / 2
                    + door_parameter["sunken_offset"][1]
                    + parameter["curve_size"][2] / 2,
                    door_parameter["size"][1] / 2 - parameter["curve_size"][2] / 2,
                )[0]

            else:
                parameter["curve_size"][2] *= randRange(1, 0.7, 1.3)[0]
                vertical_move_up = randRange(
                    1,
                    0.7 * door_parameter["size"][1] / 2,
                    door_parameter["size"][1] / 2 - parameter["curve_size"][2] / 2,
                )[0]

            mounting_seperation = randRange(
                1, 0.5 * door_parameter["size"][0], 0.9 * door_parameter["size"][0]
            )[0]
            curve_size_diff = parameter["curve_size"][0] - parameter["curve_size"][1]
            mul_curve_size_diff = randRange(
                1, 0.5 * curve_size_diff, 1.2 * curve_size_diff
            )[0]
            parameter["curve_size"][0] = randRange(
                1, mounting_seperation, mounting_seperation
            )[0]
            parameter["curve_size"][1] = (
                parameter["curve_size"][0] - curve_size_diff * mul_curve_size_diff
            )

            parameter["curve_exist_angle"][0] = (
                2
                * np.arcsin(mounting_seperation / 2 / parameter["curve_size"][1])
                / np.pi
                * 180
            )

            parameter["rotation"] = np.array(door_parameter["rotation"])
            parameter["rotation"][2] = 90
            parameter["position"] = np.array(door_parameter["position"])
            cos_door_rot = np.cos(door_parameter["rotation"][0] / 180 * np.pi)
            sin_door_rot = np.sin(door_parameter["rotation"][0] / 180 * np.pi)
            parameter["position"][1] += (
                vertical_move_up * cos_door_rot
                - door_parameter["size"][2] * sin_door_rot
            )
            parameter["position"][2] += (
                door_parameter["size"][2] * cos_door_rot
                + vertical_move_up * sin_door_rot
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Flat_Tray":
            parameter["size"][0] = randRange(1, 0.95, 1.0)[0] * (
                body_parameter["size"][0]
                - body_parameter["thickness"][2]
                - body_parameter["thickness"][3]
            )
            parameter["size"][1] *= randRange(1, 0.7, 3.0)[0]
            parameter["size"][2] = randRange(1, 0.85, 0.95)[0] * (
                body_parameter["size"][2] - body_parameter["thickness"][4]
            )

            parameter["rotation"] = np.array(body_parameter["rotation"])
            parameter["position"][0], parameter["position"][2] = (
                body_parameter["position"][0],
                body_parameter["position"][2],
            )
            if concepts[0]["template"] == "Double_Layer_Body":
                parameter["position"][1] = randRange(
                    1,
                    body_parameter["clapboard_offset"][0]
                    + body_parameter["clapboard_size"][0] / 2
                    + parameter["size"][1] / 2,
                    body_parameter["size"][1] / 2
                    - body_parameter["thickness"][0]
                    - parameter["size"][1] / 2,
                )[0]
            else:
                parameter["position"][1] = randRange(
                    1,
                    -body_parameter["size"][1] / 2
                    + body_parameter["thickness"][1]
                    + parameter["size"][1] / 2,
                    body_parameter["size"][1] / 2
                    - body_parameter["thickness"][0]
                    - parameter["size"][1] / 2,
                )[0]

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Drawer_Like_Tray":
            parameter["outer_size"][0] = randRange(1, 0.95, 1.0)[0] * (
                body_parameter["size"][0]
                - body_parameter["thickness"][2]
                - body_parameter["thickness"][3]
            )
            parameter["outer_size"][1] *= randRange(1, 0.7, 1.3)[0]
            parameter["outer_size"][2] = randRange(1, 0.85, 0.95)[0] * (
                body_parameter["size"][2] - body_parameter["thickness"][4]
            )
            new_size_diff = randRange(
                3, 0.7 * parameter["outer_size"], 0.95 * parameter["outer_size"]
            )
            parameter["inner_size"] = new_size_diff

            parameter["rotation"] = np.array(body_parameter["rotation"])
            parameter["position"][0], parameter["position"][2] = (
                body_parameter["position"][0],
                body_parameter["position"][2],
            )
            if concepts[0]["template"] == "Double_Layer_Body":
                parameter["position"][1] = randRange(
                    1,
                    body_parameter["clapboard_offset"][0]
                    + body_parameter["clapboard_size"][0] / 2
                    + parameter["outer_size"][1] / 2,
                    body_parameter["size"][1] / 2
                    - body_parameter["thickness"][0]
                    - parameter["outer_size"][1] / 2,
                )[0]
            else:
                parameter["position"][1] = randRange(
                    1,
                    -body_parameter["size"][1] / 2
                    + body_parameter["thickness"][1]
                    + parameter["outer_size"][1] / 2,
                    body_parameter["size"][1] / 2
                    - body_parameter["thickness"][0]
                    - parameter["outer_size"][1] / 2,
                )[0]

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Controller_With_Button":
            parameter["num_buttons"][0] = np.random.choice(range(4, 11))

            parameter["bottom_size"][0] = body_parameter["size"][0]
            parameter["bottom_size"][1] = randRange(
                1, 0.9 * body_parameter["thickness"][0], body_parameter["thickness"][0]
            )[0]
            parameter["bottom_size"][3] = randRange(
                1, 0.8 * door_parameter["size"][2], 1.2 * door_parameter["size"][2]
            )[0]
            parameter["bottom_size"][2] = randRange(
                1, 0.5 * parameter["bottom_size"][3], parameter["bottom_size"][3]
            )[0]

            standard_button_size = parameter["button_1_size"]
            standard_button_size *= randRange(3, 0.8, 1.2)
            if standard_button_size[1] > parameter["bottom_size"][1]:
                standard_button_size[1] = (
                    randRange(1, 0.5, 0.7)[0] * parameter["bottom_size"][1]
                )
            if (
                standard_button_size[0] * parameter["num_buttons"][0]
                > parameter["bottom_size"][0]
            ):
                standard_button_size[0] = (
                    randRange(1, 0.5, 0.7)[0]
                    * parameter["bottom_size"][0]
                    / parameter["num_buttons"][0]
                )

            comprehensive_button_seperate = randRange(
                1,
                parameter["num_buttons"][0] * standard_button_size[0],
                0.9 * parameter["bottom_size"][0],
            )[0]
            button_seperate = (
                comprehensive_button_seperate / (parameter["num_buttons"][0] - 1)
                - standard_button_size[0]
            )
            for i in range(1, parameter["num_buttons"][0] + 1):
                parameter[f"button_{i}_size"] = standard_button_size
                parameter[f"button_{i}_offset"][0] = (
                    -comprehensive_button_seperate / 2
                    + (i - 1) * (button_seperate + standard_button_size[0])
                )
                parameter[f"button_{i}_offset"][1] = 0

            if parameter["num_buttons"][0] < 10:
                for i in range(parameter["num_buttons"][0] + 1, 10):
                    parameter[f"button_{i}_size"][:] = 0
                    parameter[f"button_{i}_offset"][:] = 0

            parameter["position"][0] = 0
            parameter["position"][1] = (
                body_parameter["size"][1] / 2 - body_parameter["thickness"][0] / 2
            )
            parameter["position"][2] = body_parameter["size"][2] / 2

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cuboidal_Baffle":
            parameter["size"][0] = randRange(
                1,
                min(0.7 * body_parameter["size"][0], 0.7 * parameter["size"][0]),
                body_parameter["size"][0],
            )[0]
            parameter["size"][1] *= randRange(1, 0.7, 1.3)[0]
            parameter["size"][2] = randRange(
                1,
                min(0.05 * body_parameter["size"][2], 0.7 * parameter["size"][2]),
                0.1 * body_parameter["size"][2],
            )[0]

            parameter["rotation"] = np.array(body_parameter["rotation"])

            parameter["position"][0] = randRange(
                1,
                -body_parameter["size"][0] / 2 + parameter["size"][0] / 2,
                body_parameter["size"][0] / 2 - parameter["size"][0] / 2,
            )[0]
            parameter["position"][1] = body_parameter["size"][1] / 2
            if "Flat_Top" in templates:
                concepts[indexes["Flat_Top"]]["parameters"]["size"][1] = (
                    concepts[indexes["Flat_Top"]]["parameters"]["size"][1]
                    * randRange(1, 0.5, 1.2)[0]
                )
                parameter["position"][1] += concepts[indexes["Flat_Top"]]["parameters"][
                    "size"
                ][1]
            elif "Top_With_Burner" in templates:
                concepts[indexes["Top_With_Burner"]]["parameters"]["bottom_size"][1] = (
                    concepts[indexes["Top_With_Burner"]]["parameters"]["bottom_size"][1]
                    * randRange(1, 0.5, 1.2)[0]
                )
                parameter["position"][1] += concepts[indexes["Top_With_Burner"]][
                    "parameters"
                ]["bottom_size"][1]
            parameter["position"][2] = randRange(
                1,
                -body_parameter["size"][2] / 2 + parameter["size"][2] / 2,
                0.95 * (-body_parameter["size"][2] / 2 + parameter["size"][2] / 2),
            )[0]
            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Flat_Top":
            parameter["size"][[0, 2]] = body_parameter["size"][[0, 2]]
            if "Cuboidal_Baffle" not in templates:
                parameter["size"][1] *= randRange(1, 0.5, 1.2)[0]

            parameter["rotation"] = np.array(body_parameter["rotation"])
            parameter["position"][[0, 2]] = 0
            parameter["position"][1] = body_parameter["size"][1] / 2

            if "Controller_With_Button" in templates:
                controller_upper_width = concepts[indexes["Controller_With_Button"]][
                    "parameters"
                ]["bottom_size"][2]
                parameter["size"][2] += controller_upper_width
                parameter["position"][2] += controller_upper_width / 2

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Top_With_Burner":
            parameter["bottom_size"][0], parameter["bottom_size"][2] = (
                body_parameter["size"][0],
                body_parameter["size"][2],
            )
            if "Cuboidal_Baffle" not in templates:
                parameter["bottom_size"][1] *= randRange(1, 0.5, 1.2)[0]

            parameter["rotation"] = np.array(body_parameter["rotation"])
            parameter["position"][[0, 2]] = 0
            parameter["position"][1] = body_parameter["size"][1] / 2

            num_burners = np.random.choice(range(1, 7))
            parameter["num_burners"][0] = num_burners

            if "Controller_With_Button" in templates:
                controller_upper_width = concepts[indexes["Controller_With_Button"]][
                    "parameters"
                ]["bottom_size"][2]
                parameter["bottom_size"][2] += controller_upper_width
                parameter["position"][2] += controller_upper_width / 2

            if "Cuboidal_Baffle" in templates:
                burner_move_z = concepts[indexes["Cuboidal_Baffle"]]["parameters"][
                    "size"
                ][2]
                comprehensive_burner_seperate = np.zeros(2)
                comprehensive_burner_seperate[0] = (
                    randRange(1, 0.8, 0.95)[0] * body_parameter["size"][0]
                )
                comprehensive_burner_seperate[1] = randRange(1, 0.8, 0.95)[0] * (
                    body_parameter["size"][2] - burner_move_z
                )

            else:
                burner_move_z = 0
                comprehensive_burner_seperate = np.zeros(2)
                comprehensive_burner_seperate[0] = (
                    randRange(1, 0.8, 0.95)[0] * body_parameter["size"][0]
                )
                comprehensive_burner_seperate[1] = (
                    randRange(1, 0.8, 0.95)[0] * body_parameter["size"][2]
                )

            standard_burner_size = np.zeros(3)
            standard_burner_size[0] = (
                randRange(1, 0.9, 0.95)[0]
                * comprehensive_burner_seperate[0]
                / ((num_burners + 1) // 2)
            )
            standard_burner_size[1] = (
                parameter["burner_1_size"][1] * randRange(1, 0.8, 1.3)[0]
            )
            standard_burner_size[2] = (
                randRange(1, 0.9, 0.95)[0] * comprehensive_burner_seperate[1] / 2
            )

            final_burner_size = copy.deepcopy(standard_burner_size)
            final_burner_size[[0, 2]] = np.min(final_burner_size[[0, 2]])

            standard_burner_thickness = np.array(
                [randRange(1, 0.05, 0.1)[0] * final_burner_size[0]]
            )
            standard_burner_central_size = np.zeros(2)
            standard_burner_central_size[0] = (
                randRange(1, 0.5, 0.8)[0]
                * min(
                    [
                        final_burner_size[0] - standard_burner_thickness[0],
                        body_parameter["size"][2] / 3,
                    ]
                )
                / 2
            )
            standard_burner_central_size[1] = (
                randRange(1, 0.8, 1)[0] * standard_burner_size[1]
            )

            burner_seperate = np.zeros(2)
            burner_seperate[0] = (
                comprehensive_burner_seperate[0] / ((num_burners + 1) // 2)
                - standard_burner_size[0]
            )
            burner_seperate[1] = (
                comprehensive_burner_seperate[1] - standard_burner_size[2]
            )

            comprehensive_burner_seperate -= standard_burner_size[[0, 2]]

            if num_burners % 2 == 0:
                for i in range(1, num_burners + 1, 2):
                    parameter[f"burner_{i}_size"] = final_burner_size
                    parameter[f"burner_{i}_central_size"] = standard_burner_central_size
                    parameter[f"burner_{i}_thickness"] = standard_burner_thickness
                    parameter[f"burner_{i}_central_offset"][:] = 0
                    parameter[f"burner_{i}_offset"][0] = -comprehensive_burner_seperate[
                        0
                    ] / 2 + (i // 2) * (standard_burner_size[0] + burner_seperate[0])
                    parameter[f"burner_{i}_offset"][1] = (
                        -comprehensive_burner_seperate[1] / 2 + burner_move_z / 2
                    )

                    parameter[f"burner_{i + 1}_size"] = final_burner_size
                    parameter[f"burner_{i + 1}_central_size"] = (
                        standard_burner_central_size
                    )
                    parameter[f"burner_{i + 1}_thickness"] = standard_burner_thickness
                    parameter[f"burner_{i + 1}_central_offset"][:] = 0
                    parameter[f"burner_{i + 1}_offset"][0] = (
                        -comprehensive_burner_seperate[0] / 2
                        + (i // 2) * (standard_burner_size[0] + burner_seperate[0])
                    )
                    parameter[f"burner_{i + 1}_offset"][1] = (
                        comprehensive_burner_seperate[1] / 2 + burner_move_z / 2
                    )
            else:
                seen_single_burner = False
                i = 1
                while i <= num_burners:
                    if not seen_single_burner and (
                        random.random() < 0.5 or i == num_burners
                    ):
                        parameter[f"burner_{i}_size"][[0, 1]] = final_burner_size[
                            [0, 1]
                        ]
                        parameter[f"burner_{i}_size"][2] = (
                            final_burner_size[2] + burner_seperate[1]
                        )

                        parameter[f"burner_{i}_central_size"] = (
                            standard_burner_central_size
                        )
                        parameter[f"burner_{i}_thickness"] = standard_burner_thickness
                        parameter[f"burner_{i}_central_offset"][:] = 0
                        parameter[f"burner_{i}_offset"][0] = (
                            -comprehensive_burner_seperate[0] / 2
                            + (i // 2) * (standard_burner_size[0] + burner_seperate[0])
                        )
                        parameter[f"burner_{i}_offset"][1] = burner_move_z / 2

                        i += 1
                        seen_single_burner = True

                    else:
                        parameter[f"burner_{i}_size"] = final_burner_size
                        parameter[f"burner_{i}_central_size"] = (
                            standard_burner_central_size
                        )
                        parameter[f"burner_{i}_thickness"] = standard_burner_thickness
                        parameter[f"burner_{i}_central_offset"][:] = 0
                        parameter[f"burner_{i}_offset"][0] = (
                            -comprehensive_burner_seperate[0] / 2
                            + (i // 2) * (standard_burner_size[0] + burner_seperate[0])
                        )
                        parameter[f"burner_{i}_offset"][1] = (
                            -comprehensive_burner_seperate[1] / 2 + burner_move_z / 2
                        )

                        parameter[f"burner_{i + 1}_size"] = final_burner_size
                        parameter[f"burner_{i + 1}_central_size"] = (
                            standard_burner_central_size
                        )
                        parameter[f"burner_{i + 1}_thickness"] = (
                            standard_burner_thickness
                        )
                        parameter[f"burner_{i + 1}_central_offset"][:] = 0
                        parameter[f"burner_{i + 1}_offset"][0] = (
                            -comprehensive_burner_seperate[0] / 2
                            + (i // 2) * (standard_burner_size[0] + burner_seperate[0])
                        )
                        parameter[f"burner_{i + 1}_offset"][1] = (
                            comprehensive_burner_seperate[1] / 2 + burner_move_z / 2
                        )

                        i += 2

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Multilevel_Leg":
            parameter["num_legs"][0] = np.random.choice([1, 2, 3, 4])

            if parameter["num_legs"][0] == 1:
                parameter["front_legs_size"][0] = (
                    randRange(1, 0.8, 1.0)[0] * body_parameter["size"][0]
                )
                parameter["front_legs_size"][1] *= randRange(1, 0.8, 1.0)[0]
                parameter["front_legs_size"][2] = (
                    randRange(1, 0.8, 1.0)[0] * body_parameter["size"][2]
                )

            else:
                mul_front_leg_size = randRange(
                    parameter["front_legs_size"].shape[0], 0.7, 1.3
                )
                parameter["front_legs_size"] *= mul_front_leg_size
                if parameter["front_legs_size"][2] > body_parameter["size"][2]:
                    parameter["front_legs_size"][2] = body_parameter["size"][2]

                mul_rear_leg_size = randRange(
                    parameter["rear_legs_size"].shape[0], 0.7, 1.3
                )
                parameter["rear_legs_size"] *= mul_rear_leg_size
                parameter["rear_legs_size"][1] = parameter["front_legs_size"][1]

            parameter["legs_separation"][0] = randRange(1, 0.9, 1.0)[0] * (
                body_parameter["size"][0] - parameter["front_legs_size"][0]
            )
            parameter["legs_separation"][1] = randRange(1, 0.9, 1.0)[0] * (
                body_parameter["size"][0] - parameter["rear_legs_size"][0]
            )
            if parameter["num_legs"][0] > 2:
                parameter["legs_separation"][2] = randRange(1, 0.9, 1.0)[0] * (
                    body_parameter["size"][2]
                    - parameter["front_legs_size"][2] / 2
                    - parameter["rear_legs_size"][2] / 2
                )
            else:
                parameter["legs_separation"][2] = 0

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = (
                body_parameter["position"][1] - body_parameter["size"][1] / 2
            )
            parameter["position"][2] = (
                body_parameter["position"][2]
                + randRange(
                    1,
                    -body_parameter["size"][2] / 2
                    - (
                        -parameter["legs_separation"][2] / 2
                        - parameter["rear_legs_size"][2] / 2
                    ),
                    body_parameter["size"][2] / 2
                    - (
                        parameter["legs_separation"][2] / 2
                        + parameter["front_legs_size"][2] / 2
                    ),
                )[0]
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

    return concepts


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
        oven_type = get_oven_type()
        existing_concept_templates = concept_template_existence(oven_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, oven_type)

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
            oven_type = get_oven_type()
            existing_concept_templates = concept_template_existence(oven_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, oven_type)
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
