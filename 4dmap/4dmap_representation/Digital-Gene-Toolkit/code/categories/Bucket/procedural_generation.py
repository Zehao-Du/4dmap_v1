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


def get_bucket_type():
    total_type = ["prismatic", "cylindrical"]
    weights = [1, 1]
    box_type = random.choices(total_type, weights=weights, k=1)[0]
    return box_type


def concept_template_existence(bucket_type):
    concept_template_variation = {
        "body": {"template": ["Cylindrical_Body", "Prismatic_Body"], "necessary": True},
        "cover": {"template": ["Single_Cylinder"], "necessary": False},
        "handle": {
            "template": [
                "Trifold_Handle",
                "Curved_Handle",
                "Round_U_Handle",
                "Flat_U_Handle",
            ],
            "necessary": True,
        },
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part_name == "body":
            if bucket_type == "prismatic":
                templates.append("Prismatic_Body")
            elif bucket_type == "cylindrical":
                templates.append("Cylindrical_Body")
        elif part_name != "cover":
            templates.append(random.choice(part["template"]))
        elif part_name == "cover":
            if bucket_type == "cylindrical":
                existence = random.randint(0, 1)
                if existence > 0.5:
                    templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, bucket_type):
    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    is_prismatic: bool = False
    is_cuboid: bool = False
    body_inner_radius: float = 0
    body_outer_radius: float = 0
    body_width: float = 0
    body_length: float = 0
    body_thickness: float = 0
    body_height: float = 0
    body_difference: float = 0
    cover_existence: bool = False
    cover_angle: float = 0

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Cylindrical_Body":
            parameter["outer_size"] = np.array(
                [
                    randRange(1, 0.5, 0.7)[0],
                    randRange(1, 0.4, 0.6)[0],
                    randRange(1, 0.8, 1.8)[0],
                ]
            )

            body_height = parameter["outer_size"][2]
            body_difference = parameter["outer_size"][0] - parameter["outer_size"][1]
            body_thickness = randRange(1, 0.02, 0.06)[0]

            parameter["inner_size"] = np.array(
                [
                    parameter["outer_size"][0]
                    - body_thickness * randRange(1, 0.9, 1.1)[0],
                    parameter["outer_size"][1]
                    - body_thickness * randRange(1, 0.9, 1.1)[0],
                    parameter["outer_size"][2]
                    - body_thickness * randRange(1, 0.9, 1.1)[0],
                ]
            )

            body_outer_radius = parameter["outer_size"][0]
            body_inner_radius = parameter["inner_size"][0]

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Prismatic_Body":
            is_prismatic = True

            parameter["outer_size"] = np.array(
                [
                    randRange(1, 0.5, 0.7)[0],
                    randRange(1, 0.4, 0.6)[0],
                    randRange(1, 0.8, 1.8)[0],
                ]
            )

            body_height = parameter["outer_size"][2]
            body_difference = parameter["outer_size"][0] - parameter["outer_size"][1]
            body_thickness = randRange(1, 0.02, 0.06)[0]

            parameter["inner_size"] = np.array(
                [
                    parameter["outer_size"][0]
                    - body_thickness * randRange(1, 0.9, 1.1)[0],
                    parameter["outer_size"][1]
                    - body_thickness * randRange(1, 0.9, 1.1)[0],
                    parameter["outer_size"][2]
                    - body_thickness * randRange(1, 0.9, 1.1)[0],
                ]
            )

            body_outer_radius = parameter["outer_size"][0]
            body_inner_radius = parameter["inner_size"][0]

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Single_Cylinder":
            cover_existence = True

            parameter["size"] = np.array(
                [
                    randRange(1, body_inner_radius, body_outer_radius)[0],
                    body_thickness * randRange(1, 0.9, 1.1)[0],
                ]
            )

            all_position = ["cover", "open"]
            weight_of_position = [1, 1]
            position = random.choices(all_position, weight_of_position, k=1)[0]
            if position == "cover":
                parameter["position"] = np.array(
                    [0, body_height / 2 + parameter["size"][1] / 2, 0]
                )
            if position == "open":
                r = randRange(1, 0, body_inner_radius * 0.7)[0]
                cover_angle = randRange(1, 0, 2 * np.pi)[0]
                parameter["position"] = np.array(
                    [
                        r * np.cos(cover_angle),
                        body_height / 2 + parameter["size"][1] / 2,
                        r * np.sin(cover_angle),
                    ]
                )

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Curved_Handle":
            all_position = ["on", "in", "out"]
            weight_of_position = [1, 1, 1]
            position = random.choices(all_position, weight_of_position, k=1)[0]

            if cover_existence:
                position = "out"
            if is_cuboid or is_prismatic:
                while position == "in":
                    position = random.choices(all_position, weight_of_position, k=1)[0]

            if position == "on":
                parameter["rotation"] = np.array(
                    [randRange(1, 0, 80)[0], randRange(1, 0, 360)[0], 0]
                )
                parameter["exist_angle"] = np.array([randRange(1, 195, 200)[0]])
                height_angle = (np.deg2rad(parameter["exist_angle"][0]) - np.pi) / 2
                width_angle = np.deg2rad(parameter["exist_angle"][0] / 2)
                if parameter["exist_angle"][0] > 180:
                    width_angle = np.deg2rad((360 - parameter["exist_angle"][0]) / 2)

                distance = randRange(1, body_inner_radius, body_outer_radius)[0]
                if is_prismatic:
                    distance = randRange(1, body_inner_radius, body_outer_radius)[
                        0
                    ] / np.sqrt(2)
                if is_cuboid:
                    distance = (
                        randRange(1, body_width - body_thickness, body_width)[0] / 2
                    )

                parameter["radius"] = np.array(
                    [
                        distance / np.sin(width_angle),
                        randRange(
                            1,
                            0.008,
                            body_thickness * 0.2
                            + body_outer_radius * 0.06
                            + body_width / 2 * 0.06,
                        )[0],
                    ]
                )
                if is_prismatic:
                    parameter["radius"][1] = randRange(
                        1,
                        0.008,
                        body_thickness * 0.2
                        + body_outer_radius / np.sqrt(2) * 0.06
                        + body_width / 2 * 0.06,
                    )[0]

                parameter["position"] = np.array(
                    [
                        np.sin(np.deg2rad(parameter["rotation"][1]))
                        * (
                            parameter["radius"][0]
                            * np.sin(height_angle)
                            * np.cos(np.deg2rad(90 - parameter["rotation"][0]))
                        ),
                        body_height / 2
                        + parameter["radius"][0]
                        * np.sin(height_angle)
                        * np.cos(np.deg2rad(parameter["rotation"][0])),
                        np.cos(np.deg2rad(parameter["rotation"][1]))
                        * (
                            parameter["radius"][0]
                            * np.sin(height_angle)
                            * np.cos(np.deg2rad(90 - parameter["rotation"][0]))
                        ),
                    ]
                )

                parameter["rotation"][2] -= (parameter["exist_angle"][0] - 180) / 2

            if position == "in":
                parameter["rotation"] = np.array(
                    [randRange(1, 0, 70)[0], randRange(1, 0, 360)[0], 0]
                )
                parameter["exist_angle"] = np.array([randRange(1, 175, 180)[0]])
                height_angle = (np.deg2rad(parameter["exist_angle"][0]) - np.pi) / 2
                width_angle = np.deg2rad(parameter["exist_angle"][0] / 2)
                if parameter["exist_angle"][0] > 180:
                    width_angle = np.deg2rad((360 - parameter["exist_angle"][0]) / 2)

                parameter["radius"][1] = randRange(
                    1,
                    0.008,
                    body_thickness * 0.2
                    + body_outer_radius * 0.05
                    + body_width / 2 * 0.05,
                )[0]
                if is_prismatic:
                    parameter["radius"][1] = randRange(
                        1,
                        0.008,
                        body_thickness * 0.2
                        + body_outer_radius / np.sqrt(2) * 0.05
                        + body_width / 2 * 0.05,
                    )[0]

                distance = body_inner_radius - 0.008
                if is_prismatic:
                    distance = (
                        body_inner_radius / np.sqrt(2)
                        - parameter["radius"][1] * np.cos(height_angle)
                        + 0.008
                    )
                if is_cuboid:
                    distance = (body_width - body_thickness * 2) / 2
                if body_difference > 0.15:
                    distance -= 0.02
                if body_difference < 0:
                    distance += 0.01

                parameter["radius"][0] = distance / np.sin(width_angle)

                parameter["position"] = np.array(
                    [
                        np.sin(np.deg2rad(parameter["rotation"][1]))
                        * (
                            parameter["radius"][0]
                            * np.sin(height_angle)
                            * np.cos(np.deg2rad(90 - parameter["rotation"][0]))
                        ),
                        body_height * randRange(1, 0.4, 0.45)[0]
                        + parameter["radius"][0]
                        * np.sin(height_angle)
                        * np.cos(np.deg2rad(parameter["rotation"][0])),
                        np.cos(np.deg2rad(parameter["rotation"][1]))
                        * (
                            parameter["radius"][0]
                            * np.sin(height_angle)
                            * np.cos(np.deg2rad(90 - parameter["rotation"][0]))
                        ),
                    ]
                )

                parameter["rotation"][2] -= (parameter["exist_angle"][0] - 180) / 2

            if position == "out":
                parameter["exist_angle"] = np.array([randRange(1, 190, 250)[0]])
                if is_prismatic:
                    parameter["exist_angle"] = np.array([randRange(1, 240, 250)[0]])
                if is_cuboid:
                    parameter["exist_angle"] = np.array([randRange(1, 240, 250)[0]])
                if cover_existence:
                    parameter["exist_angle"] = np.array([randRange(1, 230, 250)[0]])
                height_angle = (np.deg2rad(parameter["exist_angle"][0]) - np.pi) / 2
                width_angle = np.deg2rad(parameter["exist_angle"][0] / 2)
                if parameter["exist_angle"][0] > 180:
                    width_angle = np.deg2rad((360 - parameter["exist_angle"][0]) / 2)

                parameter["radius"][1] = randRange(
                    1,
                    0.008,
                    body_thickness * 0.2
                    + body_outer_radius * 0.05
                    + body_width / 2 * 0.05,
                )[0]
                if is_prismatic:
                    parameter["radius"][1] = randRange(
                        1,
                        0.008,
                        body_thickness * 0.2
                        + body_outer_radius / np.sqrt(2) * 0.05
                        + body_width / 2 * 0.05,
                    )[0]

                distance = body_outer_radius - 0.008
                if is_prismatic:
                    distance = (
                        body_outer_radius / np.sqrt(2)
                        - 0.008
                        + 0.2 * parameter["radius"][1]
                    )
                if is_cuboid:
                    distance = body_width / 2
                if body_difference > 0.15:
                    distance -= 0.02
                if body_difference < 0:
                    distance += 0.01

                parameter["radius"][0] = distance / np.sin(width_angle)

                parameter["rotation"] = np.array([0.0, randRange(1, 0, 360)[0], 0.0])
                all_rotation_angle = ["acute", "obtuse"]
                weight_of_angle = [1, 1]
                rotation_angle = random.choices(all_rotation_angle, weight_of_angle)[0]
                if rotation_angle == "acute":
                    parameter["rotation"][0] = randRange(1, 0, 90)[0]
                if rotation_angle == "obtuse":
                    parameter["rotation"][0] = randRange(1, 90, 130)[0]
                if is_cuboid:
                    parameter["rotation"][0] = randRange(1, 0, 60)[0]
                    rotation_angle = "acute"

                if cover_angle != 0:
                    parameter["rotation"][1] = np.mod(
                        450 - np.rad2deg(cover_angle), 360
                    )
                    all_lean = ["close", "opposite"]
                    weight_of_lean = [1, 2]
                    lean = random.choices(all_lean, weight_of_lean)[0]
                    if lean == "close":
                        parameter["rotation"][0] = randRange(1, 0, 30)[0]
                    if lean == "opposite":
                        if rotation_angle == "acute":
                            parameter["rotation"][0] = -randRange(1, 0, 90)[0]
                        if rotation_angle == "obtuse":
                            parameter["rotation"][0] = -randRange(1, 90, 130)[0]

                parameter["rotation"][2] -= (parameter["exist_angle"][0] - 180) / 2

                parameter["position"] = np.array(
                    [
                        np.sin(np.deg2rad(parameter["rotation"][1]))
                        * (
                            parameter["radius"][0]
                            * np.sin(height_angle)
                            * np.cos(np.deg2rad(90 - parameter["rotation"][0]))
                        ),
                        body_height * randRange(1, 0.3, 0.42)[0]
                        + parameter["radius"][0]
                        * np.sin(height_angle)
                        * np.cos(np.deg2rad(parameter["rotation"][0])),
                        np.cos(np.deg2rad(parameter["rotation"][1]))
                        * (
                            parameter["radius"][0]
                            * np.sin(height_angle)
                            * np.cos(np.deg2rad(90 - parameter["rotation"][0]))
                        ),
                    ]
                )

            if is_prismatic:
                all_direction = ["x", "z"]
                weight_of_direction = [1, 1]
                direction = random.choices(all_direction, weight_of_direction, k=1)[0]
                prismatic_direction = random.randint(0, 1) * 2 - 1
                if direction == "x":
                    parameter["rotation"][0] *= prismatic_direction
                    parameter["rotation"][1] = 0
                    parameter["position"][0] = 0
                    parameter["position"][2] = prismatic_direction * (
                        parameter["radius"][0]
                        * np.sin(height_angle)
                        * np.cos(
                            np.deg2rad(
                                90 - prismatic_direction * parameter["rotation"][0]
                            )
                        )
                    )
                else:
                    parameter["rotation"][0] *= prismatic_direction
                    parameter["rotation"][1] = 90
                    parameter["position"][2] = 0
                    parameter["position"][0] = prismatic_direction * (
                        parameter["radius"][0]
                        * np.sin(height_angle)
                        * np.cos(
                            np.deg2rad(
                                90 - prismatic_direction * parameter["rotation"][0]
                            )
                        )
                    )

            if is_cuboid:
                prismatic_direction = random.randint(0, 1) * 2 - 1
                parameter["rotation"][0] *= prismatic_direction
                parameter["rotation"][1] = 90
                parameter["position"][2] = 0
                parameter["position"][0] = prismatic_direction * (
                    parameter["radius"][0]
                    * np.sin(height_angle)
                    * np.cos(
                        np.deg2rad(90 - prismatic_direction * parameter["rotation"][0])
                    )
                )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Trifold_Handle":
            all_position = ["on", "in", "out"]
            weight_of_position = [1, 1, 1]
            position = random.choices(all_position, weight_of_position, k=1)[0]

            if cover_existence:
                position = "out"

            parameter["vertical_thickness"] = np.array(
                [
                    randRange(
                        1,
                        0.02,
                        body_thickness * 0.3
                        + body_outer_radius * 0.06
                        + body_width * 0.06,
                    )[0],
                    randRange(
                        1,
                        0.02,
                        body_thickness * 0.5
                        + body_outer_radius * 0.14
                        + body_width * 0.14,
                    )[0],
                ]
            )
            if is_prismatic:
                parameter["vertical_thickness"] = np.array(
                    [
                        randRange(
                            1,
                            0.02,
                            body_thickness * 0.3
                            + body_outer_radius / np.sqrt(2) * 0.06
                            + body_width * 0.06,
                        )[0],
                        randRange(
                            1,
                            0.02,
                            body_thickness * 0.5
                            + body_outer_radius / np.sqrt(2) * 0.14
                            + body_width * 0.14,
                        )[0],
                    ]
                )

            parameter["vertical_length"] = np.array([randRange(1, 0.5, 0.7)[0], 0.0])
            parameter["vertical_length"][1] = parameter["vertical_length"][0]

            parameter["horizontal_thickness"] = np.array(
                [
                    randRange(
                        1,
                        0.02,
                        body_thickness * 0.3
                        + body_outer_radius * 0.06
                        + body_width / 2 * 0.06,
                    )[0],
                    parameter["vertical_thickness"][1],
                ]
            )
            if is_prismatic:
                parameter["horizontal_thickness"][0] = randRange(
                    1,
                    0.02,
                    body_thickness * 0.3
                    + body_outer_radius / np.sqrt(2) * 0.06
                    + body_width / 2 * 0.06,
                )[0]

            if position == "on":
                parameter["vertical_length"] = np.array(
                    [randRange(1, 0.3, 0.7)[0], 0.0]
                )
                if is_cuboid:
                    parameter["vertical_length"][0] = randRange(
                        1, body_width * 0.3, 0.3
                    )[0]
                parameter["vertical_length"][1] = parameter["vertical_length"][0]
                parameter["vertical_rotation"] = np.array(
                    [(random.randint(0, 1) * 2 - 1) * randRange(1, 0, 25)[0], 0.0]
                )
                parameter["vertical_rotation"][1] = -parameter["vertical_rotation"][0]
                parameter["vertical_separation"] = np.array(
                    [randRange(1, body_inner_radius * 2, body_outer_radius * 2)[0]]
                )
                if is_prismatic:
                    parameter["vertical_separation"] = np.array(
                        [
                            randRange(1, body_inner_radius * 2, body_outer_radius * 2)[
                                0
                            ]
                            / np.sqrt(2)
                        ]
                    )
                if is_cuboid:
                    parameter["vertical_separation"] = np.array(
                        [randRange(1, body_width - body_thickness * 2, body_width)[0]]
                    )

                parameter["position"] = np.array([0, body_height / 2, 0])

                parameter["rotation"] = np.array(
                    [
                        (random.randint(0, 1) * 2 - 1) * randRange(1, 0, 80)[0],
                        randRange(1, 0, 360)[0],
                        0,
                    ]
                )

            if position == "in":
                parameter["vertical_length"] = np.array(
                    [randRange(1, 0.4, 0.7)[0], 0.0]
                )
                if is_cuboid:
                    parameter["vertical_length"][0] = randRange(
                        1, body_width * 0.3, 0.3
                    )[0]
                    parameter["vertical_thickness"][0] = randRange(1, 0.008, 0.05)[0]
                parameter["vertical_length"][1] = parameter["vertical_length"][0]
                parameter["vertical_rotation"] = np.array([randRange(1, 0, 20)[0], 0.0])
                parameter["vertical_rotation"][1] = -parameter["vertical_rotation"][0]
                parameter["vertical_separation"] = np.array(
                    [body_inner_radius * 2 - parameter["vertical_thickness"][0]]
                )
                if is_prismatic:
                    parameter["vertical_separation"] = np.array(
                        [
                            body_inner_radius / np.sqrt(2) * 2
                            - parameter["vertical_thickness"][0]
                        ]
                    )
                if is_cuboid:
                    parameter["vertical_separation"] = np.array(
                        [
                            body_width
                            - body_thickness * 2
                            - parameter["vertical_thickness"][0]
                        ]
                    )
                if body_difference < 0:
                    parameter["vertical_separation"][0] += 0.01
                if body_difference > 0.15:
                    parameter["vertical_separation"][0] -= 0.02

                parameter["position"] = np.array(
                    [0, body_height * randRange(1, 0.3, 0.45)[0], 0]
                )
                if is_cuboid:
                    parameter["position"][1] = body_height * randRange(1, 0.4, 0.45)[0]

                parameter["rotation"] = np.array(
                    [
                        (random.randint(0, 1) * 2 - 1) * randRange(1, 0, 50)[0],
                        randRange(1, 0, 360)[0],
                        0,
                    ]
                )

            if position == "out":
                parameter["vertical_length"] = np.array(
                    [randRange(1, body_outer_radius * 1.4, 0.7)[0], 0.0]
                )
                if is_prismatic:
                    parameter["vertical_length"][0] = randRange(
                        1, body_outer_radius / np.sqrt(2) * 1.4, 0.7
                    )[0]
                if is_cuboid:
                    parameter["vertical_length"][0] = randRange(
                        1, body_width * 0.3, 0.3
                    )[0]
                    parameter["vertical_thickness"][0] = randRange(1, 0.03, 0.05)[0]
                parameter["vertical_length"][1] = parameter["vertical_length"][0]
                parameter["vertical_rotation"] = np.array(
                    [-randRange(1, 7, 10)[0], 0.0]
                )
                parameter["vertical_rotation"][1] = -parameter["vertical_rotation"][0]
                parameter["vertical_separation"] = np.array(
                    [body_outer_radius * 2 + parameter["vertical_thickness"][0] - 0.015]
                )
                if is_prismatic:
                    parameter["vertical_separation"] = np.array(
                        [
                            body_outer_radius / np.sqrt(2) * 2
                            + parameter["vertical_thickness"][0]
                            - 0.015
                        ]
                    )
                if is_cuboid:
                    parameter["vertical_separation"] = np.array([body_width * 1.04])
                if body_difference < 0:
                    parameter["vertical_separation"][0] += 0.01
                if body_difference > 0.1:
                    parameter["vertical_separation"][0] -= 0.03

                parameter["position"] = np.array(
                    [0, body_height * randRange(1, 0.35, 0.4)[0], 0]
                )

                parameter["rotation"] = np.array([0.0, randRange(1, 0, 360)[0], 0])

                all_rotation_angle = ["acute", "obtuse"]
                weight_of_angle = [1, 1]
                rotation_angle = random.choices(all_rotation_angle, weight_of_angle)[0]
                if rotation_angle == "acute":
                    parameter["rotation"][0] = (
                        random.randint(0, 1) * 2 - 1
                    ) * randRange(1, 0, 90)[0]
                if rotation_angle == "obtuse":
                    parameter["rotation"][0] = (
                        random.randint(0, 1) * 2 - 1
                    ) * randRange(1, 90, 110)[0]

                if cover_angle != 0:
                    parameter["rotation"][1] = np.mod(
                        450 - np.rad2deg(cover_angle), 360
                    )
                    all_lean = ["close", "opposite"]
                    weight_of_lean = [1, 2]
                    lean = random.choices(all_lean, weight_of_lean)[0]
                    if lean == "close":
                        parameter["rotation"][0] = randRange(1, 0, 30)[0]
                    if lean == "opposite":
                        if rotation_angle == "acute":
                            parameter["rotation"][0] = -randRange(1, 0, 90)[0]
                        if rotation_angle == "obtuse":
                            parameter["rotation"][0] = -randRange(1, 90, 110)[0]

            if is_prismatic:
                all_direction = ["x", "z"]
                weight_of_direction = [1, 1]
                direction = random.choices(all_direction, weight_of_direction, k=1)[0]
                if direction == "x":
                    parameter["rotation"][1] = 0
                if direction == "z":
                    parameter["rotation"][1] = 90

            if is_cuboid:
                parameter["rotation"][0] = randRange(1, 0, 50)[0]
                parameter["rotation"][1] = 90

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Round_U_Handle":
            parameter["inner_radius"] = np.array(
                [
                    randRange(
                        1,
                        0.01,
                        body_thickness * 0.05
                        + body_outer_radius * 0.065
                        + body_width / 2 * 0.09,
                    )[0]
                ]
            )
            if is_prismatic:
                parameter["inner_radius"] = np.array(
                    [
                        randRange(
                            1,
                            0.01,
                            body_thickness * 0.05
                            + body_outer_radius / np.sqrt(2) * 0.065
                            + body_width / 2 * 0.09,
                        )[0]
                    ]
                )

            parameter["vertical_separation"] = np.array([body_outer_radius * 2])

            parameter["vertical_length"] = np.array([randRange(1, 0.012, 0.35)[0]])

            all_position = ["on", "in", "out"]
            weight_of_position = [1, 1, 1]
            position = random.choices(all_position, weight_of_position, k=1)[0]
            if body_difference < 0:
                while position == "in":
                    position = random.choices(all_position, weight_of_position, k=1)[0]
            if cover_existence:
                position = "out"

            if position == "on":
                parameter["vertical_separation"] = np.array(
                    [randRange(1, body_inner_radius * 2, body_outer_radius * 2)[0]]
                )
                if is_prismatic:
                    parameter["vertical_separation"] = np.array(
                        [
                            randRange(1, body_inner_radius * 2, body_outer_radius * 2)[
                                0
                            ]
                            / np.sqrt(2)
                        ]
                    )
                if is_cuboid:
                    parameter["vertical_separation"] = np.array(
                        [randRange(1, body_width - body_thickness * 2, body_width)[0]]
                    )

                parameter["position"] = np.array([0.0, body_height / 2, 0.0])

                parameter["rotation"] = np.array(
                    [
                        (random.randint(0, 1) * 2 - 1) * randRange(1, 0, 90)[0],
                        randRange(1, 0, 360)[0],
                        0,
                    ]
                )

            if position == "in":
                parameter["vertical_length"] = np.array([randRange(1, 0.3, 0.4)[0]])
                if is_prismatic:
                    parameter["vertical_length"] = np.array(
                        [randRange(1, body_outer_radius / np.sqrt(2), 0.4)[0]]
                    )
                if is_cuboid:
                    parameter["vertical_length"] = np.array(
                        [randRange(1, body_width / 2, 0.4)[0]]
                    )
                parameter["vertical_separation"] = np.array(
                    [body_inner_radius * 2 - parameter["inner_radius"][0] * 2]
                )
                if is_prismatic:
                    parameter["vertical_separation"] = np.array(
                        [
                            body_inner_radius * 2 / np.sqrt(2)
                            - parameter["inner_radius"][0] * 2
                            - 0.008
                        ]
                    )
                if is_cuboid:
                    parameter["vertical_separation"] = np.array(
                        [
                            body_width
                            - body_thickness * 2
                            - parameter["inner_radius"][0] * 2
                            - 0.008
                        ]
                    )

                parameter["position"] = np.array(
                    [0.0, body_height * randRange(1, 0.4, 0.45)[0], 0.0]
                )

                parameter["rotation"] = np.array(
                    [
                        (random.randint(0, 1) * 2 - 1) * randRange(1, 0, 30)[0],
                        randRange(1, 0, 360)[0],
                        0,
                    ]
                )

            if position == "out":
                if body_difference > 0:
                    parameter["vertical_length"][0] = randRange(
                        1, body_outer_radius * 0.15, body_outer_radius * 0.4
                    )[0]
                    if is_prismatic:
                        parameter["vertical_length"][0] = randRange(
                            1,
                            body_outer_radius / np.sqrt(2) * 1.5,
                            body_outer_radius / np.sqrt(2) * 1.6,
                        )[0]
                    if is_cuboid:
                        parameter["vertical_length"][0] = randRange(
                            1, body_length / 2 * 1.15, body_length / 2 * 1.2
                        )[0]
                parameter["vertical_separation"] = np.array(
                    [body_outer_radius * 2 + parameter["inner_radius"][0] * 2]
                )
                if is_prismatic:
                    parameter["vertical_separation"] = np.array(
                        [
                            body_outer_radius * 2 / np.sqrt(2)
                            + parameter["inner_radius"][0] * 2
                        ]
                    )
                if is_cuboid:
                    parameter["vertical_separation"] = np.array(
                        [body_width + parameter["inner_radius"][0] * 2]
                    )

                parameter["position"] = np.array(
                    [0.0, body_height * randRange(1, 0.35, 0.45)[0], 0.0]
                )

                x_direction = random.randint(0, 1) * 2 - 1

                parameter["rotation"] = np.array(
                    [x_direction * randRange(1, 0, 90)[0], randRange(1, 0, 360)[0], 0.0]
                )

                if body_difference > 0:
                    all_rotation_angle = ["acute", "obtuse"]
                    weight_of_angle = [1, 1]
                    rotation_angle = random.choices(
                        all_rotation_angle, weight_of_angle
                    )[0]
                    if rotation_angle == "acute":
                        parameter["rotation"][0] = x_direction * randRange(1, 0, 60)[0]
                    if rotation_angle == "obtuse":
                        parameter["vertical_length"][0] = randRange(1, 0.5, 0.65)[0]
                        parameter["rotation"][0] = (
                            x_direction * randRange(1, 100, 125)[0]
                        )

                if cover_angle != 0:
                    parameter["rotation"][1] = np.mod(
                        450 - np.rad2deg(cover_angle), 360
                    )
                    all_lean = ["close", "opposite"]
                    weight_of_lean = [1, 2]
                    lean = random.choices(all_lean, weight_of_lean)[0]
                    if lean == "close":
                        parameter["rotation"][0] = randRange(1, 0, 30)[0]
                        x_direction = 1
                        if body_difference > 0:
                            rotation_angle = "acute"
                    if lean == "opposite":
                        if parameter["rotation"][0] > 0:
                            x_direction = -1
                            parameter["rotation"][0] = -randRange(1, 0, 130)[0]
                            if body_difference > 0:
                                if rotation_angle == "acute":
                                    parameter["rotation"][0] = (
                                        x_direction * randRange(1, 0, 60)[0]
                                    )
                                if rotation_angle == "obtuse":
                                    parameter["rotation"][0] = (
                                        x_direction * randRange(1, 100, 130)[0]
                                    )

                if body_difference > 0:
                    y_angle = np.deg2rad(parameter["rotation"][1])
                    parameter["position"][0] = (
                        -x_direction
                        * np.sin(y_angle)
                        * (
                            (body_height / 2 - parameter["position"][1])
                            * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                        )
                    )
                    parameter["position"][2] = (
                        -x_direction
                        * np.cos(y_angle)
                        * (
                            (body_height / 2 - parameter["position"][1])
                            * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                        )
                    )
                    if rotation_angle == "obtuse":
                        parameter["position"][1] = (
                            body_height * randRange(1, 0.51, 0.55)[0]
                        )
                        parameter["position"][0] = (
                            -x_direction
                            * np.sin(y_angle)
                            * (
                                (-body_height / 2 + parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )
                        )
                        parameter["position"][2] = (
                            -x_direction
                            * np.cos(y_angle)
                            * (
                                (-body_height / 2 + parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )
                        )

            if is_prismatic:
                all_direction = ["x", "z"]
                weight_of_direction = [1, 1]
                direction = random.choices(all_direction, weight_of_direction, k=1)[0]

                if direction == "x":
                    parameter["rotation"][1] = 0
                    if position == "out" and body_difference > 0:
                        parameter["position"][0] = 0
                        if rotation_angle == "acute":
                            parameter["position"][2] = -x_direction * (
                                (body_height / 2 - parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )
                        if rotation_angle == "obtuse":
                            parameter["position"][2] = (
                                -x_direction
                                * (-body_height / 2 + parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )

                if direction == "z":
                    parameter["rotation"][1] = 90
                    if position == "out" and body_difference > 0:
                        parameter["position"][2] = 0
                        if rotation_angle == "acute":
                            parameter["position"][0] = -x_direction * (
                                (body_height / 2 - parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )
                        if rotation_angle == "obtuse":
                            parameter["position"][0] = (
                                -x_direction
                                * (-body_height / 2 + parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )

            if is_cuboid:
                parameter["rotation"][1] = 90
                if position == "out" and body_difference > 0:
                    parameter["position"][2] = 0
                    if rotation_angle == "acute":
                        parameter["position"][0] = -x_direction * (
                            (body_height / 2 - parameter["position"][1])
                            * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                        )
                    if rotation_angle == "obtuse":
                        parameter["position"][0] = (
                            -x_direction
                            * (-body_height / 2 + parameter["position"][1])
                            * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                        )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Flat_U_Handle":
            parameter["vertical_size"] = np.array(
                [
                    0.0,
                    randRange(
                        1,
                        0.2,
                        body_thickness * 8 + body_outer_radius * 0.1 + body_width * 0.1,
                    )[0],
                    0.0,
                ]
            )

            parameter["vertical_size"][0] = randRange(
                1,
                0.02,
                body_thickness * 0.1 + body_outer_radius * 0.11 + body_width / 2 * 0.16,
            )[0]
            parameter["vertical_size"][2] = randRange(
                1,
                0.02,
                body_thickness * 0.1 + body_outer_radius * 0.14 + body_width / 2 * 0.2,
            )[0]
            if is_prismatic:
                parameter["vertical_size"][0] = randRange(
                    1,
                    0.02,
                    body_thickness * 0.1
                    + body_outer_radius / np.sqrt(2) * 0.11
                    + body_width / 2 * 0.16,
                )[0]
                parameter["vertical_size"][2] = randRange(
                    1,
                    0.02,
                    body_thickness * 0.05
                    + body_outer_radius / np.sqrt(2) * 0.14
                    + body_width / 2 * 0.2,
                )[0]

            all_position = ["on", "in", "out"]
            weight_of_position = [1, 1, 1]
            position = random.choices(all_position, weight_of_position, k=1)[0]
            if body_difference < 0:
                while position == "in":
                    position = random.choices(all_position, weight_of_position, k=1)[0]

            if cover_existence:
                position = "out"
            position = "out"

            if position == "on":
                parameter["vertical_separation"] = np.array(
                    [randRange(1, body_inner_radius * 2, body_outer_radius * 2)[0]]
                )
                if is_prismatic:
                    parameter["vertical_separation"] = np.array(
                        [
                            randRange(1, body_inner_radius * 2, body_outer_radius * 2)[
                                0
                            ]
                            / np.sqrt(2)
                        ]
                    )
                if is_cuboid:
                    parameter["vertical_separation"] = np.array(
                        [randRange(1, body_width - body_thickness * 2, body_width)[0]]
                    )

                parameter["position"] = np.array([0.0, body_height / 2, 0.0])

                parameter["rotation"] = np.array(
                    [
                        (random.randint(0, 1) * 2 - 1) * randRange(1, 0, 80)[0],
                        randRange(1, 0, 360)[0],
                        0,
                    ]
                )

            if position == "in":
                parameter["vertical_separation"] = np.array(
                    [body_inner_radius * 2 - parameter["vertical_size"][0] - 0.008]
                )
                if is_prismatic:
                    parameter["vertical_separation"] = np.array(
                        [
                            body_inner_radius * 2 / np.sqrt(2)
                            - parameter["vertical_size"][0]
                            - 0.008
                        ]
                    )
                if is_cuboid:
                    parameter["vertical_size"][0] = randRange(1, 0.02, 0.05)[0]
                    parameter["vertical_separation"] = np.array(
                        [
                            body_width
                            - body_thickness * 2
                            - parameter["vertical_size"][0]
                        ]
                    )

                parameter["position"] = np.array(
                    [0.0, body_height * randRange(1, 0.4, 0.45)[0], 0.0]
                )

                parameter["rotation"] = np.array(
                    [
                        (random.randint(0, 1) * 2 - 1) * randRange(1, 0, 30)[0],
                        randRange(1, 0, 360)[0],
                        0,
                    ]
                )

            if position == "out":
                if body_difference > 0:
                    parameter["vertical_size"][1] = randRange(
                        1, body_outer_radius * 0.15, body_outer_radius * 0.4
                    )[0]
                    if is_prismatic:
                        parameter["vertical_size"][1] = randRange(
                            1,
                            body_outer_radius / np.sqrt(2) * 1.5,
                            body_outer_radius / np.sqrt(2) * 1.6,
                        )[0]
                    if is_cuboid:
                        parameter["vertical_size"][1] = randRange(
                            1, body_length / 2 * 1.15, body_length / 2 * 1.2
                        )[0]

                parameter["vertical_separation"] = np.array(
                    [body_outer_radius * 2 + parameter["vertical_size"][0]]
                )
                if is_prismatic:
                    parameter["vertical_separation"] = np.array(
                        [
                            body_outer_radius * 2 / np.sqrt(2)
                            + parameter["vertical_size"][0]
                        ]
                    )
                if is_cuboid:
                    parameter["vertical_separation"] = np.array(
                        [body_width + parameter["vertical_size"][0]]
                    )

                parameter["position"] = np.array(
                    [0.0, body_height * randRange(1, 0.4, 0.45)[0], 0.0]
                )

                x_direction = random.randint(0, 1) * 2 - 1

                parameter["rotation"] = np.array(
                    [x_direction * randRange(1, 0, 90)[0], randRange(1, 0, 360)[0], 0.0]
                )

                if body_difference > 0:
                    all_rotation_angle = ["acute", "obtuse"]
                    weight_of_angle = [1, 1]
                    rotation_angle = random.choices(
                        all_rotation_angle, weight_of_angle
                    )[0]
                    if rotation_angle == "acute":
                        parameter["rotation"][0] = x_direction * randRange(1, 0, 60)[0]
                    if rotation_angle == "obtuse":
                        parameter["vertical_size"][1] = randRange(1, 0.55, 0.6)[0]
                        parameter["rotation"][0] = (
                            x_direction * randRange(1, 100, 125)[0]
                        )

                if cover_angle != 0:
                    parameter["rotation"][1] = np.mod(
                        450 - np.rad2deg(cover_angle), 360
                    )
                    all_lean = ["close", "opposite"]
                    weight_of_lean = [1, 2]
                    lean = random.choices(all_lean, weight_of_lean)[0]
                    if lean == "close":
                        parameter["rotation"][0] = randRange(1, 0, 30)[0]
                        x_direction = 1
                        if body_difference > 0:
                            rotation_angle = "acute"
                    if lean == "opposite":
                        if parameter["rotation"][0] > 0:
                            x_direction = -1
                            parameter["rotation"][0] = -randRange(1, 0, 130)[0]
                            if body_difference > 0:
                                if rotation_angle == "acute":
                                    parameter["rotation"][0] = (
                                        x_direction * randRange(1, 0, 60)[0]
                                    )
                                if rotation_angle == "obtuse":
                                    parameter["rotation"][0] = (
                                        x_direction * randRange(1, 100, 130)[0]
                                    )

                if body_difference > 0:
                    y_angle = np.deg2rad(parameter["rotation"][1])
                    parameter["position"][0] = (
                        -x_direction
                        * np.sin(y_angle)
                        * (
                            (body_height / 2 - parameter["position"][1])
                            * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                        )
                    )
                    parameter["position"][2] = (
                        -x_direction
                        * np.cos(y_angle)
                        * (
                            (body_height / 2 - parameter["position"][1])
                            * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                        )
                    )
                    if rotation_angle == "obtuse":
                        parameter["position"][1] = (
                            body_height * randRange(1, 0.51, 0.55)[0]
                        )
                        parameter["position"][0] = (
                            -x_direction
                            * np.sin(y_angle)
                            * (
                                (-body_height / 2 + parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )
                        )
                        parameter["position"][2] = (
                            -x_direction
                            * np.cos(y_angle)
                            * (
                                (-body_height / 2 + parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )
                        )

            if is_prismatic:
                all_direction = ["x", "z"]
                weight_of_direction = [1, 1]
                direction = random.choices(all_direction, weight_of_direction, k=1)[0]

                if direction == "x":
                    parameter["rotation"][1] = 0
                    if position == "out" and body_difference > 0:
                        parameter["position"][0] = 0
                        if rotation_angle == "acute":
                            parameter["position"][2] = -x_direction * (
                                (body_height / 2 - parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )
                        if rotation_angle == "obtuse":
                            parameter["position"][2] = (
                                -x_direction
                                * (-body_height / 2 + parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )

                if direction == "z":
                    parameter["rotation"][1] = 90
                    if position == "out" and body_difference > 0:
                        parameter["position"][2] = 0
                        if rotation_angle == "acute":
                            parameter["position"][0] = -x_direction * (
                                (body_height / 2 - parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )
                        if rotation_angle == "obtuse":
                            parameter["position"][0] = (
                                -x_direction
                                * (-body_height / 2 + parameter["position"][1])
                                * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                            )

            if is_cuboid:
                parameter["rotation"][1] = 90
                if position == "out" and body_difference > 0:
                    parameter["position"][2] = 0
                    if rotation_angle == "acute":
                        parameter["position"][0] = -x_direction * (
                            (body_height / 2 - parameter["position"][1])
                            * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                        )
                    if rotation_angle == "obtuse":
                        parameter["position"][0] = (
                            -x_direction
                            * (-body_height / 2 + parameter["position"][1])
                            * np.abs(np.tan(np.deg2rad(parameter["rotation"][0])))
                        )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

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
        bucket_type = get_bucket_type()
        existing_concept_templates = concept_template_existence(bucket_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, bucket_type)

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
            bucket_type = get_bucket_type()
            existing_concept_templates = concept_template_existence(bucket_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, bucket_type)
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
