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


def get_dishwasher_type():
    dishwasher_type = ["single_layer_dishwasher", "double_layer_dishwasher"]
    weights = [1, 1]
    dishwasher_type = random.choices(dishwasher_type, weights=weights, k=1)[0]
    return dishwasher_type


def concept_template_existence(dishwasher_type):
    if dishwasher_type == "single_layer_dishwasher":
        body_template = ["Cuboidal_Body"]
    elif dishwasher_type == "double_layer_dishwasher":
        body_template = ["Double_Layer_Body"]
    door_template = ["Cuboidal_Door", "Sunken_Door"]
    handle_template = [
        "Cuboidal_Handle",
        "Trifold_Handle",
        "Trifold_Curve_Handle",
        "Curve_Handle",
    ]
    tray_template = ["Drawer_Like_Tray"]
    cover_template = ["Cuboidal_Topcover"]
    leg_template = ["Single_Leg", "Multilevel_Leg"]

    concept_template_variation = {
        "body": {"template": body_template, "necessary": True},
        "door": {"template": door_template, "necessary": True},
        "handle": {"template": handle_template, "necessary": True},
        "tray": {"template": tray_template, "necessary": False},
        "cover": {"template": cover_template, "necessary": False},
        "leg": {"template": leg_template, "necessary": True},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"] == False:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))
        elif part["necessary"] == True:
            templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, dishwasher_type):
    new_concepts = []
    door_type = ""
    rot_orien = ""

    for concept in concepts:
        template = concept["template"]
        parameters = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Cuboidal_Body":
            parameters["size"] = randRange(3, 0.7, 1.3) * parameters["size"]
            parameters["size"][:2] = abs(np.sort(-parameters["size"][:2]))

            parameters["thickness"] = randRange(5, 0.02, 0.07) * parameters["size"][0]
            parameters["thickness"][1] = parameters["thickness"][0]
            parameters["thickness"][3] = parameters["thickness"][2]

            parameters["position"] = np.zeros(3)
            parameters["rotation"] = np.zeros(3)

        elif template == "Double_Layer_Body":
            parameters["size"] = randRange(3, 0.7, 1.3) * parameters["size"]
            parameters["size"][:2] = abs(np.sort(-parameters["size"][:2]))

            parameters["thickness"] = randRange(5, 0.02, 0.07) * parameters["size"][0]
            parameters["thickness"][1] = parameters["thickness"][0]
            parameters["thickness"][3] = parameters["thickness"][2]

            parameters["position"] = np.zeros(3)
            parameters["rotation"] = np.zeros(3)

            parameters["clapboard_size"] = np.array(
                [
                    parameters["size"][0] * random.randint(5, 15) / 100,
                    (parameters["size"][2] - 2 * parameters["thickness"][4])
                    * random.randint(80, 100)
                    / 100,
                    parameters["size"][1] - 2 * parameters["thickness"][2],
                ]
            )
            parameters["clapboard_offset"] = np.array(
                [
                    np.random.randint(-20, 20)
                    / 100
                    * (
                        parameters["size"][1]
                        - 2 * parameters["thickness"][0]
                        - 0.5 * parameters["clapboard_size"][1]
                    )
                ]
            )
            parameters["position"] = np.zeros(3)
            parameters["rotation"] = np.zeros(3)

        elif template == "Cuboidal_Door":
            body_params = new_concepts[0]["parameters"]
            door_type = random.choice(["left", "right", "below"])

            parameters["size"] = np.array(
                [
                    body_params["size"][0],
                    body_params["size"][1] - 1.8 * body_params["thickness"][0],
                    body_params["thickness"][4] * random.randint(50, 70) / 100,
                ]
            )
            rotation = random.randint(0, 90)
            rot_orien = door_type

            if rot_orien == "below":
                parameters["rotation"] = np.array([rotation, 0, 0])
                parameters["position"] = np.array(
                    [
                        0,
                        -0.5
                        * parameters["size"][1]
                        * (1 - np.cos(rotation * np.pi / 180)),
                        0.5 * body_params["size"][2]
                        + 0.5 * parameters["size"][1] * np.sin(rotation * np.pi / 180),
                    ]
                )
            elif rot_orien == "left":
                parameters["rotation"] = np.array([0, -rotation, 0])
                parameters["position"] = np.array(
                    [
                        0.5 * parameters["size"][0] * np.cos(rotation * np.pi / 180)
                        - 0.5 * parameters["size"][0],
                        0,
                        0.5 * body_params["size"][2]
                        + 0.5 * parameters["size"][0] * np.sin(rotation * np.pi / 180),
                    ]
                )
            elif rot_orien == "right":
                parameters["rotation"] = np.array([0, rotation, 0])
                parameters["position"] = np.array(
                    [
                        -0.5 * parameters["size"][0] * np.cos(rotation * np.pi / 180)
                        + 0.5 * parameters["size"][0],
                        0,
                        0.5 * body_params["size"][2]
                        + 0.5 * parameters["size"][0] * np.sin(rotation * np.pi / 180),
                    ]
                )

        elif template == "Sunken_Door":
            body_params = new_concepts[0]["parameters"]
            door_type = random.choice(["left", "right", "below"])

            parameters["size"] = np.array(
                [
                    body_params["size"][0],
                    body_params["size"][1] - 1.8 * body_params["thickness"][0],
                    body_params["thickness"][4] * random.randint(50, 70) / 100,
                ]
            )
            rotation = random.randint(0, 90)
            rot_orien = door_type

            if rot_orien == "below":
                parameters["rotation"] = np.array([rotation, 0, 0])
                parameters["position"] = np.array(
                    [
                        0,
                        -0.5
                        * parameters["size"][1]
                        * (1 - np.cos(rotation * np.pi / 180)),
                        0.5 * body_params["size"][2]
                        + 0.5 * parameters["size"][1] * np.sin(rotation * np.pi / 180),
                    ]
                )
            elif rot_orien == "left":
                parameters["rotation"] = np.array([0, -rotation, 0])
                parameters["position"] = np.array(
                    [
                        0.5 * parameters["size"][0] * np.cos(rotation * np.pi / 180)
                        - 0.5 * parameters["size"][0],
                        0,
                        0.5 * body_params["size"][2]
                        + 0.5 * parameters["size"][0] * np.sin(rotation * np.pi / 180),
                    ]
                )
            elif rot_orien == "right":
                parameters["rotation"] = np.array([0, rotation, 0])
                parameters["position"] = np.array(
                    [
                        -0.5 * parameters["size"][0] * np.cos(rotation * np.pi / 180)
                        + 0.5 * parameters["size"][0],
                        0,
                        0.5 * body_params["size"][2]
                        + 0.5 * parameters["size"][0] * np.sin(rotation * np.pi / 180),
                    ]
                )
            parameters["sunken_offset"] = np.zeros(2)
            parameters["sunken_size"] = randRange(3, 0.6, 0.8) * parameters["size"]
            parameters["sunken_offset"] = np.zeros(2)

        elif template == "Drawer_Like_Tray":
            body_params = new_concepts[0]["parameters"]
            body_type = new_concepts[0]["template"]
            parameters["outer_size"] = np.array(
                [
                    (
                        (
                            body_params["size"][0]
                            - 2 * np.array(body_params["thickness"][0])
                        )
                        * randRange(1, 0.4, 0.6)[0]
                    ),
                    body_params["thickness"][0] * random.randint(80, 120) / 100,
                    (
                        body_params["size"][2]
                        - 2
                        * np.array(body_params["thickness"][2])
                        * randRange(1, 0.4, 0.6)[0]
                    ),
                ]
            )
            parameters["inner_size"] = randRange(3, 0.4, 0.6) * parameters["outer_size"]
            if body_type == "Double_Layer_Body":
                parameters["position"] = np.array(body_params["position"])
                parameters["position"][1] += (
                    1 * body_params["clapboard_size"][0]
                    + body_params["clapboard_offset"][0]
                )
                parameters["rotation"] = np.zeros(3)
            else:
                parameters["position"] = np.zeros(3)
                parameters["rotation"] = np.zeros(3)

        elif template == "Single_Leg":
            body_params = {
                k: np.array(v) for k, v in new_concepts[0]["parameters"].items()
            }
            body_type = new_concepts[0]["template"]

            parameters["size"] = np.array(
                [
                    body_params["size"][0] * random.randint(60, 90) / 100,
                    body_params["size"][1] * random.randint(3, 15) / 100,
                    body_params["size"][2] * random.randint(60, 90) / 100,
                ]
            )

            parameters["position"] = np.array(
                [
                    body_params["position"][0],
                    body_params["position"][2],
                    body_params["position"][1],
                ]
            )
            parameters["position"][1] -= body_params["size"][1] * 0.5
            parameters["rotation"] = np.zeros(3)

        elif template == "Multilevel_Leg":
            body_params = {
                k: np.array(v) for k, v in new_concepts[0]["parameters"].items()
            }
            body_type = new_concepts[0]["template"]

            has_top_part = np.array([np.random.randint(0, 2)])
            parameters["has_top_part"] = has_top_part
            if has_top_part == 1:
                parameters["top_size"] = np.array(
                    [
                        body_params["size"][0] * random.randint(60, 90) / 100,
                        body_params["size"][1] * random.randint(3, 10) / 100,
                        body_params["size"][2] * random.randint(60, 90) / 100,
                    ]
                )
                attached_size = parameters["top_size"]
            else:
                attached_size = body_params["size"]
                attached_size[2] = parameters["top_size"][2]
                parameters["top_size"] = np.zeros(3)

            num_legs = np.array([random.choice([1, 4])])
            parameters["num_legs"] = num_legs
            if num_legs == 1:
                parameters["front_legs_size"] = np.array(
                    [
                        attached_size[0] * random.randint(80, 90) / 100,
                        attached_size[1] * random.randint(10, 15) / 100,
                        attached_size[2] * random.randint(80, 90) / 100,
                    ]
                )
                parameters["rear_legs_size"] = parameters["front_legs_size"]
                parameters["legs_separation"] = np.zeros(
                    parameters["legs_separation"].shape
                )
            else:
                parameters["front_legs_size"] = np.array(
                    [
                        attached_size[0]
                        * random.randint(80, 90)
                        / 100
                        / parameters["num_legs"][0],
                        attached_size[1] * random.randint(3, 15) / 100,
                        attached_size[2]
                        * random.randint(80, 90)
                        / 100
                        / parameters["num_legs"][0],
                    ]
                )
                parameters["rear_legs_size"] = parameters["front_legs_size"]
                parameters["legs_separation"] = (
                    (attached_size - parameters["front_legs_size"])
                    * random.randint(40, 80)
                    / 100
                )
                parameters["legs_separation"][1] = parameters["legs_separation"][0]

            parameters["top_bottom_offset"] = np.zeros(
                parameters["top_bottom_offset"].shape
            )
            parameters["position"] = body_params["position"]
            parameters["position"][1] -= body_params["size"][1] * 0.5
            parameters["rotation"] = np.zeros(3)

        elif template == "Cuboidal_Handle":
            body_params = {
                k: np.array(v) for k, v in new_concepts[0]["parameters"].items()
            }
            body_type = new_concepts[0]["template"]
            door_params = {
                k: np.array(v) for k, v in new_concepts[1]["parameters"].items()
            }
            rotation = np.abs(door_params["rotation"].sum())
            specific_pos = 0
            specific_rot = np.zeros(3)
            if door_type == "left":
                attached_size = door_params["size"][1]
            elif door_type == "right":
                attached_size = door_params["size"][1]
            else:
                attached_size = door_params["size"][0]

            parameters["size"] = np.array(
                [
                    randRange(3, 0.7, 1.3) * parameters["size"],
                    np.repeat(0.7 * np.sort(door_params["size"])[1], 3),
                ]
            ).min(axis=0)
            specific_pos = parameters["size"][1]
            specific_rot[2] += 90

            offset = np.zeros(3)

            if new_concepts[1]["template"] == "Sunken_Door":
                move_ratio = random.randint(43, 46) / 100
            else:
                move_ratio = random.randint(30, 46) / 100

            if rot_orien == "below":
                parameters["position"] = door_params["position"]
                offset[0] = random.randint(-10, 10) / 100 * door_params["size"][0]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + move_ratio * door_params["size"][1] * np.sin(rotation * np.pi / 180)
                offset[1] += -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + move_ratio * door_params["size"][1] * np.cos(rotation * np.pi / 180)

                specific_rot[2] += 90

            elif rot_orien == "left":
                parameters["position"] = door_params["position"]
                offset[1] = -random.randint(-10, 10) / 100 * door_params["size"][1]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.sin(
                    rotation * np.pi / 180
                )
                offset[0] += -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.cos(
                    rotation * np.pi / 180
                )

            elif rot_orien == "right":
                parameters["position"] = door_params["position"]
                offset[1] = -random.randint(-10, 10) / 100 * door_params["size"][1]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.sin(
                    rotation * np.pi / 180
                )
                offset[0] -= -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.cos(
                    rotation * np.pi / 180
                )

            parameters["position"] = door_params["position"] + offset
            parameters["rotation"] = door_params["rotation"] + specific_rot

        elif template == "Trifold_Handle":
            body_params = {
                k: np.array(v) for k, v in new_concepts[0]["parameters"].items()
            }
            body_type = new_concepts[0]["template"]
            door_params = {
                k: np.array(v) for k, v in new_concepts[1]["parameters"].items()
            }
            rotation = np.abs(door_params["rotation"].sum())
            specific_pos = 0
            specific_rot = np.zeros(3)
            if door_type == "left":
                attached_size = door_params["size"][1]
            elif door_type == "right":
                attached_size = door_params["size"][1]
            else:
                attached_size = door_params["size"][0]

            ampli_ratio = random.choice(
                [random.randint(60, 100) / 100, random.randint(10, 50) / 100]
            )

            parameters["mounting_size"] = np.array(
                [
                    randRange(3, 0.8, 1.2) * parameters["mounting_size"] * ampli_ratio,
                    np.ones(3) * 0.05,
                ]
            ).max(axis=0)
            parameters["grip_size"] = np.array(
                [
                    random.randint(70, 100) / 100 * parameters["mounting_size"][2],
                    min(
                        randRange(1, 0.7, 1.3)[0] * parameters["grip_size"][1],
                        0.7 * attached_size,
                    ),
                    random.randint(70, 100) / 100 * parameters["mounting_size"][0],
                ]
            )
            parameters["mounting_seperation"] = np.array(
                [parameters["grip_size"][1] * random.randint(50, 80) / 100]
            )

            offset = np.zeros(3)

            if new_concepts[1]["template"] == "Sunken_Door":
                move_ratio = random.randint(43, 46) / 100
            else:
                move_ratio = random.randint(30, 46) / 100

            if rot_orien == "below":
                parameters["position"] = door_params["position"]
                offset[0] = random.randint(-10, 10) / 100 * door_params["size"][0]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + move_ratio * door_params["size"][1] * np.sin(rotation * np.pi / 180)
                offset[1] += -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + move_ratio * door_params["size"][1] * np.cos(rotation * np.pi / 180)

                specific_rot[2] += 90

            elif rot_orien == "left":
                parameters["position"] = door_params["position"]
                offset[1] = -random.randint(-10, 10) / 100 * door_params["size"][1]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.sin(
                    rotation * np.pi / 180
                )
                offset[0] += -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.cos(
                    rotation * np.pi / 180
                )

            elif rot_orien == "right":
                parameters["position"] = door_params["position"]
                offset[1] = -random.randint(-10, 10) / 100 * door_params["size"][1]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.sin(
                    rotation * np.pi / 180
                )
                offset[0] -= -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.cos(
                    rotation * np.pi / 180
                )

            parameters["position"] = door_params["position"] + offset
            parameters["rotation"] = door_params["rotation"] + specific_rot

        elif template == "Trifold_Curve_Handle":
            body_params = {
                k: np.array(v) for k, v in new_concepts[0]["parameters"].items()
            }
            body_type = new_concepts[0]["template"]
            door_params = {
                k: np.array(v) for k, v in new_concepts[1]["parameters"].items()
            }
            rotation = np.abs(door_params["rotation"].sum())
            specific_pos = 0
            specific_rot = np.zeros(3)

            if door_type == "left":
                attached_size = np.sort(door_params["size"])[1]
            elif door_type == "right":
                attached_size = np.sort(door_params["size"])[1]
            else:
                attached_size = door_params["size"].max()
            parameters["curve_exist_angle"] *= random.randint(80, 120) / 100
            angle = parameters["curve_exist_angle"][0] * np.pi / 180
            parameters["curve_size"][2] *= random.randint(80, 120) / 100

            chord = random.randint(50, 90) / 100 * attached_size
            parameters["curve_size"][0] = chord / 2 / np.sin(angle / 2) * angle
            parameters["curve_size"][1] = (
                parameters["curve_size"][0] / angle - parameters["curve_size"][2]
            ) * angle
            parameters["mounting_size"] *= random.randint(80, 120) / 100
            parameters["mounting_seperation"] = np.array(
                [parameters["curve_size"][1] / angle * np.sin(angle / 2)]
            )

            specific_pos = parameters["mounting_size"][2]

            offset = np.zeros(3)

            if new_concepts[1]["template"] == "Sunken_Door":
                move_ratio = random.randint(43, 46) / 100
            else:
                move_ratio = random.randint(30, 46) / 100

            if rot_orien == "below":
                parameters["position"] = door_params["position"]
                offset[0] = random.randint(-10, 10) / 100 * door_params["size"][0]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + move_ratio * door_params["size"][1] * np.sin(rotation * np.pi / 180)
                offset[1] += -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + move_ratio * door_params["size"][1] * np.cos(rotation * np.pi / 180)

                specific_rot[2] += 90

            elif rot_orien == "left":
                parameters["position"] = door_params["position"]
                offset[1] = -random.randint(-10, 10) / 100 * door_params["size"][1]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.sin(
                    rotation * np.pi / 180
                )
                offset[0] += -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.cos(
                    rotation * np.pi / 180
                )

            elif rot_orien == "right":
                parameters["position"] = door_params["position"]
                offset[1] = -random.randint(-10, 10) / 100 * door_params["size"][1]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.sin(
                    rotation * np.pi / 180
                )
                offset[0] -= -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.cos(
                    rotation * np.pi / 180
                )

            parameters["position"] = door_params["position"] + offset
            parameters["rotation"] = door_params["rotation"] + specific_rot

        elif template == "Curve_Handle":
            body_params = {
                k: np.array(v) for k, v in new_concepts[0]["parameters"].items()
            }
            body_type = new_concepts[0]["template"]
            door_params = {
                k: np.array(v) for k, v in new_concepts[1]["parameters"].items()
            }
            rotation = np.abs(door_params["rotation"].sum())
            specific_pos = 0
            specific_rot = np.zeros(3)

            if door_type == "left":
                attached_size = np.sort(door_params["size"])[1]
            elif door_type == "right":
                attached_size = np.sort(door_params["size"])[1]
            else:
                attached_size = door_params["size"].max()
            parameters["curve_exist_angle"] *= random.randint(80, 120) / 100
            angle = parameters["curve_exist_angle"][0] * np.pi / 180
            parameters["curve_size"][2] *= random.randint(80, 120) / 100

            chord = random.randint(50, 90) / 100 * attached_size
            parameters["curve_size"][0] = chord / 2 / np.sin(angle / 2) * angle
            parameters["curve_size"][1] = (
                parameters["curve_size"][0] / angle - parameters["curve_size"][2]
            ) * angle

            specific_pos = parameters["curve_size"][2]

            offset = np.zeros(3)

            if new_concepts[1]["template"] == "Sunken_Door":
                move_ratio = random.randint(43, 46) / 100
            else:
                move_ratio = random.randint(30, 46) / 100

            if rot_orien == "below":
                parameters["position"] = door_params["position"]
                offset[0] = random.randint(-10, 10) / 100 * door_params["size"][0]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + move_ratio * door_params["size"][1] * np.sin(rotation * np.pi / 180)
                offset[1] += -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + move_ratio * door_params["size"][1] * np.cos(rotation * np.pi / 180)

                specific_rot[2] += 90

            elif rot_orien == "left":
                parameters["position"] = door_params["position"]
                offset[1] = -random.randint(-10, 10) / 100 * door_params["size"][1]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.sin(
                    rotation * np.pi / 180
                )
                offset[0] += -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.cos(
                    rotation * np.pi / 180
                )

            elif rot_orien == "right":
                parameters["position"] = door_params["position"]
                offset[1] = -random.randint(-10, 10) / 100 * door_params["size"][1]
                offset[2] += 0.5 * (specific_pos + door_params["size"][2]) * np.cos(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.sin(
                    rotation * np.pi / 180
                )
                offset[0] -= -0.5 * (specific_pos + door_params["size"][2]) * np.sin(
                    rotation * np.pi / 180
                ) + (move_ratio) * door_params["size"][0] * np.cos(
                    rotation * np.pi / 180
                )

            parameters["position"] = door_params["position"] + offset
            parameters["rotation"] = door_params["rotation"] + specific_rot

        elif template == "Cuboidal_Topcover":
            body_params = {
                k: np.array(v) for k, v in new_concepts[0]["parameters"].items()
            }
            parameters["size"] = np.array(
                [randRange(3, 0.8, 1.2) * parameters["size"], 0.8 * body_params["size"]]
            ).min(axis=0)
            parameters["position"] = np.array(
                [
                    body_params["position"][0],
                    body_params["position"][2],
                    body_params["position"][1],
                ]
            )
            parameters["position"][1] += body_params["size"][1] * 0.5
            parameters["rotation"] = np.zeros(3)

        new_concepts.append(
            {
                "template": template,
                "parameters": {k: v.tolist() for k, v in parameters.items()},
            }
        )

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
        dishwasher_type = get_dishwasher_type()
        existing_concept_templates = concept_template_existence(dishwasher_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, dishwasher_type)

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
            dishwasher_type = get_dishwasher_type()
            existing_concept_templates = concept_template_existence(dishwasher_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, dishwasher_type)
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
