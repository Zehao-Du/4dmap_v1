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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", default=False, action="store_true")
    parser.add_argument(
        "--gen_num", default=10, type=int, help="number of objects to generate"
    )
    args = parser.parse_args()
    return args


def get_window_type():
    window_type = ["vertical", "horizontal"]
    weights = [1, 1]
    window_type = random.choices(window_type, weights=weights, k=1)[0]
    return "horizontal"


def concept_template_existence(window_type):
    if window_type == "vertical":
        concept_template_variation = {
            "window": {"template": ["VerticalSlid_Window"], "necessary": True},
            "frame": {"template": ["Standard_Windowframe"], "necessary": True},
            "handle": {"template": ["Cuboidal_Handle"], "necessary": True},
        }
    elif window_type == "horizontal":
        concept_template_variation = {
            "window": {
                "template": ["Symmetrical_Window", "Asymmetrical_Window"],
                "necessary": True,
            },
            "frame": {"template": ["Standard_Windowframe"], "necessary": True},
            "handle": {
                "template": ["Cuboidal_Handle", "Arched_Handle", "LShaped_Handle"],
                "necessary": True,
            },
        }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, window_type):
    new_concepts = []

    frame_x: float = 0
    frame_y: float = 0
    window_symmetric_type: int = 0
    num_of_window: int = 0
    offset_x = None
    offset_y = None
    window_x: float = 0
    window_y: float = 0
    window_z: float = 0
    Frame_xy: float = 0

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Standard_Windowframe":
            if window_type == "vertical":
                parameter["inner_size"][0] = window_x
                parameter["inner_size"][1] = num_of_window * window_y
            elif window_type == "horizontal":
                parameter["inner_size"][0] = num_of_window * window_x
                parameter["inner_size"][1] = window_y

            Frame_xy = randRange(1, 0.03, 0.06)[0]

            parameter["size"][0] = parameter["inner_size"][0] + 2 * Frame_xy
            parameter["size"][1] = parameter["inner_size"][1] + 2 * Frame_xy
            parameter["size"][2] = window_z * randRange(1, 1, 1.2)[0]

            parameter["inner_outer_offset"] = np.array([0, 0])
            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Symmetrical_Window":
            parameter["number_of_window"][0] = random.choice(range(1, 7))
            num_of_window = parameter["number_of_window"][0]
            window_symmetric_type = 0

            parameter["size_0"] = np.array(
                [randRange(1, 0.6, 0.9)[0], randRange(1, 0.019, 0.06)[0]]
            )
            if num_of_window == 6:
                parameter["size_0"][0] *= 0.78
            if num_of_window <= 2:
                parameter["size_0"][0] *= 1.5

            parameter["size_1"] = parameter["size_0"]
            parameter["size_2"] = parameter["size_0"]

            window_x = parameter["size_0"][0]
            window_y = randRange(1, 1.44, 1.92)[0]
            if (num_of_window % 2) == 0:
                window_z = parameter["size_0"][1] * 0.5 * num_of_window
            else:
                window_z = parameter["size_0"][1] * 0.5 * (num_of_window + 1)

            frame_x = randRange(1, 0.048, 0.084)[0]
            frame_y = frame_x * randRange(1, 0.8, 1)[0]
            parameter["glass_size_0"] = np.array(
                [
                    window_x - 2 * frame_y,
                    window_y - 2 * frame_x,
                    randRange(1, 0.006, 0.012)[0],
                ]
            )
            parameter["glass_size_1"] = parameter["glass_size_0"]
            parameter["glass_size_2"] = parameter["glass_size_0"]

            parameter["outside_frame_inner_size"] = np.array(
                [window_x * num_of_window, window_y]
            )

            parameter["glass_offset_0"] = np.array([0, 0, 0])
            parameter["glass_offset_1"] = np.array([0, 0, 0])
            parameter["glass_offset_2"] = np.array([0, 0, 0])
            if random.random() < 2 / 3:
                if num_of_window == 1:
                    parameter["offset_x"][0] = 0
                elif num_of_window == 2:
                    parameter["offset_x"][0] = -0.5 * window_x
                    parameter["offset_x"][1] = 0.5 * window_x
                elif num_of_window == 3:
                    parameter["offset_x"][0] = randRange(1, -window_x, -0.7 * window_x)[
                        0
                    ]
                    parameter["offset_x"][1] = randRange(
                        1, -0.2 * window_x, 0.2 * window_x
                    )[0]
                    parameter["offset_x"][2] = randRange(1, 0.7 * window_x, window_x)[0]
                elif num_of_window == 4:
                    parameter["offset_x"][0] = randRange(
                        1, -1.5 * window_x, -1.2 * window_x
                    )[0]
                    parameter["offset_x"][1] = randRange(
                        1, -1.2 * window_x, -0.5 * window_x
                    )[0]
                    parameter["offset_x"][2] = randRange(
                        1, 0.5 * window_x, 1.2 * window_x
                    )[0]
                    parameter["offset_x"][3] = randRange(
                        1, 1.2 * window_x, 1.5 * window_x
                    )[0]
                elif num_of_window == 5:
                    parameter["offset_x"][0] = randRange(
                        1, -2 * window_x, -1.6 * window_x
                    )[0]
                    parameter["offset_x"][1] = randRange(
                        1, -1.3 * window_x, -0.5 * window_x
                    )[0]
                    parameter["offset_x"][2] = 0
                    parameter["offset_x"][3] = randRange(
                        1, 0.5 * window_x, 1.3 * window_x
                    )[0]
                    parameter["offset_x"][4] = randRange(
                        1, 1.6 * window_x, 2 * window_x
                    )[0]
                elif num_of_window == 6:
                    parameter["offset_x"][0] = randRange(
                        1, -2.5 * window_x, -2.3 * window_x
                    )[0]
                    parameter["offset_x"][1] = randRange(
                        1, -2 * window_x, -1.5 * window_x
                    )[0]
                    parameter["offset_x"][2] = randRange(1, -window_x, -0.5 * window_x)[
                        0
                    ]
                    parameter["offset_x"][3] = randRange(1, 0.5 * window_x, window_x)[0]
                    parameter["offset_x"][4] = randRange(
                        1, 1.5 * window_x, 2 * window_x
                    )[0]
                    parameter["offset_x"][5] = randRange(
                        1, 2.3 * window_x, 2.5 * window_x
                    )[0]
            else:
                if num_of_window == 1:
                    parameter["offset_x"][0] = 0
                elif num_of_window == 2:
                    parameter["offset_x"][0] = -0.5 * window_x
                    parameter["offset_x"][1] = 0.5 * window_x
                elif num_of_window == 3:
                    parameter["offset_x"][0] = -window_x
                    parameter["offset_x"][1] = 0
                    parameter["offset_x"][2] = window_x
                elif num_of_window == 4:
                    parameter["offset_x"][0] = -1.5 * window_x
                    parameter["offset_x"][1] = -0.5 * window_x
                    parameter["offset_x"][2] = 0.5 * window_x
                    parameter["offset_x"][3] = 1.5 * window_x
                elif num_of_window == 5:
                    parameter["offset_x"][0] = -2 * window_x
                    parameter["offset_x"][1] = -window_x
                    parameter["offset_x"][2] = 0
                    parameter["offset_x"][3] = window_x
                    parameter["offset_x"][4] = 2 * window_x
                elif num_of_window == 6:
                    parameter["offset_x"][0] = -2.5 * window_x
                    parameter["offset_x"][1] = -1.5 * window_x
                    parameter["offset_x"][2] = -0.5 * window_x
                    parameter["offset_x"][3] = 0.5 * window_x
                    parameter["offset_x"][4] = 1.5 * window_x
                    parameter["offset_x"][5] = 2.5 * window_x

            offset_x = parameter["offset_x"]

            parameter["symmetryOrNot"][0] = 1
            parameter["offset_z"][0] = 0.5 * parameter["size_0"][1] - window_z
            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Asymmetrical_Window":
            parameter["number_of_window"][0] = random.choice(range(2, 5))
            num_of_window = parameter["number_of_window"][0]
            window_symmetric_type = 1
            parameter["size_0"] = np.array(
                [randRange(1, 0.9, 0.9)[0], randRange(1, 0.019, 0.06)[0]]
            )
            if num_of_window == 6:
                parameter["size_0"][0] *= 0.78
            if num_of_window == 1:
                parameter["size_0"][0] *= 1.5
            parameter["size_1"] = parameter["size_0"]
            parameter["size_2"] = parameter["size_0"]
            parameter["size_3"] = parameter["size_0"]

            window_x = parameter["size_0"][0]
            window_y = randRange(1, 1.44, 1.92)[0]
            window_z = parameter["size_0"][1] * num_of_window

            parameter["outside_frame_inner_size"] = np.array(
                [window_x * num_of_window, window_y]
            )

            frame_x = randRange(1, 0.048, 0.084)[0]
            frame_y = frame_x * randRange(1, 0.8, 1)[0]
            parameter["glass_size_0"] = np.array(
                [
                    window_x - 2 * frame_y,
                    window_y - 2 * frame_x,
                    randRange(1, 0.006, 0.012)[0],
                ]
            )
            parameter["glass_size_1"] = parameter["glass_size_0"]
            parameter["glass_size_2"] = parameter["glass_size_0"]
            parameter["glass_size_3"] = parameter["glass_size_0"]

            parameter["glass_offset_0"] = np.array([0, 0, 0])
            parameter["glass_offset_1"] = np.array([0, 0, 0])
            parameter["glass_offset_2"] = np.array([0, 0, 0])
            parameter["glass_offset_3"] = np.array([0, 0, 0])

            if random.random() < 2 / 3:
                if num_of_window == 1:
                    parameter["offset_x"][0] = 0
                elif num_of_window == 2:
                    parameter["offset_x"][0] = randRange(
                        1, -0.5 * window_x, -0.2 * window_x
                    )[0]
                    parameter["offset_x"][1] = randRange(
                        1, 0.2 * window_x, 0.5 * window_x
                    )[0]
                elif num_of_window == 3:
                    parameter["offset_x"][0] = randRange(1, -window_x, -0.7 * window_x)[
                        0
                    ]
                    parameter["offset_x"][1] = randRange(
                        1, -0.2 * window_x, 0.2 * window_x
                    )[0]
                    parameter["offset_x"][2] = randRange(1, 0.7 * window_x, window_x)[0]
                elif num_of_window == 4:
                    parameter["offset_x"][0] = randRange(
                        1, -1.5 * window_x, -1.2 * window_x
                    )[0]
                    parameter["offset_x"][1] = randRange(
                        1, -1.2 * window_x, -0.5 * window_x
                    )[0]
                    parameter["offset_x"][2] = randRange(
                        1, 0.5 * window_x, 1.2 * window_x
                    )[0]
                    parameter["offset_x"][3] = randRange(
                        1, 1.2 * window_x, 1.5 * window_x
                    )[0]
                elif num_of_window == 5:
                    parameter["offset_x"][0] = randRange(
                        1, -2 * window_x, -1.6 * window_x
                    )[0]
                    parameter["offset_x"][1] = randRange(
                        1, -1.3 * window_x, -0.5 * window_x
                    )[0]
                    parameter["offset_x"][2] = 0
                    parameter["offset_x"][3] = randRange(
                        1, 0.5 * window_x, 1.3 * window_x
                    )[0]
                    parameter["offset_x"][4] = randRange(
                        1, 1.6 * window_x, 2 * window_x
                    )[0]
                elif num_of_window == 6:
                    parameter["offset_x"][0] = randRange(
                        1, -2.5 * window_x, -2.3 * window_x
                    )[0]
                    parameter["offset_x"][1] = randRange(
                        1, -2 * window_x, -1.5 * window_x
                    )[0]
                    parameter["offset_x"][2] = randRange(1, -window_x, -0.5 * window_x)[
                        0
                    ]
                    parameter["offset_x"][3] = randRange(1, 0.5 * window_x, window_x)[0]
                    parameter["offset_x"][4] = randRange(
                        1, 1.5 * window_x, 2 * window_x
                    )[0]
                    parameter["offset_x"][5] = randRange(
                        1, 2.3 * window_x, 2.5 * window_x
                    )[0]
            else:
                if num_of_window == 1:
                    parameter["offset_x"][0] = 0
                elif num_of_window == 2:
                    parameter["offset_x"][0] = -0.5 * window_x
                    parameter["offset_x"][1] = 0.5 * window_x
                elif num_of_window == 3:
                    parameter["offset_x"][0] = -window_x
                    parameter["offset_x"][1] = 0
                    parameter["offset_x"][2] = window_x
                elif num_of_window == 4:
                    parameter["offset_x"][0] = -1.5 * window_x
                    parameter["offset_x"][1] = -0.5 * window_x
                    parameter["offset_x"][2] = 0.5 * window_x
                    parameter["offset_x"][3] = 1.5 * window_x
                elif num_of_window == 5:
                    parameter["offset_x"][0] = -2 * window_x
                    parameter["offset_x"][1] = -window_x
                    parameter["offset_x"][2] = 0
                    parameter["offset_x"][3] = window_x
                    parameter["offset_x"][4] = 2 * window_x
                elif num_of_window == 6:
                    parameter["offset_x"][0] = -2.5 * window_x
                    parameter["offset_x"][1] = -1.5 * window_x
                    parameter["offset_x"][2] = -0.5 * window_x
                    parameter["offset_x"][3] = 0.5 * window_x
                    parameter["offset_x"][4] = 1.5 * window_x
                    parameter["offset_x"][5] = 2.5 * window_x

            offset_x = parameter["offset_x"]

            parameter["offset_z"][0] = (
                (1 - num_of_window) * parameter["size_0"][1] * 0.5
            )
            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "VerticalSlid_Window":
            window_symmetric_type = 2
            parameter["number_of_window"][0] = 2
            num_of_window = parameter["number_of_window"][0]
            parameter["size_0"] = np.array(
                [randRange(1, 0.72, 0.9)[0], randRange(1, 0.019, 0.06)[0]]
            )
            parameter["size_1"] = parameter["size_0"]
            window_x = randRange(1, 1.5, 1.92)[0]
            window_y = parameter["size_0"][0]
            window_z = parameter["size_0"][1]

            frame_x = randRange(1, 0.048, 0.084)[0]
            frame_y = frame_x * randRange(1, 0.8, 1)[0]
            parameter["glass_size_0"] = np.array(
                [
                    window_x - 2 * frame_y,
                    window_y - 2 * frame_x,
                    randRange(1, 0.006, 0.012)[0],
                ]
            )
            parameter["glass_size_1"] = parameter["glass_size_0"]
            parameter["outside_frame_inner_size"] = np.array(
                [window_x, window_y * num_of_window]
            )

            parameter["outside_frame_inner_outer_offset"] = np.array([0, 0])
            parameter["glass_offset_0"] = np.array([0, 0, 0])
            parameter["glass_offset_1"] = np.array([0, 0, 0])
            if random.choice([0, 0, 1]):
                parameter["offset_y"][0] = 0.5 * window_y
                parameter["offset_y"][1] = -0.5 * window_y
            else:
                parameter["offset_y"][0] = randRange(
                    1, 0.1 * window_y, 0.35 * window_y
                )[0]
                parameter["offset_y"][1] = randRange(
                    1, -0.35 * window_y, -0.1 * window_y
                )[0]

            offset_y = parameter["offset_y"]
            parameter["offset_z"][0] = 0
            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cuboidal_Handle":
            parameter["window_type"] = window_symmetric_type

            parameter["size"] = np.array(
                [
                    randRange(1, 0.03, 0.04)[0],
                    randRange(1, 0.15, 0.2)[0],
                    randRange(1, 0.03, 0.04)[0],
                ]
            )
            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            if window_type == "horizontal":
                if num_of_window == 1:
                    parameter["num_of_handle"] = np.array([1, 0])
                    parameter["offset_x"] = np.array(
                        [0.5 * window_x - 0.5 * frame_y, 0, 0, 0]
                    )
                    parameter["handle_z_position"] = np.array([0, 0])
                elif num_of_window == 2:
                    parameter["num_of_handle"] = np.array([1, 1])
                    parameter["offset_x"] = np.array(
                        [
                            offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                            0,
                            offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                            0,
                        ]
                    )
                    parameter["handle_z_position"] = np.array([0, 0])
                    if random.choice([0, 1]):
                        parameter["num_of_handle"] = np.array([2, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                offset_x[1] - 0.5 * window_x + 0.5 * frame_y,
                                0,
                                0,
                            ]
                        )
                        parameter["handle_z_position"] = np.array([0, 1])
                    if window_symmetric_type == 0:
                        parameter["handle_z_position"] = np.array([0, 0])
                        parameter["num_of_handle"] = np.array([2, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                                0,
                                offset_x[1] + 0.5 * window_x - 0.5 * frame_y,
                                0,
                            ]
                        )
                elif num_of_window == 3:
                    parameter["num_of_handle"] = np.array([2, 1])
                    parameter["offset_x"] = np.array(
                        [
                            offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                            offset_x[2] + 0.5 * window_x - 0.5 * frame_y,
                            offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                            0,
                        ]
                    )
                    parameter["handle_z_position"] = np.array([0, 2])
                    if window_symmetric_type == 0:
                        parameter["num_of_handle"] = np.array([2, 0])
                        parameter["handle_z_position"] = np.array([0, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                offset_x[2] + 0.5 * window_x - 0.5 * frame_y,
                                0,
                                0,
                            ]
                        )
                elif num_of_window >= 4:
                    parameter["num_of_handle"] = np.array([2, 2])
                    parameter["offset_x"] = np.array(
                        [
                            offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                            offset_x[3] - 0.5 * window_x + 0.5 * frame_y,
                            offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                            offset_x[3] + 0.5 * window_x - 0.5 * frame_y,
                        ]
                    )
                    parameter["handle_z_position"] = np.array([0, 3])
                    if window_symmetric_type == 0:
                        if num_of_window == 4:
                            parameter["num_of_handle"] = np.array([2, 0])
                        parameter["handle_z_position"] = np.array([0, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                offset_x[3] + 0.5 * window_x - 0.5 * frame_y,
                                offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                                offset_x[3] - 0.5 * window_x + 0.5 * frame_y,
                            ]
                        )
                        if num_of_window == 5:
                            parameter["handle_z_position"] = np.array([-1, -1])
                            parameter["offset_x"] = np.array(
                                [
                                    offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                    offset_x[4] + 0.5 * window_x - 0.5 * frame_y,
                                    offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                                    offset_x[4] - 0.5 * window_x + 0.5 * frame_y,
                                ]
                            )
                        elif num_of_window == 6:
                            parameter["handle_z_position"] = np.array([-1, 0])
                parameter["position"] = np.array(
                    [0, 0, new_concepts[0]["parameters"]["offset_z"][0]]
                )
                parameter["windows_size"] = {
                    "size_0": new_concepts[0]["parameters"]["size_0"],
                    "size_1": new_concepts[0]["parameters"]["size_1"],
                    "size_2": new_concepts[0]["parameters"]["size_2"],
                    "size_3": new_concepts[0]["parameters"]["size_2"],
                }
            else:
                parameter["num_of_handle"] = np.array([1, 0])
                parameter["offset_x"] = np.array([0, 0, 0, 0])
                parameter["size"] = np.array(
                    [
                        randRange(1, 0.15, 0.2)[0],
                        randRange(1, 0.03, 0.04)[0],
                        randRange(1, 0.03, 0.04)[0],
                    ]
                )
                parameter["position"] = np.array(
                    [0, offset_y[1] + 0.4 * window_y - parameter["size"][1], window_z]
                )
                parameter["handle_z_position"] = np.array([1, 1])

                parameter["windows_size"] = {
                    "size_0": [0, 0],
                    "size_1": [0, 0],
                    "size_2": [0, 0],
                    "size_3": [0, 0],
                }

            concept["parameters"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in parameter.items()
            }

            new_concepts.append(concept)

        elif template == "Arched_Handle":
            parameter["window_type"] = window_symmetric_type
            parameter["bottom_size"][0] = randRange(1, 0.015, 0.025)[0]
            parameter["bottom_size"][1] = parameter["bottom_size"][0]
            parameter["bottom_size"][2] = randRange(1, 0.055, 0.065)[0]
            parameter["seperation"][0] = window_y * randRange(1, 0.15, 0.2)[0]
            parameter["outer_size"][0] = (
                parameter["seperation"][0] * randRange(1, 1, 1.2)[0]
            )
            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            if window_type == "horizontal":
                if num_of_window == 1:
                    parameter["num_of_handle"] = np.array([1, 0])
                    parameter["offset_x"] = np.array(
                        [0.5 * window_x - 0.5 * frame_y, 0, 0, 0]
                    )
                    parameter["handle_z_position"] = np.array([0, 0])
                elif num_of_window == 2:
                    parameter["num_of_handle"] = np.array([1, 1])
                    parameter["offset_x"] = np.array(
                        [
                            offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                            0,
                            offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                            0,
                        ]
                    )
                    parameter["handle_z_position"] = np.array([0, 0])
                    if random.choice([0, 1]):
                        parameter["num_of_handle"] = np.array([2, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                offset_x[1] - 0.5 * window_x + 0.5 * frame_y,
                                0,
                                0,
                            ]
                        )
                        parameter["handle_z_position"] = np.array([0, 1])
                    if window_symmetric_type == 0:
                        parameter["handle_z_position"] = np.array([0, 0])
                        parameter["num_of_handle"] = np.array([2, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                                0,
                                offset_x[1] + 0.5 * window_x - 0.5 * frame_y,
                                0,
                            ]
                        )
                elif num_of_window == 3:
                    parameter["num_of_handle"] = np.array([2, 1])
                    parameter["offset_x"] = np.array(
                        [
                            offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                            offset_x[2] + 0.5 * window_x - 0.5 * frame_y,
                            offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                            0,
                        ]
                    )
                    parameter["handle_z_position"] = np.array([0, 2])
                    if window_symmetric_type == 0:
                        parameter["num_of_handle"] = np.array([2, 0])
                        parameter["handle_z_position"] = np.array([0, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                offset_x[2] + 0.5 * window_x - 0.5 * frame_y,
                                0,
                                0,
                            ]
                        )
                elif num_of_window >= 4:
                    parameter["num_of_handle"] = np.array([2, 2])
                    parameter["offset_x"] = np.array(
                        [
                            offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                            offset_x[3] - 0.5 * window_x + 0.5 * frame_y,
                            offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                            offset_x[3] + 0.5 * window_x - 0.5 * frame_y,
                        ]
                    )
                    parameter["handle_z_position"] = np.array([0, 3])
                    if window_symmetric_type == 0:
                        parameter["handle_z_position"] = np.array([0, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                offset_x[3] + 0.5 * window_x - 0.5 * frame_y,
                                offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                                offset_x[3] - 0.5 * window_x + 0.5 * frame_y,
                            ]
                        )
                        if num_of_window == 5:
                            parameter["handle_z_position"] = np.array([-1, -1])
                            parameter["offset_x"] = np.array(
                                [
                                    offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                    offset_x[4] + 0.5 * window_x - 0.5 * frame_y,
                                    offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                                    offset_x[4] - 0.5 * window_x + 0.5 * frame_y,
                                ]
                            )
                        elif num_of_window == 6:
                            parameter["handle_z_position"] = np.array([-1, 0])
                parameter["position"] = np.array(
                    [0, 0, new_concepts[0]["parameters"]["offset_z"][0]]
                )
            else:
                parameter["num_of_handle"] = np.array([1, 0])
                parameter["offset_x"] = np.array([0, 0, 0, 0])

                parameter["position"] = np.array([0, offset_y[1], window_z])
                parameter["handle_z_position"] = np.array([1, 1])

            parameter["thinner_handle"][0] = 0
            parameter["windows_size"] = {
                "size_0": new_concepts[0]["parameters"]["size_0"],
                "size_1": new_concepts[0]["parameters"]["size_1"],
                "size_2": new_concepts[0]["parameters"]["size_2"],
                "size_3": new_concepts[0]["parameters"]["size_2"],
            }

            concept["parameters"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in parameter.items()
            }

            new_concepts.append(concept)

        elif template == "LShaped_Handle":
            parameter["window_type"] = window_symmetric_type
            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            if window_type == "horizontal":
                if num_of_window == 1:
                    parameter["num_of_handle"] = np.array([1, 0])
                    parameter["offset_x"] = np.array(
                        [0.5 * window_x - 0.5 * frame_y, 0, 0, 0]
                    )
                    parameter["handle_z_position"] = np.array([0, 0])
                elif num_of_window == 2:
                    parameter["num_of_handle"] = np.array([1, 1])
                    parameter["offset_x"] = np.array(
                        [
                            offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                            0,
                            offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                            0,
                        ]
                    )
                    parameter["handle_z_position"] = np.array([0, 0])
                    if random.choice([0, 1]):
                        parameter["num_of_handle"] = np.array([2, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                offset_x[1] - 0.5 * window_x + 0.5 * frame_y,
                                0,
                                0,
                            ]
                        )
                        parameter["handle_z_position"] = np.array([0, 1])
                    if window_symmetric_type == 0:
                        parameter["handle_z_position"] = np.array([0, 0])
                        parameter["num_of_handle"] = np.array([2, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                                0,
                                offset_x[1] + 0.5 * window_x - 0.5 * frame_y,
                                0,
                            ]
                        )
                elif num_of_window == 3:
                    parameter["num_of_handle"] = np.array([2, 1])
                    parameter["offset_x"] = np.array(
                        [
                            offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                            offset_x[2] + 0.5 * window_x - 0.5 * frame_y,
                            offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                            0,
                        ]
                    )
                    parameter["handle_z_position"] = np.array([0, 2])
                    if window_symmetric_type == 0:
                        parameter["num_of_handle"] = np.array([2, 0])
                        parameter["handle_z_position"] = np.array([0, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                offset_x[2] + 0.5 * window_x - 0.5 * frame_y,
                                0,
                                0,
                            ]
                        )
                elif num_of_window >= 4:
                    parameter["num_of_handle"] = np.array([2, 2])
                    parameter["offset_x"] = np.array(
                        [
                            offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                            offset_x[3] - 0.5 * window_x + 0.5 * frame_y,
                            offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                            offset_x[3] + 0.5 * window_x - 0.5 * frame_y,
                        ]
                    )
                    parameter["handle_z_position"] = np.array([0, 3])
                    if window_symmetric_type == 0:
                        if num_of_window == 4:
                            parameter["num_of_handle"] = np.array([2, 0])
                        parameter["handle_z_position"] = np.array([0, 0])
                        parameter["offset_x"] = np.array(
                            [
                                offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                offset_x[3] + 0.5 * window_x - 0.5 * frame_y,
                                offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                                offset_x[3] - 0.5 * window_x + 0.5 * frame_y,
                            ]
                        )
                        if num_of_window == 5:
                            parameter["handle_z_position"] = np.array([-1, -1])
                            parameter["offset_x"] = np.array(
                                [
                                    offset_x[0] - 0.5 * window_x + 0.5 * frame_y,
                                    offset_x[4] + 0.5 * window_x - 0.5 * frame_y,
                                    offset_x[0] + 0.5 * window_x - 0.5 * frame_y,
                                    offset_x[4] - 0.5 * window_x + 0.5 * frame_y,
                                ]
                            )
                        elif num_of_window == 6:
                            parameter["handle_z_position"] = np.array([-1, 0])
                parameter["position"] = np.array(
                    [0, 0, new_concepts[0]["parameters"]["offset_z"][0]]
                )
            else:
                parameter["num_of_handle"] = np.array([1, 0])
                parameter["offset_x"] = np.array([0, 0, 0, 0])

                parameter["position"] = np.array([0, offset_y[1], window_z])
                parameter["handle_z_position"] = np.array([1, 1])

            parameter["size_bottom"][0] = randRange(1, 0.04, 0.06)[0]
            parameter["size_bottom"][1] = parameter["size_bottom"][0]
            parameter["size_bottom"][2] = randRange(1, 0.008, 0.012)[0]

            parameter["size_middle"][0] = randRange(1, 0.035, 0.04)[0]
            parameter["size_middle"][1] = parameter["size_middle"][0]
            parameter["size_middle"][2] = randRange(1, 0.055, 0.065)[0]

            parameter["size_top"][0] = parameter["size_middle"][0]
            parameter["size_top"][2] = parameter["size_top"][0]
            parameter["size_top"][1] = randRange(1, 0.16, 0.21)[0]

            parameter["offset_middle_y"][0] = 0
            parameter["offset_top_y"][0] = (
                -0.5 * parameter["size_top"][1] + 0.5 * parameter["size_bottom"][1]
            )

            parameter["windows_size"] = {
                "size_0": new_concepts[0]["parameters"]["size_0"],
                "size_1": new_concepts[0]["parameters"]["size_1"],
                "size_2": new_concepts[0]["parameters"]["size_2"],
                "size_3": new_concepts[0]["parameters"]["size_2"],
            }

            concept["parameters"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in parameter.items()
            }

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
        window_type = get_window_type()
        existing_concept_templates = concept_template_existence(window_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, window_type)

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
            window_type = get_window_type()
            existing_concept_templates = concept_template_existence(window_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, window_type)
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
