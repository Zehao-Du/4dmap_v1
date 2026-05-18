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


def get_chair_type():
    chair_type = ["Regular_chair", "Sofa", "Cantilever_chair"]
    weights = [1, 1, 1]
    chair_type = random.choices(chair_type, weights=weights, k=1)[0]
    return chair_type


def concept_template_existence(chair_type):
    necessary = [0, 0, 0, 0]
    seat_template = ["Regular_seat"]
    back_template = ["Solid_back"]
    leg_template = ["Regular_leg", "Regular_with_splat_leg"]
    armrest_template = ["Solid_armrest"]
    if chair_type == "Regular_chair":
        seat_template = ["Regular_seat", "Round_seat"]
        back_template = [
            "Solid_back",
            "Ladder_back",
            "Splat_back",
            "Latice_back",
            "Slat_back",
        ]
        leg_template = [
            "Regular_leg",
            "Star_leg",
            "Regular_leg_with_splat",
            "Barstool_leg",
        ]
        armrest_template = ["Office_armrest", "Solid_armrest"]
        necessary = [1, 1, 1, 0.5]
    elif chair_type == "Sofa":
        seat_template = ["Regular_seat"]
        back_template = ["Solid_back"]
        leg_template = ["Regular_leg_with_splat"]
        armrest_template = ["Solid_armrest"]
        necessary = [1, 1, 1, 1]
    elif chair_type == "Cantilever_chair":
        seat_template = ["Regular_seat", "Round_seat"]
        back_template = [
            "Solid_back",
            "Ladder_back",
            "Splat_back",
            "Latice_back",
            "Slat_back",
        ]
        leg_template = ["C_shaped_office_leg"]
        armrest_template = ["Office_armrest"]
        necessary = [1, 1, 1, 1]

    concept_template_variation = {
        "seat": {"template": seat_template, "necessary": necessary[0]},
        "back": {"template": back_template, "necessary": necessary[1]},
        "leg": {"template": leg_template, "necessary": necessary[2]},
        "armrest": {"template": armrest_template, "necessary": necessary[3]},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"] == 0.5:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))
        elif part["necessary"] == 1:
            templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, chair_type):
    new_concepts = []
    leg_type = ""
    seat_type = ""
    back_type = ""
    even_sign = 0
    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if chair_type == "Regular_chair":
            if template == "Regular_seat":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                if parameter["size"][0] < parameter["size"][2]:
                    parameter["size"][0] = (
                        parameter["size"][2] * randRange(1, 0.9, 1.2)[0]
                    )
                seat_type = "Regular"
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Round_seat":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                seat_type = "Round"
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_leg":
                seat_parameter = concepts[0]["parameters"]
                num_of_legs = 4
                parameter["number_of_legs"] = np.array([int(num_of_legs)])
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                parameter["symmetry_mode"][0] = 1
                seat_size = [0, 0, 0]

                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                elif seat_type == "Round":
                    leg_position_angle = np.random.uniform(30, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * np.cos(leg_position_angle) * 2
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * np.sin(leg_position_angle) * 2
                    )

                leg_height = parameter["front_legs_size"][1] * randRange(1, 0.8, 1.2)[0]
                if random.random() < 0.5:
                    leg_height = seat_size[0] * randRange(1, 1.0, 1.2)[0]
                if random.random() < 0.2:
                    parameter["front_rotation"][0] = -np.random.randint(0, 15)
                    parameter["front_rotation"][1] = -np.random.randint(0, 15)
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = np.random.uniform(0, 15)
                    parameter["rear_rotation"][1] = -np.random.uniform(0, 15)
                else:
                    parameter["front_rotation"][0] = -np.random.randint(0, 15)
                    parameter["front_rotation"][1] = -np.random.randint(0, 15)
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = -parameter["front_rotation"][0]
                    parameter["rear_rotation"][1] = parameter["front_rotation"][1]

                parameter["front_legs_size"][0] = (
                    seat_size[0] / 14 * randRange(1, 0.8, 1.2)[0]
                )
                leg_projection_x = leg_height * np.tan(
                    parameter["front_rotation"][0] / 180 * np.pi
                )
                leg_projection_z = leg_height * np.tan(
                    parameter["front_rotation"][1] / 180 * np.pi
                )
                parameter["front_legs_size"][1] = np.sqrt(
                    leg_projection_x**2 + leg_projection_z**2 + leg_height**2
                )
                parameter["front_legs_size"][2] = (
                    seat_size[2] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][0] = (
                    parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][2] = (
                    parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                leg_projection_x = leg_height * np.tan(
                    parameter["rear_rotation"][0] / 180 * np.pi
                )
                leg_projection_z = leg_height * np.tan(
                    parameter["rear_rotation"][1] / 180 * np.pi
                )
                parameter["rear_legs_size"][1] = np.sqrt(
                    leg_projection_x**2 + leg_projection_z**2 + leg_height**2
                )

                parameter["legs_separation"][0] = (
                    seat_size[0]
                    - parameter["front_legs_size"][1]
                    * np.sin(parameter["front_rotation"][0] / 180 * np.pi)
                    * np.sin(parameter["front_rotation"][1] / 180 * np.pi)
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][1] = (
                    seat_size[0] - parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][2] = (
                    seat_size[2]
                    - parameter["rear_legs_size"][1]
                    * np.sin(parameter["rear_rotation"][0] / 180 * np.pi)
                    / 2
                ) * randRange(1, 0.7, 1.0)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

                new_concepts.append(concept)

            elif template == "Regular_leg_with_splat":
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                probabilities = [
                    0.5777777777777777,
                    0.5777777777777777,
                    0.9111111111111111,
                    0.9111111111111111,
                ]
                parameter["bridging_bars_existance"] = np.array(
                    [np.random.choice([0, 1], p=[1 - p, p]) for p in probabilities]
                )
                seat_size = [0, 0, 0]

                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                elif seat_type == "Round":
                    leg_position_angle = np.random.uniform(30, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * np.cos(leg_position_angle) * 2
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * np.sin(leg_position_angle) * 2
                    )
                leg_height = parameter["front_legs_size"][1] * randRange(1, 0.8, 1.2)[0]
                if random.random() < 0.5:
                    leg_height = seat_size[0] * randRange(1, 1.2, 1.5)[0]
                if random.random() < 0.2:
                    parameter["front_rotation"][0] = -np.random.randint(0, 15)
                    parameter["front_rotation"][1] = -np.random.randint(0, 15)
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = np.random.uniform(0, 15)
                    parameter["rear_rotation"][1] = -np.random.uniform(0, 15)
                else:
                    parameter["front_rotation"][0] = -np.random.randint(0, 15)
                    parameter["front_rotation"][1] = -np.random.randint(0, 15)
                    parameter["central_rotation"][0] = 0
                    parameter["rear_rotation"][0] = -parameter["front_rotation"][0]
                    parameter["rear_rotation"][1] = parameter["front_rotation"][1]

                parameter["front_legs_size"][0] = (
                    seat_size[0] / 14 * randRange(1, 0.8, 1.2)[0]
                )
                leg_projection_x = leg_height * np.tan(
                    parameter["front_rotation"][0] / 180 * np.pi
                )
                leg_projection_z = leg_height * np.tan(
                    parameter["front_rotation"][1] / 180 * np.pi
                )
                parameter["front_legs_size"][1] = np.sqrt(
                    leg_projection_x**2 + leg_projection_z**2 + leg_height**2
                )
                parameter["front_legs_size"][2] = (
                    seat_size[2] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][0] = (
                    parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][2] = (
                    parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                leg_projection_x = leg_height * np.tan(
                    parameter["rear_rotation"][0] / 180 * np.pi
                )
                leg_projection_z = leg_height * np.tan(
                    parameter["rear_rotation"][1] / 180 * np.pi
                )
                parameter["rear_legs_size"][1] = np.sqrt(
                    leg_projection_x**2 + leg_projection_z**2 + leg_height**2
                )

                parameter["legs_separation"][0] = (
                    seat_size[0]
                    - parameter["front_legs_size"][1]
                    * np.sin(parameter["front_rotation"][0] / 180 * np.pi)
                    * np.sin(parameter["front_rotation"][1] / 180 * np.pi)
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][1] = (
                    seat_size[0] - parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][2] = (
                    seat_size[2]
                    - parameter["rear_legs_size"][1]
                    * np.sin(parameter["rear_rotation"][0] / 180 * np.pi)
                    / 2
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["front_rear_bridging_bars_sizes"][0] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["front_rear_bridging_bars_sizes"][1] = np.minimum(
                    parameter["front_legs_size"][2], parameter["rear_legs_size"][2]
                )
                parameter["left_right_bridging_bars_sizes"][1] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["left_right_bridging_bars_sizes"][0] = np.minimum(
                    parameter["front_legs_size"][0], parameter["rear_legs_size"][0]
                )

                parameter["front_rear_bridging_bars_offset"][0] = (
                    leg_height / 2 * randRange(1, 0.1, 0.8)[0]
                )
                parameter["left_right_bridging_bars_offset"][0] = (
                    leg_height / 2 * randRange(1, 0.1, 0.8)[0]
                )

                parameter["front_rear_bridging_bars_offset"][1] = (
                    parameter["front_rear_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_bridging_bars_offset"][1] = (
                    parameter["left_right_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

                new_concepts.append(concept)

            elif template == "Star_leg":
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = (
                    seat_parameter["position"][1] - seat_parameter["size"][1] / 2
                )
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]

                seat_size = [0, 0, 0]
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                elif seat_type == "Round":
                    seat_size[0] = seat_parameter["size"][0] * 2
                    seat_size[1] = seat_parameter["size"][1]

                parameter["number_of_sub_legs"] = np.array(
                    [int(np.random.randint(3, 5))]
                )
                parameter["horizontal_rotation"][0] = 0

                parameter["vertical_sizes"][0] = (
                    seat_size[1] * 5 / 13 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["vertical_sizes"][1] = (
                    seat_size[0] / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["tilt_angle"][0] *= randRange(1, 0.8, 1.2)
                parameter["sub_sizes"][0] = parameter["vertical_sizes"][0] / 2
                parameter["sub_sizes"][1] = parameter["vertical_sizes"][0] * 1.5
                parameter["sub_sizes"][2] = parameter["vertical_sizes"][1] * 4 / 5

                parameter["central_rotation"][0] *= randRange(1, 0.8, 1.2)[0]

                parameter["position"][1] -= parameter["vertical_sizes"][1] / 2

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Barstool_leg":
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = (
                    seat_parameter["position"][1] - seat_parameter["size"][1] / 2
                )
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                elif seat_type == "Round":
                    seat_size[0] = seat_parameter["size"][0] * 2
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][0] * 2
                parameter["horizontal_rotation"][0] = 0
                parameter["vertical_sizes"][0] = (
                    seat_size[0] / 12 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["vertical_sizes"][1] = (
                    seat_size[0] * 2 / 3 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bottom_sizes"][0] = (
                    seat_size[0] / 2 * randRange(1, 0.6, 1.0)[0]
                )
                parameter["bottom_sizes"][1] = (
                    seat_size[1] / 2 * randRange(1, 0.8, 1.2)[0]
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Solid_back":
                back_type = "Solid_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]

                parameter["size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][1] *= randRange(1, 0.8, 1.2)[0]

                parameter["back_rotation"][0] *= randRange(1, 0.8, 1.2)[0]
                if seat_type == "Regular":
                    parameter["size"][0] = (
                        seat_parameter["size"][0] * randRange(1, 0.8, 1.1)[0]
                    )
                    parameter["position"][2] -= seat_parameter["size"][2] / 2
                    parameter["size"][1] = (
                        seat_parameter["size"][0] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["size"][2] = (
                        seat_parameter["size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                elif seat_type == "Round":
                    back_position_angle = np.random.uniform(60, 90) / 180 * np.pi
                    parameter["position"][2] -= seat_parameter["size"][0] * np.sin(
                        back_position_angle
                    )
                    parameter["size"][0] = (
                        seat_parameter["size"][0] * 2 * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["size"][1] = (
                        seat_parameter["size"][0] * 2 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["size"][2] = (
                        seat_parameter["size"][1] * randRange(1, 0.8, 1.2)[0]
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Ladder_back":
                back_type = "Ladder_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    total_seat_size_x = seat_parameter["size"][0]
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                elif seat_type == "Round":
                    back_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    total_seat_size_x = seat_parameter["size"][0] * 2
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(back_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(back_position_angle)
                    )

                parameter["main_vertical_piece_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_vertical_piece_size"][1] = (
                    total_seat_size_x * randRange(1, 0.8, 1.2)[0]
                )
                if seat_type == "Regular":
                    parameter["main_vertical_separation"][0] = (
                        seat_size[0] - parameter["main_vertical_piece_size"][0]
                    )
                elif seat_type == "Round":
                    parameter["main_vertical_separation"][0] = seat_size[0]
                if random.random() < 0.5:
                    parameter["main_horizontal_piece_size"][0] = (
                        parameter["main_vertical_separation"][0]
                        + parameter["main_vertical_piece_size"][0]
                    )
                    parameter["main_horizontal_piece_size"][1] *= randRange(
                        1, 0.8, 1.2
                    )[0]
                    parameter["main_vertical_piece_size"][1] = (
                        total_seat_size_x * 2 / 3 * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["main_horizontal_piece_size"][2] = (
                        seat_size[1] * randRange(1, 0.5, 1.0)[0]
                    )
                else:
                    parameter["main_horizontal_piece_size"][0] = (
                        parameter["main_vertical_separation"][0]
                        + parameter["main_vertical_piece_size"][0]
                    )
                    parameter["main_horizontal_piece_size"][1] = (
                        seat_size[1] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["main_horizontal_piece_size"][2] = (
                        seat_size[1] * randRange(1, 0.5, 1.0)[0]
                    )

                parameter["main_vertical_piece_size"][2] = parameter[
                    "main_horizontal_piece_size"
                ][2]
                parameter["sub_offset"][0] = 0
                number_of_subs = int(np.random.randint(0, 5))
                parameter["number_of_subs"][0] = int(number_of_subs)
                parameter["interval_between_subs"][0] = (
                    parameter["main_vertical_piece_size"][1] / 2
                    - parameter["sub_offset"][0]
                ) / (parameter["number_of_subs"][0] + 1)
                parameter["sub_horizontal_piece_size"][1] = (
                    parameter["main_vertical_piece_size"][2] * randRange(1, 0.8, 1.0)[0]
                )
                parameter["sub_horizontal_piece_size"][0] = (
                    parameter["interval_between_subs"][0]
                    / 2
                    * randRange(1, 0.5, 0.7)[0]
                )
                parameter["back_rotation"][0] *= randRange(1, 0.8, 1.5)[0]
                parameter["position"][2] -= (
                    seat_size[2] / 2 - parameter["main_vertical_piece_size"][2] / 2
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["number_of_subs"][0] = int(number_of_subs)
                new_concepts.append(concept)

            elif template == "Splat_back":
                back_type = "Splat_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                    total_seat_size_x = seat_parameter["size"][0]
                elif seat_type == "Round":
                    back_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(back_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(back_position_angle)
                    )
                    total_seat_size_x = seat_parameter["size"][0] * 2

                number_of_subs = np.random.randint(0, 4)
                parameter["main_vertical_piece_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_vertical_piece_size"][1] = (
                    total_seat_size_x * randRange(1, 0.8, 1.2)[0]
                )
                if seat_type == "Regular":
                    parameter["main_vertical_separation"][0] = (
                        seat_size[0] - parameter["main_vertical_piece_size"][0]
                    )
                elif seat_type == "Round":
                    parameter["main_vertical_separation"][0] = seat_size[0]
                if random.random() < 0.5:
                    parameter["main_horizontal_piece_size"][0] = (
                        parameter["main_vertical_separation"][0]
                        + parameter["main_vertical_piece_size"][0]
                    ) * randRange(1, 0.8, 1.2)[0]
                    parameter["main_horizontal_piece_size"][1] *= randRange(
                        1, 0.8, 1.2
                    )[0]
                    parameter["main_vertical_piece_size"][1] = (
                        total_seat_size_x * 2 / 3 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["main_horizontal_piece_size"][2] = (
                        seat_size[1] * randRange(1, 0.5, 1.0)[0]
                    )
                else:
                    parameter["main_horizontal_piece_size"][0] = (
                        parameter["main_vertical_separation"][0]
                        + parameter["main_vertical_piece_size"][0]
                    ) * randRange(1, 0.8, 1.2)[0]
                    parameter["main_horizontal_piece_size"][1] = (
                        seat_size[1] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["main_horizontal_piece_size"][2] = (
                        seat_size[1] * randRange(1, 0.5, 1.0)[0]
                    )

                parameter["main_vertical_piece_size"][2] = parameter[
                    "main_horizontal_piece_size"
                ][2]
                parameter["interval_between_subs"][0] = (
                    seat_size[0] - parameter["main_vertical_piece_size"][0]
                ) / (number_of_subs + 1)
                parameter["sub_offset"][0] = -(
                    (seat_size[0] - parameter["main_vertical_piece_size"][0]) / 2
                    - parameter["interval_between_subs"][0]
                )
                parameter["sub_vertical_piece_size"][0] = (
                    parameter["interval_between_subs"][0]
                    / 4
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_vertical_piece_size"][1] = parameter[
                    "main_vertical_piece_size"
                ][2]
                parameter["back_rotation"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["position"][2] -= (
                    seat_size[2] / 2 - parameter["main_vertical_piece_size"][2] / 2
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["number_of_subs"][0] = int(number_of_subs)
                new_concepts.append(concept)

            elif template == "Latice_back":
                back_type = "Latice_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                    total_seat_size_x = seat_size[0]
                elif seat_type == "Round":
                    back_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(back_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(back_position_angle)
                    )
                    total_seat_size_x = seat_parameter["size"][0] * 2

                number_of_subs = np.random.randint(0, 5)

                parameter["main_vertical_piece_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_vertical_piece_size"][1] = (
                    total_seat_size_x * randRange(1, 0.8, 1.2)[0]
                )

                if seat_type == "Regular":
                    parameter["main_vertical_separation"][0] = (
                        seat_size[0] - parameter["main_vertical_piece_size"][0]
                    )
                elif seat_type == "Round":
                    parameter["main_vertical_separation"][0] = seat_size[0]

                parameter["main_vertical_piece_size"][2] = parameter[
                    "main_horizontal_piece_size"
                ][2]

                parameter["main_horizontal_piece_size"][0] = (
                    parameter["main_vertical_separation"][0]
                    + parameter["main_vertical_piece_size"][0]
                )
                parameter["main_horizontal_piece_size"][1] = (
                    seat_size[1] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_horizontal_piece_size"][2] = (
                    seat_size[1] * randRange(1, 0.5, 1.0)[0]
                )

                parameter["sub_horizontal_offset"][0] = (
                    -parameter["main_vertical_piece_size"][1]
                    / 2
                    * randRange(1, 0.8, 1.0)[0]
                )

                parameter["sub_horizontal_piece_size"][0] = (
                    parameter["main_vertical_piece_size"][1]
                    / 10
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_horizontal_piece_size"][1] = (
                    parameter["main_vertical_piece_size"][2] * randRange(1, 0.8, 1.0)[0]
                )

                parameter["interval_between_subs"][0] = (
                    seat_size[0] - parameter["main_vertical_piece_size"][0]
                ) / (number_of_subs + 1)

                parameter["sub_vertical_offset"][0] = -(
                    (seat_size[0] - parameter["main_vertical_piece_size"][0]) / 2
                    - parameter["interval_between_subs"][0]
                )

                parameter["sub_vertical_piece_size"][0] = (
                    parameter["interval_between_subs"][0]
                    / 4
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_vertical_piece_size"][1] = (
                    parameter["main_vertical_piece_size"][2] * randRange(1, 0.8, 1.0)[0]
                )

                parameter["position"][2] -= (
                    seat_size[2] / 2 - parameter["main_vertical_piece_size"][2] / 2
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["number_of_subs"][0] = int(number_of_subs)
                new_concepts.append(concept)

            elif template == "Slat_back":
                back_type = "Slat_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                    total_seat_size_x = seat_size[0]
                elif seat_type == "Round":
                    back_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(back_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(back_position_angle)
                    )
                    total_seat_size_x = seat_parameter["size"][0] * 2

                parameter["main_vertical_piece_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_vertical_piece_size"][1] = (
                    total_seat_size_x * randRange(1, 0.8, 1.2)[0]
                )
                if seat_type == "Regular":
                    parameter["main_vertical_separation"][0] = (
                        seat_size[0] - parameter["main_vertical_piece_size"][0]
                    )
                elif seat_type == "Round":
                    parameter["main_vertical_separation"][0] = seat_size[0]

                parameter["main_vertical_piece_size"][2] = (
                    seat_size[1] * randRange(1, 0.5, 1.0)[0]
                )
                parameter["sub_horizontal_offset"][0] = (
                    parameter["main_vertical_piece_size"][1]
                    / 2
                    * randRange(1, 0.6, 0.9)[0]
                )
                number_of_subs = int(np.random.randint(2, 6))
                parameter["number_of_subs"][0] = int(number_of_subs)
                parameter["interval_between_subs"][0] = (
                    parameter["main_vertical_piece_size"][1] / 2
                    + parameter["sub_horizontal_offset"][0]
                ) / (parameter["number_of_subs"][0] + 1)
                parameter["sub_horizontal_piece_size"][1] = (
                    parameter["main_vertical_piece_size"][2] * randRange(1, 0.8, 1.0)[0]
                )
                parameter["sub_horizontal_piece_size"][0] = (
                    parameter["interval_between_subs"][0]
                    / 2
                    * randRange(1, 0.5, 0.7)[0]
                )
                parameter["back_rotation"][0] *= randRange(1, 0.8, 1.5)
                parameter["position"][2] -= (
                    seat_size[2] / 2 - parameter["main_vertical_piece_size"][2] / 2
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["number_of_subs"][0] = int(number_of_subs)
                new_concepts.append(concept)

            elif template == "Solid_armrest":
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                    total_seat_size_x = seat_parameter["size"][0]
                elif seat_type == "Round":
                    armrest_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(armrest_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(armrest_position_angle)
                    )
                    total_seat_size_x = seat_parameter["size"][0] * 2

                armrest_height = seat_size[0] * 3 / 2 * randRange(1, 0.8, 1.2)[0]

                parameter["size"][0] = seat_size[0] / 8 * randRange(1, 0.7, 1.5)[0]
                parameter["size"][1] = armrest_height / 3 * randRange(1, 0.8, 1.2)[0]
                parameter["size"][2] = seat_size[2] * randRange(1, 0.7, 0.9)[0]
                parameter["armrest_rotation"] *= randRange(
                    parameter["armrest_rotation"].shape[0], 0.8, 1.2
                )
                if seat_type == "Regular":
                    parameter["armrest_separation"][0] = (
                        total_seat_size_x
                        + parameter["size"][0] * randRange(1, 0.8, 1.0)[0]
                    )
                elif seat_type == "Round":
                    parameter["armrest_separation"][0] = total_seat_size_x
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Office_armrest":
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                horizontal_support_size_z = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                elif seat_type == "Round":
                    seat_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(seat_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(seat_position_angle)
                    )
                    horizontal_support_position_angle = (
                        np.random.uniform(40, 50) / 180 * np.pi
                    )
                    horizontal_support_size_z = seat_parameter["size"][0] * (
                        1 + np.sin(horizontal_support_position_angle)
                    )

                armrest_height = seat_size[0] / 3 * randRange(1, 0.8, 1.2)[0]

                rotation_sign = 1 if random.random() < 0.5 else -1
                parameter["vertical_support_rotation"][0] = np.random.uniform(-45, 45)
                vertical_support_angle = (
                    parameter["vertical_support_rotation"][0] / 180 * np.pi
                )
                if seat_type == "Regular":
                    parameter["horizontal_support_sizes"][0] = (
                        seat_size[0] / 8 * randRange(1, 0.8, 1.3)[0]
                    )
                    parameter["horizontal_support_sizes"][1] = (
                        parameter["horizontal_support_sizes"][0]
                        * 5
                        / 7
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["horizontal_support_sizes"][2] = (
                        seat_size[2] * randRange(1, 0.7, 0.9)[0]
                    )
                    parameter["vertical_support_sizes"][0] = (
                        parameter["horizontal_support_sizes"][0]
                        * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["vertical_support_sizes"][1] = (
                        armrest_height * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["vertical_support_sizes"][2] = (
                        parameter["horizontal_support_sizes"][0]
                        * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["armrest_rotation"] *= rotation_sign * randRange(
                        parameter["armrest_rotation"].shape[0], 0.8, 1.2
                    )
                    parameter["armrest_separation"][0] = (
                        seat_size[0] * randRange(1, 0.8, 1.0)[0]
                    )

                    range_of_vertical_support_offset_z = seat_parameter["size"][
                        2
                    ] / 2 + (
                        parameter["vertical_support_sizes"][1]
                        * np.sin(vertical_support_angle)
                        / 2
                    )
                    if range_of_vertical_support_offset_z < 0:
                        range_of_vertical_support_offset_z = 0
                        vertical_support_angle = np.arcsin(
                            -seat_parameter["size"][2]
                            / parameter["vertical_support_sizes"][1]
                        )
                        parameter["vertical_support_rotation"][0] = (
                            vertical_support_angle * 180 / np.pi
                        )
                    if vertical_support_angle > 0:
                        range_of_horizontal_arm_length = (
                            parameter["horizontal_support_sizes"][2] / 2
                            - parameter["vertical_support_sizes"][1]
                            * np.sin(vertical_support_angle)
                            / 2
                        )
                        range_of_vertical_support_offset_z = np.minimum(
                            range_of_vertical_support_offset_z,
                            range_of_horizontal_arm_length,
                        )
                    parameter["supports_contact_offset"][0] = (
                        range_of_vertical_support_offset_z * randRange(1, 0.5, 1.0)[0]
                    )
                    if range_of_vertical_support_offset_z < 0:
                        parameter["supports_contact_offset"][0] = (
                            range_of_vertical_support_offset_z
                            * randRange(1, 1.0, 1.2)[0]
                        )

                elif seat_type == "Round":
                    parameter["horizontal_support_sizes"][0] = (
                        seat_size[0] / 8 * randRange(1, 0.8, 1.3)[0]
                    )
                    parameter["horizontal_support_sizes"][1] = (
                        parameter["horizontal_support_sizes"][0]
                        * 5
                        / 7
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["horizontal_support_sizes"][2] = (
                        horizontal_support_size_z * randRange(1, 0.7, 0.9)[0]
                    )
                    parameter["vertical_support_sizes"][0] = (
                        parameter["horizontal_support_sizes"][0]
                        * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["vertical_support_sizes"][1] = (
                        armrest_height * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["vertical_support_sizes"][2] = (
                        parameter["horizontal_support_sizes"][0]
                        * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["armrest_rotation"] *= randRange(
                        parameter["armrest_rotation"].shape[0], 0.8, 1.2
                    )
                    armrest_position_angle = np.random.uniform(10, 30) / 180 * np.pi
                    armrest_position_x = (
                        seat_parameter["size"][0] * 2 * np.cos(armrest_position_angle)
                    )
                    armrest_position_z = seat_parameter["size"][0] * np.sin(
                        armrest_position_angle
                    )
                    parameter["armrest_separation"][0] = armrest_position_x

                    range_of_vertical_support_offset_z = armrest_position_z + (
                        parameter["vertical_support_sizes"][1]
                        * np.sin(vertical_support_angle)
                        / 2
                    )
                    if vertical_support_angle > 0:
                        range_of_horizontal_arm_length = (
                            parameter["horizontal_support_sizes"][2] / 2
                            - parameter["vertical_support_sizes"][1]
                            * np.sin(vertical_support_angle)
                            / 2
                        )
                        range_of_vertical_support_offset_z = np.minimum(
                            range_of_vertical_support_offset_z,
                            range_of_horizontal_arm_length,
                        )
                    parameter["supports_contact_offset"][0] = (
                        range_of_vertical_support_offset_z * randRange(1, 0.5, 1.0)[0]
                    )
                    if range_of_vertical_support_offset_z < 0:
                        parameter["supports_contact_offset"][0] = (
                            range_of_vertical_support_offset_z
                            * randRange(1, 1.0, 1.2)[0]
                        )

                parameter["horizontal_support_rotation"] *= randRange(1, 0.8, 1.2)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif chair_type == "Sofa":
            if template == "Regular_seat":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                parameter["size"][1] = (
                    parameter["size"][0] / 3 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = (
                    parameter["size"][0] * 4 / 5 * randRange(1, 0.8, 1.2)[0]
                )
                seat_type = "Regular"
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Solid_back":
                back_type = "Solid_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]

                parameter["back_rotation"][0] = 0
                if random.random() < 0.6:
                    parameter["back_rotation"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][0] = (
                    seat_parameter["size"][0] * randRange(1, 0.8, 1.0)[0]
                )
                parameter["size"][1] = (
                    seat_parameter["size"][0] * randRange(1, 0.5, 1.0)[0]
                )
                if random.random() < 0.3:
                    parameter["size"][1] = (
                        seat_parameter["size"][0] / 2 * randRange(1, 0.8, 1.0)[0]
                    )
                    even_sign = 1
                else:
                    even_sign = 0
                parameter["size"][2] = (
                    seat_parameter["size"][0] / 4 * randRange(1, 0.6, 0.8)[0]
                )
                parameter["position"][2] -= (
                    seat_parameter["size"][2] / 2 - parameter["size"][2] / 2
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_leg":
                seat_parameter = concepts[0]["parameters"]
                num_of_legs = 4
                parameter["number_of_legs"] = np.array([int(num_of_legs)])
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                parameter["symmetry_mode"][0] = 1
                seat_size = [0, 0, 0]
                if seat_type == "Regular":
                    seat_size = seat_parameter["size"]

                leg_height = seat_size[1] / 6 * randRange(1, 0.8, 1.2)[0]
                parameter["front_rotation"][0] = 0
                parameter["front_rotation"][1] = 0
                parameter["central_rotation"][0] = 0
                parameter["rear_rotation"][0] = 0
                parameter["rear_rotation"][1] = 0

                parameter["front_legs_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                leg_projection_x = leg_height * np.tan(
                    parameter["front_rotation"][0] / 180 * np.pi
                )
                leg_projection_z = leg_height * np.tan(
                    parameter["front_rotation"][1] / 180 * np.pi
                )
                parameter["front_legs_size"][1] = np.sqrt(
                    leg_projection_x**2 + leg_projection_z**2 + leg_height**2
                )
                parameter["front_legs_size"][2] = (
                    seat_size[2] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][0] = (
                    parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][2] = (
                    parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                leg_projection_x = leg_height * np.tan(
                    parameter["rear_rotation"][0] / 180 * np.pi
                )
                leg_projection_z = leg_height * np.tan(
                    parameter["rear_rotation"][1] / 180 * np.pi
                )
                parameter["rear_legs_size"][1] = np.sqrt(
                    leg_projection_x**2 + leg_projection_z**2 + leg_height**2
                )

                parameter["legs_separation"][0] = (
                    seat_size[0] - parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][1] = (
                    seat_size[0] - parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][2] = (
                    seat_size[2] - parameter["front_legs_size"][2]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["position"][1] -= seat_parameter["size"][1] / 2
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Regular_leg_with_splat":
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]

                probabilities = [
                    0.5777777777777777,
                    0.5777777777777777,
                    0.9111111111111111,
                    0.9111111111111111,
                ]
                parameter["bridging_bars_existance"] = np.array(
                    [np.random.choice([0, 1], p=[1 - p, p]) for p in probabilities]
                )
                seat_size = seat_parameter["size"]
                leg_height = seat_size[1] / 5 * randRange(1, 0.8, 1.2)[0]
                parameter["front_rotation"][0] = 0
                parameter["front_rotation"][1] = 0
                parameter["central_rotation"][0] = 0
                parameter["rear_rotation"][0] = -parameter["front_rotation"][0]
                parameter["rear_rotation"][1] = parameter["front_rotation"][1]

                parameter["front_legs_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                leg_projection_x = leg_height * np.tan(
                    parameter["front_rotation"][0] / 180 * np.pi
                )
                leg_projection_z = leg_height * np.tan(
                    parameter["front_rotation"][1] / 180 * np.pi
                )
                parameter["front_legs_size"][1] = np.sqrt(
                    leg_projection_x**2 + leg_projection_z**2 + leg_height**2
                )
                parameter["front_legs_size"][2] = (
                    seat_size[2] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][0] = (
                    parameter["front_legs_size"][0] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rear_legs_size"][2] = (
                    parameter["front_legs_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                leg_projection_x = leg_height * np.tan(
                    parameter["rear_rotation"][0] / 180 * np.pi
                )
                leg_projection_z = leg_height * np.tan(
                    parameter["rear_rotation"][1] / 180 * np.pi
                )
                parameter["rear_legs_size"][1] = np.sqrt(
                    leg_projection_x**2 + leg_projection_z**2 + leg_height**2
                )

                parameter["legs_separation"][0] = (
                    seat_size[0] - parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][1] = (
                    seat_size[0] - parameter["front_legs_size"][0]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["legs_separation"][2] = (
                    seat_size[2] - parameter["front_legs_size"][2]
                ) * randRange(1, 0.7, 1.0)[0]
                parameter["front_rear_bridging_bars_sizes"][0] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["front_rear_bridging_bars_sizes"][0] = np.minimum(
                    parameter["front_rear_bridging_bars_sizes"][0],
                    parameter["front_legs_size"][1] * randRange(1, 0.2, 0.6)[0],
                )
                parameter["front_rear_bridging_bars_sizes"][1] = np.minimum(
                    parameter["front_legs_size"][2], parameter["rear_legs_size"][2]
                )
                parameter["left_right_bridging_bars_sizes"][1] *= randRange(
                    1, 0.5, 1.2
                )[0]
                parameter["left_right_bridging_bars_sizes"][1] = np.minimum(
                    parameter["front_rear_bridging_bars_sizes"][0],
                    parameter["front_legs_size"][1] * randRange(1, 0.2, 0.6)[0],
                )
                parameter["left_right_bridging_bars_sizes"][0] = np.minimum(
                    parameter["front_legs_size"][0], parameter["rear_legs_size"][0]
                )

                parameter["front_rear_bridging_bars_offset"][0] = (
                    leg_height / 2 * randRange(1, 0.1, 0.8)[0]
                )
                parameter["left_right_bridging_bars_offset"][0] = (
                    leg_height / 2 * randRange(1, 0.1, 0.8)[0]
                )

                parameter["front_rear_bridging_bars_offset"][1] = (
                    parameter["front_rear_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["left_right_bridging_bars_offset"][1] = (
                    parameter["left_right_bridging_bars_offset"][0]
                    * randRange(1, 0.8, 1.2)[0]
                )

                parameter["position"][1] -= seat_parameter["size"][1] / 2
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

                new_concepts.append(concept)

            elif template == "Solid_armrest":
                if back_type != "Solid_back":
                    pass
                seat_parameter = concepts[0]["parameters"]
                back_parameter = concepts[1]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                    total_seat_size_x = seat_parameter["size"][0]
                parameter["size"][0] = seat_size[0] / 8 * randRange(1, 1.0, 1.5)[0]
                parameter["size"][1] = (
                    seat_parameter["size"][1] * randRange(1, 0.8, 1.0)[0]
                )
                parameter["size"][2] = seat_size[2] * randRange(1, 0.8, 1.0)[0]
                parameter["armrest_rotation"] *= randRange(
                    parameter["armrest_rotation"].shape[0], 0.8, 1.2
                )
                parameter["armrest_separation"][0] = (
                    total_seat_size_x - parameter["size"][0] * randRange(1, 0.0, 1.0)[0]
                )

                if even_sign == 1:
                    parameter["size"][1] = back_parameter["size"][1]
                    parameter["armrest_rotation"][0] = 0
                    parameter["armrest_rotation"][1] = 0
                else:
                    parameter["position"][1] += (
                        seat_parameter["size"][1] / 2 * randRange(1, 0.8, 0.9)[0]
                    )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif chair_type == "Cantilever_chair":
            if template == "Regular_seat":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                seat_type = "Regular"
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Round_seat":
                parameter["size"] *= randRange(parameter["size"].shape[0], 0.8, 1.2)
                seat_type = "Round"
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Solid_back":
                back_type = "Solid_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                parameter["size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size"][1] *= randRange(1, 0.8, 1.2)[0]
                parameter["back_rotation"][0] *= randRange(1, 0.8, 1.2)[0]
                if seat_type == "Regular":
                    parameter["size"][0] = (
                        seat_parameter["size"][0] * randRange(1, 0.8, 1.1)[0]
                    )
                    parameter["position"][2] -= seat_parameter["size"][2] / 2
                    parameter["size"][1] = (
                        seat_parameter["size"][0] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["size"][2] = (
                        seat_parameter["size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                elif seat_type == "Round":
                    seat_position_angle = np.random.uniform(60, 90) / 180 * np.pi
                    parameter["position"][2] -= seat_parameter["size"][0] * np.sin(
                        seat_position_angle
                    )
                    parameter["size"][0] = (
                        seat_parameter["size"][0] * 2 * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["size"][1] = (
                        seat_parameter["size"][0] * 2 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["size"][2] = (
                        seat_parameter["size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Ladder_back":
                back_type = "Ladder_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    total_seat_size_x = seat_parameter["size"][0]
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                elif seat_type == "Round":
                    seat_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    total_seat_size_x = seat_parameter["size"][0] * 2
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(seat_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(seat_position_angle)
                    )

                parameter["main_vertical_piece_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_vertical_piece_size"][1] = (
                    total_seat_size_x * randRange(1, 0.8, 1.2)[0]
                )
                if seat_type == "Regular":
                    parameter["main_vertical_separation"][0] = (
                        seat_size[0] - parameter["main_vertical_piece_size"][0]
                    )
                elif seat_type == "Round":
                    parameter["main_vertical_separation"][0] = seat_size[0]
                if random.random() < 0.5:
                    parameter["main_horizontal_piece_size"][0] = (
                        parameter["main_vertical_separation"][0]
                        + parameter["main_vertical_piece_size"][0]
                    )
                    parameter["main_horizontal_piece_size"][1] *= randRange(
                        1, 0.8, 1.2
                    )[0]
                    parameter["main_vertical_piece_size"][1] = (
                        total_seat_size_x * 2 / 3 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["main_horizontal_piece_size"][2] = (
                        seat_size[1] * randRange(1, 0.5, 1.0)[0]
                    )
                else:
                    parameter["main_horizontal_piece_size"][0] = (
                        parameter["main_vertical_separation"][0]
                        + parameter["main_vertical_piece_size"][0]
                    )
                    parameter["main_horizontal_piece_size"][1] = (
                        seat_size[1] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["main_horizontal_piece_size"][2] = (
                        seat_size[1] * randRange(1, 0.5, 1.0)[0]
                    )

                parameter["main_vertical_piece_size"][2] = parameter[
                    "main_horizontal_piece_size"
                ][2]
                parameter["sub_offset"][0] = 0
                number_of_subs = int(np.random.randint(0, 5))
                parameter["number_of_subs"][0] = int(number_of_subs)
                parameter["interval_between_subs"][0] = (
                    parameter["main_vertical_piece_size"][1] / 2
                    - parameter["sub_offset"][0]
                ) / (parameter["number_of_subs"][0] + 1)
                parameter["sub_horizontal_piece_size"][1] = (
                    parameter["main_vertical_piece_size"][2] * randRange(1, 0.8, 1.0)[0]
                )
                parameter["sub_horizontal_piece_size"][0] = (
                    parameter["interval_between_subs"][0]
                    / 2
                    * randRange(1, 0.5, 0.7)[0]
                )
                parameter["back_rotation"][0] *= randRange(1, 0.8, 1.5)[0]
                parameter["position"][2] -= (
                    seat_size[2] / 2 - parameter["main_vertical_piece_size"][2] / 2
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["number_of_subs"][0] = int(number_of_subs)
                new_concepts.append(concept)

            elif template == "Splat_back":
                back_type = "Splat_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                    total_seat_size_x = seat_parameter["size"][0]
                elif seat_type == "Round":
                    seat_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(seat_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(seat_position_angle)
                    )
                    total_seat_size_x = seat_parameter["size"][0] * 2

                number_of_subs = np.random.randint(0, 4)
                parameter["main_vertical_piece_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_vertical_piece_size"][1] = (
                    total_seat_size_x * randRange(1, 0.8, 1.2)[0]
                )
                if seat_type == "Regular":
                    parameter["main_vertical_separation"][0] = (
                        seat_size[0] - parameter["main_vertical_piece_size"][0]
                    )
                elif seat_type == "Round":
                    parameter["main_vertical_separation"][0] = seat_size[0]
                if random.random() < 0.5:
                    parameter["main_horizontal_piece_size"][0] = (
                        parameter["main_vertical_separation"][0]
                        + parameter["main_vertical_piece_size"][0]
                    ) * randRange(1, 0.8, 1.2)[0]
                    parameter["main_horizontal_piece_size"][1] *= randRange(
                        1, 0.8, 1.2
                    )[0]
                    parameter["main_vertical_piece_size"][1] = (
                        total_seat_size_x * 2 / 3 * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["main_horizontal_piece_size"][2] = (
                        seat_size[1] * randRange(1, 0.5, 1.0)[0]
                    )
                else:
                    parameter["main_horizontal_piece_size"][0] = (
                        parameter["main_vertical_separation"][0]
                        + parameter["main_vertical_piece_size"][0]
                    ) * randRange(1, 0.8, 1.2)[0]
                    parameter["main_horizontal_piece_size"][1] = (
                        seat_size[1] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["main_horizontal_piece_size"][2] = (
                        seat_size[1] * randRange(1, 0.5, 1.0)[0]
                    )

                parameter["main_vertical_piece_size"][2] = parameter[
                    "main_horizontal_piece_size"
                ][2]
                parameter["interval_between_subs"][0] = (
                    seat_size[0] - parameter["main_vertical_piece_size"][0]
                ) / (number_of_subs + 1)
                parameter["sub_offset"][0] = -(
                    (seat_size[0] - parameter["main_vertical_piece_size"][0]) / 2
                    - parameter["interval_between_subs"][0]
                )
                parameter["sub_vertical_piece_size"][0] = (
                    parameter["interval_between_subs"][0]
                    / 4
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_vertical_piece_size"][1] = parameter[
                    "main_vertical_piece_size"
                ][2]
                parameter["back_rotation"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["position"][2] -= (
                    seat_size[2] / 2 - parameter["main_vertical_piece_size"][2] / 2
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["number_of_subs"][0] = int(number_of_subs)
                new_concepts.append(concept)

            elif template == "Latice_back":
                back_type = "Latice_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                    total_seat_size_x = seat_size[0]
                elif seat_type == "Round":
                    seat_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(seat_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(seat_position_angle)
                    )
                    total_seat_size_x = seat_parameter["size"][0] * 2

                number_of_subs = np.random.randint(0, 5)

                parameter["main_vertical_piece_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_vertical_piece_size"][1] = (
                    total_seat_size_x * randRange(1, 0.8, 1.2)[0]
                )

                if seat_type == "Regular":
                    parameter["main_vertical_separation"][0] = (
                        seat_size[0] - parameter["main_vertical_piece_size"][0]
                    )
                elif seat_type == "Round":
                    parameter["main_vertical_separation"][0] = seat_size[0]

                parameter["main_vertical_piece_size"][2] = parameter[
                    "main_horizontal_piece_size"
                ][2]

                parameter["main_horizontal_piece_size"][0] = (
                    parameter["main_vertical_separation"][0]
                    + parameter["main_vertical_piece_size"][0]
                )
                parameter["main_horizontal_piece_size"][1] = (
                    seat_size[1] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_horizontal_piece_size"][2] = (
                    seat_size[1] * randRange(1, 0.5, 1.0)[0]
                )

                parameter["sub_horizontal_offset"][0] = (
                    -parameter["main_vertical_piece_size"][1]
                    / 2
                    * randRange(1, 0.8, 1.0)[0]
                )

                parameter["sub_horizontal_piece_size"][0] = (
                    parameter["main_vertical_piece_size"][1]
                    / 10
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_horizontal_piece_size"][1] = (
                    parameter["main_vertical_piece_size"][2] * randRange(1, 0.8, 1.0)[0]
                )

                parameter["interval_between_subs"][0] = (
                    seat_size[0] - parameter["main_vertical_piece_size"][0]
                ) / (number_of_subs + 1)

                parameter["sub_vertical_offset"][0] = -(
                    (seat_size[0] - parameter["main_vertical_piece_size"][0]) / 2
                    - parameter["interval_between_subs"][0]
                )

                parameter["sub_vertical_piece_size"][0] = (
                    parameter["interval_between_subs"][0]
                    / 4
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_vertical_piece_size"][1] = (
                    parameter["main_vertical_piece_size"][2] * randRange(1, 0.8, 1.0)[0]
                )

                parameter["position"][2] -= (
                    seat_size[2] / 2 - parameter["main_vertical_piece_size"][2] / 2
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["number_of_subs"][0] = int(number_of_subs)
                new_concepts.append(concept)

            elif template == "Slat_back":
                back_type = "Slat_back"
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                total_seat_size_x = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                    total_seat_size_x = seat_size[0]
                elif seat_type == "Round":
                    seat_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(seat_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(seat_position_angle)
                    )
                    total_seat_size_x = seat_parameter["size"][0] * 2

                parameter["main_vertical_piece_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_vertical_piece_size"][1] = (
                    total_seat_size_x * randRange(1, 0.8, 1.2)[0]
                )
                if seat_type == "Regular":
                    parameter["main_vertical_separation"][0] = (
                        seat_size[0] - parameter["main_vertical_piece_size"][0]
                    )
                elif seat_type == "Round":
                    parameter["main_vertical_separation"][0] = seat_size[0]

                parameter["main_vertical_piece_size"][2] = (
                    seat_size[1] * randRange(1, 0.5, 1.0)[0]
                )
                parameter["sub_horizontal_offset"][0] = (
                    parameter["main_vertical_piece_size"][1]
                    / 2
                    * randRange(1, 0.6, 0.9)[0]
                )
                number_of_subs = int(np.random.randint(2, 6))
                parameter["number_of_subs"][0] = int(number_of_subs)
                parameter["interval_between_subs"][0] = (
                    parameter["main_vertical_piece_size"][1] / 2
                    + parameter["sub_horizontal_offset"][0]
                ) / (parameter["number_of_subs"][0] + 1)
                parameter["sub_horizontal_piece_size"][1] = (
                    parameter["main_vertical_piece_size"][2] * randRange(1, 0.8, 1.0)[0]
                )
                parameter["sub_horizontal_piece_size"][0] = (
                    parameter["interval_between_subs"][0]
                    / 2
                    * randRange(1, 0.5, 0.7)[0]
                )
                parameter["back_rotation"][0] *= randRange(1, 0.8, 1.5)[0]
                parameter["position"][2] -= (
                    seat_size[2] / 2 - parameter["main_vertical_piece_size"][2] / 2
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                concept["parameters"]["number_of_subs"][0] = int(number_of_subs)
                new_concepts.append(concept)

            elif template == "C_shaped_office_leg":
                seat_parameter = concepts[0]["parameters"]
                back_parameter = concepts[1]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                elif seat_type == "Round":
                    seat_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(seat_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(seat_position_angle)
                    )
                if back_type == "Solid_back":
                    armrest_height = back_parameter["size"][1]
                else:
                    armrest_height = back_parameter["main_vertical_piece_size"][1]

                parameter["vertical_leg_size"][0] = (
                    seat_size[0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["vertical_leg_size"][1] = (
                    armrest_height * randRange(1, 0.8, 0.9)[0]
                )
                parameter["vertical_leg_size"][2] = np.maximum(
                    parameter["vertical_leg_size"][2] * randRange(1, 0.8, 1.2)[0],
                    seat_size[1] / 2 * randRange(1, 0.8, 1.2)[0],
                )

                parameter["vertical_leg_separation"][0] = (
                    seat_size[0] - parameter["vertical_leg_size"][0]
                ) * randRange(1, 0.8, 1.0)[0]
                parameter["horizontal_z_leg_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["horizontal_z_leg_size"][1] = (
                    seat_size[2] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["horizontal_x_leg_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["vertical_leg_rotation"][0] = -np.random.uniform(0, 30)
                parameter["vertical_leg_rotation"][1] = np.random.uniform(0, 30)
                parameter["horizontal_leg_rotation"][0] = 0
                parameter["horizontal_leg_rotation"][1] = np.random.uniform(0, 15)
                parameter["position"][2] += seat_size[2] / 2 * randRange(1, 0.8, 0.9)[0]
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Office_armrest":
                seat_parameter = concepts[0]["parameters"]
                parameter["position"][0] = seat_parameter["position"][0]
                parameter["position"][1] = seat_parameter["position"][1]
                parameter["position"][2] = seat_parameter["position"][2]
                parameter["rotation"][0] = seat_parameter["rotation"][0]
                parameter["rotation"][1] = seat_parameter["rotation"][1]
                parameter["rotation"][2] = seat_parameter["rotation"][2]
                seat_size = [0, 0, 0]
                horizontal_support_size_z = 0
                if seat_type == "Regular":
                    seat_size[0] = seat_parameter["size"][0]
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = seat_parameter["size"][2]
                elif seat_type == "Round":
                    seat_position_angle = np.random.uniform(50, 60) / 180 * np.pi
                    seat_size[0] = (
                        seat_parameter["size"][0] * 2 * np.cos(seat_position_angle)
                    )
                    seat_size[1] = seat_parameter["size"][1]
                    seat_size[2] = (
                        seat_parameter["size"][0] * 2 * np.sin(seat_position_angle)
                    )
                    horizontal_support_position_angle = (
                        np.random.uniform(40, 50) / 180 * np.pi
                    )
                    horizontal_support_size_z = seat_parameter["size"][0] * (
                        1 + np.sin(horizontal_support_position_angle)
                    )
                armrest_height = seat_size[0] / 3 * randRange(1, 0.8, 1.2)[0]
                rotation_sign = 1 if random.random() < 0.5 else -1
                parameter["vertical_support_rotation"][0] = np.random.uniform(-45, 45)
                vertical_support_angle = (
                    parameter["vertical_support_rotation"][0] / 180 * np.pi
                )
                if seat_type == "Regular":
                    parameter["horizontal_support_sizes"][0] = (
                        seat_size[0] / 8 * randRange(1, 0.8, 1.3)[0]
                    )
                    parameter["horizontal_support_sizes"][1] = (
                        parameter["horizontal_support_sizes"][0]
                        * 5
                        / 7
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["horizontal_support_sizes"][2] = (
                        seat_size[2] * randRange(1, 0.7, 0.9)[0]
                    )
                    parameter["vertical_support_sizes"][0] = (
                        parameter["horizontal_support_sizes"][0]
                        * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["vertical_support_sizes"][1] = (
                        armrest_height * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["vertical_support_sizes"][2] = (
                        parameter["horizontal_support_sizes"][0]
                        * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["armrest_rotation"] *= rotation_sign * randRange(
                        parameter["armrest_rotation"].shape[0], 0.8, 1.2
                    )
                    parameter["armrest_separation"][0] = (
                        seat_size[0] * randRange(1, 0.8, 1.0)[0]
                    )
                    range_of_vertical_support_offset_z = seat_parameter["size"][
                        2
                    ] / 2 + (
                        parameter["vertical_support_sizes"][1]
                        * np.sin(vertical_support_angle)
                        / 2
                    )

                    if range_of_vertical_support_offset_z < 0:
                        range_of_vertical_support_offset_z = 0
                        vertical_support_angle = np.arcsin(
                            -seat_parameter["size"][2]
                            / parameter["vertical_support_sizes"][1]
                        )
                        parameter["vertical_support_rotation"][0] = (
                            vertical_support_angle * 180 / np.pi
                        )
                    if vertical_support_angle > 0:
                        range_of_horizontal_arm_length = (
                            parameter["horizontal_support_sizes"][2] / 2
                            - parameter["vertical_support_sizes"][1]
                            * np.sin(vertical_support_angle)
                            / 2
                        )
                        range_of_vertical_support_offset_z = np.minimum(
                            range_of_vertical_support_offset_z,
                            range_of_horizontal_arm_length,
                        )
                    parameter["supports_contact_offset"][0] = (
                        range_of_vertical_support_offset_z * randRange(1, 0.5, 1.0)[0]
                    )
                    if range_of_vertical_support_offset_z < 0:
                        parameter["supports_contact_offset"][0] = (
                            range_of_vertical_support_offset_z
                            * randRange(1, 1.0, 1.2)[0]
                        )

                elif seat_type == "Round":
                    parameter["horizontal_support_sizes"][0] = (
                        seat_size[0] / 8 * randRange(1, 0.8, 1.3)[0]
                    )
                    parameter["horizontal_support_sizes"][1] = (
                        parameter["horizontal_support_sizes"][0]
                        * 5
                        / 7
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["horizontal_support_sizes"][2] = (
                        horizontal_support_size_z * randRange(1, 0.7, 0.9)[0]
                    )
                    parameter["vertical_support_sizes"][0] = (
                        parameter["horizontal_support_sizes"][0]
                        * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["vertical_support_sizes"][1] = (
                        armrest_height * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["vertical_support_sizes"][2] = (
                        parameter["horizontal_support_sizes"][0]
                        * randRange(1, 0.8, 1.0)[0]
                    )
                    parameter["armrest_rotation"] *= randRange(
                        parameter["armrest_rotation"].shape[0], 0.8, 1.2
                    )
                    armrest_position_angle = np.random.uniform(10, 30) / 180 * np.pi
                    armrest_position_x = (
                        seat_parameter["size"][0] * 2 * np.cos(armrest_position_angle)
                    )
                    armrest_position_z = seat_parameter["size"][0] * np.sin(
                        armrest_position_angle
                    )
                    parameter["armrest_separation"][0] = armrest_position_x
                    range_of_vertical_support_offset_z = armrest_position_z + (
                        parameter["vertical_support_sizes"][1]
                        * np.sin(vertical_support_angle)
                        / 2
                    )
                    if vertical_support_angle > 0:
                        range_of_horizontal_arm_length = (
                            parameter["horizontal_support_sizes"][2] / 2
                            - parameter["vertical_support_sizes"][1]
                            * np.sin(vertical_support_angle)
                            / 2
                        )
                        range_of_vertical_support_offset_z = np.minimum(
                            range_of_vertical_support_offset_z,
                            range_of_horizontal_arm_length,
                        )
                    parameter["supports_contact_offset"][0] = (
                        range_of_vertical_support_offset_z * randRange(1, 0.5, 1.0)[0]
                    )
                    if range_of_vertical_support_offset_z < 0:
                        parameter["supports_contact_offset"][0] = (
                            range_of_vertical_support_offset_z
                            * randRange(1, 1.0, 1.2)[0]
                        )
                parameter["horizontal_support_rotation"] *= randRange(1, 0.8, 1.2)[0]
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
        chair_type = get_chair_type()
        existing_concept_templates = concept_template_existence(chair_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, chair_type)

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
            chair_type = get_chair_type()
            existing_concept_templates = concept_template_existence(chair_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, chair_type)
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
