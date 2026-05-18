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


def get_faucet_type():
    faucet_type = ["type1_faucet", "type2_faucet", "type3_faucet", "type4_faucet"]
    weights = [1, 1, 1, 1]
    faucet_type = random.choices(faucet_type, weights=weights, k=1)[0]
    return faucet_type


def concept_template_existence(faucet_type):
    if faucet_type == "type1_faucet":
        base_template = ["Cuboidal_Base", "Cylindrical_Base"]
        spout_template = ["Curved_Spout", "Trifold_Spout", "Quadfold_Spout"]
        switch_template = ["HandleY_Switch", "RotaryY_Switch"]
        necessary = [1, 1, 1]
    elif faucet_type == "type2_faucet":
        base_template = ["Cylindrical_Base"]
        spout_template = ["Cylindrical_Spout", "Cuboidal_Spout"]
        switch_template = ["RegularY_Switch", "Lever_Switch"]
        necessary = [1, 1, 1]
    elif faucet_type == "type3_faucet":
        base_template = ["UShapedXZ_Base"]
        spout_template = ["Curved_Spout"]
        switch_template = ["RotaryZ_Switch"]
        necessary = [1, 1, 1]
    elif faucet_type == "type4_faucet":
        base_template = ["Cuboidal_Base", "Cylindrical_Base"]
        spout_template = ["Curved_Spout"]
        switch_template = ["RotaryX_Switch"]
        necessary = [1, 1, 1]

    concept_template_variation = {
        "base": {"template": base_template, "necessary": necessary[0]},
        "spout": {"template": spout_template, "necessary": necessary[1]},
        "switch": {"template": switch_template, "necessary": necessary[2]},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"] == 1:
            templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, faucet_type):
    new_concepts = []

    base_type = ""
    spout_type = concepts[1]["template"]
    faucet_total_height = 0
    main_base_height = 0

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if faucet_type == "type1_faucet":
            if template == "Cuboidal_Base":
                base_type = "Cuboidal_Base"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size_1"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size_1"][1] = (
                    parameter["size_1"][0] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_1"][2] = (
                    parameter["size_1"][0] * randRange(1, 0.8, 1.2)[0]
                )
                faucet_total_height = (
                    parameter["size_1"][0] * 4.0 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_0"][0] = (
                    parameter["size_1"][0] / 2 * randRange(1, 0.7, 1.0)[0]
                )
                parameter["size_0"][1] = (
                    parameter["size_1"][0] * 2.4 * randRange(1, 0.8, 1.2)[0]
                )
                main_base_height = parameter["size_0"][1]
                if random.random() < 0.2 or spout_type == "Quadfold_Spout":
                    parameter["size_0"][1] = (
                        parameter["size_1"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                parameter["size_0"][2] = (
                    parameter["size_0"][0] * randRange(1, 1.0, 1.1)[0]
                )

                parameter["offset_1"][0] = 0
                parameter["offset_1"][1] = (
                    parameter["size_1"][1] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["offset_1"][2] = (
                    (parameter["size_1"][2] - parameter["size_0"][2])
                    / 2
                    * randRange(1, 0.0, 1.0)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["number_of_box"] = np.array([int(2)])
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cylindrical_Base":
                base_type = "Cylindrical_Base"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size_1"][0] *= randRange(1, 0.8, 1.2)[0]
                faucet_total_height = (
                    parameter["size_1"][0] * 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_1"][1] = (
                    parameter["size_1"][0] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_0"][1] = (
                    parameter["size_1"][0] * 2.44 * randRange(1, 0.8, 1.2)[0]
                )
                main_base_height = parameter["size_0"][1] * 1.5
                if random.random() < 0.2 or spout_type == "Quadfold_Spout":
                    parameter["size_0"][1] = (
                        parameter["size_1"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                parameter["size_0"][0] = (
                    parameter["size_1"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["offset_1"][0] = 0
                parameter["offset_1"][1] = (
                    parameter["size_1"][1] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["offset_1"][2] = (
                    parameter["size_0"][0]
                    / 3
                    * randRange(1, 0.8, 1.0)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["number_of_cylinder"] = np.array([int(2)])

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Curved_Spout":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = base_parameter["position"][1]
                parameter["position"][2] = base_parameter["position"][2]
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]
                parameter["R"][0] = (
                    base_parameter["size_0"][0] / 2
                    if base_type == "Cuboidal_Base"
                    else base_parameter["size_0"][0]
                ) * randRange(1, 0.95, 1.05)[0]
                parameter["L"][0] *= randRange(1, 0.8, 1.2)[0]
                spout_offset_z = -(
                    base_parameter["size_0"][2] / 2
                    if base_type == "Cuboidal_Base"
                    else base_parameter["size_0"][0]
                )
                parameter["position"][2] += spout_offset_z
                origin_spout_height = faucet_total_height - base_parameter["size_0"][1]
                parameter["bottom0"][0] = 0
                parameter["bottom0"][2] = 0
                parameter["bottom0"][1] = 0
                parameter["bottom1"][0] = parameter["bottom0"][0]
                parameter["bottom1"][1] = (
                    origin_spout_height * randRange(1, 0.8, 1.2)[0]
                    + parameter["bottom0"][1]
                )
                parameter["bottom1"][2] = parameter["bottom0"][2]
                parameter["center"][0] = parameter["bottom0"][0]
                parameter["center"][1] = parameter["bottom0"][1]
                parameter["center"][2] = main_base_height * randRange(1, 0.4, 0.6)[0]

                parameter["spout_rotation"][0] = 90 + (
                    45 * randRange(1, 0.9, 1.4)[0]
                    if random.random() < 0.8
                    else -15 * randRange(1, 0.8, 1.2)[0]
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Trifold_Spout":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = base_parameter["position"][1]
                parameter["position"][2] = base_parameter["position"][2]
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                parameter["R"][0] = (
                    base_parameter["size_0"][0] / 2
                    if base_type == "Cuboidal_Base"
                    else base_parameter["size_0"][0]
                ) * randRange(1, 0.95, 1.05)[0]
                spout_offset_z = -(
                    base_parameter["size_0"][2] / 2
                    if base_type == "Cuboidal_Base"
                    else base_parameter["size_0"][0]
                )
                parameter["position"][2] += spout_offset_z

                parameter["position0"][0] = 0
                parameter["position0"][1] = 0
                parameter["position0"][2] = 0
                parameter["position1"][0] = 0
                parameter["position1"][1] = (
                    faucet_total_height - base_parameter["size_0"][1]
                )
                parameter["position1"][2] = (
                    0.01
                    * randRange(1, 0.6, 1.0)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["position2"][0] = 0
                parameter["position2"][1] = main_base_height * randRange(1, 0.0, 0.3)[0]
                parameter["position2"][2] = (
                    main_base_height * 2 * randRange(1, 0.5, 0.8)[0]
                )
                parameter["position3"][0] = 0
                parameter["position3"][1] = (
                    -main_base_height * randRange(1, 0.4, 0.9)[0]
                )
                parameter["position3"][2] = main_base_height * randRange(1, 0.0, 0.2)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Quadfold_Spout":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = base_parameter["position"][1]
                parameter["position"][2] = base_parameter["position"][2]
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                parameter["R"][0] = (
                    base_parameter["size_0"][0] / 2
                    if base_type == "Cuboidal_Base"
                    else base_parameter["size_0"][0]
                ) * randRange(1, 0.95, 1.05)[0]
                spout_offset_z = -(
                    base_parameter["size_0"][2] / 2
                    if base_type == "Cuboidal_Base"
                    else base_parameter["size_0"][0]
                )
                parameter["position"][2] += spout_offset_z

                parameter["position0"][0] = 0
                parameter["position0"][1] = 0
                parameter["position0"][2] = 0
                parameter["position1"][0] = 0
                parameter["position1"][1] = (
                    base_parameter["size_1"][0] * randRange(1, 1.0, 1.2)[0]
                )
                parameter["position1"][2] = (
                    -0.01
                    * randRange(1, 0.6, 1.0)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["position2"][0] = 0
                parameter["position2"][1] = (
                    parameter["position1"][1] * 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position2"][2] = (
                    -parameter["position2"][1] * 37 / 54 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position3"][0] = 0
                parameter["position3"][1] = (
                    parameter["position2"][1] * 31 / 54 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position3"][2] = 0
                parameter["position4"][0] = 0
                parameter["position4"][1] = (
                    parameter["position3"][1] / 3 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position4"][2] = (
                    -parameter["position2"][2] * 17 / 37 * randRange(1, 0.8, 1.2)[0]
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "HandleY_Switch":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = (
                    base_parameter["position"][1]
                    - base_parameter["size_0"][1]
                    - base_parameter["size_1"][1]
                    + base_parameter["offset_1"][1]
                )
                parameter["position"][2] = base_parameter["position"][2]
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                num_of_switch = np.random.randint(1, 3)
                parameter["number_of_switch"] = np.array([int(num_of_switch)])
                parameter["number_of_cube"] = np.array([int(3)])

                parameter["bottom_size"][0] = (
                    base_parameter["size_1"][0] * 3 / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bottom_size"][1] = (
                    base_parameter["size_1"][1] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["bottom_size"][2] = (
                    parameter["bottom_size"][0] * 15 / 14 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["middle_size"][0] = (
                    parameter["bottom_size"][0] * 2 / 7 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["middle_size"][1] = (
                    parameter["middle_size"][0] * 6.5 * randRange(1, 0.8, 1.0)[0]
                )
                parameter["middle_size"][2] = (
                    parameter["middle_size"][0] * 9 / 8 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["top_size"][0] = (
                    parameter["bottom_size"][0] * 3 / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["top_size"][1] = (
                    parameter["bottom_size"][1] * 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["top_size"][2] = (
                    parameter["top_size"][0] / 4 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["bottom_middle_offset"][0] = (
                    (parameter["bottom_size"][0] - parameter["middle_size"][0])
                    / 2
                    * randRange(1, 0.0, 0.9)[0]
                )
                parameter["bottom_middle_offset"][1] = (
                    (parameter["bottom_size"][2] - parameter["middle_size"][2])
                    / 2
                    * randRange(1, 0.0, 0.9)[0]
                )

                parameter["bottom_top_offset"][0] = (
                    (
                        (parameter["top_size"][0] - parameter["middle_size"][0])
                        / 2
                        * randRange(1, 0.0, 0.9)[0]
                    )
                    * (-1 if random.random() < 0.5 else 1)
                ) + parameter["bottom_middle_offset"][0]
                parameter["bottom_top_offset"][1] = (
                    -(
                        (parameter["top_size"][2] - parameter["middle_size"][2])
                        / 2
                        * randRange(1, 0.8, 0.9)[0]
                    )
                    + parameter["bottom_middle_offset"][1]
                )
                parameter["offset_0_x"][0] = (
                    -base_parameter["size_1"][0] * 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["offset_1_x"][0] = -parameter["offset_0_x"][0]
                parameter["offset_yz"][0] = (
                    0.005
                    * randRange(1, 0.8, 1.2)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["offset_yz"][1] = (
                    parameter["bottom_size"][2] / 6 * randRange(1, 0.8, 1.2)[0]
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "RotaryY_Switch":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = (
                    base_parameter["position"][1]
                    - base_parameter["size_0"][1]
                    - base_parameter["size_1"][1]
                    + base_parameter["offset_1"][1]
                )
                parameter["position"][2] = base_parameter["position"][2]
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                num_of_switch = np.random.randint(1, 3)
                number_of_sub = np.random.randint(1, 5)
                parameter["number_of_switch"] = np.array([int(num_of_switch)])
                parameter["number_of_sub"] = np.array([int(number_of_sub)])
                base_size = [
                    base_parameter["size_1"][0]
                    * (1 / 2 if base_type == "Cuboidal_Base" else 1),
                    base_parameter["size_1"][1],
                ]
                parameter["main_size_1"][0] = base_size[0] * randRange(1, 0.8, 1.2)[0]
                parameter["main_size_1"][1] = base_size[1] * randRange(1, 0.8, 1.2)[0]
                parameter["main_size_2"][0] = (
                    parameter["main_size_1"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_size_2"][1] = (
                    parameter["main_size_1"][1] * 8 * randRange(1, 0.8, 1.2)[0]
                )
                if number_of_sub < 3:
                    parameter["sub_size"][0] = (
                        parameter["main_size_2"][0] * randRange(1, 1.05, 1.1)[0]
                    )
                    parameter["sub_size"][1] = (
                        parameter["sub_size"][0] * 17 / 16 * randRange(1, 0.9, 1.1)[0]
                    )
                    parameter["sub_size"][2] = (
                        parameter["sub_size"][0] * 5 * randRange(1, 0.8, 1.0)[0]
                    )
                elif number_of_sub > 2:
                    parameter["sub_size"][0] = (
                        parameter["main_size_2"][0] / 2 * randRange(1, 1.05, 1.1)[0]
                    )
                    parameter["sub_size"][1] = (
                        parameter["sub_size"][0] * 17 / 16 * randRange(1, 0.9, 1.1)[0]
                    )
                    parameter["sub_size"][2] = (
                        parameter["sub_size"][0] * 5 * randRange(1, 0.8, 1.0)[0]
                    )
                parameter["sub_offset"][0] = parameter["sub_size"][1]
                parameter["tilt_angle"][0] = (
                    0 if random.random() < 0.5 else 5 * randRange(1, 0.0, 0.8)[0]
                )
                parameter["rotation0"][0] = -np.random.uniform(0, 180)
                parameter["rotation1"][0] = np.random.uniform(0, 180)
                parameter["offset_x"][0] = -base_size[0] * 4 * randRange(1, 0.8, 1.2)[0]
                parameter["offset_x"][1] = -parameter["offset_x"][0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif faucet_type == "type2_faucet":
            if template == "Cylindrical_Base":
                base_type = "Cylindrical_Base"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size_1"][0] *= randRange(1, 0.8, 1.2)[0]
                faucet_total_height = (
                    parameter["size_1"][0] * 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_1"][1] = (
                    parameter["size_1"][0] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_0"][1] = (
                    parameter["size_1"][0] * 5 * randRange(1, 0.6, 1.2)[0]
                )
                main_base_height = parameter["size_0"][1] / 2
                parameter["size_0"][0] = (
                    parameter["size_1"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["offset_1"][0] = 0
                parameter["offset_1"][1] = (
                    parameter["size_1"][1] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["offset_1"][2] = (
                    parameter["size_0"][0]
                    / 3
                    * randRange(1, 0.8, 1.0)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["number_of_cylinder"] = np.array([int(2)])

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cylindrical_Spout":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = (
                    base_parameter["position"][1]
                    - base_parameter["size_0"][1] / 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position"][2] = base_parameter["position"][2]
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                parameter["main_part_size"][0] = (
                    base_parameter["size_0"][0] * randRange(1, 0.7, 0.9)[0]
                )
                parameter["main_part_size"][1] = (
                    base_parameter["size_0"][1] * 0.7 * randRange(1, 0.8, 1.3)[0]
                )
                parameter["head_size"][0] = (
                    parameter["main_part_size"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["head_size"][1] = (
                    parameter["head_size"][0] * randRange(1, 0.8, 1.0)[0]
                )
                parameter["head_offset"][0] = 0
                parameter["head_offset"][1] = 0
                parameter["head_offset"][2] = 0
                parameter["rotation_mainpart"][0] = (
                    10
                    * randRange(1, 0.8, 1.2)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["rotation_head"][0] = 0

                parameter["position"][1] -= parameter["main_part_size"][0]
                parameter["position"][2] -= (
                    parameter["main_part_size"][0]
                    * 2
                    * np.sin(np.abs(parameter["rotation_mainpart"][0] / 180 * np.pi))
                    * randRange(1, 1.1, 1.15)[0]
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cuboidal_Spout":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = (
                    base_parameter["position"][1]
                    - base_parameter["size_0"][1] / 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position"][2] = base_parameter["position"][2]
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                parameter["main_part_size"][0] = (
                    base_parameter["size_0"][0] * randRange(1, 0.9, 1.2)[0]
                )
                parameter["main_part_size"][1] = (
                    parameter["main_part_size"][0] * 18 / 23 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_part_size"][2] = (
                    base_parameter["size_0"][1] * 0.7 * randRange(1, 0.8, 1.4)[0]
                )
                parameter["head_size"][0] = (
                    parameter["main_part_size"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["head_size"][1] = (
                    parameter["head_size"][0] * randRange(1, 0.8, 1.0)[0]
                )
                parameter["head_offset"][0] = 0
                parameter["head_offset"][1] = 0
                parameter["head_offset"][2] = (
                    -parameter["main_part_size"][1] / 2 * randRange(1, 0.0, 0.2)[0]
                )
                parameter["rotation_mainpart"][0] = (
                    10
                    * randRange(1, 0.8, 1.2)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["rotation_head"][0] = 0

                parameter["position"][1] -= parameter["main_part_size"][0]
                parameter["position"][2] -= (
                    parameter["main_part_size"][0]
                    * 2
                    * np.sin(np.abs(parameter["rotation_mainpart"][0] / 180 * np.pi))
                    * randRange(1, 1.1, 1.15)[0]
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "RegularY_Switch":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = base_parameter["position"][1]
                parameter["position"][2] = (
                    base_parameter["position"][2] - base_parameter["size_0"][0]
                )
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                parameter["size"][0] = (
                    base_parameter["size_0"][0] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["size"][1] = (
                    parameter["size"][0] * 5 / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_offset"][0] = (
                    parameter["size"][1] * randRange(1, 0.0, 0.15)[0]
                )
                parameter["sub_offset"][0] = 0
                parameter["sub_offset"][0] = 0
                parameter["sub_size"][0] = (
                    parameter["size"][1] / 6 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][1] = (
                    parameter["size"][0] * 2 * randRange(1, 1.0, 1.2)[0]
                )
                parameter["rotation_Y"][0] = np.random.uniform(-90, 90)
                parameter["rotation_X"][0] = 0
                parameter["sub_rotation"][0] = 5 * randRange(1, 0.8, 1.2)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Lever_Switch":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = base_parameter["position"][1]
                parameter["position"][2] = (
                    base_parameter["position"][2] - base_parameter["size_0"][0]
                )
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                parameter["size"][0] = (
                    base_parameter["size_0"][0] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["size"][1] = (
                    parameter["size"][0] * 5 / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["R"][0] = parameter["size"][0] / 3 * randRange(1, 1.0, 1.2)[0]
                parameter["position0"][0] = 0
                parameter["position0"][1] = (
                    -parameter["R"][0] * 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position0"][2] = (
                    -parameter["size"][0] * randRange(1, 1.0, 1.2)[0]
                )
                parameter["position1"][0] = 0
                parameter["position1"][1] = (
                    parameter["R"][0] * 2.5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position1"][2] = (
                    parameter["position1"][1] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["position2"][0] = 0
                parameter["position2"][1] = -parameter["position0"][1]
                parameter["position2"][2] = (
                    parameter["position2"][1] * 37 / 16 * randRange(1, 0.8, 1.2)[0]
                )
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif faucet_type == "type3_faucet":
            if template == "UShapedXZ_Base":
                base_type = "UShapedXZ_Base"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["R"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size_tube"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size_tube"][1] = (
                    parameter["size_tube"][0] / 6.5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_base"][1] = (
                    parameter["size_tube"][1] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_base"][0] = (
                    parameter["size_base"][1] * 10 / 13 * randRange(1, 0.8, 1.2)[0]
                )

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Curved_Spout":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = base_parameter["position"][1]
                parameter["position"][2] = (
                    base_parameter["position"][2]
                    + base_parameter["size_base"][1]
                    + base_parameter["size_tube"][1]
                )
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                parameter["R"][0] = base_parameter["R"][0] * randRange(1, 0.8, 1.1)[0]
                parameter["L"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["position"][2] -= (
                    (base_parameter["R"][0] - parameter["R"][0])
                    / 2
                    * randRange(1, 0.0, 1.0)[0]
                )
                origin_spout_height = (
                    base_parameter["size_tube"][0] * 0.6345 * randRange(1, 0.8, 1.2)[0]
                )
                main_base_height = (
                    origin_spout_height * 1.06 * randRange(1, 0.8, 1.2)[0]
                )
                origin_offset_z = parameter["R"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                parameter["bottom0"][0] = 0
                parameter["bottom0"][2] = 0
                parameter["bottom0"][1] = 0
                parameter["bottom1"][0] = parameter["bottom0"][0]
                parameter["bottom1"][1] = (
                    origin_spout_height * randRange(1, 0.8, 1.2)[0]
                    + parameter["bottom0"][1]
                )
                parameter["bottom1"][2] = (
                    parameter["bottom0"][2]
                    + origin_offset_z * randRange(1, 0.0, 0.5)[0]
                )
                parameter["center"][0] = parameter["bottom0"][0]
                parameter["center"][1] = parameter["bottom0"][1]
                parameter["center"][2] = main_base_height * randRange(1, 0.8, 1.0)[0]

                parameter["spout_rotation"][0] = 90 + 45 * randRange(1, 1.0, 1.4)[0]

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "RotaryZ_Switch":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = base_parameter["position"][1]
                parameter["position"][2] = (
                    base_parameter["position"][2]
                    + base_parameter["size_base"][1]
                    + base_parameter["size_tube"][1]
                    + base_parameter["R"][0]
                )
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                parameter["number_of_switch"] = np.array([1])
                num_of_switch = 1
                parameter["main_size_1"][0] = (
                    base_parameter["R"][0] * 1.5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_size_1"][1] = (
                    base_parameter["size_tube"][1] * 2.4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_size_2"][0] = (
                    parameter["main_size_1"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_size_2"][1] = (
                    parameter["main_size_2"][0] * 3 / 11 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][0] = (
                    parameter["main_size_1"][1] * 42 / 35 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][1] = (
                    parameter["main_size_2"][1] * 6 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][2] = (
                    parameter["main_size_2"][0] * 4 / 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["number_of_sub"] = np.array([1])
                parameter["sub_offset"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["tilt_angle"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["rotation0"][0] = np.random.uniform(-180, 180)
                parameter["rotation1"][0] = np.random.uniform(-180, 180)
                parameter["offset_x"][0] = -(
                    base_parameter["size_tube"][0] / 2 - parameter["main_size_1"][0]
                )
                parameter["offset_x"][1] = -parameter["offset_x"][0]
                if num_of_switch == 1:
                    parameter["offset_x"][0] = 0
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

        elif faucet_type == "type4_faucet":
            if template == "Cuboidal_Base":
                base_type = "Cuboidal_Base"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size_1"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["size_1"][1] = (
                    parameter["size_1"][0] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_1"][2] = (
                    parameter["size_1"][0] * randRange(1, 0.8, 1.2)[0]
                )
                faucet_total_height = (
                    parameter["size_1"][0] * 3 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_0"][0] = (
                    parameter["size_1"][0] / 2 * randRange(1, 0.7, 1.0)[0]
                )
                parameter["size_0"][1] = (
                    parameter["size_1"][0] * 2 * randRange(1, 0.8, 1.2)[0]
                )
                main_base_height = parameter["size_0"][1]
                parameter["size_0"][2] = (
                    parameter["size_0"][0] * randRange(1, 1.0, 1.1)[0]
                )

                parameter["offset_1"][0] = 0
                parameter["offset_1"][1] = (
                    parameter["size_1"][1] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["offset_1"][2] = (
                    (parameter["size_1"][2] - parameter["size_0"][2])
                    / 2
                    * randRange(1, 0.0, 1.0)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["number_of_box"] = np.array([int(2)])
                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Cylindrical_Base":
                base_type = "Cylindrical_Base"
                parameter["position"][0] = 0
                parameter["position"][1] = 0
                parameter["position"][2] = 0
                parameter["rotation"][0] = 0
                parameter["rotation"][1] = 0
                parameter["rotation"][2] = 0
                parameter["size_1"][0] *= randRange(1, 0.8, 1.2)[0]
                faucet_total_height = (
                    parameter["size_1"][0] * 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_1"][1] = (
                    parameter["size_1"][0] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size_0"][1] = (
                    parameter["size_1"][0] * 2.44 * randRange(1, 0.8, 1.2)[0]
                )
                main_base_height = parameter["size_0"][1] * 1.5
                parameter["size_0"][0] = (
                    parameter["size_1"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["offset_1"][0] = 0
                parameter["offset_1"][1] = (
                    parameter["size_1"][1] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["offset_1"][2] = (
                    parameter["size_0"][0]
                    / 3
                    * randRange(1, 0.8, 1.0)[0]
                    * (-1 if random.random() < 0.5 else 1)
                )
                parameter["number_of_cylinder"] = np.array([int(2)])

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "Curved_Spout":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = base_parameter["position"][1]
                parameter["position"][2] = base_parameter["position"][2]
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]
                parameter["R"][0] = (
                    base_parameter["size_0"][0] / 2
                    if base_type == "Cuboidal_Base"
                    else base_parameter["size_0"][0]
                ) * randRange(1, 0.95, 1.05)[0]
                parameter["L"][0] *= randRange(1, 0.8, 1.2)[0]
                spout_offset_z = -(
                    base_parameter["size_0"][2] / 2
                    if base_type == "Cuboidal_Base"
                    else base_parameter["size_0"][0]
                )
                parameter["position"][2] += spout_offset_z
                origin_spout_height = faucet_total_height - base_parameter["size_0"][1]
                parameter["bottom0"][0] = 0
                parameter["bottom0"][2] = 0
                parameter["bottom0"][1] = 0
                parameter["bottom1"][0] = parameter["bottom0"][0]
                parameter["bottom1"][1] = (
                    origin_spout_height * randRange(1, 0.8, 1.2)[0]
                    + parameter["bottom0"][1]
                )
                parameter["bottom1"][2] = parameter["bottom0"][2]
                parameter["center"][0] = parameter["bottom0"][0]
                parameter["center"][1] = parameter["bottom0"][1]
                parameter["center"][2] = main_base_height * randRange(1, 0.4, 0.6)[0]

                parameter["spout_rotation"][0] = 90 + (45 * randRange(1, 1.1, 1.4)[0])

                for k, v in parameter.items():
                    if isinstance(v, list):
                        parameter[k] = np.array(v)
                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)

            elif template == "RotaryX_Switch":
                base_parameter = new_concepts[0]["parameters"]
                parameter["position"][0] = base_parameter["position"][0]
                parameter["position"][1] = base_parameter["position"][1]
                parameter["position"][2] = base_parameter["position"][2]
                parameter["rotation"][0] = base_parameter["rotation"][0]
                parameter["rotation"][1] = base_parameter["rotation"][1]
                parameter["rotation"][2] = base_parameter["rotation"][2]

                base_size = [0, 0]
                base_size[0] = (
                    base_parameter["size_0"][0]
                    if base_type == "Cylindrical_Base"
                    else np.minimum(
                        base_parameter["size_0"][0], base_parameter["size_0"][2]
                    )
                    / 2
                )
                base_size[1] = (
                    base_parameter["size_0"][1]
                    if base_type == "Cylindrical_Base"
                    else base_parameter["size_0"][1] / 2
                )

                parameter["position"][2] -= base_size[0]
                exist_left = 0 if random.random() < 0.5 else 1
                parameter["existence_of_switch"] = np.array([int(exist_left), int(1)])
                parameter["main_size_1"][0] = base_size[0] * randRange(1, 0.8, 1.2)[0]
                parameter["main_size_1"][1] = (
                    base_size[1] * 3 / 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_size_2"][0] = (
                    parameter["main_size_1"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["main_size_2"][1] = (
                    parameter["main_size_1"][1] / 2.5 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["position"][1] -= (
                    parameter["main_size_1"][0] * randRange(1, 0.0, 1.0)[0]
                )
                parameter["position"][1] -= (
                    base_parameter["size_0"][1] * randRange(1, 0.5, 0.7)[0]
                )
                parameter["position"][1] = np.maximum(
                    -base_parameter["size_0"][1] + parameter["main_size_1"][0],
                    parameter["position"][1],
                )

                parameter["sub_size"][0] = (
                    parameter["main_size_2"][0] * 13 / 11 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][1] = (
                    parameter["main_size_2"][1] * 22 / 13 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["sub_size"][2] = (
                    parameter["sub_size"][0] * randRange(1, 0.8, 1.2)[0]
                )

                number_of_sub = random.randint(1, 5)
                parameter["number_of_sub"] = np.array([int(number_of_sub)])
                parameter["sub_offset"][0] = (
                    -parameter["main_size_2"][1] / 10 * randRange(1, 0.8, 1.2)[0]
                )

                parameter["tilt_angle"][0] = 0
                parameter["rotation0"][0] = np.random.uniform(-180, 180)
                parameter["rotation1"][0] = np.random.uniform(-180, 180)
                parameter["interval"] = -base_size[0] * randRange(1, 0.0, 0.0)[0]
                parameter["offset_x"][0] = (
                    -parameter["main_size_1"][1] / 2 * randRange(1, 0.0, 0.5)[0]
                )
                parameter["offset_x"][1] = -parameter["offset_x"][0]

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
        faucet_type = get_faucet_type()
        existing_concept_templates = concept_template_existence(faucet_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, faucet_type)

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
            faucet_type = get_faucet_type()
            existing_concept_templates = concept_template_existence(faucet_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, faucet_type)
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
