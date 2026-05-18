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


def get_switch_type():
    total_type = [
        "Cuboid",
        "Lever",
        "Lever_and_knob",
        "Cuboid_and_knob",
        "Round",
        "Slip",
        "Knob",
    ]
    weights = [1, 1, 1, 1, 1, 1, 1]
    switch_type = random.choices(total_type, weights=weights, k=1)[0]

    return switch_type


def concept_template_existence(switch_type):
    if switch_type == "Cuboid":
        concept_template_variation = {
            "base": {"template": ["Standard_Base", "Frame_Base"], "necessary": True},
            "switch": {"template": ["FlipX_Switch", "FlipY_Switch"], "necessary": True},
        }
    if switch_type == "Lever":
        concept_template_variation = {
            "switch": {"template": ["Lever_Switch"], "necessary": True},
            "base": {"template": ["Standard_Base"], "necessary": True},
            "plug": {"template": ["Cuboidal_Plug"], "necessary": True},
        }
    if switch_type == "Lever_and_knob":
        concept_template_variation = {
            "switch": {"template": ["Lever_Switch"], "necessary": True},
            "base": {"template": ["Standard_Base"], "necessary": True},
            "knob": {"template": ["Standard_Knob"], "necessary": True},
        }
    if switch_type == "Cuboid_and_knob":
        concept_template_variation = {
            "switch": {"template": ["FlipX_Switch"], "necessary": True},
            "base": {"template": ["Standard_Base"], "necessary": True},
            "knob": {"template": ["Standard_Knob"], "necessary": True},
        }
    if switch_type == "Round":
        concept_template_variation = {
            "switch": {"template": ["Round_Switch"], "necessary": True},
            "base": {"template": ["Standard_Base"], "necessary": True},
            "plug": {
                "template": ["Cuboidal_Plug", "Cylindrical_Plug"],
                "necessary": True,
            },
        }
    if switch_type == "Slip":
        concept_template_variation = {
            "switch": {"template": ["FlipY_Switch"], "necessary": True},
            "base": {"template": ["Standard_Base"], "necessary": True},
            "plug": {
                "template": ["Cuboidal_Plug", "Cylindrical_Plug"],
                "necessary": True,
            },
        }
    elif switch_type == "Knob":
        concept_template_variation = {
            "base": {"template": ["Standard_Base"], "necessary": True},
            "knob": {"template": ["Standard_Knob"], "necessary": True},
            "plug": {"template": ["Cuboidal_Plug"], "necessary": True},
        }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, switch_type):
    new_concepts = []

    base_x: float = 0
    base_y: float = 0
    base_z: float = 0
    num_of_switch: int = 0
    switch_offset: float = 0

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Standard_Base":
            parameter["back_part_size"] = np.array([0.0, 0.0, 0.0])
            parameter["back_part_offset"] = np.array([0, 0])

            parameter["has_back_part"][0] = 0
            parameter["rotation"] = np.array([0, 0, 0])
            parameter["position"] = np.array([0, 0, 0])

            if switch_type == "Slip":
                parameter["size"][0] = base_x
                parameter["size"][1] = base_y
                parameter["size"][2] = base_z

            elif switch_type == "Lever":
                parameter["size"][0] = randRange(1, 0.72, 1)[0]
                parameter["size"][1] = parameter["size"][0]
                parameter["size"][2] = randRange(1, 0.08, 0.18)[0]
                if num_of_switch != 0:
                    parameter["size"][0] *= num_of_switch * 0.3 + 1

            elif switch_type == "Cuboid":
                parameter["size"][0] = randRange(1, 0.72, 1)[0]
                parameter["size"][1] = parameter["size"][0]
                parameter["size"][2] = randRange(1, 0.08, 0.12)[0]
                parameter["has_back_part"][0] = random.choice([0, 1])

            elif switch_type == "Round":
                parameter["size"][0] = base_x
                parameter["size"][1] = randRange(1, 0.74, 1.3)[0]
                parameter["size"][2] = base_z

            elif switch_type == "Knob":
                parameter["size"][0] = randRange(1, 0.72, 1)[0]
                parameter["size"][1] = parameter["size"][0]
                parameter["size"][2] = randRange(1, 0.08, 0.12)[0]

            elif switch_type == "Cuboid_and_knob" or switch_type == "Lever_and_knob":
                parameter["size"][0] = base_x
                parameter["size"][1] = base_y
                parameter["size"][2] = randRange(1, 0.08, 0.12)[0]
                parameter["has_back_part"][0] = random.choice([0, 1])

            if parameter["has_back_part"][0] == 1:
                parameter["back_part_size"][0] = (
                    parameter["size"][0] * randRange(1, 0.4, 0.6)[0]
                )
                parameter["back_part_size"][1] = parameter["back_part_size"][0]
                parameter["back_part_size"][2] = randRange(1, 0.06, 0.09)[0]

            base_x = parameter["size"][0]
            base_y = parameter["size"][1]
            base_z = parameter["size"][2] + parameter["back_part_size"][2]

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Frame_Base":
            parameter["size"][0] = randRange(1, 0.72, 0.9)[0]
            parameter["size"][1] = parameter["size"][0]

            if switch_type == "Cuboid_and_knob":
                parameter["size"][0] = randRange(1, 0.72, 0.9)[0] * 1.5

            parameter["size"][2] = randRange(1, 0.07, 0.08)[0]
            parameter["inner_size"][0] = parameter["size"][1] - 0.09
            parameter["inner_size"][1] = parameter["inner_size"][0]
            parameter["inner_outer_offset"] = np.array([0, 0])
            parameter["rotation"] = np.array([0, 0, 0])
            parameter["position"] = np.array([0, 0, 0])
            base_x = parameter["inner_size"][0]
            base_z = parameter["size"][2]

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Cuboidal_Plug":
            parameter["column_of_contact"][0] = random.choice(range(1, 3))
            parameter["row_of_contact"][0] = random.choice(range(2, 4))
            parameter["size"][0] = randRange(1, 0.09, 0.12)[0]
            parameter["size"][2] = randRange(1, 0.4, 0.6)[0]
            parameter["size"][1] = randRange(1, 0.03, 0.045)[0]

            if parameter["column_of_contact"][0] > 1:
                parameter["interval"][0] = (
                    base_x
                    * randRange(1, 0.5, 0.8)[0]
                    / (parameter["column_of_contact"][0] + 1)
                )
            else:
                parameter["interval"][0] = 0

            parameter["interval"][1] = (
                base_y
                / (parameter["row_of_contact"][0] + 1)
                * randRange(1, 0.7, 0.9)[0]
            )

            parameter["position"] = np.array(
                [
                    -(parameter["column_of_contact"][0] - 1)
                    * 0.5
                    * (parameter["interval"][0] + parameter["size"][0]),
                    -(parameter["row_of_contact"][0] - 1)
                    * 0.5
                    * (parameter["interval"][1] + parameter["size"][1]),
                    -base_z,
                ]
            )
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Cylindrical_Plug":
            parameter["column_of_contact"][0] = random.choice(range(1, 3))
            parameter["row_of_contact"][0] = random.choice(range(3, 4))

            parameter["size"][0] = randRange(1, 0.04, 0.06)[0]
            parameter["size"][1] = randRange(1, 0.4, 0.6)[0]
            if parameter["column_of_contact"][0] > 1:
                parameter["interval"][0] = (
                    base_x
                    * randRange(1, 0.5, 0.8)[0]
                    / (parameter["column_of_contact"][0] + 1)
                )
            else:
                parameter["interval"][0] = 0

            parameter["interval"][1] = (
                base_y
                / (parameter["row_of_contact"][0] + 1)
                * randRange(1, 0.7, 0.9)[0]
            )

            parameter["position"] = np.array(
                [
                    -(parameter["column_of_contact"][0] - 1)
                    * 0.5
                    * (parameter["interval"][0] + parameter["size"][0]),
                    -(parameter["row_of_contact"][0] - 1)
                    * 0.5
                    * (parameter["interval"][1] + 0.5 * parameter["size"][0]),
                    -base_z,
                ]
            )
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "FlipX_Switch":
            if switch_type == "Cuboid":
                parameter["number_of_switch"][0] = random.choice(range(1, 6))
                number_of_switch = parameter["number_of_switch"][0]
                parameter["separation"] = np.array([randRange(1, 0.02, 0.028)[0], 0, 0])
                parameter["size"][0] = (
                    base_x - parameter["separation"][0] * (number_of_switch - 1)
                ) / number_of_switch
                parameter["size"][2] = randRange(1, 0.06, 0.08)[0]
                if number_of_switch == 1:
                    parameter["position"] = np.array([0, 0, 0])
                    parameter["size"][2] = randRange(1, 0.08, 0.1)[0]
                else:
                    parameter["position"] = np.array(
                        [
                            -(number_of_switch - 1)
                            * 0.5
                            * (parameter["size"][0] + parameter["separation"][0]),
                            0,
                            0,
                        ]
                    )

                if random.random() < 0.5:
                    parameter["switch_rotation"][0] = (
                        np.arcsin(2 * parameter["size"][2] / base_x) * 90 / np.pi
                    )
                else:
                    parameter["switch_rotation"][0] = (
                        -np.arcsin(2 * parameter["size"][2] / base_x) * 90 / np.pi
                    )

                parameter["size"][1] = base_x * np.cos(
                    parameter["switch_rotation"][0] * np.pi / 180
                )
                parameter["rotation"] = np.array([0, 0, 0])

            elif switch_type == "Cuboid_and_knob":
                parameter["number_of_switch"][0] = 2
                base_x = randRange(1, 1.152, 1.5)[0]
                parameter["separation"] = np.array(
                    [base_x * randRange(1, 0.3, 0.45)[0], 0, 0]
                )

                parameter["size"][0] = (
                    (base_x - parameter["separation"][0])
                    / 2
                    * randRange(1, 0.45, 0.55)[0]
                )
                parameter["size"][2] = randRange(1, 0.06, 0.08)[0]
                parameter["position"] = np.array(
                    [
                        -parameter["separation"][0] * 0.5 - parameter["size"][0] * 0.5,
                        0,
                        0,
                    ]
                )

                if random.random() < 0.5:
                    parameter["switch_rotation"][0] = (
                        np.arcsin(2 * parameter["size"][2] / base_x) * 90 / np.pi
                    )
                else:
                    parameter["switch_rotation"][0] = (
                        -np.arcsin(2 * parameter["size"][2] / base_x) * 90 / np.pi
                    )

                base_y = randRange(1, 0.72, 1)[0]
                parameter["size"][1] = (
                    base_y
                    * np.cos(parameter["switch_rotation"][0] * np.pi / 180)
                    * randRange(1, 0.6, 0.8)[0]
                )

                parameter["rotation"] = np.array([0, 0, 0])

            elif switch_type == "Slip":
                parameter["number_of_switch"][0] = 1
                parameter["size"][1] = randRange(1, 0.06, 0.08)[0]
                parameter["size"][0] = randRange(1, 0.04, 0.06)[0]
                parameter["size"][2] = randRange(1, 0.1, 0.12)[0]
                parameter["separation"] = np.array([0, 0, 0])
                parameter["switch_rotation"][0] = 0

                parameter["rotation"] = np.array([0, 0, 0])
                parameter["position"] = np.array([0, 0, base_z])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "FlipY_Switch":
            if switch_type == "Cuboid":
                parameter["number_of_switch"][0] = random.choice(range(1, 5))
                number_of_switch = parameter["number_of_switch"][0]
                parameter["separation"] = np.array([randRange(1, 0.02, 0.028)[0], 0, 0])
                parameter["size"][0] = (
                    base_x - parameter["separation"][0] * (number_of_switch - 1)
                ) / number_of_switch
                parameter["size"][2] = randRange(1, 0.04, 0.06)[0]
                if number_of_switch == 1:
                    parameter["position"] = np.array([0, 0, 0])
                else:
                    parameter["position"] = np.array(
                        [
                            -(number_of_switch - 1)
                            * 0.5
                            * (parameter["size"][0] + parameter["separation"][0]),
                            0,
                            0,
                        ]
                    )

                if random.random() < 0.5:
                    parameter["switch_rotation"][0] = (
                        np.arcsin(parameter["size"][2] / parameter["size"][0])
                        * 180
                        / np.pi
                    )
                else:
                    parameter["switch_rotation"][0] = (
                        -np.arcsin(parameter["size"][2] / parameter["size"][0])
                        * 180
                        / np.pi
                    )

                parameter["size"][1] = base_x
                parameter["rotation"] = np.array([0, 0, 0])

            elif switch_type == "Slip":
                parameter["number_of_switch"][0] = random.choices(
                    [1, 2, 3, 4, 5, 6], weights=[3, 1, 1, 1, 1, 1], k=1
                )[0]
                num_of_switch = parameter["number_of_switch"][0]

                parameter["size"][0] = randRange(1, 0.1, 0.14)[0]
                parameter["size"][1] = randRange(1, 0.04, 0.06)[0]
                parameter["size"][2] = randRange(1, 0.24, 0.36)[0]

                parameter["separation"] = np.array(
                    [randRange(1, 0.15, 0.24)[0] * 4 / num_of_switch, 0, 0]
                )

                parameter["switch_rotation"][0] = 0

                base_y = randRange(1, 0.72, 1)[0]
                base_z = randRange(1, 0.12, 0.18)[0]
                parameter["position"] = np.array([0, 0, 0])
                parameter["rotation"] = np.array([0, 0, 0])

                base_x = randRange(1, 0.72, 1)[0]
                if num_of_switch == 1:
                    parameter["position"] = np.array(
                        [0, -base_y * randRange(1, 0.16, 0.3)[0], 0]
                    )
                    if random.random() < 0.5:
                        parameter["position"][1] *= -1
                    parameter["size"][1] *= 1.4
                    parameter["size"][2] = randRange(1, 0.2, 0.28)[0]
                else:
                    if num_of_switch >= 4:
                        base_x *= randRange(1, 1.6, 1.8)[0]
                        parameter["separation"] = np.array(
                            [randRange(1, 0.16, 0.23)[0] * 4 / num_of_switch, 0, 0]
                        )
                    else:
                        parameter["separation"] = np.array(
                            [randRange(1, 0.05, 0.08)[0] * 4 / num_of_switch, 0, 0]
                        )

                    parameter["size"][0] *= 0.8
                    parameter["size"][2] = randRange(1, 0.24, 0.36)[0]
                    parameter["position"] = np.array(
                        [
                            -(num_of_switch - 1)
                            * 0.5
                            * (parameter["separation"][0] + parameter["size"][0]),
                            -base_y * randRange(1, 0.1, 0.16)[0],
                            0,
                        ]
                    )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Lever_Switch":
            parameter["switch_rotation"][0] = randRange(1, 20, 30)[0] * random.choice(
                [-1, 1]
            )

            parameter["switch_rotation"][1] = parameter["switch_rotation"][
                0
            ] * random.choice([-1, 1])
            parameter["switch_rotation"][2] = parameter["switch_rotation"][
                0
            ] * random.choice([-1, 1])
            parameter["switch_rotation"][3] = parameter["switch_rotation"][
                0
            ] * random.choice([-1, 1])

            if switch_type == "Lever":
                parameter["number_of_switch"][0] = random.choices(
                    [1, 2, 3, 4], weights=[3, 1, 1, 1], k=1
                )[0]
                num_of_switch = parameter["number_of_switch"][0]

                if num_of_switch == 1:
                    parameter["separation"] = np.array([0, 0, 0])
                    parameter["base_size"] = np.array(
                        [randRange(1, 0.15, 0.18)[0], randRange(1, 0.09, 0.135)[0]]
                    )
                    parameter["main_size"] = np.array(
                        [
                            randRange(1, 0.044, 0.048)[0],
                            randRange(1, 0.077, 0.081)[0],
                            randRange(1, 0.32, 0.35)[0],
                        ]
                    )
                    parameter["inter_offset"] = np.array(
                        [0, 0, -parameter["main_size"][0]]
                    )
                    parameter["rotation"] = np.array([0, 0, 0])
                    parameter["position"] = np.array([0, 0, base_z])
                else:
                    parameter["base_size"] = np.array(
                        [randRange(1, 0.12, 0.15)[0], randRange(1, 0.045, 0.09)[0]]
                    )
                    parameter["separation"] = np.array(
                        [
                            randRange(1, 0.07, 0.09)[0] * 4 / num_of_switch
                            + parameter["base_size"][0] * 2,
                            0,
                            0,
                        ]
                    )
                    parameter["main_size"] = np.array(
                        [
                            randRange(1, 0.044, 0.048)[0],
                            randRange(1, 0.077, 0.081)[0],
                            randRange(1, 0.32, 0.35)[0],
                        ]
                    )
                    parameter["inter_offset"] = np.array(
                        [0, 0, -parameter["main_size"][0]]
                    )
                    parameter["position"] = np.array(
                        [
                            -(num_of_switch - 1) * 0.5 * parameter["separation"][0],
                            0,
                            base_z,
                        ]
                    )
                    parameter["rotation"] = np.array([0, 0, 0])

            elif switch_type == "Lever_and_knob":
                parameter["number_of_switch"][0] = 1
                base_x = randRange(1, 1.152, 1.5)[0]
                base_y = randRange(1, 0.72, 1)[0]
                parameter["separation"] = np.array([0, 0, 0])
                parameter["base_size"] = np.array(
                    [randRange(1, 0.13, 0.19)[0], randRange(1, 0.07, 0.135)[0]]
                )
                parameter["main_size"] = np.array(
                    [
                        randRange(1, 0.044, 0.048)[0],
                        randRange(1, 0.077, 0.081)[0],
                        randRange(1, 0.32, 0.35)[0],
                    ]
                )
                parameter["inter_offset"] = np.array([0, 0, 0])

                parameter["position"] = np.array(
                    [-base_x * randRange(1, 0.2, 0.24)[0], 0, base_z]
                )
                if random.choice([0, 1]):
                    parameter["position"][0] *= -1

                switch_offset = -parameter["position"][0]

                parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Standard_Knob":
            parameter["size"] = np.array(
                [randRange(1, 0.12, 0.15)[0], randRange(1, 0.07, 0.28)[0]]
            )
            if switch_type == "Lever_and_knob":
                parameter["position"] = np.array([switch_offset, 0, 0])
            elif switch_type == "Cuboid_and_knob":
                parameter["position"] = np.array([0, 0, 0])
            elif switch_type == "Knob":
                parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Round_Switch":
            parameter["number_of_switch"][0] = random.choices(
                [1, 2, 3, 4], weights=[3, 1, 1, 1], k=1
            )[0]
            num_of_switch = parameter["number_of_switch"][0]
            parameter["size"][0] = randRange(1, 0.1, 0.12)[0]

            parameter["offset_1"] = np.array([0, 0, 0])
            parameter["offset_2"] = parameter["offset_1"]
            parameter["offset_3"] = parameter["offset_1"]
            parameter["offset_4"] = parameter["offset_1"]

            base_z = randRange(1, 0.12, 0.18)[0]

            parameter["offset_Z"][0] = base_z
            base_x = randRange(1, 0.72, 1)[0]

            parameter["switch_rotation"][0] = randRange(1, 20, 30)[0] * random.choice(
                [-1, 1]
            )

            parameter["switch_rotation"][1] = parameter["switch_rotation"][
                0
            ] * random.choice([-1, 1])
            parameter["switch_rotation"][2] = parameter["switch_rotation"][
                0
            ] * random.choice([-1, 1])
            parameter["switch_rotation"][3] = parameter["switch_rotation"][
                0
            ] * random.choice([-1, 1])

            parameter["size"][1] = (
                2
                * parameter["size"][0]
                * np.tan(parameter["switch_rotation"][0] * np.pi / 180)
            )
            if parameter["size"][1] < 0:
                parameter["size"][1] *= -1
            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            if num_of_switch == 1:
                parameter["size"][0] *= 1.3
                parameter["size"][1] *= 1.3
            if num_of_switch == 2:
                parameter["offset_1"] = np.array([-base_x / 6, 0, 0])
                parameter["offset_2"] = np.array([base_x / 6, 0, 0])
            if num_of_switch == 3:
                parameter["offset_1"] = np.array([-base_x / 4, 0, 0])
                parameter["offset_2"] = np.array([base_x / 4, 0, 0])
                parameter["offset_3"] = np.array(
                    [0, randRange(1, 0.16, 0.23)[0] * base_x, 0]
                )
            if num_of_switch == 4:
                base_x *= 1.5
                parameter["offset_1"] = np.array([-base_x / 10, 0, 0])
                parameter["offset_2"] = np.array([base_x / 10, 0, 0])
                parameter["offset_3"] = np.array([-base_x * 0.3, 0, 0])
                parameter["offset_4"] = np.array([base_x * 0.3, 0, 0])

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
        switch_type = get_switch_type()
        existing_concept_templates = concept_template_existence(switch_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, switch_type)

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
            switch_type = get_switch_type()
            existing_concept_templates = concept_template_existence(switch_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, switch_type)
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
