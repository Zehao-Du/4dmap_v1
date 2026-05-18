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


def get_pliers_type():
    pliers_type = ["regular"]
    weights = [1]
    pliers_type = random.choices(pliers_type, weights=weights, k=1)[0]
    return pliers_type


def concept_template_existence(pliers_type):
    if pliers_type == "regular":
        concept_template_variation = {
            "shaft": {
                "template": ["Round_Shaft", "Rectangular_Shaft"],
                "necessary": True,
            },
            "handle": {
                "template": [
                    "Straight_Handle",
                    "Rear_Curved_Handle",
                    "Middle_Curved_Handle",
                ],
                "necessary": True,
            },
            "gripper": {
                "template": ["Cusp_Gripper", "Curved_Gripper"],
                "necessary": True,
            },
            "baffle": {
                "template": ["Rectangular_Baffle", "Curved_Baffle"],
                "necessary": False,
            },
        }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"] == False:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))
        elif part["necessary"] == True:
            templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, pliers_type):
    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}
    new_concepts = []
    shaft_params = None
    shaft_type = templates[0]
    layer_3_rotation = 0
    handle_rotation = 0
    for concept in concepts:
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}
        template = concept["template"]
        parameter["position"] = np.ones(3)
        if template == "Round_Shaft":
            parameter["size"] *= randRange(2, 0.7, 1.3)
            parameter["has_central_shaft"] = np.array(
                random.choices([0, 1], weights=[1, 1], k=1)
            )
            if parameter["has_central_shaft"]:
                parameter["central_shaft_size"] = np.array(
                    [
                        randRange(1, 0.4, 0.6)[0] * parameter["size"][0],
                        parameter["size"][1],
                    ]
                )
                parameter["central_shaft_offset"] = np.zeros(3)
            parameter["position"] = parameter["rotation"] = np.zeros(3)
            shaft_params = parameter
            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
        elif template == "Rectangular_Shaft":
            handle_rotation = randRange(1, 5, 45)[0]
            layer_3_rotation = randRange(1, 20, 50)[0]
            parameter["num_layers"] = np.array([random.choice([2, 3])])
            parameter["num_layers"] = np.array([3])
            for i in range(parameter["num_layers"][0]):
                if i != 0 and i != 2:
                    parameter[f"layer_{i + 1}_size"][1] = parameter[f"layer_{i}_size"][
                        1
                    ]
                    parameter[f"layer_{i + 1}_offset"] = np.array([0, 0])
            parameter["layer_1_size"] *= randRange(3, 0.7, 1.2)
            parameter["layer_2_size"] = parameter["layer_1_size"]
            parameter["layer_3_size"] *= randRange(3, 0.8, 1.5)
            parameter["layer_3_size"][1] = parameter["layer_2_size"][1] = parameter[
                "layer_1_size"
            ][1]
            parameter["layer_3_size"][::-2] = np.sort(parameter["layer_3_size"][::2])
            parameter["layer_2_offset"] = np.array([0, 0])
            parameter["layer_3_offset"] = np.array([0, 0])

            parameter["has_central_shaft"] = np.array(
                random.choices([0, 1], weights=[1, 1], k=1)
            )
            parameter["central_shaft_size"] *= randRange(2, 0.7, 1.3)
            parameter["central_shaft_offset"] = np.array([0, 0, 0])
            parameter["layer_rotation"] = np.array(
                [-handle_rotation, handle_rotation, layer_3_rotation]
            )
            parameter["position"] = np.zeros(3)
            shaft_params = parameter
            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Straight_Handle":
            if "Round_Shaft" in templates:
                parameter["front_size"] *= randRange(3, 0.7, 1.3)
                parameter["behind_size"] *= randRange(3, 0.7, 1.3)
                parameter["handle_separation"] = np.array([shaft_params["size"][0]])
                parameter["front_behind_offset"] = np.array([0, 0])
                parameter["left_right_offset"] = np.array([0])
                handle_rotation = randRange(1, 0, 50)[0]
                parameter["handle_rotation"] = np.array(
                    [handle_rotation, handle_rotation * randRange(1, 0.8, 1)[0]]
                )
                parameter["position"] = np.array([0, 0, 0])
            elif "Rectangular_Shaft" in templates:
                parameter["front_size"] *= randRange(3, 0.7, 1.3)
                parameter["front_size"][1] = shaft_params["layer_1_size"][1]
                parameter["behind_size"] *= randRange(3, 0.7, 1.3)
                parameter["handle_separation"] *= randRange(1, 0.8, 1.2)
                parameter["front_behind_offset"] = np.array([0, 0])
                parameter["left_right_offset"] = np.array(
                    [shaft_params["layer_1_size"][1]]
                )
                parameter["handle_rotation"] = np.array(
                    [handle_rotation, handle_rotation * randRange(1, 0.8, 1)[0]]
                )
                parameter["position"] = np.array(
                    [0, -0.5 * parameter["front_size"][1], 0]
                )
            handle_params = parameter
            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
        elif template == "Rear_Curved_Handle":
            if "Round_Shaft" in templates:
                parameter["front_size"] *= randRange(3, 0.7, 1.3)
                parameter["front_size"][1] = (
                    shaft_params["size"][1] * randRange(1, 0.8, 1)[0]
                )
                behind_width = parameter["behind_size"][0] - parameter["behind_size"][1]
                behind_width *= randRange(1, 0.8, 1.2)[0]

                parameter["behind_size"] = np.array(
                    [
                        parameter["behind_size"][0] * randRange(1, 0.8, 1.2)[0],
                        0,
                        parameter["front_size"][1] * randRange(1, 0.8, 1.2)[0],
                    ]
                )
                parameter["behind_size"][1] = parameter["behind_size"][0] - behind_width
                parameter["handle_separation"] = np.array([shaft_params["size"][0]])
                parameter["left_right_offset"] = np.array([0])
                parameter["exist_angle"] = np.array([40]) * randRange(1, 0.6, 1.3)[0]
                parameter["handle_rotation"] = np.array([16, 20]) * randRange(
                    2, 0.8, 1.2
                )
                parameter["handle_rotation"][0] = parameter["handle_rotation"][1] = (
                    parameter["handle_rotation"][0] + parameter["handle_rotation"][1]
                ) * 0.5
                offset_angle = np.arctan(
                    2
                    * parameter["behind_size"][0]
                    * np.power(np.sin(parameter["exist_angle"][0] / 2 * np.pi / 180), 2)
                    / (
                        parameter["front_size"][2]
                        + parameter["behind_size"][:2].mean()
                        * np.sin(parameter["exist_angle"][0] * np.pi / 180)
                    )
                )
                offset_angle *= 180 / np.pi
                parameter["handle_rotation"] += (
                    -parameter["handle_rotation"][0] + offset_angle
                )
                handle_rotation = randRange(1, 0, 45)[0]
                parameter["front_behind_offset"] = np.array(
                    [
                        -parameter["front_size"][0]
                        / 2
                        * np.cos(handle_rotation * np.pi / 180),
                        0,
                    ]
                )
                parameter["handle_rotation"] += handle_rotation
                parameter["position"] = np.array([0, 0, 0])
                parameter["rotation"] = np.zeros(3)

            elif "Rectangular_Shaft" in templates:
                parameter["front_size"] *= randRange(3, 0.7, 1.3)
                parameter["front_size"][1] = shaft_params["layer_1_size"][1]
                behind_width = parameter["behind_size"][0] - parameter["behind_size"][1]
                behind_width *= randRange(1, 0.8, 1.2)[0]

                parameter["behind_size"] = np.array(
                    [
                        parameter["behind_size"][0] * randRange(1, 0.7, 1.3)[0],
                        0,
                        parameter["front_size"][1] * randRange(1, 0.8, 1.2)[0],
                    ]
                )
                parameter["behind_size"][1] = parameter["behind_size"][0] - behind_width
                parameter["handle_separation"] = np.array(
                    [
                        shaft_params["layer_1_size"][2]
                        * np.sin(handle_rotation * np.pi / 180)
                    ]
                )
                parameter["exist_angle"] = np.array([40]) * randRange(1, 0.6, 1.3)[0]

                parameter["handle_rotation"] = np.array([16, 20]) * randRange(
                    2, 0.7, 1.3
                )
                offset_angle = np.arctan(
                    2
                    * parameter["behind_size"][0]
                    * np.power(np.sin(parameter["exist_angle"][0] / 2 * np.pi / 180), 2)
                    / (
                        parameter["front_size"][2]
                        + parameter["behind_size"][:2].mean()
                        * np.sin(parameter["exist_angle"][0] * np.pi / 180)
                    )
                )
                offset_angle *= 180 / np.pi
                parameter["handle_rotation"] += (
                    -parameter["handle_rotation"][0] + offset_angle
                )
                handle_rotation = shaft_params["layer_rotation"][1]
                parameter["handle_rotation"] += handle_rotation

                parameter["left_right_offset"] = np.array([parameter["front_size"][1]])
                parameter["front_behind_offset"] = np.array(
                    [
                        -parameter["front_size"][0]
                        / 2
                        * np.cos(handle_rotation * np.pi / 180),
                        0,
                    ]
                )
                parameter["position"] = np.array(
                    [
                        0,
                        -0.5 * parameter["front_size"][1],
                        -0.5
                        * shaft_params["layer_1_size"][2]
                        * np.cos(handle_rotation * np.pi / 180),
                    ]
                )
                parameter["rotation"] = np.zeros(3)
            handle_params = parameter
            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Middle_Curved_Handle":
            if "Round_Shaft" in templates:
                parameter["front_size"] *= randRange(3, 0.7, 1.3)
                parameter["front_size"][1] = (
                    shaft_params["size"][1] * randRange(1, 0.8, 1)[0]
                )
                middle_width = parameter["middle_size"][0] - parameter["middle_size"][1]
                middle_width *= randRange(1, 0.8, 1.2)[0]

                parameter["middle_size"] = np.array(
                    [
                        parameter["middle_size"][0] * randRange(1, 0.7, 1.3)[0],
                        0,
                        parameter["front_size"][1] * randRange(1, 0.8, 1.2)[0],
                    ]
                )
                parameter["middle_size"][1] = parameter["middle_size"][0] - middle_width
                parameter["behind_size"] *= randRange(3, 0.7, 1.3)
                parameter["behind_size"][1] = parameter["front_size"][1]
                parameter["handle_separation"] = np.array([shaft_params["size"][0]])
                parameter["exist_angle"] = np.array([40]) * randRange(1, 0.6, 1.3)[0]
                parameter["middle_behind_offset"] = np.array([0, 0])

                parameter["handle_rotation"] = np.array([11, 30, 7]) * randRange(
                    3, 0.7, 1.3
                )
                parameter["handle_rotation"][:2] = (
                    parameter["handle_rotation"][:2].mean(keepdims=1).repeat(2)
                )
                offset_angle = np.arctan(
                    2
                    * parameter["middle_size"][0]
                    * np.power(np.sin(parameter["exist_angle"][0] / 2 * np.pi / 180), 2)
                    / (
                        parameter["front_size"][2]
                        + parameter["middle_size"][:2].mean()
                        * np.sin(parameter["exist_angle"][0] * np.pi / 180)
                    )
                )
                offset_angle *= 180 / np.pi
                parameter["handle_rotation"] += (
                    -parameter["handle_rotation"][0] + offset_angle
                )
                handle_rotation = randRange(1, 0, 45)[0]
                parameter["handle_rotation"] += handle_rotation

                parameter["front_middle_offset"] = np.array(
                    [
                        -parameter["front_size"][0]
                        * np.sin(handle_rotation * np.pi / 180),
                        0,
                    ]
                )
                parameter["left_right_offset"] = np.array([0])

                parameter["position"] = np.array([0, 0, 0])

            elif "Rectangular_Shaft" in templates:
                parameter["front_size"] *= randRange(3, 0.7, 1.3)
                parameter["front_size"][1] = shaft_params["layer_1_size"][1]
                middle_width = parameter["middle_size"][0] - parameter["middle_size"][1]
                middle_width *= randRange(1, 0.8, 1.2)[0]

                parameter["middle_size"] = np.array(
                    [
                        parameter["middle_size"][0] * randRange(1, 0.7, 1.3)[0],
                        0,
                        parameter["front_size"][1] * randRange(1, 0.8, 1.2)[0],
                    ]
                )
                parameter["middle_size"][1] = parameter["middle_size"][0] - middle_width

                parameter["behind_size"] *= randRange(3, 0.8, 1.2)
                parameter["behind_size"][1] = parameter["front_size"][1]
                parameter["handle_separation"] = np.array(
                    [
                        shaft_params["layer_1_size"][2]
                        * np.sin(handle_rotation * np.pi / 180)
                    ]
                )
                parameter["exist_angle"] = np.array([40]) * randRange(1, 0.6, 1.3)[0]

                parameter["handle_rotation"] = np.array([11, 30, 7]) * randRange(
                    3, 0.7, 1.3
                )
                offset_angle = np.arctan(
                    2
                    * parameter["middle_size"][0]
                    * np.power(np.sin(parameter["exist_angle"][0] / 2 * np.pi / 180), 2)
                    / (
                        parameter["front_size"][2]
                        + parameter["middle_size"][:2].mean()
                        * np.sin(parameter["exist_angle"][0] * np.pi / 180)
                    )
                )
                offset_angle *= 180 / np.pi
                parameter["handle_rotation"] += -parameter["handle_rotation"][0]
                handle_rotation = shaft_params["layer_rotation"][1]
                parameter["handle_rotation"] += handle_rotation + offset_angle

                parameter["left_right_offset"] = np.array([parameter["front_size"][1]])
                parameter["front_middle_offset"] = np.array(
                    [
                        -parameter["front_size"][0]
                        * np.sin(handle_rotation * np.pi / 180),
                        0,
                    ]
                )
                parameter["middle_behind_offset"] = np.array([0, 0])

                parameter["position"] = np.array(
                    [
                        0,
                        -0.5 * shaft_params["layer_1_size"][1],
                        -0.5
                        * shaft_params["layer_1_size"][2]
                        * np.cos(handle_rotation * np.pi / 180),
                    ]
                )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
            handle_params = parameter

        elif template == "Cusp_Gripper":
            if "Round_Shaft" in templates:
                parameter["behind_size"] *= randRange(4, 0.8, 1.2)
                parameter["front_size"] *= randRange(4, 0.8, 1.2)
                parameter["gripper_separation"] = np.array(
                    [shaft_params["size"][0] * np.sin(handle_rotation * np.pi / 180)]
                )
                parameter["rotation"] = np.zeros(3)
                parameter["gripper_rotation"] = np.array([handle_rotation])
                parameter["position"] = np.zeros(3)
                parameter["position"][2] = shaft_params["size"][0] * np.cos(
                    handle_rotation * np.pi / 180
                )
            elif "Rectangular_Shaft" in templates:
                if shaft_params["num_layers"][0] == 2:
                    parameter["behind_size"] *= randRange(4, 0.8, 1.2)
                    parameter["front_size"] *= randRange(4, 0.8, 1.2)
                    parameter["gripper_separation"] = np.array(
                        [
                            shaft_params["layer_1_size"][2]
                            * 0.5
                            * np.sin(handle_rotation * np.pi / 180)
                        ]
                    )
                    parameter["rotation"] = np.zeros(3)
                    parameter["gripper_rotation"] = np.array([handle_rotation])
                    parameter["position"] = np.zeros(3)
                    parameter["position"][2] = (
                        shaft_params["layer_1_size"][0]
                        * np.cos(handle_rotation * np.pi / 180)
                        * 0.5
                    )
                elif shaft_params["num_layers"][0] == 3:
                    parameter["behind_size"] *= randRange(4, 0.5, 0.8)
                    parameter["front_size"] *= randRange(4, 0.5, 0.8)
                    parameter["front_size"][1] = shaft_params["layer_3_size"][1]
                    parameter["gripper_separation"] = np.array(
                        [shaft_params["layer_3_size"][0] * handle_rotation / 50]
                    )
                    parameter["rotation"] = np.array([0, layer_3_rotation, 0])
                    parameter["gripper_rotation"] = np.array([0])
                    parameter["position"] = np.array(
                        [
                            0.5
                            * shaft_params["layer_3_size"][2]
                            * np.sin(layer_3_rotation * np.pi / 180),
                            shaft_params["layer_2_size"][1]
                            + 0.5 * shaft_params["layer_3_size"][1],
                            0.5
                            * shaft_params["layer_3_size"][2]
                            * np.cos(layer_3_rotation * np.pi / 180),
                        ]
                    )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
        elif template == "Curved_Gripper":
            if "Round_Shaft" in templates:
                parameter["radius"] *= randRange(2, 0.8, 1.2)
                parameter["thickness"] *= randRange(1, 0.8, 1.2)

                parameter["gripper_separation"] = np.array(
                    [shaft_params["size"][0] * np.sin(handle_rotation * np.pi / 180)]
                )
                parameter["rotation"] = np.zeros(3)
                parameter["gripper_rotation"] = np.array([handle_rotation])
                parameter["position"] = np.zeros(3)
                parameter["position"][2] = (
                    shaft_params["size"][0]
                    * np.cos(handle_rotation * np.pi / 180)
                    * 0.5
                )
            elif "Rectangular_Shaft" in templates:
                if shaft_params["num_layers"][0] == 2:
                    parameter["radius"] *= randRange(2, 0.8, 1.2)
                    parameter["thickness"] *= randRange(1, 0.8, 1.2)

                    parameter["gripper_separation"] = np.array(
                        [
                            shaft_params["layer_1_size"][2]
                            * 0.5
                            * np.sin(handle_rotation * np.pi / 180)
                        ]
                    )
                    parameter["rotation"] = np.zeros(3)
                    parameter["gripper_rotation"] = np.array([handle_rotation])
                    parameter["position"] = np.zeros(3)
                    parameter["position"][2] = (
                        shaft_params["layer_1_size"][0]
                        * np.cos(handle_rotation * np.pi / 180)
                        * 0.5
                    )
                elif shaft_params["num_layers"][0] == 3:
                    parameter["radius"] *= randRange(2, 0.5, 0.8)
                    parameter["thickness"][0] = shaft_params["layer_3_size"][1]
                    parameter["gripper_separation"] = np.array(
                        [shaft_params["layer_3_size"][0] * handle_rotation / 50]
                    )
                    parameter["rotation"] = np.array([0, layer_3_rotation, 0])
                    parameter["gripper_rotation"] = np.array([0])
                    parameter["position"] = np.array(
                        [
                            0.5
                            * shaft_params["layer_3_size"][2]
                            * np.sin(layer_3_rotation * np.pi / 180),
                            shaft_params["layer_2_size"][1]
                            + 0.5 * shaft_params["layer_3_size"][1],
                            0.5
                            * shaft_params["layer_3_size"][2]
                            * np.cos(layer_3_rotation * np.pi / 180),
                        ]
                    )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Rectangular_Baffle":
            baffle_offset = parameter["size"][0] * randRange(1, 0.1, 0.3)[0]
            parameter["size"] = handle_params["front_size"] * randRange(3, 1.5, 2.5)
            parameter["size"][2] = 0.03 * randRange(1, 0.7, 1.3)[0]
            z_move_distance = randRange(1, 0.7, 0.9)[0] * handle_params["front_size"][2]
            parameter["baffle_separation"] = 2 * np.array(
                [
                    z_move_distance * np.sin(handle_rotation * np.pi / 180)
                    + handle_params["front_size"][0] / 2
                    + baffle_offset * np.cos(handle_rotation * np.pi / 180)
                ]
            )
            parameter["rotation"] = np.array([0, 0, 0])
            parameter["position"] = (
                np.array(
                    [
                        0,
                        0,
                        -z_move_distance * np.cos(handle_rotation * np.pi / 180)
                        + baffle_offset * np.sin(handle_rotation * np.pi / 180),
                    ]
                )
                + handle_params["position"]
            )
            parameter["baffle_rotation"] = np.array([-handle_rotation])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Curved_Baffle":
            z_move_distance = randRange(1, 0.6, 0.8)[0] * handle_params["front_size"][2]
            origin_radius = np.sqrt(
                z_move_distance**2
                + 0.25 * handle_params["handle_separation"][0] ** 2
                + 2
                * z_move_distance
                * 0.5
                * handle_params["handle_separation"][0]
                * np.sin(handle_params["handle_rotation"][0] * np.pi / 180)
            )
            parameter["radius"][::-1] = np.sort(
                np.array(
                    [
                        np.sqrt(
                            2 * origin_radius**2
                            + 2
                            * origin_radius
                            * z_move_distance
                            * np.cos(handle_rotation * np.pi / 180)
                        )
                    ]
                ).repeat(2)
                * randRange(2, 0.9, 1.1)
            )
            parameter["height"] *= randRange(1, 0.8, 1.2)[0]
            parameter["exist_angle"] = (
                np.array(
                    [
                        180
                        / 2
                        / np.pi
                        * np.arccos(
                            z_move_distance
                            * np.cos(handle_params["handle_rotation"][0] * np.pi / 180)
                            / origin_radius
                        )
                    ]
                )
                + randRange(1, 1.2, 1.3)
                * np.arctan(
                    0.5 * handle_params["front_size"][0] / parameter["radius"][0]
                )
                * 180
                / np.pi
            )

            parameter["seperation_rotation"] = np.array([0])
            parameter["position"] = (
                np.array([0, 0, origin_radius]) + handle_params["position"]
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
        pliers_type = get_pliers_type()
        existing_concept_templates = concept_template_existence(pliers_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, pliers_type)
        print(new_concepts)
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
            pliers_type = get_pliers_type()
            existing_concept_templates = concept_template_existence(pliers_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, pliers_type)
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
