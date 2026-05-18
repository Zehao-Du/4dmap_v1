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


def get_knife_type():
    total_type = ["cutter", "spring", "stagger"]
    weights = [1, 1, 1]
    knife_type = random.choices(total_type, weights=weights, k=1)[0]
    return knife_type


def concept_template_existence(knife_type):
    if knife_type == "cutter":
        concept_template_variation = {
            "handle": {"template": ["Enveloping_Handle"], "necessary": True},
            "blade": {"template": ["Cusp_Blade"], "necessary": True},
            "button": {"template": ["Regular_Button"], "necessary": True},
        }
    elif knife_type == "spring":
        concept_template_variation = {
            "handle": {"template": ["Multideck_Handle"], "necessary": True},
            "blade": {"template": ["Cusp_Blade"], "necessary": True},
        }

    elif knife_type == "stagger":
        concept_template_variation = {
            "handle": {
                "template": [
                    "Cylindrical_Handle",
                    "Cuboidal_Handle",
                    "T_Shaped_Handle",
                    "Curved_Handle",
                ],
                "necessary": True,
            },
            "blade": {"template": ["Curved_Blade"], "necessary": True},
        }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, knife_type):
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    handle_params = None
    knife_move_distance = 0

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}
        parameter["position"] = parameter["rotation"] = np.ones(3)
        if template == "Enveloping_Handle":
            knife_width = (
                parameter["size"][0] - parameter["thickness"][1] * 2
            ) * randRange(1, 0.9, 1.1)[0]
            parameter["size"] *= randRange(3, 0.8, 1.3)

            parameter["thickness"][1] = (parameter["size"][0] - knife_width) / 2
            parameter["thickness"][::2] *= randRange(2, 0.8, 1.2)[1]
            parameter["gap_width"] *= randRange(1, 0.7, 1.3)
            parameter["position"] = parameter["rotation"] = np.zeros(3)
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
            handle_params = parameter
        elif template == "Multideck_Handle":
            parameter["bottom_size"] *= randRange(3, 0.6, 1.4)
            parameter["beside_seperation"] = parameter["bottom_size"][0] * randRange(
                1, 0.6, 0.7
            )
            parameter["beside_size"][1] *= randRange(1, 0.6, 1.4)[0]
            parameter["beside_size"][0] = (
                parameter["bottom_size"][0] - parameter["beside_seperation"][0]
            )
            parameter["beside_size"][2] = parameter["bottom_size"][2]
            parameter["beside_offset"] = np.zeros(2)
            parameter["position"] = parameter["rotation"] = np.zeros(3)
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
            handle_params = parameter
        elif template == "Cylindrical_Handle":
            template_existing_variance = ["frustum", "cylinder"]
            template_type = random.choice(template_existing_variance)
            parameter["size"] *= randRange(3, 0.6, 1.4)

            if template_type == "cylinder":
                parameter["size"][:2] = np.mean(
                    parameter["size"][:2], keepdims=1
                ).repeat(2)
            elif template_type == "frustum":
                parameter["size"][1::-1] = np.sort(parameter["size"][:2])
            parameter["position"] = parameter["rotation"] = np.zeros(3)
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
            handle_params = parameter
        elif template == "Cuboidal_Handle":
            parameter["size"] *= randRange(3, 0.6, 1.4)
            parameter["position"] = parameter["rotation"] = np.zeros(3)
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
            handle_params = parameter
        elif template == "T_Shaped_Handle":
            parameter["main_size"] *= randRange(3, 0.7, 1.3)
            parameter["bottom_size"][::2] = (
                parameter["main_size"][::2] * randRange(1, 1.2, 1.8)[0]
            )
            parameter["bottom_size"][1] = parameter["main_size"][1] / (
                randRange(1, 0.8, 1.2)[0] * 4.5
            )
            parameter["bottom_offset"] = np.array(
                random.choice(
                    [
                        [0, 0],
                        [
                            0,
                            0.5
                            * (parameter["bottom_size"][2] - parameter["main_size"][2]),
                        ],
                    ]
                )
            )
            parameter["position"] = parameter["rotation"] = np.zeros(3)
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
            handle_params = parameter
        elif template == "Curved_Handle":
            parameter["exist_angle"] *= randRange(1, 0.8, 1.2)[0]
            handle_width = (
                parameter["radius"][0] - parameter["radius"][1]
            ) * randRange(1, 0.7, 1.3)[0]

            parameter["radius"][0] *= randRange(1, 0.7, 1.3)[0]
            parameter["thickness"][0] *= randRange(1, 0.7, 1.3)[0]
            parameter["radius"][1] = parameter["radius"][0] - handle_width
            parameter["position"] = parameter["rotation"] = np.zeros(3)

            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
            handle_params = parameter
        elif template == "Cusp_Blade":
            if "Enveloping_Handle" in templates:
                blade_tip_ratio = randRange(1, 0.15, 0.25)[0]
                parameter["root_size"] = np.array(
                    [
                        handle_params["size"][0]
                        - handle_params["thickness"][1] * 2
                        - 0.005,
                        (handle_params["size"][1] - handle_params["thickness"][0])
                        * (1 - blade_tip_ratio),
                        handle_params["size"][2]
                        - handle_params["thickness"][2] * 2
                        - 0.005,
                        handle_params["size"][2]
                        - handle_params["thickness"][2] * 2
                        - 0.005,
                    ]
                )
                parameter["tip_length"] = np.array(
                    [
                        (handle_params["size"][1] - handle_params["thickness"][0])
                        * blade_tip_ratio
                    ]
                )
                parameter["root_z_offset"] = np.zeros(1)
                parameter["tip_z_offset"] = np.array([0.5 * parameter["root_size"][2]])
                knife_move_distance = (
                    handle_params["size"][1] - handle_params["thickness"][0]
                ) * (np.random.rand() * 0.8)
                parameter["rotation"] = np.zeros(3)
                parameter["position"] = np.array(
                    [
                        0,
                        knife_move_distance
                        - (
                            0.5 * handle_params["size"][1]
                            - handle_params["thickness"][0]
                        ),
                        0,
                    ]
                )
            elif "Multideck_Handle" in templates:
                rotation = random.choice(np.arange(180))
                blade_tip_ratio = randRange(1, 0.15, 0.25)[0]

                parameter["root_size"] = np.array(
                    [
                        handle_params["bottom_size"][0]
                        - handle_params["beside_size"][0] * 2,
                        handle_params["beside_size"][1] * (1 - blade_tip_ratio),
                        *np.array(
                            [
                                handle_params["bottom_size"][2]
                                * randRange(1, 0.6, 0.9)[0]
                            ]
                        ).repeat(2),
                    ]
                )
                parameter["tip_length"] = np.array(
                    [handle_params["beside_size"][1] * blade_tip_ratio]
                )
                parameter["root_z_offset"] = np.zeros(1)
                parameter["tip_z_offset"] = np.zeros(1)
                parameter["position"] = np.array(
                    [0, handle_params["beside_size"][1] / 2, 0]
                )

                parameter["rotation"] = np.array([rotation, 0, 0])
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
        elif template == "Curved_Blade":
            knife_existing_variance = [
                "equal_back_and_forth_size",
                "unequal_back_and_forth_size",
            ]

            if templates[0] == "Cylindrical_Handle":
                parameter["root_size"][0] *= randRange(1, 0.7, 1.3)[0]

                knife_type = random.choice(knife_existing_variance)
                if knife_type == "unequal_back_and_forth_size":
                    parameter["root_size"][1] = (
                        handle_params["size"][2] * randRange(1, 0.7, 1.1)[0]
                    )
                    parameter["root_size"][2] = (
                        handle_params["size"][0] * randRange(1, 1, 1.6)[0]
                    )
                    parameter["root_size"][3] = (
                        parameter["root_size"][2] * randRange(1, 1.2, 1.4)[0]
                    )
                    parameter["root_z_offset"] = np.zeros(1)
                    parameter["tip_length"] *= randRange(1, 0.7, 1)[0]
                    parameter["tip_angle"] *= randRange(1, 0.5, 0.7)[0]
                elif knife_type == "equal_back_and_forth_size":
                    parameter["root_size"][1] = (
                        handle_params["size"][2] * randRange(1, 0.7, 1.1)[0]
                    )
                    parameter["root_size"][2] = (
                        handle_params["size"][0] * randRange(1, 1.2, 1.6)[0]
                    )
                    parameter["root_size"][3] = parameter["root_size"][2]
                    parameter["root_z_offset"] = np.zeros(1)
                    parameter["tip_length"] *= randRange(1, 1, 1.2)[0]
                    parameter["tip_angle"] *= randRange(1, 0.6, 1)[0]
                parameter["position"] = np.array(
                    [0, 0.5 * handle_params["size"][2], 0.5 * parameter["root_size"][0]]
                )
                parameter["rotation"] = np.zeros(3)
            elif "Cuboidal_Handle" in templates:
                parameter["root_size"][0] *= randRange(1, 0.7, 1.3)[0]

                knife_existing_variance = [
                    "equal_back_and_forth_size",
                    "unequal_back_and_forth_size",
                ]
                knife_type = random.choice(knife_existing_variance)
                if knife_type == "unequal_back_and_forth_size":
                    parameter["root_size"][1] = (
                        handle_params["size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["root_size"][2] = (
                        handle_params["size"][2] * randRange(1, 0.7, 0.9)[0]
                    )
                    parameter["root_size"][3] = (
                        parameter["root_size"][2] * randRange(1, 1.2, 1.4)[0]
                    )
                    parameter["root_z_offset"] = np.zeros(1)
                    parameter["tip_length"] *= randRange(1, 0.7, 1.3)[0]
                    parameter["tip_angle"] *= randRange(1, 1, 1.3)[0]
                elif knife_type == "equal_back_and_forth_size":
                    parameter["root_size"][1] = (
                        handle_params["size"][1] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["root_size"][2] = (
                        handle_params["size"][2] * randRange(1, 0.6, 0.8)[0]
                    )
                    parameter["root_size"][3] = parameter["root_size"][2]
                    parameter["root_z_offset"] = np.zeros(1)
                    parameter["tip_length"] *= randRange(1, 1, 1.3)[0]
                    parameter["tip_angle"] *= randRange(1, 0.6, 1)[0]
                parameter["position"] = np.array(
                    [0, 0.5 * handle_params["size"][1], 0.5 * parameter["root_size"][0]]
                )
                parameter["rotation"] = np.zeros(3)
            elif "Curved_Handle" in templates:
                parameter["root_size"][0] *= randRange(1, 0.7, 1.3)[0]
                knife_type = random.choice(knife_existing_variance)
                if knife_type == "unequal_back_and_forth_size":
                    parameter["root_size"][1] = (
                        handle_params["radius"][0] * randRange(1, 0.2, 0.4)[0]
                    )
                    parameter["root_size"][2] = (
                        handle_params["radius"][0] - handle_params["radius"][1]
                    ) * randRange(1, 0.7, 1)[0]
                    parameter["root_size"][3] = (
                        parameter["root_size"][2] * randRange(1, 1.4, 2)[0]
                    )
                    parameter["root_z_offset"] = np.zeros(1)
                    parameter["tip_length"] = handle_params["radius"][0] * randRange(
                        1, 0.8, 1.1
                    )
                    parameter["tip_angle"] *= randRange(1, 1, 1.3)[0]

                elif knife_type == "equal_back_and_forth_size":
                    parameter["root_size"][1] = (
                        handle_params["radius"][0] * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["root_size"][2] = (
                        handle_params["radius"][0] - handle_params["radius"][1]
                    ) * randRange(1, 1.3, 1.5)[0]
                    parameter["root_size"][3] = parameter["root_size"][2]
                    parameter["root_z_offset"] = np.array(
                        [
                            0.5 * parameter["root_size"][2]
                            - 0.5
                            * (handle_params["radius"][0] - handle_params["radius"][1])
                        ]
                    )
                    parameter["tip_length"] *= randRange(1, 0.6, 0.8)[0]
                    parameter["tip_angle"] *= randRange(1, 0.6, 1)[0]
                parameter["position"] = np.array(
                    [
                        0,
                        0,
                        0.5 * parameter["root_size"][2]
                        - 0.5
                        * (handle_params["radius"][0] - handle_params["radius"][1]),
                    ]
                )
                parameter["rotation"] = np.zeros(3)
            elif "T_Shaped_Handle" in templates:
                parameter["root_size"][0] *= randRange(1, 0.7, 1.3)[0]
                parameter["root_size"][1] = (
                    handle_params["main_size"][1] * randRange(1, 0.8, 1.2)[0]
                )
                knife_existing_variance = [
                    "equal_back_and_forth_size",
                    "unequal_back_and_forth_size",
                ]
                knife_type = random.choice(knife_existing_variance)

                parameter["root_size"][2] = (
                    handle_params["main_size"][2] * randRange(1, 0.6, 0.8)[0]
                )
                parameter["root_size"][3] = (
                    parameter["root_size"][2] * randRange(1, 1.3, 1.5)[0]
                )
                parameter["root_z_offset"] = np.zeros(1)
                parameter["tip_length"] *= randRange(1, 0.7, 1.3)[0]
                parameter["tip_angle"] *= randRange(1, 0.6, 1)[0]
                parameter["position"] = np.array(
                    [0, 0.5 * (handle_params["main_size"][1]), 0]
                )
                parameter["rotation"] = np.zeros(3)
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
        elif template == "Regular_Button":
            parameter["size"][0] = handle_params["gap_width"][0] * (
                np.random.rand() * 0.1 + 1
            )
            parameter["size"][1] = parameter["size"][0]
            parameter["size"][2] *= randRange(1, 0.8, 1.2)[0]
            parameter["rotation"] = np.zeros(3)
            parameter["position"] = np.array(
                [
                    0.5 * parameter["size"][2],
                    knife_move_distance
                    - (0.5 * handle_params["size"][1] - handle_params["thickness"][0]),
                    0,
                ]
            )
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
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
        knife_type = get_knife_type()
        existing_concept_templates = concept_template_existence(knife_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, knife_type)

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
            knife_type = get_knife_type()
            existing_concept_templates = concept_template_existence(knife_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, knife_type)
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
