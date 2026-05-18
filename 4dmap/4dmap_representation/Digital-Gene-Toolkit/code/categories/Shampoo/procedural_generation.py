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


def get_shampoo_type():
    total_type = ["Toothpaste", "Cylindrical", "Cuboidal"]
    weights = [1, 1, 1]
    shampoo_type = random.choices(total_type, weights=weights, k=1)[0]
    return shampoo_type


def concept_template_existence(shampoo_type):
    if shampoo_type == "Toothpaste":
        concept_template_variation = {
            "body": {"template": ["Toothpaste_body"], "necessary": True},
            "cap": {"template": ["Cylindrical_cap"], "necessary": True},
        }
    elif shampoo_type == "Cylindrical":
        concept_template_variation = {
            "body": {"template": ["Cylindrical_body"], "necessary": True},
            "cap": {
                "template": ["Cylindrical_cap", "Regular_nozzle"],
                "necessary": True,
            },
        }
    elif shampoo_type == "Cuboidal":
        concept_template_variation = {
            "body": {"template": ["Cuboidal_body"], "necessary": True},
            "nozzle": {"template": ["Regular_nozzle"], "necessary": True},
        }
    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, shampoo_type):
    new_concepts = []
    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}
    existing_cuboidal_templates = ["fat_body", "slim_body"]
    existing_cylindrical_templates = None
    body_params = None
    body_height = 0
    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}
        parameter["position"] = np.ones(3)
        parameter["rotation"] = np.zeros(3)

        if template == "Toothpaste_body":
            parameter["radius"] = (
                randRange(parameter["radius"].shape[0], 0.6, 1.4) * parameter["radius"]
            )
            parameter["height"] *= randRange(parameter["height"].shape[0], 0.6, 1.4)
            parameter["bottom_length"] *= randRange(
                parameter["bottom_length"].shape[0], 0.6, 1.4
            )
            parameter["position"] = parameter["rotation"] = np.zeros(3)
            body_params = parameter
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )

        elif template == "Cylindrical_body":
            if "Cylindrical_cap" in templates:
                parameter["num_of_part"] = num_of_part = np.array(
                    [random.choice([2, 3, 4])]
                )
                num_of_part = num_of_part[0]

                for i in range(2 * parameter["num_of_part"][0] + 1):
                    parameter["all_sizes"][i] = randRange(
                        1,
                        0.8 * parameter["all_sizes"][i],
                        1.2 * parameter["all_sizes"][i],
                    )[0]
                x_z_ratio_existence = [1, np.random.rand() * (1.4 - 1.2) + 1.2]
                parameter["x_z_ratio"] = np.random.choice(x_z_ratio_existence, 1)
                parameter["position"] = np.zeros(3)
                parameter["rotation"] = np.array([0, 0, 0])

            elif "Regular_nozzle" in templates:
                parameter["num_of_part"] = num_of_part = np.array(
                    [random.choice([3, 4])]
                )
                num_of_part = num_of_part[0]
                for i in range(2 * parameter["num_of_part"][0] + 1):
                    parameter["all_sizes"][i] = randRange(
                        1,
                        0.8 * parameter["all_sizes"][i],
                        1.2 * parameter["all_sizes"][i],
                    )[0]

                parameter["x_z_ratio"] = randRange(1, 0.3, 1.7)
                parameter["position"] = np.zeros(3)
                parameter["rotation"] = np.array([0, 0, 0])
            body_params = parameter
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
        elif template == "Cuboidal_body":
            existing_cuboidal_templates = ["fat_body", "slim_body"]
            cuboidal_body_type = random.choices(
                existing_cuboidal_templates, weights=[1, 1], k=1
            )[0]
            if cuboidal_body_type == "fat_body":
                parameter["size"] = randRange(3, 0.8, 1.2) * parameter["size"]
            else:
                parameter["size"] = np.array(
                    [
                        (np.random.rand() * (1.3 - 0.7) + 0.7) * parameter["size"][0],
                        0,
                        (np.random.rand() * (1.3 - 0.7) + 0.7) * parameter["size"][0],
                    ]
                )
                parameter["size"][1] = (
                    np.random.rand() * (1.6 - 1.3) + 1.3
                ) * parameter["size"][0]
            parameter["position"] = parameter["rotation"] = np.zeros(3)
            body_params = parameter
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
        elif template == "Cylindrical_cap":
            if "Toothpaste_body" in templates:
                parameter["outer_size"] = np.array(
                    [
                        body_params["radius"][0],
                        body_params["radius"][0],
                        (np.random.rand() * (1.3 - 0.7) + 0.7)
                        * parameter["outer_size"][2],
                    ]
                )
                parameter["inner_size"] = (
                    randRange(3, 0.6, 0.8) * parameter["outer_size"]
                )
                parameter["x_z_ratio"] = np.array([1])
                parameter["position"] = np.array(
                    [
                        0,
                        0.5 * body_params["height"][0]
                        + 0.5 * parameter["inner_size"][2],
                        0,
                    ]
                )
                parameter["rotation"] = np.zeros(3)
            elif "Cylindrical_body" in templates:
                parameter["outer_size"] = np.array(
                    [
                        randRange(1, 0.7, 0.8)[0]
                        * body_params["all_sizes"][
                            1 + 2 * (body_params["num_of_part"][0] - 1)
                        ],
                        body_params["all_sizes"][
                            1 + 2 * (body_params["num_of_part"][0] - 1)
                        ],
                        (np.random.rand() * (1.3 - 0.7) + 0.7)
                        * parameter["outer_size"][2],
                    ]
                )
                parameter["outer_size"][1] = parameter["outer_size"][0]
                parameter["inner_size"] = (
                    randRange(3, 0.6, 0.8) * parameter["outer_size"]
                )
                x_z_ratio_existence = [1, np.random.rand() * (0.4) + 1]

                parameter["x_z_ratio"] = np.random.choice(x_z_ratio_existence, 1)

                parameter["position"] = np.array(
                    [
                        0,
                        0.5 * parameter["inner_size"][2]
                        + sum(
                            body_params["all_sizes"][
                                2 : 2 * (body_params["num_of_part"][0]) + 1 : 2
                            ]
                        )
                        - body_params["all_sizes"][2]
                        + 0.5 * body_params["all_sizes"][1],
                        0,
                    ]
                )
                parameter["rotation"] = np.zeros(3)
            new_concepts.append(
                {
                    "template": template,
                    "parameters": {k: v.tolist() for k, v in parameter.items()},
                }
            )
        elif template == "Regular_nozzle":
            delta_height = 0

            parameter["num_of_part"] = np.random.choice([3, 4, 5], 1)
            parameter["num_of_nozzle"] = np.random.choice([1, 2], 1)
            if "Cylindrical_body" in templates:
                parameter["parts_params"][0] = (
                    0.8
                    * body_params["all_sizes"][body_params["num_of_part"][0] * 2 - 1]
                    * min(body_params["x_z_ratio"][0], 1 / body_params["x_z_ratio"][0])
                )
                parameter["parts_params"][1] = (
                    randRange(1, 0.2, 0.4)[0] * parameter["parts_params"][0]
                )
                body_height = (
                    sum(
                        body_params["all_sizes"][
                            2 : 2 * body_params["num_of_part"][0] + 1 : 2
                        ]
                    )
                    - body_params["all_sizes"][2]
                    + 0.5 * body_params["all_sizes"][1]
                )
            else:
                parameter["parts_params"][0] = (np.random.rand() * 0.2 + 0.2) * min(
                    body_params["size"][0], body_params["size"][2]
                )
                parameter["parts_params"][1] = (
                    randRange(1, 0.2, 0.4)[0] * parameter["parts_params"][0]
                )
                body_height = body_params["size"][1] * 0.5

            for i in range(parameter["num_of_part"][0] - 2):
                if i == 0:
                    delta_height += parameter["parts_params"][1]
                    continue
                else:
                    parameter["parts_params"][i * 2] = (
                        randRange(1, 0.8, 1.05)[0]
                        * parameter["parts_params"][i * 2 - 2]
                    )
                    parameter["parts_params"][i * 2 + 1] = (
                        randRange(1, 0.9, 1.1)[0] * parameter["parts_params"][i * 2 - 1]
                    )
                    delta_height += parameter["parts_params"][i * 2 + 1]
            if "Cylindrical_body" in templates:
                parameter["parts_params"][2 * parameter["num_of_part"][0] - 4] = (
                    randRange(1, 0.4, 0.6)[0]
                    * parameter["parts_params"][2 * parameter["num_of_part"][0] - 6]
                )
                parameter["parts_params"][2 * parameter["num_of_part"][0] - 3] = (
                    body_height / 6 * randRange(1, 0.7, 1.3)[0]
                )
            else:
                parameter["parts_params"][2 * parameter["num_of_part"][0] - 4] = (
                    randRange(1, 0.4, 0.6)[0]
                    * parameter["parts_params"][2 * parameter["num_of_part"][0] - 6]
                )
                parameter["parts_params"][2 * parameter["num_of_part"][0] - 3] = (
                    body_height / 3 * randRange(1, 0.7, 1.3)[0]
                )
            parameter["parts_params"][2 * parameter["num_of_part"][0] - 2] = (
                parameter["parts_params"][0] * randRange(1, 0.8, 1)[0]
            )
            parameter["parts_params"][2 * parameter["num_of_part"][0] - 1] = (
                randRange(1, 0.4, 0.6)[0]
                * parameter["parts_params"][2 * parameter["num_of_part"][0] - 2]
            )
            delta_height += parameter["parts_params"][i]

            parameter["nozzle_size"] = np.array(
                [
                    randRange(1, 0.5, 0.7)[0]
                    * parameter["parts_params"][2 * parameter["num_of_part"][0] - 1]
                ]
            ).repeat(2)
            if parameter["num_of_nozzle"][0] == 1:
                parameter["nozzle_length"] = np.array(
                    [
                        (np.random.rand() * (1.2 - 0.8) + 0.8)
                        * parameter["parts_params"][2 * parameter["num_of_part"][0] - 3]
                    ]
                )
            else:
                parameter["nozzle_length"] = np.array(
                    [
                        (np.random.rand() * (1.2 - 0.8) + 0.8)
                        * parameter["parts_params"][
                            2 * parameter["num_of_part"][0] - 3
                        ],
                        parameter["nozzle_size"][0],
                    ]
                )

            parameter["nozzle_rotation"] = np.zeros(3)

            parameter["nozzle_offset"] = (
                np.array(
                    [
                        -1
                        * parameter["parts_params"][2 * parameter["num_of_part"][0] - 1]
                    ]
                )
                * randRange(1, 0.3, 0.5)[0]
            )

            parameter["position"] = np.array([0, body_height, 0])

            parameter["rotation"] = np.zeros(3)
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
        shampoo_type = get_shampoo_type()
        existing_concept_templates = concept_template_existence(shampoo_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, shampoo_type)

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
            shampoo_type = get_shampoo_type()
            existing_concept_templates = concept_template_existence(shampoo_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, shampoo_type)
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
