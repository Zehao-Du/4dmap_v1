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


def get_trashcan_type():
    total_type = ["cuboidal", "cylindrical"]
    weights = [1, 1]
    trashcan_type = random.choices(total_type, weights=weights, k=1)[0]
    return trashcan_type


def concept_template_existence(trashcan_type):
    if trashcan_type == "cuboidal":
        concept_template_variation = {
            "body": {"template": ["Prismatic_Body"], "necessary": True},
            "shell": {"template": ["Cuboidal_Shell"], "necessary": False},
            "cover": {
                "template": [
                    "Cuboidal_Cover",
                    "Double_Layer_Cuboidal_Cover",
                    "Cuboidal_Hollow_Cover",
                    "Holed_Cuboidal_Cover",
                ],
                "necessary": True,
            },
        }
    elif trashcan_type == "cylindrical":
        concept_template_variation = {
            "body": {
                "template": ["Cylindrical_Body", "Separated_Cylindrical_Body"],
                "necessary": True,
            },
            "shell": {"template": ["Cylindrical_Shell"], "necessary": False},
            "cover": {
                "template": [
                    "Cylindrical_Cover",
                    "Cylindrical_Hollow_Cover",
                    "Holed_Cylindrical_Cover",
                ],
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


def jitter_parameters(concepts, trashcan_type):
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    body_params = None

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Prismatic_Body":
            parameter["top_size"] = parameter["top_size"] * randRange(2, 0.7, 1.3)
            parameter["bottom_size"] = parameter["bottom_size"] * randRange(2, 0.7, 1.3)
            parameter["height"][0] = parameter["height"][0] * randRange(1, 0.7, 1.3)[0]
            parameter["height"][1] = parameter["height"][0]
            parameter["top_offset"] = np.array([0, 0])
            parameter["thickness"] = parameter["thickness"] * randRange(3, 0.7, 1.3)

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
            body_params = parameter

        elif template == "Cylindrical_Body":
            parameter["outer_size"] = parameter["outer_size"] * randRange(3, 0.7, 1.3)
            parameter["inner_size"] = parameter["outer_size"] * randRange(3, 0.8, 0.95)

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
            body_params = parameter

        elif template == "Separated_Cylindrical_Body":
            parameter["outer_size"] = parameter["outer_size"] * randRange(3, 0.7, 1.3)
            parameter["inner_size"] = parameter["outer_size"] * randRange(3, 0.8, 0.95)
            parameter["clapboard_size"] = np.array(
                [
                    min(parameter["inner_size"][0], parameter["inner_size"][1]) * 2,
                    parameter["inner_size"][2] - parameter["inner_size"][0],
                    (parameter["outer_size"][2] - parameter["inner_size"][2])
                    * randRange(1, 0.8, 0.9)[0],
                ]
            )

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)
            body_params = parameter

        elif template == "Cuboidal_Shell":
            parameter["inner_size"] = np.array(
                [
                    randRange(1, 1.05, 1.4)[0]
                    * max(body_params["top_size"][0], body_params["bottom_size"][0]),
                    randRange(1, 1.05, 1.4)[0]
                    * max(body_params["top_size"][1], body_params["bottom_size"][1]),
                ]
            )
            if "Cuboidal_Cover" in templates:
                parameter["outer_size"] = np.array(
                    [
                        randRange(1, 1.1, 1.2)[0] * parameter["inner_size"][0],
                        body_params["height"][0] + body_params["top_size"][1] / 2,
                        randRange(1, 1.1, 1.2)[0] * parameter["inner_size"][1],
                    ]
                )
            else:
                parameter["outer_size"] = np.array(
                    [
                        randRange(1, 1.1, 1.2)[0] * parameter["inner_size"][0],
                        body_params["height"][0],
                        randRange(1, 1.1, 1.2)[0] * parameter["inner_size"][1],
                    ]
                )

            parameter["position"] = np.array(
                [0, (parameter["outer_size"][1] - body_params["height"][0]) / 2, 0]
            )
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cylindrical_Shell":
            parameter["inner_size"] = np.array(
                [
                    randRange(1, 1.05, 1.4)[0] * body_params["outer_size"][0],
                    randRange(1, 1.05, 1.4)[0] * body_params["outer_size"][1],
                ]
            )
            if "Cylindrical_Cover" in templates:
                parameter["outer_size"] = np.array(
                    [
                        randRange(1, 1.1, 1.2)[0] * parameter["inner_size"][0],
                        randRange(1, 1.1, 1.2)[0] * parameter["inner_size"][1],
                        randRange(1, 1, 1.2)[0] * body_params["outer_size"][2]
                        + body_params["outer_size"][0],
                    ]
                )
            else:
                parameter["outer_size"] = np.array(
                    [
                        randRange(1, 1.1, 1.2)[0] * parameter["inner_size"][0],
                        randRange(1, 1.1, 1.2)[0] * parameter["inner_size"][1],
                        randRange(1, 1, 1.2)[0] * body_params["outer_size"][2],
                    ]
                )

            parameter["position"] = np.array(
                [0, (parameter["outer_size"][2] - body_params["outer_size"][2]) / 2, 0]
            )
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cuboidal_Cover":
            parameter["bottom_size"][0] = (
                randRange(1, 1.0, 1.2)[0] * body_params["top_size"][0]
            )
            parameter["bottom_size"][1] = (
                randRange(1, 1.0, 1.2)[0] * body_params["top_size"][1]
            )
            parameter["top_size"] = randRange(2, 0.7, 1.1) * parameter["bottom_size"]
            parameter["height"] = randRange(2, 0.8, 1.2) * parameter["height"]
            parameter["top_offset"] = np.array([0, 0])

            parameter["rotation"] = np.array([randRange(1, -45, 0)[0], 0, 0])
            parameter["position"] = np.array(
                [
                    body_params["top_offset"][0],
                    0.5 * body_params["height"][0]
                    - (1 - np.cos(parameter["rotation"][0] * np.pi / 180))
                    * 0.5
                    * parameter["height"][0],
                    body_params["top_offset"][1]
                    - body_params["top_size"][1] / 2
                    - np.sin(parameter["rotation"][0] * np.pi / 180)
                    * 0.5
                    * parameter["height"][0],
                ]
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Double_Layer_Cuboidal_Cover":
            parameter["bottom_size"] = np.array(
                [
                    randRange(1, 1, 1.1)[0] * body_params["top_size"][0],
                    randRange(1, 0.8, 1.2)[0] * parameter["bottom_size"][1],
                    randRange(1, 1, 1.1)[0] * body_params["top_size"][1],
                ]
            )
            parameter["top_size"] = np.array(
                [
                    randRange(1, 0.5, 0.9)[0] * parameter["bottom_size"][0],
                    randRange(1, 0.4, 0.8)[0] * parameter["bottom_size"][1],
                    randRange(1, 0.5, 0.9)[0] * parameter["bottom_size"][2],
                ]
            )
            parameter["top_offset"] = np.array([0, 0])

            parameter["rotation"] = np.array([randRange(1, -45, 0)[0], 0, 0])
            parameter["position"] = np.array(
                [
                    body_params["top_offset"][0],
                    0.5 * body_params["height"][0],
                    body_params["top_offset"][1] - parameter["bottom_size"][2] / 2,
                ]
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Holed_Cuboidal_Cover":
            parameter["outer_size"] = np.array(
                [
                    body_params["top_size"][0],
                    randRange(1, 0.8, 1.2)[0] * parameter["outer_size"][1],
                    body_params["top_size"][1],
                ]
            )
            parameter["inner_size"] = randRange(3, 0.8, 0.95) * parameter["outer_size"]
            parameter["front_behind_hole_size"] = np.array(
                [
                    randRange(1, 0.6, 0.9)[0] * parameter["inner_size"][0],
                    randRange(1, 0.6, 0.9)[0] * parameter["inner_size"][1],
                ]
            )
            parameter["front_behind_hole_size"] = np.array(
                [
                    randRange(1, 0.6, 0.9)[0] * parameter["inner_size"][2],
                    parameter["front_behind_hole_size"][1],
                ]
            )
            parameter["has_hole"] = np.array(
                random.choice(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [1, 1, 1, 1],
                    ]
                )
            )

            parameter["rotation"] = np.array([0, 0, 0])
            parameter["position"] = np.array(
                [
                    body_params["top_offset"][0],
                    0.5 * body_params["height"][0],
                    body_params["top_offset"][1],
                ]
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cuboidal_Hollow_Cover":
            parameter["outer_size"] = np.array(
                [
                    body_params["top_size"][0],
                    randRange(1, 0.02, 0.04)[0],
                    body_params["top_size"][1],
                ]
            )
            parameter["inner_size"] = np.array(
                [
                    randRange(1, 0.5, 0.8)[0] * parameter["outer_size"][0],
                    randRange(1, 0.5, 0.8)[0] * parameter["outer_size"][2],
                ]
            )
            parameter["inner_offset"] = (
                randRange(2, 0.8, 1.2) * parameter["inner_offset"]
            )

            parameter["rotation"] = np.array([0, 0, 0])
            parameter["position"] = np.array(
                [
                    body_params["top_offset"][0],
                    0.5 * body_params["height"][0],
                    body_params["top_offset"][1],
                ]
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cylindrical_Cover":
            parameter["size"] = np.array(
                [
                    body_params["inner_size"][0],
                    randRange(1, 0.7, 1.3)[0] * parameter["size"][1],
                ]
            )

            parameter["rotation"] = np.array([randRange(1, -45, 45)[0], 0, 0])
            parameter["position"] = np.array([0, 0.5 * body_params["inner_size"][2], 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Holed_Cylindrical_Cover":
            parameter["num_sides"] = np.array([np.random.randint(1, 5)])
            parameter["radius"][0] = body_params["outer_size"][0]
            parameter["radius"][1] = randRange(
                1, body_params["inner_size"][0], parameter["radius"][0]
            )[0]

            parameter["height"][0] = randRange(1, 0.7, 1.3)[0] * parameter["height"][0]
            parameter["height"][1] = randRange(1, 0.7, 0.9)[0] * parameter["height"][0]
            parameter["height"][2] = randRange(1, 0.7, 1.3)[0] * parameter["height"][2]

            parameter["exist_angle"] = np.array([180 / parameter["num_sides"][0] / 2])

            parameter["rotation"] = np.array([0, 0, 0])
            parameter["position"] = np.array([0, 0.5 * body_params["inner_size"][2], 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cylindrical_Hollow_Cover":
            parameter["size"][0] = body_params["outer_size"][0]
            parameter["size"][1] = randRange(1, 0.3, 0.6)[0] * parameter["size"][0]
            parameter["size"][2] = randRange(1, 0.7, 1.3)[0] * parameter["size"][2]

            parameter["rotation"] = np.array([0, 0, 0])
            parameter["position"] = np.array([0, 0.5 * body_params["inner_size"][2], 0])

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
        trashcan_type = get_trashcan_type()
        existing_concept_templates = concept_template_existence(trashcan_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, trashcan_type)

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
            trashcan_type = get_trashcan_type()
            existing_concept_templates = concept_template_existence(trashcan_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, trashcan_type)
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
