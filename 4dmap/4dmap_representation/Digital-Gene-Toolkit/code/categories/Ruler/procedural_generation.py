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


def get_ruler_type():
    total_type = ["Symmetrical_Ruler", "Asymmetrical_Ruler"]
    weights = [1, 0]
    ruler_type = random.choices(total_type, weights=weights, k=1)[0]
    return ruler_type


def concept_template_existence(ruler_type):
    if ruler_type == "Symmetrical_Ruler":
        concept_template_variation = {
            "body": {"template": ["Symmetrical_body"], "necessary": True},
            "shaft": {"template": ["Regular_shaft"], "necessary": True},
        }
    elif ruler_type == "Asymmetrical_Ruler":
        concept_template_variation = {
            "body": {"template": ["Asymmetrical_body"], "necessary": True},
            "shaft": {"template": ["Regular_shaft"], "necessary": True},
        }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, ruler_type):
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Symmetrical_body":
            parameter["size"] = np.array([1.7, 0.25, 0.06]) * randRange(3, 0.7, 1.3)

            parameter["body_rotation"][0] = randRange(1, 0, 90)[0]

            if parameter["body_rotation"][0] > 30 and random.random() < 0.5:
                parameter["separation"][0] = (
                    -parameter["size"][1]
                    / 2
                    / np.sin(parameter["body_rotation"][0] / 180 * np.pi)
                )
            else:
                parameter["separation"][0] = 0

            parameter["left_right_offset"][0] = (
                parameter["size"][2] / 2 * randRange(1, 1.2, 2)[0]
            )

            parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Asymmetrical_body":
            parameter["body_rotation"][0] = randRange(1, 0, 90)[0]

            parameter["left_size"] = np.array([1.7, 0.25, 0.06]) * randRange(
                3, 0.7, 1.3
            )

            parameter["right_size"][0] = 1.7 * randRange(1, 0.7, 1.3)[0]
            parameter["right_size"][1] = parameter["left_size"][1]
            parameter["right_size"][2] = parameter["left_size"][2]

            if parameter["body_rotation"][0] > 30 and random.random() < 0.5:
                parameter["separation"][0] = (
                    -parameter["left_size"][1]
                    / 2
                    / np.sin(parameter["body_rotation"][0] / 180 * np.pi)
                )  #
            else:
                parameter["separation"][0] = 0

            parameter["left_right_offset"][0] = (
                (parameter["left_size"][2] + parameter["right_size"][2])
                / 4
                * randRange(1, 1.2, 2)[0]
            )

            parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Regular_shaft":
            if new_concepts[0]["template"] == "Symmetrical_body":
                parameter["size"][0] = (
                    new_concepts[0]["parameters"]["size"][1] * randRange(1, 0.4, 0.6)[0]
                )
                parameter["size"][1] = (
                    new_concepts[0]["parameters"]["left_right_offset"][0] * 2
                    - new_concepts[0]["parameters"]["size"][2]
                )
            elif new_concepts[0]["template"] == "Asymmetrical_body":
                parameter["size"][0] = (
                    new_concepts[0]["parameters"]["left_size"][1]
                    * randRange(1, 0.4, 0.6)[0]
                )
                parameter["size"][1] = (
                    new_concepts[0]["parameters"]["left_right_offset"][0] * 2
                    - (
                        new_concepts[0]["parameters"]["left_size"][2]
                        + new_concepts[0]["parameters"]["right_size"][2]
                    )
                    / 2
                )

            parameter["position"] = np.array([0, 0, 0])

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
        ruler_type = get_ruler_type()
        existing_concept_templates = concept_template_existence(ruler_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, ruler_type)

        vertices, faces = get_overall_model(new_concepts)
        pointcloud = get_overall_pointcloud(vertices, faces)
        opt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))
        coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        vertices = o3d.utility.Vector3dVector(vertices)
        faces = o3d.utility.Vector3iVector(faces)
        opt_mesh = o3d.geometry.TriangleMesh(vertices, faces)
        opt_mesh.compute_vertex_normals()
        coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        o3d.visualization.draw_geometries([opt_mesh, coordframe])

    else:
        concept_list = []
        for obj_idx in range(args.gen_num):
            ruler_type = get_ruler_type()
            existing_concept_templates = concept_template_existence(ruler_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, ruler_type)
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
