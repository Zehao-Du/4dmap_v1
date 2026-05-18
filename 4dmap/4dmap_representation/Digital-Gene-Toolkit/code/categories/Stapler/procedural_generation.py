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


def get_stapler_type():
    stapler_type = ["regular"]
    weights = [1]
    stapler_type = random.choices(stapler_type, weights=weights, k=1)[0]
    return stapler_type


def concept_template_existence(stapler_type):
    concept_template_variation = {
        "body": {"template": ["Standard_Body"], "necessary": True},
        "cover": {"template": ["Simplified_Cover", "Carved_Cover"], "necessary": True},
        "magazine": {
            "template": ["Carved_Magazine", "Complex_Magazine"],
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


def jitter_parameters(concepts, stapler_type):
    new_concepts = []

    smallerL: float = 0
    body_x_size: float = 0
    body_y_size: float = 0
    body_z_size: float = 0
    body_thick: float = 0
    body_hold_seperation: float = 0
    shaft_y_offset: float = 0
    shaft_z_offset: float = 0
    cover_rotation: float = 0

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Standard_Body":
            parameter["base_size"] = np.array(
                [
                    randRange(1, 0.3, 0.45)[0],
                    randRange(1, 0.05, 0.09)[0],
                    randRange(1, 1.15, 1.8)[0],
                ]
            )
            body_x_size = parameter["base_size"][0]
            body_y_size = parameter["base_size"][1]
            body_z_size = parameter["base_size"][2]

            parameter["beside_size"] = np.array(
                [
                    parameter["base_size"][0] * randRange(1, 0.21, 0.25)[0],
                    randRange(1, 0.2, 0.24)[0],
                    randRange(1, 0.2, 0.24)[0],
                ]
            )

            if parameter["beside_size"][1] >= parameter["beside_size"][2]:
                smallerL = parameter["beside_size"][2]
            else:
                smallerL = parameter["beside_size"][1]

            parameter["beside_seperation"][0] = (
                parameter["base_size"][0] - parameter["beside_size"][0] * 2
            ) * randRange(1, 0.7, 0.9)[0]
            body_hold_seperation = (
                parameter["beside_seperation"][0] - parameter["beside_size"][0]
            )
            parameter["beside_offset_z"][0] = (
                0.5
                * (parameter["base_size"][2] - parameter["beside_size"][2])
                * randRange(1, 0.1, 0.2)[0]
            )

            parameter["has_shaft"] = np.array([np.random.randint(0, 2)])
            parameter["shaft_central_size"] = np.array(
                [
                    randRange(1, 0.03, 0.045)[0],
                    parameter["beside_seperation"][0] + 2 * parameter["beside_size"][0],
                ]
            )
            parameter["shaft_beside_size"] = np.array(
                [randRange(1, 0.4, 0.5)[0] * smallerL, randRange(1, 0.012, 0.015)[0]]
            )
            parameter["shaft_offset"] = np.array([0, 0])
            shaft_y_offset = parameter["beside_size"][1]
            shaft_z_offset = (
                -0.5 * body_z_size
                + 0.5 * parameter["beside_size"][2]
                + parameter["beside_offset_z"][0]
                - parameter["shaft_central_size"][0]
            )

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Simplified_Cover":
            parameter["size"] = np.array(
                [body_x_size, body_x_size * randRange(1, 0.65, 0.75)[0], body_z_size]
            )
            parameter["position"] = np.array([0, 0.5 * body_y_size, -0.5 * body_z_size])
            cover_rotation = randRange(1, -125, -8)[0]
            parameter["rotation"] = np.array([cover_rotation, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Carved_Cover":
            body_thick = randRange(1, 0.012, 0.02)[0]
            parameter["outer_size"] = np.array(
                [
                    body_x_size + body_thick * 2,
                    body_x_size * randRange(1, 0.65, 0.75)[0],
                    body_z_size + body_thick * 2,
                ]
            )
            parameter["inner_size"] = np.array(
                [body_x_size, body_x_size * randRange(1, 0.4, 0.56)[0], body_z_size]
            )

            parameter["position"] = np.array([0, 0.5 * body_y_size, -0.5 * body_z_size])
            cover_rotation = randRange(1, -125, -8)[0]
            parameter["rotation"] = np.array([cover_rotation, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Carved_Magazine":
            parameter["outer_size"] = np.array(
                [
                    body_hold_seperation * randRange(1, 0.8, 0.9)[0],
                    shaft_y_offset,
                    body_z_size * randRange(1, 0.67, 0.8)[0],
                ]
            )
            parameter["inner_size"] = np.array(
                [
                    parameter["outer_size"][0] - 2 * body_thick,
                    parameter["outer_size"][1] - body_thick,
                    parameter["outer_size"][2] - 2 * body_thick,
                ]
            )
            parameter["rotation"] = np.array(
                [cover_rotation * randRange(1, 0.05, 0.8)[0], 0, 0]
            )
            parameter["position"] = np.array([0, 0.5 * body_y_size, shaft_z_offset])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Complex_Magazine":
            parameter["size"] = np.array(
                [
                    body_hold_seperation * randRange(1, 0.8, 0.9)[0],
                    shaft_y_offset,
                    body_z_size * randRange(1, 0.67, 0.8)[0],
                ]
            )
            parameter["thickness"][0] = (
                0.5 * parameter["size"][0] * randRange(1, 0.4, 0.7)[0]
            )
            parameter["front_height"][0] = (
                parameter["size"][1] * randRange(1, 1.1, 1.3)[0]
            )
            parameter["beside_length"][0] = (
                parameter["size"][2] * randRange(1, 0.7, 0.8)[0]
            )
            parameter["beside_offset"][0] = randRange(1, -0.2, 0.2)[0] * (
                parameter["size"][2] - parameter["beside_length"][0]
            )

            parameter["rotation"] = np.array(
                [cover_rotation * randRange(1, 0.05, 0.8)[0], 0, 0]
            )
            parameter["position"] = np.array([0, 0.5 * body_y_size, shaft_z_offset])

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
        stapler_type = get_stapler_type()
        existing_concept_templates = concept_template_existence(stapler_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, stapler_type)

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
            stapler_type = get_stapler_type()
            existing_concept_templates = concept_template_existence(stapler_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, stapler_type)
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
