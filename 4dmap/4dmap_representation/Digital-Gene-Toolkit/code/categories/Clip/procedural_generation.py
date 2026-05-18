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


def get_clip_type():
    clip_type = ["A-clip", "streamline-clip", "T-A-clip", "T-streamline-clip"]
    weights = [1, 1, 1, 1]
    clip_type = random.choices(clip_type, weights=weights, k=1)[0]
    return clip_type


def concept_template_existence(clip_type):
    necessary = [1, 1]
    lever_template = ["Regular_lever"]
    jaw_template = ["Regular_jaw", "Curved_jaw"]
    if clip_type == "A-clip":
        necessary = [1, 0]
    elif clip_type == "streamline-clip":
        jaw_template = ["Curved_jaw"]
    elif clip_type == "T-A-clip":
        jaw_template = ["Regular_jaw"]
    elif clip_type == "T-streamline-clip":
        necessary = [1, 1]
        jaw_template = ["Curved_jaw"]

    concept_template_variation = {
        "lever": {"template": lever_template, "necessary": necessary[0]},
        "jaw": {"template": jaw_template, "necessary": necessary[1]},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"] == 0.5:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))
        elif part["necessary"] == 1:
            templates.append(random.choice(part["template"]))
    return templates


def get_target_angle(separation, radius, central_angle):
    x = separation
    r = radius
    sin_theta = np.sin(central_angle / 180 * np.pi)
    cos_theta = np.cos(central_angle / 180 * np.pi)
    tmp_1 = 1 - cos_theta
    tmp_2 = sin_theta
    beta = np.arccos(tmp_1 / np.sqrt(tmp_1**2 + tmp_2**2))
    tmp_3 = (x / r) / np.sqrt(tmp_1**2 + tmp_2**2)
    alpha = np.arccos(tmp_3) - beta
    return alpha


def jitter_parameters(concepts, clip_type):
    new_concepts = []

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Regular_lever":
            parameter["position"][0] = 0
            parameter["position"][1] = 0
            parameter["position"][2] = 0
            parameter["rotation"][0] = 0
            parameter["rotation"][1] = 0
            parameter["rotation"][2] = 0
            if clip_type == "A-clip":
                parameter["level_handle_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["level_handle_size"][1] = (
                    parameter["level_handle_size"][0] * 3 * randRange(1, 0.8, 1.4)[0]
                )
                parameter["level_handle_size"][2] = (
                    parameter["level_handle_size"][0]
                    * 3
                    / 8
                    * randRange(1, 0.8, 1.5)[0]
                )

                support_size_0 = (
                    parameter["level_handle_size"][0] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_1 = (
                    parameter["level_handle_size"][1] / 5 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_2 = (
                    parameter["level_handle_size"][2] * randRange(1, 0.8, 1.5)[0]
                )
                separation_distance = (
                    (parameter["level_handle_size"][0] - support_size_0 * 2)
                    / 2
                    * randRange(1, 0.5, 0.9)[0]
                )
                if random.random() < 0.5:
                    support_size_0 = (
                        parameter["level_handle_size"][0]
                        / 2
                        * randRange(1, 0.7, 0.95)[0]
                    )
                    separation_distance = 0
                parameter["level_support_size"][0] = support_size_0
                parameter["level_support_size"][1] = support_size_1
                parameter["level_support_size"][2] = support_size_2
                parameter["level_support_seperation"][0] = separation_distance
                parameter["level_handle_offset"][0] = (
                    parameter["level_handle_size"][1] / 2 * randRange(1, -0.5, 0.3)[0]
                )
                max_rotation_angle = (
                    np.arctan(
                        parameter["level_support_size"][2]
                        / parameter["level_handle_size"][1]
                    )
                    * 180
                    / np.pi
                )
                parameter["level_handle_rotation"][0] = (
                    max_rotation_angle * randRange(1, 0.8, 1.2)[0]
                )

            elif clip_type == "streamline-clip":
                parameter["level_handle_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["level_handle_size"][1] = (
                    parameter["level_handle_size"][0]
                    * np.random.uniform(2, 3)
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["level_handle_size"][2] = (
                    parameter["level_handle_size"][0] / 2 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_0 = (
                    parameter["level_handle_size"][0] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_1 = (
                    parameter["level_handle_size"][1] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_2 = (
                    parameter["level_handle_size"][1] / 3 * randRange(1, 0.8, 1.1)[0]
                )
                separation_distance = (
                    (parameter["level_handle_size"][0] - support_size_0 * 2)
                    / 2
                    * randRange(1, 0.5, 0.9)[0]
                )
                if random.random() < 0.5:
                    support_size_0 = (
                        parameter["level_handle_size"][0]
                        / 2
                        * randRange(1, 0.7, 0.95)[0]
                    )
                    separation_distance = 0
                parameter["level_support_size"][0] = support_size_0
                parameter["level_support_size"][1] = support_size_1
                parameter["level_support_size"][2] = support_size_2
                parameter["level_support_seperation"][0] = separation_distance
                parameter["level_handle_offset"][0] = (
                    parameter["level_handle_size"][1] / 2 * randRange(1, -0.5, 0)[0]
                )
                max_rotation_angle = 25
                parameter["level_handle_rotation"][0] = (
                    max_rotation_angle * randRange(1, 0.8, 1.2)[0]
                )

            elif clip_type == "T-A-clip":
                parameter["level_handle_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["level_handle_size"][1] = (
                    parameter["level_handle_size"][0]
                    * 8
                    / 5
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["level_handle_size"][2] = (
                    parameter["level_handle_size"][0] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_0 = (
                    parameter["level_handle_size"][0] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_1 = (
                    parameter["level_handle_size"][1] / 5 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_2 = (
                    parameter["level_handle_size"][2] * randRange(1, 0.8, 1.5)[0]
                )
                separation_distance = (
                    (parameter["level_handle_size"][0] - support_size_0 * 2)
                    / 2
                    * randRange(1, 0.5, 0.9)[0]
                )
                if random.random() < 0.5:
                    support_size_0 = (
                        parameter["level_handle_size"][0]
                        / 2
                        * randRange(1, 0.7, 0.95)[0]
                    )
                    separation_distance = 0
                parameter["level_support_size"][0] = support_size_0
                parameter["level_support_size"][1] = support_size_1
                parameter["level_support_size"][2] = support_size_2
                parameter["level_support_seperation"][0] = separation_distance
                parameter["level_handle_offset"][0] = (
                    parameter["level_handle_size"][1] / 2 * randRange(1, -0.5, 0)[0]
                )
                max_rotation_angle = (
                    np.arctan(
                        parameter["level_support_size"][2]
                        / parameter["level_handle_size"][1]
                    )
                    * 180
                    / np.pi
                )
                parameter["level_handle_rotation"][0] = (
                    max_rotation_angle * randRange(1, 0.8, 0.9)[0]
                )

            elif clip_type == "T-streamline-clip":
                parameter["level_handle_size"][0] *= randRange(1, 0.8, 1.2)[0]
                parameter["level_handle_size"][1] = (
                    parameter["level_handle_size"][0] * 2 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["level_handle_size"][2] = (
                    parameter["level_handle_size"][0] / 5 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_0 = (
                    parameter["level_handle_size"][0] / 8 * randRange(1, 0.8, 1.2)[0]
                )
                support_size_1 = (
                    parameter["level_handle_size"][1] / 3 * randRange(1, 0.7, 1.2)[0]
                )
                support_size_2 = (
                    parameter["level_handle_size"][1] / 3 * randRange(1, 0.8, 1.2)[0]
                )
                separation_distance = (
                    (parameter["level_handle_size"][0] - support_size_0 * 2)
                    / 2
                    * randRange(1, 0.5, 0.9)[0]
                )
                if random.random() < 0.5:
                    support_size_0 = (
                        parameter["level_handle_size"][0]
                        / 2
                        * randRange(1, 0.7, 0.95)[0]
                    )
                    separation_distance = 0
                parameter["level_support_size"][0] = support_size_0
                parameter["level_support_size"][1] = support_size_1
                parameter["level_support_size"][2] = support_size_2
                parameter["level_support_seperation"][0] = separation_distance

                parameter["level_handle_offset"][0] = (
                    -(
                        parameter["level_handle_size"][1]
                        - parameter["level_support_size"][1]
                    )
                    / 2
                    * randRange(1, 0.5, 0.95)[0]
                )
                max_rotation_angle = 20
                parameter["level_handle_rotation"][0] = (
                    max_rotation_angle * randRange(1, 0.8, 1.2)[0]
                )
            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)
            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Regular_jaw":
            lever_parameter = new_concepts[0]["parameters"]
            parameter["position"][0] = lever_parameter["position"][0]
            parameter["position"][1] = lever_parameter["position"][1]
            parameter["position"][2] = lever_parameter["position"][2]
            parameter["rotation"][0] = lever_parameter["rotation"][0]
            parameter["rotation"][1] = lever_parameter["rotation"][1]
            parameter["rotation"][2] = lever_parameter["rotation"][2]
            if clip_type == "T-A-clip":
                parameter["size"][0] = (
                    lever_parameter["level_handle_size"][0]
                    * 4
                    * randRange(1, 0.5, 1.2)[0]
                )
                parameter["size"][1] = (
                    lever_parameter["level_handle_size"][1]
                    / 2
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = (
                    lever_parameter["level_handle_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                additional_offset_y = (
                    lever_parameter["level_handle_size"][1]
                    * np.cos(lever_parameter["level_handle_rotation"][0] / 180 * np.pi)
                    / 2
                    + lever_parameter["level_handle_offset"][0]
                )
                parameter["position"][1] += (
                    additional_offset_y
                    + parameter["size"][1] / 2 * randRange(1, 0.8, 0.9)[0]
                )
                parameter["jaw_separation"][0] = lever_parameter["level_support_size"][
                    2
                ] - 2 * (
                    lever_parameter["level_support_size"][1] / 2
                    - lever_parameter["level_handle_offset"][0]
                ) * np.sin(lever_parameter["level_handle_rotation"][0] / 180 * np.pi)
                parameter["jaw_separation"][0] = np.maximum(
                    parameter["jaw_separation"][0],
                    parameter["size"][2] / 2 * randRange(1, 0.8, 1.2)[0],
                )
                max_rotation_angle = (
                    np.arcsin(
                        parameter["jaw_separation"][0] / (2 * parameter["size"][1])
                    )
                    * 180
                    / np.pi
                )
                parameter["jaw_rotation"][0] = (
                    max_rotation_angle * randRange(1, 0.8, 1.0)[0]
                )
            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)
            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Curved_jaw":
            lever_parameter = new_concepts[0]["parameters"]
            parameter["position"][0] = lever_parameter["position"][0]
            parameter["position"][1] = lever_parameter["position"][1]
            parameter["position"][2] = lever_parameter["position"][2]
            parameter["rotation"][0] = lever_parameter["rotation"][0]
            parameter["rotation"][1] = lever_parameter["rotation"][1]
            parameter["rotation"][2] = lever_parameter["rotation"][2]
            if clip_type == "streamline-clip":
                thickness = (
                    lever_parameter["level_handle_size"][2] * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][0] *= randRange(1, 0.8, 1.2)
                parameter["size"][1] = parameter["size"][0] - thickness
                parameter["size"][2] = lever_parameter["level_handle_size"][0] * (
                    randRange(1, 1.2, 1.3)[0]
                    if random.random() < 0.5
                    else randRange(1, 0.8, 1.2)[0]
                )
                additional_offset_y = (
                    lever_parameter["level_handle_size"][1]
                    * np.cos(lever_parameter["level_handle_rotation"][0] / 180 * np.pi)
                    / 2
                    + lever_parameter["level_handle_offset"][0]
                )
                parameter["jaw_separation"][0] = (
                    (
                        lever_parameter["level_support_size"][2]
                        + lever_parameter["level_handle_size"][2]
                    )
                    / 2
                    - lever_parameter["level_handle_offset"][0]
                    * np.sin(lever_parameter["level_handle_rotation"][0] / 180 * np.pi)
                    - lever_parameter["level_handle_size"][1]
                    / 2
                    * np.sin(lever_parameter["level_handle_rotation"][0] / 180 * np.pi)
                )
                parameter["position"][1] += additional_offset_y
                avg_radius = (parameter["size"][0] + parameter["size"][1]) / 2
                parameter["central_angle"][0] = 90 * randRange(1, 0.8, 1.2)[0]
                parameter["jaw_rotation"][0] = (
                    -get_target_angle(
                        parameter["jaw_separation"][0],
                        avg_radius,
                        parameter["central_angle"][0],
                    )
                    * 180
                    / np.pi
                )

            elif clip_type == "T-streamline-clip":
                thickness = (
                    lever_parameter["level_handle_size"][2]
                    / 2
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][0] = (
                    lever_parameter["level_handle_size"][1]
                    / 3
                    * randRange(1, 0.8, 1.0)[0]
                )
                parameter["size"][1] = parameter["size"][0] - thickness
                parameter["size"][2] = (
                    lever_parameter["level_handle_size"][0]
                    * np.random.uniform(2, 4)
                    * randRange(1, 0.8, 1.2)[0]
                )
                additional_offset_y = (
                    lever_parameter["level_handle_size"][1]
                    * np.cos(lever_parameter["level_handle_rotation"][0] / 180 * np.pi)
                    / 2
                    + lever_parameter["level_handle_offset"][0]
                )

                parameter["jaw_separation"][0] = (
                    (
                        lever_parameter["level_support_size"][2]
                        + lever_parameter["level_handle_size"][2]
                    )
                    / 2
                    - lever_parameter["level_handle_offset"][0]
                    * np.sin(lever_parameter["level_handle_rotation"][0] / 180 * np.pi)
                    - lever_parameter["level_handle_size"][1]
                    / 2
                    * np.sin(lever_parameter["level_handle_rotation"][0] / 180 * np.pi)
                )
                parameter["position"][1] += additional_offset_y
                parameter["central_angle"][0] = 90 * randRange(1, 0.8, 1.0)[0]
                avg_radius = (parameter["size"][0] + parameter["size"][1]) / 2
                parameter["jaw_rotation"][0] = (
                    -get_target_angle(
                        parameter["jaw_separation"][0],
                        avg_radius,
                        parameter["central_angle"][0],
                    )
                    * 180
                    / np.pi
                )

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
        clip_type = get_clip_type()
        existing_concept_templates = concept_template_existence(clip_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, clip_type)

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
            clip_type = get_clip_type()
            existing_concept_templates = concept_template_existence(clip_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, clip_type)
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
