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


def get_box_type():
    total_type = ["chest", "carton"]
    weights = [1, 1]
    box_type = random.choices(total_type, weights=weights, k=1)[0]
    return box_type


def concept_template_existence(box_type):
    concept_template_variation = {
        "body": {"template": ["Cuboidal_Body"], "necessary": True},
        "cover": {"template": ["Fourfold_Cover", "Regular_Cover"], "necessary": True},
        "leg": {"template": ["Cuboidal_Leg"], "necessary": False},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part_name == "cover":
            if box_type == "chest":
                templates.append("Regular_Cover")
            elif box_type == "carton":
                templates.append("Fourfold_Cover")
        else:
            if not part["necessary"]:
                if random.random() < 0.5:
                    templates.append(random.choice(part["template"]))
            else:
                templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, box_type):
    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Cuboidal_Body":
            size_mul_top = randRange(parameter["top_size"].shape[0], 0.7, 1.3)
            size_diff = parameter["top_size"] - parameter["bottom_size"]
            size_mul_top_bottom_diff = randRange(size_diff.shape[0], 0.9, 1.1)
            parameter["top_size"] = parameter["top_size"] * size_mul_top
            parameter["bottom_size"] = (
                parameter["top_size"] + size_diff * size_mul_top_bottom_diff
            )

            mul_thickness = randRange(parameter["thickness"].shape[0], 0.7, 1.3)
            mul_height = randRange(parameter["height"].shape[0], 0.7, 1.3)
            parameter["thickness"] = parameter["thickness"] * mul_thickness
            parameter["height"] = parameter["height"] * mul_height

            add_top_bottom_offset = randRange(
                parameter["top_bottom_offset"].shape[0], -0.05, 0.05
            )
            parameter["top_bottom_offset"] = (
                parameter["top_bottom_offset"] + add_top_bottom_offset
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Fourfold_Cover":
            body_parameter = concepts[0]["parameters"]

            parameter["has_cover"] = np.array(
                [1 if np.random.random() < 0.8 else 0 for _ in range(4)]
            )

            mul_cover_width = randRange(1, 0.7, 1.3)[0]
            parameter["front_behind_size"][0] = body_parameter["top_size"][0]
            if parameter["has_cover"][0] + parameter["has_cover"][1] < 2:
                parameter["front_behind_size"][1] = body_parameter["top_size"][1]
            else:
                parameter["front_behind_size"][1] = (
                    randRange(1, 0.5, 0.7)[0] * body_parameter["top_size"][1]
                )
            parameter["front_behind_size"][2] *= mul_cover_width

            parameter["left_right_size"][2] = body_parameter["top_size"][1]
            if parameter["has_cover"][2] + parameter["has_cover"][3] < 2:
                parameter["left_right_size"][1] = body_parameter["top_size"][0]
            else:
                parameter["left_right_size"][1] = (
                    randRange(1, 0.5, 0.7)[0] * body_parameter["top_size"][0]
                )
            parameter["left_right_size"][0] *= mul_cover_width

            mul_cover_rotation = randRange(
                parameter["cover_rotation"].shape[0], -0.5, 1.5
            )
            parameter["cover_rotation"] *= mul_cover_rotation

            parameter["cover_separation"] = np.array(body_parameter["top_size"])[[1, 0]]

            parameter["position"][[0, 2]] = (
                np.array(body_parameter["top_bottom_offset"])
                + np.array(body_parameter["position"])[[0, 2]]
            )
            parameter["position"][1] = (
                body_parameter["height"][0] / 2 + body_parameter["position"][1]
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Regular_Cover":
            body_parameter = concepts[0]["parameters"]

            size_diff = parameter["outer_size"] - parameter["inner_size"]
            size_mul_diff = randRange(size_diff.shape[0], 0.5, 1.5)
            parameter["outer_size"][[0, 2]] = np.array(body_parameter["top_size"])
            parameter["outer_size"][1] *= randRange(1, 0.7, 1.3)[0]
            parameter["inner_size"] = (
                parameter["outer_size"] - size_diff * size_mul_diff
            )

            parameter["position"][0] = (
                body_parameter["top_bottom_offset"][0] + body_parameter["position"][0]
            )
            parameter["position"][1] = (
                body_parameter["height"][0] / 2 + body_parameter["position"][1]
            )
            parameter["position"][2] = (
                body_parameter["top_bottom_offset"][1]
                + body_parameter["position"][2]
                - body_parameter["top_size"][1] / 2
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Cuboidal_Leg":
            body_parameter = concepts[0]["parameters"]

            parameter["num_legs"][0] = np.random.choice([1, 2, 3, 4])

            if parameter["num_legs"][0] == 1:
                parameter["front_legs_size"][0] = (
                    randRange(1, 0.8, 1.0)[0] * body_parameter["bottom_size"][0]
                )
                parameter["front_legs_size"][1] *= randRange(1, 0.8, 1.0)[0]
                parameter["front_legs_size"][2] = (
                    randRange(1, 0.8, 1.0)[0] * body_parameter["bottom_size"][1]
                )

            else:
                mul_front_leg_size = randRange(
                    parameter["front_legs_size"].shape[0], 0.7, 1.3
                )
                parameter["front_legs_size"] *= mul_front_leg_size

                mul_rear_leg_size = randRange(
                    parameter["rear_legs_size"].shape[0], 0.7, 1.3
                )
                parameter["rear_legs_size"] *= mul_rear_leg_size
                parameter["rear_legs_size"][1] = parameter["front_legs_size"][1]

            parameter["legs_separation"][0] = randRange(1, 0.9, 1.0)[0] * (
                body_parameter["bottom_size"][0] - parameter["front_legs_size"][0]
            )
            parameter["legs_separation"][1] = randRange(1, 0.9, 1.0)[0] * (
                body_parameter["bottom_size"][0] - parameter["rear_legs_size"][0]
            )
            if parameter["num_legs"][0] > 2:
                parameter["legs_separation"][2] = randRange(1, 0.9, 1.0)[0] * (
                    body_parameter["bottom_size"][1]
                    - parameter["front_legs_size"][2] / 2
                    - parameter["rear_legs_size"][2] / 2
                )
            else:
                parameter["legs_separation"][2] = 0

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = (
                body_parameter["position"][1] - body_parameter["height"][0] / 2
            )
            parameter["position"][2] = (
                body_parameter["position"][2]
                + randRange(
                    1,
                    -body_parameter["bottom_size"][1] / 2
                    - (
                        -parameter["legs_separation"][2] / 2
                        - parameter["rear_legs_size"][2] / 2
                    ),
                    body_parameter["bottom_size"][1] / 2
                    - (
                        parameter["legs_separation"][2] / 2
                        + parameter["front_legs_size"][2] / 2
                    ),
                )[0]
            )

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

    return concepts


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
        box_type = get_box_type()
        existing_concept_templates = concept_template_existence(box_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, box_type)

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
            box_type = get_box_type()
            existing_concept_templates = concept_template_existence(box_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, box_type)
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
