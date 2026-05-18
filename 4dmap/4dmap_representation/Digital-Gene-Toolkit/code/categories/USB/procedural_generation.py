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


def get_usb_type():
    total_type = ["regular"]
    weights = [1]
    usb_type = random.choices(total_type, weights=weights, k=1)[0]
    return usb_type


def concept_template_existence(usb_type):
    concept_template_variation = {
        "body": {"template": ["Regular_Body", "RoundEnded_Body"], "necessary": True},
        "connector": {
            "template": ["Simplied_Connector", "Regular_Connector"],
            "necessary": True,
        },
        "cap": {
            "template": ["Regular_Cap", "SquareEnded_Cap", "RoundEnded_Cap"],
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


def jitter_parameters(concepts, usb_type):
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Regular_Body":
            size_mul = randRange(parameter["size"].shape[0], 0.5, 1.5)
            parameter["size"] *= size_mul

            parameter["has_back_part"][0] = 1 if np.random.random() < 0.6 else 0
            parameter["has_side_part"][0] = 1 if np.random.random() < 0.6 else 0

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "RoundEnded_Body":
            size_mul = np.random.random(parameter["size"].shape[0]) + 0.5
            parameter["size"] *= size_mul

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Simplied_Connector":
            body_parameter = concepts[0]["parameters"]

            maximum_size_x = 0.9 * body_parameter["size"][0]
            minimum_size_x = 0.6 * body_parameter["size"][0]
            parameter["size"][0] = randRange(1, minimum_size_x, maximum_size_x)[0]
            parameter["size"][2] = randRange(1, minimum_size_x, maximum_size_x)[0]
            maximum_size_y = 0.9 * body_parameter["size"][1]
            minimum_size_y = 0.6 * body_parameter["size"][1]
            parameter["size"][1] = randRange(1, minimum_size_y, maximum_size_y)[0]

            parameter["position"] = np.array(body_parameter["position"])
            parameter["rotation"] = np.array(body_parameter["rotation"])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Regular_Connector":
            body_parameter = concepts[0]["parameters"]

            maximum_size_x = 0.9 * body_parameter["size"][0]
            minimum_size_x = 0.6 * body_parameter["size"][0]
            parameter["size"][0] = randRange(1, minimum_size_x, maximum_size_x)[0]
            parameter["size"][2] = randRange(1, minimum_size_x, maximum_size_x)[0]
            maximum_size_y = 0.9 * body_parameter["size"][1]
            minimum_size_y = 0.6 * body_parameter["size"][1]
            parameter["size"][1] = randRange(1, minimum_size_y, maximum_size_y)[0]

            maximum_thickness = 0.2 * min(parameter["size"][0], parameter["size"][1])
            parameter["thickness"][0] = randRange(1, 0, maximum_thickness)[0]

            parameter["position"] = np.array(body_parameter["position"])
            parameter["rotation"] = np.array(body_parameter["rotation"])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Regular_Cap":
            body_parameter = concepts[0]["parameters"]
            connector_parameter = concepts[1]["parameters"]

            parameter["size"][0] = body_parameter["size"][0]
            parameter["size"][1] = body_parameter["size"][1]
            minimum_size_z = connector_parameter["size"][2] * 1.2
            maximum_size_z = connector_parameter["size"][2] * 1.5
            parameter["size"][2] = randRange(1, minimum_size_z, maximum_size_z)[0]

            parameter["inner_size"] = randRange(
                parameter["inner_size"].shape[0],
                np.array(connector_parameter["size"]),
                parameter["size"],
            )

            parameter["inner_outer_offset"] = np.zeros(
                parameter["inner_outer_offset"].shape[0]
            )
            maximum_offset_z = connector_parameter["size"][2] * 1.2
            minimum_offset_z = connector_parameter["size"][2] * 0.2

            parameter["position"] = np.array(
                [
                    body_parameter["position"][0],
                    body_parameter["position"][1],
                    body_parameter["position"][2]
                    + randRange(1, minimum_offset_z, maximum_offset_z)[0],
                ]
            )
            parameter["rotation"] = np.array(body_parameter["rotation"])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "SquareEnded_Cap":
            body_parameter = concepts[0]["parameters"]
            connector_parameter = concepts[1]["parameters"]

            parameter["cap_rotation"][0] = randRange(1, -180.0, 180.0)[0]
            virtual_shaft_offset = (
                randRange(
                    1, -body_parameter["size"][2] / 4, body_parameter["size"][2] / 4
                )[0]
                - body_parameter["size"][2] / 2
            )

            parameter["size"][0] = randRange(1, 1.0, 1.2)[0] * body_parameter["size"][0]
            parameter["size"][1] = randRange(1, 1.0, 1.5)[0] * parameter["size"][1]

            parameter["position"] = np.array(
                [
                    body_parameter["position"][0],
                    body_parameter["position"][1],
                    body_parameter["position"][2]
                    + virtual_shaft_offset
                    + parameter["size"][0] / 2,
                ]
            )
            parameter["rotation"] = np.array(body_parameter["rotation"])

            if concepts[0]["template"] == "Regular_Body":
                size_z_back_val = np.linalg.norm(
                    [
                        parameter["position"][2]
                        - parameter["size"][0] / 2
                        - (
                            -body_parameter["size"][2]
                            - (
                                body_parameter["size"][1]
                                / 2
                                * body_parameter["has_back_part"][0]
                            )
                        ),
                        body_parameter["size"][0],
                    ]
                )
                size_z_front_val = np.linalg.norm(
                    [
                        parameter["position"][2]
                        - parameter["size"][0] / 2
                        - connector_parameter["size"][2],
                        body_parameter["size"][0],
                    ]
                )
            elif concepts[0]["template"] == "RoundEnded_Body":
                size_z_back_val = np.linalg.norm(
                    [
                        parameter["position"][2]
                        - parameter["size"][0] / 2
                        - (
                            -body_parameter["size"][2] - (body_parameter["size"][0] / 2)
                        ),
                        body_parameter["size"][0],
                    ]
                )
                size_z_front_val = np.linalg.norm(
                    [
                        parameter["position"][2]
                        - parameter["size"][0] / 2
                        - connector_parameter["size"][2],
                        body_parameter["size"][0],
                    ]
                )
            parameter["size"][2] = (
                np.max([size_z_back_val, size_z_front_val]) * randRange(1, 1.0, 1.2)[0]
            )

            parameter["proximal_interval"][0] = (
                randRange(1, 1.0, 1.1)[0] * body_parameter["size"][1]
            )

            parameter["inclination"][0] = randRange(1, 0.0, 5.0)[0]

            parameter["has_shaft"][0] = 1
            if parameter["has_shaft"][0]:
                maximum_shaft_radius = min(
                    0.9 * body_parameter["size"][0] / 2,
                    abs(parameter["position"][2] - body_parameter["size"][0] / 2),
                )
                parameter["shaft_size"][0] = (
                    randRange(1, 0.6, 1.0)[0] * maximum_shaft_radius
                )
                parameter["shaft_size"][1] = randRange(1, 1.0, 1.3)[0] * max(
                    (parameter["proximal_interval"][0] - body_parameter["size"][1])
                    + parameter["size"][1],
                    0.02,
                )
                parameter["shaft_offset"][0] = -parameter["size"][0] / 2
                parameter["shaft_interval"] = np.array(body_parameter["size"][1])
            else:
                parameter["shaft_size"][:] = 0.0
                parameter["shaft_offset"][0] = 0.0
                parameter["shaft_interval"] = np.array(0.0)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "RoundEnded_Cap":
            body_parameter = concepts[0]["parameters"]
            connector_parameter = concepts[1]["parameters"]

            parameter["cap_rotation"][0] = randRange(1, -180.0, 180.0)[0]
            virtual_shaft_offset = (
                randRange(
                    1, -body_parameter["size"][2] / 4, body_parameter["size"][2] / 4
                )[0]
                - body_parameter["size"][2] / 2
            )

            parameter["size"][0] = randRange(1, 1.0, 1.2)[0] * body_parameter["size"][0]
            parameter["size"][1] = randRange(1, 1.0, 1.5)[0] * parameter["size"][1]

            parameter["position"] = np.array(
                [
                    body_parameter["position"][0],
                    body_parameter["position"][1],
                    body_parameter["position"][2]
                    + virtual_shaft_offset
                    + parameter["size"][0] / 2,
                ]
            )
            parameter["rotation"] = np.array(body_parameter["rotation"])

            if concepts[0]["template"] == "Regular_Body":
                size_z_back_val = np.linalg.norm(
                    [
                        parameter["position"][2]
                        - parameter["size"][0] / 2
                        - (
                            -body_parameter["size"][2]
                            - (
                                body_parameter["size"][1]
                                / 2
                                * body_parameter["has_back_part"][0]
                            )
                        ),
                        body_parameter["size"][0],
                    ]
                )
                size_z_front_val = np.linalg.norm(
                    [
                        parameter["position"][2]
                        - parameter["size"][0] / 2
                        - connector_parameter["size"][2],
                        body_parameter["size"][0],
                    ]
                )
            elif concepts[0]["template"] == "RoundEnded_Body":
                size_z_back_val = np.linalg.norm(
                    [
                        parameter["position"][2]
                        - parameter["size"][0] / 2
                        - (
                            -body_parameter["size"][2] - (body_parameter["size"][0] / 2)
                        ),
                        body_parameter["size"][0],
                    ]
                )
                size_z_front_val = np.linalg.norm(
                    [
                        parameter["position"][2]
                        - parameter["size"][0] / 2
                        - connector_parameter["size"][2],
                        body_parameter["size"][0],
                    ]
                )
            parameter["size"][2] = (
                np.max([size_z_back_val, size_z_front_val]) * randRange(1, 1.0, 1.2)[0]
            )

            parameter["proximal_interval"][0] = (
                randRange(1, 1.0, 1.1)[0] * body_parameter["size"][1]
            )

            parameter["inclination"][0] = randRange(1, 0.0, 5.0)[0]

            parameter["has_shaft"][0] = 1
            if parameter["has_shaft"][0]:
                maximum_shaft_radius = min(
                    0.9 * body_parameter["size"][0] / 2,
                    abs(parameter["position"][2] - body_parameter["size"][0] / 2),
                )
                parameter["shaft_size"][0] = (
                    randRange(1, 0.6, 1.0)[0] * maximum_shaft_radius
                )
                parameter["shaft_size"][1] = randRange(1, 1.0, 1.3)[0] * max(
                    (parameter["proximal_interval"][0] - body_parameter["size"][1]) / 2
                    + parameter["size"][1],
                    0.02,
                )
                parameter["shaft_offset"][0] = -parameter["size"][0] / 2
                parameter["shaft_interval"] = np.array(body_parameter["size"][1])
            else:
                parameter["shaft_size"][:] = 0.0
                parameter["shaft_offset"][0] = 0.0
                parameter["shaft_interval"] = np.array(0.0)

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
        usb_type = get_usb_type()
        existing_concept_templates = concept_template_existence(usb_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, usb_type)

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
            usb_type = get_usb_type()
            existing_concept_templates = concept_template_existence(usb_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, usb_type)
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
