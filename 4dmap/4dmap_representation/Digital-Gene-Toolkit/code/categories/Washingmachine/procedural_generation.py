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


def get_washingmachine_type():
    total_type = ["horizontal_roller", "vertical_roller"]
    weights = [1, 1]
    washingmachine_type = random.choices(total_type, weights=weights, k=1)[0]
    return washingmachine_type


def concept_template_existence(washingmachine_type):
    if washingmachine_type == "horizontal_roller":
        concept_template_variation = {
            "body": {"template": ["Front_Facing_Roller_Body"], "necessary": True},
            "door": {"template": ["Roller_Door"], "necessary": True},
            "controller": {"template": ["Controller_With_Button"], "necessary": True},
        }
    elif washingmachine_type == "vertical_roller":
        concept_template_variation = {
            "body": {
                "template": ["Upright_Roller_Body", "Cuboidal_Body"],
                "necessary": True,
            },
            "door": {"template": ["Cuboidal_Door"], "necessary": True},
            "controller": {"template": ["Controller_With_Button"], "necessary": True},
        }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, washingmachine_type):
    new_concepts = []

    body_param = None
    body_type = None
    door_param = None

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if washingmachine_type == "horizontal_roller":
            if template == "Front_Facing_Roller_Body":
                parameter["outer_size"] = (
                    randRange(3, 0.7, 1.3) * parameter["outer_size"]
                )
                parameter["inner_size"] = np.array(
                    [
                        randRange(1, 0.35, 0.45)[0]
                        * min(parameter["outer_size"][0], parameter["outer_size"][1]),
                        randRange(1, 0.6, 0.8)[0] * parameter["outer_size"][2],
                    ]
                )
                parameter["inner_offset"] = np.array([0, 0])

                parameter["position"] = np.array([0, 0, 0])
                parameter["rotation"] = np.array([0, 0, 0])

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)
                body_param = parameter
                body_type = "Front_Facing_Roller_Body"

            elif template == "Roller_Door":
                parameter["circle_size"] = np.array(
                    [
                        body_param["inner_size"][0],
                        body_param["inner_size"][0] * randRange(1, 0.85, 0.98)[0],
                        parameter["circle_size"][2] * randRange(1, 0.7, 1.3)[0],
                    ]
                )
                parameter["middle_size"] = parameter["middle_size"] * randRange(
                    2, 0.7, 1.3
                )
                parameter["middle_offset"] = parameter["middle_offset"] * randRange(
                    2, 0.7, 1.3
                )

                parameter["rotation"] = np.array([0, randRange(1, -90, 0)[0], 0])
                parameter["position"] = np.array(
                    [
                        body_param["inner_offset"][0]
                        - parameter["circle_size"][0]
                        + np.sin(parameter["rotation"][1] * np.pi / 180)
                        * parameter["circle_size"][2]
                        / 2,
                        body_param["inner_offset"][1],
                        body_param["outer_size"][2] / 2
                        - np.cos(parameter["rotation"][1] * np.pi / 180)
                        * parameter["circle_size"][2]
                        / 2,
                    ]
                )

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)
                door_param = parameter

        elif washingmachine_type == "vertical_roller":
            if template == "Cuboidal_Body":
                parameter["outer_size"] = (
                    randRange(3, 0.7, 1.3) * parameter["outer_size"]
                )
                parameter["thickness"] = (
                    randRange(3, 0.05, 0.1) * parameter["outer_size"]
                )

                parameter["position"] = np.array([0, 0, 0])
                parameter["rotation"] = np.array([0, 0, 0])

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)
                body_param = parameter
                body_type = "Cuboidal_Body"

            elif template == "Upright_Roller_Body":
                parameter["outer_size"] = (
                    randRange(3, 0.7, 1.3) * parameter["outer_size"]
                )
                parameter["inner_size"] = np.array(
                    [
                        randRange(1, 0.35, 0.45)[0]
                        * min(parameter["outer_size"][0], parameter["outer_size"][2]),
                        randRange(1, 0.6, 0.8)[0] * parameter["outer_size"][1],
                    ]
                )
                parameter["inner_offset"] = np.array([0, 0])

                parameter["position"] = np.array([0, 0, 0])
                parameter["rotation"] = np.array([0, 0, 0])

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)
                body_param = parameter
                body_type = "Upright_Roller_Body"

            elif template == "Cuboidal_Door":
                if body_type == "Cuboidal_Body":
                    parameter["size"] = np.array(
                        [
                            body_param["outer_size"][0],
                            body_param["outer_size"][1] * randRange(1, 0.01, 0.025)[0],
                            body_param["outer_size"][2] * randRange(1, 0.8, 0.9)[0],
                        ]
                    )

                    door_rotation_type = random.choice(["front", "left", "right"])
                    door_rotation_type = "front"
                    if door_rotation_type == "front":
                        parameter["rotation"] = np.array(
                            [randRange(1, -90, 0)[0], 0, 0]
                        )
                        parameter["position"] = np.array(
                            [
                                0,
                                body_param["outer_size"][1] / 2
                                + parameter["size"][2]
                                * np.sin(-parameter["rotation"][0] * np.pi / 180)
                                / 2,
                                body_param["outer_size"][2] / 2
                                - parameter["size"][2]
                                * (
                                    1
                                    - np.cos(parameter["rotation"][0] * np.pi / 180) / 2
                                ),
                            ]
                        )
                    elif door_rotation_type == "left":
                        parameter["rotation"] = np.array([0, 0, randRange(1, 0, 90)[0]])
                        parameter["position"] = np.array(
                            [
                                -parameter["size"][0]
                                / 2
                                * (1 - np.cos(parameter["rotation"][2] * np.pi / 180)),
                                body_param["outer_size"][1] / 2
                                + parameter["size"][0]
                                * np.sin(parameter["rotation"][2] * np.pi / 180)
                                / 2,
                                (body_param["outer_size"][2] - parameter["size"][2])
                                / 2,
                            ]
                        )
                    elif door_rotation_type == "right":
                        parameter["rotation"] = np.array(
                            [0, 0, randRange(1, -90, 0)[0]]
                        )
                        parameter["position"] = np.array(
                            [
                                parameter["size"][0]
                                / 2
                                * (1 - np.cos(-parameter["rotation"][2] * np.pi / 180)),
                                body_param["outer_size"][1] / 2
                                + parameter["size"][0]
                                * np.sin(-parameter["rotation"][2] * np.pi / 180)
                                / 2,
                                (body_param["outer_size"][2] - parameter["size"][2])
                                / 2,
                            ]
                        )

                elif body_type == "Upright_Roller_Body":
                    parameter["size"] = np.array(
                        [
                            body_param["outer_size"][0],
                            body_param["outer_size"][1] * randRange(1, 0.01, 0.025)[0],
                            body_param["outer_size"][2] * randRange(1, 0.8, 0.9)[0],
                        ]
                    )

                    door_rotation_type = random.choice(["front", "left", "right"])
                    door_rotation_type = "front"
                    if door_rotation_type == "front":
                        parameter["rotation"] = np.array(
                            [randRange(1, -90, 0)[0], 0, 0]
                        )
                        parameter["position"] = np.array(
                            [
                                0,
                                body_param["outer_size"][1] / 2
                                + parameter["size"][2]
                                * np.sin(-parameter["rotation"][0] * np.pi / 180)
                                / 2,
                                body_param["outer_size"][2] / 2
                                - parameter["size"][2]
                                * (
                                    1
                                    - np.cos(parameter["rotation"][0] * np.pi / 180) / 2
                                ),
                            ]
                        )
                    elif door_rotation_type == "left":
                        parameter["rotation"] = np.array([0, 0, randRange(1, 0, 90)[0]])
                        parameter["position"] = np.array(
                            [
                                -parameter["size"][0]
                                / 2
                                * (1 - np.cos(parameter["rotation"][2] * np.pi / 180)),
                                body_param["outer_size"][1] / 2
                                + parameter["size"][0]
                                * np.sin(parameter["rotation"][2] * np.pi / 180)
                                / 2,
                                (body_param["outer_size"][2] - parameter["size"][2])
                                / 2,
                            ]
                        )
                    elif door_rotation_type == "right":
                        parameter["rotation"] = np.array(
                            [0, 0, randRange(1, -90, 0)[0]]
                        )
                        parameter["position"] = np.array(
                            [
                                parameter["size"][0]
                                / 2
                                * (1 - np.cos(-parameter["rotation"][2] * np.pi / 180)),
                                body_param["outer_size"][1] / 2
                                + parameter["size"][0]
                                * np.sin(-parameter["rotation"][2] * np.pi / 180)
                                / 2,
                                (body_param["outer_size"][2] - parameter["size"][2])
                                / 2,
                            ]
                        )

                concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
                new_concepts.append(concept)
                door_param = parameter

        if template == "Controller_With_Button":
            if body_type == "Cuboidal_Body":
                parameter["bottom_size"] = np.array(
                    [
                        body_param["outer_size"][0],
                        parameter["bottom_size"][1] * randRange(1, 0.7, 1.3)[0],
                        parameter["bottom_size"][2] * randRange(1, 0.7, 1.3)[0],
                        body_param["outer_size"][2] - door_param["size"][2],
                    ]
                )
                parameter["position"] = np.array(
                    [
                        0,
                        body_param["outer_size"][1] / 2
                        + parameter["bottom_size"][1] / 2,
                        -body_param["outer_size"][2] / 2,
                    ]
                )
                parameter["rotation"] = np.array([0, 0, 0])

            elif body_type == "Upright_Roller_Body":
                parameter["bottom_size"] = np.array(
                    [
                        body_param["outer_size"][0],
                        parameter["bottom_size"][1] * randRange(1, 0.7, 1.3)[0],
                        parameter["bottom_size"][2] * randRange(1, 0.7, 1.3)[0],
                        body_param["outer_size"][2] - door_param["size"][2],
                    ]
                )
                parameter["position"] = np.array(
                    [
                        0,
                        body_param["outer_size"][1] / 2
                        + parameter["bottom_size"][1] / 2,
                        -body_param["outer_size"][2] / 2,
                    ]
                )
                parameter["rotation"] = np.array([0, 0, 0])

            elif body_type == "Front_Facing_Roller_Body":
                parameter["bottom_size"] = np.array(
                    [
                        body_param["outer_size"][0],
                        (
                            body_param["outer_size"][1] / 2
                            - door_param["circle_size"][0]
                            - body_param["inner_offset"][1]
                        )
                        * randRange(1, 0.7, 1)[0],
                        body_param["outer_size"][2] * randRange(1, 0.01, 0.1)[0],
                        body_param["outer_size"][2] * randRange(1, 0.1, 0.15)[0],
                    ]
                )
                parameter["position"] = np.array(
                    [
                        0,
                        body_param["outer_size"][1] / 2
                        + parameter["bottom_size"][1] / 2,
                        -body_param["outer_size"][2] / 2,
                    ]
                )
                parameter["rotation"] = np.array([0, 0, 0])

            parameter["num_buttons"] = [np.random.randint(2, 5)]
            button_panel_length = (
                parameter["bottom_size"][0] * randRange(1, 0.4, 0.8)[0]
            )
            button_size = np.array(
                [
                    button_panel_length / (parameter["num_buttons"][0] + 1) / 2,
                    parameter["bottom_size"][1] * randRange(1, 0.1, 0.24)[0],
                    parameter["button_1_size"][2] * randRange(1, 0.8, 1.2)[0],
                ]
            )
            for button_idx in range(parameter["num_buttons"][0]):
                parameter["button_%d_size" % (button_idx + 1)] = button_size
                parameter["button_%d_offset" % (button_idx + 1)] = np.array(
                    [
                        -button_panel_length / 2
                        + button_panel_length
                        / (parameter["num_buttons"][0] + 1)
                        * (button_idx + 1),
                        0,
                    ]
                )

            concept["parameters"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in parameter.items()
            }
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
        washingmachine_type = get_washingmachine_type()
        existing_concept_templates = concept_template_existence(washingmachine_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, washingmachine_type)

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
            washingmachine_type = get_washingmachine_type()
            existing_concept_templates = concept_template_existence(washingmachine_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, washingmachine_type)
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
