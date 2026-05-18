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


def get_dispenser_type():
    dispenser_type = ["Spray", "Press"]
    weights = [1, 1]
    dispenser_type = random.choices(dispenser_type, weights=weights, k=1)[0]
    return dispenser_type


def concept_template_existence(dishwasher_type):
    if dishwasher_type == "Spray":
        nozzle_template = ["Spray_Nozzle"]
    elif dishwasher_type == "Press":
        nozzle_template = ["Press_Nozzle"]
    concept_template_variation = {
        "body": {"template": ["Cuboidal_Body", "Multilevel_Body"], "necessary": True},
        "nozzle": {"template": nozzle_template, "necessary": True},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if part["necessary"] == False:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))
        elif part["necessary"] == True:
            templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, dishwasher_type):
    body_height = 0
    body_top_size = 0
    nozzle_multiplier = 0
    new_concepts = []

    templates = [concept["template"] for concept in concepts]

    body_type = ""
    nozzle_type = ""

    for concept in concepts:
        parameters = {p: np.array(v) for p, v in concept["parameters"].items()}
        template = concept["template"]

        if template == "Multilevel_Body":
            body_type = random.choice(["bottle", "based", "steri", "vase", "slim"])
            if "Spray_Nozzle" in templates:
                body_type = "slim"

            if body_type == "bottle":
                parameters["num_levels"][0] = 3
                parameters["level_1_size"] = np.array([0.32, 0.25, 0.21])
                parameters["level_2_size"] = np.array([0.11, 0.39])
                parameters["level_3_size"] = np.array([0.086, 0.49])
                parameters["level_4_size"] = np.array([0.25, 0.2])
            elif body_type == "based":
                parameters["num_levels"][0] = 3
                parameters["level_1_size"] = np.array([0.277, 0.15, 0.375])
                parameters["level_2_size"] = np.array([0.277, 0.9])
                parameters["level_3_size"] = np.array([0.22, 0.13])
                parameters["level_4_size"] = np.array([0.25, 0.2])
            elif body_type == "steri":
                parameters["num_levels"][0] = 4
                parameters["level_1_size"] = np.array([0.35, 0.08, 0.3])
                parameters["level_2_size"] = np.array([0.241, 1.375])
                parameters["level_3_size"] = np.array([0.35, 0.25])
                parameters["level_4_size"] = np.array([0.2, 0.25])
            elif body_type == "steri":
                parameters["num_levels"][0] = 1
                parameters["level_1_size"] = np.array([0.336, 1.025, 0.336])
                parameters["level_2_size"] = np.array([0.241, 1.375])
                parameters["level_3_size"] = np.array([0.35, 0.25])
                parameters["level_4_size"] = np.array([0.2, 0.25])
            elif body_type == "slim":
                parameters["num_levels"][0] = 4
                parameters["level_1_size"] = np.array([0.29, 0.19, 0.22])
                parameters["level_2_size"] = np.array([0.29, 0.84])
                parameters["level_3_size"] = np.array([0.11, 0.22])
                parameters["level_4_size"] = np.array([0.11, 0.112])

            for i in range(parameters["num_levels"][0]):
                parameters[f"level_{i + 1}_size"] = randRange(
                    parameters[f"level_{i + 1}_size"].shape[0],
                    0.75 * parameters[f"level_{i + 1}_size"],
                    1.35 * parameters[f"level_{i + 1}_size"],
                )
                body_height += parameters[f"level_{i + 1}_size"][1]
            body_top_size = parameters["level_%d_size" % parameters["num_levels"][0]][0]
            body_height -= parameters["level_1_size"][1] * 0.5
            parameters["position"] = np.zeros(3)
            parameters["rotation"] = np.array([0, 45, 0])

        if template == "Cuboidal_Body":
            body_type = random.choice(["chubby", "slim", "regular"])

            if body_type == "chubby":
                parameters["size"] = np.array([0.84, 0.479, 1.649])
            elif body_type == "slim":
                parameters["size"] = np.array([0.437, 1.31, 0.439])
            elif body_type == "regular":
                parameters["size"] = np.array([1.13, 1.038, 1.11])

            parameters["size"] = (
                randRange(
                    parameters["size"].shape[0],
                    0.7 * parameters["size"],
                    1.3 * parameters["size"],
                )
                * random.randrange(85, 115, step=5)
                / 100
            )
            body_height = parameters["size"][1]
            body_height *= 0.5
            parameters["position"] = np.zeros(3)
            parameters["rotation"] = np.array([0, 45, 0])
            body_top_size = min(parameters["size"][0], parameters["size"][2]) / 2

        nozzle_multiplier = 0.5

        if template == "Press_Nozzle":
            nozzle_type = random.choice(["small", "tall", "large", "slim"])
            if body_type == "chubby":
                nozzle_type = "small"
            elif body_type == "regular":
                nozzle_type = "large"

            if nozzle_type == "small":
                parameters["num_levels"][0] = 2
                parameters["level_1_size"] = np.array([0.07, 0.048])
                parameters["level_2_size"] = np.array([0.1, 0.18])
                parameters["level_3_size"] = np.array([0.1, 0.05])
                parameters["level_4_size"] = np.array([0.1, 0.05])
                parameters["level_5_size"] = np.array([0.1, 0.05])
                parameters["num_nozzles"][0] = 1
                parameters["nozzle_size"] = np.array([0.12, 0.1])
                parameters["nozzle_length"] = np.array([0.01, 0.1])
                parameters["nozzle_offset"] = np.array([-0.075])
                parameters["nozzle_rotation"] = np.array([0, 0])
            elif nozzle_type == "tall":
                parameters["num_levels"][0] = 1
                parameters["level_1_size"] = np.array([0.055, 0.34])
                parameters["level_2_size"] = np.array([0.1, 0.18])
                parameters["level_3_size"] = np.array([0.1, 0.05])
                parameters["level_4_size"] = np.array([0.1, 0.05])
                parameters["level_5_size"] = np.array([0.1, 0.05])
                parameters["num_nozzles"][0] = 1
                parameters["nozzle_size"] = np.array([0.1, 0.1])
                parameters["nozzle_length"] = np.array([0.44, 0.1])
                parameters["nozzle_offset"] = np.array([-0.0535])
                parameters["nozzle_rotation"] = np.array([0, 0])
            elif nozzle_type == "large":
                parameters["num_levels"][0] = 2
                parameters["level_1_size"] = np.array([0.08, 0.102])
                parameters["level_2_size"] = np.array([0.2, 0.2])
                parameters["level_3_size"] = np.array([0.1, 0.05])
                parameters["level_4_size"] = np.array([0.1, 0.05])
                parameters["level_5_size"] = np.array([0.1, 0.05])
                parameters["num_nozzles"][0] = 1
                parameters["nozzle_size"] = np.array([0.1, 0.1])
                parameters["nozzle_length"] = np.array([0.15, 0.08])
                parameters["nozzle_offset"] = np.array([-0.075])
                parameters["nozzle_rotation"] = np.array([0, 0])
            elif nozzle_type == "slim":
                parameters["num_levels"][0] = 3
                parameters["level_1_size"] = np.array([0.08, 0.162])
                parameters["level_2_size"] = np.array([0.018, 0.12])
                parameters["level_3_size"] = np.array([0.07, 0.11])
                parameters["level_4_size"] = np.array([0.1, 0.05])
                parameters["level_5_size"] = np.array([0.1, 0.05])
                parameters["num_nozzles"][0] = 1
                parameters["nozzle_size"] = np.array([0.04, 0.02])
                parameters["nozzle_length"] = np.array([0.405, 0.1])
                parameters["nozzle_offset"] = np.array([0])
                parameters["nozzle_rotation"] = np.array([0, 0])

            delta_height = 0
            k = 0
            for i in range(1, parameters["num_levels"][0] + 1):
                parameters[f"level_{i}_size"] = randRange(
                    parameters[f"level_{i}_size"].shape[0],
                    0.7 * parameters[f"level_{i}_size"],
                    1.3 * parameters[f"level_{i}_size"],
                )

            k = body_top_size * nozzle_multiplier / parameters["level_1_size"][0]
            k = max(0.7, min(k, 1.3))

            for i in range(1, parameters["num_levels"][0] + 1):
                parameters["level_%d_size" % i] *= k

            delta_height += parameters[f"level_{i}_size"]
            parameters["num_levels"] = np.array(parameters["num_levels"])
            parameters["num_nozzles"] = np.array([random.randint(1, 2)])

            if nozzle_type == "tall":
                parameters["nozzle_size"] = max(
                    0.4 * parameters["level_%d_size" % parameters["num_levels"][0]][0],
                    randRange(1, 3, 6)[0] / 200,
                ) * np.array([1, 1])
                parameters["num_nozzles"] = np.array([2])
            else:
                parameters["nozzle_size"] = max(
                    0.2 * parameters["level_%d_size" % parameters["num_levels"][0]][1],
                    randRange(1, 3, 6)[0] / 200,
                ) * np.array([1, 1])
            parameters["nozzle_length"] = np.array(
                [
                    max(parameters["nozzle_size"][0], parameters["nozzle_size"][1])
                    * randRange(1, 3, 4.5)[0],
                    max(parameters["nozzle_size"][0], parameters["nozzle_size"][1])
                    * randRange(1, 1.5, 2.5)[0],
                ]
            )

            if parameters["num_nozzles"][0] == 1:
                parameters["nozzle_rotation"] = randRange(1, -10, 10)
            else:
                parameters["nozzle_rotation"] = np.array([0, randRange(1, 5, 20)[0]])

            parameters["nozzle_offset"] = (
                np.array(
                    [-1 * parameters["level_%d_size" % parameters["num_levels"][0]][1]]
                )
                * randRange(1, 0.5, 0.3)[0]
            )

            parameters["position"] = np.array([0, body_height, 0])
            if new_concepts[0]["template"] == "Cuboidal_Body" and body_type == "chubby":
                cusize = new_concepts[0]["parameters"]["size"]
                if random.choice([0, 1, 1]) == 1:
                    parameters["position"][cusize.index(max(cusize))] = 0.4 * max(
                        cusize
                    )
                    theta = -np.pi / 4
                    parameters["position"] = np.array(
                        [
                            [np.cos(theta), 0, -np.sin(theta)],
                            [0, 1, 0],
                            [np.sin(theta), 0, np.cos(theta)],
                        ]
                    ).dot(parameters["position"])

            parameters["rotation"] = np.array([0, 45, 0])

        if template == "Spray_Nozzle":
            nozzle_type = random.choice(["mid", "hand", "trig"])

            if nozzle_type == "mid":
                parameters["bottom_size"] = np.array([0.21, 0.14])
                parameters["middle_size"] = np.array([0.01, 0.01, 0.01])
                parameters["top_size"] = np.array([0.25, 0.207, 0.52])
                parameters["top_offset"] = np.array([0.02, 0.15])
                parameters["top_rotation"] = np.array([0])
                parameters["nozzle_size"] = np.array([0.06, 0.05])
                parameters["handle_size"] = np.array([0.115, 0.3, 0.1])
                parameters["handle_offset"] = np.array([-0.027])
                parameters["handle_rotation"] = np.array([0])
            elif nozzle_type == "hand":
                parameters["bottom_size"] = np.array([0.09, 0.12])
                parameters["middle_size"] = np.array([0.12, 0.118, 0.215])
                parameters["top_size"] = np.array([0.08, 0.064, 0.37])
                parameters["top_offset"] = np.array([0.00, 0.013])
                parameters["top_rotation"] = np.array([0])
                parameters["nozzle_size"] = np.array([0.04, 0.073])
                parameters["handle_size"] = np.array([0.03, 0.26, 0.04])
                parameters["handle_offset"] = np.array([-0.135])
                parameters["handle_rotation"] = np.array([0])
            elif nozzle_type == "trig":
                parameters["bottom_size"] = np.array([0.0581, 0.09])
                parameters["middle_size"] = np.array([0.268, 0.01, 0.15])
                parameters["top_size"] = np.array([0.268, 0.09, 0.17])
                parameters["top_offset"] = np.array([0.00, 0.01])
                parameters["top_rotation"] = np.array([0])
                parameters["nozzle_size"] = np.array([0.026, 0.044])
                parameters["handle_size"] = np.array([0.01, 0.01, 0.01])
                parameters["handle_offset"] = np.array([0])
                parameters["handle_rotation"] = np.array([0])

            delta_height = parameters["bottom_size"][1] + parameters["middle_size"][1]
            for name, val in parameters.items():
                if name == "handle_size":
                    parameters[name] = randRange(
                        parameters[name].shape[0],
                        0.5 * parameters[name],
                        1.5 * parameters[name],
                    )
                else:
                    parameters[name] = randRange(
                        parameters[name].shape[0],
                        0.7 * parameters[name],
                        1.3 * parameters[name],
                    )

            k = max(
                0.8,
                min(
                    1.2,
                    body_top_size * nozzle_multiplier / parameters["bottom_size"][0],
                ),
            )

            for name, val in parameters.items():
                parameters[name] *= k
            if nozzle_type == "mid":
                parameters["handle_rotation"] = randRange(1, -5, 5)
            else:
                parameters["handle_rotation"] = randRange(1, -20, 5)
            parameters["position"] = np.array([0, body_height, 0])
            parameters["rotation"] = np.array([0, randRange(1, 40, 50)[0], 0])
            if nozzle_type == "hand":
                parameters["top_rotation"] = randRange(1, -5, 5)
            else:
                parameters["top_rotation"] = randRange(1, -20, 5)

            if True in (parameters["handle_size"] < 1e-5):
                parameters["handle_size"] = np.zeros(3)
            else:
                if True in (parameters["middle_size"] < 1e-5):
                    while (
                        parameters["top_offset"][0]
                        + 0.5 * parameters["top_size"][2]
                        - parameters["bottom_size"][0]
                        / np.cos(parameters["top_rotation"][0] * np.pi / 180)
                        + parameters["handle_offset"][0]
                        - parameters["handle_size"][2] / 2
                        + parameters["handle_size"][1]
                        * np.sin(parameters["handle_rotation"] * np.pi / 180)
                        < 0.05
                    ):
                        parameters["handle_size"] *= 0.98
                        parameters["handle_offset"] *= 0.75
                        parameters["top_offset"] *= 1.2
                        parameters["top_size"][2] *= 1.1
                        parameters["bottom_size"] *= 0.8
                else:
                    while (
                        parameters["top_offset"][0]
                        + 0.5 * parameters["top_size"][2]
                        - parameters["middle_size"][2]
                        / 2
                        / np.cos(parameters["top_rotation"][0] * np.pi / 180)
                        + parameters["handle_offset"][0]
                        - parameters["handle_size"][2] / 2
                        + parameters["handle_size"][1]
                        * np.sin(parameters["handle_rotation"] * np.pi / 180)
                        < 0.05
                    ):
                        parameters["handle_size"] *= 0.98
                        parameters["handle_offset"] *= 0.75
                        parameters["top_offset"] *= 1.1
                        parameters["top_size"][2] *= 1.1
                        parameters["middle_size"] *= 0.8
                        parameters["bottom_size"] *= 0.8
            delta_height = parameters["bottom_size"][1] + parameters["middle_size"][1]
            while (
                parameters["handle_size"][1]
                * np.cos(parameters["top_rotation"] * np.pi / 180)
                + (
                    parameters["top_offset"][0]
                    + 0.5 * parameters["top_size"][2]
                    + parameters["handle_offset"][0]
                    - parameters["handle_size"][2] / 2
                )
                * np.sin(parameters["top_rotation"] * np.pi / 180)
                > delta_height - 0.02
            ):
                parameters["handle_size"][1] *= 0.9

            if True in (parameters["handle_size"] < 1e-5):
                parameters["handle_size"] = np.zeros(3)

        new_concepts.append(
            {
                "template": template,
                "parameters": {p: v.tolist() for p, v in parameters.items()},
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
        dispenser_type = get_dispenser_type()
        existing_concept_templates = concept_template_existence(dispenser_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, dispenser_type)

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
            dispenser_type = get_dispenser_type()
            existing_concept_templates = concept_template_existence(dispenser_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, dispenser_type)
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
