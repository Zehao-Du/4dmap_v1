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


def get_bottle_type():
    total_type = ["wine", "juice", "water", "others"]
    weights = [1, 1, 1, 1]
    bottle_type = random.choices(total_type, weights=weights, k=1)[0]
    return bottle_type


def concept_template_existence(bottle_type):
    concept_template_variation = {
        "body": {"template": ["Multilevel_Body"], "necessary": True},
        "lid": {"template": ["Cylindrical_Lid"], "necessary": True},
    }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        else:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, bottle_type):
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Multilevel_Body":
            total_body_height = 0
            for i in range(4):
                total_body_height += parameter["level_" + str(i + 1) + "_size"][1]
            size_mul = randRange(1, 0.6, 1.4)
            level_1_size = parameter["level_1_size"] * size_mul
            level_2_size = parameter["level_2_size"] * size_mul
            level_3_size = parameter["level_3_size"] * size_mul
            level_4_size = parameter["level_4_size"] * size_mul

            if bottle_type == "wine":
                num_of_level = np.random.randint(3, 5)
                height_wine_mul = randRange(1, 0.8, 1.5)[0]
                height_neck_mul = randRange(1, 0.6, 1.2)[0]
                height_mouth_mul = randRange(1, 0.8, 1.2)[0]
                avg_height = total_body_height * randRange(1, 1.1, 1.5)[0]
                wine_body_height = avg_height * 2 / 4 * height_wine_mul
                wine_neck_height = avg_height / 4 * height_neck_mul
                wine_mouth_height = avg_height / 4 * height_mouth_mul
                if num_of_level == 3:
                    level_1_size[0] = np.minimum(level_1_size[0], level_1_size[2])
                    level_1_size[2] = level_1_size[0]
                    level_2_size = level_1_size * randRange(1, 0.2, 0.4)[0]
                    level_3_size = level_2_size * randRange(1, 0.9, 1.1)[0]
                    level_1_size[1] = wine_body_height
                    level_2_size[1] = wine_neck_height
                    level_3_size[1] = wine_mouth_height
                if num_of_level == 4:
                    num_level_neck = np.random.randint(1, 3)
                    if num_level_neck == 1:
                        level_2_size[0] = (
                            np.maximum(
                                level_2_size[0],
                                level_1_size[0] * randRange(1, 1.0, 1.2)[0],
                            )
                            * randRange(1, 1.1, 1.3)[0]
                        )
                        level_1_size[2] = level_2_size[0] * randRange(1, 0.5, 0.8)[0]
                        level_1_size[0] = level_2_size[0]
                        level_3_size = level_2_size * randRange(1, 0.2, 0.4)[0]
                        level_4_size = level_3_size * randRange(1, 0.9, 1.1)[0]
                        level_1_size[1] = wine_neck_height * randRange(1, 0.3, 0.6)[0]
                        level_2_size[1] = wine_body_height
                        level_3_size[1] = wine_neck_height
                        level_4_size[1] = wine_mouth_height
                    if num_level_neck == 2:
                        level_1_size[0] = np.maximum(
                            level_1_size[0], level_1_size[2] * randRange(1, 1.0, 1.3)[0]
                        )
                        level_1_size[2] = level_1_size[0] * randRange(1, 0.9, 1.1)[0]
                        level_2_size = level_1_size * randRange(1, 0.5, 0.8)[0]
                        level_3_size = level_1_size * randRange(1, 0.2, 0.4)[0]
                        level_4_size = level_3_size * randRange(1, 0.9, 1.1)[0]
                        level_1_size[1] = wine_body_height
                        level_2_size[1] = wine_neck_height * randRange(1, 0.4, 0.6)[0]
                        level_3_size[1] = wine_neck_height * randRange(1, 0.4, 0.6)[0]
                        level_4_size[1] = wine_mouth_height

            elif bottle_type == "juice":
                necessary_levels = ["b", "d"]
                optional_levels = ["a", "c"]
                all_combinations = []
                for i in range(2 ** len(optional_levels)):
                    combination = necessary_levels.copy()
                    for j in range(len(optional_levels)):
                        if (i >> j) & 1:
                            combination.append(optional_levels[j])
                    all_combinations.append(combination)
                selected_combination = random.choice(all_combinations)
                num_of_level = len(selected_combination)
                base_mul = randRange(1, 0.5, 0.8)[0]
                height_body_mul = randRange(1, 0.8, 1.5)[0]
                height_neck_mul = randRange(1, 0.8, 1.2)[0]
                avg_height = total_body_height / 3
                body_height_juice = 2 * avg_height * height_body_mul
                neck_height_juice = avg_height * height_neck_mul
                if "a" in selected_combination:
                    level_1_size[0] = (
                        np.minimum(level_2_size[0], level_1_size[0])
                        * randRange(1, 1.2, 1.4)[0]
                    )
                    level_1_size[2] = level_1_size[0] * base_mul
                    level_1_size[1] = neck_height_juice * randRange(1, 0.2, 0.4)[0]
                    if "c" in selected_combination:
                        level_2_size[0] = level_1_size[0] * randRange(1, 0.7, 0.9)[0]
                        level_3_size[0] = level_1_size[0] * randRange(1, 0.9, 1.1)[0]
                        level_3_size[1] = body_height_juice * random.uniform(0.5, 0.9)
                        level_2_size[1] = body_height_juice - level_3_size[1]
                        level_4_size[0] = level_3_size[0] * randRange(1, 0.2, 0.4)[0]
                        level_4_size[1] = neck_height_juice
                    else:
                        level_2_size[1] = body_height_juice
                        level_2_size[0] = level_1_size[0]
                        level_3_size[0] = level_2_size[0] * randRange(1, 0.2, 0.4)[0]
                        level_3_size[1] = neck_height_juice
                else:
                    level_1_size[2] = (
                        np.minimum(level_1_size[0], level_1_size[2])
                        * randRange(1, 1.2, 1.4)[0]
                    )
                    if "c" in selected_combination:
                        level_1_size[0] = level_1_size[2] * randRange(1, 0.7, 0.9)[0]
                        level_2_size[0] = level_1_size[2] * randRange(1, 0.8, 1.0)[0]
                        level_2_size[1] = body_height_juice * random.uniform(0.5, 0.9)
                        level_1_size[1] = body_height_juice - level_2_size[1]
                        level_3_size[0] = level_2_size[0] * randRange(1, 0.2, 0.4)[0]
                        level_3_size[1] = neck_height_juice
                    else:
                        level_1_size[1] = body_height_juice
                        level_1_size[0] = level_1_size[2]
                        level_2_size[0] = level_1_size[0] * randRange(1, 0.2, 0.4)[0]
                        level_2_size[1] = neck_height_juice

            elif bottle_type == "water":
                num_of_level = np.random.randint(3, 5)
                max_radius = np.maximum(level_1_size[0], level_1_size[2])
                min_radius = np.minimum(level_1_size[0], level_1_size[2])
                for idx in range(num_of_level):
                    if max_radius < locals()["level_" + str(idx + 1) + "_size"][0]:
                        max_radius = locals()["level_" + str(idx + 1) + "_size"][0]
                    if min_radius > locals()["level_" + str(idx + 1) + "_size"][0]:
                        min_radius = locals()["level_" + str(idx + 1) + "_size"][0]
                avg_height = max_radius * randRange(1, 1.5, 1.8)[0]
                body_total_height = avg_height * randRange(1, 0.7, 1.1)[0]
                for idx in range(num_of_level):
                    if idx == (num_of_level - 1):
                        locals()["level_" + str(idx + 1) + "_size"][0] = (
                            min_radius * randRange(1, 0.7, 0.9)[0]
                        )
                        locals()["level_" + str(idx + 1) + "_size"][1] = (
                            avg_height / 3 * randRange(1, 0.8, 1.2)[0]
                        )
                    elif idx == 0:
                        level_1_size[0] = (
                            locals()["level_" + str(idx + 1) + "_size"][0] * 0.4
                            + max_radius * 0.6
                        ) * randRange(1, 0.8, 1.2)[0]
                        level_1_size[1] = avg_height / 3 * randRange(1, 0.7, 1.1)[0]
                        level_1_size[2] = level_1_size[0] * randRange(1, 0.3, 0.6)[0]
                    else:
                        next_radius = (
                            locals()["level_" + str(idx) + "_size"][0] * 0.4
                            + max_radius * 0.6
                        )
                        locals()["level_" + str(idx + 1) + "_size"][0] = (
                            next_radius * randRange(1, 0.8, 1.2)[0]
                        )
                        if idx == 1:
                            locals()["level_" + str(idx + 1) + "_size"][1] = (
                                body_total_height * randRange(1, 0, 1)[0]
                            )
                            body_total_height -= locals()[
                                "level_" + str(idx + 1) + "_size"
                            ][1]
                        elif idx == 2:
                            locals()["level_" + str(idx + 1) + "_size"][1] = (
                                body_total_height
                            )

            elif bottle_type == "others":
                num_of_level = np.random.randint(1, 5)
                avg_height = total_body_height
                num_of_body_level = np.random.randint(1, num_of_level + 1)
                if num_of_level == 1:
                    level_1_size[2] = level_1_size[0] = np.maximum(
                        level_1_size[0], avg_height / 2
                    )
                    level_1_size[1] = avg_height
                else:
                    level_1_size[2] = (
                        total_body_height * 2 / 3 * randRange(1, 0.5, 0.6)[0]
                    )
                    for i in range(num_of_level):
                        if i == num_of_body_level - 1:
                            locals()["level_" + str(i + 1) + "_size"][0] = level_1_size[
                                2
                            ]
                            locals()["level_" + str(i + 1) + "_size"][1] = (
                                avg_height * 5 / 6 * randRange(1, 0.8, 1.2)[0]
                            )
                        else:
                            locals()["level_" + str(i + 1) + "_size"][0] = (
                                level_1_size[2] * randRange(1, 0.9, 1.1)[0]
                            )
                            locals()["level_" + str(i + 1) + "_size"][1] = (
                                avg_height
                                / (6 * (num_of_level - 1))
                                * randRange(1, 0.9, 1.1)[0]
                            )

            parameter["level_1_size"] = level_1_size
            parameter["level_2_size"] = level_2_size
            parameter["level_3_size"] = level_3_size
            parameter["level_4_size"] = level_4_size
            parameter["num_levels"][0] = num_of_level
            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cylindrical_Lid":
            body_parameter = concepts[0]["parameters"]
            num_of_level = int(body_parameter["num_levels"][0])
            parameter["position"][[0, 2]] = np.array(body_parameter["position"])[[0, 2]]
            total_body_height = 0
            for i in range(num_of_level):
                total_body_height += body_parameter["level_" + str(i + 1) + "_size"][1]

            if bottle_type == "wine":
                parameter["inner_size"][1] = body_parameter[
                    "level_" + str(num_of_level) + "_size"
                ][0]
                parameter["inner_size"][0] = parameter["inner_size"][1]
                parameter["inner_size"][2] = (
                    body_parameter["level_" + str(num_of_level) + "_size"][1]
                    * randRange(1, 0.2, 0.3)[0]
                )

                parameter["outer_size"][0] = (
                    parameter["inner_size"][0] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["outer_size"][1] = (
                    parameter["inner_size"][1] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["outer_size"][2] = (
                    parameter["inner_size"][2] * randRange(1, 1.1, 1.3)[0]
                )

            elif bottle_type == "juice":
                parameter["inner_size"][1] = body_parameter[
                    "level_" + str(num_of_level) + "_size"
                ][0]
                parameter["inner_size"][0] = parameter["inner_size"][1]
                parameter["inner_size"][2] = (
                    body_parameter["level_" + str(num_of_level) + "_size"][1]
                    * randRange(1, 0.2, 0.5)[0]
                )

                parameter["outer_size"][0] = (
                    parameter["inner_size"][0] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["outer_size"][1] = (
                    parameter["inner_size"][1] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["outer_size"][2] = (
                    parameter["inner_size"][2] * randRange(1, 1.1, 1.3)[0]
                )

            elif bottle_type == "water":
                parameter["inner_size"][1] = body_parameter[
                    "level_" + str(num_of_level) + "_size"
                ][0]
                parameter["inner_size"][0] = parameter["inner_size"][1]
                parameter["inner_size"][2] = (
                    body_parameter["level_" + str(num_of_level) + "_size"][1]
                    * randRange(1, 0.2, 0.3)[0]
                )

                parameter["outer_size"][0] = (
                    parameter["inner_size"][0] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["outer_size"][1] = (
                    parameter["inner_size"][1] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["outer_size"][2] = (
                    parameter["inner_size"][2] * randRange(1, 1.1, 1.3)[0]
                )

            elif bottle_type == "others":
                parameter["inner_size"][1] = (
                    body_parameter["level_" + str(num_of_level) + "_size"][0]
                    * randRange(1, 0.3, 0.5)[0]
                )
                parameter["inner_size"][0] = parameter["inner_size"][1]
                parameter["inner_size"][2] = (
                    total_body_height * randRange(1, 0.2, 0.3)[0]
                )

                parameter["outer_size"][0] = (
                    parameter["inner_size"][0] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["outer_size"][1] = (
                    parameter["inner_size"][1] * randRange(1, 1.1, 1.2)[0]
                )
                parameter["outer_size"][2] = (
                    parameter["inner_size"][2] * randRange(1, 1.1, 1.2)[0]
                )

            parameter["position"][1] = (
                -body_parameter["level_1_size"][1] / 2
                + total_body_height
                + parameter["outer_size"][2] / 2
            )
            parameter["position"][1] += body_parameter["position"][1]

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
        bottle_type = get_bottle_type()
        existing_concept_templates = concept_template_existence(bottle_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, bottle_type)

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
            bottle_type = get_bottle_type()
            existing_concept_templates = concept_template_existence(bottle_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, bottle_type)
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
