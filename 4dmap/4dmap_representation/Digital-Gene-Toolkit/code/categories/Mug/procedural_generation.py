import argparse
import copy
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


def get_mug_type():
    mug_type = [
        "no_cover_or_coaster",
        "has_cover",
        "has_coaster",
        "has_cover_and_coaster",
    ]
    weights = [1, 1, 1, 1]
    mug_type = random.choices(mug_type, weights=weights, k=1)[0]
    return mug_type


def concept_template_existence(mug_type):
    if mug_type == "no_cover_or_coaster":
        concept_template_variation = {
            "body": {
                "template": ["Cylindrical_Body", "Prismatic_Body", "Multilevel_Body"],
                "necessary": True,
            },
            "handle": {
                "template": ["Curved_Handle", "Trifold_Handle"],
                "necessary": True,
            },
        }
    elif mug_type == "has_cover":
        concept_template_variation = {
            "body": {
                "template": ["Cylindrical_Body", "Prismatic_Body", "Multilevel_Body"],
                "necessary": True,
            },
            "handle": {
                "template": ["Curved_Handle", "Trifold_Handle"],
                "necessary": True,
            },
            "cover": {"template": ["Single_Cylinder"], "necessary": True},
        }
    elif mug_type == "has_coaster":
        concept_template_variation = {
            "body": {
                "template": ["Cylindrical_Body", "Prismatic_Body", "Multilevel_Body"],
                "necessary": True,
            },
            "handle": {
                "template": ["Curved_Handle", "Trifold_Handle"],
                "necessary": True,
            },
            "coaster": {"template": ["Single_Cylinder"], "necessary": True},
        }
    elif mug_type == "has_cover_and_coaster":
        concept_template_variation = {
            "body": {
                "template": ["Cylindrical_Body", "Prismatic_Body", "Multilevel_Body"],
                "necessary": True,
            },
            "handle": {
                "template": ["Curved_Handle", "Trifold_Handle"],
                "necessary": True,
            },
            "cover": {"template": ["Single_Cylinder"], "necessary": True},
            "coaster": {"template": ["Single_Cylinder"], "necessary": True},
        }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part["necessary"]:
            templates.append(random.choice(part["template"]))
        elif random.random() < 0.5:
            templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, mug_type):
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    body_height = 0
    body_bottom_radius = 0
    coaster_y_position = 0
    body_top_radius = 0
    cover_y_position = 0
    define_cover = False
    define_coaster = False

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Cylindrical_Body":
            size_diff = parameter["outer_size"] - parameter["inner_size"]
            size_mul_inner = randRange(parameter["inner_size"].shape[0], 0.55, 0.85)
            size_mul_diff = randRange(size_diff.shape[0], 0.5, 1.5)
            parameter["inner_size"] = parameter["inner_size"] * size_mul_inner
            parameter["outer_size"] = (
                parameter["inner_size"] + size_diff * size_mul_diff
            )

            body_height = parameter["inner_size"][2]
            body_bottom_radius = parameter["outer_size"][1]
            coaster_y_position = -parameter["inner_size"][2] / 2
            body_top_radius = parameter["outer_size"][0]
            cover_y_position = parameter["outer_size"][2] / 2

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Prismatic_Body":
            size_diff = parameter["outer_size"] - parameter["inner_size"]
            size_mul_inner = randRange(parameter["outer_size"].shape[0], 0.85, 1.15)
            size_mul_diff = randRange(size_diff.shape[0], 0.5, 1.5)
            parameter["inner_size"] = parameter["inner_size"] * size_mul_inner
            parameter["outer_size"] = (
                parameter["inner_size"] + size_diff * size_mul_diff
            )

            body_height = parameter["outer_size"][2]
            body_bottom_radius = parameter["outer_size"][1]
            coaster_y_position = -parameter["inner_size"][2] / 2
            body_top_radius = parameter["outer_size"][0]
            cover_y_position = parameter["outer_size"][2] / 2

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Multilevel_Body":
            new_num_levels = np.random.randint(1, 5)

            old_height = 0
            old_num_levelss = parameter["num_levels"][0]
            bottom_thickness = (
                parameter["level_1_height"][0] - parameter["level_1_height"][1]
            )
            for idx in range(parameter["num_levels"][0]):
                key_height = "level_%d_height" % (idx + 1)
                if idx == 0:
                    old_height += parameter[key_height][1]
                else:
                    old_height += parameter[key_height][0]

            old_radius = parameter["level_%d_top_radius" % parameter["num_levels"][0]]

            parameter["num_levels"][0] = new_num_levels
            avg_height = old_height / new_num_levels
            for idx in range(parameter["num_levels"][0]):
                key_height = "level_%d_height" % (idx + 1)
                if idx == 0:
                    parameter[key_height][1] = copy.deepcopy(avg_height)
                    parameter[key_height][0] = avg_height + bottom_thickness
                else:
                    parameter[key_height][0] = copy.deepcopy(avg_height)
                if idx >= old_num_levelss:
                    parameter["level_%d_top_radius" % (idx + 1)] = copy.deepcopy(
                        old_radius
                    )

            for idx in range(parameter["num_levels"][0], 4):
                key_height = "level_%d_height" % (idx + 1)
                parameter[key_height][0] = 0

            old_height = (
                old_height
                - parameter["level_1_height"][1]
                + parameter["level_1_height"][0]
            )

            new_height = 0
            size_diff = np.abs(
                parameter["level_1_bottom_radius"][0]
                - parameter["level_1_bottom_radius"][1]
            )
            size_mul_inner = randRange(1, 0.9, 1.1)[0]
            size_mul_diff = randRange(1, 0.5, 1.5)[0]
            parameter["level_1_bottom_radius"][1] = (
                parameter["level_1_bottom_radius"][1] * size_mul_inner
            )
            parameter["level_1_bottom_radius"][0] = (
                parameter["level_1_bottom_radius"][1] + size_diff * size_mul_diff
            )

            size_diff = (
                parameter["level_1_top_radius"][0] - parameter["level_1_top_radius"][1]
            )
            top_radius_mul_inner = randRange(1, 0.9, 1.1)[0]
            parameter["level_1_top_radius"][1] *= top_radius_mul_inner
            top_radius_mul_diff = randRange(1, 0.5, 1.5)[0]
            parameter["level_1_top_radius"][0] = (
                parameter["level_1_top_radius"][1]
                + parameter["level_1_top_radius"][0] * size_diff * top_radius_mul_diff
            )

            size_diff = parameter["level_1_height"][0] - parameter["level_1_height"][1]
            size_mul_height_inner = randRange(1, 0.9, 1.1)[0]
            size_mul_diff = randRange(1, 0.5, 1.5)[0]
            parameter["level_1_height"][1] = (
                parameter["level_1_height"][1] * size_mul_height_inner
            )
            parameter["level_1_height"][0] = (
                parameter["level_1_height"][1] + size_diff * size_mul_diff
            )
            new_height += parameter["level_1_height"][0]

            for idx in range(1, parameter["num_levels"][0]):
                key = "level_%d_top_radius" % (idx + 1)
                key_height = "level_%d_height" % (idx + 1)
                size_diff = parameter[key][0] - parameter[key][1]
                size_mul_inner = randRange(1, 0.9, 1.1)[0]
                size_mul_diff = randRange(1, 0.5, 1.5)[0]
                parameter[key][1] = parameter[key][1] * size_mul_inner
                parameter[key][0] = parameter[key][1] + size_diff * size_mul_diff
                size_mul_height = randRange(parameter[key_height].shape[0], 0.9, 1.1)
                parameter[key_height] = parameter[key_height] * size_mul_height
                new_height += parameter[key_height][0]

            body_height = new_height
            body_bottom_radius = parameter["level_1_bottom_radius"][0]
            coaster_y_position = -parameter["level_1_height"][0] / 2
            body_top_radius = parameter[
                "level_%d_top_radius" % (parameter["num_levels"][0])
            ][0]
            cover_y_position = body_height - parameter["level_1_height"][0] / 2

            parameter["position"] = np.array([0, 0, 0])
            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Trifold_Handle":
            horizontal_thickness_mul = randRange(
                parameter["horizontal_thickness"].shape[0], 0.7, 1.3
            )
            horizontal_thickness_mul = (
                np.random.random(parameter["horizontal_thickness"].shape[0]) * 0.6 + 0.7
            )
            parameter["horizontal_thickness"] *= horizontal_thickness_mul

            horizontal_length_mul = randRange(
                parameter["horizontal_length"].shape[0], 0.7, 1.3
            )
            parameter["horizontal_length"] *= horizontal_length_mul

            parameter["vertical_thickness"][0] = parameter["horizontal_thickness"][0]
            vertical_thickness_mul = randRange(1, 0.7, 1.3)[0]
            parameter["vertical_thickness"][1] = min(
                vertical_thickness_mul * parameter["vertical_thickness"][1],
                body_height * 0.9,
            )

            parameter["horizontal_rotation"] = randRange(
                parameter["horizontal_rotation"].shape[0], -30.0, 30.0
            )

            maximum_horizontal_separation = (
                body_height - parameter["horizontal_thickness"][0]
            )
            minimum_horizontal_separation = body_height / 2
            parameter["horizontal_separation"] = randRange(
                parameter["horizontal_separation"].shape[0],
                minimum_horizontal_separation,
                maximum_horizontal_separation,
            )

            parameter["mounting_offset"] = np.array([0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Curved_Handle":
            radius_mul = randRange(1, 0.5, 1.5)[0]
            parameter["radius"][1] *= radius_mul
            maximum_radius = 0.95 * (body_height - parameter["radius"][1]) / 2
            minimum_radius = body_height / 4
            parameter["radius"][0] = randRange(1, minimum_radius, maximum_radius)[0]

            parameter["central_angle"] = np.array([180])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Single_Cylinder":
            if (
                mug_type == "has_coaster" or mug_type == "has_cover_and_coaster"
            ) and not define_coaster:
                coaster_concept = copy.deepcopy(concept)
                coaster_parameter = copy.deepcopy(parameter)

                coaster_parameter["size"][1] *= randRange(1, 0.5, 1.5)[0]
                coaster_parameter["size"][0] = (
                    body_bottom_radius * randRange(1, 1.3, 2.0)[0]
                )

                coaster_parameter["position"] = np.array(
                    [0, coaster_y_position - coaster_parameter["size"][1] / 2, 0]
                )
                coaster_parameter["rotation"] = np.array([0, 0, 0])

                coaster_concept["parameters"] = {
                    k: v.tolist() for k, v in coaster_parameter.items()
                }
                new_concepts.append(coaster_concept)

                define_coaster = True

            if (
                mug_type == "has_cover" or mug_type == "has_cover_and_coaster"
            ) and not define_cover:
                cover_concept = copy.deepcopy(concept)
                cover_parameter = copy.deepcopy(parameter)

                cover_parameter["size"][1] *= randRange(1, 0.3, 1.2)[0]
                cover_parameter["size"][0] = (
                    body_top_radius * randRange(1, 0.98, 1.02)[0]
                )

                cover_parameter["position"] = np.array(
                    [
                        body_top_radius * randRange(1, -0.05, 0.05)[0],
                        cover_y_position + cover_parameter["size"][1] / 2,
                        body_top_radius * randRange(1, -0.05, 0.05)[0],
                    ]
                )
                cover_parameter["rotation"] = np.array([0, 0, 0])

                cover_concept["parameters"] = {
                    k: v.tolist() for k, v in cover_parameter.items()
                }
                new_concepts.append(cover_concept)

                define_cover = True

    def constraints(concepts):
        templates = [concept["template"] for concept in concepts]
        indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

        if "Multilevel_Body" in templates:
            """make multiLevel_body continuous between adjacent level"""
            """new param form makes this para unnecessary"""
            pass

        if "Trifold_Handle" in templates:
            module = eval(concepts[0]["template"])
            component = module(**concepts[0]["parameters"])
            body_vertices = component.vertices
            body_faces = component.faces
            body_pointcloud = get_overall_pointcloud(body_vertices, body_faces)

            """switch handle height"""
            trifold_handle_param = concepts[indexes["Trifold_Handle"]]["parameters"]
            module = eval("Trifold_Handle")
            component = module(**trifold_handle_param)
            handle_vertices = component.vertices
            handle_faces = component.faces
            handle_pointcloud = get_overall_pointcloud(handle_vertices, handle_faces)

            handle_top = np.max(handle_pointcloud[:, 1])
            handle_bottom = np.min(handle_pointcloud[:, 1])

            body_top = np.max(body_pointcloud[:, 1])
            body_bottom = np.min(body_pointcloud[:, 1])

            if handle_top > body_top:
                minmove = handle_top - body_top
                maxmove = handle_bottom - body_bottom
                handle_move_y = np.random.random() * (maxmove - minmove) + minmove
                concepts[indexes["Trifold_Handle"]]["parameters"]["position"][1] -= (
                    handle_move_y
                )

            elif handle_bottom < body_bottom:
                maxmove = body_top - handle_top
                minmove = body_bottom - handle_bottom
                handle_move_y = np.random.random() * (maxmove - minmove) + minmove
                concepts[indexes["Trifold_Handle"]]["parameters"]["position"][1] += (
                    handle_move_y
                )

            trifold_handle_param = concepts[indexes["Trifold_Handle"]]["parameters"]
            rotation = [x / 180 * np.pi for x in trifold_handle_param["rotation"]]
            module = eval("Trifold_Handle")
            component = module(**trifold_handle_param)
            handle_vertices = component.vertices
            handle_faces = component.faces
            handle_pointcloud = get_overall_pointcloud(handle_vertices, handle_faces)
            handle_upper_end = np.mean(handle_vertices[[2, 3]], axis=0)
            handle_lower_end = np.mean(handle_vertices[[6 + 8, 7 + 8]], axis=0)

            on_face_handle_upper_end = body_pointcloud[
                np.abs(body_pointcloud[:, 1] - handle_upper_end[1])
                < trifold_handle_param["horizontal_thickness"][0] / 2
            ]
            on_face_handle_lower_end = body_pointcloud[
                np.abs(body_pointcloud[:, 1] - handle_lower_end[1])
                < trifold_handle_param["horizontal_thickness"][1] / 2
            ]

            if len(on_face_handle_upper_end) == 0:
                on_face_handle_upper_end = body_pointcloud
            if len(on_face_handle_lower_end) == 0:
                on_face_handle_lower_end = body_pointcloud

            additional_mounting_offset = np.min(
                np.linalg.norm(on_face_handle_upper_end - handle_upper_end, axis=1)
            )
            additional_lower_offset = np.min(
                np.linalg.norm(on_face_handle_lower_end - handle_lower_end, axis=1)
            )

            new_concepts = copy.deepcopy(concepts)
            new_concepts[indexes["Trifold_Handle"]]["parameters"]["position"][2] -= (
                additional_lower_offset
            )
            new_concepts[indexes["Trifold_Handle"]]["parameters"]["mounting_offset"][
                0
            ] += additional_lower_offset - additional_mounting_offset

            concepts = new_concepts

        if "Curved_Handle" in templates:
            module = eval(concepts[0]["template"])
            component = module(**concepts[0]["parameters"])
            body_vertices = component.vertices
            body_faces = component.faces
            body_pointcloud = get_overall_pointcloud(body_vertices, body_faces)

            """switch handle height"""
            curved_handle_param = concepts[indexes["Curved_Handle"]]["parameters"]
            rotation = [x / 180 * np.pi for x in curved_handle_param["rotation"]]
            module = eval("Curved_Handle")
            component = module(**curved_handle_param)
            handle_vertices = component.vertices
            handle_faces = component.faces
            handle_pointcloud = get_overall_pointcloud(handle_vertices, handle_faces)
            handle_upper_end = np.array([[0, curved_handle_param["radius"][0], 0]])
            handle_upper_end = apply_transformation(
                handle_upper_end, curved_handle_param["position"], rotation
            )
            handle_lower_end = np.array(
                [
                    [
                        0,
                        curved_handle_param["radius"][0]
                        * np.cos(curved_handle_param["central_angle"][0] / 180 * np.pi),
                        curved_handle_param["radius"][0]
                        * np.sin(curved_handle_param["central_angle"][0] / 180 * np.pi),
                    ]
                ]
            )
            handle_lower_end = apply_transformation(
                handle_lower_end, curved_handle_param["position"], rotation
            )

            handle_top = handle_upper_end[0, 1] + curved_handle_param["radius"][1]
            handle_bottom = handle_lower_end[0, 1] - curved_handle_param["radius"][1]

            body_top = np.max(body_pointcloud[:, 1])
            body_bottom = np.min(body_pointcloud[:, 1])

            if handle_top > body_top:
                minmove = handle_top - body_top
                maxmove = handle_bottom - body_bottom
                handle_move_y = np.random.random() * (maxmove - minmove) + minmove
                concepts[indexes["Curved_Handle"]]["parameters"]["position"][1] -= (
                    handle_move_y
                )

            elif handle_bottom < body_bottom:
                maxmove = body_top - handle_top
                minmove = body_bottom - handle_bottom
                handle_move_y = np.random.random() * (maxmove - minmove) + minmove
                concepts[indexes["Curved_Handle"]]["parameters"]["position"][1] += (
                    handle_move_y
                )

            """stick curved_handle to mug body"""
            new_concepts = copy.deepcopy(concepts)

            curved_handle_param = concepts[indexes["Curved_Handle"]]["parameters"]
            rotation = [x / 180 * np.pi for x in curved_handle_param["rotation"]]
            handle_upper_end = np.array([[0, curved_handle_param["radius"][0], 0]])
            handle_upper_end = apply_transformation(
                handle_upper_end, curved_handle_param["position"], rotation
            )
            handle_lower_end = np.array(
                [
                    [
                        0,
                        curved_handle_param["radius"][0]
                        * np.cos(curved_handle_param["central_angle"][0] / 180 * np.pi),
                        curved_handle_param["radius"][0]
                        * np.sin(curved_handle_param["central_angle"][0] / 180 * np.pi),
                    ]
                ]
            )
            handle_lower_end = apply_transformation(
                handle_lower_end, curved_handle_param["position"], rotation
            )

            near_body_pointcloud = body_pointcloud[
                np.abs(
                    np.linalg.norm(
                        body_pointcloud - curved_handle_param["position"], axis=1
                    )
                    - curved_handle_param["radius"][0]
                )
                < 0.02
            ]

            near2handle_upper_end = near_body_pointcloud[
                np.argmin(
                    np.linalg.norm(near_body_pointcloud - handle_upper_end, axis=1)
                )
            ]
            upper_end_z_move = near2handle_upper_end[2] - handle_upper_end[0][2]
            handle_upper_end[0][2] += upper_end_z_move
            handle_lower_end[0][2] += upper_end_z_move
            near2handle_lower_end = near_body_pointcloud[
                np.argmin(
                    np.linalg.norm(near_body_pointcloud - handle_lower_end, axis=1)
                )
            ]

            new_concepts[indexes["Curved_Handle"]]["parameters"]["position"][2] += (
                upper_end_z_move
            )

            best_central_angle = (
                360
                - np.arccos(
                    np.dot(
                        near2handle_upper_end - curved_handle_param["position"],
                        near2handle_lower_end - curved_handle_param["position"],
                    )
                    / (
                        np.linalg.norm(
                            near2handle_upper_end - curved_handle_param["position"]
                        )
                        * np.linalg.norm(
                            near2handle_lower_end - curved_handle_param["position"]
                        )
                    )
                )
                / np.pi
                * 180
            )

            new_concepts[indexes["Curved_Handle"]]["parameters"]["central_angle"] = [
                best_central_angle
            ]

            concepts = new_concepts

        return concepts

    new_concepts = constraints(new_concepts)

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
        mug_type = get_mug_type()
        existing_concept_templates = concept_template_existence(mug_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, mug_type)

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
            mug_type = get_mug_type()
            existing_concept_templates = concept_template_existence(mug_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, mug_type)
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
