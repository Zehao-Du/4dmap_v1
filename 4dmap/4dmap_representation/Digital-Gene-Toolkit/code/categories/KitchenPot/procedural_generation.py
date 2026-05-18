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


def get_kitchenpot_type():
    total_type = ["pot", "pan"]
    weights = [1, 1]
    kitchenpot_type = random.choices(total_type, weights=weights, k=1)[0]
    return kitchenpot_type


def concept_template_existence(kitchenpot_type):
    concept_template_variation = {
        "body": {"template": ["Cylindrical_Body"], "necessary": True},
        "cover": {
            "template": [
                "Cylindrical_Cover",
                "Carved_Cylindrical_Cover",
                "Semi_Spherical_Cover",
            ],
            "necessary": True,
        },
        "tophandle": {
            "template": [
                "Cuboidal_Tophandle",
                "Trifold_Tophandle",
                "Semi_Ring_Tophandle",
                "Multilevel_Tophandle",
            ],
            "necessary": True,
        },
        "sidehandle": {
            "template": [
                "Trifold_Sidehandle",
                "L_Shaped_Sidehandle",
                "Cuboidal_Sidehandle",
            ],
            "necessary": True,
        },
    }

    templates = []

    has_cover = False
    for part_name, part in concept_template_variation.items():
        if part_name == "body":
            templates.append(random.choice(part["template"]))
        elif part_name == "cover":
            if kitchenpot_type == "pot" or random.random() < 0.5:
                has_cover = True
                templates.append(random.choice(part["template"]))
        elif part_name == "tophandle":
            if has_cover:
                templates.append(random.choice(part["template"]))
        elif part_name == "sidehandle":
            templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, kitchenpot_type):
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    body_outer_radius: float = 0
    body_inner_radius: float = 0
    body_height: float = 0
    body_difference: float = 0
    cover_height: float = 0
    cover_angle: float = 0

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if kitchenpot_type == "pot":
            if template == "Cylindrical_Body":
                parameter["outer_size"] = np.array(
                    [
                        random.uniform(0.6, 0.8),
                        random.uniform(0.6, 0.8),
                        random.uniform(0.7, 1.1),
                    ]
                )

                thickness = np.array(
                    [random.uniform(0.02, 0.12), random.uniform(0.02, 0.12)]
                )

                body_outer_radius = parameter["outer_size"][0]
                body_height = parameter["outer_size"][2]
                body_difference = (
                    parameter["outer_size"][0] - parameter["outer_size"][1]
                )
                if body_difference < 0:
                    body_difference = 0

                parameter["inner_size"] = np.array(
                    [
                        parameter["outer_size"][0] - thickness[0],
                        parameter["outer_size"][1] - thickness[0],
                        parameter["outer_size"][2] - thickness[1],
                    ]
                )
                body_inner_radius = parameter["inner_size"][0]

                parameter["position"] = np.zeros(3)
                parameter["rotation"] = np.zeros(3)

            elif template == "Cylindrical_Cover":
                parameter["size"] = np.array(
                    [
                        random.uniform(body_inner_radius, body_outer_radius),
                        random.uniform(0.012, 0.05),
                    ]
                )

                cover_height = parameter["size"][1]

                parameter["position"] = np.array(
                    [0, body_height / 2 + cover_height / 2, 0]
                )
                parameter["rotation"] = np.array([0, random.uniform(0, 360), 0])

                cover_angle = parameter["rotation"][1]

            elif template == "Carved_Cylindrical_Cover":
                parameter["outer_size"] = np.array(
                    [
                        0.0,
                        random.uniform(body_outer_radius - 0.01, body_outer_radius),
                        random.uniform(0.1, 0.14),
                    ]
                )
                parameter["outer_size"][0] = parameter["outer_size"][
                    1
                ] - random.uniform(0.012, 0.022)

                cover_height = parameter["outer_size"][2]

                scale = random.uniform(0.3, 0.5)

                parameter["inner_size"] = np.array(
                    [
                        parameter["outer_size"][0] * scale,
                        parameter["outer_size"][1] - random.uniform(0.2, 0.3),
                        parameter["outer_size"][2] * random.uniform(0.4, 0.8),
                    ]
                )
                parameter["position"] = np.array(
                    [0, body_height / 2 + cover_height / 2, 0]
                )
                parameter["rotation"] = np.array([0, random.uniform(0, 360), 0])

                cover_angle = parameter["rotation"][1]

            elif template == "Semi_Spherical_Cover":
                parameter["radius"] = np.array(
                    [body_outer_radius * random.uniform(1.5, 3)]
                )
                angle = np.arcsin(body_outer_radius / parameter["radius"][0])

                cover_height = parameter["radius"][0] * (1 - np.cos(angle)) - 0.03

                angle = np.rad2deg(angle)
                parameter["exist_angle"] = np.array([angle])

                parameter["position"] = np.array([0, body_height / 2, 0])
                parameter["rotation"] = np.array([0, random.uniform(0, 360), 0])
                cover_angle = parameter["rotation"][1]

            elif template == "Cuboidal_Tophandle":
                parameter["size"] = np.array(
                    [
                        random.uniform(0.1, 0.32),
                        random.uniform(0.1, 0.32),
                        random.uniform(0.1, 0.32),
                    ]
                )
                parameter["position"] = np.array([0, body_height / 2 + cover_height, 0])
                parameter["rotation"] = np.array([0, cover_angle, 0])

            elif template == "Trifold_Tophandle":
                parameter["mounting_size"] = np.array(
                    [
                        random.uniform(0.03, 0.14),
                        random.uniform(0.05, 0.2),
                        random.uniform(0.03, 0.14),
                    ]
                )

                parameter["mounting_seperation"] = np.array([random.uniform(0.4, 0.5)])
                parameter["grip_size"] = np.array(
                    [
                        parameter["mounting_seperation"][0]
                        + parameter["mounting_size"][0] * 2
                        + random.uniform(0.05, 0.1),
                        random.uniform(0.05, 0.07),
                        parameter["mounting_size"][2] * random.uniform(1, 1.2),
                    ]
                )

                parameter["mounting_rotation"] = np.array(
                    [(random.randint(0, 1) * 2 - 1) * random.uniform(0, 15)]
                )
                parameter["position"] = np.array([0, body_height / 2 + cover_height, 0])
                parameter["rotation"] = np.array([0, cover_angle, 0])

            elif template == "Semi_Ring_Tophandle":
                parameter["curve_size"] = np.array(
                    [random.uniform(0.12, 0.45), 0.0, random.uniform(0.05, 0.3)]
                )
                ring_width = random.uniform(0.03, 0.08)
                parameter["curve_size"][1] = parameter["curve_size"][0] - ring_width
                parameter["curve_exist_angle"] = np.array([random.uniform(100, 160)])
                exist_angle = np.deg2rad(parameter["curve_exist_angle"][0] / 2)
                parameter["position"] = np.array(
                    [
                        0,
                        body_height / 2
                        + cover_height
                        + ring_width * np.cos(exist_angle),
                        0,
                    ]
                )
                parameter["rotation"] = np.array([0, cover_angle, 0])

            elif template == "Multilevel_Tophandle":
                parameter["num_levels"] = np.array([random.randint(1, 4)])

                parameter["level_1_size"] = np.array(
                    [
                        random.uniform(0.05, 0.18),
                        random.uniform(0.08, 0.18),
                        random.uniform(0.05, 0.09),
                    ]
                )
                parameter["level_2_size"] = np.array(
                    [
                        random.uniform(0.05, 0.18),
                        parameter["level_1_size"][0],
                        random.uniform(0.05, 0.09),
                    ]
                )
                parameter["level_3_size"] = np.array(
                    [
                        random.uniform(0.05, 0.18),
                        parameter["level_2_size"][0],
                        random.uniform(0.08, 0.1),
                    ]
                )
                parameter["level_4_size"] = np.array(
                    [
                        random.uniform(0.05, 0.15),
                        parameter["level_3_size"][0],
                        random.uniform(0.08, 0.1),
                    ]
                )
                parameter["position"] = np.array([0, body_height / 2 + cover_height, 0])
                parameter["rotation"] = np.array([0, cover_angle, 0])

            elif template == "Trifold_Sidehandle":
                parameter["mounting_size"] = np.array(
                    [
                        random.uniform(0.15, 0.3),
                        random.uniform(0.02, 0.1),
                        random.uniform(0.02, 0.1),
                    ]
                )
                parameter["mounting_seperation"] = np.array([random.uniform(0.3, 0.5)])
                parameter["grip_size"] = np.array(
                    [
                        random.uniform(0.02, 0.09),
                        parameter["mounting_size"][1] * random.uniform(1, 1.2),
                        parameter["mounting_seperation"][0]
                        + parameter["mounting_size"][2] * 2
                        + random.uniform(0.05, 0.1),
                    ]
                )
                parameter["mounting_rotation"] = np.array(
                    [(random.randint(0, 1) * 2 - 1) * random.uniform(0, 15)]
                )
                parameter["handle_seperation"] = np.array(
                    [
                        body_outer_radius * 2
                        - body_difference * 0.75
                        - parameter["mounting_seperation"][0] * 0.15
                    ]
                )
                parameter["whole_rotation"] = np.array(
                    [(random.randint(0, 1) * 2 - 1) * random.uniform(0, 15)]
                )
                parameter["position"] = np.array(
                    [0, random.uniform(0.15, 0.45) * body_height, 0]
                )
                parameter["rotation"] = np.zeros(3)

            elif template == "L_Shaped_Sidehandle":
                parameter["bottom_size"] = np.array(
                    [
                        random.uniform(0.03, 0.1),
                        random.uniform(0.001, 0.2),
                        random.uniform(0.06, 0.4),
                    ]
                )
                parameter["top_size"] = np.array(
                    [
                        random.uniform(0.1, 0.3),
                        random.uniform(0.02, 0.05),
                        parameter["bottom_size"][2],
                    ]
                )
                parameter["handle_seperation"] = np.array(
                    [body_outer_radius * 2 - body_difference * 0.33]
                )
                parameter["position"] = np.array(
                    [0, random.uniform(0.1, 0.35) * body_height, 0]
                )
                parameter["rotation"] = np.zeros(3)

            elif template == "Cuboidal_Sidehandle":
                parameter["size"] = np.array(
                    [
                        random.uniform(0.5, 1),
                        random.uniform(0.1, 0.25),
                        random.uniform(0.1, 0.25),
                    ]
                )

                parameter["rotation"] = np.array([0, 0, random.uniform(0, 15)])
                parameter["position"] = np.array(
                    [
                        body_outer_radius
                        + parameter["size"][0]
                        * 0.5
                        * np.cos(np.deg2rad(parameter["rotation"][2]))
                        - body_difference * 1.4,
                        random.uniform(0.05, 0.35) * body_height
                        + parameter["size"][0]
                        * 0.5
                        * np.sin(np.deg2rad(parameter["rotation"][2])),
                        parameter["size"][2] / 2,
                    ]
                )

        elif kitchenpot_type == "pan":
            if template == "Cylindrical_Body":
                parameter["outer_size"] = np.array(
                    [random.uniform(0.6, 0.8), 0.0, random.uniform(0.25, 0.4)]
                )
                parameter["outer_size"][1] = parameter["outer_size"][
                    0
                ] * random.uniform(0.7, 1)

                thickness = np.array(
                    [random.uniform(0.02, 0.08), random.uniform(0.02, 0.08)]
                )

                body_outer_radius = parameter["outer_size"][0]
                body_height = parameter["outer_size"][2]
                body_difference = (
                    parameter["outer_size"][0] - parameter["outer_size"][1]
                )

                parameter["inner_size"] = np.array(
                    [
                        parameter["outer_size"][0] - thickness[0],
                        parameter["outer_size"][1] - thickness[0],
                        parameter["outer_size"][2] - thickness[1],
                    ]
                )
                body_inner_radius = parameter["inner_size"][0]

                parameter["position"] = np.zeros(3)
                parameter["rotation"] = np.zeros(3)

            elif template == "Cylindrical_Cover":
                parameter["size"] = np.array(
                    [
                        random.uniform(body_inner_radius, body_outer_radius),
                        random.uniform(0.012, 0.022),
                    ]
                )

                cover_height = parameter["size"][1]

                parameter["position"] = np.array(
                    [0, body_height / 2 + cover_height / 2, 0]
                )
                parameter["rotation"] = np.array([0, random.uniform(0, 360), 0])

            elif template == "Carved_Cylindrical_Cover":
                parameter["outer_size"] = np.array(
                    [
                        0.0,
                        random.uniform(body_outer_radius - 0.01, body_outer_radius),
                        random.uniform(0.1, 0.14),
                    ]
                )
                parameter["outer_size"][0] = parameter["outer_size"][
                    1
                ] - random.uniform(0.012, 0.022)

                cover_height = parameter["outer_size"][2]

                scale = random.uniform(0.3, 0.5)

                parameter["inner_size"] = np.array(
                    [
                        parameter["outer_size"][0] * scale,
                        parameter["outer_size"][1] - random.uniform(0.2, 0.3),
                        parameter["outer_size"][2] * random.uniform(0.4, 0.8),
                    ]
                )
                parameter["position"] = np.array(
                    [0, body_height / 2 + cover_height / 2, 0]
                )
                parameter["rotation"] = np.array([0, random.uniform(0, 360), 0])

                cover_angle = parameter["rotation"][1]

            elif template == "Semi_Spherical_Cover":
                parameter["radius"] = np.array(
                    [body_outer_radius * random.uniform(1.5, 2)]
                )
                angle = np.arcsin(body_outer_radius / parameter["radius"][0])

                cover_height = parameter["radius"][0] * (1 - np.cos(angle)) - 0.03

                angle = np.rad2deg(angle)
                parameter["exist_angle"] = np.array([angle])

                parameter["position"] = np.array([0, body_height / 2, 0])
                parameter["rotation"] = np.array([0, random.uniform(0, 360), 0])
                cover_angle = parameter["rotation"][1]

            elif template == "Cuboidal_Tophandle":
                parameter["size"] = np.array(
                    [
                        random.uniform(0.1, 0.2),
                        random.uniform(0.1, body_height * 0.5),
                        random.uniform(0.1, 0.3),
                    ]
                )
                parameter["position"] = np.array([0, body_height / 2 + cover_height, 0])
                parameter["rotation"] = np.array([0, cover_angle, 0])

            elif template == "Trifold_Tophandle":
                parameter["mounting_size"] = np.array(
                    [
                        random.uniform(0.03, 0.07),
                        random.uniform(0.1, 0.2),
                        random.uniform(0.03, 0.09),
                    ]
                )

                parameter["mounting_seperation"] = np.array([random.uniform(0.4, 0.5)])
                parameter["grip_size"] = np.array(
                    [
                        parameter["mounting_seperation"][0]
                        + parameter["mounting_size"][0] * 2
                        + random.uniform(0.05, 0.1),
                        random.uniform(0.05, 0.07),
                        parameter["mounting_size"][2],
                    ]
                )

                parameter["mounting_rotation"] = np.array(
                    [(random.randint(0, 1) * 2 - 1) * random.uniform(0, 15)]
                )
                parameter["position"] = np.array([0, body_height / 2 + cover_height, 0])
                parameter["rotation"] = np.array([0, cover_angle, 0])

            elif template == "Semi_Ring_Tophandle":
                parameter["curve_size"] = np.array(
                    [random.uniform(0.15, 0.3), 0.0, random.uniform(0.05, 0.15)]
                )
                ring_width = random.uniform(0.03, 0.08)
                parameter["curve_size"][1] = parameter["curve_size"][0] - ring_width
                parameter["curve_exist_angle"] = np.array([random.uniform(100, 160)])
                exist_angle = np.deg2rad(parameter["curve_exist_angle"][0] / 2)
                parameter["position"] = np.array(
                    [
                        0,
                        body_height / 2
                        + cover_height
                        + ring_width * np.cos(exist_angle),
                        0,
                    ]
                )
                parameter["rotation"] = np.array([0, cover_angle, 0])

            elif template == "Multilevel_Tophandle":
                parameter["num_levels"] = np.array([random.randint(1, 4)])

                parameter["level_1_size"] = np.array(
                    [
                        random.uniform(0.05, 0.2),
                        random.uniform(0.05, 0.2),
                        random.uniform(0.05, 0.07),
                    ]
                )
                parameter["level_2_size"] = np.array(
                    [
                        random.uniform(0.1, 0.14),
                        parameter["level_1_size"][0],
                        random.uniform(0.02, 0.05),
                    ]
                )
                parameter["level_3_size"] = np.array(
                    [
                        random.uniform(0.08, 0.14),
                        parameter["level_2_size"][0],
                        random.uniform(0.02, 0.05),
                    ]
                )
                parameter["level_4_size"] = np.array(
                    [
                        random.uniform(0.08, 0.1),
                        parameter["level_3_size"][0],
                        random.uniform(0.02, 0.05),
                    ]
                )
                parameter["position"] = np.array([0, body_height / 2 + cover_height, 0])
                parameter["rotation"] = np.array([0, cover_angle, 0])
            elif template == "Trifold_Sidehandle":
                parameter["mounting_size"] = np.array(
                    [
                        random.uniform(0.14, 0.27),
                        random.uniform(0.03, 0.07),
                        random.uniform(0.03, 0.05),
                    ]
                )
                parameter["mounting_seperation"] = np.array([random.uniform(0.3, 0.5)])
                parameter["grip_size"] = np.array(
                    [
                        random.uniform(0.06, 0.08),
                        parameter["mounting_size"][1] * random.uniform(1, 1.2),
                        parameter["mounting_seperation"][0]
                        + parameter["mounting_size"][2] * 2
                        + random.uniform(0.05, 0.1),
                    ]
                )
                parameter["mounting_rotation"] = np.array(
                    [(random.randint(0, 1) * 2 - 1) * random.uniform(0, 15)]
                )
                parameter["handle_seperation"] = np.array(
                    [
                        body_outer_radius * 2
                        - body_difference * 1.4
                        - parameter["mounting_seperation"][0] * 0.15
                    ]
                )
                parameter["whole_rotation"] = np.array(
                    [(random.randint(0, 1) * 2 - 1) * random.uniform(0, 15)]
                )
                parameter["position"] = np.array(
                    [0, random.uniform(0.05, 0.1) * body_height, 0]
                )
                parameter["rotation"] = np.zeros(3)

            elif template == "L_Shaped_Sidehandle":
                parameter["bottom_size"] = np.array(
                    [
                        random.uniform(0.06, 0.08),
                        random.uniform(0.001, 0.03),
                        random.uniform(0.12, 0.25),
                    ]
                )
                parameter["top_size"] = np.array(
                    [
                        random.uniform(0.12, 0.25),
                        random.uniform(0.04, 0.05),
                        parameter["bottom_size"][2],
                    ]
                )
                parameter["handle_seperation"] = np.array(
                    [body_outer_radius * 2 - body_difference * 0.7]
                )
                parameter["position"] = np.array(
                    [0, random.uniform(0.03, 0.07) * body_height, 0]
                )
                parameter["rotation"] = np.zeros(3)

            elif template == "Cuboidal_Sidehandle":
                parameter["size"] = np.array(
                    [
                        random.uniform(0.5, 1),
                        random.uniform(0.12, 0.18),
                        random.uniform(0.1, 0.2),
                    ]
                )

                parameter["rotation"] = np.array([0, 0, random.uniform(0, 15)])
                parameter["position"] = np.array(
                    [
                        body_outer_radius
                        + parameter["size"][0]
                        * 0.5
                        * np.cos(np.deg2rad(parameter["rotation"][2]))
                        - body_difference * 1.3,
                        random.uniform(0.05, 0.1) * body_height
                        + parameter["size"][0]
                        * 0.5
                        * np.sin(np.deg2rad(parameter["rotation"][2])),
                        parameter["size"][2] / 2,
                    ]
                )

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
        kitchenpot_type = get_kitchenpot_type()
        existing_concept_templates = concept_template_existence(kitchenpot_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, kitchenpot_type)

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
            kitchenpot_type = get_kitchenpot_type()
            existing_concept_templates = concept_template_existence(kitchenpot_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, kitchenpot_type)
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
