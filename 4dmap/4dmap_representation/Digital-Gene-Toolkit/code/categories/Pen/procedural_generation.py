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


def get_pen_type():
    total_type = ["capped", "pushing"]
    weights = [1, 1]
    pen_type = random.choices(total_type, weights=weights, k=1)[0]
    return pen_type


def concept_template_existence(pen_type):
    concept_template_variation = {
        "barrel": {
            "template": ["Cylindrical_Barrel", "Double_Layer_Barrel"],
            "necessary": True,
        },
        "cap": {"template": ["Single_Cap"], "necessary": True},
        "clip": {"template": ["Trifold_Clip", "Curved_Clip"], "necessary": True},
        "button": {
            "template": ["Cylindrical_Button", "Bistratal_Button"],
            "necessary": True,
        },
        "refill": {"template": ["Cylindrical_Refill"], "necessary": True},
    }

    templates = []

    for part_name, part in concept_template_variation.items():
        if part_name != "cap":
            templates.append(random.choice(part["template"]))
        if part_name == "cap":
            if pen_type == "capped":
                templates.append(random.choice(part["template"]))

    return templates


def jitter_parameters(concepts, pen_type):
    new_concepts = []

    if_double: bool = False
    barrel_angle: float = 0
    barrel_bottom_radius: float = 0
    barrel_bottom_inside_radius: float = 0
    barrel_upper_radius: float = 0
    barrel_height: float = 0
    double_barrel_height: float = 0
    if_curved: bool = False
    curve_angle: float = 0
    cap_height: float = 0
    cap_radius: float = 0
    cap_bottom_radius: float = 0
    cap_position_y: float = 0

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Cylindrical_Barrel":
            if pen_type == "capped":
                parameter["size"] = np.array(
                    [
                        randRange(1, 0.06, 0.1)[0],
                        randRange(1, 0.06, 0.1)[0],
                        randRange(1, 1, 2.3)[0],
                    ]
                )
                parameter["thickness"] = randRange(1, 0.01, 0.016)
                parameter["position"] = np.array([0, 0, 0])
                parameter["rotation"] = np.array([0, 0, 0])

            elif pen_type == "pushing":
                parameter["size"] = np.array(
                    [
                        randRange(1, 0.06, 0.1)[0],
                        randRange(1, 0.06, 0.1)[0],
                        randRange(1, 1.2, 2.3)[0],
                    ]
                )
                parameter["thickness"] = randRange(1, 0.01, 0.016)
                parameter["position"] = np.array([0, 0, 0])
                parameter["rotation"] = np.array([0, 0, 0])

            barrel_angle = np.arctan(
                (parameter["size"][1] - parameter["size"][0]) / parameter["size"][2]
            )
            barrel_bottom_radius = parameter["size"][1]
            barrel_bottom_inside_radius = (
                parameter["size"][1] - parameter["thickness"][0]
            )
            barrel_upper_radius = parameter["size"][0]
            barrel_height = parameter["size"][2]

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        if template == "Double_Layer_Barrel":
            if_double = True
            total_height = randRange(1, 1, 2.3)[0]
            if pen_type == "capped":
                parameter["main_size"] = np.array(
                    [
                        randRange(1, 0.06, 0.08)[0],
                        randRange(1, 0.06, 0.11)[0],
                        total_height * randRange(1, 0.8, 0.95)[0],
                    ]
                )
                parameter["bottom_size"] = np.array(
                    [
                        randRange(1, 0.025, 0.045)[0],
                        total_height - parameter["main_size"][2],
                    ]
                )
                parameter["thickness"] = randRange(1, 0.009, 0.015)
                parameter["position"] = np.array([0, 0, 0])
                parameter["rotation"] = np.array([0, 0, 0])

            elif pen_type == "pushing":
                parameter["main_size"] = np.array(
                    [
                        randRange(1, 0.06, 0.08)[0],
                        randRange(1, 0.06, 0.15)[0],
                        total_height * randRange(1, 0.4, 0.9)[0],
                    ]
                )
                parameter["bottom_size"] = np.array(
                    [
                        randRange(1, 0.025, 0.045)[0],
                        total_height - parameter["main_size"][2],
                    ]
                )
                parameter["thickness"] = randRange(1, 0.009, 0.015)
                parameter["position"] = np.array([0, 0, 0])
                parameter["rotation"] = np.array([0, 0, 0])

            barrel_angle = np.arctan(
                (parameter["main_size"][1] - parameter["main_size"][0])
                / parameter["main_size"][2]
            )
            barrel_bottom_radius = parameter["main_size"][1]
            barrel_bottom_inside_radius = (
                parameter["bottom_size"][0] - parameter["thickness"][0]
            )
            barrel_upper_radius = parameter["main_size"][0]
            barrel_height = parameter["main_size"][2]
            double_barrel_height = parameter["bottom_size"][1]

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Single_Cap":
            if pen_type == "capped":
                all_shape = ["straight", "curved"]
                weight_of_shape = [1, 1]
                cap_shape = random.choices(all_shape, weight_of_shape, k=1)[0]

                if cap_shape == "straight":
                    parameter["inner_size"] = np.array(
                        [
                            barrel_bottom_radius,
                            barrel_bottom_radius,
                            randRange(1, 0.4, 0.5)[0],
                        ]
                    )
                    parameter["outer_size"] = np.array(
                        [
                            parameter["inner_size"][0] + randRange(1, 0.005, 0.02)[0],
                            0.1,
                            parameter["inner_size"][2] + randRange(1, 0.005, 0.02)[0],
                        ]
                    )
                    parameter["outer_size"][1] = parameter["outer_size"][0]

                if cap_shape == "curved":
                    if_curved = True
                    parameter["inner_size"] = np.array(
                        [barrel_bottom_radius, 0.1, randRange(1, 0.4, 0.5)[0]]
                    )
                    parameter["inner_size"][1] = parameter["inner_size"][0] - 0.03

                    parameter["outer_size"] = np.array(
                        [
                            parameter["inner_size"][0] + randRange(1, 0.005, 0.02)[0],
                            0.1,
                            parameter["inner_size"][2] + randRange(1, 0.005, 0.02)[0],
                        ]
                    )
                    parameter["outer_size"][1] = (
                        parameter["outer_size"][0] - randRange(1, 0.025, 0.035)[0]
                    )

                    curve_angle = np.arctan(
                        (parameter["outer_size"][0] - parameter["outer_size"][1])
                        / parameter["outer_size"][2]
                    )

                parameter["position"] = np.array(
                    [
                        0,
                        randRange(
                            1,
                            -(barrel_height + double_barrel_height) / 2 - 0.15,
                            -(barrel_height + double_barrel_height),
                        )[0],
                        0,
                    ]
                )

                parameter["rotation"] = np.array([0, 0, 0])

                cap_height = parameter["outer_size"][2]
                cap_radius = parameter["outer_size"][0]
                cap_bottom_radius = parameter["outer_size"][1]

            elif pen_type == "pushing":
                pass

            cap_position_y = parameter["position"][1]

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Trifold_Clip":
            if pen_type == "capped":
                parameter["clip_root_size"] = np.array(
                    [
                        randRange(1, 0.04, 0.05)[0],
                        randRange(1, 0.09, 0.11)[0],
                        randRange(1, 0.015, 0.017)[0],
                    ]
                )
                parameter["clip_vertical_size"] = np.array(
                    [
                        randRange(1, 0.045, 0.055)[0],
                        randRange(1, cap_height - 0.1, cap_height + 0.1)[0],
                        randRange(1, 0.015, 0.02)[0],
                    ]
                )
                parameter["clip_tip_size"] = np.array(
                    [
                        randRange(1, 0.043, 0.045)[0],
                        randRange(1, 0.085, 0.095)[0],
                        randRange(1, 0.02, 0.025)[0],
                    ]
                )
                parameter["clip_offset"] = np.array([0, 0])
                all_direction = ["left", "right", "middle"]
                weight_of_direction = [1, 1, 1]
                direction = random.choices(all_direction, weight_of_direction, k=1)[0]

                if direction == "left":
                    parameter["position"] = np.array(
                        [
                            -cap_radius,
                            cap_position_y
                            + (parameter["clip_vertical_size"][1] - cap_height / 2),
                            0,
                        ]
                    )
                    parameter["rotation"] = np.array([0, -90, 0])

                    if if_curved:
                        parameter["rotation"] = np.array(
                            [0, -90, np.rad2deg(curve_angle)]
                        )

                if direction == "right":
                    parameter["position"] = np.array(
                        [
                            cap_radius,
                            cap_position_y
                            + (parameter["clip_vertical_size"][1] - cap_height / 2),
                            0,
                        ]
                    )
                    parameter["rotation"] = np.array([0, 90, 0])
                    if if_curved:
                        parameter["rotation"] = np.array(
                            [0, 90, -np.rad2deg(curve_angle)]
                        )

                if direction == "middle":
                    parameter["position"] = np.array(
                        [
                            0,
                            cap_position_y
                            + (parameter["clip_vertical_size"][1] - cap_height / 2),
                            cap_radius,
                        ]
                    )
                    parameter["rotation"] = np.array([0, 0, 0])
                    if if_curved:
                        parameter["rotation"] = np.array(
                            [np.rad2deg(curve_angle), 0, 0]
                        )

            elif pen_type == "pushing":
                parameter["clip_root_size"] = np.array(
                    [
                        randRange(1, 0.043, 0.045)[0],
                        randRange(1, 0.085, 0.095)[0],
                        randRange(1, 0.02, 0.025)[0],
                    ]
                )
                parameter["clip_vertical_size"] = np.array(
                    [
                        randRange(1, 0.045, 0.055)[0],
                        randRange(1, 0.3, 0.5)[0],
                        randRange(1, 0.015, 0.02)[0],
                    ]
                )
                parameter["clip_tip_size"] = np.array(
                    [
                        randRange(1, 0.04, 0.05)[0],
                        randRange(1, 0.09, 0.11)[0],
                        randRange(1, 0.015, 0.017)[0],
                    ]
                )
                parameter["clip_offset"] = np.array([0, 0])
                all_direction = ["left", "right", "middle"]
                weight_of_direction = [1, 1, 1]
                direction = random.choices(all_direction, weight_of_direction, k=1)[0]

                if direction == "left":
                    parameter["position"] = np.array(
                        [-barrel_upper_radius, barrel_height / 2, 0]
                    )
                    parameter["rotation"] = np.array(
                        [0, -90, -np.rad2deg(barrel_angle)]
                    )

                if direction == "right":
                    parameter["position"] = np.array(
                        [barrel_upper_radius, barrel_height / 2, 0]
                    )
                    parameter["rotation"] = np.array([0, 90, np.rad2deg(barrel_angle)])

                if direction == "middle":
                    parameter["position"] = np.array(
                        [0, barrel_height / 2, barrel_upper_radius]
                    )
                    parameter["rotation"] = np.array([-np.rad2deg(barrel_angle), 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Curved_Clip":
            if pen_type == "capped":
                parameter["clip_curve_size"] = np.array(
                    [randRange(1, 0.8, 0.86)[0], 0.1, randRange(1, 0.04, 0.05)[0]]
                )
                parameter["clip_curve_size"][1] = (
                    parameter["clip_curve_size"][0] - 0.015
                )
                parameter["clip_curve_exist_angle"] = randRange(1, 20, 25)
                sin_half_angle = np.sin(
                    np.deg2rad(parameter["clip_curve_exist_angle"][0] / 2)
                )
                clip_height = parameter["clip_curve_size"][1] * sin_half_angle * 2
                all_direction = ["left", "right", "middle"]
                weight_of_direction = [1, 1, 1]
                direction = random.choices(all_direction, weight_of_direction, k=1)[0]
                if direction == "left":
                    parameter["position"] = np.array(
                        [
                            -cap_radius - 0.015,
                            cap_position_y + (clip_height - cap_height / 2),
                            0,
                        ]
                    )
                    parameter["rotation"] = np.array([0, -90, 0])

                    if if_curved:
                        parameter["position"][0] = -cap_bottom_radius - 0.033
                        parameter["rotation"] = np.array(
                            [0, -90, np.rad2deg(curve_angle)]
                        )

                if direction == "right":
                    parameter["position"] = np.array(
                        [
                            cap_radius + 0.015,
                            cap_position_y + (clip_height - cap_height / 2),
                            0,
                        ]
                    )
                    parameter["rotation"] = np.array([0, 90, 0])

                    if if_curved:
                        parameter["position"][0] = cap_bottom_radius + 0.033
                        parameter["rotation"] = np.array(
                            [0, 90, -np.rad2deg(curve_angle)]
                        )

                if direction == "middle":
                    parameter["position"] = np.array(
                        [
                            0,
                            cap_position_y + (clip_height - cap_height / 2),
                            cap_radius + 0.015,
                        ]
                    )
                    parameter["rotation"] = np.array([0, 0, 0])

                    if if_curved:
                        parameter["position"][2] = cap_bottom_radius + 0.033
                        parameter["rotation"] = np.array(
                            [np.rad2deg(curve_angle), 0, 0]
                        )

            elif pen_type == "pushing":
                parameter["clip_curve_size"] = np.array(
                    [randRange(1, 0.7, 0.9)[0], 0.1, randRange(1, 0.04, 0.05)[0]]
                )
                parameter["clip_curve_size"][1] = (
                    parameter["clip_curve_size"][0] - 0.015
                )
                parameter["clip_curve_exist_angle"] = randRange(1, 20, 25)

                all_direction = ["left", "right", "middle"]
                weight_of_direction = [1, 1, 1]
                direction = random.choices(all_direction, weight_of_direction, k=1)[0]

                if direction == "left":
                    parameter["position"] = np.array(
                        [-barrel_upper_radius - 0.015, barrel_height / 2, 0]
                    )
                    parameter["rotation"] = np.array(
                        [0, -90, -np.rad2deg(barrel_angle)]
                    )

                if direction == "right":
                    parameter["position"] = np.array(
                        [barrel_upper_radius + 0.015, barrel_height / 2, 0]
                    )
                    parameter["rotation"] = np.array([0, 90, np.rad2deg(barrel_angle)])

                if direction == "middle":
                    parameter["position"] = np.array(
                        [0, barrel_height / 2, barrel_upper_radius + 0.015]
                    )
                    parameter["rotation"] = np.array([-np.rad2deg(barrel_angle), 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cylindrical_Button":
            if pen_type == "capped":
                parameter["size"] = np.array(
                    [
                        randRange(1, 0.05, 0.055)[0],
                        randRange(1, barrel_upper_radius - 0.006, barrel_upper_radius)[
                            0
                        ],
                        randRange(1, 0.02, 0.12)[0],
                    ]
                )
                parameter["position"] = np.array([0, barrel_height / 2, 0])
                parameter["rotation"] = np.array([0, 0, 0])

            elif pen_type == "pushing":
                parameter["size"] = np.array(
                    [
                        randRange(1, 0.05, 0.055)[0],
                        randRange(1, barrel_upper_radius - 0.006, barrel_upper_radius)[
                            0
                        ],
                        randRange(1, 0.06, 0.14)[0],
                    ]
                )
                parameter["position"] = np.array([0, barrel_height / 2, 0])
                parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Bistratal_Button":
            if pen_type == "capped":
                parameter["bottom_size"] = np.array(
                    [
                        randRange(1, 0.05, 0.056)[0],
                        randRange(1, barrel_upper_radius - 0.006, barrel_upper_radius)[
                            0
                        ],
                        randRange(1, 0.02, 0.06)[0],
                    ]
                )
                parameter["top_size"] = np.array(
                    [
                        randRange(1, 0.04, 0.05)[0],
                        randRange(1, 0.043, 0.05)[0],
                        randRange(1, 0.004, 0.008)[0],
                    ]
                )
                parameter["position"] = np.array([0, barrel_height / 2, 0])
                parameter["rotation"] = np.array([0, 0, 0])

            elif pen_type == "pushing":
                parameter["bottom_size"] = np.array(
                    [
                        randRange(1, 0.05, 0.056)[0],
                        randRange(1, barrel_upper_radius - 0.006, barrel_upper_radius)[
                            0
                        ],
                        randRange(1, 0.02, 0.06)[0],
                    ]
                )
                parameter["top_size"] = np.array(
                    [
                        randRange(1, 0.04, 0.05)[0],
                        randRange(1, 0.043, 0.05)[0],
                        randRange(1, 0.004, 0.006)[0],
                    ]
                )
                parameter["position"] = np.array([0, barrel_height / 2, 0])
                parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cylindrical_Refill":
            if pen_type == "capped":
                parameter["size"] = np.array([0.06, barrel_height])
                if if_double:
                    parameter["size"][0] = barrel_bottom_inside_radius

                parameter["tip_radius"] = np.array(
                    [parameter["size"][0] * randRange(1, 0.5, 0.8)[0]]
                )
                parameter["tip_height"] = np.array(
                    [randRange(1, 0.073, 0.083)[0], randRange(1, 0.06, 0.1)[0]]
                )

                if if_double:
                    parameter["tip_height"] = np.array(
                        [randRange(1, 0.03, 0.04)[0], randRange(1, 0.06, 0.1)[0]]
                    )

                all_tip = [
                    np.array([0, 0]),
                    np.array([-0.5 * parameter["tip_radius"][0], 0]),
                    np.array([0.5 * parameter["tip_radius"][0], 0]),
                ]

                parameter["tip_offset"] = all_tip[1]
                parameter["position"] = np.array([0, -0.05, 0])
                if if_double:
                    parameter["position"][1] = -double_barrel_height + 0.01

                parameter["rotation"] = np.array([0, 0, 0])

            elif pen_type == "pushing":
                parameter["size"] = np.array([0.06, barrel_height])
                if if_double:
                    parameter["size"][0] = barrel_bottom_inside_radius

                parameter["tip_radius"] = np.array(
                    [parameter["size"][0] * randRange(1, 0.5, 0.8)[0]]
                )

                parameter["tip_height"] = np.array(
                    [randRange(1, 0.03, 0.07)[0], randRange(1, 0.06, 0.1)[0]]
                )

                all_tip = [
                    np.array([0, 0]),
                    np.array([-0.5 * parameter["tip_radius"][0], 0]),
                    np.array([0.5 * parameter["tip_radius"][0], 0]),
                ]

                parameter["tip_offset"] = all_tip[1]
                parameter["position"] = np.array([0, -0.05, 0])

                if if_double:
                    all_position = ["inside", "outside"]
                    weight_of_position = [1, 1]
                    position = random.choices(all_position, weight_of_position, k=1)[0]

                    if position == "inside":
                        parameter["position"] = np.array(
                            [0, -double_barrel_height / 2 + 0.1, 0]
                        )

                    if position == "outside":
                        parameter["position"] = np.array(
                            [0, -double_barrel_height + 0.04, 0]
                        )

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
        pen_type = get_pen_type()
        existing_concept_templates = concept_template_existence(pen_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, pen_type)

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
            pen_type = get_pen_type()
            existing_concept_templates = concept_template_existence(pen_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, pen_type)
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
