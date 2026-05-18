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


def get_door_type():
    door_type = "regular_door"
    return door_type


def concept_template_existence(door_type):
    concept_template_variation = {
        "door": {"template": ["Standard_Door"], "necessary": True},
        "frame": {"template": ["Standard_Doorframe"], "necessary": True},
        "hinge": {"template": ["Standard_Hinge"], "necessary": True},
        "handle": {
            "template": [
                "LShape_Handle",
                "PiShape_Handle",
                "Cylindrical_Handle",
                "Spherical_Handle",
            ],
            "necessary": True,
        },
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        templates.append(random.choice(part["template"]))
    return templates


def jitter_parameters(concepts, door_type):
    is_double: bool = False
    door_size = np.array([0, 0, 0])
    existence_of_door = np.array([0, 0])
    door_rotation = np.array([0, 0, 0])

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Standard_Door":
            parameter["existence_of_door"] = np.random.randint(low=0, high=2, size=2)
            while (
                parameter["existence_of_door"][0] == 0
                and parameter["existence_of_door"][1] == 0
            ):
                parameter["existence_of_door"] = np.random.randint(
                    low=0, high=2, size=2
                )

            if (
                parameter["existence_of_door"][0] == 1
                and parameter["existence_of_door"][1] == 1
            ):
                is_double = True
            existence_of_door = np.copy(parameter["existence_of_door"])

            parameter["size"] = np.array(
                [
                    randRange(1, 0.6, 0.95)[0],
                    randRange(1, 1.35, 1.95)[0],
                    randRange(1, 0.02, 0.05)[0],
                ]
            )

            door_size = np.copy(parameter["size"])

            parameter["door_rotation"] = np.array(
                [randRange(1, 0, 90)[0], randRange(1, 0, 90)[0]]
            )

            door_rotation = np.copy(parameter["door_rotation"])

            parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Standard_Doorframe":
            parameter["door_size"] = np.copy(door_size)
            if is_double:
                parameter["door_size"][0] = door_size[0] * 2

            parameter["existence_of_doorframe"] = np.random.randint(
                low=0, high=2, size=2
            )

            parameter["main_outer_size"] = np.array(
                [
                    parameter["door_size"][0] + randRange(1, 0.03, 0.3)[0],
                    parameter["door_size"][1] + randRange(1, 0.01, 0.18)[0],
                    randRange(1, 0.02, 0.105)[0],
                ]
            )

            parameter["main_inner_outer_offset"] = np.array([0, 0])

            all_main_move = ["default", "forward", "backward"]
            weight_of_main_move = [2, 1, 1]
            main_move = random.choices(all_main_move, weight_of_main_move, k=1)[0]
            parameter["main_offset"] = np.array([0.0, 0.0])
            # if main_move == 'forward':
            #     parameter['main_offset'] = np.array([
            #         0,
            #         randRange(1, 0, parameter['door_size'][2]/2)[0]
            #     ])
            # if main_move == 'backward':
            #     parameter['main_offset'] = np.array([
            #         0,
            #         -randRange(1, 0, parameter['door_size'][2] / 2)[0]
            #     ])

            # if parameter['existence_of_doorframe'][0] == 0 and parameter['existence_of_doorframe'][1] == 0:
            #     if existence_of_door[0] == 0 or existence_of_door[1] == 0:
            #         if existence_of_door[0]:
            #             parameter['main_offset'][0] = -randRange(1, 0, parameter['door_size'][0] / 2)[0]
            #         if existence_of_door[1]:
            #             parameter['main_offset'][0] = randRange(1, 0, parameter['door_size'][0] / 2)[0]

            difference_x = parameter["main_outer_size"][0] - parameter["door_size"][0]
            difference_y = parameter["main_outer_size"][1] - parameter["door_size"][1]

            difference = randRange(1, 0.3, 1.5)[0]

            parameter["sub1_outer_size"] = np.array(
                [
                    parameter["door_size"][0] + difference * difference_x,
                    parameter["door_size"][1] + difference * difference_y,
                    randRange(1, 0.03, 0.085)[0],
                ]
            )

            parameter["sub1_inner_size"][0] = parameter["door_size"][0]
            parameter["sub1_inner_size"][1] = parameter["door_size"][1]

            parameter["sub1_inner_outer_offset"] = np.array([0, 0])

            parameter["sub1_offset"] = np.array([0.0, 0.0])

            parameter["sub2_outer_size"] = np.array(
                [
                    parameter["door_size"][0] + difference * difference_x,
                    parameter["door_size"][1] + difference * difference_y,
                    randRange(1, 0.03, 0.085)[0],
                ]
            )

            if (
                parameter["existence_of_doorframe"][0]
                and parameter["existence_of_doorframe"][1]
            ):
                parameter["sub2_outer_size"] = parameter["sub1_outer_size"]

            parameter["sub2_inner_size"][0] = parameter["door_size"][0]
            parameter["sub2_inner_size"][1] = parameter["door_size"][1]

            parameter["sub2_inner_outer_offset"] = np.array([0, 0])

            parameter["sub2_offset"] = np.array([0.0, 0.0])

            all_sub_y_move = ["default", "up"]
            weight_of_sub_y_move = [1, 1]
            sub_y_move = random.choices(all_sub_y_move, weight_of_sub_y_move, k=1)[0]
            if sub_y_move == "up":
                parameter["sub1_offset"][1] = randRange(1, 0, difference_y)[0]
                parameter["sub2_offset"][1] = parameter["sub1_offset"][1]
                if difference < 1:
                    parameter["sub1_offset"][1] = randRange(
                        1, 0, difference_y * (1 - difference)
                    )[0]
                    parameter["sub2_offset"][1] = parameter["sub1_offset"][1]

            if existence_of_door[0] == 0 or existence_of_door[1] == 0:
                all_sub_x_move = ["default", "x_move"]
                weight_of_sub_x_move = [1, 1]
                sub_x_move = random.choices(all_sub_x_move, weight_of_sub_x_move, k=1)[
                    0
                ]
                if sub_x_move == "x_move":
                    if existence_of_door[0]:
                        parameter["sub1_offset"][0] = -randRange(
                            1, 0, difference_x / 2
                        )[0]
                        parameter["sub2_offset"][0] = parameter["sub1_offset"][0]
                        if difference < 1:
                            parameter["sub1_offset"][0] = -randRange(
                                1, 0, difference_x / 2 * (1 - difference)
                            )[0]
                            parameter["sub2_offset"][0] = parameter["sub1_offset"][0]
                    if existence_of_door[1]:
                        parameter["sub1_offset"][0] = -randRange(
                            1, 0, difference_x / 2
                        )[0]
                        parameter["sub2_offset"][0] = parameter["sub1_offset"][0]
                        if difference < 1:
                            parameter["sub1_offset"][0] = -randRange(
                                1, 0, difference_x / 2 * (1 - difference)
                            )[0]
                            parameter["sub2_offset"][0] = parameter["sub1_offset"][0]

            parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Standard_Hinge":
            parameter["existence_of_door"] = existence_of_door

            parameter["number_of_hinge"] = np.array([np.random.randint(low=2, high=5)])

            parameter["size"] = np.array(
                [randRange(1, 0.004, 0.005)[0], randRange(1, 0.08, 0.095)[0]]
            )

            original_height = door_size[1] * randRange(1, 0.05, 0.2)[0]
            if parameter["number_of_hinge"] == 5:
                original_height = door_size[1] * randRange(1, 0.05, 0.1)[0]
            if parameter["number_of_hinge"] == 2:
                original_height = door_size[1] * randRange(1, 0.05, 0.3)[0]
            if parameter["number_of_hinge"] == 1:
                original_height = door_size[1] / 2 * randRange(1, 0.8, 1.2)[0]

            parameter["separation"] = np.array(
                [
                    door_size[1]
                    * 0.65
                    * randRange(1, 0.8, 1)[0]
                    * (1 / (parameter["number_of_hinge"][0] - 1)),
                    door_size[1]
                    * 0.65
                    * randRange(1, 0.8, 1)[0]
                    * (1 / (parameter["number_of_hinge"][0] - 1)),
                    door_size[1]
                    * 0.65
                    * randRange(1, 0.8, 1)[0]
                    * (1 / (parameter["number_of_hinge"][0] - 1)),
                    door_size[1]
                    * 0.65
                    * randRange(1, 0.8, 1)[0]
                    * (1 / (parameter["number_of_hinge"][0] - 1)),
                ]
            )

            parameter["offset_1"] = np.array(
                [-door_size[0] / 2, -door_size[1] / 2 + original_height, 0]
            )

            if existence_of_door[0] == 0 and existence_of_door[1] == 1:
                parameter["offset_1"] = np.array(
                    [door_size[0] / 2, -door_size[1] / 2 + original_height, 0]
                )

            parameter["offset_2"] = np.array([0.0, 0.0, 0.0])

            if is_double:
                parameter["offset_1"][0] = -door_size[0]
                parameter["offset_2"] = np.array(
                    [door_size[0], -door_size[1] / 2 + original_height, 0]
                )

            parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "LShape_Handle":
            parameter["existence_of_door"] = existence_of_door
            parameter["door_rotation"] = door_rotation
            parameter["door_size"] = door_size

            parameter["existence_of_handle"] = np.random.randint(low=0, high=2, size=2)
            while (
                parameter["existence_of_handle"][0] == 0
                and parameter["existence_of_handle"][1] == 0
            ):
                parameter["existence_of_handle"] = np.random.randint(
                    low=0, high=2, size=2
                )

            parameter["fixed_part_size"] = np.array(
                [
                    randRange(1, 0.03, 0.05)[0],
                    randRange(1, 0.03, 0.05)[0],
                    randRange(1, 0.008, 0.01)[0],
                ]
            )

            parameter["vertical_movable_size"] = np.array(
                [
                    randRange(1, 0.02, 0.03)[0],
                    randRange(1, 0.0125, 0.0225)[0],
                    randRange(1, 0.01, 0.014)[0],
                ]
            )

            parameter["horizontal_movable_size"] = np.array(
                [
                    randRange(1, 0.08, 0.09)[0],
                    randRange(1, 0.01, 0.02)[0],
                    randRange(1, 0.01, 0.02)[0],
                ]
            )

            parameter["interpiece_offset"] = np.array([0, 0])

            all_position = ["up", "down"]
            weight_of_position = [1, 1]
            y_position = random.choices(all_position, weight_of_position, k=1)[0]
            if y_position == "up":
                parameter["offset_x"] = np.array(
                    [
                        door_size[0] * randRange(1, 0.3, 0.45)[0],
                        randRange(1, 0, 0.18)[0],
                    ]
                )
            elif y_position == "down":
                parameter["offset_x"] = np.array(
                    [
                        door_size[0] * randRange(1, 0.3, 0.45)[0],
                        -randRange(1, 0, 0.18)[0],
                    ]
                )

            all_direction = ["left", "right"]
            weight_of_direction = [1, 1]
            direction = random.choices(all_direction, weights=weight_of_direction, k=1)[
                0
            ]
            # if direction == 'left':
            #     parameter['handle_rotation'] = np.array([
            #         randRange(1, 0, 45)[0]
            #         ])
            # elif direction == 'right':
            #     parameter['handle_rotation'] = np.array([
            #         -randRange(1, 0, 45)[0]
            #     ])

            parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "PiShape_Handle":
            parameter["existence_of_door"] = existence_of_door
            parameter["door_rotation"] = door_rotation
            parameter["door_size"] = door_size

            parameter["existence_of_handle"] = np.random.randint(low=0, high=2, size=2)
            while (
                parameter["existence_of_handle"][0] == 0
                and parameter["existence_of_handle"][1] == 0
            ):
                parameter["existence_of_handle"] = np.random.randint(
                    low=0, high=2, size=2
                )

            parameter["main_size"] = np.array([0, 0, randRange(1, 0.009, 0.0095)[0]])

            parameter["sub_size"] = np.array(
                [
                    randRange(1, 0.02, 0.03)[0],
                    randRange(1, 0.0125, 0.0225)[0],
                    randRange(1, 0.01, 0.015)[0],
                ]
            )

            parameter["separation"] = np.array([randRange(1, 0.08, 0.09)[0]])

            parameter["main_size"][0] = (
                parameter["sub_size"][0] + randRange(1, 0, 0.01)[0]
            )
            parameter["main_size"][1] = (
                parameter["separation"][0]
                + 2 * parameter["sub_size"][1]
                + randRange(1, 0, 0.03)[0]
            )

            parameter["interpiece_offset"] = np.array([0])

            all_position = ["up", "down"]
            weight_of_position = [1, 1]
            y_position = random.choices(all_position, weight_of_position, k=1)[0]
            if y_position == "up":
                parameter["offset_x"] = np.array(
                    [
                        -door_size[0] * randRange(1, 0.3, 0.45)[0],
                        randRange(1, 0, 0.18)[0],
                    ]
                )
            elif y_position == "down":
                parameter["offset_x"] = np.array(
                    [
                        -door_size[0] * randRange(1, 0.3, 0.45)[0],
                        -randRange(1, 0, 0.18)[0],
                    ]
                )

            all_direction = ["default", "left", "right"]
            weight_of_direction = [2, 1, 1]
            direction = random.choices(all_direction, weights=weight_of_direction, k=1)[
                0
            ]
            # if direction == 'left':
            #     parameter['handle_rotation'] = np.array([
            #         randRange(1, 0, 45)[0]
            #         ])
            # elif direction == 'right':
            #     parameter['handle_rotation'] = np.array([
            #         -randRange(1, 0, 45)[0]
            #     ])
            # if direction == 'default':
            #     parameter['handle_rotation'] = np.array([0.0])
            # else:
            #     parameter['main_size'][1] += (parameter['separation'][0] / np.cos(np.radians(parameter['handle_rotation'][0])))-parameter['separation'][0]

            parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Cylindrical_Handle":
            parameter["existence_of_door"] = existence_of_door
            parameter["door_rotation"] = door_rotation
            parameter["door_size"] = door_size

            parameter["existence_of_handle"] = np.random.randint(low=0, high=2, size=2)
            while (
                parameter["existence_of_handle"][0] == 0
                and parameter["existence_of_handle"][1] == 0
            ):
                parameter["existence_of_handle"] = np.random.randint(
                    low=0, high=2, size=2
                )

            parameter["fixed_part_size"] = np.array(
                [randRange(1, 0.02, 0.04)[0], randRange(1, 0.005, 0.006)[0]]
            )

            parameter["sub_size"] = np.array(
                [randRange(1, 0.006, 0.009)[0], randRange(1, 0.01, 0.025)[0]]
            )

            parameter["main_size"] = np.array(
                [randRange(1, 0.018, 0.04)[0], randRange(1, 0.023, 0.043)[0]]
            )

            all_position = ["up", "down"]
            weight_of_position = [1, 1]
            y_position = random.choices(all_position, weight_of_position, k=1)[0]
            if y_position == "up":
                parameter["offset_x"] = np.array(
                    [
                        door_size[0] * randRange(1, 0.3, 0.45)[0],
                        randRange(1, 0, 0.18)[0],
                    ]
                )
            elif y_position == "down":
                parameter["offset_x"] = np.array(
                    [
                        door_size[0] * randRange(1, 0.3, 0.45)[0],
                        -randRange(1, 0, 0.18)[0],
                    ]
                )

            # parameter['handle_rotation'] = np.array([randRange(1, 0, 45)[0]])

            parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}

        elif template == "Spherical_Handle":
            parameter["existence_of_door"] = existence_of_door
            parameter["door_rotation"] = door_rotation
            parameter["door_size"] = door_size

            parameter["existence_of_handle"] = np.random.randint(low=0, high=2, size=2)
            while (
                parameter["existence_of_handle"][0] == 0
                and parameter["existence_of_handle"][1] == 0
            ):
                parameter["existence_of_handle"] = np.random.randint(
                    low=0, high=2, size=2
                )

            parameter["fixed_part_size"] = np.array(
                [randRange(1, 0.02, 0.035)[0], randRange(1, 0.002, 0.003)[0]]
            )

            parameter["sub_size"] = np.array(
                [randRange(1, 0.006, 0.01)[0], randRange(1, 0.002, 0.02)[0]]
            )

            base_radius = randRange(1, 0.02, 0.05)[0]

            parameter["main_size"] = np.array(
                [
                    base_radius * randRange(1, 0.7, 1)[0],
                    base_radius * randRange(1, 0.7, 1)[0],
                    base_radius * randRange(1, 0.7, 1)[0],
                ]
            )

            all_position = ["up", "down"]
            weight_of_position = [1, 1]
            y_position = random.choices(all_position, weight_of_position, k=1)[0]
            if y_position == "up":
                parameter["offset_x"] = np.array(
                    [
                        door_size[0] * randRange(1, 0.3, 0.45)[0],
                        randRange(1, 0, 0.18)[0],
                    ]
                )
            elif y_position == "down":
                parameter["offset_x"] = np.array(
                    [
                        door_size[0] * randRange(1, 0.3, 0.45)[0],
                        -randRange(1, 0, 0.18)[0],
                    ]
                )

            # parameter['handle_rotation'] = np.array([randRange(1, 0, 45)[0]])

            parameter["position"] = np.array([0, 0, 0])

            parameter["rotation"] = np.array([0, 0, 0])

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
        door_type = get_door_type()
        existing_concept_templates = concept_template_existence(door_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, door_type)

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
            door_type = get_door_type()
            existing_concept_templates = concept_template_existence(door_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, door_type)
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
