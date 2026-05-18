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


def get_kettle_type():
    total_type = ["semi_spherical", "spherical_cylindrical", "cylindrical"]
    weights = [1, 1, 1]
    kettle_type = random.choices(total_type, weights=weights, k=1)[0]
    return kettle_type


def concept_template_existence(kettle_type):
    if kettle_type == "semi_spherical":
        body_template = ["Semi_Spherical_Body"]
    elif kettle_type == "spherical_cylindrical":
        body_template = ["Spherical_Cylindrical_Body"]
    elif kettle_type == "cylindrical":
        body_template = ["Multilevel_Body"]
    concept_template_variation = {
        "body": {"template": body_template, "necessary": True},
        "cover": {"template": ["Standard_Cover"], "necessary": True},
        "handle": {
            "template": [
                "Trifold_Handle",
                "Curved_Handle",
                "Ring_Handle",
                "Cylindrical_Handle",
            ],
            "necessary": True,
        },
        "U_handle": {
            "template": ["Round_U_Handle", "Flat_U_Handle"],
            "necessary": True,
        },
        "spout": {"template": ["Straight_Spout", "Curved_Spout"], "necessary": True},
    }

    templates = []
    for part_name, part in concept_template_variation.items():
        if not part["necessary"]:
            if random.random() < 0.5:
                templates.append(random.choice(part["template"]))
        else:
            templates.append(random.choice(part["template"]))

    return templates


def get_handle_position(cur_angle, ellipse_b, ellipse_a):
    pos_x = (ellipse_a * ellipse_b) / np.sqrt(
        (ellipse_a * ellipse_a * np.tan(cur_angle) * np.tan(cur_angle))
        + (ellipse_b * ellipse_b)
    )
    pos_y = np.tan(cur_angle) * pos_x
    return [pos_x, pos_y]


def get_spout_position(half_height, ellipse_b1, ellipse_b2, ellipse_a):
    x1 = ellipse_a * np.sqrt(1 - (half_height**2 / (ellipse_b1**2)))
    x2 = ellipse_a * np.sqrt(1 - (half_height**2 / (ellipse_b2**2)))
    return np.minimum(x1, x2)


def get_center(point_up, point_down, radius, horizontal_radius):
    middle_point = [
        (point_up[0] + point_down[0]) / 2,
        (point_up[1] - point_down[1]) / 2,
    ]
    half_point_dis = np.sqrt(
        (point_up[0] - middle_point[0]) ** 2 + (point_up[1] - middle_point[1]) ** 2
    )
    least_radius = np.sqrt((horizontal_radius - point_up[0]) ** 2 + half_point_dis**2)
    if radius < least_radius:
        radius = least_radius * randRange(1, 1.0, 1.1)[0]
    if point_up[0] != point_down[0]:
        slop_up_down = (point_up[1] + point_down[1]) / (point_up[0] - point_down[0])
        slop_new = -1 / slop_up_down
        new_angle = np.arctan(slop_new)
        basic_offset = np.sqrt(radius**2 - half_point_dis**2)
        pos_y_offset = basic_offset * np.sin(new_angle)
        pos_x_offset = basic_offset * np.cos(new_angle)
        center_point = [pos_x_offset + middle_point[0], pos_y_offset + middle_point[1]]
        additional_rot_angle = new_angle / np.pi * 180
    else:
        center_point = [0, 0]
        basic_offset = np.sqrt(radius**2 - ((point_up[1] + point_down[1]) / 2) ** 2)
        center_point[0] = basic_offset + point_up[0]
        center_point[1] = point_up[1] - (point_up[1] + point_down[1]) / 2
        additional_rot_angle = 0
    empty_angle = np.arcsin(half_point_dis / radius) / np.pi * 180
    exist_angle = 360 - empty_angle * 2

    return radius, center_point, exist_angle, additional_rot_angle


def multilevel_get_center(point_up, point_down, cur_level, horizontal_radius):
    middle_point = [
        (point_up[0] + point_down[0]) / 2,
        (point_up[1] + point_down[1]) / 2,
    ]
    half_point_dis = np.sqrt(
        (point_up[0] - middle_point[0]) ** 2 + (point_up[1] - middle_point[1]) ** 2
    )
    least_radius = np.sqrt((horizontal_radius / 2) ** 2 + half_point_dis**2)
    if cur_level == 1:
        least_radius = half_point_dis * randRange(1, 1.0, 1.1)[0]
    radius = least_radius * randRange(1, 1.0, 1.1)[0]
    if point_up[0] != point_down[0]:
        slop_up_down = (point_up[1] - point_down[1]) / (point_up[0] - point_down[0])
        slop_new = -1 / slop_up_down
        new_angle = np.arctan(slop_new)
        basic_offset = np.sqrt(radius**2 - half_point_dis**2)
        pos_y_offset = basic_offset * np.sin(new_angle)
        pos_x_offset = basic_offset * np.cos(new_angle)
        center_point = [pos_x_offset + middle_point[0], pos_y_offset + middle_point[1]]
        additional_rot_angle = new_angle / np.pi * 180
    else:
        center_point = [0, 0]
        basic_offset = np.sqrt(radius**2 - ((point_up[1] + point_down[1]) / 2) ** 2)
        center_point[0] = basic_offset + point_up[0]
        center_point[1] = point_up[1] - (point_up[1] + point_down[1]) / 2
        additional_rot_angle = 0
    empty_angle = np.arcsin(half_point_dis / radius) / np.pi * 180
    exist_angle = 360 - empty_angle * 2

    return radius, center_point, exist_angle, additional_rot_angle


def jitter_parameters(concepts, kettle_type):
    body_type = ""
    new_concepts = []

    templates = [concept["template"] for concept in concepts]
    indexes = {concept["template"]: i for i, concept in enumerate(concepts)}

    for concept in concepts:
        template = concept["template"]
        parameter = {k: np.array(v) for k, v in concept["parameters"].items()}

        if template == "Semi_Spherical_Body":
            body_type = "Semi_Spherical_Body"
            horizontal_axis = (
                parameter["horizontal_axis"][0] * randRange(1, 0.7, 1.3)[0]
            )
            parameter["horizontal_axis"][0] = horizontal_axis
            vertical_mul = randRange(parameter["vertical_axis"].shape[0], 0.9, 1.3)
            parameter["vertical_axis"] *= vertical_mul
            exist_mul = randRange(parameter["exist_angle"].shape[0], 0.7, 1.3)
            parameter["exist_angle"] *= exist_mul
            bottom_mul = randRange(parameter["bottom_size"].shape[0], 0.7, 1.3)
            parameter["bottom_size"] *= bottom_mul
            parameter["x_z_ratio"][0] = 1
            # parameter['thickness'][0] *= randRange(1,0.7,1.3)[0]
            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Spherical_Cylindrical_Body":
            body_type = "Spherical_Cylindrical_Body"
            horizontal_axis = (
                parameter["horizontal_axis"][0] * randRange(1, 0.7, 1.3)[0]
            )
            parameter["horizontal_axis"][0] = horizontal_axis
            vertical_mul = randRange(parameter["vertical_axis"].shape[0], 0.9, 1.3)
            parameter["vertical_axis"] *= vertical_mul
            exist_mul = randRange(parameter["exist_angle"].shape[0], 0.7, 1.3)
            parameter["exist_angle"] *= exist_mul
            bottom_mul = randRange(parameter["bottom_size"].shape[0], 0.9, 1.3)
            parameter["bottom_size"] *= bottom_mul
            parameter["x_z_ratio"][0] = 1
            # thickness_mul = randRange(parameter['thickness'].shape[0],0.7,1.3)
            # parameter['thickness'] *= thickness_mul
            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Multilevel_Body":
            body_type = "Multilevel_Body"
            parameter["num_levels"][0] = np.random.randint(1, 6)
            parameter["num_levels"] = np.array(parameter["num_levels"], dtype=int)
            level_1_bottom_diff = (
                parameter["level_1_bottom_radius"][0]
                - parameter["level_1_bottom_radius"][1]
            )
            bottom_radius_diff = level_1_bottom_diff * randRange(1, 0.7, 1.3)[0]
            parameter["level_1_bottom_radius"][0] *= randRange(1, 0.7, 1.3)[0]
            parameter["level_1_bottom_radius"][1] = (
                parameter["level_1_bottom_radius"][0] - bottom_radius_diff
            )
            for i in range(parameter["num_levels"][0]):
                cur_height = "level_" + str(i + 1) + "_height"
                cur_top = "level_" + str(i + 1) + "_top_radius"
                if i == 0:
                    top_radius_diff = parameter[cur_top][0] - parameter[cur_top][1]
                    top_radius_diff *= randRange(1, 0.7, 1.3)[0]
                    parameter[cur_top][0] *= randRange(1, 0.8, 1.2)[0]
                    parameter[cur_top][1] = parameter[cur_top][0] - top_radius_diff
                    height_diff = parameter[cur_height][0] - parameter[cur_height][1]
                    height_diff *= randRange(1, 0.7, 1.3)[0]
                    parameter[cur_height][0] *= randRange(1, 0.7, 1.3)[0]
                    parameter[cur_height][1] = parameter[cur_height][0] - height_diff
                else:
                    top_radius_diff = parameter[cur_top][0] - parameter[cur_top][1]
                    top_radius_diff *= randRange(1, 0.7, 1.3)[0]
                    parameter[cur_top][0] *= randRange(1, 0.8, 1.2)[0]
                    tmp_radius = (
                        parameter["level_" + str(i) + "_top_radius"][0]
                        * randRange(1, 0.8, 0.9)[0]
                    )
                    parameter[cur_top][0] = np.minimum(
                        parameter[cur_top][0], tmp_radius
                    )
                    parameter[cur_top][1] = parameter[cur_top][0] - top_radius_diff
                    parameter[cur_height][0] *= randRange(1, 0.7, 1.3)[0]
            parameter["x_z_ratio"][0] = 1
            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Standard_Cover":
            body_parameter = concepts[0]["parameters"]

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"] = body_parameter["rotation"]
            parameter["num_knobs"][0] = np.random.randint(1, 6)
            parameter["num_knobs"] = np.array(parameter["num_knobs"], dtype=int)

            size_diff = parameter["outer_size"] - parameter["inner_size"]
            size_diff *= randRange(size_diff.shape[0], 0.7, 1.3)

            if body_type == "Semi_Spherical_Body":
                cur_body_exist_angle = body_parameter["exist_angle"][0] * np.pi / 180
                cur_inner_size_1 = body_parameter["horizontal_axis"][0] * np.cos(
                    cur_body_exist_angle
                )
                # if cur_inner_size_1 < 0:
                #     cur_inner_size_1 += body_parameter['thickness'][0]

                parameter["position"][1] += body_parameter["vertical_axis"][0] * np.sin(
                    cur_body_exist_angle
                )

            elif body_type == "Spherical_Cylindrical_Body":
                cur_body_exist_angle = body_parameter["exist_angle"][0] * np.pi / 180
                cur_inner_size_1 = body_parameter["horizontal_axis"][0] * np.cos(
                    cur_body_exist_angle
                )
                # cur_inner_size_1 -= body_parameter['thickness'][0]
                # if cur_inner_size_1 < 0:
                #     cur_inner_size_1 += body_parameter['thickness'][0]

                parameter["position"][1] += body_parameter["vertical_axis"][0] * np.sin(
                    cur_body_exist_angle
                )

            else:
                number_of_cur_levels = int(body_parameter["num_levels"][0])
                cur_inner_size_1 = body_parameter[
                    "level_" + str(number_of_cur_levels) + "_top_radius"
                ][1]
                total_height = -body_parameter["level_1_height"][0] / 2
                for i in range(number_of_cur_levels):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]

                parameter["position"][1] += total_height

            cur_mul = cur_inner_size_1 / parameter["inner_size"][1]
            parameter["inner_size"][1] = cur_inner_size_1
            parameter["inner_size"][0] *= cur_mul * randRange(1, 0.7, 1.3)[0]
            parameter["inner_size"][2] *= cur_mul * randRange(1, 0.7, 1.3)[0]
            parameter["outer_size"] = parameter["inner_size"] + size_diff

            for i in range(int(parameter["num_knobs"][0])):
                string_new = "knob_" + str(i + 1) + "_size"
                string_old = "knob_" + str(i) + "_size"
                knob_size_mul = randRange(parameter[string_new].shape[0], 0.7, 1.3)
                parameter[string_new] *= knob_size_mul
                if i > 0:
                    parameter[string_new][1] = parameter[string_old][0]

            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Trifold_Handle":
            body_parameter = concepts[0]["parameters"]
            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            thickness_mul = randRange(
                parameter["horizontal_thickness"].shape[0], 0.7, 1.3
            )
            length_mul = randRange(parameter["horizontal_length"].shape[0], 0.7, 1.3)
            vertical_mul = randRange(parameter["vertical_thickness"].shape[0], 0.7, 1.3)
            horizontal_mul = randRange(
                parameter["horizontal_rotation"].shape[0], 0.7, 1.3
            )
            separation_mul = randRange(
                parameter["horizontal_separation"].shape[0], 0.7, 1.3
            )
            offset_mul = randRange(parameter["mounting_offset"].shape[0], 0.9, 1.1)

            parameter["horizontal_thickness"] *= thickness_mul
            parameter["horizontal_length"] *= length_mul
            parameter["vertical_thickness"] *= vertical_mul
            parameter["horizontal_rotation"] *= horizontal_mul
            parameter["horizontal_separation"] *= separation_mul
            parameter["mounting_offset"] *= offset_mul

            if body_type == "Semi_Spherical_Body":
                exist_angle_up = body_parameter["exist_angle"][0]
                exist_angle_down = body_parameter["exist_angle"][1]
                body_radius_up = body_parameter["vertical_axis"][0]
                body_radius_down = body_parameter["vertical_axis"][1]
                body_horizontal_radius = body_parameter["horizontal_axis"][0]

                cur_angle_up = (
                    np.random.uniform(5, exist_angle_up * 2 / 3) / 180 * np.pi
                )
                cur_angle_down = (
                    np.random.uniform(5, exist_angle_down * 2 / 3) / 180 * np.pi
                )

                handle_position_up = get_handle_position(
                    cur_angle_up, body_radius_up, body_horizontal_radius
                )
                handle_position_down = get_handle_position(
                    cur_angle_down, body_radius_down, body_horizontal_radius
                )

                cur_vertical_total_height = (
                    handle_position_up[1] + handle_position_down[1]
                ) * randRange(1, 1.0, 1.2)[0]
                parameter["horizontal_separation"][0] = (
                    handle_position_up[1] + handle_position_down[1]
                )

                virtual_handle_length_up = (
                    cur_vertical_total_height - parameter["horizontal_separation"][0]
                ) / 2
                virtual_handle_length_down = (
                    cur_vertical_total_height - parameter["horizontal_separation"][0]
                ) / 2

                basic_offset_x = (
                    body_parameter["horizontal_axis"][0] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                total_offset_x_up = basic_offset_x + (
                    body_parameter["horizontal_axis"][0] - handle_position_up[0]
                )
                total_offset_x_down = basic_offset_x + (
                    body_parameter["horizontal_axis"][0] - handle_position_down[0]
                )
                parameter["horizontal_rotation"][0] = (
                    np.arctan(virtual_handle_length_up / total_offset_x_up)
                    / np.pi
                    * 180
                )
                parameter["horizontal_rotation"][1] = (
                    -np.arctan(virtual_handle_length_down / total_offset_x_down)
                    / np.pi
                    * 180
                )
                parameter["horizontal_length"][0] = (
                    np.sqrt(virtual_handle_length_up**2 + total_offset_x_up**2)
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["horizontal_length"][1] = (
                    np.sqrt(virtual_handle_length_down**2 + total_offset_x_down**2)
                    * randRange(1, 0.8, 1.2)[0]
                )

                real_offset_y = (
                    -(handle_position_up[1] + handle_position_down[1]) / 2
                    + handle_position_up[1]
                )
                top_offset = -handle_position_down[0] + handle_position_up[0]
                parameter["mounting_offset"][0] = top_offset
                parameter["position"][1] += real_offset_y
                parameter["position"][2] -= handle_position_down[0]

            elif body_type == "Spherical_Cylindrical_Body":
                exist_angle_up = body_parameter["exist_angle"][0]
                body_radius_up = body_parameter["vertical_axis"][0]
                body_bottom_radius = body_parameter["bottom_size"][0]
                body_horizontal_radius = body_parameter["horizontal_axis"][0]
                bottom_height = body_parameter["bottom_size"][1]
                cur_angle_up = (
                    np.random.uniform(5, exist_angle_up * 2 / 3) / 180 * np.pi
                )

                handle_position_up = get_handle_position(
                    cur_angle_up, body_radius_up, body_horizontal_radius
                )
                handle_position_down = [0, 0]
                handle_position_down[0] = np.random.uniform(
                    (body_bottom_radius + body_horizontal_radius) / 2,
                    body_horizontal_radius,
                )
                handle_position_down[1] = (
                    handle_position_down[0] - body_bottom_radius
                ) * (bottom_height / (body_horizontal_radius - body_bottom_radius))
                handle_position_down[1] = bottom_height - handle_position_down[1]

                cur_vertical_total_height = (
                    handle_position_up[1] + handle_position_down[1]
                ) * randRange(1, 1.0, 1.2)[0]
                parameter["horizontal_separation"][0] = (
                    handle_position_up[1] + handle_position_down[1]
                )

                virtual_handle_length_up = (
                    cur_vertical_total_height - parameter["horizontal_separation"][0]
                ) / 2
                virtual_handle_length_down = (
                    cur_vertical_total_height - parameter["horizontal_separation"][0]
                ) / 2

                basic_offset_x = (
                    body_parameter["horizontal_axis"][0] / 4 * randRange(1, 0.8, 1.2)[0]
                )
                total_offset_x_up = basic_offset_x + (
                    body_parameter["horizontal_axis"][0] - handle_position_up[0]
                )
                total_offset_x_down = basic_offset_x + (
                    body_parameter["horizontal_axis"][0] - handle_position_down[0]
                )
                parameter["horizontal_rotation"][0] = (
                    np.arctan(virtual_handle_length_up / total_offset_x_up)
                    / np.pi
                    * 180
                )
                parameter["horizontal_rotation"][1] = (
                    -np.arctan(virtual_handle_length_down / total_offset_x_down)
                    / np.pi
                    * 180
                )
                parameter["horizontal_length"][0] = (
                    np.sqrt(virtual_handle_length_up**2 + total_offset_x_up**2)
                    * randRange(1, 0.8, 1.2)[0]
                )
                parameter["horizontal_length"][1] = (
                    np.sqrt(virtual_handle_length_down**2 + total_offset_x_down**2)
                    * randRange(1, 0.8, 1.2)[0]
                )

                real_offset_y = (
                    -(handle_position_up[1] + handle_position_down[1]) / 2
                    + handle_position_up[1]
                )
                top_offset = -handle_position_down[0] + handle_position_up[0]
                parameter["mounting_offset"][0] = top_offset
                parameter["position"][1] += real_offset_y
                parameter["position"][2] -= handle_position_down[0]

            elif body_type == "Multilevel_Body":
                cur_num = body_parameter["num_levels"][0]
                total_height = -body_parameter["level_1_height"][0] / 2
                last_body_level = int(cur_num / 2)
                if last_body_level == 0:
                    last_body_level = 1
                top_body_level = cur_num
                target_handle_level = np.random.randint(
                    last_body_level, top_body_level + 1
                )
                target_level_radius_up = body_parameter[
                    "level_" + str(target_handle_level) + "_top_radius"
                ][0]
                target_level_height = body_parameter[
                    "level_" + str(target_handle_level) + "_height"
                ][0]
                for i in range(target_handle_level - 1):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]
                if target_handle_level == 1:
                    target_level_radius_down = body_parameter["level_1_bottom_radius"][
                        0
                    ]
                else:
                    target_level_radius_down = body_parameter[
                        "level_" + str(target_handle_level - 1) + "_top_radius"
                    ][0]
                slop = target_level_height / (
                    target_level_radius_up - target_level_radius_down
                )
                handle_pos_point_1 = [0, 0]
                handle_pos_point_2 = [0, 0]
                radius_min = np.minimum(
                    target_level_radius_down, target_level_radius_up
                )
                radius_max = np.maximum(
                    target_level_radius_down, target_level_radius_up
                )
                handle_pos_point_1[0] = np.random.uniform(
                    radius_min, (radius_min * 3 / 4 + radius_max / 4)
                )
                handle_pos_point_2[0] = np.random.uniform(
                    (radius_min / 4 + radius_max * 3 / 4), radius_max
                )
                if slop < 0:
                    handle_pos_point_1[1] = slop * (
                        -handle_pos_point_1[0] + target_level_radius_up
                    )
                    handle_pos_point_2[1] = -slop * (
                        target_level_radius_down - handle_pos_point_2[0]
                    )
                    height_of_up_from_half_line = (
                        target_level_height / 2 - handle_pos_point_1[1]
                    )
                    height_of_down_from_half_line = (
                        target_level_height / 2 - handle_pos_point_2[1]
                    )
                else:
                    handle_pos_point_1[1] = slop * (
                        handle_pos_point_1[0] - target_level_radius_down
                    )
                    handle_pos_point_2[1] = slop * (
                        target_level_radius_up - handle_pos_point_2[0]
                    )
                    height_of_up_from_half_line = (
                        target_level_height / 2 - handle_pos_point_2[1]
                    )
                    height_of_down_from_half_line = (
                        target_level_height / 2 - handle_pos_point_1[1]
                    )

                cur_vertical_total_height = (
                    height_of_up_from_half_line + height_of_down_from_half_line
                ) * randRange(1, 1.0, 1.2)[0]
                parameter["horizontal_separation"][0] = (
                    height_of_up_from_half_line + height_of_down_from_half_line
                )

                virtual_handle_length_up = (
                    cur_vertical_total_height - parameter["horizontal_separation"][0]
                ) / 2
                virtual_handle_length_down = (
                    cur_vertical_total_height - parameter["horizontal_separation"][0]
                ) / 2

                basic_offset_x = (
                    (target_level_radius_down + target_level_radius_up)
                    / 4
                    * randRange(1, 0.7, 1.2)[0]
                )
                if slop < 0:
                    total_offset_x_up = basic_offset_x + (
                        (target_level_radius_up + target_level_radius_down) / 2
                        - handle_pos_point_1[0]
                    )
                    total_offset_x_down = basic_offset_x + (
                        (target_level_radius_up + target_level_radius_down) / 2
                        - handle_pos_point_2[0]
                    )
                    parameter["horizontal_rotation"][0] = (
                        np.arctan(virtual_handle_length_up / total_offset_x_up)
                        / np.pi
                        * 180
                    )
                    parameter["horizontal_rotation"][1] = (
                        -np.arctan(virtual_handle_length_down / total_offset_x_down)
                        / np.pi
                        * 180
                    )
                    parameter["horizontal_length"][0] = (
                        np.sqrt(virtual_handle_length_up**2 + total_offset_x_up**2)
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["horizontal_length"][1] = (
                        np.sqrt(virtual_handle_length_down**2 + total_offset_x_down**2)
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    top_offset = -handle_pos_point_1[0] + handle_pos_point_2[0]
                    parameter["position"][2] -= handle_pos_point_2[0]
                else:
                    total_offset_x_up = basic_offset_x + (
                        (target_level_radius_up + target_level_radius_down) / 2
                        - handle_pos_point_2[0]
                    )
                    total_offset_x_down = basic_offset_x + (
                        (target_level_radius_up + target_level_radius_down) / 2
                        - handle_pos_point_1[0]
                    )
                    parameter["horizontal_rotation"][0] = (
                        np.arctan(virtual_handle_length_up / total_offset_x_up)
                        / np.pi
                        * 180
                    )
                    parameter["horizontal_rotation"][1] = (
                        -np.arctan(virtual_handle_length_down / total_offset_x_down)
                        / np.pi
                        * 180
                    )
                    parameter["horizontal_length"][0] = (
                        np.sqrt(virtual_handle_length_up**2 + total_offset_x_up**2)
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["horizontal_length"][1] = (
                        np.sqrt(virtual_handle_length_down**2 + total_offset_x_down**2)
                        * randRange(1, 0.8, 1.2)[0]
                    )
                    top_offset = -handle_pos_point_2[0] + handle_pos_point_1[0]
                    parameter["position"][2] -= handle_pos_point_1[0]

                real_offset_y = (
                    -(height_of_up_from_half_line + height_of_down_from_half_line) / 2
                    + height_of_up_from_half_line
                )
                parameter["mounting_offset"][0] = -top_offset
                parameter["position"][1] += (
                    real_offset_y + total_height + target_level_height / 2
                )

            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Curved_Handle":
            body_parameter = concepts[0]["parameters"]

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            size_mul = randRange(parameter["radius"].shape[0], 0.7, 1.3)
            exist_mul = randRange(parameter["exist_angle"].shape[0], 0.85, 1.2)
            parameter["radius"] *= size_mul
            parameter["exist_angle"] *= exist_mul

            if body_type == "Semi_Spherical_Body":
                parameter["radius"][0] = (
                    body_parameter["horizontal_axis"][0] / 3 * randRange(1, 0.8, 1.2)[0]
                )
                exist_angle_up = body_parameter["exist_angle"][0]
                exist_angle_down = body_parameter["exist_angle"][1]
                body_radius_up = body_parameter["vertical_axis"][0]
                body_radius_down = body_parameter["vertical_axis"][1]
                body_horizontal_radius = body_parameter["horizontal_axis"][0]

                cur_angle_up = (
                    np.random.uniform(5, exist_angle_up * 2 / 3) / 180 * np.pi
                )
                cur_angle_down = (
                    np.random.uniform(5, exist_angle_down * 2 / 3) / 180 * np.pi
                )

                handle_position_up = get_handle_position(
                    cur_angle_up, body_radius_up, body_horizontal_radius
                )
                handle_position_down = get_handle_position(
                    cur_angle_down, body_radius_down, body_horizontal_radius
                )

                (
                    parameter["radius"][0],
                    center_point,
                    parameter["exist_angle"][0],
                    additional_curve_rot_angle,
                ) = get_center(
                    handle_position_up,
                    handle_position_down,
                    parameter["radius"][0],
                    body_parameter["horizontal_axis"][0],
                )
                parameter["radius"][1] = (
                    parameter["radius"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rotation"][0] += additional_curve_rot_angle
                parameter["position"][1] += center_point[1]
                parameter["position"][2] -= center_point[0]

            elif body_type == "Spherical_Cylindrical_Body":
                parameter["radius"][0] = (
                    body_parameter["horizontal_axis"][0] / 3 * randRange(1, 0.8, 1.2)[0]
                )
                exist_angle_up = body_parameter["exist_angle"][0]
                body_radius_up = body_parameter["vertical_axis"][0]
                body_bottom_radius = body_parameter["bottom_size"][0]
                body_horizontal_radius = body_parameter["horizontal_axis"][0]
                bottom_height = body_parameter["bottom_size"][1]
                cur_angle_up = (
                    np.random.uniform(5, exist_angle_up * 2 / 3) / 180 * np.pi
                )

                handle_position_up = get_handle_position(
                    cur_angle_up, body_radius_up, body_horizontal_radius
                )
                handle_position_down = [0, 0]
                handle_position_down[0] = np.random.uniform(
                    (body_bottom_radius + body_horizontal_radius) / 2,
                    body_horizontal_radius,
                )
                handle_position_down[1] = (
                    handle_position_down[0] - body_bottom_radius
                ) * (bottom_height / (body_horizontal_radius - body_bottom_radius))
                handle_position_down[1] = bottom_height - handle_position_down[1]

                (
                    parameter["radius"][0],
                    center_point,
                    parameter["exist_angle"][0],
                    additional_curve_rot_angle,
                ) = get_center(
                    handle_position_up,
                    handle_position_down,
                    parameter["radius"][0],
                    body_parameter["horizontal_axis"][0],
                )
                parameter["radius"][1] = (
                    parameter["radius"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rotation"][0] += additional_curve_rot_angle
                parameter["position"][1] += center_point[1]
                parameter["position"][2] -= center_point[0]

            elif body_type == "Multilevel_Body":
                cur_num = body_parameter["num_levels"][0]
                total_height = -body_parameter["level_1_height"][0] / 2
                last_body_level = int(cur_num / 2)
                if last_body_level == 0:
                    last_body_level = 1
                top_body_level = cur_num
                target_handle_level = np.random.randint(
                    last_body_level, top_body_level + 1
                )
                target_level_radius_up = body_parameter[
                    "level_" + str(target_handle_level) + "_top_radius"
                ][0]
                target_level_height = body_parameter[
                    "level_" + str(target_handle_level) + "_height"
                ][0]
                for i in range(target_handle_level - 1):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]
                if target_handle_level == 1:
                    target_level_radius_down = body_parameter["level_1_bottom_radius"][
                        0
                    ]
                else:
                    target_level_radius_down = body_parameter[
                        "level_" + str(target_handle_level - 1) + "_top_radius"
                    ][0]
                slop = target_level_height / (
                    target_level_radius_up - target_level_radius_down
                )
                handle_pos_point_1 = [0, 0]
                handle_pos_point_2 = [0, 0]
                radius_min = np.minimum(
                    target_level_radius_down, target_level_radius_up
                )
                radius_max = np.maximum(
                    target_level_radius_down, target_level_radius_up
                )
                handle_pos_point_1[0] = np.random.uniform(
                    radius_min, (radius_min * 3 / 4 + radius_max / 4)
                )
                handle_pos_point_2[0] = np.random.uniform(
                    (radius_min / 4 + radius_max * 3 / 4), radius_max
                )
                if slop < 0:
                    handle_pos_point_1[1] = slop * (
                        -handle_pos_point_1[0] + target_level_radius_up
                    )
                    handle_pos_point_2[1] = -slop * (
                        target_level_radius_down - handle_pos_point_2[0]
                    )

                    handle_pos_point_1[1] = target_level_height - handle_pos_point_1[1]
                    handle_pos_point_1[1] += total_height
                    handle_pos_point_2[1] += total_height
                else:
                    handle_pos_point_1[1] = slop * (
                        handle_pos_point_1[0] - target_level_radius_down
                    )
                    handle_pos_point_2[1] = slop * (
                        target_level_radius_up - handle_pos_point_2[0]
                    )

                    handle_pos_point_2[1] = target_level_height - handle_pos_point_2[1]
                    handle_pos_point_1[1] += total_height
                    handle_pos_point_2[1] += total_height

                if slop < 0:
                    (
                        parameter["radius"][0],
                        center_point,
                        parameter["exist_angle"][0],
                        additional_curve_rot_angle,
                    ) = multilevel_get_center(
                        handle_pos_point_1,
                        handle_pos_point_2,
                        target_handle_level,
                        (radius_min + radius_max) / 2,
                    )
                    parameter["radius"][1] = (
                        parameter["radius"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rotation"][0] += additional_curve_rot_angle
                    parameter["position"][1] += center_point[1]
                    parameter["position"][2] -= center_point[0]
                else:
                    (
                        parameter["radius"][0],
                        center_point,
                        parameter["exist_angle"][0],
                        additional_curve_rot_angle,
                    ) = multilevel_get_center(
                        handle_pos_point_2,
                        handle_pos_point_1,
                        target_handle_level,
                        (radius_min + radius_max) / 2,
                    )
                    parameter["radius"][1] = (
                        parameter["radius"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rotation"][0] += additional_curve_rot_angle
                    parameter["position"][1] += center_point[1]
                    parameter["position"][2] -= center_point[0]

            parameter["rotation"][0] -= (parameter["exist_angle"][0] - 180) / 2

            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Ring_Handle":
            body_parameter = concepts[0]["parameters"]

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
            diff_radius = (parameter["size"][0] - parameter["size"][1]) * randRange(
                1, 0.8, 1.2
            )[0]
            exist_mul = randRange(parameter["exist_angle"].shape[0], 0.85, 1.3)
            parameter["size"] *= size_mul
            parameter["size"][1] = parameter["size"][0] - diff_radius
            parameter["exist_angle"] *= exist_mul

            if body_type == "Semi_Spherical_Body":
                parameter["size"][0] = (
                    body_parameter["horizontal_axis"][0] / 3 * randRange(1, 0.8, 1.2)[0]
                )
                exist_angle_up = body_parameter["exist_angle"][0]
                exist_angle_down = body_parameter["exist_angle"][1]
                body_radius_up = body_parameter["vertical_axis"][0]
                body_radius_down = body_parameter["vertical_axis"][1]
                body_horizontal_radius = body_parameter["horizontal_axis"][0]

                cur_angle_up = (
                    np.random.uniform(5, exist_angle_up * 2 / 3) / 180 * np.pi
                )
                cur_angle_down = (
                    np.random.uniform(5, exist_angle_down * 2 / 3) / 180 * np.pi
                )

                handle_position_up = get_handle_position(
                    cur_angle_up, body_radius_up, body_horizontal_radius
                )
                handle_position_down = get_handle_position(
                    cur_angle_down, body_radius_down, body_horizontal_radius
                )

                (
                    parameter["size"][0],
                    center_point,
                    parameter["exist_angle"][0],
                    additional_curve_rot_angle,
                ) = get_center(
                    handle_position_up,
                    handle_position_down,
                    parameter["size"][0],
                    body_parameter["horizontal_axis"][0],
                )
                parameter["size"][1] = (
                    parameter["size"][0]
                    - parameter["size"][0] / 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = (
                    parameter["size"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rotation"][0] += additional_curve_rot_angle
                parameter["position"][1] += center_point[1]
                parameter["position"][2] -= center_point[0]

            elif body_type == "Spherical_Cylindrical_Body":
                parameter["size"][0] = (
                    body_parameter["horizontal_axis"][0] / 3 * randRange(1, 0.8, 1.2)[0]
                )
                exist_angle_up = body_parameter["exist_angle"][0]
                body_radius_up = body_parameter["vertical_axis"][0]
                body_bottom_radius = body_parameter["bottom_size"][0]
                body_horizontal_radius = body_parameter["horizontal_axis"][0]
                bottom_height = body_parameter["bottom_size"][1]
                cur_angle_up = (
                    np.random.uniform(5, exist_angle_up * 2 / 3) / 180 * np.pi
                )

                handle_position_up = get_handle_position(
                    cur_angle_up, body_radius_up, body_horizontal_radius
                )
                handle_position_down = [0, 0]
                handle_position_down[0] = np.random.uniform(
                    (body_bottom_radius + body_horizontal_radius) / 2,
                    body_horizontal_radius,
                )
                handle_position_down[1] = (
                    handle_position_down[0] - body_bottom_radius
                ) * (bottom_height / (body_horizontal_radius - body_bottom_radius))
                handle_position_down[1] = bottom_height - handle_position_down[1]

                (
                    parameter["size"][0],
                    center_point,
                    parameter["exist_angle"][0],
                    additional_curve_rot_angle,
                ) = get_center(
                    handle_position_up,
                    handle_position_down,
                    parameter["size"][0],
                    body_parameter["horizontal_axis"][0],
                )
                parameter["size"][1] = (
                    parameter["size"][0]
                    - parameter["size"][0] / 5 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["size"][2] = (
                    parameter["size"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                )
                parameter["rotation"][0] += additional_curve_rot_angle
                parameter["position"][1] += center_point[1]
                parameter["position"][2] -= center_point[0]

            elif body_type == "Multilevel_Body":
                cur_num = body_parameter["num_levels"][0]
                total_height = -body_parameter["level_1_height"][0] / 2
                last_body_level = int(cur_num / 2)
                if last_body_level == 0:
                    last_body_level = 1
                top_body_level = cur_num
                target_handle_level = np.random.randint(
                    last_body_level, top_body_level + 1
                )
                target_level_radius_up = body_parameter[
                    "level_" + str(target_handle_level) + "_top_radius"
                ][0]
                target_level_height = body_parameter[
                    "level_" + str(target_handle_level) + "_height"
                ][0]
                for i in range(target_handle_level - 1):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]
                if target_handle_level == 1:
                    target_level_radius_down = body_parameter["level_1_bottom_radius"][
                        0
                    ]
                else:
                    target_level_radius_down = body_parameter[
                        "level_" + str(target_handle_level - 1) + "_top_radius"
                    ][0]
                slop = target_level_height / (
                    target_level_radius_up - target_level_radius_down
                )
                handle_pos_point_1 = [0, 0]
                handle_pos_point_2 = [0, 0]
                radius_min = np.minimum(
                    target_level_radius_down, target_level_radius_up
                )
                radius_max = np.maximum(
                    target_level_radius_down, target_level_radius_up
                )
                handle_pos_point_1[0] = np.random.uniform(
                    radius_min, (radius_min * 3 / 4 + radius_max / 4)
                )
                handle_pos_point_2[0] = np.random.uniform(
                    (radius_min / 4 + radius_max * 3 / 4), radius_max
                )
                if slop < 0:
                    handle_pos_point_1[1] = slop * (
                        -handle_pos_point_1[0] + target_level_radius_up
                    )
                    handle_pos_point_2[1] = -slop * (
                        target_level_radius_down - handle_pos_point_2[0]
                    )

                    handle_pos_point_1[1] = target_level_height - handle_pos_point_1[1]
                    handle_pos_point_1[1] += total_height
                    handle_pos_point_2[1] += total_height
                else:
                    handle_pos_point_1[1] = slop * (
                        handle_pos_point_1[0] - target_level_radius_down
                    )
                    handle_pos_point_2[1] = slop * (
                        target_level_radius_up - handle_pos_point_2[0]
                    )

                    handle_pos_point_2[1] = target_level_height - handle_pos_point_2[1]
                    handle_pos_point_1[1] += total_height
                    handle_pos_point_2[1] += total_height

                if slop < 0:
                    (
                        parameter["size"][0],
                        center_point,
                        parameter["exist_angle"][0],
                        additional_curve_rot_angle,
                    ) = multilevel_get_center(
                        handle_pos_point_1,
                        handle_pos_point_2,
                        target_handle_level,
                        (radius_min + radius_max) / 2,
                    )
                    parameter["size"][1] = (
                        parameter["size"][0]
                        - parameter["size"][0] / 5 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["size"][2] = (
                        parameter["size"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rotation"][0] += additional_curve_rot_angle
                    parameter["position"][1] += center_point[1]
                    parameter["position"][2] -= center_point[0]
                else:
                    (
                        parameter["size"][0],
                        center_point,
                        parameter["exist_angle"][0],
                        additional_curve_rot_angle,
                    ) = multilevel_get_center(
                        handle_pos_point_2,
                        handle_pos_point_1,
                        target_handle_level,
                        (radius_min + radius_max) / 2,
                    )
                    parameter["size"][1] = (
                        parameter["size"][0]
                        - parameter["size"][0] / 5 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["size"][2] = (
                        parameter["size"][0] / 10 * randRange(1, 0.8, 1.2)[0]
                    )
                    parameter["rotation"][0] += additional_curve_rot_angle
                    parameter["position"][1] += center_point[1]
                    parameter["position"][2] -= center_point[0]

            parameter["rotation"][0] -= (parameter["exist_angle"][0] - 180) / 2

            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Cylindrical_Handle":
            body_parameter = concepts[0]["parameters"]
            size_mul = randRange(parameter["size"].shape[0], 0.7, 1.3)
            parameter["size"] *= size_mul
            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            if body_type == "Semi_Spherical_Body":
                offset_distance = body_parameter["horizontal_axis"][0]
                parameter["position"][2] -= offset_distance * randRange(1, 0.8, 0.9)[0]

            elif body_type == "Spherical_Cylindrical_Body":
                offset_distance = body_parameter["horizontal_axis"][0]
                parameter["position"][2] -= offset_distance * randRange(1, 0.8, 0.9)[0]

            elif body_type == "Multilevel_Body":
                cur_num = body_parameter["num_levels"][0]
                total_height = 0
                minimum_radius = body_parameter["level_1_top_radius"][0]
                for i in range(cur_num):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]
                    if (
                        minimum_radius
                        > body_parameter["level_" + str(i + 1) + "_top_radius"][0]
                    ):
                        minimum_radius = body_parameter[
                            "level_" + str(i + 1) + "_top_radius"
                        ][0]
                parameter["position"][1] += (
                    total_height - body_parameter["level_1_height"][0]
                ) / 2
                parameter["position"][2] -= minimum_radius * randRange(1, 0.8, 0.9)[0]

            parameter["rotation"][0] += 10 * randRange(1, 0.7, 1.3)[0]
            parameter["rotation"][2] += 10 * randRange(1, 0.7, 1.3)[0]

            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Round_U_Handle":
            body_parameter = concepts[0]["parameters"]
            size_mul_mounting = randRange(
                parameter["mounting_radius"].shape[0], 0.7, 1.3
            )
            parameter["mounting_radius"] *= size_mul_mounting
            size_mul_length = randRange(parameter["vertical_length"].shape[0], 0.7, 1.3)
            parameter["vertical_length"] *= size_mul_length

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            if body_type == "Semi_Spherical_Body":
                cur_horizontal_radius = body_parameter["horizontal_axis"][0]
                body_radius_up = body_parameter["vertical_axis"][0]
                exist_angle = body_parameter["exist_angle"][0]
                cur_angle = exist_angle * randRange(1, 0.1, 0.9)[0] / 180 * np.pi

                parameter["position"][1] += body_radius_up * np.sin(cur_angle)

                parameter["vertical_separation"][0] = (
                    2
                    * cur_horizontal_radius
                    * np.cos(cur_angle)
                    * randRange(1, 0.8, 0.9)[0]
                )

                cover_parameter = concepts[1]["parameters"]
                num_of_cover = cover_parameter["num_knobs"][0]
                cover_height = cover_parameter["outer_size"][2]
                exist_angle_up = exist_angle / 180 * np.pi
                for i in range(num_of_cover):
                    cover_height += cover_parameter["knob_" + str(i + 1) + "_size"][2]
                if parameter["vertical_length"][0] + parameter["vertical_separation"][
                    0
                ] / 2 + parameter["position"][1] <= (
                    cover_height + (body_radius_up * np.sin(exist_angle_up))
                ):
                    parameter["vertical_length"][0] = (
                        (cover_height + (body_radius_up * np.sin(exist_angle_up)))
                        - (
                            parameter["vertical_separation"][0] / 2
                            + parameter["position"][1]
                        )
                    ) * randRange(1, 1.1, 1.2)[0]

            elif body_type == "Spherical_Cylindrical_Body":
                cur_horizontal_radius = body_parameter["horizontal_axis"][0]
                body_radius_up = body_parameter["vertical_axis"][0]
                exist_angle = body_parameter["exist_angle"][0]
                cur_angle = exist_angle * randRange(1, 0.1, 0.9)[0] / 180 * np.pi

                parameter["position"][1] += body_radius_up * np.sin(cur_angle)

                parameter["vertical_separation"][0] = (
                    2
                    * cur_horizontal_radius
                    * np.cos(cur_angle)
                    * randRange(1, 0.8, 0.9)[0]
                )

                cover_parameter = concepts[1]["parameters"]
                num_of_cover = cover_parameter["num_knobs"][0]
                cover_height = cover_parameter["outer_size"][2]
                exist_angle_up = exist_angle / 180 * np.pi
                for i in range(num_of_cover):
                    cover_height += cover_parameter["knob_" + str(i + 1) + "_size"][2]

                parameter["vertical_length"][0] = (
                    (cover_height + (body_radius_up * np.sin(exist_angle_up)))
                    - (
                        parameter["vertical_separation"][0] / 2
                        + parameter["position"][1]
                    )
                ) * randRange(1, 1.1, 1.2)[0]

            elif body_type == "Multilevel_Body":
                cur_num = body_parameter["num_levels"][0]
                total_height = 0
                minimum_radius = body_parameter["level_1_top_radius"][0]
                for i in range(cur_num):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]
                    if (
                        minimum_radius
                        > body_parameter["level_" + str(i + 1) + "_top_radius"][0]
                    ):
                        minimum_radius = body_parameter[
                            "level_" + str(i + 1) + "_top_radius"
                        ][0]
                parameter["position"][1] += (
                    total_height - body_parameter["level_1_height"][0] / 2
                )
                parameter["vertical_separation"][0] = (
                    body_parameter["level_" + str(cur_num) + "_top_radius"][0]
                    * 2
                    * randRange(1, 0.8, 0.9)[0]
                )

                cover_parameter = concepts[1]["parameters"]
                num_of_cover = cover_parameter["num_knobs"][0]
                cover_height = cover_parameter["outer_size"][2]
                for i in range(num_of_cover):
                    cover_height += cover_parameter["knob_" + str(i + 1) + "_size"][2]
                if (
                    parameter["vertical_length"][0]
                    + parameter["vertical_separation"][0] / 2
                    <= cover_height
                ):
                    parameter["vertical_length"][0] = (
                        cover_height - parameter["vertical_separation"][0] / 2
                    ) * randRange(1, 1.1, 1.3)[0]

            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Flat_U_Handle":
            body_parameter = concepts[0]["parameters"]
            parameter["vertical_size"] *= randRange(
                parameter["vertical_size"].shape[0], 0.7, 1.3
            )

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            if body_type == "Semi_Spherical_Body":
                cur_horizontal_radius = body_parameter["horizontal_axis"][0]
                body_radius_up = body_parameter["vertical_axis"][0]
                exist_angle = body_parameter["exist_angle"][0]
                cur_angle = exist_angle * randRange(1, 0.1, 0.9)[0] / 180 * np.pi

                parameter["position"][1] += body_radius_up * np.sin(cur_angle)

                parameter["vertical_separation"][0] = (
                    2
                    * cur_horizontal_radius
                    * np.cos(cur_angle)
                    * randRange(1, 0.8, 0.9)[0]
                )

                cover_parameter = concepts[1]["parameters"]
                num_of_cover = cover_parameter["num_knobs"][0]
                cover_height = cover_parameter["outer_size"][2]
                exist_angle_up = exist_angle / 180 * np.pi
                for i in range(num_of_cover):
                    cover_height += cover_parameter["knob_" + str(i + 1) + "_size"][2]
                if parameter["vertical_size"][1] + (
                    parameter["vertical_separation"][0] - parameter["vertical_size"][0]
                ) / 2 + parameter["position"][1] <= (
                    cover_height + (body_radius_up * np.sin(exist_angle_up))
                ):
                    parameter["vertical_size"][1] = (
                        cover_height
                        + (body_radius_up * np.sin(exist_angle_up))
                        - (
                            (
                                parameter["vertical_separation"][0]
                                - parameter["vertical_size"][0]
                            )
                            / 2
                            + parameter["position"][1]
                        )
                    ) * randRange(1, 1.1, 1.3)[0]

            elif body_type == "Spherical_Cylindrical_Body":
                cur_horizontal_radius = body_parameter["horizontal_axis"][0]
                body_radius_up = body_parameter["vertical_axis"][0]
                exist_angle = body_parameter["exist_angle"][0]
                cur_angle = exist_angle * randRange(1, 0.1, 0.9)[0] / 180 * np.pi

                parameter["position"][1] += body_radius_up * np.sin(cur_angle)

                parameter["vertical_separation"][0] = (
                    2
                    * cur_horizontal_radius
                    * np.cos(cur_angle)
                    * randRange(1, 0.8, 0.9)[0]
                )

                cover_parameter = concepts[1]["parameters"]
                num_of_cover = cover_parameter["num_knobs"][0]
                cover_height = cover_parameter["outer_size"][2]
                exist_angle_up = exist_angle / 180 * np.pi
                for i in range(num_of_cover):
                    cover_height += cover_parameter["knob_" + str(i + 1) + "_size"][2]
                if parameter["vertical_size"][1] + (
                    parameter["vertical_separation"][0] - parameter["vertical_size"][0]
                ) / 2 + parameter["position"][1] <= (
                    cover_height + (body_radius_up * np.sin(exist_angle_up))
                ):
                    parameter["vertical_size"][1] = (
                        cover_height
                        + (body_radius_up * np.sin(exist_angle_up))
                        - (
                            (
                                parameter["vertical_separation"][0]
                                - parameter["vertical_size"][0]
                            )
                            / 2
                            + parameter["position"][1]
                        )
                    ) * randRange(1, 1.1, 1.3)[0]

            elif body_type == "Multilevel_Body":
                cur_num = body_parameter["num_levels"][0]
                total_height = 0
                minimum_radius = body_parameter["level_1_top_radius"][0]
                for i in range(cur_num):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]
                    if (
                        minimum_radius
                        > body_parameter["level_" + str(i + 1) + "_top_radius"][0]
                    ):
                        minimum_radius = body_parameter[
                            "level_" + str(i + 1) + "_top_radius"
                        ][0]
                parameter["position"][1] += (
                    total_height - body_parameter["level_1_height"][0] / 2
                )
                parameter["vertical_separation"][0] = (
                    body_parameter["level_" + str(cur_num) + "_top_radius"][0]
                    + body_parameter["level_" + str(cur_num) + "_top_radius"][0]
                ) * randRange(1, 0.8, 0.9)[0]

                cover_parameter = concepts[1]["parameters"]
                num_of_cover = cover_parameter["num_knobs"][0]
                cover_height = cover_parameter["outer_size"][2]
                for i in range(num_of_cover):
                    cover_height += cover_parameter["knob_" + str(i + 1) + "_size"][2]
                if (
                    parameter["vertical_size"][1]
                    + (
                        parameter["vertical_separation"][0]
                        - parameter["vertical_size"][0]
                    )
                    / 2
                    + parameter["position"][1]
                    <= cover_height
                ):
                    parameter["vertical_size"][1] = (
                        cover_height
                        - (
                            parameter["vertical_separation"][0]
                            - parameter["vertical_size"][0]
                        )
                        / 2
                    ) * randRange(1, 1.2, 1.3)[0]

            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Straight_Spout":
            body_parameter = concepts[0]["parameters"]
            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]
            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            parameter["num_of_sub_spouts"][0] = 1
            parameter["num_of_sub_spouts"] = np.array(
                parameter["num_of_sub_spouts"], dtype=int
            )
            for i in range(parameter["num_of_sub_spouts"][0]):
                size_mul_1 = randRange(
                    parameter["spout_" + str(i + 1) + "_radius"].shape[0], 0.8, 1.2
                )
                size_mul_2 = randRange(
                    parameter["spout_" + str(i + 1) + "_thinkness"].shape[0], 0.8, 1
                )
                size_mul_3 = randRange(
                    parameter["spout_" + str(i + 1) + "_length"].shape[0], 0.8, 1.2
                )
                size_mul_4 = randRange(
                    parameter["spout_" + str(i + 1) + "_generatrix_offset"].shape[0],
                    0.8,
                    1.2,
                )
                parameter["spout_" + str(i + 1) + "_radius"] *= size_mul_1
                if i == 0:
                    parameter["spout_" + str(i + 1) + "_thinkness"] *= size_mul_2
                else:
                    parameter["spout_" + str(i + 1) + "_thinkness"][0] = parameter[
                        "spout_" + str(i) + "_thinkness"
                    ][0]
                parameter["spout_" + str(i + 1) + "_length"] *= size_mul_3
                parameter["spout_" + str(i + 1) + "_generatrix_offset"] *= size_mul_4

            exist_angle_up = parameter["spout_1_rotation"][0] / 180 * np.pi
            parameter["spout_1_generatrix_offset"][0] = (
                parameter["spout_1_length"][1] - parameter["spout_1_length"][0]
            ) / 2 - (2 * parameter["spout_1_radius"][1] * np.tan(exist_angle_up))

            for i in range(parameter["num_of_sub_spouts"][0] - 1):
                cur_string = "spout_" + str(i + 2)
                last_string = "spout_" + str(i + 1)
                tmp_x_z_ratio = 1
                if (
                    parameter[cur_string + "_rotation"][0]
                    > parameter[last_string + "_rotation"][0]
                ):
                    parameter[cur_string + "_rotation"][0] *= randRange(1, 0.8, 1.2)[0]
                    cur_length_diff = (
                        parameter[cur_string + "_length"][1]
                        - parameter[cur_string + "_length"][0]
                    )
                    last_cur_length_diff = (
                        parameter[last_string + "_length"][1]
                        + parameter[last_string + "_generatrix_offset"][0]
                        - (
                            parameter[last_string + "_length"][1]
                            - parameter[last_string + "_length"][0]
                        )
                        / 2
                        - parameter[last_string + "_length"][0]
                    )
                    last_angle = np.arctan(
                        2
                        * (parameter[last_string + "_radius"][0])
                        * tmp_x_z_ratio
                        / last_cur_length_diff
                    )

                    tilt_length = last_cur_length_diff / np.cos(last_angle)
                    cur_real_rotation = (
                        np.pi / 2
                        - (
                            (
                                parameter[cur_string + "_rotation"][0]
                                - parameter[last_string + "_rotation"][0]
                            )
                            / 180
                            * np.pi
                        )
                        - last_angle
                    )

                    cur_offset = (
                        np.sin(cur_real_rotation) * tilt_length + cur_length_diff / 2
                    )
                    parameter[cur_string + "_generatrix_offset"][0] = cur_offset
                    parameter[cur_string + "_radius"][1] = (
                        tilt_length * np.cos(cur_real_rotation) / 2
                    )
                    parameter[cur_string + "_thinkness"][0] = (
                        parameter[last_string + "_thinkness"][0]
                        * parameter[cur_string + "_radius"][1]
                        / (parameter[last_string + "_radius"][0] * tmp_x_z_ratio)
                    ) * randRange(1, 0.7, 0.8)[0]
                else:
                    is_offset_negative = 0
                    parameter[cur_string + "_rotation"][0] *= randRange(1, 0.8, 1.2)[0]
                    cur_length_diff = (
                        parameter[cur_string + "_length"][1]
                        - parameter[cur_string + "_length"][0]
                    )
                    last_cur_length_diff = (
                        parameter[last_string + "_length"][1]
                        + parameter[last_string + "_generatrix_offset"][0]
                        - (
                            parameter[last_string + "_length"][1]
                            - parameter[last_string + "_length"][0]
                        )
                        / 2
                        - parameter[last_string + "_length"][0]
                    )
                    if last_cur_length_diff < 0:
                        last_cur_length_diff = -last_cur_length_diff
                        is_offset_negative = 1
                    last_angle = np.arctan(
                        2
                        * parameter[last_string + "_radius"][0]
                        * tmp_x_z_ratio
                        / last_cur_length_diff
                    )

                    tilt_length = np.sqrt(
                        last_cur_length_diff * last_cur_length_diff
                        + (
                            4
                            * parameter[last_string + "_radius"][0]
                            * tmp_x_z_ratio
                            * tmp_x_z_ratio
                            * parameter[last_string + "_radius"][0]
                        )
                    )
                    rotation_diff = (
                        (
                            -parameter[cur_string + "_rotation"][0]
                            + parameter[last_string + "_rotation"][0]
                        )
                        / 180
                        * np.pi
                    )

                    if is_offset_negative == 1:
                        cur_real_rotation = np.pi / 2 - rotation_diff - last_angle
                        cur_offset = (
                            -np.sin(cur_real_rotation) * tilt_length
                            + cur_length_diff / 2
                        )
                    elif is_offset_negative == 0:
                        cur_real_rotation = np.pi / 2 + rotation_diff - last_angle
                        cur_offset = (
                            np.sin(cur_real_rotation) * tilt_length
                            + cur_length_diff / 2
                        )
                    else:
                        cur_real_rotation = np.pi / 2 - rotation_diff - last_angle
                        cur_offset = (
                            -np.sin(cur_real_rotation) * tilt_length
                            + cur_length_diff / 2
                        )

                    parameter[cur_string + "_generatrix_offset"][0] = cur_offset
                    parameter[cur_string + "_radius"][1] = (
                        tilt_length * np.cos(cur_real_rotation) / 2
                    )
                    parameter[cur_string + "_thinkness"][0] = (
                        parameter[last_string + "_thinkness"][0]
                        * parameter[cur_string + "_radius"][1]
                        / (parameter[last_string + "_radius"][0] * tmp_x_z_ratio)
                    ) * randRange(1, 0.7, 0.8)[0]

                radius_pre = parameter["spout_%d_radius" % (i + 1)][0]
                parameter["spout_%d_radius" % (i + 2)][1] = radius_pre

            spout_height = (
                2
                * parameter["spout_1_radius"][1]
                / np.cos(parameter["spout_1_rotation"][0] / 180 * np.pi)
            )

            if body_type == "Semi_Spherical_Body":
                cur_horizontal_radius = body_parameter["horizontal_axis"][0]
                body_radius_up = body_parameter["vertical_axis"][0]
                body_radius_down = body_parameter["vertical_axis"][1]
                exist_angle_up = body_parameter["exist_angle"][0] / 180 * np.pi
                exist_angle_down = body_parameter["exist_angle"][1] / 180 * np.pi
                total_height = body_radius_up * np.sin(
                    exist_angle_up
                ) + body_radius_down * np.sin(exist_angle_down)
                height_diff = total_height / 2 - body_radius_down * np.sin(
                    exist_angle_down
                )

                spout_up_angle = np.arcsin(spout_height / 2 / body_radius_up)
                spout_down_angle = np.arcsin(spout_height / 2 / body_radius_down)
                offset_distance = cur_horizontal_radius * np.minimum(
                    np.cos(spout_up_angle), np.cos(spout_down_angle)
                )
                if body_radius_up * np.sin(
                    exist_angle_up
                ) > 1.5 * body_radius_down * np.sin(exist_angle_down):
                    parameter["position"][1] += height_diff
                    height_down = spout_height - height_diff
                    tmp_angle_down = np.arcsin(height_down / body_radius_down)
                    tmp_angle_up = np.arcsin(height_diff / body_radius_up)
                    offset_distance = cur_horizontal_radius * np.cos(tmp_angle_up)
                    if height_down > 0:
                        offset_distance = np.minimum(
                            offset_distance,
                            cur_horizontal_radius * np.cos(tmp_angle_down),
                        )

                parameter["position"][2] += offset_distance

            elif body_type == "Spherical_Cylindrical_Body":
                cur_horizontal_radius = body_parameter["horizontal_axis"][0]
                body_radius_up = body_parameter["vertical_axis"][0]
                height_of_bottom_part = body_parameter["bottom_size"][1]
                exist_angle_up = body_parameter["exist_angle"][0] / 180 * np.pi
                total_height = (
                    body_radius_up * np.sin(exist_angle_up) + height_of_bottom_part
                )
                height_diff = total_height / 2 - height_of_bottom_part

                spout_up_angle = np.arcsin(spout_height / (2 * body_radius_up))
                spout_down_angle = np.arctan(
                    (cur_horizontal_radius - body_parameter["bottom_size"][0])
                    / height_of_bottom_part
                )
                offset_distance = np.minimum(
                    cur_horizontal_radius * np.cos(spout_up_angle),
                    body_parameter["bottom_size"][0]
                    + (height_of_bottom_part - (spout_height / 2))
                    * np.tan(spout_down_angle),
                )
                if (
                    body_radius_up * np.sin(exist_angle_up)
                    > 1.5 * height_of_bottom_part
                ):
                    parameter["position"][1] += height_diff
                    height_down = spout_height - height_diff
                    tmp_angle_up = np.arcsin(height_diff / body_radius_up)
                    offset_distance = cur_horizontal_radius * np.cos(tmp_angle_up)
                    if height_down > 0:
                        offset_distance = np.minimum(
                            offset_distance,
                            body_parameter["bottom_size"][0]
                            + (height_of_bottom_part - height_down)
                            * np.tan(spout_down_angle),
                        )
                parameter["position"][2] += offset_distance

            elif body_type == "Multilevel_Body":
                cur_num = body_parameter["num_levels"][0]
                total_height = 0
                minimum_radius = body_parameter["level_1_top_radius"][0]
                for i in range(cur_num):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]
                    if (
                        minimum_radius
                        > body_parameter["level_" + str(i + 1) + "_top_radius"][0]
                    ):
                        minimum_radius = body_parameter[
                            "level_" + str(i + 1) + "_top_radius"
                        ][0]
                if cur_num > 1:
                    parameter["position"][1] += (
                        total_height
                        - (body_parameter["level_1_height"][0] / 2)
                        - (
                            body_parameter["level_" + str(cur_num) + "_height"][0]
                            * randRange(1, 0.8, 0.9)[0]
                        )
                    )
                    parameter["position"][2] += (
                        body_parameter["level_" + str(cur_num - 1) + "_top_radius"][0]
                        * randRange(1, 0.89, 0.92)[0]
                    )
                else:
                    parameter["position"][1] += 0
                    parameter["position"][2] += (
                        (
                            body_parameter["level_1_bottom_radius"][0]
                            + body_parameter["level_1_top_radius"][0]
                        )
                        / 2
                        * randRange(1, 0.89, 0.92)[0]
                    )

            parameter["position"][1] -= (
                -(parameter["spout_1_length"][0] + parameter["spout_1_length"][1])
                / 4
                * np.sin(parameter["spout_1_rotation"][0] / 180 * np.pi)
            )
            parameter["position"][2] -= (
                -(parameter["spout_1_length"][0] + parameter["spout_1_length"][1])
                / 4
                * np.cos(parameter["spout_1_rotation"][0] / 180 * np.pi)
            )

            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Curved_Spout":
            body_parameter = concepts[0]["parameters"]
            size_mul_1 = randRange(parameter["central_radius"].shape[0], 0.7, 1.3)
            size_mul_2 = randRange(parameter["exist_angle"].shape[0], 0.7, 1.1)
            size_mul_3 = randRange(parameter["torus_radius"].shape[0], 0.7, 1.3)
            parameter["central_radius"] *= size_mul_1
            parameter["exist_angle"] *= size_mul_2
            parameter["torus_radius"] *= size_mul_3

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]

            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            if body_type == "Semi_Spherical_Body":
                offset_distance = body_parameter["horizontal_axis"][0]
                parameter["position"][2] += offset_distance * randRange(1, 0.8, 0.9)[0]

            elif body_type == "Spherical_Cylindrical_Body":
                offset_distance = body_parameter["horizontal_axis"][0]
                parameter["position"][2] += offset_distance * randRange(1, 0.8, 0.9)[0]

            elif body_type == "Multilevel_Body":
                cur_num = body_parameter["num_levels"][0]
                total_height = 0
                minimum_radius = body_parameter["level_1_top_radius"][0]
                for i in range(cur_num):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]
                    if (
                        minimum_radius
                        > body_parameter["level_" + str(i + 1) + "_top_radius"][0]
                    ):
                        minimum_radius = body_parameter[
                            "level_" + str(i + 1) + "_top_radius"
                        ][0]
                if cur_num > 1:
                    parameter["position"][1] += (
                        total_height
                        - (body_parameter["level_1_height"][0] / 2)
                        - (body_parameter["level_" + str(cur_num) + "_height"][0])
                        * randRange(1, 0.8, 0.9)[0]
                    )
                    parameter["position"][2] += (
                        body_parameter["level_" + str(cur_num - 1) + "_top_radius"][0]
                        * randRange(1, 0.89, 0.92)[0]
                    )
                else:
                    parameter["position"][1] += 0
                    parameter["position"][2] += (
                        (
                            body_parameter["level_1_bottom_radius"][0]
                            + body_parameter["level_1_top_radius"][0]
                        )
                        / 2
                        * randRange(1, 0.89, 0.92)[0]
                    )

            for k, v in parameter.items():
                if isinstance(v, list):
                    parameter[k] = np.array(v)

            concept["parameters"] = {k: v.tolist() for k, v in parameter.items()}
            new_concepts.append(concept)

        elif template == "Triangular_Spout":
            body_parameter = concepts[0]["parameters"]
            size_mul_1 = randRange(parameter["top_outer_size"].shape[0], 0.8, 1.2)
            size_mul_2 = randRange(parameter["bottom_outer_size"].shape[0], 0.8, 1.2)
            size_mul_3 = randRange(parameter["top_inner_size"].shape[0], 0.8, 1.2)
            size_mul_4 = randRange(parameter["bottom_inner_size"].shape[0], 0.8, 1.2)
            size_mul_5 = randRange(parameter["height"].shape[0], 0.8, 1.2)

            top_outer_inner_diff = (
                parameter["top_outer_size"] - parameter["top_inner_size"]
            )
            bottom_outer_inner_diff = (
                parameter["bottom_outer_size"] - parameter["bottom_inner_size"]
            )

            parameter["top_outer_size"] *= size_mul_1
            parameter["bottom_outer_size"] *= size_mul_2

            top_outer_inner_diff *= size_mul_3
            bottom_outer_inner_diff *= size_mul_4

            parameter["top_inner_size"] = (
                parameter["top_outer_size"] - top_outer_inner_diff
            )
            parameter["bottom_inner_size"] = (
                parameter["bottom_outer_size"] - bottom_outer_inner_diff
            )
            parameter["height"] *= size_mul_5

            parameter["position"][0] = body_parameter["position"][0]
            parameter["position"][1] = body_parameter["position"][1]
            parameter["position"][2] = body_parameter["position"][2]

            parameter["rotation"][0] = body_parameter["rotation"][0]
            parameter["rotation"][1] = body_parameter["rotation"][1]
            parameter["rotation"][2] = body_parameter["rotation"][2]

            parameter["spout_rotation"][0] *= randRange(1, 0.7, 1.3)[0]

            if body_type == "Semi_Spherical_Body":
                offset_distance = body_parameter["horizontal_axis"][0]
                parameter["position"][2] += offset_distance * randRange(1, 0.8, 0.9)[0]

            elif body_type == "Spherical_Cylindrical_Body":
                offset_distance = body_parameter["horizontal_axis"][0]
                parameter["position"][2] += offset_distance * randRange(1, 0.8, 0.9)[0]

            elif body_type == "Multilevel_Body":
                cur_num = body_parameter["num_levels"][0]
                total_height = 0
                minimum_radius = body_parameter["level_1_top_radius"][0]
                for i in range(cur_num):
                    total_height += body_parameter["level_" + str(i + 1) + "_height"][0]
                    if (
                        minimum_radius
                        > body_parameter["level_" + str(i + 1) + "_top_radius"][0]
                    ):
                        minimum_radius = body_parameter[
                            "level_" + str(i + 1) + "_top_radius"
                        ][0]
                if cur_num > 1:
                    parameter["position"][1] += (
                        total_height
                        - (body_parameter["level_1_height"][0] / 2)
                        - (body_parameter["level_" + str(cur_num) + "_height"][0])
                        * randRange(1, 0.8, 0.9)[0]
                    )
                    parameter["position"][2] += (
                        body_parameter["level_" + str(cur_num - 1) + "_top_radius"][0]
                        * randRange(1, 0.89, 0.92)[0]
                    )
                else:
                    parameter["position"][1] += 0
                    parameter["position"][2] += (
                        (
                            body_parameter["level_1_bottom_radius"][0]
                            + body_parameter["level_1_top_radius"][0]
                        )
                        / 2
                        * randRange(1, 0.89, 0.92)[0]
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
        kettle_type = get_kettle_type()
        existing_concept_templates = concept_template_existence(kettle_type)
        default_params = json.load(open("default_params.json", "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = jitter_parameters(concepts, kettle_type)

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
            kettle_type = get_kettle_type()
            existing_concept_templates = concept_template_existence(kettle_type)
            default_params_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/default_params.json"
            )
            default_params = json.load(open(default_params_path, "r"))
            concepts = []
            for template in existing_concept_templates:
                concepts.append(
                    {"template": template, "parameters": default_params[template]}
                )
            new_concepts = jitter_parameters(concepts, kettle_type)
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
