import os
import pickle
import sys
from copy import deepcopy
from typing import Any, Dict

import imageio
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

from .concept_template import *
from code.utils import COLOR20


from .gif_utils import (
    GenGifError,
    analyze_actionable_parts_from_conceptualization,
    apply_transformation,
    assemble_concepts,
    assemble_concepts_instance,
    debug_draw_move_axis,
    find_concept_idx_by_idx,
    find_crucial_parts,
    setup_camera_to_view_meshes,
)


class Trifold_Handle_GraspPose:
    def __init__(
        self,
        vertical_length,
        vertical_thickness,
        vertical_position,
        vertical_rotation,
        position,
        rotation,
    ):
        self.vertical_length = vertical_length
        self.vertical_thickness = vertical_thickness
        self.vertical_position = vertical_position
        self.vertical_rotation = vertical_rotation
        self.position = position
        self.rotation = rotation

    def get_grasp_pose(self, omega):
        """
        omega: -pi/3 ~ pi/3
        """
        R = Rot.from_euler("Y", omega, degrees=False).as_matrix()

        T = np.array([[0, self.vertical_length / 2 * np.sin(omega * 0.8), 0]])
        T = apply_transformation(
            T, position=self.vertical_position, rotation=self.vertical_rotation
        )
        T = apply_transformation(T, position=self.position, rotation=self.rotation)
        T = T[0]
        return R, T


class Curved_Handle_GraspPose:
    def __init__(self, radius, central_angle, position, rotation):
        self.radius = radius
        self.central_angle = central_angle
        self.position = position
        self.rotation = rotation

    def get_grasp_pose(self, omega):
        """
        omega: -pi/3 ~ pi/3
        """
        R = Rot.from_euler(
            "xy", [self.central_angle / 2 * np.sin(omega * 0.6), omega], degrees=False
        ).as_matrix()

        T = np.array(
            [
                [
                    self.radius
                    * np.cos(self.central_angle / 2 * (1 + np.sin(omega * 0.6))),
                    0,
                    self.radius
                    * np.sin(self.central_angle / 2 * (1 + np.sin(omega * 0.6))),
                ]
            ]
        )
        T = apply_transformation(T, position=[0, 0, 0], rotation=[0, 0, np.pi / 2])
        T = apply_transformation(T, position=self.position, rotation=self.rotation)
        T = T[0]
        return R, T


def generate_grasp_knowledge_gif(
    concepts: Dict[str, Any] = None,
    output_path: str = "./output_gifs/animation.gif",
    camera_params_path=None,
    num_frames: int = 40,
    width: int = 640,
    height: int = 480,
    duration: float = 2000 / 30,
    debug: bool = False,
):
    """
    Generate an animated GIF showing the gripper manipulating movable multi-concept 3D objects, 
    where each part moves along its specified axis (translation or rotation).
    
    The gripper's manipulation parameters vary from -1 to 1, generating a variety of feasible grasp poses.
    Supports "back-and-forth" motion for each part: starting from the initial position, reaching maximum displacement, then returning.

    Parameters:
    ----------
    concepts : Dict[str, Any]
        A dictionary of concept definitions. Each concept includes a template name and corresponding parameters.
    output_path : str
        Output path for the GIF file (including filename). Example: "./output/animation.gif".
    camera_params_path : str or None
        Optional path to load Open3D pinhole camera parameters. If None, the view is set automatically.
    num_frames : int
        Total number of frames in the animation (for the full back-and-forth movement).
        PLEASE NOTE THAT: num_frames for grasp is not the true number of frames, it should mutiply by 3 or more 
        More frames result in smoother animation but longer generation time.
    duration : float
        Display time per frame in milliseconds, controlling the playback speed. Longer durations may cause stuttering playback.
    
    Main Workflow:
    --------------
    1. Create an Open3D visualization window and load all mesh parts.
    2. If debug mode is enabled, draw motion axes and centers for verification.
    3. Set up the camera view (from file or auto-calculated).
    4. Compute the gripper's motion parameters and generate its initial pose and per-frame transformations.
    5. For each concept, compute its motion axis, center, amplitude, and determine per-frame transformation.
    6. Apply gripper and object part movements frame by frame (forward then backward), update geometry, and capture images.
    7. Save all captured frames as an animated GIF using imageio.
    """
    template_names = [c["template"] for c in concepts]
    if not find_crucial_parts(template_names, part_names=["handle"]):
        raise GenGifError("no movable part, please check concepts list")

    vertices_list = {}
    faces_list = {}
    total_num_vertices = {}

    templates = []

    for c in concepts:
        module = eval(c["template"])
        component = module(**c["parameters"])

        vertices_list.setdefault(component.semantic, [])
        faces_list.setdefault(component.semantic, [])
        total_num_vertices.setdefault(component.semantic, 0)

        vertices_list[component.semantic].append(component.vertices)
        faces_list[component.semantic].append(
            component.faces + total_num_vertices[component.semantic]
        )
        total_num_vertices[component.semantic] += len(component.vertices)

        if c["template"] in ["Trifold_Handle"]:
            horizontal_rotation = [
                x / 180 * np.pi for x in c["parameters"]["horizontal_rotation"]
            ]
            delta_y = (
                c["parameters"]["horizontal_separation"][0]
                - c["parameters"]["horizontal_length"][0]
                * np.sin(horizontal_rotation[0])
                + c["parameters"]["horizontal_length"][1]
                * np.sin(horizontal_rotation[1])
            )
            delta_z = (
                c["parameters"]["mounting_offset"][0]
                - c["parameters"]["horizontal_length"][1]
                * np.cos(horizontal_rotation[1])
                + c["parameters"]["horizontal_length"][0]
                * np.cos(horizontal_rotation[0])
            )
            vertical_length = (
                np.sqrt(delta_y * delta_y + delta_z * delta_z)
                + c["parameters"]["horizontal_thickness"][1]
            )
            vertical_rotation = np.arctan(delta_z / delta_y)
            vertical_y_offset = (
                -c["parameters"]["horizontal_length"][0]
                * np.sin(horizontal_rotation[0])
                - c["parameters"]["horizontal_length"][1]
                * np.sin(horizontal_rotation[1])
            ) / 2
            vertical_z_offset = (
                c["parameters"]["horizontal_length"][1] * np.cos(horizontal_rotation[1])
                + c["parameters"]["mounting_offset"][0]
                + c["parameters"]["horizontal_length"][0]
                * np.cos(horizontal_rotation[0])
            ) / 2
            grasp_pose = Trifold_Handle_GraspPose(
                vertical_length,
                c["parameters"]["vertical_thickness"][0],
                [
                    0,
                    vertical_y_offset,
                    vertical_z_offset + c["parameters"]["vertical_thickness"][1] / 2,
                ],
                [vertical_rotation, 0, 0],
                c["parameters"]["position"],
                [x / 180 * np.pi for x in c["parameters"]["position"]],
            )

        elif c["template"] in ["Curved_Handle"]:
            grasp_pose = Curved_Handle_GraspPose(
                c["parameters"]["radius"][0],
                c["parameters"]["central_angle"][0] / 180 * np.pi,
                c["parameters"]["position"],
                [x / 180 * np.pi for x in c["parameters"]["rotation"]],
            )

        templates.append(c["template"])

    final_vertices = {}
    final_faces = {}
    part_meshes = {}
    color_idx = 0
    for k in vertices_list.keys():
        final_vertices[k] = np.concatenate(vertices_list[k])
        aabb_min = final_vertices[k].min(axis=0)
        aabb_max = final_vertices[k].max(axis=0)
        if np.all(aabb_min == aabb_max):
            continue
        final_faces[k] = np.concatenate(faces_list[k])
        part_meshes[k] = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(final_vertices[k]),
            o3d.utility.Vector3iVector(final_faces[k]),
        )
        part_meshes[k].compute_vertex_normals()
        part_meshes[k].paint_uniform_color(np.array(COLOR20[color_idx]))
        color_idx += 1

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=width, height=height)

    images = []
    if camera_params_path is not None:
        camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
    else:
        camera_params = setup_camera_to_view_meshes(
            part_meshes, width, height, front_offset_ratio=1.5
        )

    angle_per_frame = np.pi / num_frames
    omega = -np.pi / 3
    phase = 0

    current_file_path = os.path.abspath(__file__)
    gripper_stl_path = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))}/assets/standard_gripper.stl"
    gripper_mesh = o3d.io.read_triangle_mesh(gripper_stl_path)

    gripper_mesh.rotate(
        Rot.from_euler("Y", np.pi / 2, degrees=False).as_matrix(), center=[0, 0, 0]
    )
    standard_gripper = gripper_mesh
    standard_gripper.scale(0.2, center=[0, 0, 0])
    standard_gripper.compute_vertex_normals()
    standard_gripper.vertices = o3d.utility.Vector3dVector(
        apply_transformation(
            np.array(standard_gripper.vertices),
            position=[0, 0, 0.16],
            rotation=[0, np.pi / 2, np.pi / 2],
            rotation_order="YZX",
        )
    )

    for i in range(num_frames):
        R, T = grasp_pose.get_grasp_pose(omega)
        if phase == 0:
            omega += angle_per_frame
            if omega >= np.pi / 3:
                phase = 1
        else:
            omega -= angle_per_frame

        gripper = deepcopy(standard_gripper)
        gripper.rotate(R, center=[0, 0, 0])
        gripper.translate(T)

        vis.clear_geometries()
        for k, mesh in part_meshes.items():
            vis.add_geometry(mesh)
        vis.add_geometry(gripper)
        # vis.add_geometry(sphere)
        ctr = vis.get_view_control()

        ctr.convert_from_pinhole_camera_parameters(camera_params)

        vis.poll_events()
        vis.update_renderer()

        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        images.append((img * 255).astype(np.uint8))

    angle_per_frame = np.pi / 180 * 120 / num_frames
    ct_clockwise_rot_mat = Rot.from_euler(
        "Y", [-angle_per_frame], degrees=False
    ).as_matrix()[0]
    clockwise_rot_mat = Rot.from_euler(
        "Y", [angle_per_frame], degrees=False
    ).as_matrix()[0]
    total_rotation = 0
    phase = 0

    for i in range(num_frames):
        if phase == 0:
            gripper.rotate(ct_clockwise_rot_mat, center=[0, 0, 0])
            part_meshes["Body"].rotate(ct_clockwise_rot_mat, center=[0, 0, 0])
            part_meshes["Handle"].rotate(ct_clockwise_rot_mat, center=[0, 0, 0])
            total_rotation -= angle_per_frame
            if total_rotation <= -np.pi / 6:
                phase += 1
        elif phase == 1:
            gripper.rotate(clockwise_rot_mat, center=[0, 0, 0])
            part_meshes["Body"].rotate(clockwise_rot_mat, center=[0, 0, 0])
            part_meshes["Handle"].rotate(clockwise_rot_mat, center=[0, 0, 0])
            total_rotation += angle_per_frame
            if total_rotation >= np.pi / 6:
                phase += 1
        else:
            gripper.rotate(ct_clockwise_rot_mat, center=[0, 0, 0])
            part_meshes["Body"].rotate(ct_clockwise_rot_mat, center=[0, 0, 0])
            part_meshes["Handle"].rotate(ct_clockwise_rot_mat, center=[0, 0, 0])
            total_rotation -= angle_per_frame

        vis.clear_geometries()
        for k, mesh in part_meshes.items():
            vis.add_geometry(mesh)
        vis.add_geometry(gripper)
        # vis.add_geometry(sphere)
        ctr = vis.get_view_control()

        ctr.convert_from_pinhole_camera_parameters(camera_params)

        vis.poll_events()
        vis.update_renderer()

        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        images.append((img * 255).astype(np.uint8))

    gif_path = os.path.dirname(output_path)
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    vis.destroy_window()
    


def generate_instance_knowledge_gif(
    concepts: Dict[str, Any] = None,
    output_path: str = "./output_gifs/animation.gif",
    camera_params_path=None,
    num_frames: int = 40,
    width: int = 640,
    height: int = 480,
    duration: float = 2000 / 30,
):
    """
    Generate a rotating animation of the assembled 3D mesh object and save it as a GIF file.
    The object rotates continuously around the Y-axis (from its initial orientation), 
    making a full 360-degree rotation over the course of the animation.

    Parameters:
    ----------
    concepts : Dict[str, Any]
        A dictionary of concept definitions. Each concept should include a template name and parameters.
    output_path : str
        Output path for the GIF file (including filename). e.g., "./output/animation.gif".
    camera_params_path : str or None, optional
        Path to load Open3D pinhole camera parameters. If None, the view is set automatically.
    num_frames : int
        Total number of frames in the animation (for the full back-and-forth movement). More frames result in smoother animation but longer generation time.
    width : int
        Width of the rendered image window.
    height : int
        Height of the rendered image window.
    duration : float
        Display time per frame in milliseconds, controlling the playback speed. Longer durations may cause stuttering playback.

    Main Workflow:
    --------------
    1. Create an Open3D visualizer window with specified resolution.
    2. Assemble and load all mesh parts from the given concepts.
    3. Set up the camera view using provided or automatically computed parameters.
    4. Rotate the object by small increments per frame around the Y-axis.
    5. Save the collected images as an animated GIF using imageio.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=width, height=height)

    part_meshes = assemble_concepts_instance(concepts)
    for mesh in part_meshes.values():
        vis.add_geometry(mesh)

    if camera_params_path is not None:
        camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
    else:
        camera_params = setup_camera_to_view_meshes(
            part_meshes, width, height, front_offset_ratio=1.5
        )
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    angle_per_frame = 360 / num_frames 
    R = Rot.from_euler("y", angles=angle_per_frame, degrees=True).as_matrix()[:3, :3]
    images = []
    for i in range(num_frames):
        for mesh in part_meshes.values():
            mesh.rotate(R, center=[0, 0, 0])
            vis.update_geometry(mesh)

        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        images.append((img * 255).astype(np.uint8))

    gif_path = os.path.dirname(output_path)
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    vis.destroy_window()


def generate_affordance_knowledge_gif(
    concepts: Dict[str, Any] = None,
    output_path: str = "./output_gifs/animation.gif",
    camera_params_path=None,
    num_frames: int = 40,
    width: int = 640,
    height: int = 480,
    duration: float = 2000 / 30,
    debug: bool = False,
):
    """
    Generate an animated GIF showing movable parts of a multi-concept 3D object, with each part moving along its own axis (translation or rotation).
    Supports "back-and-forth" motion for each part: starting from the initial position, reaching maximum displacement, then returning.

    Parameters:
    ----------
    concepts : Dict[str, Any]
        A dictionary of concept definitions. Each concept should include a template name and parameters.
    output_path : str
        Output path for the GIF file (including filename). e.g., "./output/animation.gif".
    camera_params_path : str or None, optional
        Path to load Open3D pinhole camera parameters. If None, the view is set automatically.
    num_frames : int
        Total number of frames in the animation (for the full back-and-forth movement). More frames result in smoother animation but longer generation time.
    width : int
        Width of the rendered image window.
    height : int
        Height of the rendered image window.
    duration : float
        Display time per frame in milliseconds, controlling the playback speed. Longer durations may cause stuttering playback.
    debug : bool
        Whether to enable debug mode. If True, visual aids (e.g., motion axes) will be drawn for verification.

    Main Workflow:
    --------------
    1. Create an Open3D visualization window and load all mesh parts.
    2. If debug mode is enabled, draw motion axes for reference.
    3. Set up the camera view (from file or auto-calculated).
    4. For each concept, compute its motion axis, center, and amplitude; calculate per-frame transformations.
    5. For each frame, apply forward or backward motion, update geometry, and capture screen.
    6. Save all captured frames as an animated GIF using imageio.
    """
    template_names = [c["template"] for c in concepts]
    if not find_crucial_parts(template_names, part_names=["cylinder"]):
        raise GenGifError("no movable part, please check concepts list")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=width, height=height)

    part_meshes = assemble_concepts(concepts)
    for mesh in part_meshes.values():
        vis.add_geometry(mesh)

    actionable_parts_info = analyze_actionable_parts_from_conceptualization(concepts)
    if actionable_parts_info is None:
        raise GenGifError("no movable part, please check concepts list")
    (
        template_names,
        move_types,
        move_centers,
        move_axises,
        move_state_and_limits,
        concept_idxes,
    ) = (
        actionable_parts_info["template_names"],
        actionable_parts_info["move_types"],
        actionable_parts_info["move_centers"],
        actionable_parts_info["move_axises"],
        actionable_parts_info["move_state_and_limits"],
        actionable_parts_info["concept_idxes"],
    )

    # visualize move_axises and move_centers
    if debug:
        debug_draw_move_axis(vis, move_centers, move_axises)

    if camera_params_path is not None:
        camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
    else:
        camera_params = setup_camera_to_view_meshes(
            part_meshes, width, height, front_offset_ratio=1.5
        )
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    images = []
    move_steps = []
    turn_times = []
    for move_axis, move_state_and_limit in zip(move_axises, move_state_and_limits):
        min_dist, max_dist, move_dist = move_state_and_limit
        total_dist_range = (max_dist - move_dist) + (max_dist - min_dist)
        turn_times.append(
            int(np.ceil(num_frames * (max_dist - move_dist) / total_dist_range))
        )
        dist_per_frame = total_dist_range / num_frames
        move_axis = move_axis / np.linalg.norm(move_axis)
        move_steps.append(dist_per_frame * move_axis)

    for i in range(num_frames):
        for semantic, mesh in part_meshes.items():
            global_idx = int(semantic.split("_")[-1])
            if "Cylinder" not in semantic:
                continue
            find_idx = find_concept_idx_by_idx(
                concepts, concept_idxes[0], "Single_Cylinder"
            )
            if global_idx != find_idx:
                continue
            # Direction Control: First Half Forward,Second Half Backward
            direction = 1 if i < turn_times[0] else -1
            trans = move_steps[0]
            mesh.translate(direction * trans, relative=True)
            vis.update_geometry(mesh)

        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        images.append((img * 255).astype(np.uint8))

    gif_path = os.path.dirname(output_path)
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    vis.destroy_window()
    