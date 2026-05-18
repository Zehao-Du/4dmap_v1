import os
import pickle
import sys
from copy import deepcopy
from typing import Any, Dict

import imageio
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot


from ...base_knowledge import get_manip_knowledge
from .gif_utils import (
    GenGifError,
    assemble_concepts,
    assemble_concepts_instance,
    find_crucial_parts,
    setup_camera_to_view_meshes,
)


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
    template_names = [cc["template"] for cc in concepts]
    if not find_crucial_parts(template_names, part_names=["switch"]):
        raise GenGifError("no movable part, please check concepts list")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=width, height=height)

    part_meshes = assemble_concepts(concepts)
    for mesh in part_meshes.values():
        vis.add_geometry(mesh)

    current_file_path = os.path.abspath(__file__)
    gripper_stl_path = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))}/assets/standard_gripper.stl"
    gripper_mesh = o3d.io.read_triangle_mesh(gripper_stl_path)
    manipulation_knowledge = {"state": "random-middle", "primact_type": "pulling"}
    grasp_pose, force_direction, gripper_width_ratio, manip_params_size = (
        get_manip_knowledge(concepts, "Switch", manipulation_knowledge)
    )
    gripper_mesh.scale(0.2, center=[0, 0, 0])
    gripper_mesh.paint_uniform_color([230 / 255, 25 / 255, 75 / 255])
    gripper_mesh.compute_vertex_normals()
    origin_gripper_mesh = deepcopy(gripper_mesh)
    gripper_mesh = gripper_mesh.transform(grasp_pose)
    vis.add_geometry(gripper_mesh)

    if camera_params_path is not None:
        camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
    else:
        camera_params = setup_camera_to_view_meshes(
            part_meshes, width, height, front_offset_ratio=1.5
        )
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    init_camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    # Adjust manipulation_params from 0 -> 1 -> -1 -> 0 to control gripper movement
    gripper_frames_num = 2 * num_frames
    gripper_params = []
    cur_gripper_param = 0
    per_frame_delta = 4 / gripper_frames_num
    for i in range(gripper_frames_num):
        if i < gripper_frames_num // 4:
            gripper_params.append([cur_gripper_param] * (manip_params_size - 1))
            cur_gripper_param += per_frame_delta
        elif i < 3 * gripper_frames_num // 4:
            gripper_params.append([cur_gripper_param] * (manip_params_size - 1))
            cur_gripper_param -= per_frame_delta
        else:
            gripper_params.append([cur_gripper_param] * (manip_params_size - 1))
            cur_gripper_param += per_frame_delta

    for i in range(gripper_frames_num):
        gripper_params[i] += [0]

    images = []
    for concept in concepts:
        if "switch" in concept["template"].lower():
            switch_concept = concept
            break
    assert switch_concept is not None, "No switch concept found in the list."

    switch_rotations = switch_concept["parameters"]["switch_rotation"][0:1]

    if switch_concept["template"] == "Lever_Switch":
        min_angle, max_angle = -20, 20  # degree
    else:
        min_angle, max_angle = -10, 10  # degree
    for switch_rotation in switch_rotations:
        min_angle = min(min_angle, switch_rotation)
        max_angle = max(max_angle, switch_rotation)
        if switch_rotation < 0:
            total_angle_range = max_angle - switch_rotation
        else:
            total_angle_range = min_angle - switch_rotation
        angle_per_frame = total_angle_range / num_frames


    for i in range(3 * num_frames):
        if i < 2 * num_frames:
            vis.clear_geometries()
            for mesh in part_meshes.values():
                vis.add_geometry(mesh)
            grasp_pose, force_direction, gripper_width_ratio, _ = get_manip_knowledge(
                concepts,
                "Switch",
                manipulation_knowledge,
                gripper_params[i][:manip_params_size],
            )
            gripper_mesh = deepcopy(origin_gripper_mesh).transform(grasp_pose)
            vis.add_geometry(gripper_mesh)
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(init_camera_params)

        else:
            vis.clear_geometries()
            switch_concept["parameters"]["switch_rotation"][0] += (
                angle_per_frame  # * direction
            )

            part_meshes = assemble_concepts(concepts)
            for mesh in part_meshes.values():
                vis.add_geometry(mesh)

            grasp_pose, force_direction, gripper_width_ratio, _ = get_manip_knowledge(
                concepts, "Switch", manipulation_knowledge
            )
            gripper_mesh = deepcopy(origin_gripper_mesh).transform(grasp_pose)
            vis.add_geometry(gripper_mesh)
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(init_camera_params)

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
    template_names = [cc["template"] for cc in concepts]
    if not find_crucial_parts(template_names, part_names=["switch"]):
        raise GenGifError("no movable part, please check concepts list")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=width, height=height)

    part_meshes = assemble_concepts(concepts)
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
    init_camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    images = []
    turn_times = []
    for concept in concepts:
        if "switch" in concept["template"].lower():
            switch_concept = concept
            break
    assert switch_concept is not None, "No switch concept found in the list."

    switch_rotations = switch_concept["parameters"]["switch_rotation"][0:1]

    if switch_concept["template"] == "Lever_Switch":
        min_angle, max_angle = -20, 20  # degree
    else:
        min_angle, max_angle = -10, 10  # degree
    for switch_rotation in switch_rotations:
        min_angle = min(min_angle, switch_rotation)
        max_angle = max(max_angle, switch_rotation)
        total_angle_range = (max_angle - switch_rotation) + (max_angle - min_angle)
        turn_times.append(
            int(np.ceil(num_frames * (max_angle - switch_rotation) / total_angle_range))
        )
        angle_per_frame = total_angle_range / num_frames

    for i in range(num_frames):
        vis.clear_geometries()
        # Direction Control: First Half Forward,Second Half Backward
        for idx in range(len(turn_times)):
            direction = 1 if i < turn_times[idx] else -1
            switch_concept["parameters"]["switch_rotation"][0] += (
                direction * angle_per_frame
            )

        part_meshes = assemble_concepts(concepts)
        for mesh in part_meshes.values():
            vis.add_geometry(mesh)
        if debug:
            coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=3, origin=[0, 0, 0]
            )
            vis.add_geometry(coordinates)

        # Reset the camera view after executing vis.clear_geometries()
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(init_camera_params)

        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        images.append((img * 255).astype(np.uint8))

    gif_path = os.path.dirname(output_path)
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    vis.destroy_window()