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
    debug_draw_move_axis,
    find_crucial_parts,
    find_text_in_list,
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
    if not find_crucial_parts(template_names, part_names=["press", "spray"]):
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
    press_concept = None
    spray_concept = None

    for concept in concepts:
        if "press" in concept["template"].lower():
            press_concept = concept
        if "spray" in concept["template"].lower():
            spray_concept = concept
    assert press_concept is not None or spray_concept is not None, (
        "No standard nozzle found in the list"
    )


    if press_concept is not None:
        press_translation = press_concept["parameters"]["level_1_size"][1]
        min_translation, max_translation = press_translation / 2, press_translation
        total_translation_range = max_translation - min_translation
        turn_times.append(
            int(np.ceil(num_frames * press_translation / 2 / total_translation_range))
        )
        translation_per_frame = total_translation_range / num_frames
        rotation_per_frame = 360 / num_frames

        for i in range(num_frames):
            vis.clear_geometries()
            # Direction Control: First Half Backward, Second Half Forward
            direction = -1 if i <= num_frames // 2 else 1
            press_concept["parameters"]["position"][1] += (
                direction * translation_per_frame
            )
            press_concept["parameters"]["rotation"][1] += rotation_per_frame

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

    elif spray_concept is not None:
        spray_rotation = spray_concept["parameters"]["handle_rotation"][0]
        spray_rotation = max(spray_rotation, 0.01)  # ensure non-zero rotation
        min_angle, max_angle = spray_rotation / 2, spray_rotation
        total_angle_range = max_angle - min_angle
        turn_times.append(int(np.ceil(num_frames * spray_rotation / total_angle_range)))
        angle_per_frame = total_angle_range / num_frames
        rotation_per_frame = 360 / num_frames

        for i in range(num_frames):
            vis.clear_geometries()
            # Direction Control: First Half Backward, Second Half Forward
            direction = -1 if i <= num_frames // 2 else 1
            spray_concept["parameters"]["handle_rotation"][0] += (
                direction * angle_per_frame
            )
            spray_concept["parameters"]["rotation"][1] += rotation_per_frame

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
