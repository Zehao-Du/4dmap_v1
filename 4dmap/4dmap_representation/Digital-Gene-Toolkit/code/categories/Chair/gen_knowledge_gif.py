import json
import os
import pickle
import sys
from typing import Any, Dict

import imageio
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot


from .gif_utils import (
    assemble_concepts_instance,
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
    
