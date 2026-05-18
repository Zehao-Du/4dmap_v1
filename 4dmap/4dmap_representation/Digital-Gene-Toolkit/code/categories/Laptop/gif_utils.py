import random

import numpy as np
import open3d as o3d

from .concept_template import *
from code.utils import COLOR20


class GenGifError(Exception):
    """Custom exception class for invalid concept input with invalid parameters."""
    pass


def find_crucial_parts(template_names, part_names):
    """
    Check if any of the part names is contained (case-insensitive) in any template name.
    
    Args:
        template_names (list of str): List of template names to search within.
        part_names (list of str): List of part names to search for. e.g., ["lid", "handle"].
        
    Returns:
        bool: True if any part name is found in any template name, False otherwise.
    """
    for template_name in template_names:
        for part_name in part_names:
            if part_name.lower() in template_name.lower():
                return True
    return False


def find_text_in_list(text, lst):
    """Case-insensitive text matching. Returns the index, or -1 if not found."""
    for idx, item in enumerate(lst):
        if text.lower() in item.lower():
            return idx
    return -1


def assemble_concepts(concepts):
    """
    Assemble 3D mesh parts from a list of concepts.

    Args:
        concepts (list of dict): List of concept definitions.
            Each dict should have:
                - 'template' (str): Name of the template class/module.
                - 'parameters' (dict): Parameters for creating the component.

    Returns:
        dict: Keys are semantic labels, values are assembled TriangleMesh objects.
    """
    vertices_list = {}
    faces_list = {}
    total_num_vertices = {}
    part_meshes = {}
    for c in concepts:
        module_name = c["template"]

        try:
            module = eval(module_name)
            component = module(**c["parameters"])

            vertices_list.setdefault(component.semantic, [])
            faces_list.setdefault(component.semantic, [])
            total_num_vertices.setdefault(component.semantic, 0)

            vertices_list[component.semantic].append(component.vertices)
            faces_list[component.semantic].append(
                component.faces + total_num_vertices[component.semantic]
            )
            total_num_vertices[component.semantic] += len(component.vertices)
        except Exception as e:
            print(f"Failed to load or instantiate template '{module_name}': {e}")

    color_idx = 0
    for semantic, vertices in vertices_list.items():
        final_vertices = np.concatenate(vertices)
        # compute AABB for mesh and check if it's degenerate
        aabb_min = final_vertices.min(axis=0)
        aabb_max = final_vertices.max(axis=0)
        if np.all(aabb_min == aabb_max):
            continue
        final_faces = np.concatenate(faces_list[semantic])
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(final_vertices),
            o3d.utility.Vector3iVector(final_faces),
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array(COLOR20[color_idx % len(COLOR20)]))
        part_meshes[semantic] = mesh
        color_idx += 1

    return part_meshes


def assemble_concepts_instance(concepts):
    """Similar to assemble_concepts, this function assembles the object mesh based on the concept list. However, coloring is applied at the instance level(e.g. tophandle and sidehandle), rather than the semantic level."""
    vertices_list = {}
    faces_list = {}
    total_num_vertices = {}
    part_meshes = {}
    for c in concepts:
        module_name = c["template"]

        try:
            module = eval(module_name)
            component = module(**c["parameters"])

            if "connector" in module_name.lower():
                num_connector = int(component.number_of_connector[0])
                per_connector_vertices = len(component.vertices) // num_connector
                per_connector_faces = len(component.faces) // num_connector
                assert per_connector_vertices * num_connector == len(
                    component.vertices
                ), (
                    f"The number of vertices does not match for each button: {per_connector_vertices} * {num_connector} != {len(component.vertices)}"
                )
                for connector_idx in range(num_connector):
                    instance_name = f"{component.semantic}_{connector_idx}"
                    vertices_list.setdefault(instance_name, [])
                    faces_list.setdefault(instance_name, [])
                    total_num_vertices.setdefault(instance_name, 0)

                    vertices_list[instance_name].append(
                        component.vertices[
                            connector_idx * per_connector_vertices : (connector_idx + 1)
                            * per_connector_vertices
                        ]
                    )
                    faces_list[instance_name].append(
                        component.faces[
                            connector_idx * per_connector_faces : (connector_idx + 1)
                            * per_connector_faces
                        ]
                        - connector_idx * per_connector_vertices
                    )
                    total_num_vertices[instance_name] += (
                        connector_idx * per_connector_vertices
                    )
            else:
                vertices_list.setdefault(component.semantic, [])
                faces_list.setdefault(component.semantic, [])
                total_num_vertices.setdefault(component.semantic, 0)

                vertices_list[component.semantic].append(component.vertices)
                faces_list[component.semantic].append(
                    component.faces + total_num_vertices[component.semantic]
                )
                total_num_vertices[component.semantic] += len(component.vertices)
        except Exception as e:
            print(f"Failed to load or instantiate template '{module_name}': {e}")

    color_idx = 0
    for semantic, vertices in vertices_list.items():
        final_vertices = np.concatenate(vertices)
        # compute AABB for mesh and check if it's degenerate
        aabb_min = final_vertices.min(axis=0)
        aabb_max = final_vertices.max(axis=0)
        if np.all(aabb_min == aabb_max):
            continue
        final_faces = np.concatenate(faces_list[semantic])
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(final_vertices),
            o3d.utility.Vector3iVector(final_faces),
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array(COLOR20[color_idx % len(COLOR20)]))
        part_meshes[semantic] = mesh
        color_idx += 1

    return part_meshes


def compute_lookat_open3d(cam_pos, target, up):
    """
    Compute the LookAt extrinsic matrix (camera-to-world transformation).

    :param cam_pos: Camera position (numpy array)
    :param target: Target point to look at (numpy array)
    :param up: Up direction vector (numpy array)
    :return: Extrinsic matrix (4x4 numpy array)
    """
    up = -up / np.linalg.norm(up)
    z = cam_pos - target
    z = -z

    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    t = -R.T @ cam_pos
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = t
    return extrinsic


def setup_camera_to_view_meshes(
    part_meshes, width=640, height=480, front_offset_ratio=1.5
):
    """
    Automatically calculate camera parameters so that the object is centered in view,
    with a perspective from slightly above and to the front-right or front-left.

    :param part_meshes: Dictionary of o3d.geometry.TriangleMesh objects.
    :param width: Width of the rendered image.
    :param height: Height of the rendered image.
    :param front_offset_ratio: Ratio of camera distance to the object's size.
    :return: open3d.camera.PinholeCameraParameters
    """
    # Merge all meshes to compute overall AABB (bounding box)
    all_mesh = o3d.geometry.TriangleMesh()
    for mesh in part_meshes.values():
        all_mesh += mesh

    bbox = all_mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    # center = np.array([0, 0, 0])  # Uncomment to force camera to look at origin
    extent = bbox.get_extent()
    diagonal_length = np.linalg.norm(extent)

    # Set camera distance based on object size
    distance = diagonal_length * front_offset_ratio

    # Define viewing direction (slightly to the right/left and above front)
    # x-left, y-down, z-backward
    uniform0_5 = random.uniform(0.5, 1)
    mul_or_div = random.choice([1, -1])
    if mul_or_div == 1:
        uniform0_5 = 1 / uniform0_5
    view_x = uniform0_5 * random.choice([-1, 1])
    view_y = 1
    view_x = view_x / np.sqrt(view_x**2 + view_y**2) * np.sqrt(2)
    view_y = view_y / np.sqrt(view_x**2 + view_y**2) * np.sqrt(2)

    # Normalize view direction
    view_dir = np.array([view_x, view_y, 1])
    view_dir = view_dir / np.linalg.norm(view_dir)
    camera_position = center + view_dir * distance

    up_vector = np.array([0.0, 1.0, 0.0])  # Keep Y-axis as up direction

    # Compute LookAt extrinsic matrix
    extrinsic = compute_lookat_open3d(camera_position, center, up_vector)

    # Create PinholeCameraParameters object
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.extrinsic = extrinsic
    cam_params.intrinsic.set_intrinsics(
        width, height, fx=600, fy=600, cx=width // 2 - 0.5, cy=height // 2 - 0.5
    )

    return cam_params


def debug_draw_move_axis(vis, move_centers, move_axises):
    """
    Draw movement axes (translation or rotation) and their center points in an Open3D visualization window.
    This is a helper function to verify whether the motion axes and centers used for GIF generation are correct.

    Parameters:
        move_centers (list of list or np.ndarray): List of 3D center points. Each center is a list or array [x, y, z].
        move_axises (list of list or np.ndarray): List of 3D direction vectors. Each vector is [dx, dy, dz].
    """
    for move_center, move_axis in zip(move_centers, move_axises):
        move_center = np.array(move_center, dtype=np.float64)
        move_axis = np.array(move_axis, dtype=np.float64)
        axis_length = 2.0  # Length of the drawn axis line
        axis_end = move_center + move_axis * axis_length

        points = [move_center, axis_end]  # Two endpoints of the line segment
        lines = [[0, 1]]  # Line connects point 0 to point 1
        colors = [[1, 0, 0]]  # Red color for the axis line
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

        # Draw a small green sphere at the first center point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(move_centers[0])
        sphere.paint_uniform_color([0, 1, 0])  # Green color
        vis.add_geometry(sphere)


def analyze_actionable_parts_from_conceptualization(conceptualization):
    """
    Analyzes the object's conceptualization to identify actionable parts and their motion axes.
    Args:
        conceptualization (list): A list of dictionaries, where each dictionary describes a
                         component and its parameters (including position, rotation)
                         and optionally semantic information.
    Returns:
        dict: A dictionary containing information about actionable parts and their motion axes.
            - template name(list[str]): The template name of the part concept.
            - move type(list[str]): The type of motion (e.g., "revolute", "prismatic").
            - move center (list[float]): The center of rotation or translation.
            - move axis (List[float]): The axis of rotation or translation.
            - move_state_and_limits (list[Tuple[float, float, float]]): The limits of the motion (min, max, start). in degrees.
    """
    actionable_parts_info = {}

    def find_concept_by_template_name(template_name):
        for concept in conceptualization:
            if template_name.lower() in concept["template"].lower():
                return concept
        return None

    # find screen_concept and handle_concept
    screen_concept = find_concept_by_template_name("screen")
    concept_rotation = np.array(screen_concept["parameters"]["rotation"])
    base_rotation = np.array(screen_concept["parameters"]["rotation"])
    rot_diff = concept_rotation - base_rotation
    assert np.all(np.abs(rot_diff) == 0), " base and screen rotation should be the same"

    screen_rotation = screen_concept["parameters"]["screen_rotation"]
    concept_pos = np.array(screen_concept["parameters"]["position"])
    size = np.array(screen_concept["parameters"]["size"])
    offset = np.array(screen_concept["parameters"]["offset"])

    rot_axis = np.array([1, 0, 0])
    rot_axis = apply_transformation(rot_axis, [0, 0, 0], concept_rotation)

    assert np.sum(np.abs(screen_rotation) != 0) <= 1, (
        "Door rotation should be along one axis only."
    )

    # note that don't modify the original screen_rotation
    screen_rotation_rad = screen_rotation[0] / 180 * np.pi
    back_mesh_position = [
        0,
        offset[0] + size[1] * np.cos(screen_rotation_rad) / 2,
        offset[1],
    ]
    back_mesh_rotation = [screen_rotation_rad, 0, 0]
    axis_pos = [0, -size[1] / 2, size[2] / 2]
    axis_pos = apply_transformation(axis_pos, back_mesh_position, back_mesh_rotation)
    axis_pos = apply_transformation(axis_pos, concept_pos, concept_rotation)

    move_state_and_limits = []
    assert -90 <= screen_rotation[0] <= 90, (
        "Screen rotation should be between -90 and 90 degrees."
    )
    screen_state_and_limit = np.array([-90, 90, screen_rotation[0]])
    move_state_and_limits.append(screen_state_and_limit)

    actionable_parts_info["template_names"] = [screen_concept["template"]]
    actionable_parts_info["move_types"] = ["revolute"]
    actionable_parts_info["move_centers"] = [axis_pos]
    actionable_parts_info["move_axises"] = [rot_axis]
    actionable_parts_info["move_state_and_limits"] = move_state_and_limits
    return actionable_parts_info
