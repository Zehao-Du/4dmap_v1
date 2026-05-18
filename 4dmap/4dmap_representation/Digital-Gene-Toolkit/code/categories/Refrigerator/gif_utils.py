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


def find_concept_idx_by_idx(concepts, local_idx, template_name):
    """Find the global index of a concept by its occurrence order among concepts of the same type.
    local_idx starts from 1, and global_idx starts from 0."""
    for global_idx, concept in enumerate(concepts):
        if concept["template"].lower() == template_name.lower():
            if local_idx == 1:
                return global_idx
            local_idx -= 1
    raise ValueError(
        f"Concept with template '{template_name}' not found in conceptualization."
    )
    
    
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
    part_meshes = {}
    color_map = {}  # record semantic to color index mapping

    for idx, c in enumerate(concepts):
        module_name = c["template"]
        try:
            module = eval(module_name)
            component = module(**c["parameters"])

            name = f"{module_name}_{idx}"
            semantic = component.semantic

            # assign a color index to the current semantic if not already assigned
            if semantic not in color_map:
                color_map[semantic] = len(color_map) % len(COLOR20)

            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(component.vertices),
                o3d.utility.Vector3iVector(component.faces),
            )
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(np.array(COLOR20[color_map[semantic]]))

            part_meshes[name] = mesh
        except Exception as e:
            print(f"Failed to load or instantiate template '{module_name}': {e}")

    return part_meshes


def assemble_concepts_instance(concepts):
    """Similar to assemble_concepts, this function assembles the object mesh based on the concept list. However, coloring is applied at the instance level(e.g. tophandle and sidehandle), rather than the semantic level."""
    vertices_list = {}
    faces_list = {}
    total_num_vertices = {}
    part_meshes = {}
    for concept_idx, c in enumerate(concepts):
        module_name = c["template"]

        try:
            module = eval(module_name)
            component = module(**c["parameters"])
            if "leg" in module_name.lower():
                num_legs = component.num_legs
                num_instance = int(num_legs[0])
            else:
                num_instance = 1

            per_instance_vertices = len(component.vertices) // num_instance
            per_instance_faces = len(component.faces) // num_instance
            assert per_instance_vertices * num_instance == len(component.vertices), (
                f"The number of vertices does not match for each button: {per_instance_vertices} * {num_instance} != {len(component.vertices)}"
            )
            for instance_idx in range(num_instance):
                instance_name = f"{module_name}_{concept_idx}_{instance_idx}"
                vertices_list.setdefault(instance_name, [])
                faces_list.setdefault(instance_name, [])
                total_num_vertices.setdefault(instance_name, 0)

                vertices_list[instance_name].append(
                    component.vertices[
                        instance_idx * per_instance_vertices : (instance_idx + 1)
                        * per_instance_vertices
                    ]
                )
                faces_list[instance_name].append(
                    component.faces[
                        instance_idx * per_instance_faces : (instance_idx + 1)
                        * per_instance_faces
                    ]
                    - instance_idx * per_instance_vertices
                )
                total_num_vertices[instance_name] += (
                    instance_idx * per_instance_vertices
                )
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
            - move axis (list[float]): The axis of motion.
            - move_state_and_limits (list[Tuple[float, float, float]]): The limits of the motion (min, max, start). in degrees.
            - concept_idxes (list[int]): The indices of the concepts in the conceptualization.
    """
    actionable_parts_info = {}
    concept_pairs = get_door_handle_vessel_pairs(conceptualization)

    for concept_pair in concept_pairs:
        door_concept = concept_pair["door"][0]
        handle_concept = concept_pair["handle"][0]
        vessel_concepts = concept_pair.get("vessel", [])
        door_rotation = door_concept["parameters"]["rotation"]
        axis_pos = np.array(door_concept["parameters"]["position"])

        # Handle the case where door_rotation is zero, determine rotation axis based on handle position
        if len(concept_pairs) == 1 and door_rotation[1] == 0:
            handle_pos_x = handle_concept["parameters"]["position"][0]
            door_pos_x = door_concept["parameters"]["position"][0]
            rotation_axis = [0, 1, 0]  # Default to y-axis rotation
            if handle_pos_x > door_pos_x:
                # Left axis
                axis_pos += np.array([-door_concept["parameters"]["size"][0] / 2, 0, 0])
                rotation_angle = -0.001
            else:
                # Right axis
                axis_pos += np.array([door_concept["parameters"]["size"][0] / 2, 0, 0])
                rotation_angle = 0.001
        else:
            idx2axis = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
            rotation_axis_idx = np.argmax(np.abs(door_rotation))
            if np.array_equal(door_rotation, [0, 0, 0]):
                rotation_axis_idx = 1  # Default to y-axis rotation
            rotation_axis = idx2axis[rotation_axis_idx]
            rotation_angle = door_rotation[rotation_axis_idx]
            rotation_angle_rad = np.radians(np.array(rotation_angle))
            assert np.sum(np.abs(door_rotation) != 0) <= 1, (
                "Door rotation should be along one axis only."
            )
            assert rotation_axis_idx == 1, "Door rotation should be along y axis."
            # Rotation is divided into four cases, left and right on the y-axis
            if rotation_angle > 0:  # y-axis left
                rot_offset = np.array(
                    [
                        np.cos(rotation_angle_rad)
                        * door_concept["parameters"]["size"][0]
                        / 2,
                        0,
                        -np.sin(rotation_angle_rad)
                        * door_concept["parameters"]["size"][0]
                        / 2,
                    ]
                )
            elif rotation_angle <= 0:  # y-axis right
                rot_offset = np.array(
                    [
                        -np.cos(rotation_angle_rad)
                        * door_concept["parameters"]["size"][0]
                        / 2,
                        0,
                        np.sin(rotation_angle_rad)
                        * door_concept["parameters"]["size"][0]
                        / 2,
                    ]
                )
            else:
                raise GenGifError("Rotation angle should not be zero.")
            axis_pos += rot_offset

        move_state_and_limits = np.zeros((3,), dtype=np.float32)
        if rotation_angle > 0:
            move_state_and_limits = np.array([0, 90, rotation_angle])
        else:
            move_state_and_limits = np.array([-90, 0, rotation_angle])
        num_concepts = len(vessel_concepts) + 2  # door + handle + vessels
        actionable_parts_info.setdefault("template_names", []).extend(
            [
                door_concept["template"],
                handle_concept["template"],
            ]
        )
        actionable_parts_info.setdefault("template_names", []).extend(
            [vessel_concept["template"] for vessel_concept in vessel_concepts]
        )
        actionable_parts_info.setdefault("move_types", []).extend(
            ["revolute"] * num_concepts
        )
        actionable_parts_info.setdefault("move_centers", []).extend(
            [axis_pos] * num_concepts
        )
        actionable_parts_info.setdefault("move_axises", []).extend(
            [rotation_axis] * num_concepts
        )
        actionable_parts_info.setdefault("move_state_and_limits", []).extend(
            [move_state_and_limits] * num_concepts
        )

        # find the global index of these concepts in the conceptualization
        vessel_concept_idxes = []
        door_concept_idx = -1
        handle_concept_idx = -1
        for concept_idx, concept in enumerate(conceptualization):
            if deep_dict_compare(concept, door_concept):
                door_concept_idx = concept_idx
            elif deep_dict_compare(concept, handle_concept):
                handle_concept_idx = concept_idx
            else:
                for vessel_concept in vessel_concepts:
                    if deep_dict_compare(concept, vessel_concept):
                        vessel_concept_idxes.append(concept_idx)
            if (
                door_concept_idx != -1
                and handle_concept_idx != -1
                and (len(vessel_concept_idxes) == len(vessel_concepts))
            ):
                break

        actionable_parts_info["concept_idxes"] = [door_concept_idx, handle_concept_idx]
        actionable_parts_info["concept_idxes"].extend(vessel_concept_idxes)

    return actionable_parts_info


def deep_dict_compare(obj1, obj2):
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            return False
        for key in obj1:
            if not deep_dict_compare(obj1[key], obj2[key]):
                return False
        return True
    elif isinstance(obj1, list) and isinstance(obj2, list):
        if len(obj1) != len(obj2):
            return False
        for v1, v2 in zip(obj1, obj2):
            if not deep_dict_compare(v1, v2):
                return False
        return True
    elif isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        return np.array_equal(obj1, obj2)
    elif isinstance(obj1, np.ndarray) and isinstance(obj2, list):
        return np.array_equal(obj1, np.array(obj2))
    elif isinstance(obj1, list) and isinstance(obj2, np.ndarray):
        return np.array_equal(np.array(obj1), obj2)
    else:
        return obj1 == obj2


def get_door_handle_vessel_pairs(conceptualization):
    """Returns:
    List[Dict[str, Any]]: A list of dictionaries, each containing the concept dictionaries for a door, handle, and vessel. Each dictionary has the following keys:
    - "door": The concept dictionary for the door, value is list[concept]
    - "handle": The concept dictionary for the handle, value is list[concept]
    - "vessel": The concept dictionary for the vessel, value is list[concept]
    """
    concept_pairs = []

    def find_concept_by_template_name(template_name):
        concepts = []
        for concept in conceptualization:
            if template_name.lower() in concept["template"].lower():
                concepts.append(concept)
        return concepts

    # find door_concept and handle_concept
    door_concepts = find_concept_by_template_name("door")
    handle_concepts = find_concept_by_template_name("handle")
    vessel_concepts = find_concept_by_template_name("vessel")

    for door_concept in door_concepts:
        door_rotation_angle = door_concept["parameters"]["rotation"][1]
        find = False
        for handle_concept in handle_concepts:
            handle_rotation_angle = handle_concept["parameters"]["rotation"][1]
            if abs(door_rotation_angle - handle_rotation_angle) < 1e-5:
                find = True
                concept_dict = {
                    "door": [door_concept],
                    "handle": [handle_concept],
                    "vessel": [],
                }
                break  # ensure one door corresponds to one handle
        assert find, (
            f"Refrigerator has door {door_concept['template']} with rotation angle {door_rotation_angle} but no handle with same rotation angle."
        )

        concept_pairs.append(concept_dict)

    for vessel_idx, vessel_concept in enumerate(vessel_concepts):
        vessel_rotation_angle = vessel_concept["parameters"]["rotation"][1]
        find = False
        for door_idx, door_concept in enumerate(door_concepts):
            door_rotation_angle = door_concept["parameters"]["rotation"][1]
            if abs(vessel_rotation_angle - door_rotation_angle) < 1e-5:
                find = True
                concept_pairs[door_idx].setdefault("vessel", []).append(vessel_concept)
        assert find, (
            f"Refrigerator has vessel {vessel_concept['template']} with rotation angle {vessel_rotation_angle} but no door with same rotation angle."
        )

    return concept_pairs
