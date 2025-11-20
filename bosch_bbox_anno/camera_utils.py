"""
Camera parameter extraction and manipulation utilities for Bosch point cloud rendering.
"""

import numpy as np
import trimesh
import json
from typing import List, Dict, Tuple, Optional
import os.path as osp


def extract_camera_parameters_from_glb(glb_path: str) -> List[Dict]:
    """
    Extract camera intrinsics and extrinsics from GLB file.

    Args:
        glb_path: Path to the GLB file

    Returns:
        List of camera dictionaries with intrinsics, extrinsics, and names
    """
    try:
        # Load the GLB scene
        scene = trimesh.load(glb_path)

        cameras = []

        # GLB files store camera information in the scene graph
        # We need to parse the scene to find camera nodes
        if hasattr(scene, 'graph') and hasattr(scene.graph, 'nodes'):
            for node_name, node in scene.graph.nodes.items():
                if hasattr(node, 'camera') and node.camera is not None:
                    camera_info = {
                        'name': node_name,
                        'intrinsics': None,
                        'extrinsics': node.transform,
                        'camera_model': 'pinhole'  # Default assumption
                    }

                    # Try to extract intrinsics from camera object
                    if hasattr(node.camera, 'focal') and hasattr(node.camera, 'resolution'):
                        # GLTF camera format
                        focal = node.camera.focal
                        resolution = node.camera.resolution
                        if len(focal) >= 2 and len(resolution) >= 2:
                            fx, fy = focal[0], focal[1]
                            cx, cy = resolution[0] / 2, resolution[1] / 2
                            camera_info['intrinsics'] = np.array([
                                [fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]
                            ])
                    elif hasattr(node.camera, 'intrinsics'):
                        # Direct intrinsics matrix
                        camera_info['intrinsics'] = node.camera.intrinsics

                    cameras.append(camera_info)

        # If no cameras found in scene graph, try alternative methods
        if not cameras:
            # Check if camera information is stored in scene metadata
            if hasattr(scene, 'metadata') and scene.metadata:
                if 'cameras' in scene.metadata:
                    cameras = scene.metadata['cameras']

        return cameras

    except Exception as e:
        print(f"Error extracting cameras from GLB: {e}")
        return []


def extract_camera_parameters_from_predictions(predictions_path: str) -> List[Dict]:
    """
    Extract camera parameters from MapAnything predictions file.

    Args:
        predictions_path: Path to the predictions .npz file

    Returns:
        List of camera dictionaries
    """
    try:
        # Load predictions
        data = np.load(predictions_path, allow_pickle=True)

        cameras = []

        if 'camera_poses' in data:
            camera_poses = data['camera_poses']
            if camera_poses is not None:
                for i, pose in enumerate(camera_poses):
                    cameras.append({
                        'name': f'camera_{i}',
                        'intrinsics': None,  # Will be extracted separately if available
                        'extrinsics': pose,
                        'camera_model': 'pinhole'
                    })

        # Try to get intrinsics from predictions if available
        if 'intrinsic' in data:
            intrinsics = data['intrinsic']
            for i, camera in enumerate(cameras):
                if i < len(intrinsics):
                    camera['intrinsics'] = intrinsics[i]

        return cameras

    except Exception as e:
        print(f"Error extracting cameras from predictions: {e}")
        return []


def save_camera_parameters(cameras: List[Dict], output_path: str):
    """
    Save camera parameters to JSON file.

    Args:
        cameras: List of camera dictionaries
        output_path: Output JSON file path
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_cameras = []
    for camera in cameras:
        cam_dict = camera.copy()
        if 'intrinsics' in cam_dict and cam_dict['intrinsics'] is not None:
            cam_dict['intrinsics'] = cam_dict['intrinsics'].tolist()
        if 'extrinsics' in cam_dict and cam_dict['extrinsics'] is not None:
            cam_dict['extrinsics'] = cam_dict['extrinsics'].tolist()
        serializable_cameras.append(cam_dict)

    with open(output_path, 'w') as f:
        json.dump(serializable_cameras, f, indent=2)


def load_camera_parameters(json_path: str) -> List[Dict]:
    """
    Load camera parameters from JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        List of camera dictionaries
    """
    with open(json_path, 'r') as f:
        cameras = json.load(f)

    # Convert lists back to numpy arrays
    for camera in cameras:
        if 'intrinsics' in camera and camera['intrinsics'] is not None:
            camera['intrinsics'] = np.array(camera['intrinsics'])
        if 'extrinsics' in camera and camera['extrinsics'] is not None:
            camera['extrinsics'] = np.array(camera['extrinsics'])

    return cameras


def get_camera_view_matrix(extrinsics: np.ndarray) -> np.ndarray:
    """
    Convert camera-to-world extrinsics to view matrix (world-to-camera).

    Args:
        extrinsics: 4x4 camera-to-world transformation matrix

    Returns:
        4x4 view matrix (world-to-camera)
    """
    return np.linalg.inv(extrinsics)


def create_camera_frustum_mesh(intrinsics: np.ndarray, extrinsics: np.ndarray,
                              frustum_length: float = 1.0) -> trimesh.Trimesh:
    """
    Create a camera frustum mesh for visualization.

    Args:
        intrinsics: 3x3 camera intrinsics matrix
        extrinsics: 4x4 camera-to-world transformation matrix
        frustum_length: Length of the frustum

    Returns:
        Trimesh object representing the camera frustum
    """
    if intrinsics is None:
        # Create a default frustum if no intrinsics available
        intrinsics = np.array([
            [1000, 0, 640],
            [0, 1000, 480],
            [0, 0, 1]
        ])

    # Camera principal point and focal lengths
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Image plane dimensions (assuming 1280x720 for visualization)
    img_width, img_height = 1280, 720

    # Calculate corner points on image plane
    corners_2d = np.array([
        [0, 0],
        [img_width, 0],
        [img_width, img_height],
        [0, img_height]
    ])

    # Convert to camera coordinates
    corners_camera = np.zeros((4, 3))
    for i, (x, y) in enumerate(corners_2d):
        # Convert to normalized device coordinates
        x_ndc = (x - cx) / fx
        y_ndc = (y - cy) / fy
        # Project to 3D at frustum_length distance
        corners_camera[i] = np.array([x_ndc * frustum_length, y_ndc * frustum_length, frustum_length])

    # Camera center in camera coordinates
    camera_center = np.array([0, 0, 0])

    # Create frustum vertices (camera center + 4 corners)
    vertices = np.vstack([camera_center, corners_camera])

    # Create faces for the frustum pyramid
    faces = [
        [0, 1, 2],  # front face
        [0, 2, 3],  # right face
        [0, 3, 4],  # back face
        [0, 4, 1],  # left face
        [1, 2, 3, 4]  # base face
    ]

    # Transform vertices to world coordinates
    vertices_homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    vertices_world = (extrinsics @ vertices_homogeneous.T).T[:, :3]

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces)

    return mesh
