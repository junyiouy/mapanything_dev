#!/usr/bin/env python3
"""
Script to load COLMAP data, convert to OpenCV format, and visualize cameras and point cloud.
Uses first frame as reference coordinate system.
"""

import os
import os.path as osp
import numpy as np
import trimesh
import sys

# Add mapanything to path
sys.path.append('/wekafs/ict/junyiouy/map-anything')

from mapanything.utils.geometry import colmap_to_opencv_intrinsics
from misc.opendv_utils import integrate_camera_into_scene

# Import COLMAP readers
colmap_reader_path = '/wekafs/ict/junyiouy/map-anything/data_processing/wai_processing/third_party/mvsanywhere/src/mvsanywhere/datasets'
if colmap_reader_path not in sys.path:
    sys.path.append(colmap_reader_path)
from read_write_colmap_model import read_cameras_binary, read_images_binary, read_points3D_binary

def load_colmap_data(colmap_path):
    """
    Load COLMAP data from binary files.

    Args:
        colmap_path: Path to directory containing cameras.bin, images.bin, points3D.bin

    Returns:
        cameras, images, points3D: COLMAP data structures
    """
    cameras = read_cameras_binary(osp.join(colmap_path, 'cameras.bin'))
    images = read_images_binary(osp.join(colmap_path, 'images.bin'))
    points3D = read_points3D_binary(osp.join(colmap_path, 'points3D.bin'))

    return cameras, images, points3D


def get_colmap_to_opencv_matrix():
    """
    Get the coordinate conversion matrix from COLMAP to OpenCV.

    COLMAP: +X right, +Y up, +Z backward
    OpenCV: +X right, +Y down, +Z forward

    Returns:
        4x4 conversion matrix
    """
    # Flip Y axis
    matrix = np.eye(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def process_colmap_poses(images):
    """
    Process COLMAP image poses and convert to OpenCV camera-to-world format.

    Args:
        images: COLMAP images dict

    Returns:
        c2w_poses: List of 4x4 camera-to-world matrices
        image_ids: List of corresponding image IDs
        image_names: List of image names
    """
    c2w_poses = []
    image_ids = []
    image_names = []

    for img_id, img in images.items():
        # Get the 4x4 pose matrix (world-to-camera in COLMAP)
        R = img.qvec2rotmat()  # 3x3 rotation matrix
        t = img.tvec  # 3x1 translation vector

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        # Convert to OpenCV camera-to-world
        c2w = np.linalg.inv(w2c)
        c2w[1, 1] *= -1
        c2w[2, 2] *= -1

        c2w_poses.append(c2w)
        image_ids.append(img_id)
        image_names.append(img.name)

    return c2w_poses, image_ids, image_names

def transform_to_reference_frame(poses, points, reference_idx=0):
    """
    Transform all poses and points to the reference frame (first pose).

    Args:
        poses: List of 4x4 camera-to-world matrices
        points: Nx3 array of 3D points
        reference_idx: Index of reference pose

    Returns:
        transformed_poses: List of transformed poses
        transformed_points: Transformed 3D points
    """
    if len(poses) == 0:
        return poses, points

    # Get reference pose (camera-to-world of first frame)
    ref_pose = poses[reference_idx]

    # Transform all poses to be relative to reference
    transformed_poses = []
    for pose in poses:
        # pose_relative = ref_pose_inv @ pose
        # Since both are c2w, the relative pose is ref_pose_inv @ pose
        ref_pose_inv = np.linalg.inv(ref_pose)
        pose_relative = ref_pose_inv @ pose
        transformed_poses.append(pose_relative)

    # Transform points to reference frame
    if points is not None and len(points) > 0:
        # Points are in world coordinates, transform to reference camera frame
        transformed_points = (ref_pose_inv @ np.column_stack([points, np.ones(len(points))]).T).T[:, :3]
    else:
        transformed_points = points

    return transformed_poses, transformed_points

def create_visualization_scene(poses, points, colors=None, scene_scale=None):
    """
    Create trimesh scene with cameras and point cloud.

    Args:
        poses: List of 4x4 camera-to-world matrices
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (0-255)
        scene_scale: Scale for camera visualization

    Returns:
        scene: trimesh.Scene object
    """
    scene = trimesh.Scene()

    # Add point cloud
    if points is not None and len(points) > 0:
        if colors is not None:
            # Convert colors to 0-1 range if needed
            if colors.max() > 1.0:
                colors = colors / 255.0
            point_cloud = trimesh.PointCloud(vertices=points, colors=colors)
        else:
            point_cloud = trimesh.PointCloud(vertices=points)
        scene.add_geometry(point_cloud)

    # Determine scene scale if not provided
    if scene_scale is None and points is not None and len(points) > 0:
        lower_percentile = np.percentile(points, 5, axis=0)
        upper_percentile = np.percentile(points, 95, axis=0)
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)
    elif scene_scale is None:
        scene_scale = 1.0

    # Add cameras
    import matplotlib
    matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    colormap = plt.cm.gist_rainbow
    for i, pose in enumerate(poses):
        rgba_color = colormap(i / len(poses))
        current_color = tuple(int(255 * x) for x in rgba_color[:3])
        integrate_camera_into_scene(scene, pose, current_color, scene_scale)

    return scene

def main():
    # Data paths
    data_root = '/wekafs/ict/junyiouy/map-anything/bosch_data/1755150222210'
    colmap_path = osp.join(data_root, 'sparse', '0')
    images_base_dir = osp.join(data_root, 'images', 'rig000001')

    print("Loading COLMAP data...")
    cameras, images, points3D = load_colmap_data(colmap_path)

    print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")

    # Print camera intrinsics
    print("\nCamera intrinsics:")
    for cam_id, cam in cameras.items():
        print(f"Camera {cam_id}: model={cam.model}, width={cam.width}, height={cam.height}")
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params
            print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params
            print(f"  f={f:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        else:
            print(f"  params={cam.params}")

    print("Processing camera poses...")
    c2w_poses, image_ids, image_names = process_colmap_poses(images)

    # Extract point cloud data
    if len(points3D) > 0:
        points = np.array([pt.xyz for pt in points3D.values()])
        colors = np.array([pt.rgb for pt in points3D.values()])
        print(f"Point cloud: {len(points)} points")
    else:
        points = None
        colors = None
        print("No 3D points found")

    print("Transforming to reference coordinate system...")
    # Use first pose as reference
    transformed_poses, transformed_points = transform_to_reference_frame(c2w_poses, points, reference_idx=0)

    print("Creating visualization...")
    scene = create_visualization_scene(transformed_poses, transformed_points, colors)

    # Export to GLB
    output_path = '/wekafs/ict/junyiouy/map-anything/colmap_visualization.glb'
    scene.export(output_path)
    print(f"Visualization saved to: {output_path}")

    # Also show scene info
    print(f"Scene contains {len(scene.geometry)} geometries")
    print(f"Number of cameras: {len(transformed_poses)}")
    if transformed_points is not None:
        print(f"Number of points: {len(transformed_points)}")

if __name__ == "__main__":
    main()
