#!/usr/bin/env python3
"""
Script to run MapAnything inference on Bosch COLMAP data.

This script loads COLMAP data (images, camera intrinsics, and extrinsics),
converts them to MapAnything's expected format, runs inference, and saves results.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
import os.path as osp

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

import sys
sys.path.append('/wekafs/ict/junyiouy/map-anything')
# Import GLB creation function
from misc.opendv_utils import predictions_to_glb, convert_mapanything_to_pi3_format

# Add mapanything to path
sys.path.append('/wekafs/ict/junyiouy/map-anything')

from mapanything.utils.geometry import colmap_to_opencv_intrinsics

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


def load_image_for_preprocessing(image_path):
    """
    Load image in format expected by MapAnything's preprocess_inputs.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image: Image in RGB format
    """
    # Load image
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    return img


def prepare_views_for_colmap(images_base_dir, cameras, images, modalities, metric_scale=True):
    """
    Prepare all views for COLMAP data in MapAnything's expected format.

    Args:
        images_base_dir: Base directory containing the images
        cameras: COLMAP cameras dict
        images: COLMAP images dict
        modalities: List of modalities to include
        metric_scale: Whether inputs are in metric scale

    Returns:
        list: List of view dictionaries for MapAnything's preprocess_inputs
    """
    views = []

    # Process each image
    for img_id, img in images.items():
        # Get camera intrinsics
        cam_id = img.camera_id
        if cam_id not in cameras:
            print(f"Warning: Camera {cam_id} not found for image {img_id}")
            continue

        camera = cameras[cam_id]

        # Load image
        image_path = osp.join(images_base_dir, img.name)
        if not osp.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        image = load_image_for_preprocessing(image_path)

        # Create base view dictionary
        view = {
            'img': image,  # PIL Image
            'data_norm_type': ['dinov2'],  # Will be updated to match model
            'is_metric_scale': torch.tensor([metric_scale]),
        }

        # Add modalities based on selection
        if 'intrinsics' in modalities:
            # Build intrinsics matrix from COLMAP camera parameters
            if camera.model == "PINHOLE":
                fx, fy, cx, cy = camera.params
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            elif camera.model == "SIMPLE_PINHOLE":
                f, cx, cy = camera.params
                K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported camera model: {camera.model}")

            # Convert COLMAP intrinsics to OpenCV format
            opencv_intrinsics = colmap_to_opencv_intrinsics(K)
            view['intrinsics'] = opencv_intrinsics.astype(np.float32)

        if 'poses' in modalities:
            # Get camera-to-world pose
            R = img.qvec2rotmat()
            t = img.tvec

            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t

            # Convert to OpenCV camera-to-world
            c2w = np.linalg.inv(w2c)
            c2w[1, 1] *= -1  # Flip Y axis
            c2w[2, 2] *= -1  # Flip Z axis

            view['camera_poses'] = c2w.astype(np.float32)

        # Note: depth and ray_directions not implemented for COLMAP
        if 'depth' in modalities:
            print("Warning: Depth modality requested but not available for COLMAP data")
        if 'ray_directions' in modalities:
            print("Warning: Ray directions modality requested but not implemented for COLMAP")

        views.append(view)

    return views


def run_inference(model, views, memory_efficient=False):
    """
    Run MapAnything inference on the prepared views.

    Args:
        model: MapAnything model instance
        views: List of view dictionaries
        memory_efficient: Whether to use memory efficient inference

    Returns:
        list: List of prediction dictionaries
    """
    if len(views) == 0:
        return []

    print(f"Running inference on {len(views)} views...")

    # Preprocess inputs
    processed_views = preprocess_inputs(views)

    # Run inference
    predictions = model.infer(
        processed_views,
        memory_efficient_inference=memory_efficient,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        confidence_percentile=20,
    )

    return predictions


def save_results(predictions, output_dir, sequence_name,
                 depth_validity_check=True, conf_threshold=0.0, max_points=-1):
    """
    Save inference results as GLB scene with cameras and point cloud.

    Args:
        predictions: List of prediction dictionaries from MapAnything
        output_dir: Output directory path
        sequence_name: Name for the output files
        depth_validity_check: Whether to check depth validity (> 0)
        conf_threshold: Confidence threshold for filtering (0.0 = no filtering)
        max_points: Maximum number of points to keep (-1 = no limit)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert MapAnything predictions to Pi3 format for GLB creation
    pi3_predictions = convert_mapanything_to_pi3_format(predictions)

    # Apply additional filtering if needed
    if conf_threshold > 0.0 or max_points > 0:
        points = pi3_predictions['points']
        images = pi3_predictions['images']
        conf = pi3_predictions['conf']

        # Apply confidence threshold
        if conf_threshold > 0.0:
            valid_mask = conf >= conf_threshold
        else:
            valid_mask = np.ones(len(points), dtype=bool)

        # Apply point limit
        if max_points > 0 and np.sum(valid_mask) > max_points:
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > max_points:
                sampled_indices = np.random.choice(valid_indices, size=max_points, replace=False)
                valid_mask = np.zeros(len(points), dtype=bool)
                valid_mask[sampled_indices] = True

        # Filter data
        pi3_predictions['points'] = points[valid_mask]
        pi3_predictions['images'] = images[valid_mask]
        pi3_predictions['conf'] = conf[valid_mask]

    # Create GLB scene with cameras and point cloud
    print("Creating GLB scene with cameras and point cloud...")
    scene_3d = predictions_to_glb(pi3_predictions, show_cam=True)

    # Save GLB file
    glb_path = os.path.join(output_dir, f"{sequence_name}_scene.glb")
    print(f"Saving GLB scene to: {glb_path}")
    scene_3d.export(glb_path)

    print(f"Saved GLB scene with {len(pi3_predictions['points'])} points and {len(pi3_predictions['camera_poses']) if pi3_predictions['camera_poses'] is not None else 0} cameras")


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # randomly pick which True positions to keep
    sampled_indices = np.random.choice(
        true_indices, size=max_trues, replace=False
    )  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)


def validate_modalities(modalities):
    """验证模态组合的有效性"""
    # 必须包含图像
    if 'image' not in modalities:
        raise ValueError("Image modality is required")

    # 内参和光线方向不能同时使用
    if 'intrinsics' in modalities and 'ray_directions' in modalities:
        raise ValueError("Cannot use both intrinsics and ray_directions")

    # 如果使用深度，必须有标定信息
    if 'depth' in modalities:
        if 'intrinsics' not in modalities and 'ray_directions' not in modalities:
            raise ValueError("Depth requires calibration info (intrinsics or ray_directions)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Run MapAnything on Bosch COLMAP data")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to Bosch data root directory (containing sparse/ and images/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs='+',
        default=['image', 'intrinsics', 'poses'],
        choices=['image', 'intrinsics', 'poses', 'depth', 'ray_directions'],
        help="Modalities to use (default: image intrinsics poses)"
    )
    parser.add_argument(
        "--no_metric_scale",
        action="store_true",
        default=False,
        help="Use non-metric scale for inputs"
    )
    parser.add_argument(
        "--no_memory_efficient",
        action="store_true",
        help="Use memory efficient inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--no_depth_validity_check",
        action="store_true",
        default=False,
        help="Skip depth validity check (keep points with depth <= 0)"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for filtering points (default: 0.0, no filtering)"
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=-1,
        help="Maximum number of points to keep (-1 for no limit, default: -1)"
    )

    args = parser.parse_args()

    # Validate modalities
    validate_modalities(args.modalities)
    print(f"Using modalities: {args.modalities}")
    print(f"Metric scale: {not args.no_metric_scale}")

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up paths
    colmap_path = osp.join(args.data_root, 'sparse', '0')
    images_base_dir = osp.join(args.data_root, 'images')

    # Check if paths exist
    if not osp.exists(colmap_path):
        print(f"Error: COLMAP path not found: {colmap_path}")
        sys.exit(1)

    if not osp.exists(images_base_dir):
        print(f"Error: Images directory not found: {images_base_dir}")
        sys.exit(1)

    # Load COLMAP data
    print(f"Loading COLMAP data from: {colmap_path}")
    cameras, images, points3D = load_colmap_data(colmap_path)
    print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")

    # Get sequence name from data root
    sequence_name = osp.basename(args.data_root)

    # Prepare views for MapAnything
    print("Preparing views for MapAnything...")
    views = prepare_views_for_colmap(
        images_base_dir, cameras, images, args.modalities, not args.no_metric_scale
    )
    print(f"Prepared {len(views)} views")

    if len(views) == 0:
        print("No valid views found, exiting...")
        sys.exit(1)

    # Initialize MapAnything model
    print("Loading MapAnything model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    # Update data_norm_type in views to match model
    data_norm_type = model.encoder.data_norm_type
    print(f"Using data normalization type: {data_norm_type}")

    for view in views:
        view['data_norm_type'] = [data_norm_type]

    # Run inference
    predictions = run_inference(model, views, not args.no_memory_efficient)

    if len(predictions) == 0:
        print("No predictions generated, exiting...")
        sys.exit(1)

    # Save results
    save_results(
        predictions, args.output_dir, sequence_name,
        not args.no_depth_validity_check, args.conf_threshold, args.max_points
    )

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()

#  python scripts/infer_bosch_data.py --data_root /wekafs/ict/junyiouy/map-anything/bosch_data/1755150222210 --output_dir /wekafs/ict/junyiouy/map-anything/misc_files/bosch_res_no_metric
