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
from misc.opendv_utils import predictions_to_glb, convert_mapanything_to_pi3_format, transform_points

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
        camera_name = img.name.split('/')[-2]
        if not osp.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        image = load_image_for_preprocessing(image_path)

        # Create base view dictionary
        view = {
            'img': image,  # PIL Image
            'data_norm_type': ['dinov2'],  # Will be updated to match model
            'is_metric_scale': torch.tensor([metric_scale]),
            'camera_name': camera_name
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


def group_images_by_rig_sequence(images, cameras, rig_ranges):
    """
    Group images by rig sequence ranges.

    Args:
        images: COLMAP images dict
        cameras: COLMAP cameras dict
        rig_ranges: List of tuples [(start, end), ...] defining rig number ranges

    Returns:
        rig_groups: Dict[group_name, List[image_data]]
        camera_names: Set of unique camera names
    """
    rig_groups = {}
    camera_names = set()

    for img_id, img in images.items():
        # Parse image path: rigXXXXXX/camera/image.jpg
        parts = img.name.split('/')

        if len(parts) >= 3:
            rig_str = parts[-3]  # rigXXXXXX
            camera_name = parts[-2]

            # Extract rig number
            try:
                rig_num = int(rig_str.replace('rig', ''))
            except ValueError:
                print(f"Warning: Could not parse rig number from {rig_str}")
                continue

            camera_names.add(camera_name)

            # Find which range this rig belongs to
            for start, end in rig_ranges:
                if start <= rig_num <= end:
                    group_name = f"rig{start:06d}-{end:06d}"
                    if group_name not in rig_groups:
                        rig_groups[group_name] = []
                    rig_groups[group_name].append({
                        'img_id': img_id,
                        'img': img,
                        'camera_name': camera_name,
                        'rig_num': rig_num
                    })
                    break
        else:
            print(f"Warning: Unexpected image path structure: {img.name}")

    return rig_groups, camera_names


def group_images_by_timestamp(images, cameras):
    """
    Group images by timestamp, assuming timestamp is encoded in image name.

    Args:
        images: COLMAP images dict
        cameras: COLMAP cameras dict

    Returns:
        timestamp_groups: Dict[timestamp, List[image_data]]
        camera_names: Set of unique camera names
    """
    timestamp_groups = {}
    camera_names = set()

    for img_id, img in images.items():
        # Extract timestamp from image name (adjust parsing logic as needed)
        # New format: rigXXXXXX/camera/image.jpg
        parts = img.name.split('/')

        # For path like 'rig000001/FrontCam02/image000001.jpg':
        # parts[-3] = 'rig000001' (timestamp)
        # parts[-2] = 'FrontCam02' (camera name)
        # parts[-1] = 'image000001.jpg' (image file)
        if len(parts) >= 3:
            timestamp = parts[-3]
            camera_name = parts[-2]
        else:
            # Fallback for old format or unexpected structure
            print(f"Warning: Unexpected image path structure: {img.name}")
            timestamp = parts[-1] if len(parts) >= 1 else f"unknown_{img_id}"
            camera_name = parts[-2] if len(parts) >= 2 else "unknown"

        camera_names.add(camera_name)

        if timestamp not in timestamp_groups:
            timestamp_groups[timestamp] = []

        timestamp_groups[timestamp].append({
            'img_id': img_id,
            'img': img,
            'camera_name': camera_name
        })

    # Sort timestamps (assuming they are sortable)
    sorted_timestamps = sorted(timestamp_groups.keys())

    return {ts: timestamp_groups[ts] for ts in sorted_timestamps}, camera_names


def create_timestamp_chunks(sorted_timestamps, chunk_size, overlap_size):
    """
    Create overlapping chunks from sorted timestamps.

    Args:
        sorted_timestamps: List of sorted timestamp strings
        chunk_size: Number of timestamps per chunk
        overlap_size: Number of overlapping timestamps between chunks

    Returns:
        list: List of chunk timestamp lists
    """
    chunks = []
    step_size = chunk_size - overlap_size

    for start_idx in range(0, len(sorted_timestamps), step_size):
        end_idx = min(start_idx + chunk_size, len(sorted_timestamps))
        chunk_timestamps = sorted_timestamps[start_idx:end_idx]
        chunks.append(chunk_timestamps)

        if end_idx == len(sorted_timestamps):
            break

    return chunks


def prepare_single_view_for_colmap(images_base_dir, cameras, img, modalities, metric_scale=True):
    """
    Prepare a single view for COLMAP data (extracted from prepare_views_for_colmap).

    Returns:
        view dict or None if invalid
    """
    # Get camera intrinsics
    cam_id = img.camera_id
    if cam_id not in cameras:
        return None

    camera = cameras[cam_id]

    # Load image
    image_path = osp.join(images_base_dir, img.name)
    camera_name = img.name.split('/')[-2] if len(img.name.split('/')) >= 2 else 'unknown'

    if not osp.exists(image_path):
        return None

    image = load_image_for_preprocessing(image_path)

    # Create view dictionary
    view = {
        'img': image,
        'data_norm_type': ['dinov2'],  # Will be updated
        'is_metric_scale': torch.tensor([metric_scale]),
        'camera_name': camera_name
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

    return view


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
        confidence_percentile=10,
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
    parser.add_argument(
        "--chunk_timestamps",
        action="store_true",
        help="Enable timestamp-based chunking for multi-camera sequences"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of timestamps per chunk (when chunk_timestamps is enabled)"
    )
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=1,
        help="Number of overlapping timestamps between chunks"
    )
    parser.add_argument(
        "--ref_camera",
        type=str,
        default=None,
        help="Reference camera name for chunk alignment (e.g., 'FrontCam02'). If specified, this camera's poses will be used as reference for alignment between chunks."
    )
    parser.add_argument(
        "--rig_ranges",
        type=str,
        default=None,
        help="Rig sequence ranges to process separately, format: '1-68,69-172,173-258'"
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

    # Parse rig ranges if specified
    rig_ranges = None
    if args.rig_ranges:
        rig_ranges = []
        for range_str in args.rig_ranges.split(','):
            start, end = map(int, range_str.strip().split('-'))
            rig_ranges.append((start, end))
        print(f"Processing rig ranges: {rig_ranges}")

    if rig_ranges:
        # Process each rig range separately
        print("Processing rig ranges separately...")

        # Group images by rig sequence
        rig_groups, camera_names = group_images_by_rig_sequence(images, cameras, rig_ranges)
        print(f"Found {len(rig_groups)} rig groups, {len(camera_names)} cameras")
        print(f"Camera names: {sorted(camera_names)}")

        # Initialize MapAnything model
        print("Loading MapAnything model...")
        model = MapAnything.from_pretrained("facebook/map-anything").to(device)
        model.eval()

        # Update data_norm_type to match model
        data_norm_type = model.encoder.data_norm_type
        print(f"Using data normalization type: {data_norm_type}")

        # Process each rig group
        for group_name, img_data_list in rig_groups.items():
            print(f"Processing rig group: {group_name} ({len(img_data_list)} images)")

            # Create filtered images dict for this group
            group_images = {data['img_id']: data['img'] for data in img_data_list}

            if args.chunk_timestamps:
                # Apply timestamp chunking within this rig group
                print(f"  Applying timestamp chunking for {group_name}...")

                # Group by timestamp within this rig group
                group_timestamp_groups, _ = group_images_by_timestamp(group_images, cameras)
                sorted_timestamps = list(group_timestamp_groups.keys())

                # Create chunks for this rig group
                chunks = create_timestamp_chunks(sorted_timestamps, args.chunk_size, args.overlap_size)
                print(f"  Split {group_name} into {len(chunks)} timestamp chunks")

                # Collect results for all chunks in this rig group
                chunk_results = []

                # Process each chunk within this rig group
                for i, chunk_timestamps in enumerate(chunks):
                    print(f"    Processing chunk {i}: timestamps {chunk_timestamps[0]} to {chunk_timestamps[-1]}")

                    # Collect all images for this chunk
                    chunk_views = []
                    ref_view_indices = []  # Track reference camera views

                    for ts in chunk_timestamps:
                        for img_data in group_timestamp_groups[ts]:
                            # Create view for this image
                            view = prepare_single_view_for_colmap(
                                images_base_dir, cameras, img_data['img'],
                                args.modalities, not args.no_metric_scale
                            )
                            if view is not None:
                                chunk_views.append(view)
                                # Track reference camera views
                                if args.ref_camera and view.get('camera_name') == args.ref_camera:
                                    ref_view_indices.append(len(chunk_views) - 1)

                    if len(chunk_views) == 0:
                        print(f"    No views found for chunk {i} in {group_name}, skipping...")
                        continue

                    # Update data_norm_type
                    for view in chunk_views:
                        view['data_norm_type'] = [data_norm_type]

                    # drop chunk_views['camera_name'] for inference
                    for view in chunk_views:
                        if 'camera_name' in view:
                            del view['camera_name']

                    # Run inference on chunk
                    predictions = run_inference(model, chunk_views, not args.no_memory_efficient)

                    # Convert and store results
                    pi3_predictions = convert_mapanything_to_pi3_format(predictions)
                    chunk_results.append({
                        'points': pi3_predictions['points'],
                        'images': pi3_predictions['images'],
                        'camera_poses': pi3_predictions['camera_poses'],
                        'conf': pi3_predictions['conf'],
                        'timestamps': chunk_timestamps,
                        'ref_view_indices': ref_view_indices if args.ref_camera else None
                    })
                    print(f"    Chunk {i}: {len(pi3_predictions['points'])} points")

                # Align chunks within this rig group
                aligned_chunks = []
                cum_T = np.eye(4)  # cumulative transform to first chunk

                for i, chunk in enumerate(chunk_results):
                    if i == 0:
                        # First chunk, no alignment
                        aligned_chunks.append({
                            'points': chunk['points'],
                            'images': chunk['images'],
                            'camera_poses': chunk['camera_poses'],
                            'cum_T': cum_T.copy(),
                            'ref_view_indices': chunk.get('ref_view_indices')
                        })
                    else:
                        # Compute alignment transform using reference camera
                        aligned = False
                        if args.ref_camera and chunk_results[i-1].get('ref_view_indices') and chunk.get('ref_view_indices'):
                            # Use reference camera poses for alignment
                            prev_ref_idx = chunk_results[i-1]['ref_view_indices'][-1]  # Last ref view in previous chunk
                            curr_ref_idx = chunk['ref_view_indices'][0]  # First ref view in current chunk

                            if (chunk_results[i-1]['camera_poses'] is not None and
                                chunk['camera_poses'] is not None and
                                prev_ref_idx < len(chunk_results[i-1]['camera_poses']) and
                                curr_ref_idx < len(chunk['camera_poses'])):

                                c2w_prev_ref = chunk_results[i-1]['camera_poses'][prev_ref_idx]
                                c2w_curr_ref = chunk['camera_poses'][curr_ref_idx]
                                T_align = c2w_prev_ref @ np.linalg.inv(c2w_curr_ref)
                                cum_T = cum_T @ T_align

                                print(f"Aligned chunk {i} using reference camera '{args.ref_camera}'")
                                aligned = True

                        if not aligned:
                            # Fallback: use any available poses for alignment
                            if chunk_results[i-1]['camera_poses'] is not None and chunk['camera_poses'] is not None:
                                # Use first pose of each chunk for alignment
                                c2w_prev_first = chunk_results[i-1]['camera_poses'][0]
                                c2w_curr_first = chunk['camera_poses'][0]
                                T_align = c2w_prev_first @ np.linalg.inv(c2w_curr_first)
                                cum_T = cum_T @ T_align
                                print(f"Aligned chunk {i} using fallback pose alignment")
                            else:
                                print(f"Warning: Could not align chunk {i}, no poses available")

                    # Transform points and poses
                    points_aligned = transform_points(cum_T, chunk['points'])
                    camera_poses_aligned = cum_T @ chunk['camera_poses'] if chunk['camera_poses'] is not None else None

                    aligned_chunks.append({
                        'points': points_aligned,
                        'images': chunk['images'],
                        'camera_poses': camera_poses_aligned,
                        'cum_T': cum_T.copy(),
                        'ref_view_indices': chunk.get('ref_view_indices')
                    })
                    print(f"Aligned chunk {i}, cum_T:\n{cum_T}")

                # Merge all chunks for this rig group
                all_points = []
                all_images = []
                all_camera_poses = []

                for chunk in aligned_chunks:
                    all_points.append(chunk['points'])
                    all_images.append(chunk['images'])
                    if chunk['camera_poses'] is not None:
                        all_camera_poses.append(chunk['camera_poses'])

                all_points = np.concatenate(all_points, axis=0)
                all_images = np.concatenate(all_images, axis=0)
                if all_camera_poses:
                    all_camera_poses = np.concatenate(all_camera_poses, axis=0)
                else:
                    all_camera_poses = None

                print(f"Merged {group_name}: {len(all_points)} points, {len(all_camera_poses) if all_camera_poses is not None else 0} camera poses")

                # Save merged results for this rig group
                predictions = {
                    'points': all_points,
                    'images': all_images,
                    'camera_poses': all_camera_poses,
                    'conf': np.ones(len(all_points))  # Default confidence
                }

                os.makedirs(args.output_dir, exist_ok=True)
                print("Creating GLB scene with cameras and point cloud...")
                scene_3d = predictions_to_glb(predictions, show_cam=True)

                # Save GLB file
                glb_path = os.path.join(args.output_dir, f"{sequence_name}_{group_name}_scene.glb")
                print(f"Saving GLB scene to: {glb_path}")
                scene_3d.export(glb_path)

                print(f"Saved GLB scene with {len(all_points)} points and {len(all_camera_poses) if all_camera_poses is not None else 0} cameras")
            else:
                # Single-pass processing for this rig group
                views = prepare_views_for_colmap(
                    images_base_dir, cameras, group_images, args.modalities, not args.no_metric_scale
                )
                print(f"Prepared {len(views)} views for {group_name}")

                if len(views) == 0:
                    print(f"No valid views found for {group_name}, skipping...")
                    continue

                # Update data_norm_type
                for view in views:
                    view['data_norm_type'] = [data_norm_type]
                    if 'camera_name' in view:
                        del view['camera_name']

                # Run inference
                predictions = run_inference(model, views, not args.no_memory_efficient)

                if len(predictions) == 0:
                    print(f"No predictions generated for {group_name}, skipping...")
                    continue

                # Save results for this rig group
                group_sequence_name = f"{sequence_name}_{group_name}"
                save_results(
                    predictions, args.output_dir, group_sequence_name,
                    not args.no_depth_validity_check, args.conf_threshold, args.max_points
                )

        print("\nAll rig groups processed!")
        return

    if args.chunk_timestamps:
        # Group images by timestamp
        print("Grouping images by timestamp...")
        timestamp_groups, camera_names = group_images_by_timestamp(images, cameras)
        sorted_timestamps = list(timestamp_groups.keys())

        print(f"Found {len(sorted_timestamps)} timestamps, {len(camera_names)} cameras")
        print(f"Camera names: {sorted(camera_names)}")

        # Validate reference camera if specified
        if args.ref_camera and args.ref_camera not in camera_names:
            print(f"Warning: Specified reference camera '{args.ref_camera}' not found in data")
            print(f"Available cameras: {sorted(camera_names)}")
            args.ref_camera = None

        if args.ref_camera:
            print(f"Using '{args.ref_camera}' as reference camera for chunk alignment")

        # Create chunks
        chunks = create_timestamp_chunks(sorted_timestamps, args.chunk_size, args.overlap_size)
        print(f"Split into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk)} timestamps ({chunk[0]} to {chunk[-1]})")

        # Initialize MapAnything model
        print("Loading MapAnything model...")
        model = MapAnything.from_pretrained("facebook/map-anything").to(device)
        model.eval()

        # Update data_norm_type to match model
        data_norm_type = model.encoder.data_norm_type
        print(f"Using data normalization type: {data_norm_type}")

        # Process each chunk
        chunk_results = []

        for i, chunk_timestamps in enumerate(chunks):
            print(f"Processing chunk {i}: timestamps {chunk_timestamps[0]} to {chunk_timestamps[-1]}")

            # Collect all images for this chunk
            chunk_views = []
            ref_view_indices = []  # Track reference camera views

            for ts in chunk_timestamps:
                for img_data in timestamp_groups[ts]:
                    # Create view for this image
                    view = prepare_single_view_for_colmap(
                        images_base_dir, cameras, img_data['img'],
                        args.modalities, not args.no_metric_scale
                    )
                    if view is not None:
                        chunk_views.append(view)
                        # Track reference camera views
                        if args.ref_camera and view.get('camera_name') == args.ref_camera:
                            ref_view_indices.append(len(chunk_views) - 1)

            if len(chunk_views) == 0:
                print(f"No views found for chunk {i}, skipping...")
                continue

            # Log reference camera info
            if args.ref_camera:
                print(f"  Found {len(ref_view_indices)} reference camera views in chunk {i}")

            # Update data_norm_type
            for view in chunk_views:
                view['data_norm_type'] = [data_norm_type]

            # drop chunk_views['camera_name'] for inference
            for view in chunk_views:
                if 'camera_name' in view:
                    del view['camera_name']

            # Run inference on chunk
            predictions = run_inference(model, chunk_views, not args.no_memory_efficient)

            # Convert and store results
            pi3_predictions = convert_mapanything_to_pi3_format(predictions)
            chunk_results.append({
                'points': pi3_predictions['points'],
                'images': pi3_predictions['images'],
                'camera_poses': pi3_predictions['camera_poses'],
                'conf': pi3_predictions['conf'],
                'timestamps': chunk_timestamps,
                'ref_view_indices': ref_view_indices if args.ref_camera else None
            })
            print(f"  Chunk {i}: {len(pi3_predictions['points'])} points")

        # Align chunks using reference camera poses
        aligned_chunks = []
        cum_T = np.eye(4)  # cumulative transform to first chunk

        for i, chunk in enumerate(chunk_results):
            if i == 0:
                # First chunk, no alignment
                aligned_chunks.append({
                    'points': chunk['points'],
                    'images': chunk['images'],
                    'camera_poses': chunk['camera_poses'],
                    'cum_T': cum_T.copy(),
                    'ref_view_indices': chunk.get('ref_view_indices')
                })
            else:
                # Compute alignment transform using reference camera
                aligned = False
                if args.ref_camera and chunk_results[i-1].get('ref_view_indices') and chunk.get('ref_view_indices'):
                    # Use reference camera poses for alignment
                    prev_ref_idx = chunk_results[i-1]['ref_view_indices'][-1]  # Last ref view in previous chunk
                    curr_ref_idx = chunk['ref_view_indices'][0]  # First ref view in current chunk

                    if (chunk_results[i-1]['camera_poses'] is not None and
                        chunk['camera_poses'] is not None and
                        prev_ref_idx < len(chunk_results[i-1]['camera_poses']) and
                        curr_ref_idx < len(chunk['camera_poses'])):

                        c2w_prev_ref = chunk_results[i-1]['camera_poses'][prev_ref_idx]
                        c2w_curr_ref = chunk['camera_poses'][curr_ref_idx]
                        T_align = c2w_prev_ref @ np.linalg.inv(c2w_curr_ref)
                        cum_T = cum_T @ T_align

                        print(f"Aligned chunk {i} using reference camera '{args.ref_camera}'")
                        aligned = True

                if not aligned:
                    # Fallback: use any available poses for alignment
                    if chunk_results[i-1]['camera_poses'] is not None and chunk['camera_poses'] is not None:
                        # Use first pose of each chunk for alignment
                        c2w_prev_first = chunk_results[i-1]['camera_poses'][0]
                        c2w_curr_first = chunk['camera_poses'][0]
                        T_align = c2w_prev_first @ np.linalg.inv(c2w_curr_first)
                        cum_T = cum_T @ T_align
                        print(f"Aligned chunk {i} using fallback pose alignment")
                    else:
                        print(f"Warning: Could not align chunk {i}, no poses available")

            # Transform points and poses
            points_aligned = transform_points(cum_T, chunk['points'])
            camera_poses_aligned = cum_T @ chunk['camera_poses'] if chunk['camera_poses'] is not None else None

            aligned_chunks.append({
                'points': points_aligned,
                'images': chunk['images'],
                'camera_poses': camera_poses_aligned,
                'cum_T': cum_T.copy(),
                'ref_view_indices': chunk.get('ref_view_indices')
            })
            print(f"Aligned chunk {i}, cum_T:\n{cum_T}")

        # Merge all chunks
        all_points = []
        all_images = []
        all_camera_poses = []

        for chunk in aligned_chunks:
            all_points.append(chunk['points'])
            all_images.append(chunk['images'])
            if chunk['camera_poses'] is not None:
                all_camera_poses.append(chunk['camera_poses'])

        all_points = np.concatenate(all_points, axis=0)
        all_images = np.concatenate(all_images, axis=0)
        if all_camera_poses:
            all_camera_poses = np.concatenate(all_camera_poses, axis=0)
        else:
            all_camera_poses = None

        print(f"Merged: {len(all_points)} points, {len(all_camera_poses) if all_camera_poses is not None else 0} camera poses")

        # Save merged results directly as GLB
        predictions = {
            'points': all_points,
            'images': all_images,
            'camera_poses': all_camera_poses,
            'conf': np.ones(len(all_points))  # Default confidence
        }

        os.makedirs(args.output_dir, exist_ok=True)
        print("Creating GLB scene with cameras and point cloud...")
        scene_3d = predictions_to_glb(predictions, show_cam=True)

        # Save GLB file
        glb_path = os.path.join(args.output_dir, f"{sequence_name}_scene.glb")
        print(f"Saving GLB scene to: {glb_path}")
        scene_3d.export(glb_path)

        print(f"Saved GLB scene with {len(all_points)} points and {len(all_camera_poses) if all_camera_poses is not None else 0} cameras")

    else:
        # Original single-pass inference
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
            if 'camera_name' in view:
                del view['camera_name']

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

#  python scripts/infer_bosch_data_chunk_new_data.py --data_root /wekafs/ict/junyiouy/map-anything/bosch_data/1973687048769372160_data --output_dir /wekafs/ict/junyiouy/map-anything/misc_files/bosch_res_no_metric_newdata_image_only --chunk_size 40 --chunk_timestamps --ref_camera FrontCam02 --no_metric_scale --modalities image --rig_ranges '1-68,69-172,173-258'
#  python scripts/infer_bosch_data_chunk_new_data.py --data_root /wekafs/ict/junyiouy/map-anything/bosch_data/1973687048769372160_data --output_dir /wekafs/ict/junyiouy/map-anything/misc_files/bosch_res_no_metric_newdata --chunk_size 40 --chunk_timestamps --ref_camera FrontCam02 --no_metric_scale --rig_ranges '1-68,69-172,173-258'