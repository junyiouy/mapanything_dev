#!/usr/bin/env python3
"""
Script to run MapAnything inference on KITTI image sequences with chunking and alignment.

This script processes a KITTI sequence by dividing it into overlapping chunks, running inference
on each chunk separately, then aligning all point clouds to the first chunk's coordinate system
using SE3 transformations from overlapping camera poses, and saving each chunk separately.
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import trimesh
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import predictions_to_glb

from time import time


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

    # center crop width with 1/2
    width, height = img.size
    new_width = width // 2
    left = (width - new_width) // 2
    right = left + new_width
    img = img.crop((left, 0, right, height))

    return img


def get_image_paths(sequence_dir, downsample_rate=1):
    """
    Get sorted list of image paths from sequence directory.

    Args:
        sequence_dir: Path to sequence directory containing images
        downsample_rate: Take every Nth image

    Returns:
        list: List of image file paths
    """
    # Find all image files (common extensions)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []

    for file in os.listdir(sequence_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(sequence_dir, file))

    # Sort by filename (assuming numerical naming)
    image_paths.sort()

    # Apply downsampling
    if downsample_rate > 1:
        image_paths = image_paths[::downsample_rate]

    return image_paths


def prepare_views_for_chunk(image_paths, chunk_indices, metric_scale=True):
    """
    Prepare all views for a specific chunk of indices.

    Args:
        image_paths: List of all image paths
        chunk_indices: List of indices for this chunk
        metric_scale: Whether inputs are in metric scale

    Returns:
        list: List of view dictionaries for MapAnything's preprocess_inputs
    """
    views = []

    # Process each index in the chunk
    for idx in sorted(chunk_indices):
        if idx >= len(image_paths):
            continue

        image_path = image_paths[idx]

        # Load image as PIL Image (preprocess_inputs will handle resizing)
        image = load_image_for_preprocessing(image_path)

        # Create base view dictionary
        view = {
            'img': image,  # PIL Image
            'data_norm_type': ['dinov2'],  # Will be updated to match model
            'is_metric_scale': torch.tensor([metric_scale]),
        }

        views.append(view)

    return views


def create_overlapping_chunks(all_indices, chunk_size, overlap_size):
    """
    Create overlapping chunks from a list of indices.

    Args:
        all_indices: Sorted list of all indices
        chunk_size: Number of indices per chunk
        overlap_size: Number of overlapping indices between chunks

    Returns:
        list: List of chunk index lists
    """
    chunks = []
    step_size = chunk_size - overlap_size

    for start_idx in range(0, len(all_indices), step_size):
        end_idx = min(start_idx + chunk_size, len(all_indices))
        chunk_indices = all_indices[start_idx:end_idx]
        chunks.append(chunk_indices)

        # Stop if we've reached the end
        if end_idx == len(all_indices):
            break

    return chunks


def run_inference_on_chunk(model, views, memory_efficient=False):
    """
    Run MapAnything inference on a chunk of views.

    Args:
        model: MapAnything model instance
        views: List of view dictionaries
        memory_efficient: Whether to use memory efficient inference

    Returns:
        list: List of prediction dictionaries
    """
    if len(views) == 0:
        return []

    print(f"Running inference on chunk with {len(views)} views...")

    # Preprocess inputs
    processed_views = preprocess_inputs(views)

    t0 = time()
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
    t1 = time()
    print(f"Inference time: {t1 - t0:.2f} seconds for {len(views)} views")

    return predictions


def find_ref_view_pose_in_chunk(chunk_predictions, chunk_indices, target_index):
    """
    Find the pose of a specific view (index) in chunk predictions.

    Args:
        chunk_predictions: Predictions from the chunk
        chunk_indices: Indices for the chunk
        target_index: Target index to find

    Returns:
        numpy.ndarray: 4x4 pose matrix, or None if not found
    """
    if target_index not in chunk_indices:
        return None

    # Find the index of the target index in the chunk
    idx_pos = chunk_indices.index(target_index)

    if idx_pos >= len(chunk_predictions):
        return None

    pred = chunk_predictions[idx_pos]

    if 'camera_poses' in pred:
        # Return the first camera pose (assuming single camera per prediction)
        return pred['camera_poses'][0].cpu().numpy()

    return None


def transform_point_cloud(points, transform_matrix):
    """
    Transform point cloud using SE3 transformation.

    Args:
        points: Point cloud array, can be (N, 3) or (B, H, W, 3)
        transform_matrix: 4x4 SE3 transformation matrix

    Returns:
        numpy.ndarray: Transformed points with same shape as input
    """
    original_shape = points.shape

    if len(original_shape) == 2:  # (N, 3)
        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])

        # Apply transformation
        transformed_homogeneous = points_homogeneous @ transform_matrix.T

        # Convert back to 3D coordinates
        transformed_points = transformed_homogeneous[:, :3]

    elif len(original_shape) == 4:  # (B, H, W, 3)
        # Reshape to (N, 3) for transformation
        points_reshaped = points.reshape(-1, 3)

        # Convert to homogeneous coordinates
        ones = np.ones((points_reshaped.shape[0], 1))
        points_homogeneous = np.hstack([points_reshaped, ones])

        # Apply transformation
        transformed_homogeneous = points_homogeneous @ transform_matrix.T

        # Convert back to 3D coordinates and reshape
        transformed_points = transformed_homogeneous[:, :3].reshape(original_shape)

    else:
        raise ValueError(f"Unsupported point cloud shape: {original_shape}")

    return transformed_points


def align_point_clouds_to_reference(predictions_list, chunks):
    """
    Align all chunk point clouds to the first chunk's coordinate system.

    For each chunk N (N > 0):
    - Find its ref view (first index) pose in chunk N
    - Find the same view pose in chunk N-1
    - Compute transform: T = pose_in_chunkN @ inv(pose_in_chunkN-1)
    - Accumulate transforms

    Note: prediction['camera_poses'] are c2w poses relative to each chunk's reference view.

    Args:
        predictions_list: List of predictions for each chunk
        chunks: List of chunk index lists

    Returns:
        list: List of aligned predictions
    """
    if len(predictions_list) <= 1:
        return predictions_list

    print("Aligning point clouds using ref view correspondences...")
    print("Note: Using c2w poses from model predictions (relative to each chunk's reference view)")

    # Compute cumulative transformations for each chunk relative to reference
    cumulative_transforms = [np.eye(4)]  # First chunk has identity transform (reference)

    for chunk_idx in range(1, len(chunks)):
        print(f"\nComputing transform for chunk {chunk_idx + 1}...")

        # Get ref view of current chunk (first index)
        current_chunk_indices = chunks[chunk_idx]
        ref_index = current_chunk_indices[0]

        print(f"  Ref view: index={ref_index}")

        # Find pose of ref view in current chunk
        pose_current = find_ref_view_pose_in_chunk(
            predictions_list[chunk_idx], current_chunk_indices, ref_index
        )

        # Find pose of same view in previous chunk
        prev_chunk_indices = chunks[chunk_idx-1]
        pose_prev = find_ref_view_pose_in_chunk(
            predictions_list[chunk_idx-1], prev_chunk_indices, ref_index
        )

        print(f"  Pose in chunk {chunk_idx + 1}: {pose_current is not None}")
        print(f"  Pose in chunk {chunk_idx}: {pose_prev is not None}")

        if pose_current is not None and pose_prev is not None:
            print(f"  Chunk {chunk_idx + 1} pose:\n{pose_current}")
            print(f"  Chunk {chunk_idx} pose:\n{pose_prev}")

            # Transform from current chunk to previous chunk: T_current_to_prev = pose_current @ inv(pose_prev)
            # Since both are c2w poses in their respective chunk coordinate systems
            T_current_to_prev = pose_prev
            print(f"  Transform from chunk {chunk_idx + 1} to chunk {chunk_idx}:\n{T_current_to_prev}")

            # Cumulative transform: current_to_ref = prev_to_ref @ current_to_prev
            T_current_to_ref = cumulative_transforms[chunk_idx-1] @ T_current_to_prev
            cumulative_transforms.append(T_current_to_ref)
            print(f"  Cumulative transform to reference:\n{T_current_to_ref}")
        else:
            print(f"  Warning: Could not find ref view poses for chunk {chunk_idx + 1}")
            print(f"  Using previous chunk's transform for chunk {chunk_idx + 1}")
            cumulative_transforms.append(cumulative_transforms[-1])

    # Apply transformations to all chunks
    aligned_predictions = []
    for chunk_idx, (predictions, T_chunk_to_ref) in enumerate(zip(predictions_list, cumulative_transforms)):
        print(f"\nApplying transform to chunk {chunk_idx + 1}...")

        # Apply transformation to point cloud
        chunk_predictions = []
        for pred in predictions:
            if 'pts3d' in pred:
                pts_local = pred['pts3d'].cpu().numpy()
                pts_aligned = transform_point_cloud(pts_local, T_chunk_to_ref)
                pred_copy = pred.copy()
                pred_copy['pts3d'] = torch.from_numpy(pts_aligned).to(pred['pts3d'].device)
                chunk_predictions.append(pred_copy)
            else:
                chunk_predictions.append(pred)

        aligned_predictions.append(chunk_predictions)

    return aligned_predictions


def save_chunk_visualizations(predictions_list, output_dir, sequence_name, chunks,
                           depth_validity_check=True, conf_threshold=0.0, max_points=-1):
    """
    Save visualization results for each chunk separately.

    Args:
        predictions_list: List of aligned predictions for each chunk
        output_dir: Output directory path
        sequence_name: Name of the sequence for file naming
        chunks: List of chunk index lists
        depth_validity_check: Whether to check depth validity (> 0)
        conf_threshold: Confidence threshold for filtering (0.0 = no filtering)
        max_points: Maximum number of points to keep (-1 = no limit)
    """
    os.makedirs(output_dir, exist_ok=True)

    for chunk_idx, (predictions, chunk_indices) in enumerate(zip(predictions_list, chunks)):
        print(f"Saving chunk {chunk_idx + 1} with {len(chunk_indices)} indices...")

        # Collect all 3D points, colors, and camera poses for this chunk
        all_points = []
        all_colors = []

        for pred in predictions:
            # Get world points (already aligned)
            pts3d = pred['pts3d'].cpu().numpy()  # (H, W, 3)
            mask = pred['mask'].squeeze(-1).cpu().numpy()  # (H, W)

            # Apply additional filtering
            valid_mask = mask > 0.5  # Start with model mask

            # Depth validity check
            if depth_validity_check and 'depth_z' in pred:
                depth_z = pred['depth_z'].cpu().numpy().squeeze(-1)
                depth_valid = depth_z > 0
                valid_mask = valid_mask & depth_valid

            # Confidence threshold
            if conf_threshold > 0.0 and 'conf' in pred:
                conf = pred['conf'].cpu().numpy()
                conf_valid = conf >= conf_threshold
                valid_mask = valid_mask & conf_valid

            # Point limit (random sampling)
            if max_points > 0 and np.sum(valid_mask) > max_points:
                valid_mask = randomly_limit_trues(valid_mask, max_points)

            # Get valid points and colors
            valid_points = pts3d[valid_mask]

            # Get colors from denormalized image
            img_no_norm = pred['img_no_norm'].cpu().numpy()  # (H, W, 3)
            valid_colors = img_no_norm[valid_mask]  # (N, 3)
            valid_colors = (valid_colors * 255).astype(np.uint8)

            all_points.append(valid_points)
            all_colors.append(valid_colors)

        # Combine all points and colors for this chunk
        if all_points:
            combined_points = np.concatenate(all_points, axis=0)
            combined_colors = np.concatenate(all_colors, axis=0)

            # Save PLY point cloud
            ply_path = os.path.join(output_dir, f"{sequence_name}_chunk{chunk_idx+1:02d}_pointcloud.ply")
            print(f"Saving PLY point cloud to: {ply_path}")

            point_cloud = trimesh.PointCloud(vertices=combined_points, colors=combined_colors)
            point_cloud.export(ply_path)

            print(f"Saved {len(combined_points)} points for chunk {chunk_idx + 1}")
        else:
            print(f"No valid points found for chunk {chunk_idx + 1}!")


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
    parser = argparse.ArgumentParser(description="Run MapAnything on KITTI sequences with chunking and alignment")
    parser.add_argument(
        "--sequence_dir",
        type=str,
        required=True,
        help="Path to KITTI sequence directory containing images"
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
        default=['image'],
        choices=['image', 'intrinsics', 'poses', 'depth', 'ray_directions'],
        help="Modalities to use (default: image)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of images per chunk (default: 100)"
    )
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=50,
        help="Number of overlapping images between chunks (default: 50)"
    )
    parser.add_argument(
        "--downsample_rate",
        type=int,
        default=1,
        help="Downsample rate: take every Nth image (default: 1)"
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
        help="Use memory efficient inference for large sequences"
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
    print(f"Chunk size: {args.chunk_size}, Overlap size: {args.overlap_size}")
    print(f"Downsample rate: {args.downsample_rate}")

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get image paths with downsampling
    print(f"Loading KITTI sequence from: {args.sequence_dir}")
    image_paths = get_image_paths(args.sequence_dir, args.downsample_rate)
    print(f"Found {len(image_paths)} images after downsampling")

    if len(image_paths) == 0:
        print("No images found in the sequence directory!")
        sys.exit(1)

    # Get sequence name from directory
    sequence_name = os.path.basename(args.sequence_dir)

    # Create indices for all images
    all_indices = list(range(len(image_paths)))

    # Create overlapping chunks
    chunks = create_overlapping_chunks(all_indices, args.chunk_size, args.overlap_size)
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} images (indices {chunk[0]} to {chunk[-1]})")

    # Initialize MapAnything model
    print("Loading MapAnything model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    # Update data_norm_type in views to match model
    data_norm_type = model.encoder.data_norm_type
    print(f"Using data normalization type: {data_norm_type}")

    # Process each chunk
    predictions_list = []
    for chunk_idx, chunk_indices in enumerate(chunks):
        print(f"\nProcessing chunk {chunk_idx + 1}/{len(chunks)}...")

        # Prepare views for this chunk
        views = prepare_views_for_chunk(
            image_paths, chunk_indices, not args.no_metric_scale
        )

        if len(views) == 0:
            print(f"No views found for chunk {chunk_idx + 1}, skipping...")
            predictions_list.append([])
            continue

        # Update data_norm_type for views
        for view in views:
            view['data_norm_type'] = [data_norm_type]

        # Run inference on this chunk
        predictions = run_inference_on_chunk(model, views, not args.no_memory_efficient)
        predictions_list.append(predictions)

    print(f"\nCompleted inference on {len(chunks)} chunks")

    # Align all point clouds to the first chunk's coordinate system
    print("\nAligning point clouds to reference coordinate system...")
    aligned_predictions = align_point_clouds_to_reference(predictions_list, chunks)

    # Save visualizations for each chunk separately
    save_chunk_visualizations(
        aligned_predictions, args.output_dir, sequence_name, chunks,
        not args.no_depth_validity_check, args.conf_threshold, args.max_points
    )

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()
