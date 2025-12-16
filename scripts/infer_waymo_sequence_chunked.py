#!/usr/bin/env python3
"""
Script to run MapAnything inference on Waymo multi-camera sequences with chunking and alignment.

This script processes a Waymo sequence by dividing it into overlapping chunks, running inference
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
import rerun as rr

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import predictions_to_glb, script_add_rerun_args

from time import time

# Import the WaymoDatasetReader
waymo_preprocessing_path = "/wekafs/ict/junyiouy/waymo_preprocessing"
dataset_reader_path = Path(waymo_preprocessing_path) / "dataset_reader.py"

if not dataset_reader_path.exists():
    print(f"Error: dataset_reader.py not found at {dataset_reader_path}")
    sys.exit(1)

# Add the waymo_preprocessing path to sys.path
sys.path.insert(0, waymo_preprocessing_path)

try:
    from dataset_reader import WaymoDatasetReader
except ImportError as e:
    print(f"Error: Could not import WaymoDatasetReader: {e}")
    print(f"Make sure waymo_preprocessing is set up correctly at {waymo_preprocessing_path}")
    sys.exit(1)


class WaymoToOpenCVConverter:
    """Convert Waymo coordinate system to OpenCV coordinate system."""

    def __init__(self):
        # Waymo(OpenGL) to OpenCV transformation matrix
        # OpenCV: +X right, +Y down, +Z forward
        # Waymo: +X front, +Y left, +Z up
        self.coord_transform = torch.tensor([
            [1,  0,  0, 0],  # OpenCV X(right) = Waymo X(front)
            [0,  -1, 0, 0],  # OpenCV Y(down) = -Waymo Z(up)
            [0,  0,  -1, 0],  # OpenCV Z(front) = -Waymo Y(left)
            [0,  0,  0, 1]
        ], dtype=torch.float32)

    def waymo_to_opencv_transform(self):
        """Get the transformation matrix from Waymo to OpenCV coordinates."""
        return self.coord_transform

    def convert_pose_waymo_to_opencv(self, waymo_pose):
        """
        Convert a pose from Waymo coordinates to OpenCV coordinates.

        Args:
            waymo_pose: 4x4 transformation matrix in Waymo coordinates

        Returns:
            4x4 transformation matrix in OpenCV coordinates
        """
        waymo_pose = torch.from_numpy(waymo_pose).float()
        coord_transform = self.waymo_to_opencv_transform()
        # opencv_pose = coord_transform @ waymo_pose @ torch.linalg.inv(coord_transform)
        opencv_pose = waymo_pose @ torch.linalg.inv(coord_transform)
        return opencv_pose.numpy()


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


def prepare_views_for_chunk(reader, camera_types, converter, modalities, chunk_timestamps, metric_scale=True):
    """
    Prepare all views for a specific chunk of timestamps.

    Args:
        reader: WaymoDatasetReader instance
        camera_types: List of camera names to use
        converter: WaymoToOpenCVConverter instance
        modalities: List of modalities to include
        chunk_timestamps: List of timestamps for this chunk
        metric_scale: Whether inputs are in metric scale

    Returns:
        list: List of view dictionaries for MapAnything's preprocess_inputs
    """
    views = []

    # Group frames by timestamp for this chunk
    frames_by_timestamp = {}
    for frame in reader.get_frames():
        if frame['timestamp'] in chunk_timestamps:
            timestamp = frame['timestamp']
            if timestamp not in frames_by_timestamp:
                frames_by_timestamp[timestamp] = []
            frames_by_timestamp[timestamp].append(frame)

    # Process each timestamp in the chunk
    for timestamp in sorted(chunk_timestamps):
        if timestamp not in frames_by_timestamp:
            continue

        timestamp_frames = frames_by_timestamp[timestamp]

        # Filter frames by camera types
        camera_frames = {frame['camera']: frame for frame in timestamp_frames
                        if frame['camera'] in camera_types}

        # Process each camera for this timestamp
        for camera_name in camera_types:
            if camera_name not in camera_frames:
                print(f"Warning: Camera {camera_name} not found for timestamp {timestamp}")
                continue

            frame = camera_frames[camera_name]

            # Load image as PIL Image (preprocess_inputs will handle resizing)
            image_path = reader.get_image_path(frame)
            image = load_image_for_preprocessing(image_path)

            # Create base view dictionary
            view = {
                'img': image,  # PIL Image
                'data_norm_type': ['dinov2'],  # Will be updated to match model
                'is_metric_scale': torch.tensor([metric_scale]),
            }

            # Add modalities based on selection
            if 'intrinsics' in modalities:
                intrinsics = reader.get_camera_intrinsics(camera_name)['intrinsic_matrix'].astype(np.float32)
                view['intrinsics'] = intrinsics

            if 'poses' in modalities:
                c2w_pose = reader.get_camera_to_world_pose(frame, camera_name)
                c2w_pose = converter.convert_pose_waymo_to_opencv(c2w_pose).astype(np.float32)
                view['camera_poses'] = c2w_pose

            # Note: depth and ray_directions not implemented yet as Waymo doesn't provide depth
            if 'depth' in modalities:
                print("Warning: Depth modality requested but Waymo dataset doesn't provide depth data")
            if 'ray_directions' in modalities:
                print("Warning: Ray directions modality requested but not implemented for Waymo")

            views.append(view)

    return views


def create_overlapping_chunks(all_timestamps, chunk_size, overlap_size):
    """
    Create overlapping chunks from a list of timestamps.

    Args:
        all_timestamps: Sorted list of all timestamps
        chunk_size: Number of timestamps per chunk
        overlap_size: Number of overlapping timestamps between chunks

    Returns:
        list: List of chunk timestamp lists
    """
    chunks = []
    step_size = chunk_size - overlap_size

    for start_idx in range(0, len(all_timestamps), step_size):
        end_idx = min(start_idx + chunk_size, len(all_timestamps))
        chunk_timestamps = all_timestamps[start_idx:end_idx]
        chunks.append(chunk_timestamps)

        # Stop if we've reached the end
        if end_idx == len(all_timestamps):
            break

    return chunks


def run_inference_on_chunk(model, views, memory_efficient=False,confidence_percentile=0.0):
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
        confidence_percentile=confidence_percentile,
    )
    t1 = time()
    print(f"Inference time: {t1 - t0:.2f} seconds for {len(views)} views")

    return predictions


def find_ref_view_pose_in_chunk(chunk_predictions, chunk_timestamps, target_timestamp, target_camera):
    """
    Find the pose of a specific view (timestamp + camera) in chunk predictions.

    Args:
        chunk_predictions: Predictions from the chunk
        chunk_timestamps: Timestamps for the chunk
        target_timestamp: Target timestamp to find
        target_camera: Target camera name

    Returns:
        numpy.ndarray: 4x4 pose matrix, or None if not found
    """
    if target_timestamp not in chunk_timestamps:
        return None

    # Find the index of the target timestamp in the chunk
    timestamp_idx = chunk_timestamps.index(target_timestamp)

    if timestamp_idx >= len(chunk_predictions):
        return None

    pred = chunk_predictions[timestamp_idx]

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


def align_point_clouds_to_reference(predictions_list, chunks, reader, camera_types, converter):
    """
    Align all chunk point clouds to the first chunk's coordinate system.

    For each chunk N (N > 0):
    - Find its ref view (first timestamp, first camera) pose in chunk N
    - Find the same view pose in chunk N-1
    - Compute transform: T = pose_in_chunkN @ inv(pose_in_chunkN-1)
    - Accumulate transforms

    Note: prediction['camera_poses'] are c2w poses relative to each chunk's reference view.

    Args:
        predictions_list: List of predictions for each chunk
        chunks: List of chunk timestamp lists
        reader: WaymoDatasetReader instance
        camera_types: List of camera names
        converter: WaymoToOpenCVConverter instance

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

        # Get ref view of current chunk (first timestamp, first camera)
        current_chunk_timestamps = chunks[chunk_idx]
        ref_timestamp = current_chunk_timestamps[0]
        ref_camera = camera_types[0]  # Use first camera type

        print(f"  Ref view: timestamp={ref_timestamp}, camera={ref_camera}")

        # Find pose of ref view in current chunk
        pose_current = find_ref_view_pose_in_chunk(
            predictions_list[chunk_idx], current_chunk_timestamps,
            ref_timestamp, ref_camera
        )

        # Find pose of same view in previous chunk
        prev_chunk_timestamps = chunks[chunk_idx-1]
        pose_prev = find_ref_view_pose_in_chunk(
            predictions_list[chunk_idx-1], prev_chunk_timestamps,
            ref_timestamp, ref_camera
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


def voxelize_points_and_get_indices(points: np.ndarray, grid_size: tuple[int, int, int]) -> np.ndarray:
    """
    将点云体素化（划分到 3D BBox 网格）并返回每个点的 BBox 索引。
    
    Args:
        points: (N, 3) float array of points.
        grid_size: (Gx, Gy, Gz) tuple defining the grid dimensions.
        
    Returns:
        (N,) int array of voxel indices.
    """
    if points.size == 0:
        return np.array([], dtype=int)
        
    # 1. 计算点云的边界
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    span = max_coords - min_coords
    
    # 避免除以零，对 span=0 的维度进行处理
    span[span == 0] = 1.0 
    
    # 2. 归一化到 [0, 1) 范围
    # 增加一个小 epsilon 避免归一化后结果为 1.0 导致索引越界
    epsilon = 1e-6 
    normalized_points = (points - min_coords) / span
    normalized_points = np.clip(normalized_points, 0.0, 1.0 - epsilon)
    
    # 3. 映射到 BBox 索引 [0, G-1]
    bbox_indices_3d = np.floor(normalized_points * np.array(grid_size)).astype(int)
    
    # 4. 转换为单个 BBox 索引 (index = x + Gx*y + Gx*Gy*z)
    Gx, Gy, Gz = grid_size
    bbox_indices_1d = (
        bbox_indices_3d[:, 0] + 
        Gx * bbox_indices_3d[:, 1] + 
        Gx * Gy * bbox_indices_3d[:, 2]
    )
    
    return bbox_indices_1d


def log_view_to_rerun(pred, view_idx, chunk_idx, timestamp, camera_name):
    """
    Log individual view data to Rerun for visualization, following ScanNet++ pattern.

    Args:
        pred: Prediction dictionary for this view
        view_idx: Index of this view within the chunk
        chunk_idx: Index of this chunk
        timestamp: Timestamp of this view
        camera_name: Camera name (e.g., 'FRONT')
    """
    if 'pts3d' not in pred or 'camera_poses' not in pred:
        return

    # Set time sequence for this view
    rr.set_time("stable_time", sequence=view_idx)

    # Get points and colors
    pts3d = pred['pts3d'].cpu().numpy().squeeze(0)  # (H, W, 3)
    mask = pred['mask'].squeeze(-1).cpu().numpy().squeeze(0)  # (H, W)
    valid_mask = mask > 0.5

    if not np.any(valid_mask):
        return

    valid_points = pts3d[valid_mask]

    # Get colors
    img_no_norm = pred['img_no_norm'].cpu().numpy().squeeze(0)  # (H, W, 3)
    valid_colors = img_no_norm[valid_mask]  # (N, 3)
    valid_colors = (valid_colors * 255).astype(np.uint8)

    # Get camera pose
    camera_pose = pred['camera_poses'][0].cpu().numpy()  # 4x4 matrix

    # Create entity path for this view
    entity_path = f"chunk_{chunk_idx+1:02d}/view_{camera_name}_{timestamp}"

    # Log point cloud for this view
    rr.log(
        f"{entity_path}/pointcloud",
        rr.Points3D(
            valid_points,
            colors=valid_colors,
            radii=0.05,  # Smaller radius for individual views
        ),
    )

    # Log camera transform
    rr.log(
        f"{entity_path}/camera/pinhole",
        rr.Transform3D(
            translation=camera_pose[:3, 3],
            mat3x3=camera_pose[:3, :3],
        ),
    )

    # Log camera pinhole and frustum if intrinsics are available
    if 'intrinsics' in pred:
        intrinsics = pred['intrinsics'].squeeze(0).detach().cpu().numpy()  # 3x3 matrix
        H, W = pts3d.shape[0], pts3d.shape[1]  # Use depth map dimensions

        # Log pinhole camera for image plane visualization
        rr.log(
            f"{entity_path}/camera/pinhole",
            rr.Pinhole(
                image_from_camera=intrinsics,
                height=H,
                width=W,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=1.
            ),
        )

        # Log RGB image if available
        if 'img_no_norm' in pred:
            rr.log(
                f"{entity_path}/camera/pinhole",
                rr.Image(pred['img_no_norm']),
            )

    print(f"Logged view {camera_name}_{timestamp} with {len(valid_points)} points")


def log_chunk_to_rerun(combined_points, combined_colors, chunk_idx, max_points=-1):
    """
    Log final (sampled) chunk data to Rerun for visualization.

    Args:
        combined_points: (N, 3) numpy array of points (already sampled/aligned).
        combined_colors: (N, 3) numpy array of colors (already sampled/aligned).
        chunk_idx: Index of this chunk.
        max_points: Max points to log (applied only if combined_points is too large).
    """
    print(f"Logging chunk {chunk_idx + 1} to Rerun...")

    # Set time sequence for this chunk
    rr.set_time("chunk", sequence=chunk_idx)

    points_to_log = combined_points
    colors_to_log = combined_colors

    # Log point cloud to Rerun
    rr.log(
        f"chunk_{chunk_idx+1:02d}/pointcloud",
        rr.Points3D(
            points_to_log,
            colors=colors_to_log,
            radii=0.01,  # Small radius for visibility
        ),
    )

    print(f"Logged {len(points_to_log)} points for chunk {chunk_idx + 1} (after Rerun-specific sampling).")


def save_chunk_visualizations(predictions_list, output_dir, sequence_name, chunks, camera_types,
                           depth_validity_check=True, conf_threshold=0.0, max_points=-1, bbox_grid_size=None, viz_rerun=False):
    """
    Save visualization results for each chunk separately.

    Args:
        predictions_list: List of aligned predictions for each chunk
        output_dir: Output directory path
        sequence_name: Name of the sequence for file naming
        chunks: List of chunk timestamp lists
        camera_types: List of camera types used
        depth_validity_check: Whether to check depth validity (> 0)
        conf_threshold: Confidence threshold for filtering (0.0 = no filtering)
        max_points: Maximum number of points to keep (-1 = no limit). *Retained for Rerun logging limit.*
        bbox_grid_size: Tuple (Gx, Gy, Gz) for density-based sampling. None to disable.
        viz_rerun: Whether to log data to Rerun
    """
    os.makedirs(output_dir, exist_ok=True)

    for chunk_idx, (predictions, chunk_timestamps) in enumerate(zip(predictions_list, chunks)):
        print(f"Saving chunk {chunk_idx + 1} with {len(chunk_timestamps)} timestamps...")

        # Collect all 3D points, colors, and camera poses for this chunk
        all_points = []
        all_colors = []

        for pred_idx, pred in enumerate(predictions):
            if 'pts3d' not in pred:
                continue
                
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

            # --- NOTE: Simple max_points sampling is NOT applied here, only filters are applied ---

            # Get valid points and colors
            valid_points = pts3d[valid_mask]

            # Get colors from denormalized image
            img_no_norm = pred['img_no_norm'].cpu().numpy()  # (H, W, 3)
            valid_colors = img_no_norm[valid_mask]  # (N, 3)
            valid_colors = (valid_colors * 255).astype(np.uint8)

            all_points.append(valid_points)
            all_colors.append(valid_colors)

            # Log individual view to Rerun (Only if Rerun is enabled)
            if viz_rerun:
                # Determine camera name and timestamp for this prediction
                timestamp_idx = pred_idx // len(camera_types)
                camera_idx = pred_idx % len(camera_types)

                if timestamp_idx < len(chunk_timestamps) and camera_idx < len(camera_types):
                    timestamp = chunk_timestamps[timestamp_idx]
                    camera_name = camera_types[camera_idx]

                    log_view_to_rerun(pred, pred_idx, chunk_idx, timestamp, camera_name)


        # Combine all points and colors for this chunk
        if all_points:
            combined_points = np.concatenate(all_points, axis=0)
            combined_colors = np.concatenate(all_colors, axis=0)
            
            # --- START: Density-Based Sampling Logic (Applied to file save and Rerun) ---
            if bbox_grid_size is not None and len(combined_points) > 0:
                print(f"Applying density-based sampling with grid size: {bbox_grid_size}...")

                # 1. Voxelization
                bbox_indices_1d = voxelize_points_and_get_indices(combined_points, bbox_grid_size)

                # 2. Count points per occupied bbox (exclude empty bboxes)
                unique_indices, counts = np.unique(bbox_indices_1d, return_counts=True)

                # Only consider bboxes that have points
                occupied_counts = counts[counts > 0]  # Exclude empty bboxes

                # if occupied_counts.size > 0:
                #     median_count = int(np.median(occupied_counts))
                # else:
                #     median_count = 0

                # median_count = 200000//max(bbox_grid_size)
                median_count = 200000000  # 保留所有点

                if median_count > 0:
                    print(f"  Total initial points: {len(combined_points)}")
                    print(f"  Number of occupied BBoxes: {len(unique_indices)}")
                    print(f"  Median point count among occupied BBoxes (Threshold M): {median_count}")

                    # 3. Conditional Downsampling Preparation
                    bbox_to_original_indices = defaultdict(list)
                    for original_idx, bbox_idx in enumerate(bbox_indices_1d):
                        bbox_to_original_indices[bbox_idx].append(original_idx)

                    final_retained_indices_list = []

                    # 4. Conditional Downsampling - only for occupied bboxes
                    for bbox_idx, count in zip(unique_indices, counts):
                        original_indices = bbox_to_original_indices[bbox_idx]
                        N_i = len(original_indices)  # 当前 BBox 的点数

                        if N_i <= median_count:
                            # 点数不足或等于中位数：保留所有点
                            retained_indices = original_indices
                        else:
                            # 点数过多：随机下采样到中位数 M
                            target_count = median_count

                            # 随机选择 target_count 个点的原始索引
                            retained_indices = np.random.choice(
                                original_indices,
                                size=target_count,
                                replace=False
                            ).tolist()

                        final_retained_indices_list.extend(retained_indices)

                    final_retained_indices = np.array(final_retained_indices_list)

                    # 5. 生成最终的、下采样后的点云
                    combined_points = combined_points[final_retained_indices]
                    combined_colors = combined_colors[final_retained_indices]
                    print(f"  Points after density-based sampling: {len(combined_points)}")
                else:
                    print("  No occupied bboxes found, no density sampling applied.")
            # --- END: Density-Based Sampling Logic ---

            # Save PLY point cloud
            ply_path = os.path.join(output_dir, f"{sequence_name}_chunk{chunk_idx+1:02d}_pointcloud.ply")
            print(f"Saving PLY point cloud to: {ply_path}")

            point_cloud = trimesh.PointCloud(vertices=combined_points, colors=combined_colors)
            point_cloud.export(ply_path)

            print(f"Saved {len(combined_points)} points for chunk {chunk_idx + 1}")
            
            # # --- RERUN LOGGING (使用经过密度采样的点云) ---
            # if viz_rerun:
            #      log_chunk_to_rerun(combined_points, combined_colors, chunk_idx, max_points)

        else:
            print(f"No valid points found for chunk {chunk_idx + 1}!")


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    
    NOTE: This function is deprecated for file saving but retained for Rerun visualization.
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
    parser = argparse.ArgumentParser(description="Run MapAnything on Waymo sequences with chunking and alignment")
    script_add_rerun_args(parser)
    parser.add_argument(
        "--sequence_dir",
        type=str,
        required=True,
        help="Path to Waymo sequence directory (containing transforms.json)"
    )
    parser.add_argument(
        "--camera_types",
        type=str,
        nargs='+',
        default=['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT'],
        help="Camera types to use (default: all 5 cameras)"
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
        "--chunk_size",
        type=int,
        default=100,
        help="Number of timestamps per chunk (default: 100)"
    )
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=50,
        help="Number of overlapping timestamps between chunks (default: 50)"
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
        help="Maximum number of points to keep (-1 for no limit, default: -1). Primarily used to limit Rerun visualization size."
    )
    parser.add_argument(
        "--bbox_grid_size",
        type=int,
        nargs=3,
        default=None,
        help="3D grid size (Gx Gy Gz) for density-based sampling. If provided (e.g., 10 10 10), it applies point density control before saving."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process (-1 for all frames, default: -1)"
    )
    parser.add_argument(
        "--downsample_rate",
        type=int,
        default=1,
        help="Downsample rate for frames (default: 1, no downsampling)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Rerun if visualization is enabled
    if args.connect or args.save or args.stdout:
        rr.init(f"MapAnything_Waymo_{os.path.basename(args.sequence_dir)}")
        if args.connect:
            rr.spawn()
        if args.save:
            rr.save(args.save)
        if args.stdout:
            # For stdout, Rerun doesn't have direct support in current version
            # You might need to implement custom logging or use save to temp file
            pass

    # Validate modalities
    validate_modalities(args.modalities)
    print(f"Using modalities: {args.modalities}")
    print(f"Metric scale: {not args.no_metric_scale}")
    print(f"Chunk size: {args.chunk_size}, Overlap size: {args.overlap_size}")
    if args.bbox_grid_size:
        print(f"Density Sampling Grid: {args.bbox_grid_size}")
    else:
        print("Density Sampling: Disabled")

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Waymo dataset reader
    print(f"Loading Waymo sequence from: {args.sequence_dir}")
    reader = WaymoDatasetReader(args.sequence_dir)

    # Initialize coordinate converter
    converter = WaymoToOpenCVConverter()

    # Get sequence name from directory
    sequence_name = os.path.basename(args.sequence_dir)

    # Get all timestamps
    all_frames = reader.get_frames()

    # downsample_rate
    if args.downsample_rate > 1:
        all_frames = all_frames[::args.downsample_rate]
        print(f"Downsampled frames by rate {args.downsample_rate}, total frames: {len(all_frames)}")

    # Get only the first args.max_frames frames if specified
    if args.max_frames > 0:
        all_frames = all_frames[:args.max_frames]

    all_timestamps = sorted(list(set(frame['timestamp'] for frame in all_frames)))
    print(f"Total timestamps in sequence: {len(all_timestamps)}")

    # Create overlapping chunks
    chunks = create_overlapping_chunks(all_timestamps, args.chunk_size, args.overlap_size)
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} timestamps (from {chunk[0]} to {chunk[-1]})")

    # Initialize MapAnything model
    print("Loading MapAnything model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    # Update data_norm_type in views to match model
    data_norm_type = model.encoder.data_norm_type
    print(f"Using data normalization type: {data_norm_type}")

    # Process each chunk
    predictions_list = []
    for chunk_idx, chunk_timestamps in enumerate(chunks):
        print(f"\nProcessing chunk {chunk_idx + 1}/{len(chunks)}...")

        # Prepare views for this chunk
        views = prepare_views_for_chunk(
            reader, args.camera_types, converter, args.modalities,
            chunk_timestamps, not args.no_metric_scale
        )

        if len(views) == 0:
            print(f"No views found for chunk {chunk_idx + 1}, skipping...")
            predictions_list.append([])
            continue

        # Update data_norm_type for views
        for view in views:
            view['data_norm_type'] = [data_norm_type]

        # Run inference on this chunk
        predictions = run_inference_on_chunk(model, views, not args.no_memory_efficient, confidence_percentile=args.conf_threshold)
        predictions_list.append(predictions)

    print(f"\nCompleted inference on {len(chunks)} chunks")

    # Align all point clouds to the first chunk's coordinate system
    print("\nAligning point clouds to reference coordinate system...")
    aligned_predictions = align_point_clouds_to_reference(
        predictions_list, chunks, reader, args.camera_types, converter
    )

    # Save visualizations for each chunk separately
    bbox_size = tuple(args.bbox_grid_size) if args.bbox_grid_size else None
    save_chunk_visualizations(
        aligned_predictions, args.output_dir, sequence_name, chunks, args.camera_types,
        not args.no_depth_validity_check, args.conf_threshold, args.max_points,
        bbox_grid_size=bbox_size,
        viz_rerun=args.connect or args.save or args.stdout
    ) 

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()
# python /wekafs/ict/junyiouy/map-anything/scripts/infer_waymo_sequence_chunked.py --sequence_dir /wekafs/ict/junyiouy/waymo_preprocessing/waymo_ns/1172406780360799916_1660_000_1680_000   --camera_types FRONT --output_dir /wekafs/ict/junyiouy/map-anything/waymo_output/single_view/waymo_output_image_no_metric_chunk_60      --no_metric_scale --max_points 30000      --modalities image  --chunk_size 60 --conf_threshold 0.2  --overlap_size 1

# python /wekafs/ict/junyiouy/map-anything/scripts/infer_waymo_sequence_chunked.py --sequence_dir /wekafs/ict/junyiouy/SCube-release/waymo_ns_test/testing/3341890853207909601_1020_000_1040_000   --camera_types FRONT --output_dir /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object      --no_metric_scale --max_points 30000      --modalities image  --chunk_size 60 --conf_threshold 0.2  --overlap_size 1

# python /wekafs/ict/junyiouy/map-anything/scripts/infer_waymo_sequence_chunked.py \
#   --sequence_dir /wekafs/ict/junyiouy/SCube-release/waymo_ns_test/testing/3341890853207909601_1020_000_1040_000 \
#   --camera_types FRONT \
#   --output_dir /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short \
#   --no_metric_scale --max_points 30000 --modalities image \
#   --chunk_size 60 --conf_threshold 0.2 --overlap_size 1 \
#   --save /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short/waymo_sequence.rrd \
#   --max_frames 40

# python /wekafs/ict/junyiouy/map-anything/scripts/infer_waymo_sequence_chunked.py \
#   --sequence_dir /wekafs/ict/junyiouy/SCube-release/waymo_ns_test/testing/3341890853207909601_1020_000_1040_000 \
#   --camera_types FRONT \
#   --output_dir /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive \
#   --no_metric_scale --max_points 30000 --modalities image \
#   --chunk_size 60 --conf_threshold 0.0 --overlap_size 1 \
#   --save /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive/waymo_sequence.rrd \
#   --max_frames 10

# python /wekafs/ict/junyiouy/map-anything/scripts/infer_waymo_sequence_chunked.py \
#   --sequence_dir /wekafs/ict/junyiouy/SCube-release/waymo_ns_test/testing/3341890853207909601_1020_000_1040_000 \
#   --camera_types FRONT \
#   --output_dir /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params \
#   --no_metric_scale --max_points 30000 --modalities image intrinsics poses\
#   --chunk_size 60 --conf_threshold 0.0 --overlap_size 1 \
#   --save /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params/waymo_sequence.rrd \
#   --max_frames 10


# python /wekafs/ict/junyiouy/map-anything/scripts/infer_waymo_sequence_chunked.py \
#   --sequence_dir /wekafs/ict/junyiouy/SCube-release/waymo_ns_test/testing/3341890853207909601_1020_000_1040_000 \
#   --camera_types FRONT \
#   --output_dir /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params_w_metric_scale \
#   --max_points 30000 --modalities image intrinsics poses\
#   --chunk_size 60 --conf_threshold 0.0 --overlap_size 1 \
#   --save /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params_w_metric_scale/waymo_sequence.rrd \
#   --max_frames 10


# python /wekafs/ict/junyiouy/map-anything/scripts/infer_waymo_sequence_chunked.py \
#   --sequence_dir /wekafs/ict/junyiouy/SCube-release/waymo_ns_test/testing/5585555620508986875_720_000_740_000 \
#   --camera_types FRONT \
#   --output_dir /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params_w_metric_scale_1 \
#   --max_points 30000 --modalities image intrinsics poses\
#   --chunk_size 60 --conf_threshold 0.0 --overlap_size 1 \
#   --save /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params_w_metric_scale_1/waymo_sequence.rrd \
#   --max_frames 10

# python /wekafs/ict/junyiouy/map-anything/scripts/infer_waymo_sequence_chunked.py \
#   --sequence_dir /wekafs/ict/junyiouy/SCube-release/waymo_ns_test/testing/2942662230423855469_880_000_900_000 \
#   --camera_types FRONT \
#   --output_dir /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params_w_metric_scale_2 \
#   --max_points 30000 --modalities image intrinsics poses\
#   --chunk_size 60 --conf_threshold 0.0 --overlap_size 1 \
#   --save /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params_w_metric_scale_2/waymo_sequence.rrd \
#   --max_frames 10

# python /wekafs/ict/junyiouy/map-anything/scripts/infer_waymo_sequence_chunked.py \
#   --sequence_dir /wekafs/ict/junyiouy/SCube-release/waymo_ns_test/testing/2942662230423855469_880_000_900_000 \
#   --camera_types FRONT \
#   --output_dir /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params_w_metric_scale_3 \
#   --max_points 30000 --modalities image intrinsics poses\
#   --chunk_size 60 --conf_threshold 0.0 --overlap_size 1 \
#   --save /wekafs/ict/junyiouy/map-anything/waymo_output/dynamic_long_object_short_adaptive_w_params_w_metric_scale_3/waymo_sequence.rrd \
#   --max_frames 60 --downsample_rate 2
