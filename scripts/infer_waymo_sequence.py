#!/usr/bin/env python3
"""
Script to run MapAnything inference on Waymo multi-camera sequences.

This script processes an entire Waymo sequence at once, using all timestamps
and specified cameras to perform multi-view 3D reconstruction.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from mapanything.models import MapAnything, init_model
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import predictions_to_glb

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
        # Waymo to OpenCV transformation matrix
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


def prepare_views_for_sequence(reader, camera_types, converter, modalities, metric_scale=True):
    """
    Prepare all views for the entire sequence in format expected by preprocess_inputs.

    Args:
        reader: WaymoDatasetReader instance
        camera_types: List of camera names to use
        converter: WaymoToOpenCVConverter instance
        modalities: List of modalities to include
        metric_scale: Whether inputs are in metric scale

    Returns:
        list: List of view dictionaries for MapAnything's preprocess_inputs
    """
    views = []

    # Get all frames
    all_frames = reader.get_frames()

    # Group frames by timestamp
    frames_by_timestamp = {}
    for frame in all_frames:
        timestamp = frame['timestamp']
        if timestamp not in frames_by_timestamp:
            frames_by_timestamp[timestamp] = []
        frames_by_timestamp[timestamp].append(frame)

    # Process each timestamp
    for timestamp, timestamp_frames in sorted(frames_by_timestamp.items()):
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
    print(f"Running inference on {len(views)} views...")

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
        # ignore_pose_scale_inputs=True,
        confidence_percentile=20,
    )
    t1 = time()
    print(f"Inference time: {t1 - t0:.2f} seconds for {len(views)} views")

    print("Inference completed!")
    return predictions


def save_visualizations(predictions, output_dir, sequence_name,
                       depth_validity_check=True, conf_threshold=0.0, max_points=-1):
    """
    Save visualization results: PLY point cloud and camera poses.

    Args:
        predictions: List of prediction dictionaries from MapAnything
        output_dir: Output directory path
        sequence_name: Name of the sequence for file naming
        depth_validity_check: Whether to check depth validity (> 0)
        conf_threshold: Confidence threshold for filtering (0.0 = no filtering)
        max_points: Maximum number of points to keep (-1 = no limit)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all 3D points, colors, and camera poses
    all_points = []
    all_colors = []
    all_camera_poses = []

    for pred in predictions:
        # Get world points
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

        # Get camera intrinsics and poses
        intrinsics = pred['intrinsics'].cpu().numpy()  # (3, 3)
        camera_pose = pred['camera_poses'].cpu().numpy()  # (4, 4)

        all_camera_poses.append({
            'intrinsics': intrinsics,
            'camera_pose': camera_pose
        })

    # Combine all points and colors
    if all_points:
        combined_points = np.concatenate(all_points, axis=0)
        combined_colors = np.concatenate(all_colors, axis=0)

        # Save PLY point cloud
        ply_path = os.path.join(output_dir, f"{sequence_name}_pointcloud.ply")
        print(f"Saving PLY point cloud to: {ply_path}")

        point_cloud = trimesh.PointCloud(vertices=combined_points, colors=combined_colors)
        point_cloud.export(ply_path)

        print(f"Saved {len(combined_points)} points to PLY file")
    else:
        print("No valid points found!")

    # Save camera poses to text file
    poses_path = os.path.join(output_dir, f"{sequence_name}_camera_poses.txt")
    print(f"Saving camera poses to: {poses_path}")

    with open(poses_path, 'w') as f:
        f.write(f"Sequence: {sequence_name}\n")
        f.write(f"Number of cameras: {len(all_camera_poses)}\n\n")

        for i, cam_data in enumerate(all_camera_poses):
            f.write(f"Camera {i}:\n")
            f.write("Intrinsics:\n")
            f.write(np.array2string(cam_data['intrinsics'], separator=', '))
            f.write("\n\nCamera Pose (c2w):\n")
            f.write(np.array2string(cam_data['camera_pose'], separator=', '))
            f.write("\n\n" + "="*50 + "\n\n")

    print("Visualization files saved!")


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
    parser = argparse.ArgumentParser(description="Run MapAnything on Waymo sequences")
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
    parser.add_argument(
        "--reference_idx",
        type=int,
        default=0,
        help="Index of the reference view (0-based, default: 0)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained checkpoint (alternative to pretrained model)"
    )


    args = parser.parse_args()

    # Validate modalities
    validate_modalities(args.modalities)
    print(f"Using modalities: {args.modalities}")
    print(f"Metric scale: {not args.no_metric_scale}")

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

    # Prepare views for the entire sequence
    print("Preparing views for sequence...")
    views = prepare_views_for_sequence(
        reader,
        args.camera_types,
        converter,
        args.modalities,
        not args.no_metric_scale
    )
    views = views[::11]
    if len(views) == 0:
        print("No views found! Check camera types and sequence data.")
        return

    # Count unique timestamps (we don't store them in views anymore)
    timestamps = set()
    for frame in reader.get_frames():
        if frame['camera'] in args.camera_types:
            timestamps.add(frame['timestamp'])
    print(f"Prepared {len(views)} views from {len(timestamps)} timestamps")

    # Initialize MapAnything model
    print("Loading MapAnything model...")
    if args.checkpoint:
        # Load from checkpoint
        print(f"Loading from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

        # Try to extract model config from checkpoint
        model_config = {
            'model_str': 'mapanything_chunked',
            'model_config': {
                'encoder': {'name': 'dinov2_large'},
                'info_sharing': {'name': 'aat_ifr_24_layers'},
                'pred_head': {'name': 'dpt_pose_scale'},
                'task': {'name': 'images_only'}
            }
        }

        # Override with saved config if available
        if 'args' in ckpt and hasattr(ckpt['args'], 'model'):
            saved_config = ckpt['args'].model
            if hasattr(saved_config, 'model_str'):
                model_config['model_str'] = saved_config.model_str
            if hasattr(saved_config, 'model_config'):
                model_config['model_config'] = saved_config.model_config
        
        print(f"Using model config: {model_config}")

        # # wrap model_config in OmegaConf for compatibility
        # from omegaconf import OmegaConf
        # model_config = OmegaConf.create(model_config)

        model = init_model(
            model_config['model_str'],
            model_config['model_config'],
            torch_hub_force_reload=False
        )

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
        print("Loaded model from checkpoint, missing keys: {}, unexpected keys: {}".format(missing_keys, unexpected_keys))

    else:
        # Load pretrained model
        model = MapAnything.from_pretrained("facebook/map-anything")
        print("Loaded pretrained model")

    model.to(device)
    model.eval()

    # Update data_norm_type in views to match model
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'data_norm_type'):
        data_norm_type = model.encoder.data_norm_type
    else:
        data_norm_type = 'dinov2'  # fallback

    print(f"Using data normalization type: {data_norm_type}")
    for view in views:
        view['data_norm_type'] = [data_norm_type]  # List format expected by preprocess_inputs

    if args.reference_idx < 0 or args.reference_idx >= len(views):
        raise ValueError(f"reference_idx {args.reference_idx} is out of range [0, {len(views)-1}]")

    # Reorder views so that reference_idx becomes the first view
    if args.reference_idx != 0 and args.reference_idx < len(views):
        reference_view = views[args.reference_idx]
        remaining_views = views[:args.reference_idx] + views[args.reference_idx+1:]
        views = [reference_view] + remaining_views
        print(f"Using view {args.reference_idx} as reference (moved to position 0)")

    # Run inference
    predictions = run_inference(model, views, not args.no_memory_efficient)

    # Save visualizations
    save_visualizations(
        predictions, args.output_dir, sequence_name,
        not args.no_depth_validity_check, args.conf_threshold, args.max_points
    )

    print("Processing completed!")


if __name__ == "__main__":
    main()
