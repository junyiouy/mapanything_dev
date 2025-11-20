#!/usr/bin/env python3
"""
Main rendering script for Bosch point cloud visualization with bounding box annotation.

This script provides a complete pipeline for:
1. Loading MapAnything predictions or GLB files
2. Extracting camera parameters
3. Rendering point clouds from multiple camera viewpoints
4. Adding and visualizing bounding box annotations
5. Exporting rendered images and annotations

Usage:
    python main_renderer.py --input /path/to/predictions.npz --output_dir ./output
    python main_renderer.py --glb_file /path/to/scene.glb --bbox_file /path/to/bboxes.json
"""

import argparse
import os
import sys
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utilities with robust import handling
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Try to import modules, handling both package and direct execution
try:
    # First try absolute imports (when run as package)
    from camera_utils import (
        extract_camera_parameters_from_glb,
        extract_camera_parameters_from_predictions,
        save_camera_parameters,
        load_camera_parameters
    )
    from pointcloud_renderer import PointCloudRenderer, BoundingBoxAnnotator
    from bbox_annotator_ui import create_bbox_annotation_app
except ImportError:
    try:
        # Fallback: try relative imports (when run as package)
        from .camera_utils import (
            extract_camera_parameters_from_glb,
            extract_camera_parameters_from_predictions,
            save_camera_parameters,
            load_camera_parameters
        )
        from .pointcloud_renderer import PointCloudRenderer, BoundingBoxAnnotator
        from .bbox_annotator_ui import create_bbox_annotation_app
    except ImportError:
        # Last resort: import directly from files
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Import camera_utils
        camera_utils = import_module_from_path("camera_utils", os.path.join(current_dir, "camera_utils.py"))
        extract_camera_parameters_from_glb = camera_utils.extract_camera_parameters_from_glb
        extract_camera_parameters_from_predictions = camera_utils.extract_camera_parameters_from_predictions
        save_camera_parameters = camera_utils.save_camera_parameters
        load_camera_parameters = camera_utils.load_camera_parameters

        # Import pointcloud_renderer
        pointcloud_renderer = import_module_from_path("pointcloud_renderer", os.path.join(current_dir, "pointcloud_renderer.py"))
        PointCloudRenderer = pointcloud_renderer.PointCloudRenderer
        BoundingBoxAnnotator = pointcloud_renderer.BoundingBoxAnnotator

        # Import bbox_annotator_ui
        bbox_annotator_ui = import_module_from_path("bbox_annotator_ui", os.path.join(current_dir, "bbox_annotator_ui.py"))
        create_bbox_annotation_app = bbox_annotator_ui.create_bbox_annotation_app


def load_point_cloud_data(input_path: str) -> tuple:
    """
    Load point cloud data from various formats.

    Args:
        input_path: Path to input file (NPZ or GLB)

    Returns:
        Tuple of (points, colors, camera_params)
    """
    input_path = Path(input_path)

    if input_path.suffix == '.npz':
        # Load from MapAnything predictions
        print(f"Loading predictions from: {input_path}")
        data = np.load(input_path, allow_pickle=True)

        points = data['points']
        colors = data.get('images', None)
        if colors is not None and colors.ndim == 4:
            colors = colors.reshape(-1, 3)  # Flatten to (N, 3)

        # Extract camera parameters
        camera_params = extract_camera_parameters_from_predictions(str(input_path))

        print(f"Loaded {len(points)} points, {len(camera_params)} cameras")

    elif input_path.suffix == '.glb':
        # Load from GLB file
        print(f"Loading GLB scene from: {input_path}")

        import trimesh
        scene = trimesh.load(input_path)

        if hasattr(scene, 'geometry') and scene.geometry:
            # Get the first geometry (assuming it's a point cloud)
            geom_name = list(scene.geometry.keys())[0]
            geometry = scene.geometry[geom_name]

            if hasattr(geometry, 'vertices'):
                points = geometry.vertices
                colors = getattr(geometry, 'visual', None)
                if colors is not None and hasattr(colors, 'vertex_colors'):
                    colors = colors.vertex_colors[:, :3]  # RGB only
                else:
                    colors = None
            else:
                raise ValueError("GLB file does not contain valid point cloud geometry")
        else:
            raise ValueError("GLB file does not contain geometry")

        # Extract camera parameters from GLB
        camera_params = extract_camera_parameters_from_glb(str(input_path))

        print(f"Loaded {len(points)} points from GLB, {len(camera_params)} cameras")

    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    return points, colors, camera_params


def render_point_cloud_with_annotations(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    camera_params: List[Dict],
    bbox_file: Optional[str] = None,
    output_dir: str = "./output",
    render_width: int = 1280,
    render_height: int = 720
):
    """
    Render point cloud with optional bounding box annotations.

    Args:
        points: Nx3 point cloud
        colors: Nx3 colors (optional)
        camera_params: Camera parameter dictionaries
        bbox_file: Path to bounding box annotations JSON
        output_dir: Output directory
        render_width: Render image width
        render_height: Render image height
    """
    print("Setting up renderer...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize renderer
    renderer = PointCloudRenderer(camera_params, render_width, render_height)

    # Add point cloud
    renderer.add_point_cloud(points, colors)

    # Load and add bounding boxes if provided
    bboxes = []
    if bbox_file and Path(bbox_file).exists():
        print(f"Loading bounding box annotations from: {bbox_file}")
        with open(bbox_file, 'r') as f:
            bboxes = json.load(f)
        renderer.add_bounding_boxes(bboxes)
        print(f"Loaded {len(bboxes)} bounding boxes")

    # Render from all cameras
    print("Rendering from all camera viewpoints...")
    render_prefix = "scene_with_bboxes" if bboxes else "scene"
    try:
        renderer.render_all_cameras(str(output_dir), render_prefix)
    finally:
        # Clean up Open3D visualizer
        if hasattr(renderer, 'close'):
            renderer.close()

    # Save scene as point cloud file
    pc_output = output_dir / "rendered_scene.ply"
    try:
        renderer.save_scene(str(pc_output))
    finally:
        # Clean up Open3D visualizer if not already closed
        if hasattr(renderer, 'close'):
            renderer.close()

    print(f"Rendering complete! Output saved to: {output_dir}")
    print(f"- Rendered images: {output_dir}/{render_prefix}_cam_*.png")
    print(f"- Point cloud: {pc_output}")

    return str(output_dir)


def extract_and_save_camera_params(input_path: str, output_dir: str = "./output"):
    """
    Extract camera parameters and save to JSON file.

    Args:
        input_path: Path to input file
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load camera parameters
    if Path(input_path).suffix == '.npz':
        cameras = extract_camera_parameters_from_predictions(input_path)
    elif Path(input_path).suffix == '.glb':
        cameras = extract_camera_parameters_from_glb(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    # Save to JSON
    json_path = output_dir / "camera_parameters.json"
    save_camera_parameters(cameras, str(json_path))

    print(f"Camera parameters saved to: {json_path}")
    print(f"Extracted {len(cameras)} camera configurations")

    return str(json_path)


def create_sample_bboxes(points: np.ndarray, num_bboxes: int = 3) -> List[Dict]:
    """
    Create sample bounding boxes for testing.

    Args:
        points: Point cloud data
        num_bboxes: Number of sample bboxes to create

    Returns:
        List of bounding box dictionaries
    """
    if len(points) == 0:
        return []

    # Get point cloud bounds
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds

    bboxes = []
    for i in range(num_bboxes):
        # Create bbox around random region
        offset = np.random.rand(3) * size * 0.5
        bbox_min = center - size * (0.2 + np.random.rand(3) * 0.3)
        bbox_max = bbox_min + size * (0.1 + np.random.rand(3) * 0.2)

        # Ensure bounds are within point cloud range
        bbox_min = np.maximum(bbox_min, min_bounds)
        bbox_max = np.minimum(bbox_max, max_bounds)

        bbox = {
            'min': bbox_min.tolist(),
            'max': bbox_max.tolist(),
            'label': f'sample_bbox_{i}',
            'num_points': 0
        }
        bboxes.append(bbox)

    return bboxes


def batch_render_multiple_scenes(input_dir: str, output_dir: str = "./batch_output",
                                bbox_dir: Optional[str] = None):
    """
    Batch render multiple scenes.

    Args:
        input_dir: Directory containing input files
        output_dir: Output directory
        bbox_dir: Directory containing bbox annotation files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all supported input files
    input_files = []
    input_files.extend(input_dir.glob("*.npz"))
    input_files.extend(input_dir.glob("*.glb"))

    print(f"Found {len(input_files)} scenes to process")

    for input_file in input_files:
        print(f"\nProcessing: {input_file.name}")

        try:
            # Load data
            points, colors, cameras = load_point_cloud_data(str(input_file))

            # Check for corresponding bbox file
            bbox_file = None
            if bbox_dir:
                bbox_path = Path(bbox_dir) / f"{input_file.stem}_bboxes.json"
                if bbox_path.exists():
                    bbox_file = str(bbox_path)

            # Create scene-specific output directory
            scene_output_dir = output_dir / input_file.stem
            scene_output_dir.mkdir(exist_ok=True)

            # Render scene
            render_point_cloud_with_annotations(
                points, colors, cameras, bbox_file, str(scene_output_dir)
            )

        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")
            continue

    print(f"\nBatch processing complete! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Bosch Point Cloud Renderer with Bounding Box Annotations"
    )

    # Input options
    parser.add_argument(
        "--input", "-i", type=str,
        help="Input file (NPZ predictions or GLB scene)"
    )
    parser.add_argument(
        "--input_dir", type=str,
        help="Input directory for batch processing"
    )
    parser.add_argument(
        "--glb_file", type=str,
        help="GLB scene file path"
    )

    # Output options
    parser.add_argument(
        "--output_dir", "-o", type=str, default="./bosch_output",
        help="Output directory"
    )

    # Annotation options
    parser.add_argument(
        "--bbox_file", type=str,
        help="Bounding box annotations JSON file"
    )
    parser.add_argument(
        "--bbox_dir", type=str,
        help="Directory containing bbox files for batch processing"
    )
    parser.add_argument(
        "--create_sample_bboxes", action="store_true",
        help="Create sample bounding boxes for testing"
    )

    # Rendering options
    parser.add_argument(
        "--render_width", type=int, default=1280,
        help="Render image width"
    )
    parser.add_argument(
        "--render_height", type=int, default=720,
        help="Render image height"
    )

    # Utility options
    parser.add_argument(
        "--extract_cameras", action="store_true",
        help="Extract and save camera parameters only"
    )
    parser.add_argument(
        "--launch_annotator", action="store_true",
        help="Launch interactive bbox annotation UI"
    )
    parser.add_argument(
        "--batch_render", action="store_true",
        help="Batch render multiple scenes"
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.input, args.input_dir, args.glb_file, args.launch_annotator]):
        parser.error("Must specify --input, --input_dir, --glb_file, or --launch_annotator")

    try:
        if args.launch_annotator:
            # Launch interactive annotation UI
            print("Launching interactive bounding box annotator...")
            create_bbox_annotation_app(args.input, args.bbox_file)

        elif args.batch_render:
            # Batch render multiple scenes
            if not args.input_dir:
                parser.error("--batch_render requires --input_dir")
            print(f"Batch rendering scenes from: {args.input_dir}")
            batch_render_multiple_scenes(
                args.input_dir, args.output_dir, args.bbox_dir
            )

        elif args.extract_cameras:
            # Extract camera parameters only
            input_file = args.input or args.glb_file
            if not input_file:
                parser.error("--extract_cameras requires --input or --glb_file")
            extract_and_save_camera_params(input_file, args.output_dir)

        else:
            # Single scene rendering
            input_file = args.input or args.glb_file
            if not input_file:
                parser.error("Must specify --input or --glb_file for single scene rendering")

            print(f"Processing single scene: {input_file}")

            # Load data
            points, colors, cameras = load_point_cloud_data(input_file)

            # Create sample bboxes if requested
            bbox_file = args.bbox_file
            if args.create_sample_bboxes and not bbox_file:
                print("Creating sample bounding boxes...")
                bboxes = create_sample_bboxes(points)
                bbox_file = str(Path(args.output_dir) / "sample_bboxes.json")
                with open(bbox_file, 'w') as f:
                    json.dump(bboxes, f, indent=2)
                print(f"Sample bboxes saved to: {bbox_file}")

            # Render scene
            render_point_cloud_with_annotations(
                points, colors, cameras, bbox_file, args.output_dir,
                args.render_width, args.render_height
            )

        print("\nProcessing completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
