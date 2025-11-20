"""
Point cloud rendering engine with bounding box support for Bosch data visualization.
Uses Open3D for headless-friendly rendering.
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import os
import os.path as osp

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: Open3D not available. Install with: pip install open3d")
    OPEN3D_AVAILABLE = False
    o3d = None

# Import camera utilities - handle both relative and absolute imports
try:
    from .camera_utils import load_camera_parameters, get_camera_view_matrix
except ImportError:
    # Fallback for direct execution
    from camera_utils import load_camera_parameters, get_camera_view_matrix


class PointCloudRenderer:
    """
    Renderer for point clouds with camera visualization and bounding box support.
    Uses Open3D for headless-friendly rendering.
    """

    def __init__(self, camera_params: List[Dict], width: int = 1280, height: int = 720):
        """
        Initialize the renderer.

        Args:
            camera_params: List of camera parameter dictionaries
            width: Render width
            height: Render height
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is required for rendering. Install with: pip install open3d")

        self.camera_params = camera_params
        self.width = width
        self.height = height

        # Store scene objects
        self.point_cloud = None
        self.bbox_geometries = []

        # Try to create visualizer (may fail in headless environments)
        self.vis = None
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=width, height=height, visible=False)
            # Set background color
            opt = self.vis.get_render_option()
            opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
            print("Open3D visualizer initialized successfully")
        except Exception as e:
            print(f"Warning: Open3D visualizer initialization failed: {e}")
            print("Rendering will be limited to point cloud export only")

    def add_point_cloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                       point_size: float = 2.0):
        """
        Add point cloud to the scene.

        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (0-255), optional
            point_size: Size of points for rendering
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Set colors
        if colors is not None:
            # Normalize colors to 0-1 range
            if colors.dtype != np.float64:
                colors_norm = colors.astype(np.float64) / 255.0
            else:
                colors_norm = colors
            pcd.colors = o3d.utility.Vector3dVector(colors_norm)

        # Store point cloud for later use
        self.point_cloud = pcd
        self.point_size = point_size

        # Add to visualizer if available
        if self.vis is not None:
            # Remove existing point cloud
            if hasattr(self, '_current_pcd') and self._current_pcd is not None:
                try:
                    self.vis.remove_geometry(self._current_pcd)
                except:
                    pass

            # Add to visualizer
            self.vis.add_geometry(pcd)
            self._current_pcd = pcd

            # Set point size
            try:
                render_option = self.vis.get_render_option()
                render_option.point_size = point_size
            except:
                pass

    def add_bounding_boxes(self, bboxes: List[Dict], colors: Optional[List] = None):
        """
        Add bounding boxes to the scene.

        Args:
            bboxes: List of bbox dictionaries with 'min' and 'max' keys
            colors: List of RGB colors for each bbox (0-1 range)
        """
        if colors is None:
            # Default colors (normalized to 0-1)
            colors = [
                [1.0, 0.0, 0.0],    # Red
                [0.0, 1.0, 0.0],    # Green
                [0.0, 0.0, 1.0],    # Blue
                [1.0, 1.0, 0.0],    # Yellow
                [1.0, 0.0, 1.0],    # Magenta
                [0.0, 1.0, 1.0],    # Cyan
            ]

        # Remove existing bbox geometries
        for bbox_geom in self.bbox_geometries:
            self.vis.remove_geometry(bbox_geom)
        self.bbox_geometries = []

        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]

            # Create Open3D oriented bounding box
            min_bounds = np.array(bbox['min'])
            max_bounds = np.array(bbox['max'])

            # Calculate center and extent
            center = (min_bounds + max_bounds) / 2
            extent = max_bounds - min_bounds

            # Create oriented bounding box (axis-aligned)
            obb = o3d.geometry.OrientedBoundingBox(center=center, extent=extent)
            obb.color = color

            # Add to visualizer
            self.vis.add_geometry(obb)
            self.bbox_geometries.append(obb)

    def render_from_camera(self, camera_idx: int, show_bboxes: bool = True,
                          show_point_cloud: bool = True) -> np.ndarray:
        """
        Render the scene from a specific camera viewpoint.

        Args:
            camera_idx: Index of the camera to use
            show_bboxes: Whether to show bounding boxes
            show_point_cloud: Whether to show point cloud

        Returns:
            Rendered RGB image as numpy array (HxWx3)
        """
        if camera_idx >= len(self.camera_params):
            raise ValueError(f"Camera index {camera_idx} out of range")

        # Set visibility
        if self.point_cloud is not None:
            # Note: Open3D doesn't have direct visibility control for point clouds
            # We could remove/add geometry instead, but that's expensive
            pass

        # Set camera viewpoint
        cam_param = self.camera_params[camera_idx]
        if cam_param.get('extrinsics') is not None:
            # Convert camera-to-world to world-to-camera for Open3D
            extrinsics = cam_param['extrinsics']
            view_matrix = np.linalg.inv(extrinsics)

            # Set view control
            ctr = self.vis.get_view_control()
            # Note: Open3D's view control is complex, we'll use a simpler approach
            # For now, we'll render from default viewpoint
            # TODO: Implement proper camera viewpoint setting

        # Update renderer
        self.vis.poll_events()
        self.vis.update_renderer()

        # Capture screen
        image = self.vis.capture_screen_float_buffer(do_render=True)

        # Convert to numpy array and RGB format
        image_np = np.asarray(image)
        if image_np.ndim == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]  # Remove alpha channel

        # Convert to uint8 (0-255 range)
        image_np = (image_np * 255).astype(np.uint8)

        return image_np

    def render_all_cameras(self, output_dir: str, prefix: str = "render",
                          show_bboxes: bool = True, show_point_cloud: bool = True):
        """
        Render from all cameras and save images.

        Args:
            output_dir: Directory to save rendered images
            prefix: Prefix for output filenames
            show_bboxes: Whether to show bounding boxes
            show_point_cloud: Whether to show point cloud
        """
        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(self.camera_params)):
            try:
                print(f"Rendering camera {i}...")
                image = self.render_from_camera(i, show_bboxes, show_point_cloud)

                # Convert to PIL Image and save
                pil_image = Image.fromarray(image)
                output_path = osp.join(output_dir, f"{prefix}_cam_{i:03d}.png")
                pil_image.save(output_path)

                print(f"Saved render: {output_path}")

            except Exception as e:
                print(f"Failed to render camera {i}: {e}")

    def save_scene(self, output_path: str):
        """
        Save the current scene as various formats.

        Args:
            output_path: Output file path (supports .ply, .pcd, etc.)
        """
        if self.point_cloud is not None:
            # Save point cloud
            o3d.io.write_point_cloud(output_path, self.point_cloud)
            print(f"Saved point cloud to: {output_path}")
        else:
            print("No point cloud to save")

    def close(self):
        """Clean up the visualizer."""
        if hasattr(self, 'vis'):
            self.vis.destroy_window()


class BoundingBoxAnnotator:
    """
    Interactive bounding box annotation tool for point clouds.
    """

    def __init__(self, point_cloud: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        Initialize the annotator.

        Args:
            point_cloud: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (0-255), optional
        """
        self.point_cloud = point_cloud
        self.colors = colors if colors is not None else np.full_like(point_cloud, 128, dtype=np.uint8)
        self.bboxes = []

    def add_bbox_from_points(self, selected_points: np.ndarray, label: str = None):
        """
        Create a bounding box from selected 3D points.

        Args:
            selected_points: Mx3 array of selected 3D points
            label: Optional label for the bbox
        """
        if len(selected_points) == 0:
            return

        # Calculate bounding box
        min_bounds = np.min(selected_points, axis=0)
        max_bounds = np.max(selected_points, axis=0)

        bbox = {
            'min': min_bounds.tolist(),
            'max': max_bounds.tolist(),
            'label': label or f'bbox_{len(self.bboxes)}',
            'num_points': len(selected_points)
        }

        self.bboxes.append(bbox)

    def add_bbox_manual(self, min_bounds: np.ndarray, max_bounds: np.ndarray, label: str = None):
        """
        Manually add a bounding box.

        Args:
            min_bounds: 3D min bounds
            max_bounds: 3D max bounds
            label: Optional label for the bbox
        """
        bbox = {
            'min': min_bounds.tolist(),
            'max': max_bounds.tolist(),
            'label': label or f'bbox_{len(self.bboxes)}',
            'num_points': 0  # Manual bbox
        }

        self.bboxes.append(bbox)

    def remove_bbox(self, index: int):
        """
        Remove a bounding box by index.

        Args:
            index: Index of bbox to remove
        """
        if 0 <= index < len(self.bboxes):
            self.bboxes.pop(index)

    def get_bboxes(self) -> List[Dict]:
        """Get all bounding boxes."""
        return self.bboxes.copy()

    def save_annotations(self, output_path: str):
        """
        Save bounding box annotations to JSON file.

        Args:
            output_path: Output JSON file path
        """
        with open(output_path, 'w') as f:
            json.dump(self.bboxes, f, indent=2)

    def load_annotations(self, json_path: str):
        """
        Load bounding box annotations from JSON file.

        Args:
            json_path: Path to JSON file
        """
        with open(json_path, 'r') as f:
            self.bboxes = json.load(f)

    def get_points_in_bbox(self, bbox_idx: int) -> np.ndarray:
        """
        Get all points within a specific bounding box.

        Args:
            bbox_idx: Index of the bounding box

        Returns:
            Nx3 array of points within the bbox
        """
        if bbox_idx >= len(self.bboxes):
            return np.array([])

        bbox = self.bboxes[bbox_idx]
        min_bounds = np.array(bbox['min'])
        max_bounds = np.array(bbox['max'])

        # Find points within bbox
        mask = np.all((self.point_cloud >= min_bounds) & (self.point_cloud <= max_bounds), axis=1)
        return self.point_cloud[mask]

    def get_bbox_statistics(self) -> List[Dict]:
        """
        Get statistics for all bounding boxes.

        Returns:
            List of dictionaries with bbox statistics
        """
        stats = []
        for i, bbox in enumerate(self.bboxes):
            min_bounds = np.array(bbox['min'])
            max_bounds = np.array(bbox['max'])

            # Calculate dimensions
            dimensions = max_bounds - min_bounds

            # Count points in bbox
            points_in_bbox = self.get_points_in_bbox(i)

            stat = {
                'index': i,
                'label': bbox['label'],
                'dimensions': dimensions.tolist(),
                'volume': float(np.prod(dimensions)),
                'num_points': len(points_in_bbox),
                'center': ((min_bounds + max_bounds) / 2).tolist()
            }

            stats.append(stat)

        return stats
