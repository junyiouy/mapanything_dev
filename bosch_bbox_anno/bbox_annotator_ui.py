"""
Interactive bounding box annotation interface for point cloud visualization.
"""

import numpy as np
import gradio as gr
import json
import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import utilities with robust import handling
import importlib.util
import os

def import_module_from_path(module_name, file_path):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Try to import modules, handling both package and direct execution
try:
    # First try absolute imports (when run as package)
    from pointcloud_renderer import PointCloudRenderer, BoundingBoxAnnotator
    from camera_utils import load_camera_parameters
except ImportError:
    try:
        # Fallback: try relative imports (when run as package)
        from .pointcloud_renderer import PointCloudRenderer, BoundingBoxAnnotator
        from .camera_utils import load_camera_parameters
    except ImportError:
        # Last resort: import directly from files
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Import pointcloud_renderer
        pointcloud_renderer = import_module_from_path("pointcloud_renderer", os.path.join(current_dir, "pointcloud_renderer.py"))
        PointCloudRenderer = pointcloud_renderer.PointCloudRenderer
        BoundingBoxAnnotator = pointcloud_renderer.BoundingBoxAnnotator

        # Import camera_utils
        camera_utils = import_module_from_path("camera_utils", os.path.join(current_dir, "camera_utils.py"))
        load_camera_parameters = camera_utils.load_camera_parameters


class BBoxAnnotationUI:
    """
    Interactive UI for bounding box annotation on point clouds.
    """

    def __init__(self, point_cloud: np.ndarray, colors: Optional[np.ndarray] = None,
                 camera_params: Optional[List[Dict]] = None):
        """
        Initialize the annotation UI.

        Args:
            point_cloud: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (0-255), optional
            camera_params: Camera parameters for rendering
        """
        self.point_cloud = point_cloud
        self.colors = colors
        self.camera_params = camera_params or []

        # Initialize components
        self.annotator = BoundingBoxAnnotator(point_cloud, colors)
        self.renderer = None

        if self.camera_params:
            self.renderer = PointCloudRenderer(self.camera_params)

        # UI state
        self.current_view_camera = 0
        self.selected_bbox_idx = None

    def create_gradio_interface(self):
        """
        Create a Gradio interface for bbox annotation.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Point Cloud Bounding Box Annotator") as interface:

            # State variables
            bbox_list_state = gr.State(value=[])
            selected_bbox_state = gr.State(value=None)

            with gr.Row():
                with gr.Column(scale=2):
                    # 3D Visualization
                    gr.Markdown("### 3D Point Cloud View")
                    point_cloud_plot = gr.Plot(label="Point Cloud")

                    # Camera controls
                    if self.camera_params:
                        with gr.Row():
                            camera_selector = gr.Dropdown(
                                choices=[f"Camera {i}" for i in range(len(self.camera_params))],
                                value="Camera 0",
                                label="View Camera"
                            )
                            render_btn = gr.Button("Render from Camera", variant="secondary")

                with gr.Column(scale=1):
                    # Bounding Box Controls
                    gr.Markdown("### Bounding Box Tools")

                    with gr.Tab("Manual BBox"):
                        gr.Markdown("Enter bounding box coordinates manually:")

                        min_x = gr.Number(label="Min X", value=0.0)
                        min_y = gr.Number(label="Min Y", value=0.0)
                        min_z = gr.Number(label="Min Z", value=0.0)
                        max_x = gr.Number(label="Max X", value=1.0)
                        max_y = gr.Number(label="Max Y", value=1.0)
                        max_z = gr.Number(label="Max Z", value=1.0)
                        bbox_label = gr.Textbox(label="Label", value="bbox_0")

                        add_bbox_btn = gr.Button("Add Bounding Box", variant="primary")

                    with gr.Tab("Statistics"):
                        bbox_stats_text = gr.Textbox(
                            label="Bounding Box Statistics",
                            lines=10,
                            interactive=False
                        )

                        refresh_stats_btn = gr.Button("Refresh Statistics")

                    # Bounding Box List
                    gr.Markdown("### Current Bounding Boxes")
                    bbox_list = gr.Dataframe(
                        headers=["Index", "Label", "Min Bounds", "Max Bounds", "Points"],
                        datatype=["number", "str", "str", "str", "number"],
                        interactive=False
                    )

                    with gr.Row():
                        selected_bbox = gr.Dropdown(
                            choices=[],
                            label="Select BBox to Edit"
                        )
                        remove_bbox_btn = gr.Button("Remove Selected", variant="stop")

            # File operations
            with gr.Row():
                save_btn = gr.Button("Save Annotations")
                load_btn = gr.Button("Load Annotations")
                export_render_btn = gr.Button("Export Rendered Views")

            file_output = gr.File(label="Download Annotations")

            # Event handlers
            def update_plot(camera_idx=0, bboxes=None):
                """Update the 3D plot with current bboxes."""
                return self._create_3d_plot(camera_idx, bboxes or [])

            def add_bbox(min_x, min_y, min_z, max_x, max_y, max_z, label, current_bboxes):
                """Add a new bounding box."""
                min_bounds = np.array([min_x, min_y, min_z])
                max_bounds = np.array([max_x, max_y, max_z])

                self.annotator.add_bbox_manual(min_bounds, max_bounds, label)

                # Update bbox list
                updated_bboxes = self.annotator.get_bboxes()
                bbox_table = self._create_bbox_table(updated_bboxes)

                # Update plot
                plot = self._create_3d_plot(self.current_view_camera, updated_bboxes)

                return plot, bbox_table, updated_bboxes

            def remove_bbox(selected_idx, current_bboxes):
                """Remove selected bounding box."""
                if selected_idx is not None and 0 <= selected_idx < len(current_bboxes):
                    self.annotator.remove_bbox(selected_idx)

                updated_bboxes = self.annotator.get_bboxes()
                bbox_table = self._create_bbox_table(updated_bboxes)
                plot = self._create_3d_plot(self.current_view_camera, updated_bboxes)

                return plot, bbox_table, updated_bboxes

            def update_bbox_list(current_bboxes):
                """Update the bbox list display."""
                return self._create_bbox_table(current_bboxes)

            def get_bbox_stats(current_bboxes):
                """Get statistics for all bboxes."""
                if not current_bboxes:
                    return "No bounding boxes to analyze."

                stats = self.annotator.get_bbox_statistics()
                stats_text = "Bounding Box Statistics:\n\n"

                for stat in stats:
                    stats_text += f"Index {stat['index']} ({stat['label']}):\n"
                    stats_text += f"  Dimensions: {stat['dimensions']}\n"
                    stats_text += f"  Volume: {stat['volume']:.3f}\n"
                    stats_text += f"  Points: {stat['num_points']}\n"
                    stats_text += f"  Center: {stat['center']}\n\n"

                return stats_text

            def save_annotations(current_bboxes):
                """Save annotations to file."""
                if not current_bboxes:
                    return None

                output_path = "bosch_bbox_annotations.json"
                self.annotator.save_annotations(output_path)
                return output_path

            def load_annotations(file_obj):
                """Load annotations from file."""
                if file_obj is None:
                    return [], [], self._create_3d_plot(0, [])

                try:
                    self.annotator.load_annotations(file_obj.name)
                    bboxes = self.annotator.get_bboxes()
                    bbox_table = self._create_bbox_table(bboxes)
                    plot = self._create_3d_plot(self.current_view_camera, bboxes)

                    return bboxes, bbox_table, plot
                except Exception as e:
                    print(f"Error loading annotations: {e}")
                    return [], [], self._create_3d_plot(0, [])

            def export_renders(current_bboxes):
                """Export rendered views from all cameras."""
                if not self.renderer:
                    return "No camera parameters available for rendering."

                # Add point cloud and bboxes to renderer
                self.renderer.add_point_cloud(self.point_cloud, self.colors)
                if current_bboxes:
                    self.renderer.add_bounding_boxes(current_bboxes)

                # Render from all cameras
                output_dir = "bosch_rendered_views"
                self.renderer.render_all_cameras(output_dir, "annotated_scene")

                return f"Rendered views saved to {output_dir}/"

            # Connect event handlers
            interface.load(
                fn=lambda: (self.annotator.get_bboxes(), self._create_bbox_table([]), self._create_3d_plot(0, [])),
                outputs=[bbox_list_state, bbox_list, point_cloud_plot]
            )

            add_bbox_btn.click(
                fn=add_bbox,
                inputs=[min_x, min_y, min_z, max_x, max_y, max_z, bbox_label, bbox_list_state],
                outputs=[point_cloud_plot, bbox_list, bbox_list_state]
            )

            remove_bbox_btn.click(
                fn=remove_bbox,
                inputs=[selected_bbox, bbox_list_state],
                outputs=[point_cloud_plot, bbox_list, bbox_list_state]
            )

            refresh_stats_btn.click(
                fn=get_bbox_stats,
                inputs=[bbox_list_state],
                outputs=[bbox_stats_text]
            )

            save_btn.click(
                fn=save_annotations,
                inputs=[bbox_list_state],
                outputs=[file_output]
            )

            load_btn.upload(
                fn=load_annotations,
                inputs=[load_btn],
                outputs=[bbox_list_state, bbox_list, point_cloud_plot]
            )

            export_render_btn.click(
                fn=export_renders,
                inputs=[bbox_list_state],
                outputs=[file_output]
            )

            if self.camera_params:
                camera_selector.change(
                    fn=lambda cam_idx: self._create_3d_plot(int(cam_idx.split()[1]), self.annotator.get_bboxes()),
                    inputs=[camera_selector],
                    outputs=[point_cloud_plot]
                )

                render_btn.click(
                    fn=lambda: "Rendered views exported!" if self.renderer else "No renderer available",
                    outputs=[file_output]
                )

        return interface

    def _create_3d_plot(self, camera_idx: int = 0, bboxes: List[Dict] = None) -> plt.Figure:
        """
        Create a 3D matplotlib plot of the point cloud with bounding boxes.

        Args:
            camera_idx: Camera index for viewpoint
            bboxes: List of bounding boxes to display

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot point cloud (downsample for performance)
        if len(self.point_cloud) > 10000:
            indices = np.random.choice(len(self.point_cloud), 10000, replace=False)
            points = self.point_cloud[indices]
            colors = self.colors[indices] if self.colors is not None else None
        else:
            points = self.point_cloud
            colors = self.colors

        if colors is not None:
            # Normalize colors for matplotlib
            colors_norm = colors.astype(float) / 255.0
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=colors_norm, s=1, alpha=0.6)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c='gray', s=1, alpha=0.6)

        # Plot bounding boxes
        if bboxes:
            colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
            for i, bbox in enumerate(bboxes):
                color = colors[i % len(colors)]
                self._plot_bbox(ax, bbox, color)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud with Bounding Boxes (Camera {camera_idx})')

        # Set equal aspect ratio
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return fig

    def _plot_bbox(self, ax: Axes3D, bbox: Dict, color: str):
        """
        Plot a single bounding box on the 3D axes.

        Args:
            ax: Matplotlib 3D axes
            bbox: Bounding box dictionary
            color: Color for the bbox
        """
        min_bounds = np.array(bbox['min'])
        max_bounds = np.array(bbox['max'])

        # Define the 8 corners of the bbox
        corners = np.array([
            [min_bounds[0], min_bounds[1], min_bounds[2]],  # 0
            [max_bounds[0], min_bounds[1], min_bounds[2]],  # 1
            [max_bounds[0], max_bounds[1], min_bounds[2]],  # 2
            [min_bounds[0], max_bounds[1], min_bounds[2]],  # 3
            [min_bounds[0], min_bounds[1], max_bounds[2]],  # 4
            [max_bounds[0], min_bounds[1], max_bounds[2]],  # 5
            [max_bounds[0], max_bounds[1], max_bounds[2]],  # 6
            [min_bounds[0], max_bounds[1], max_bounds[2]],  # 7
        ])

        # Define edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7],  # sides
        ]

        # Plot edges
        for edge in edges:
            ax.plot3D(
                [corners[edge[0], 0], corners[edge[1], 0]],
                [corners[edge[0], 1], corners[edge[1], 1]],
                [corners[edge[0], 2], corners[edge[1], 2]],
                color=color, linewidth=2
            )

    def _create_bbox_table(self, bboxes: List[Dict]) -> List[List]:
        """
        Create a table representation of bounding boxes.

        Args:
            bboxes: List of bounding box dictionaries

        Returns:
            Table data for Gradio Dataframe
        """
        table_data = []
        for i, bbox in enumerate(bboxes):
            min_str = f"[{bbox['min'][0]:.2f}, {bbox['min'][1]:.2f}, {bbox['min'][2]:.2f}]"
            max_str = f"[{bbox['max'][0]:.2f}, {bbox['max'][1]:.2f}, {bbox['max'][2]:.2f}]"

            table_data.append([
                i,
                bbox.get('label', f'bbox_{i}'),
                min_str,
                max_str,
                bbox.get('num_points', 0)
            ])

        return table_data


def create_bbox_annotation_app(point_cloud_path: str = None, camera_params_path: str = None):
    """
    Create and launch the bounding box annotation application.

    Args:
        point_cloud_path: Path to point cloud data (GLB or NPZ file)
        camera_params_path: Path to camera parameters JSON file
    """
    # Load point cloud data
    if point_cloud_path and point_cloud_path.endswith('.glb'):
        # Load from GLB file
        scene = trimesh.load(point_cloud_path)
        if hasattr(scene, 'geometry'):
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
    elif point_cloud_path and point_cloud_path.endswith('.npz'):
        # Load from NPZ file (MapAnything predictions)
        data = np.load(point_cloud_path)
        points = data['points']
        colors = data.get('images', None)
        if colors is not None and colors.ndim == 4:
            colors = colors.reshape(-1, 3)  # Flatten to (N, 3)
    else:
        # Generate sample data for testing
        print("No point cloud file provided, generating sample data...")
        points = np.random.rand(10000, 3) * 10
        colors = np.random.randint(0, 255, (10000, 3))

    # Load camera parameters
    camera_params = None
    if camera_params_path and os.path.exists(camera_params_path):
        camera_params = load_camera_parameters(camera_params_path)

    # Create annotation UI
    annotator_ui = BBoxAnnotationUI(points, colors, camera_params)

    # Launch Gradio app
    interface = annotator_ui.create_gradio_interface()
    interface.launch(share=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Point Cloud Bounding Box Annotator")
    parser.add_argument("--point_cloud", type=str, help="Path to point cloud file (GLB or NPZ)")
    parser.add_argument("--cameras", type=str, help="Path to camera parameters JSON file")

    args = parser.parse_args()

    create_bbox_annotation_app(args.point_cloud, args.cameras)
