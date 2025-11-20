"""
Integration script to add point cloud rendering and bbox annotation to the existing Gradio app.
"""

import os
import sys
import numpy as np
import gradio as gr
from pathlib import Path

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))  # Add parent directory for mapanything imports

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
    from camera_utils import extract_camera_parameters_from_predictions
    from pointcloud_renderer import PointCloudRenderer, BoundingBoxAnnotator
    from bbox_annotator_ui import BBoxAnnotationUI
except ImportError:
    try:
        # Fallback: try relative imports (when run as package)
        from .camera_utils import extract_camera_parameters_from_predictions
        from .pointcloud_renderer import PointCloudRenderer, BoundingBoxAnnotator
        from .bbox_annotator_ui import BBoxAnnotationUI
    except ImportError:
        # Last resort: import directly from files
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Import camera_utils
        camera_utils = import_module_from_path("camera_utils", os.path.join(current_dir, "camera_utils.py"))
        extract_camera_parameters_from_predictions = camera_utils.extract_camera_parameters_from_predictions

        # Import pointcloud_renderer
        pointcloud_renderer = import_module_from_path("pointcloud_renderer", os.path.join(current_dir, "pointcloud_renderer.py"))
        PointCloudRenderer = pointcloud_renderer.PointCloudRenderer
        BoundingBoxAnnotator = pointcloud_renderer.BoundingBoxAnnotator

        # Import bbox_annotator_ui
        bbox_annotator_ui = import_module_from_path("bbox_annotator_ui", os.path.join(current_dir, "bbox_annotator_ui.py"))
        BBoxAnnotationUI = bbox_annotator_ui.BBoxAnnotationUI


def add_pointcloud_rendering_tab(gradio_app, target_dir_state):
    """
    Add a point cloud rendering and bbox annotation tab to the existing Gradio app.

    Args:
        gradio_app: The main Gradio Blocks app
        target_dir_state: Gradio state for target directory

    Returns:
        Updated Gradio app with new tab
    """
    with gradio_app:

        with gr.Tab("Point Cloud Rendering & BBox Annotation"):

            gr.Markdown("""
            ## Point Cloud Rendering with Bounding Box Annotations

            This tab allows you to render point clouds from MapAnything predictions
            and add interactive bounding box annotations.
            """)

            with gr.Row():
                with gr.Column():
                    # Load predictions button
                    load_predictions_btn = gr.Button(
                        "Load Predictions from Target Directory",
                        variant="primary"
                    )

                    # Status display
                    load_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3
                    )

                    # Rendering options
                    with gr.Accordion("Rendering Options", open=True):
                        render_width = gr.Slider(
                            minimum=640, maximum=1920, value=1280, step=32,
                            label="Render Width"
                        )
                        render_height = gr.Slider(
                            minimum=480, maximum=1080, value=720, step=24,
                            label="Render Height"
                        )
                        show_cameras = gr.Checkbox(
                            label="Show Camera Frustums",
                            value=True
                        )
                        show_bboxes = gr.Checkbox(
                            label="Show Bounding Boxes",
                            value=True
                        )

                    # Render button
                    render_btn = gr.Button("Render Point Cloud", variant="secondary")

                with gr.Column():
                    # 3D Visualization placeholder
                    render_display = gr.Image(
                        label="Rendered View",
                        interactive=False
                    )

                    # Camera selector
                    camera_selector = gr.Dropdown(
                        choices=[],
                        label="Select Camera View",
                        interactive=True
                    )

            # Bounding Box Annotation Section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Bounding Box Annotation")

                    with gr.Tab("Add BBox"):
                        bbox_min_x = gr.Number(label="Min X", value=0.0)
                        bbox_min_y = gr.Number(label="Min Y", value=0.0)
                        bbox_min_z = gr.Number(label="Min Z", value=0.0)
                        bbox_max_x = gr.Number(label="Max X", value=1.0)
                        bbox_max_y = gr.Number(label="Max Y", value=1.0)
                        bbox_max_z = gr.Number(label="Max Z", value=1.0)
                        bbox_label = gr.Textbox(label="Label", value="bbox_0")

                        add_bbox_btn = gr.Button("Add Bounding Box")

                    with gr.Tab("Manage BBoxes"):
                        bbox_list = gr.Dataframe(
                            headers=["Index", "Label", "Min Bounds", "Max Bounds"],
                            interactive=False
                        )

                        selected_bbox = gr.Dropdown(
                            choices=[],
                            label="Select BBox to Remove"
                        )

                        remove_bbox_btn = gr.Button("Remove Selected", variant="stop")

                with gr.Column():
                    # Annotation statistics
                    bbox_stats = gr.Textbox(
                        label="Bounding Box Statistics",
                        lines=8,
                        interactive=False
                    )

                    # File operations
                    save_annotations_btn = gr.Button("Save Annotations")
                    load_annotations_btn = gr.UploadButton("Load Annotations")
                    export_renders_btn = gr.Button("Export All Camera Views")

            # State variables for this tab
            point_cloud_state = gr.State(None)  # (points, colors)
            camera_params_state = gr.State(None)
            bbox_annotations_state = gr.State([])
            renderer_state = gr.State(None)

            # Event handlers
            def load_predictions(target_dir):
                """Load predictions from the target directory."""
                if not target_dir or target_dir == "None":
                    return "No target directory set. Please run reconstruction first.", None, None, None, [], []

                predictions_path = os.path.join(target_dir, "predictions.npz")
                if not os.path.exists(predictions_path):
                    return f"No predictions found at {predictions_path}", None, None, None, [], []

                try:
                    # Load predictions
                    data = np.load(predictions_path, allow_pickle=True)
                    points = data['points']
                    colors = data.get('images', None)
                    if colors is not None and colors.ndim == 4:
                        colors = colors.reshape(-1, 3)

                    # Extract camera parameters
                    cameras = extract_camera_parameters_from_predictions(predictions_path)

                    # Create renderer
                    renderer = PointCloudRenderer(cameras, width=1280, height=720)
                    renderer.add_point_cloud(points, colors)

                    # Update camera selector
                    camera_choices = [f"Camera {i}" for i in range(len(cameras))]

                    status = f"Loaded {len(points)} points and {len(cameras)} cameras from {predictions_path}"

                    return status, (points, colors), cameras, renderer, camera_choices, []

                except Exception as e:
                    return f"Error loading predictions: {e}", None, None, None, [], []

            def render_from_camera(camera_idx, renderer, show_bboxes, bboxes):
                """Render from selected camera."""
                if renderer is None:
                    return None

                try:
                    # Add/remove bboxes based on checkbox
                    if show_bboxes and bboxes:
                        renderer.add_bounding_boxes(bboxes)
                    else:
                        # Remove existing bboxes by clearing and re-adding point cloud
                        renderer.scene.clear()
                        points, colors = renderer.point_cloud_data if hasattr(renderer, 'point_cloud_data') else (None, None)
                        if points is not None:
                            renderer.add_point_cloud(points, colors)

                    # Render from camera
                    camera_idx = int(camera_idx.split()[1]) if isinstance(camera_idx, str) else camera_idx
                    image = renderer.render_from_camera(camera_idx, show_bboxes=show_bboxes)

                    # Convert to PIL Image for display
                    from PIL import Image
                    pil_image = Image.fromarray(image)
                    return pil_image

                except Exception as e:
                    print(f"Rendering error: {e}")
                    return None

            def add_bounding_box(min_x, min_y, min_z, max_x, max_y, max_z, label, current_bboxes, point_cloud):
                """Add a new bounding box."""
                if point_cloud is None:
                    return current_bboxes, "No point cloud loaded"

                points, colors = point_cloud

                # Create annotator if needed
                annotator = BoundingBoxAnnotator(points, colors)

                # Add existing bboxes
                for bbox in current_bboxes:
                    annotator.add_bbox_manual(
                        np.array(bbox['min']),
                        np.array(bbox['max']),
                        bbox.get('label', 'bbox')
                    )

                # Add new bbox
                min_bounds = np.array([min_x, min_y, min_z])
                max_bounds = np.array([max_x, max_y, max_z])
                annotator.add_bbox_manual(min_bounds, max_bounds, label)

                updated_bboxes = annotator.get_bboxes()

                # Update bbox table
                table_data = []
                for i, bbox in enumerate(updated_bboxes):
                    min_str = f"[{bbox['min'][0]:.2f}, {bbox['min'][1]:.2f}, {bbox['min'][2]:.2f}]"
                    max_str = f"[{bbox['max'][0]:.2f}, {bbox['max'][1]:.2f}, {bbox['max'][2]:.2f}]"
                    table_data.append([i, bbox.get('label', f'bbox_{i}'), min_str, max_str])

                return updated_bboxes, table_data

            def remove_bounding_box(selected_idx, current_bboxes):
                """Remove selected bounding box."""
                if not current_bboxes or selected_idx is None:
                    return current_bboxes, []

                try:
                    idx = int(selected_idx)
                    if 0 <= idx < len(current_bboxes):
                        current_bboxes.pop(idx)

                    # Update table
                    table_data = []
                    for i, bbox in enumerate(current_bboxes):
                        min_str = f"[{bbox['min'][0]:.2f}, {bbox['min'][1]:.2f}, {bbox['min'][2]:.2f}]"
                        max_str = f"[{bbox['max'][0]:.2f}, {bbox['max'][1]:.2f}, {bbox['max'][2]:.2f}]"
                        table_data.append([i, bbox.get('label', f'bbox_{i}'), min_str, max_str])

                    return current_bboxes, table_data

                except Exception as e:
                    print(f"Error removing bbox: {e}")
                    return current_bboxes, []

            def get_bbox_statistics(bboxes, point_cloud):
                """Get statistics for bounding boxes."""
                if not bboxes or point_cloud is None:
                    return "No bounding boxes or point cloud available."

                points, colors = point_cloud
                annotator = BoundingBoxAnnotator(points, colors)

                # Add bboxes to annotator
                for bbox in bboxes:
                    annotator.add_bbox_manual(
                        np.array(bbox['min']),
                        np.array(bbox['max']),
                        bbox.get('label', 'bbox')
                    )

                stats = annotator.get_bbox_statistics()
                stats_text = "Bounding Box Statistics:\n\n"

                for stat in stats:
                    stats_text += f"Index {stat['index']} ({stat['label']}):\n"
                    stats_text += f"  Dimensions: {stat['dimensions']}\n"
                    stats_text += f"  Volume: {stat['volume']:.3f}\n"
                    stats_text += f"  Points: {stat['num_points']}\n\n"

                return stats_text

            def save_bbox_annotations(bboxes, target_dir):
                """Save bounding box annotations."""
                if not bboxes:
                    return None

                if not target_dir or target_dir == "None":
                    return None

                output_path = os.path.join(target_dir, "bounding_boxes.json")
                with open(output_path, 'w') as f:
                    import json
                    json.dump(bboxes, f, indent=2)

                return output_path

            def export_all_renders(renderer, bboxes, target_dir):
                """Export rendered views from all cameras."""
                if renderer is None:
                    return "No renderer available"

                if not target_dir or target_dir == "None":
                    return "No target directory set"

                try:
                    # Add bboxes to renderer if any
                    if bboxes:
                        renderer.add_bounding_boxes(bboxes)

                    # Render from all cameras
                    output_dir = os.path.join(target_dir, "rendered_views")
                    renderer.render_all_cameras(output_dir, "annotated_scene")

                    return f"Rendered views saved to {output_dir}/"

                except Exception as e:
                    return f"Error exporting renders: {e}"

            # Connect event handlers
            load_predictions_btn.click(
                fn=load_predictions,
                inputs=[target_dir_state],
                outputs=[
                    load_status, point_cloud_state, camera_params_state,
                    renderer_state, camera_selector, bbox_annotations_state
                ]
            )

            camera_selector.change(
                fn=lambda cam_idx, renderer, show_bboxes, bboxes: render_from_camera(cam_idx, renderer, show_bboxes, bboxes),
                inputs=[camera_selector, renderer_state, show_bboxes, bbox_annotations_state],
                outputs=[render_display]
            )

            render_btn.click(
                fn=lambda cam_idx, renderer, show_bboxes, bboxes: render_from_camera(cam_idx, renderer, show_bboxes, bboxes),
                inputs=[camera_selector, renderer_state, show_bboxes, bbox_annotations_state],
                outputs=[render_display]
            )

            add_bbox_btn.click(
                fn=add_bounding_box,
                inputs=[
                    bbox_min_x, bbox_min_y, bbox_min_z,
                    bbox_max_x, bbox_max_y, bbox_max_z,
                    bbox_label, bbox_annotations_state, point_cloud_state
                ],
                outputs=[bbox_annotations_state, bbox_list]
            )

            remove_bbox_btn.click(
                fn=remove_bounding_box,
                inputs=[selected_bbox, bbox_annotations_state],
                outputs=[bbox_annotations_state, bbox_list]
            )

            # Update bbox statistics when bboxes change
            bbox_annotations_state.change(
                fn=get_bbox_statistics,
                inputs=[bbox_annotations_state, point_cloud_state],
                outputs=[bbox_stats]
            )

            save_annotations_btn.click(
                fn=save_bbox_annotations,
                inputs=[bbox_annotations_state, target_dir_state],
                outputs=[load_status]
            )

            export_renders_btn.click(
                fn=export_all_renders,
                inputs=[renderer_state, bbox_annotations_state, target_dir_state],
                outputs=[load_status]
            )

    return gradio_app


def integrate_with_existing_gradio():
    """
    Example of how to integrate with the existing gradio_app.py.

    To use this integration, add the following to your gradio_app.py:

    ```python
    from bosch_bbox_anno.integrate_with_gradio import add_pointcloud_rendering_tab

    # After creating your main gradio_app Blocks object:
    gradio_app = add_pointcloud_rendering_tab(gradio_app, target_dir_output)
    ```
    """
    print("Point cloud rendering integration module loaded.")
    print("To integrate with existing Gradio app, import and call add_pointcloud_rendering_tab()")


if __name__ == "__main__":
    integrate_with_existing_gradio()
