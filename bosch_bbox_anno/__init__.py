"""
Bosch Point Cloud Rendering and Bounding Box Annotation System.

This package provides tools for:
- Extracting camera parameters from MapAnything predictions
- Rendering point clouds with multiple camera viewpoints
- Interactive bounding box annotation
- Integration with Gradio web interfaces
"""

from .camera_utils import (
    extract_camera_parameters_from_glb,
    extract_camera_parameters_from_predictions,
    save_camera_parameters,
    load_camera_parameters,
    get_camera_view_matrix,
    create_camera_frustum_mesh
)

from .pointcloud_renderer import (
    PointCloudRenderer,
    BoundingBoxAnnotator
)

from .bbox_annotator_ui import (
    BBoxAnnotationUI,
    create_bbox_annotation_app
)

from .main_renderer import (
    load_point_cloud_data,
    render_point_cloud_with_annotations,
    extract_and_save_camera_params,
    create_sample_bboxes,
    batch_render_multiple_scenes
)

__version__ = "1.0.0"
__author__ = "MapAnything Team"
__description__ = "Point cloud rendering and bbox annotation system for Bosch COLMAP data"

__all__ = [
    # Camera utilities
    'extract_camera_parameters_from_glb',
    'extract_camera_parameters_from_predictions',
    'save_camera_parameters',
    'load_camera_parameters',
    'get_camera_view_matrix',
    'create_camera_frustum_mesh',

    # Rendering components
    'PointCloudRenderer',
    'BoundingBoxAnnotator',

    # UI components
    'BBoxAnnotationUI',
    'create_bbox_annotation_app',

    # Main functions
    'load_point_cloud_data',
    'render_point_cloud_with_annotations',
    'extract_and_save_camera_params',
    'create_sample_bboxes',
    'batch_render_multiple_scenes'
]
