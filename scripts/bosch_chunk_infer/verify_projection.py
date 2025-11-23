import argparse
import json
import os
import cv2
import numpy as np

def get_bbox_corners_world(center, extent, rotation_matrix):
    """
    Calculate 8 corners of the bbox in World Coordinates.
    """
    center = np.array(center)
    extent = np.array(extent)
    R = np.array(rotation_matrix)
    
    # 1. Define canonical corners (relative to center, axis-aligned)
    dx, dy, dz = extent / 2.0
    
    # 8 corners: combinations of +/- dx, +/- dy, +/- dz
    corners_local = np.array([
        [dx, dy, dz], [dx, dy, -dz], [dx, -dy, -dz], [dx, -dy, dz],
        [-dx, dy, dz], [-dx, dy, -dz], [-dx, -dy, -dz], [-dx, -dy, dz]
    ])
    
    # 2. Rotate and Translate to World
    # P_world = R * P_local + Center
    corners_world = (R @ corners_local.T).T + center
    return corners_world

def project_points(points_world, c2w, K):
    """
    Project World Points to Image Pixels.
    """
    # 1. World -> Camera (w2c = inv(c2w))
    w2c = np.linalg.inv(np.array(c2w))
    
    # Convert to homogeneous coordinates
    ones = np.ones((points_world.shape[0], 1))
    points_world_h = np.hstack([points_world, ones])
    
    # Transform: P_cam = w2c * P_world
    points_cam = (w2c @ points_world_h.T).T # (N, 4)
    
    # Extract xyz
    xyz = points_cam[:, :3]
    
    # 2. Camera -> Pixel (P_uv = K * P_cam)
    uv_z = (np.array(K) @ xyz.T).T # (N, 3)
    
    z = uv_z[:, 2]
    
    # Avoid division by zero
    u = uv_z[:, 0] / (z + 1e-6)
    v = uv_z[:, 1] / (z + 1e-6)
    
    return np.stack([u, v], axis=1), z

def draw_3d_box(img, points_2d, depth, color=(0, 255, 0), thickness=2):
    """
    Draw cube wireframe on image.
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Top
        (4, 5), (5, 6), (6, 7), (7, 4), # Bottom
        (0, 4), (1, 5), (2, 6), (3, 7)  # Pillars
    ]
    
    h, w = img.shape[:2]
    
    if np.mean(depth) < 0:
        print("Warning: Object seems to be behind the camera.")
    
    drawn_lines = 0
    for start_idx, end_idx in edges:
        # Check if points are behind camera
        if depth[start_idx] < 0 or depth[end_idx] < 0:
            continue # Skip clipping lines behind camera
            
        pt1 = tuple(map(int, points_2d[start_idx]))
        pt2 = tuple(map(int, points_2d[end_idx]))
        
        # Check bounds roughly (optional, cv2.line handles OOB gracefully usually)
        # Just draw
        cv2.line(img, pt1, pt2, color, thickness)
        drawn_lines += 1

    if drawn_lines > 0:
        # Draw front face corner indicator
        pt_corner = tuple(map(int, points_2d[0]))
        if 0 <= pt_corner[0] < w and 0 <= pt_corner[1] < h:
            cv2.circle(img, pt_corner, 5, (0, 0, 255), -1) 
    
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_json", type=str, required=True, help="Path to bbox_annotation_frame_XXXX.json")
    parser.add_argument("--meta_json", type=str, required=True, help="Path to scene_meta.json")
    parser.add_argument("--output_path", type=str, default="projection_vis.jpg")
    
    # 【新增参数】控制投影到哪一帧
    parser.add_argument("--target_frame_idx", type=int, default=None, 
                        help="Index of the frame to project onto. If not set, uses the frame where annotation was created.")
    
    args = parser.parse_args()

    # 1. Load Data
    if not os.path.exists(args.annotation_json):
        print(f"Error: Annotation file not found: {args.annotation_json}")
        return
    if not os.path.exists(args.meta_json):
        print(f"Error: Meta file not found: {args.meta_json}")
        return

    with open(args.annotation_json, 'r') as f:
        anno = json.load(f)
    
    with open(args.meta_json, 'r') as f:
        meta = json.load(f)
        
    # 2. Determine Target Frame
    source_frame_idx = anno['frame_idx']
    
    if args.target_frame_idx is not None:
        target_frame_idx = args.target_frame_idx
        print(f"Projection Mode: Cross-Frame (Source: {source_frame_idx} -> Target: {target_frame_idx})")
    else:
        target_frame_idx = source_frame_idx
        print(f"Projection Mode: Same-Frame (Frame {source_frame_idx})")
    
    # Validate index
    if target_frame_idx < 0 or target_frame_idx >= len(meta['frames']):
        print(f"Error: Target frame index {target_frame_idx} out of bounds (Total: {len(meta['frames'])})")
        return
        
    # 3. Get Target Frame Info (Image, Pose, K)
    frame_info = meta['frames'][target_frame_idx]
    
    base_dir = os.path.dirname(args.meta_json)
    img_path = os.path.join(base_dir, frame_info['file_path'])
    
    if not os.path.exists(img_path):
        print(f"Error: Target image not found at {img_path}")
        return
        
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Failed to decode image.")
        return
    print(f"Loaded target image: {img_path} ({img.shape})")

    K = frame_info['intrinsics']
    c2w = frame_info['transform_matrix']
    
    if K is None or c2w is None:
        print("Error: Intrinsics or Pose missing for target frame.")
        return

    # 4. Get BBox in World Coordinates (from Annotation)
    bbox_data = anno['bbox']
    pts_world = get_bbox_corners_world(
        bbox_data['center'],
        bbox_data['extent'],
        bbox_data['rotation']
    )
    
    # 5. Project onto Target Camera
    pts_2d, depth = project_points(pts_world, c2w, K)
    
    # 6. Draw and Save
    img_vis = draw_3d_box(img.copy(), pts_2d, depth)
    
    cv2.imwrite(args.output_path, img_vis)
    print(f"Projection saved to {args.output_path}")

if __name__ == "__main__":
    main()