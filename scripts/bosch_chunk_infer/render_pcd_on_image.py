import argparse
import json
import os
import cv2
import numpy as np
import trimesh

def project_points(points_world, c2w, K):
    """
    Project World Points to Image Pixels.
    Returns: uv coordinates (N, 2) and depth values (N,)
    """
    # 1. World -> Camera
    w2c = np.linalg.inv(np.array(c2w))
    ones = np.ones((points_world.shape[0], 1))
    points_world_h = np.hstack([points_world, ones])
    points_cam = (w2c @ points_world_h.T).T
    
    xyz = points_cam[:, :3]
    z = xyz[:, 2]
    
    # 2. Camera -> Pixel
    uv_z = (np.array(K) @ xyz.T).T
    
    u = uv_z[:, 0] / (z + 1e-6)
    v = uv_z[:, 1] / (z + 1e-6)
    
    return np.stack([u, v], axis=1), z

def get_points_in_bbox_mask(points, bbox_data):
    """
    Check which points are inside the oriented bounding box.
    Returns: boolean mask (N,)
    """
    center = np.array(bbox_data['center'])
    extent = np.array(bbox_data['extent'])
    rotation = np.array(bbox_data['rotation']) # (3, 3) rotation matrix local->world
    
    # Transform points to BBox Local Coordinate System
    # P_local = (P_world - Center) @ R
    diff = points - center
    points_local = diff @ rotation 
    
    # Check bounds
    half_extent = extent / 2.0
    mask = np.all(np.abs(points_local) <= half_extent, axis=1)
    return mask

def main():
    parser = argparse.ArgumentParser(description="Render Point Cloud INSIDE BBox only (as Red)")
    parser.add_argument("--meta_json", type=str, required=True, help="Path to scene_meta.json")
    parser.add_argument("--annotation_json", type=str, required=True, help="Path to bbox_annotation.json")
    parser.add_argument("--target_frame_idx", type=int, required=True, help="Index of the IMAGE to project onto")
    parser.add_argument("--source_frame_idx", type=int, default=None, 
                        help="Index of the PLY to use. If None, uses the target frame's PLY.")
    parser.add_argument("--output_path", type=str, default="pcd_bbox_only_red.jpg")
    
    # Rendering options
    parser.add_argument("--point_size", type=int, default=2, help="Radius of rendered points")
    parser.add_argument("--opacity", type=float, default=0.7, help="Opacity of the overlay")
    
    args = parser.parse_args()

    # 1. Load Metadata & Annotation
    if not os.path.exists(args.meta_json): print(f"Error: {args.meta_json} not found"); return
    with open(args.meta_json, 'r') as f: meta = json.load(f)
    frames = meta['frames']
    base_dir = os.path.dirname(args.meta_json)
    
    if not os.path.exists(args.annotation_json): print(f"Error: {args.annotation_json} not found"); return
    with open(args.annotation_json, 'r') as f: anno = json.load(f)

    # 2. Get Target Frame Info
    if args.target_frame_idx >= len(frames): print("Target index out of bounds"); return
    target_info = frames[args.target_frame_idx]
    
    img_path = os.path.join(base_dir, target_info['file_path'])
    c2w = np.array(target_info['transform_matrix'])
    K = np.array(target_info['intrinsics'])
    
    img = cv2.imread(img_path)
    if img is None: print("Image not found"); return
    h, w = img.shape[:2]

    # 3. Load Source Point Cloud
    src_idx = args.source_frame_idx if args.source_frame_idx is not None else args.target_frame_idx
    source_info = frames[src_idx]
    ply_path = os.path.join(base_dir, source_info['ply_path'])
    
    print(f"Loading PCD frame {src_idx}...")
    pcd = trimesh.load(ply_path)
    points = pcd.vertices

    # 4. Calculate BBox Inclusion Mask
    print("Calculating BBox inclusion...")
    inside_bbox_mask = get_points_in_bbox_mask(points, anno['bbox'])
    num_inside = np.sum(inside_bbox_mask)
    print(f"Points inside BBox: {num_inside} / {len(points)}")

    if num_inside == 0:
        print("Warning: No points found inside the BBox. Outputting original image.")
        cv2.imwrite(args.output_path, img)
        return

    # 5. Project Points
    print("Projecting points...")
    uv, depth = project_points(points, c2w, K)
    
    # 6. Filter Valid Points (Frustum AND BBox Inclusion)
    # We only keep points that are on screen AND inside the BBox
    frustum_mask = (depth > 0.1) & \
                   (uv[:, 0] >= 0) & (uv[:, 0] < w) & \
                   (uv[:, 1] >= 0) & (uv[:, 1] < h)
    
    final_mask = frustum_mask & inside_bbox_mask # 【核心修改】取交集
    
    uv_valid = uv[final_mask]
    depth_valid = depth[final_mask]
    
    print(f"Points to render (inside BBox & visible): {len(uv_valid)}")

    # 7. Render (Painter's Algorithm)
    sort_indices = np.argsort(depth_valid)[::-1]
    uv_sorted = uv_valid[sort_indices]
    
    overlay = img.copy()
    red_color = (0, 0, 255) # BGR for Red

    for i in range(len(uv_sorted)):
        pt = tuple(map(int, uv_sorted[i]))
        # 【核心修改】统一使用红色
        cv2.circle(overlay, pt, args.point_size, red_color, -1)

    # 8. Blend
    final_img = cv2.addWeighted(overlay, args.opacity, img, 1 - args.opacity, 0)
    
    cv2.imwrite(args.output_path, final_img)
    print(f"Saved: {args.output_path}")

if __name__ == "__main__":
    main()