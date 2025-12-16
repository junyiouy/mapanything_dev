import argparse
import os
import sys
import json
import numpy as np
import torch
import trimesh
from PIL import Image
import os.path as osp
from torchvision import transforms
import cv2
import pandas as pd
import scipy.spatial.transform as spt
import OpenEXR
import Imath
from time import time
import concurrent.futures
from tqdm import tqdm

# Rerun import
try:
    import rerun as rr
except ImportError:
    rr = None

def script_add_rerun_args(parser):
    """添加 Rerun 可视化参数"""
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Connect to Rerun server for live visualization"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save visualization to RRD file"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Stream to stdout (for Rerun viewer)"
    )
    parser.add_argument(
        "--max_points_per_frame",
        type=int,
        default=-1,
        help="Maximum points per frame to log (-1 for no limit)"
    )

# MapAnything imports
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.geometry import colmap_to_opencv_intrinsics

# Add paths (assuming these paths are necessary for MapAnything)
sys.path.append('/wekafs/ict/junyiouy/map-anything')
colmap_reader_path = '/wekafs/ict/junyiouy/map-anything/data_processing/wai_processing/third_party/mvsanywhere/src/mvsanywhere/datasets'
if colmap_reader_path not in sys.path:
    sys.path.append(colmap_reader_path)

# MapAnything utilities (Keep original imports, though some are not used now)
from misc.opendv_utils import transform_points
# from read_write_colmap_model import read_cameras_binary, read_images_binary, read_points3D_binary # 移除 COLMAP 读取

# 引入 Physical AI AV dataset 接口
# 假设 physical_ai_av, DracoPy 库已正确安装
try:
    from physical_ai_av import dataset
    from DracoPy import decode as decode_draco
except ImportError as e:
    print(f"Error: {e}. Please ensure physical_ai_av and DracoPy are installed.")
    sys.exit(1)


# --- 辅助函数：从代码2引入和简化 ---

def decode_draco_pointcloud(draco_bytes):
    """解码Draco压缩的点云数据"""
    pointcloud = decode_draco(draco_bytes)
    return np.asarray(pointcloud.points)

def get_lidar_scan_data(lidar_df, scan_index):
    """
    从预加载的lidar_df获取指定索引的LiDAR时间戳T_lidar
    """
    if scan_index >= len(lidar_df):
        raise ValueError(f"Scan index {scan_index} out of bounds.")
    nearest_scan_row = lidar_df.iloc[scan_index]
    T_lidar = nearest_scan_row['reference_timestamp']
    return T_lidar

def get_camera_frame_and_timestamp_for_lidar(camera_reader, T_lidar):
    """从预加载的camera_reader获取与 T_lidar 时间上最接近的 RGB 帧及其实际时间戳 T_cam"""
    requested_timestamps = np.array([T_lidar])
    frames, T_cam_actual = camera_reader.decode_images_from_timestamps(requested_timestamps)
    return frames[0], T_cam_actual[0]


def get_intrinsics_params(camera_intrinsics_df, camera_name):
    """获取相机内参参数"""
    cam_intrinsics = camera_intrinsics_df.loc[camera_name]
    return {
        'width': int(cam_intrinsics['width']),
        'height': int(cam_intrinsics['height']),
        'cx': cam_intrinsics['cx'],
        'cy': cam_intrinsics['cy'],
        'fw_poly': np.array([
            cam_intrinsics['fw_poly_0'],
            cam_intrinsics['fw_poly_1'],
            cam_intrinsics['fw_poly_2'],
            cam_intrinsics['fw_poly_3'],
            cam_intrinsics['fw_poly_4']
        ])
    }

def get_transform_matrices(egomotion_interp, sensor_extrinsics_df, camera_name, T_lidar):
    """
    计算变换矩阵：返回 T_World_to_Camera 和 T_LiDAR_to_World 矩阵。
    """
    # T_Rig_to_World (R2W) - Ego Motion
    rig_to_world = egomotion_interp(T_lidar).pose.as_matrix()

    # T_LiDAR_to_Rig (L2R) - LiDAR Extrinsics
    lidar_row = sensor_extrinsics_df[sensor_extrinsics_df.index == "lidar_top_360fov"].iloc[0]
    lidar_rotation = spt.Rotation.from_quat([lidar_row['qx'], lidar_row['qy'], lidar_row['qz'], lidar_row['qw']])
    lidar_translation = np.array([lidar_row['x'], lidar_row['y'], lidar_row['z']])
    lidar_to_rig = np.eye(4)
    lidar_to_rig[:3, :3] = lidar_rotation.as_matrix()
    lidar_to_rig[:3, 3] = lidar_translation

    # T_Camera_to_Rig (C2R) - Camera Extrinsics
    camera_row = sensor_extrinsics_df[sensor_extrinsics_df.index == camera_name].iloc[0]
    camera_rotation = spt.Rotation.from_quat([camera_row['qx'], camera_row['qy'], camera_row['qz'], camera_row['qw']])
    camera_translation = np.array([camera_row['x'], camera_row['y'], camera_row['z']])
    camera_to_rig = np.eye(4)
    camera_to_rig[:3, :3] = camera_rotation.as_matrix()
    camera_to_rig[:3, 3] = camera_translation

    # T_LiDAR_to_World (L2W) = R2W @ L2R
    lidar_to_world = rig_to_world @ lidar_to_rig

    # T_World_to_Camera (W2C) = (C2R)^-1 @ (R2W)^-1
    world_to_camera = np.linalg.inv(camera_to_rig) @ np.linalg.inv(rig_to_world)

    return world_to_camera, lidar_to_world

def undistort_f_theta(image, intrinsics, target_fx=None):
    """
    【去畸变函数】遵循 f-theta 逆投影逻辑 (Pinhole -> f-theta 映射)。
    Returns:
        tuple: (undistorted_image, K_undistorted)
    """
    H, W = int(intrinsics['height']), int(intrinsics['width'])
    u0, v0 = float(intrinsics['cx']), float(intrinsics['cy'])

    fw_poly = intrinsics['fw_poly']
    f = target_fx if target_fx is not None else fw_poly[1]

    K_undistorted = np.array([
        [f, 0, u0],
        [0, f, v0],
        [0, 0, 1]
    ], dtype=np.float64)

    K_inv = np.linalg.inv(K_undistorted)

    U_prime, V_prime = np.meshgrid(np.arange(W), np.arange(H))
    P_prime_homogeneous = np.stack([U_prime, V_prime, np.ones_like(U_prime)], axis=-1).reshape(-1, 3)

    R_ideal = (K_inv @ P_prime_homogeneous.T).T 

    Rx, Ry, Rz = R_ideal[:, 0], R_ideal[:, 1], R_ideal[:, 2]
    R_norm = np.linalg.norm(R_ideal, axis=1)
    R_norm[R_norm == 0] = 1e-6

    cos_theta = Rz / R_norm
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    r_distorted = np.polyval(fw_poly[::-1], theta)

    R_p_norm = np.sqrt(Rx**2 + Ry**2)
    R_p_norm[R_p_norm == 0] = 1e-6

    map_x = u0 + r_distorted * (Rx / R_p_norm)
    map_y = v0 + r_distorted * (Ry / R_p_norm)

    map_x = map_x.reshape(H, W).astype(np.float32)
    map_y = map_y.reshape(H, W).astype(np.float32)

    undistorted_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return undistorted_image, K_undistorted


# --- MapAnything 流程辅助函数 (保持不变) ---

# def load_colmap_data(colmap_path): # 移除

def unnormalize_and_save_image(tensor, save_path):
    """Reverses ImageNet normalization and saves tensor as image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    img_unorm = tensor * std + mean
    img_pil = transforms.ToPILImage()(img_unorm.clamp(0, 1).cpu())
    img_pil.save(save_path)

def save_preprocessed_batch_images(processed_views, output_dir, prefix_start_idx=0):
    """Saves images from processed_views to disk."""
    img_dir = osp.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    saved_paths = []
    for i, view in enumerate(processed_views):
        file_name = f"frame_{prefix_start_idx + i:06d}.jpg"
        save_path = osp.join(img_dir, file_name)
        unnormalize_and_save_image(view['img'].squeeze(0), save_path)
        saved_paths.append(osp.join("images", file_name))
    return saved_paths

# --- 核心函数：视图准备 (重写) ---

def load_raw_data_for_scan(args):
    (scan_idx, lidar_df, camera_reader, egomotion_interp, sensor_extrinsics_df, camera_name) = args

    T_lidar = get_lidar_scan_data(lidar_df, scan_idx)
    rgb_frame_raw, T_cam_actual = get_camera_frame_and_timestamp_for_lidar(camera_reader, T_lidar)

    world_to_camera, _ = get_transform_matrices(egomotion_interp, sensor_extrinsics_df, camera_name, T_lidar)
    camera_to_world = np.linalg.inv(world_to_camera).astype(np.float32)

    return {
        'scan_idx': scan_idx,
        'T_lidar': T_lidar,
        'rgb_frame_raw': rgb_frame_raw,
        'camera_to_world': camera_to_world
    }

def process_image_data(args):
    """并行处理图像去畸变和缩放（进程安全）"""
    (raw_data, intrinsics_params, K_resized, new_W, new_H, camera_name, modalities_to_load) = args

    scan_idx = raw_data['scan_idx']
    T_lidar = raw_data['T_lidar']
    rgb_frame_raw = raw_data['rgb_frame_raw']
    camera_to_world = raw_data['camera_to_world']

    # 去畸变和缩放
    undistorted_rgb, _ = undistort_f_theta(rgb_frame_raw, intrinsics_params, target_fx=None)
    resized_rgb = cv2.resize(undistorted_rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    # 构建view
    view = {
        'img': Image.fromarray(resized_rgb, mode='RGB'),
        'is_metric_scale': torch.tensor([True]),
        'camera_name': camera_name,
        'image_name': f"{camera_name}_{T_lidar}.png",
        'T_lidar': T_lidar,
    }

    if 'intrinsics' in modalities_to_load:
        view['intrinsics'] = K_resized.astype(np.float32)

    if 'poses' in modalities_to_load:
        view['camera_poses'] = camera_to_world

    return view

def prepare_views_from_physical_ai_av_data(
    lidar_df, camera_reader, intrinsics_params, egomotion_interp, sensor_extrinsics_df,
    camera_name, modalities_to_load, max_scans_to_process, target_res=512
):
    """
    基于 Physical AI AV 数据集准备 MapAnything 的输入视图列表。
    执行去畸变、缩放和内参/姿态计算。
    """
    views = []

    # 1. 预计算缩放参数（只计算一次）
    if len(lidar_df) == 0:
        print("Error: LiDAR DataFrame is empty.")
        return []

    T_lidar_sample = get_lidar_scan_data(lidar_df, 0)
    # 图像是 RGB 格式
    rgb_frame_raw, _ = get_camera_frame_and_timestamp_for_lidar(camera_reader, T_lidar_sample)
    undistorted_rgb, K_undistorted = undistort_f_theta(rgb_frame_raw, intrinsics_params, target_fx=None)
    H, W = undistorted_rgb.shape[:2]

    # 根据长边缩放至 target_res (例如 512)
    scale = target_res / max(H, W)
    new_H, new_W = int(H * scale + 0.5), int(W * scale + 0.5)

    K_resized = K_undistorted.copy()
    K_resized[0, 0] *= scale
    K_resized[1, 1] *= scale
    K_resized[0, 2] *= scale
    K_resized[1, 2] *= scale

    print(f"Target resolution: ({new_W}, {new_H}). Resized K: {K_resized.tolist()}")

    num_scans = min(len(lidar_df), max_scans_to_process)


    print(f"Loading raw data for {num_scans} scans...")
    raw_data_list = []
    for i in tqdm(range(num_scans), desc="Loading raw data"):
        args = (i, lidar_df, camera_reader, egomotion_interp, sensor_extrinsics_df, camera_name)
        raw_data = load_raw_data_for_scan(args)
        raw_data_list.append(raw_data)


    print(f"Processing images for {num_scans} scans with parallel execution...")
    image_args_list = [(raw_data, intrinsics_params, K_resized, new_W, new_H, camera_name, modalities_to_load)
                      for raw_data in raw_data_list]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, num_scans)) as executor:
        views = list(tqdm(executor.map(process_image_data, image_args_list),
                         total=num_scans, desc="Processing images"))
        
    # sort views by T_lidar to maintain order
    views.sort(key=lambda v: v['T_lidar'])
    # delete T_lidar from views as it's no longer needed
    for v in views:
        del v['T_lidar']

    return views, new_W, new_H


# --- 核心函数：处理 Chunk (修改对齐逻辑) ---

def process_and_save_chunk(model, views, output_dir, chunk_name, global_img_counter, inference_modalities, sampling_rate=0.2, align_context=None, new_W=None, new_H=None, max_points_per_frame=-1):
    """
    Processes a chunk and returns metadata frames and next context.
    🚨 关键修改: 忽略对齐上下文，使用输入的精确姿态。
    """
    if not views: return None, None

    # --- 1. Preprocess ---
    print(f"  Pre-processing {len(views)} views...")
    for v in views: v['data_norm_type'] = [model.encoder.data_norm_type]
    processed_views = preprocess_inputs(views)

    # --- 2. Save Preprocessed Images ---
    # processed_views 中的 img 是归一化后的 tensor
    rel_img_paths = save_preprocessed_batch_images(processed_views, output_dir, global_img_counter)

    # --- 3. Prepare Inference Input ---
    inference_views = []
    for pv in processed_views:
        iv = {}
        iv['img'] = pv['img']
        iv['data_norm_type'] = pv['data_norm_type']
        iv['is_metric_scale'] = pv['is_metric_scale']
        if 'poses' in inference_modalities and 'camera_poses' in pv:
            iv['camera_poses'] = pv['camera_poses'] # 输入的精确姿态
        if 'intrinsics' in inference_modalities and 'intrinsics' in pv:
            iv['intrinsics'] = pv['intrinsics']
        if 'depth' in inference_modalities and 'depth_z' in pv:
            iv['depth_z'] = pv['depth_z']
        inference_views.append(iv)

    # --- 4. Run Inference ---
    print(f"  Running inference for {chunk_name}...")
    predictions = model.infer(
        inference_views,
        memory_efficient_inference=True,
        use_amp=True, amp_dtype="bf16",
        apply_mask=True, mask_edges=False, confidence_percentile=0
    )

    # --- 5. Alignment Calc ---

    all_raw_poses = []
    for pred in predictions:
        if 'camera_poses' in pred:
            all_raw_poses.append(pred['camera_poses'].cpu().numpy())

    if all_raw_poses:
        raw_model_poses = np.concatenate(all_raw_poses, axis=0)
    else:
        raw_model_poses = None

    ref_camera_name = align_context.get('ref_camera_name') if align_context else None
    curr_ref_indices = []
    if ref_camera_name:
        for idx, v in enumerate(views):
            if v['camera_name'] == ref_camera_name:
                curr_ref_indices.append(idx)

    cum_T = align_context.get('cum_T', np.eye(4)) if align_context else np.eye(4)

    if align_context and align_context.get('do_alignment') and raw_model_poses is not None and curr_ref_indices:
        prev_raw_pose = align_context['prev_raw_ref_pose']
        if len(curr_ref_indices) > 0:
            curr_raw_pose = raw_model_poses[curr_ref_indices[0]]
            T_delta = prev_raw_pose @ np.linalg.inv(curr_raw_pose)
            cum_T = cum_T @ T_delta
            print(f"  Aligned {chunk_name} using {ref_camera_name}")

    # --- 6. Process & Save Points ---
    points_dir = osp.join(output_dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    metadata_frames = []
    current_global_idx = global_img_counter
    current_view_idx = 0

    if raw_model_poses is not None:
        final_aligned_poses = cum_T @ raw_model_poses
    else:
        final_aligned_poses = np.stack([v['camera_poses'] for v in views])
    
    for pred in predictions:
        pts3d_batch = pred['pts3d'].cpu().numpy()      
        mask_batch = pred['mask'].cpu().numpy()       
        img_no_norm_batch = pred['img_no_norm'].cpu().numpy() 
        
        batch_size = pts3d_batch.shape[0]
        for b in range(batch_size):
            valid_mask = mask_batch[b, ..., 0] > 0.5
            if sampling_rate < 1.0:
                random_mask = np.random.rand(*valid_mask.shape) < sampling_rate
                final_mask = np.logical_and(valid_mask, random_mask)
            else:
                final_mask = valid_mask
                
            pts = pts3d_batch[b][final_mask] # MapAnything 推理的 3D 点 (在 C 坐标系下)
            colors = img_no_norm_batch[b][final_mask]
            aligned_pts = transform_points(cum_T, pts)
            
            ply_filename = f"frame_{current_global_idx:06d}.ply"
            ply_path = osp.join(points_dir, ply_filename)
            
            if len(aligned_pts) > 0:
                pcd = trimesh.PointCloud(vertices=aligned_pts, colors=(colors * 255).astype(np.uint8))
                pcd.export(ply_path)
            else:
                trimesh.PointCloud(vertices=[]).export(ply_path)

            # Rerun logging
            if rr:
                # Limit points if needed
                if max_points_per_frame > 0 and len(aligned_pts) > max_points_per_frame:
                    indices = np.random.choice(len(aligned_pts), max_points_per_frame, replace=False)
                    log_pts = aligned_pts[indices]
                    log_colors = colors[indices]
                else:
                    log_pts = aligned_pts
                    log_colors = colors

                rr.set_time("frame", sequence=current_global_idx)
                rr.log("world/pointcloud", rr.Points3D(log_pts, colors=(log_colors * 255).astype(np.uint8), radii=0.01))

                # Camera pose
                
                translation = final_aligned_poses[current_view_idx][:3, 3].tolist()
                rotation_matrix = final_aligned_poses[current_view_idx][:3, :3].tolist()
                rr.log(f"world/camera_{current_global_idx}", rr.Transform3D(translation=translation, mat3x3=rotation_matrix))

                # Pinhole intrinsics
                rr.log(f"world/camera_{current_global_idx}", rr.Pinhole(image_from_camera=np.array(processed_views[current_view_idx]['intrinsics'].tolist()), height=new_H, width=new_W, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=1.0))

                # Image
                image = img_no_norm_batch[b]
                if image is not None:
                    # Convert from (H, W, C) to (C, H, W) if needed, but img_no_norm_batch should already be in correct format
                    rr.log(f"world/camera_{current_global_idx}", rr.Image(image))


            # Metadata collection
            K = processed_views[current_view_idx]['intrinsics'].tolist()
            c2w = final_aligned_poses[current_view_idx].tolist()
            
            metadata_frames.append({
                "file_path": rel_img_paths[current_view_idx],
                "ply_path": osp.join("points", ply_filename),
                "transform_matrix": c2w,
                "intrinsics": K,
                "width": new_W,
                "height": new_H,
                "chunk_id": chunk_name
            })

            current_global_idx += 1
            current_view_idx += 1

    # --- 7. Next Context ---
    next_context = {
        "cum_T": cum_T,
        "ref_camera_name": ref_camera_name,
        "overlap_size": align_context.get('overlap_size', 1) if align_context else 1,
        "do_alignment": True
    }

    if raw_model_poses is not None and curr_ref_indices:
        overlap_size = next_context['overlap_size']
        if len(curr_ref_indices) >= overlap_size:
            idx_for_next_alignment = curr_ref_indices[-overlap_size]
            next_context['prev_raw_ref_pose'] = raw_model_poses[idx_for_next_alignment]
        else:
             next_context['prev_raw_ref_pose'] = raw_model_poses[curr_ref_indices[-1]]
    
    return metadata_frames, next_context


def main():
    parser = argparse.ArgumentParser()
    # 替换 data_root 为 dataset_path
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path to the Physical AI AV dataset.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--modalities", type=str, nargs='+', default=['image', 'intrinsics', 'poses'])
    parser.add_argument("--no_metric_scale", action="store_true") # 忽略，PAAV 数据集始终是度量尺度的
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_timestamps", action="store_true") # 忽略，我们按时间顺序分块
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--overlap_size", type=int, default=1)
    parser.add_argument("--ref_camera", type=str, default=None)
    parser.add_argument("--rig_ranges", type=str, default=None) # 忽略分组
    parser.add_argument("--point_sampling_rate", type=float, default=0.2)
    # 引入 Physical AI AV 特有参数
    parser.add_argument("--clip_id", type=str, required=True, help="The ID of the Physical AI AV clip to process.")
    parser.add_argument("--camera_name", type=str, default="camera_front_wide_120fov", help="Camera sensor name.")
    parser.add_argument("--max_scans", type=int, default=10000, help="Maximum number of LiDAR scans to process for the clip.")
    parser.add_argument("--target_resolution", type=int, default=512, help="Target resolution for the longest image side after undistortion and resizing.")

    # Add Rerun arguments
    script_add_rerun_args(parser)

    args = parser.parse_args()
    device = torch.device(args.device)

    # Initialize Rerun if any visualization option is specified
    if rr and (args.connect or args.save or args.stdout):
        rr.init(f"PhysicalAI_{args.clip_id}")
        if args.connect:
            rr.spawn()
        if args.save:
            rr.save(args.save)
        if args.stdout:
            # For stdout, Rerun doesn't have direct support in current version
            # You might need to implement custom logging or use save to temp file
            pass
    
    # 🚨 替换 COLMAP 加载
    print(f"Initializing Physical AI AV dataset at: {args.dataset_path}")
    ds = dataset.PhysicalAIAVDatasetInterface(local_dir=args.dataset_path)

    # 1. 加载 Clip 数据
    try:
        lidar_data_dict = ds.get_clip_feature(args.clip_id, "lidar_top_360fov", maybe_stream=True)
        # 提取 DataFrame
        lidar_df = next(v for v in lidar_data_dict.values() 
                        if isinstance(v, pd.DataFrame) and 'reference_timestamp' in v.columns)
        
        camera_reader = ds.get_clip_feature(args.clip_id, args.camera_name, maybe_stream=True)
        egomotion_interp = ds.get_clip_feature(args.clip_id, "egomotion", maybe_stream=True)
        sensor_extrinsics_df = ds.get_clip_feature(args.clip_id, "sensor_extrinsics", maybe_stream=True)
        camera_intrinsics_df = ds.get_clip_feature(args.clip_id, "camera_intrinsics", maybe_stream=True)
        
        intrinsics_params = get_intrinsics_params(camera_intrinsics_df, args.camera_name)
    except Exception as e:
        print(f"Error loading clip data for {args.clip_id}: {e}")
        return

    print("Loading MapAnything model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    # 2. 准备数据
    group_name = args.clip_id
    group_out_dir = osp.join(args.output_dir, group_name)
    os.makedirs(group_out_dir, exist_ok=True)
    
    print(f"\n=== Processing Group: {group_name} ===")
    
    loading_modalities = list(set(args.modalities + ['poses', 'intrinsics']))
    inference_modalities = args.modalities
    
    # 🚨 调用新的视图准备函数
    views, new_W, new_H = prepare_views_from_physical_ai_av_data(
        lidar_df, camera_reader, intrinsics_params, egomotion_interp, sensor_extrinsics_df,
        args.camera_name, loading_modalities, args.max_scans, args.target_resolution
    )

    # 3. 分块处理
    num_views = len(views)
    chunks = []
    step = args.chunk_size - args.overlap_size
    for start_idx in range(0, num_views, step):
        end_idx = min(start_idx + args.chunk_size, num_views)
        chunk_items = views[start_idx:end_idx]
        chunks.append(chunk_items)
        if end_idx == num_views: break

    print(f"Created {len(chunks)} chunks from {num_views} views")

    global_img_counter = 0

    # 初始上下文
    prev_context = {
        "cum_T": np.eye(4),
        "ref_camera_name": args.ref_camera,
        "overlap_size": args.overlap_size,
        "do_alignment": False,
        "prev_raw_ref_pose": None
    }

    all_frames_metadata = []
    camera_positions = []

    for i, chunk_items in enumerate(chunks):
        if not chunk_items: continue

        chunk_name = f"chunk_{i:03d}"
        print(f"Processing {chunk_name} ({len(chunk_items)} images)...")

        # 启用对齐如果有重叠且指定了ref_camera
        if i > 0 and args.overlap_size > 0 and args.ref_camera:
            prev_context['do_alignment'] = True
        else:
            prev_context['do_alignment'] = False

        chunk_meta_frames, prev_context = process_and_save_chunk(
            model, chunk_items, group_out_dir, chunk_name,
            global_img_counter, inference_modalities,
            sampling_rate=args.point_sampling_rate,
            align_context=prev_context, new_W=new_W, new_H=new_H,
            max_points_per_frame=args.max_points_per_frame
        )

        if chunk_meta_frames:
            all_frames_metadata.extend(chunk_meta_frames)
            for frame in chunk_meta_frames:
                camera_positions.append(np.array(frame['transform_matrix'])[:3, 3])
            global_img_counter += len(chunk_items)
        else:
            print("Chunk processing failed.")
            break

    # 4. 保存元数据
    scene_meta = {
        "group_name": group_name,
        "total_frames": len(all_frames_metadata),
        "frames": all_frames_metadata
    }
    
    scene_meta_path = osp.join(group_out_dir, "scene_meta.json")
    with open(scene_meta_path, 'w') as f:
        json.dump(scene_meta, f, indent=2)
    print(f"Saved merged metadata to {scene_meta_path}")

    # Log camera trajectory
    if rr and camera_positions:
        camera_colors = []
        for i in range(len(camera_positions)):
            t = i / (len(camera_positions)-1) if len(camera_positions)>1 else 0
            r = int(255 * t)
            g = 0
            b = int(255 * (1 - t))
            camera_colors.append([r, g, b])
        rr.log("world/camera_trajectory", rr.Points3D(camera_positions, colors=camera_colors, radii=0.05))
        if len(camera_positions) > 1:
            rr.log("world/camera_trajectory", rr.LineStrips3D([camera_positions], colors=[camera_colors[0]]))
        print(f"Logged camera trajectory with {len(camera_positions)} positions")

    print("\nProcessing completed!")

if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=0 python /wekafs/ict/junyiouy/map-anything/scripts/physicai_chunk_infer/infer_physicai_save_components.py --dataset_path /wekafs/ict/junyiouy/physical_ai_av_dataset --output_dir output_physicai --clip_id d4190223-c9fd-4671-a0f1-cc7424171971 --point_sampling_rate 0.6 --ref_camera camera_front_wide_120fov 
# python /wekafs/ict/junyiouy/map-anything/scripts/physicai_chunk_infer/server_annotator.py --meta_json output_physicai/d4190223-c9fd-4671-a0f1-cc7424171971/scene_meta.json --port 8091 
# CUDA_VISIBLE_DEVICES=1 python /wekafs/ict/junyiouy/map-anything/scripts/physicai_chunk_infer/infer_physicai_save_components.py --dataset_path /wekafs/ict/junyiouy/physical_ai_av_dataset --output_dir output_physicai --clip_id cdf494d1-cfaa-4fc3-95e4-86fc4662b092 --point_sampling_rate 0.6 --ref_camera camera_front_wide_120fov
# python /wekafs/ict/junyiouy/map-anything/scripts/physicai_chunk_infer/server_annotator.py --meta_json output_physicai/cdf494d1-cfaa-4fc3-95e4-86fc4662b092/scene_meta.json --port 8091
# export CLIP_ID=d4190223-c9fd-4671-a0f1-cc7424171971
# 95c10498-2dfb-43ff-b469-3671dc5b8b07
# CUDA_VISIBLE_DEVICES=0 python /wekafs/ict/junyiouy/map-anything/scripts/physicai_chunk_infer/infer_physicai_save_components.py --dataset_path /wekafs/ict/junyiouy/physical_ai_av_dataset --output_dir output_physicai --clip_id $CLIP_ID --point_sampling_rate 0.6 --ref_camera camera_front_wide_120fov --save /wekafs/ict/junyiouy/map-anything/output_physicai/${CLIP_ID}_recon.rrd
# python /wekafs/ict/junyiouy/map-anything/scripts/physicai_chunk_infer/server_annotator.py --meta_json output_physicai/${CLIP_ID}/scene_meta.json --port 8095 