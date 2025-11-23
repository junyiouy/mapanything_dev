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

# MapAnything imports
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.geometry import colmap_to_opencv_intrinsics

# Add paths
sys.path.append('/wekafs/ict/junyiouy/map-anything')
colmap_reader_path = '/wekafs/ict/junyiouy/map-anything/data_processing/wai_processing/third_party/mvsanywhere/src/mvsanywhere/datasets'
if colmap_reader_path not in sys.path:
    sys.path.append(colmap_reader_path)

from misc.opendv_utils import transform_points
from read_write_colmap_model import read_cameras_binary, read_images_binary, read_points3D_binary


# --- Helper Functions ---

def load_colmap_data(colmap_path):
    print(f"Loading COLMAP data from: {colmap_path}")
    cameras = read_cameras_binary(osp.join(colmap_path, 'cameras.bin'))
    images = read_images_binary(osp.join(colmap_path, 'images.bin'))
    points3D = read_points3D_binary(osp.join(colmap_path, 'points3D.bin'))
    return cameras, images, points3D

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

def prepare_views_from_images(images_base_dir, cameras, image_list, modalities_to_load, metric_scale=True):
    """Converts COLMAP image objects to MapAnything input views."""
    views = []
    for img_data in image_list:
        img_obj = img_data['img']
        cam_id = img_obj.camera_id
        if cam_id not in cameras: continue
        camera = cameras[cam_id]
        
        image_path = osp.join(images_base_dir, img_obj.name)
        if not osp.exists(image_path): continue
        
        pil_img = Image.open(image_path).convert('RGB')
        
        view = {
            'img': pil_img,
            'is_metric_scale': torch.tensor([metric_scale]),
            'camera_name': img_data.get('camera_name', 'unknown'),
            'image_name': osp.basename(img_obj.name)
        }

        if 'intrinsics' in modalities_to_load:
            if camera.model == "PINHOLE":
                fx, fy, cx, cy = camera.params
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            elif camera.model == "SIMPLE_PINHOLE":
                f, cx, cy = camera.params
                K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
            else:
                continue 
            view['intrinsics'] = colmap_to_opencv_intrinsics(K).astype(np.float32)

        if 'poses' in modalities_to_load:
            R = img_obj.qvec2rotmat()
            t = img_obj.tvec
            w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
            c2w = np.linalg.inv(w2c)
            c2w[1, 1] *= -1; c2w[2, 2] *= -1 
            view['camera_poses'] = c2w.astype(np.float32)

        views.append(view)
    return views

def process_and_save_chunk(model, views, output_dir, chunk_name, global_img_counter, inference_modalities, sampling_rate=0.2, align_context=None):
    """
    Processes a chunk and returns metadata frames and next context.
    DOES NOT save json file anymore.
    """
    if not views: return None, None

    # --- 1. Preprocess ---
    print(f"  Pre-processing {len(views)} views...")
    for v in views: v['data_norm_type'] = [model.encoder.data_norm_type]
    processed_views = preprocess_inputs(views)

    # --- 2. Save Preprocessed Images ---
    rel_img_paths = save_preprocessed_batch_images(processed_views, output_dir, global_img_counter)

    # --- 3. Prepare Inference Input ---
    inference_views = []
    for pv in processed_views:
        iv = {}
        iv['img'] = pv['img']
        iv['data_norm_type'] = pv['data_norm_type']
        iv['is_metric_scale'] = pv['is_metric_scale']
        if 'poses' in inference_modalities and 'camera_poses' in pv:
            iv['camera_poses'] = pv['camera_poses']
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
        apply_mask=True, mask_edges=True, confidence_percentile=10
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
        final_aligned_poses = np.stack([np.eye(4)] * len(views))

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
                
            pts = pts3d_batch[b][final_mask]
            colors = img_no_norm_batch[b][final_mask]
            aligned_pts = transform_points(cum_T, pts)
            
            ply_filename = f"frame_{current_global_idx:06d}.ply"
            ply_path = osp.join(points_dir, ply_filename)
            
            if len(aligned_pts) > 0:
                pcd = trimesh.PointCloud(vertices=aligned_pts, colors=(colors * 255).astype(np.uint8))
                pcd.export(ply_path)
            else:
                trimesh.PointCloud(vertices=[]).export(ply_path)

            # Metadata collection
            K = processed_views[current_view_idx]['intrinsics'].squeeze(0).cpu().numpy().tolist() \
                if 'intrinsics' in processed_views[current_view_idx] else None
            c2w = final_aligned_poses[current_view_idx].tolist()
            
            metadata_frames.append({
                "file_path": rel_img_paths[current_view_idx], 
                "ply_path": osp.join("points", ply_filename), 
                "transform_matrix": c2w,
                "intrinsics": K,
                "chunk_id": chunk_name # Helpful for debug
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

# --- Grouping Logic ---
def get_grouped_images(images, cameras, rig_ranges=None, chunk_timestamps=False):
    # ... (Keep your existing logic here unchanged) ...
    # For brevity, reusing previous implementation
    grouped_data = {}
    temp_rig_groups = {}
    if rig_ranges:
        ranges = []
        for r in rig_ranges.split(','):
            s, e = map(int, r.strip().split('-'))
            ranges.append((s, e))
        for img_id, img in images.items():
            parts = img.name.split('/')
            if len(parts) >= 3:
                try:
                    rig_num = int(parts[-3].replace('rig', ''))
                    cam_name = parts[-2]
                    for start, end in ranges:
                        if start <= rig_num <= end:
                            g_name = f"rig{start:06d}-{end:06d}"
                            if g_name not in temp_rig_groups: temp_rig_groups[g_name] = []
                            temp_rig_groups[g_name].append({'img': img, 'camera_name': cam_name, 'sort_key': img.name})
                            break
                except: pass
    else:
        temp_rig_groups["default"] = []
        for img_id, img in images.items():
            cam_name = img.name.split('/')[-2] if '/' in img.name else "unknown"
            temp_rig_groups["default"].append({'img': img, 'camera_name': cam_name, 'sort_key': img.name})

    for rig_group_name, items in temp_rig_groups.items():
        if not chunk_timestamps:
            items.sort(key=lambda x: x['sort_key'])
            grouped_data[rig_group_name] = [items]
        else:
            ts_map = {}
            for item in items:
                parts = item['img'].name.split('/')
                ts = parts[-3] if len(parts) >= 3 else parts[-1]
                if ts not in ts_map: ts_map[ts] = []
                ts_map[ts].append(item)
            sorted_ts = sorted(ts_map.keys())
            grouped_data[rig_group_name] = [ts_map[ts] for ts in sorted_ts]
    return grouped_data

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--modalities", type=str, nargs='+', default=['image', 'intrinsics', 'poses'])
    parser.add_argument("--no_metric_scale", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_timestamps", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--overlap_size", type=int, default=1)
    parser.add_argument("--ref_camera", type=str, default=None)
    parser.add_argument("--rig_ranges", type=str, default=None)
    parser.add_argument("--point_sampling_rate", type=float, default=0.2)
    
    args = parser.parse_args()
    device = torch.device(args.device)
    colmap_path = osp.join(args.data_root, 'sparse', '0')
    images_base_dir = osp.join(args.data_root, 'images')
    
    cameras, images, points3D = load_colmap_data(colmap_path)
    
    print("Loading MapAnything model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    grouped_data = get_grouped_images(images, cameras, args.rig_ranges, args.chunk_timestamps)
    loading_modalities = list(set(args.modalities + ['poses', 'intrinsics']))
    inference_modalities = args.modalities

    for group_name, time_slices in grouped_data.items():
        print(f"\n=== Processing Group: {group_name} ===")
        group_out_dir = osp.join(args.output_dir, group_name)
        os.makedirs(group_out_dir, exist_ok=True)
        
        # ... (Chunk generation logic same as before) ...
        if not args.chunk_timestamps:
            all_items_flat = []
            for ts_list in time_slices: all_items_flat.extend(ts_list)
            chunks = [all_items_flat]
            is_temporal = False
        else:
            num_timestamps = len(time_slices)
            chunks = []
            step = args.chunk_size - args.overlap_size
            for start_idx in range(0, num_timestamps, step):
                end_idx = min(start_idx + args.chunk_size, num_timestamps)
                chunk_items = []
                for i in range(start_idx, end_idx):
                    chunk_items.extend(time_slices[i])
                chunks.append(chunk_items)
                if end_idx == num_timestamps: break
            is_temporal = True

        global_img_counter = 0 
        
        prev_context = {
            "cum_T": np.eye(4),
            "ref_camera_name": args.ref_camera,
            "overlap_size": args.overlap_size,
            "do_alignment": False 
        }

        # 【关键修改】用于收集所有帧的列表
        all_frames_metadata = []

        for i, chunk_items in enumerate(chunks):
            chunk_name = f"chunk_{i:03d}"
            print(f"Processing {chunk_name} ({len(chunk_items)} images)...")
            
            views = prepare_views_from_images(
                images_base_dir, cameras, chunk_items, 
                loading_modalities, not args.no_metric_scale
            )
            
            if is_temporal and i > 0 and args.ref_camera:
                prev_context['do_alignment'] = True
            else:
                prev_context['do_alignment'] = False

            # 获取返回的元数据，而不是在函数内保存
            chunk_meta_frames, prev_context = process_and_save_chunk(
                model, views, group_out_dir, chunk_name, 
                global_img_counter, inference_modalities, 
                sampling_rate=args.point_sampling_rate,
                align_context=prev_context
            )
            
            if chunk_meta_frames:
                all_frames_metadata.extend(chunk_meta_frames)
                global_img_counter += len(views)
            else:
                print("Chunk processing failed.")
                break

        # 【关键修改】循环结束后统一保存
        scene_meta = {
            "group_name": group_name,
            "total_frames": len(all_frames_metadata),
            "frames": all_frames_metadata
        }
        
        scene_meta_path = osp.join(group_out_dir, "scene_meta.json")
        with open(scene_meta_path, 'w') as f:
            json.dump(scene_meta, f, indent=2)
        print(f"Saved merged metadata to {scene_meta_path}")

    print("\nProcessing completed!")

if __name__ == "__main__":
    main()