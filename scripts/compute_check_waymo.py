#!/usr/bin/env python3
"""
Run MapAnything inference on Waymo sequences using Hydra configuration.
Decoupled metrics version.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

import hydra
from omegaconf import DictConfig, OmegaConf

from mapanything.models import init_model
from mapanything.utils.image import preprocess_inputs

# Waymo 预处理路径
waymo_preprocessing_path = "/wekafs/ict/junyiouy/waymo_preprocessing"
sys.path.insert(0, waymo_preprocessing_path)

try:
    from dataset_reader import WaymoDatasetReader
except ImportError as e:
    print(f"Error: Could not import WaymoDatasetReader: {e}")
    sys.exit(1)

class WaymoToOpenCVConverter:
    def __init__(self):
        self.coord_transform = torch.tensor([
            [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]
        ], dtype=torch.float32)

    def convert_pose_waymo_to_opencv(self, waymo_pose):
        waymo_pose = torch.from_numpy(waymo_pose).float()
        opencv_pose = waymo_pose @ torch.linalg.inv(self.coord_transform)
        return opencv_pose.numpy()

def load_image_for_preprocessing(image_path):
    return Image.open(image_path).convert('RGB')

def prepare_views_for_sequence(reader, camera_types, converter, modalities, metric_scale=True):
    views = []
    all_frames = reader.get_frames()
    frames_by_timestamp = {}
    for frame in all_frames:
        ts = frame['timestamp']
        if ts not in frames_by_timestamp: frames_by_timestamp[ts] = []
        frames_by_timestamp[ts].append(frame)

    for ts, ts_frames in sorted(frames_by_timestamp.items()):
        camera_frames = {f['camera']: f for f in ts_frames if f['camera'] in camera_types}
        for cam_name in camera_types:
            if cam_name not in camera_frames: continue
            frame = camera_frames[cam_name]
            image = load_image_for_preprocessing(reader.get_image_path(frame))
            view = {
                'img': image,
                'data_norm_type': ['dinov2'],
                'is_metric_scale': torch.tensor([metric_scale]),
            }
            if 'intrinsics' in modalities:
                view['intrinsics'] = reader.get_camera_intrinsics(cam_name)['intrinsic_matrix'].astype(np.float32)
            if 'poses' in modalities:
                c2w = reader.get_camera_to_world_pose(frame, cam_name)
                view['camera_poses'] = converter.convert_pose_waymo_to_opencv(c2w).astype(np.float32)
            views.append(view)
    return views

def save_test_results(predictions, metrics, output_dir, sequence_name):
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "sequence_name": sequence_name,
        "num_views": len(predictions),
        "performance_metrics": metrics
    }
    output_path = os.path.join(output_dir, f"test_results_{sequence_name}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

@hydra.main(version_base=None, config_path="../configs", config_name="infer_waymo")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    reader = WaymoDatasetReader(cfg.sequence_dir)
    converter = WaymoToOpenCVConverter()
    sequence_name = os.path.basename(cfg.sequence_dir.rstrip('/'))

    views = prepare_views_for_sequence(reader, cfg.camera_types, converter, cfg.modalities, not cfg.no_metric_scale)
    # if len(views) % 2 != 0: views = views[:len(views)-1]
    views = views[:192]
    # duplicate 4 times for testing
    views = views * 4
    if not views: return

    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    if cfg.checkpoint:
        ckpt = torch.load(cfg.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)

    model.to(device)
    model.eval()

    norm_type = getattr(model.encoder, 'data_norm_type', 'dinov2')
    for v in views: v['data_norm_type'] = [norm_type]

    # 执行推理
    processed_views = preprocess_inputs(views)
    predictions = model.infer(
        processed_views,
        memory_efficient_inference=not cfg.no_memory_efficient,
        use_amp=True,
        amp_dtype="bf16",
        record_metrics=cfg.record_metrics
    )

    # 提取解耦后的性能指标
    perf_metrics = getattr(model, 'last_inference_metrics', {})
    save_test_results(predictions, perf_metrics, cfg.output_dir, sequence_name)

if __name__ == "__main__":
    main()

# # 运行命令
# CUDA_VISIBLE_DEVICES=0 python scripts/compute_check_waymo.py