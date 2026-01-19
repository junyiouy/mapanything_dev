#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path

import numpy as np
from argconf import argconf_parse
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
    get_original_scene_names,  # noqa: F401, Needed for launch_slurm.py
)


def process_arkitscenes_scene(cfg, scene_name):
    """
    Process an ARKitScenes scene into the WAI format.

    Expected structure:
    - scene_metadata.npz: trajectories, intrinsics, images, pairs
    - vga_wide/: JPG images
    - lowres_depth/: PNG depth maps
    """
    # Set up paths
    scene_root = Path(cfg.original_root) / scene_name
    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    depth_dir = target_scene_root / "depth"

    # Create directories
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {scene_name} data to WAI format ...")

    # Load scene metadata
    metadata_path = scene_root / "scene_metadata.npz"
    if not metadata_path.exists():
        print(f"Warning: scene_metadata.npz not found for {scene_name}, skipping...")
        return

    data = np.load(metadata_path)
    trajectories = data['trajectories']  # (N, 4, 4) c2w matrices
    intrinsics = data['intrinsics']    # (N, 6) [w, h, fx, fy, cx, cy]
    images = data['images']            # (N,) basenames

    wai_frames = []

    for i, basename in enumerate(images):
        # Source files
        img_src = scene_root / "vga_wide" / basename.replace('.png', '.jpg')
        depth_src = scene_root / "lowres_depth" / basename

        if not img_src.exists() or not depth_src.exists():
            print(f"Warning: Missing image or depth for {basename} in {scene_name}, skipping...")
            continue

        # Create symlinks
        frame_name = Path(basename).stem
        img_link = image_dir / f"{frame_name}.jpg"
        depth_link = depth_dir / basename

        try:
            os.symlink(img_src, img_link)
            os.symlink(depth_src, depth_link)
        except FileExistsError:
            # Symlink already exists, skip
            pass

        # Extract intrinsics [w, h, fx, fy, cx, cy]
        w, h, fx, fy, cx, cy = intrinsics[i]

        # Build WAI frame metadata
        wai_frame = {
            "frame_name": frame_name,
            "image": f"images/{frame_name}.jpg",
            "depth": f"depth/{basename}",
            "transform_matrix": trajectories[i].tolist(),
            "h": int(h),
            "w": int(w),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }
        wai_frames.append(wai_frame)

    if not wai_frames:
        print(f"Warning: No valid frames found for {scene_name}, skipping...")
        return

    # Construct overall scene metadata
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
        "frames": wai_frames,
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "depth": {"frame_key": "depth", "format": "depth"},
        },
    }

    # Save scene metadata
    with open(target_scene_root / "scene_meta.json", "w") as f:
        json.dump(scene_meta, f, indent=2)


def get_arkitscenes_scene_names(cfg):
    """Get all ARKitScenes scene names from the processed data."""
    scene_names = []

    # Check both Test and Training splits
    for split in ["Test", "Training"]:
        split_dir = Path(cfg.original_root) / split
        if not split_dir.exists():
            continue

        scene_list_file = split_dir / "scene_list.json"
        if scene_list_file.exists():
            with open(scene_list_file, "r") as f:
                split_scenes = json.load(f)
                # Prefix with split name to create full scene identifiers
                scene_names.extend([f"{split}/{scene}" for scene in split_scenes])

    return scene_names


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/arkitscenes.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(
        process_arkitscenes_scene,
        cfg,
        get_original_scene_names_func=get_arkitscenes_scene_names,
    )
