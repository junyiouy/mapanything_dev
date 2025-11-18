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


def process_waymo_scene(cfg, scene_name):
    """
    Process a WayMo scene into the WAI format.

    Expected structure:
    - Images: <scene_name>/<frame_id>_<cam_id>.jpg
    - Depth: <scene_name>/<frame_id>_<cam_id>.exr
    - Camera params: <scene_name>/<frame_id>_<cam_id>.npz (intrinsics, cam2world, distortion)
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
    wai_frames = []

    # Get all .jpg files and sort
    jpg_files = sorted(scene_root.glob("*.jpg"))
    if not jpg_files:
        print(f"No .jpg files found in {scene_name}, skipping...")
        return

    for jpg_file in jpg_files:
        base_name = jpg_file.stem  # e.g., 00000_1
        frame_id, cam_id = base_name.split('_')
        cam_id = int(cam_id)

        # Corresponding files
        exr_file = scene_root / f"{base_name}.exr"
        npz_file = scene_root / f"{base_name}.npz"

        if not exr_file.exists() or not npz_file.exists():
            print(f"Warning: Missing exr or npz for {base_name}, skipping...")
            continue

        # Load camera parameters from npz
        data = np.load(npz_file)
        intrinsics = data['intrinsics']  # 3x3 matrix
        cam2world = data['cam2world']  # 4x4 matrix
        distortion = data['distortion']  # distortion params

        # Extract intrinsics
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        h, w = intrinsics.shape[0], intrinsics.shape[1]  # Assuming square, but actually from image

        # Get image size from actual image (assuming PIL or similar)
        from PIL import Image
        with Image.open(jpg_file) as img:
            w, h = img.size

        # Create symlinks
        rel_image_path = Path("images") / f"{base_name}.jpg"
        rel_depth_path = Path("depth") / f"{base_name}.exr"

        os.symlink(jpg_file, target_scene_root / rel_image_path)
        os.symlink(exr_file, target_scene_root / rel_depth_path)

        # Store WAI frame metadata
        wai_frame = {
            "frame_name": base_name,
            "image": str(rel_image_path),
            "file_path": str(rel_image_path),
            "depth": str(rel_depth_path),
            "transform_matrix": cam2world.tolist(),
            "h": h,
            "w": w,
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "distortion": distortion.tolist(),
        }
        wai_frames.append(wai_frame)

    # Construct overall scene metadata
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",  
        "camera_convention": "opencv",
        "scale_type": "metric",
        "scene_modalities": {},
        "frames": wai_frames,
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "depth": {
                "frame_key": "depth",
                "format": "depth",
            },
        },
    }

    # Save scene metadata
    with open(target_scene_root / "scene_meta.json", "w") as f:
        json.dump(scene_meta, f, indent=2)


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/waymo.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(
        process_waymo_scene,
        cfg,
    )
