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


def process_spatialvid_scene(cfg, scene_name):
    """
    Process a spatialvid scene into the WAI format.

    Expected structure:
    - transforms.json: camera intrinsics and frame metadata
    - images/: JPG images
    - depths/: EXR depth maps
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

    # Load transforms.json
    transforms_path = scene_root / "transforms.json"
    if not transforms_path.exists():
        print(f"Warning: transforms.json not found for {scene_name}, skipping...")
        return

    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)

    camera_model = transforms_data.get("camera_model", "OPENCV")
    fl_x = transforms_data["fl_x"]
    fl_y = transforms_data["fl_y"]
    cx = transforms_data["cx"]
    cy = transforms_data["cy"]
    w = int(transforms_data["w"])
    h = int(transforms_data["h"])

    frames = transforms_data["frames"]

    wai_frames = []

    for frame in frames:
        file_path = frame["file_path"]
        depth_file_path = frame["depth_file_path"]
        transform_matrix = frame["transform_matrix"]

        # Source files
        img_src = scene_root / file_path
        depth_src = scene_root / depth_file_path

        if not img_src.exists() or not depth_src.exists():
            print(f"Warning: Missing image or depth for {file_path} in {scene_name}, skipping...")
            continue

        # Create symlinks
        frame_name = Path(file_path).stem
        img_link = image_dir / f"{frame_name}.jpg"
        depth_link = depth_dir / f"{frame_name}.exr"

        try:
            os.symlink(img_src, img_link)
            os.symlink(depth_src, depth_link)
        except FileExistsError:
            # Symlink already exists, skip
            pass

        # Build WAI frame metadata
        wai_frame = {
            "frame_name": frame_name,
            "image": f"images/{frame_name}.jpg",
            "depth": f"depth/{frame_name}.exr",
            "transform_matrix": transform_matrix,
            "h": h,
            "w": w,
            "fl_x": float(fl_x),
            "fl_y": float(fl_y),
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
        "shared_intrinsics": True,
        "camera_model": camera_model,
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


def get_spatialvid_scene_names(cfg):
    """Get all spatialvid scene names from the processed data."""
    scene_names = []

    # Traverse the directory structure: original_root/group_XXXX/scene_id/
    original_root = Path(cfg.original_root)
    if not original_root.exists():
        print(f"Warning: original_root {original_root} does not exist")
        return scene_names

    for group_dir in original_root.iterdir():
        if group_dir.is_dir() and group_dir.name.startswith("group_"):
            for scene_dir in group_dir.iterdir():
                if scene_dir.is_dir():
                    # Check if transforms.json exists
                    transforms_path = scene_dir / "transforms.json"
                    if transforms_path.exists():
                        # Scene name is group_XXXX/scene_id
                        scene_name = f"{group_dir.name}/{scene_dir.name}"
                        scene_names.append(scene_name)

    return scene_names


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/spatialvid.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(
        process_spatialvid_scene,
        cfg,
        get_original_scene_names_func=get_spatialvid_scene_names,
    )
