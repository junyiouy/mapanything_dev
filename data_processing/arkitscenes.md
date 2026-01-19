# ARKitScenes to WAI Format Conversion Plan

## Overview
This document outlines the plan to convert the processed ARKitScenes dataset to the WAI (World AI) format required by the map-anything library.

## Current Data Structure (ARKitScenes Preprocessing Output)

The ARKitScenes preprocessing script produces data in `/wekafs/ict/junyiouy/arkitscenes_processed` with the following structure:

### Directory Structure
```
arkitscenes_processed/
├── Test/
│   ├── scene_name_1/
│   │   ├── scene_metadata.npz    # trajectories, intrinsics, images, pairs
│   │   ├── vga_wide/            # JPG images
│   │   └── lowres_depth/        # PNG depth maps
│   └── scene_list.json
└── Training/
    ├── scene_name_2/
    └── ...
```

### Data Format Details

#### scene_metadata.npz Contents:
- **trajectories**: (N, 4, 4) array of camera-to-world transformation matrices
- **intrinsics**: (N, 6) array with format [width, height, fx, fy, cx, cy]
- **images**: (N,) array of image basenames (e.g., "frame_0001.png")
- **pairs**: (M, 2) array of image pair indices for training

#### Camera Coordinate System:
- **Convention**: OpenCV (right-handed, Z-forward, Y-down, X-right)
- **Pose format**: Camera-to-world (c2w) transformation matrices
- **No conversion needed** - already matches WAI requirements

## WAI Format Requirements

### Directory Structure per Scene
```
wai_arkitscenes/
├── scene_name/
│   ├── scene_meta.json
│   ├── images/          # symlinks to JPG files
│   └── depth/           # symlinks to PNG files
```

### scene_meta.json Structure
```json
{
  "scene_name": "scene_name",
  "dataset_name": "arkitscenes",
  "version": "0.1",
  "shared_intrinsics": false,
  "camera_model": "PINHOLE",
  "camera_convention": "opencv",
  "scale_type": "metric",
  "frames": [
    {
      "frame_name": "frame_0001",
      "image": "images/frame_0001.jpg",
      "depth": "depth/frame_0001.png",
      "transform_matrix": [[...], [...], [...], [...]],
      "h": 480,
      "w": 640,
      "fl_x": 525.0,
      "fl_y": 525.0,
      "cx": 319.5,
      "cy": 239.5
    }
  ],
  "frame_modalities": {
    "image": {"frame_key": "image", "format": "image"},
    "depth": {"frame_key": "depth", "format": "depth"}
  }
}
```

## Conversion Implementation

### 1. Conversion Script (arkitscenes.py)

Location: `data_processing/wai_processing/scripts/conversion/arkitscenes.py`

#### Main Function: process_arkitscenes_scene()
```python
def process_arkitscenes_scene(cfg, scene_name):
    # Set up paths
    scene_root = Path(cfg.original_root) / scene_name
    target_scene_root = Path(cfg.root) / scene_name

    # Load scene metadata
    metadata_path = scene_root / "scene_metadata.npz"
    data = np.load(metadata_path)

    # Extract data
    trajectories = data['trajectories']  # (N, 4, 4) c2w matrices
    intrinsics = data['intrinsics']    # (N, 6) [w, h, fx, fy, cx, cy]
    images = data['images']            # (N,) basenames

    # Create directories
    image_dir = target_scene_root / "images"
    depth_dir = target_scene_root / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Process each frame
    wai_frames = []
    for i, basename in enumerate(images):
        # Create symlinks
        img_src = scene_root / "vga_wide" / basename.replace('.png', '.jpg')
        depth_src = scene_root / "lowres_depth" / basename

        img_link = image_dir / f"{Path(basename).stem}.jpg"
        depth_link = depth_dir / basename

        os.symlink(img_src, img_link)
        os.symlink(depth_src, depth_link)

        # Extract intrinsics
        w, h, fx, fy, cx, cy = intrinsics[i]

        # Build frame metadata
        wai_frame = {
            "frame_name": Path(basename).stem,
            "image": f"images/{Path(basename).stem}.jpg",
            "depth": f"depth/{basename}",
            "transform_matrix": trajectories[i].tolist(),
            "h": int(h),
            "w": int(w),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy)
        }
        wai_frames.append(wai_frame)

    # Build scene metadata
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
            "depth": {"frame_key": "depth", "format": "depth"}
        }
    }

    # Save scene metadata
    with open(target_scene_root / "scene_meta.json", "w") as f:
        json.dump(scene_meta, f, indent=2)
```

### 2. Configuration Files

#### Conversion Config (arkitscenes.yaml)
Location: `data_processing/wai_processing/configs/conversion/arkitscenes.yaml`
```yaml
original_root: /wekafs/ict/junyiouy/arkitscenes_processed
root: # path of wai-formatted dataset

dataset_name: arkitscenes
version: 0.1
overwrite: True

scene_filters:
  - process_state_not: [conversion, finished]
```

#### Launch Config (arkitscenes.yaml)
Location: `data_processing/wai_processing/configs/launch/arkitscenes.yaml`
```yaml
stage: # set stage via CLI
root:  # path of wai-formatted dataset

gpus: 0
cpus: 10
mem: 20
scenes_per_job: 20
conda_env: # pass the name of our conda environment
nodelist:

num_workers: 16

stages:
  conversion:
    script: conversion/arkitscenes.py
    config: conversion/arkitscenes.yaml
    scenes_per_job: 50
```

### 3. Data Mapping Details

#### Intrinsics Mapping
- ARKitScenes: `[w, h, fx, fy, cx, cy]`
- WAI: `fl_x=fx, fl_y=fy, cx=cx, cy=cy, h=h, w=w`

#### Image/Depth File Mapping
- ARKitScenes images: `vga_wide/frame_0001.png` → `images/frame_0001.jpg`
- ARKitScenes depth: `lowres_depth/frame_0001.png` → `depth/frame_0001.png`

#### Pose Matrix
- Direct mapping: `trajectories[i]` → `transform_matrix`
- Already in correct OpenCV c2w format

## Implementation Steps

1. **Create conversion script** with process_arkitscenes_scene() function
2. **Create conversion config** defining paths and parameters
3. **Create launch config** for pipeline execution
4. **Test on sample scene** to verify correctness
5. **Run full conversion** using convert_scenes_wrapper

## Dependencies

- numpy
- json
- pathlib
- os
- Existing WAI utilities: convert_scenes_wrapper, store_data

## Expected Output

After conversion, the WAI-formatted ARKitScenes dataset will be ready for use with map-anything training and inference pipelines, maintaining all camera calibration and geometric information from the original preprocessing.
