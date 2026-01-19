# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
SpatialVid Dataset using WAI format data.
"""

import os
from pathlib import Path

import numpy as np

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.utils.wai.core import load_data, load_frame
import cv2

class SpatialVidWAI(BaseDataset):
    """
    SpatialVid dataset containing multi-view spatial video scenes.
    """

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir,
        split,
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: str = None,
        **kwargs,
    ):
        """
        Initialize the dataset attributes.
        Args:
            ROOT: Root directory of the dataset.
            dataset_metadata_dir: Path to the dataset metadata directory.
            split: Dataset split (train, val, test). For SpatialVid, all scenes are used for train.
            overfit_num_sets: If None, use all sets. Else, the dataset will be truncated to this number of sets.
            sample_specific_scene: Whether to sample a specific scene from the dataset.
            specific_scene_name: Name of the specific scene to sample.
        """
        # Initialize the dataset attributes
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self._load_data()

        # Define the dataset type flags
        self.is_metric_scale = True
        self.is_synthetic = False

    def _load_data(self):
        "Load the precomputed dataset metadata"
        cache_path = os.path.join(self.ROOT, "spatialvid_splits.npy")

        # Try to load splits from cache first
        splits = None
        if os.path.exists(cache_path):
            try:
                splits = np.load(cache_path, allow_pickle=True).item()
                print(f"Loaded splits from cache: {cache_path}")
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}, will regenerate...")
                splits = None

        # If cache doesn't exist or failed to load, traverse directory and create splits
        if splits is None:
            print("Generating scene list and splits by traversing directory...")
            all_scenes = []
            root_path = Path(self.ROOT)
            if root_path.exists():
                for group_dir in root_path.iterdir():
                    if group_dir.is_dir() and group_dir.name.startswith("group_"):
                        for scene_dir in group_dir.iterdir():
                            if scene_dir.is_dir():
                                scene_meta_path = scene_dir / "scene_meta.json"
                                if scene_meta_path.exists():
                                    scene_name = f"{group_dir.name}/{scene_dir.name}"
                                    all_scenes.append(scene_name)

            # Shuffle and split: 80% train, 10% val, 10% test
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(all_scenes)
            n_total = len(all_scenes)
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)
            n_test = n_total - n_train - n_val

            splits = {
                'train': all_scenes[:n_train],
                'val': all_scenes[n_train:n_train + n_val],
                'test': all_scenes[n_train + n_val:]
            }

            # Save cache for future use
            try:
                np.save(cache_path, splits)
                print(f"Saved splits cache to: {cache_path}")
                print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
            except Exception as e:
                print(f"Failed to save cache {cache_path}: {e}")

        # Select scenes based on split
        self.scenes = splits.get(self.split, [])
        if not self.scenes:
            print(f"Warning: No scenes found for split '{self.split}'")

        if self.sample_specific_scene:
            self.scenes = [self.specific_scene_name] if self.specific_scene_name in self.scenes else []

        if self.overfit_num_sets is not None:
            original_len = len(self.scenes)
            self.scenes = self.scenes[: self.overfit_num_sets]
            self.scenes = self.scenes * (original_len // self.overfit_num_sets)
            print(f"Overfitting to {self.overfit_num_sets} sets. Total scenes used: {len(self.scenes)}")

        self.num_of_scenes = len(self.scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        # Get the scene name of the sampled index
        scene_index = sampled_idx
        scene_name = self.scenes[scene_index]

        # Get the metadata corresponding to the scene
        scene_root = os.path.join(self.ROOT, scene_name)
        scene_meta = load_data(
            os.path.join(scene_root, "scene_meta.json"), "scene_meta"
        )
        scene_meta['camera_model'] = 'PINHOLE'  # Ensure camera model is set
        frames = scene_meta["frames"]
        num_views_in_scene = len(frames)

        # Sample views with random intervals, similar to ScanNetPP (no covisibility needed)
        max_interval = min(20, num_views_in_scene // num_views_to_sample)
        interval = np.random.randint(1, max_interval + 1)
        start_idx = np.random.randint(0, max(1, num_views_in_scene - interval * num_views_to_sample + 1))
        view_indices = [start_idx + i * interval for i in range(num_views_to_sample)]
        view_indices = [idx % num_views_in_scene for idx in view_indices]

        # Get the views corresponding to the selected view indices
        views = []
        for view_index in view_indices:
            frame_data = frames[view_index]

            # Load the data corresponding to the view
            view_data = load_frame(
                scene_root,
                frame_data["frame_name"],
                modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            # Convert necessary data to numpy
            image = view_data["image"].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            depthmap = view_data["depth"].numpy().astype(np.float32)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

            # for spatialvid, the stored depth is inverse depth.
            depthmap = 1 / depthmap

            # Ensure that the depthmap has all valid values            
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
            # resize depthmap to match image size if necessary, using nearest neighbor
            if depthmap.shape != image.shape[:2]:
                depthmap = cv2.resize(depthmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Resize the data to match the desired resolution
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image=image,
                resolution=resolution,
                depthmap=depthmap,
                intrinsics=intrinsics,
                additional_quantities=None,
            )

            # Append the view dictionary to the list of views
            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=c2w_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="SpatialVid",
                    label=scene_name,
                    instance=os.path.join("images", str(frame_data["frame_name"])),
                )
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="/wekafs/ict/junyiouy/map_anything_data/spatialvid", type=str
    )
    parser.add_argument(
        "-dmd",
        "--dataset_metadata_dir",
        default="/wekafs/ict/junyiouy/map-anything/map_anything_dataset/map-anything/mapanything_dataset_metadata",
        type=str,
    )
    parser.add_argument(
        "-nv",
        "--num_of_views",
        default=2,
        type=int,
    )
    parser.add_argument("--viz", action="store_true")

    return parser


if __name__ == "__main__":
    import rerun as rr
    from tqdm import tqdm

    from mapanything.datasets.base.base_dataset import view_name
    from mapanything.utils.image import rgb
    from mapanything.utils.viz import script_add_rerun_args


    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    dataset = SpatialVidWAI(
        num_views=args.num_of_views,
        split="train",
        covisibility_thres=0.25,
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        resolution=(518, 336),
        aug_crop=16,
        transform="colorjitter+grayscale+gaublur",
        data_norm_type="dinov2",
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "SpatialVid_Dataloader")
        rr.set_time("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    sampled_indices = np.random.choice(len(dataset), size=5, replace=False)

    for num, idx in enumerate(tqdm(sampled_indices)):
        views = dataset[idx]
        assert len(views) == args.num_of_views
        sample_name = f"{idx}"
        for view_idx in range(args.num_of_views):
            sample_name += f" {view_name(views[view_idx])}"
        print(sample_name)
        for view_idx in range(args.num_of_views):
            image = rgb(
                views[view_idx]["img"], norm_type=views[view_idx]["data_norm_type"]
            )
            depthmap = views[view_idx]["depthmap"]
            pose = views[view_idx]["camera_pose"]
            intrinsics = views[view_idx]["camera_intrinsics"]
            pts3d = views[view_idx]["pts3d"]
            valid_mask = views[view_idx]["valid_mask"]
            if "non_ambiguous_mask" in views[view_idx]:
                non_ambiguous_mask = views[view_idx]["non_ambiguous_mask"]
            else:
                non_ambiguous_mask = None
            if "prior_depth_along_ray" in views[view_idx]:
                prior_depth_along_ray = views[view_idx]["prior_depth_along_ray"]
            else:
                prior_depth_along_ray = None
            if args.viz:
                rr.set_time("stable_time", sequence=num)
                base_name = f"world/view_{view_idx}"
                pts_name = f"world/view_{view_idx}_pointcloud"
                # Log camera info and loaded data
                height, width = image.shape[0], image.shape[1]
                rr.log(
                    base_name,
                    rr.Transform3D(
                        translation=pose[:3, 3],
                        mat3x3=pose[:3, :3],
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole",
                    rr.Pinhole(
                        image_from_camera=intrinsics,
                        height=height,
                        width=width,
                        camera_xyz=rr.ViewCoordinates.RDF,
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole/rgb",
                    rr.Image(image),
                )
                rr.log(
                    f"{base_name}/pinhole/depth",
                    rr.DepthImage(depthmap),
                )
                if prior_depth_along_ray is not None:
                    rr.log(
                        f"prior_depth_along_ray_{view_idx}",
                        rr.DepthImage(prior_depth_along_ray),
                    )
                if non_ambiguous_mask is not None:
                    rr.log(
                        f"{base_name}/pinhole/non_ambiguous_mask",
                        rr.SegmentationImage(non_ambiguous_mask.astype(int)),
                    )
                # Log points in 3D
                filtered_pts = pts3d[valid_mask]
                filtered_pts_col = image[valid_mask]
                rr.log(
                    pts_name,
                    rr.Points3D(
                        positions=filtered_pts.reshape(-1, 3),
                        colors=filtered_pts_col.reshape(-1, 3),
                    ),
                )
