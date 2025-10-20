# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Waymo Dataset using WAI format data.
"""

import os

import numpy as np

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.utils.wai.core import load_data, load_frame


class WaymoWAI(BaseDataset):
    """
    Waymo dataset containing real-world driving scenes.
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
            split: Dataset split (train, val, test).
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
        self.is_metric_scale = True  # Waymo is real-world metric scale
        self.is_synthetic = False

    def _load_data(self):
        "Load the precomputed dataset metadata"
        # Load the dataset metadata corresponding to the split
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"waymo_scene_list_{self.split}.npy",
        )
        split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        # Get the list of all scenes
        if not self.sample_specific_scene:
            self.scenes = list(split_scene_list)
        else:
            self.scenes = [self.specific_scene_name]
        self.num_of_scenes = len(self.scenes)
    
    @staticmethod
    def build_manual_covisibility_matrix(num_views_in_scene):
        """
        手动构建共可视矩阵
        Args:
            num_views_in_scene: 场景中的总view数 (5t)
        Returns:
            5t×5t的共可视矩阵
        """
        num_timestamps = num_views_in_scene // 5
        pairwise_covisibility = np.zeros((num_views_in_scene, num_views_in_scene), dtype=np.float32)
        
        # 5×5相机内共可视矩阵（同一时间戳）
        intra_covisibility = np.array([
            [1.0, 1.0, 1.0, 0.0, 0.0],  # F
            [1.0, 1.0, 0.0, 1.0, 0.0],  # FL
            [1.0, 0.0, 1.0, 0.0, 1.0],  # FR
            [0.0, 1.0, 0.0, 1.0, 0.0],  # SL
            [0.0, 0.0, 1.0, 0.0, 1.0],  # SR
        ], dtype=np.float32)
        
        for i in range(num_views_in_scene):
            timestamp_i = i // 5
            camera_i = i % 5
            
            for j in range(num_views_in_scene):
                timestamp_j = j // 5
                camera_j = j % 5
                
                time_diff = abs(timestamp_i - timestamp_j)
                
                if time_diff == 0:
                    # 同一时间戳内的连接
                    pairwise_covisibility[i, j] = intra_covisibility[camera_i, camera_j]
                
                elif time_diff <= 5:
                    # 5步以内：不相邻连接（间隔连接）
                    pass
                        
                elif time_diff <= 25:
                    # 5-25步：仅front相机相邻连接
                    if camera_i == 0 and camera_j == 0:
                        pairwise_covisibility[i, j] = 1.0
                        
                # 超过25步或不符合条件的保持0.0
        
        return pairwise_covisibility
    
    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        # Get the scene name of the sampled index
        scene_index = sampled_idx
        scene_name = self.scenes[scene_index]

        # Get the metadata corresponding to the scene
        scene_root = os.path.join(self.ROOT, scene_name)
        scene_meta = load_data(
            os.path.join(scene_root, "scene_meta.json"), "scene_meta"
        )
        scene_file_names = list(scene_meta["frame_names"].keys())
        num_views_in_scene = len(scene_file_names)

        pairwise_covisibility = self.build_manual_covisibility_matrix(num_views_in_scene)
        
        # Get the indices of the N views in the scene
        view_indices = self._sample_view_indices(
            num_views_to_sample, num_views_in_scene, pairwise_covisibility
        )
        # import pdb; pdb.set_trace()

        # Get the views corresponding to the selected view indices
        views = []
        for view_index in view_indices:
            # Load the data corresponding to the view
            view_file_name = scene_file_names[view_index]
            view_data = load_frame(
                scene_root,
                view_file_name,
                modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            # Convert necessary data to numpy
            image = view_data["image"].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            depthmap = view_data["depth"].numpy().astype(np.float32)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

            # Handle incomplete GT depth map - ensure all values are valid
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

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
                    dataset="Waymo",
                    label=scene_name,
                    instance=os.path.join("images", str(view_file_name)),
                )
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="/wekafs/ict/junyiouy/map_anything_data/waymo_train", type=str
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

    dataset = WaymoWAI(
        num_views=args.num_of_views,
        split="train",
        covisibility_thres=0.25,
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        resolution=(518, 392),
        aug_crop=16,
        transform="colorjitter+grayscale+gaublur",
        data_norm_type="dinov2",
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "Waymo_Dataloader")
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
# python3 mapanything/datasets/wai/waymo.py --viz --save Waymo_log.rrd --connect False --num_of_views 24