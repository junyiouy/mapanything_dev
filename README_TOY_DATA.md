data_root: /wekafs/ict/junyiouy/map_anything_data
code_root: /wekafs/ict/junyiouy/map-anything

# Extraction
* Extract ``waymo_train_toy.tar.gz`` and place it under the data_root directory. It should look like this:
/wekafs/ict/junyiouy/map_anything_data/waymo_train_toy
└── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
    ├── _process_log.json
    ├── _process_log_backup.json
    ├── _scene.lock
    ├── _scene_meta_backup.json
    ├── covisibility
    │   └── v0
    │       └── pairwise_covisibility--990x990.npy
    ├── depth
    ....

* Extract `waymo_train_toy_scene_list.tar.gz` and put it under the code_root directory. It should look like this:
``/wekafs/ict/junyiouy/map-anything/map_anything_dataset/map-anything/mapanything_dataset_metadata/train/waymo_scene_list_train_toy.npy``

# Modify config files
* Modify ``/wekafs/ict/junyiouy/map-anything/configs/machine/dgx02.yaml`` to set to your paths.
* Modify **dataset_metadata_dir** in ``/wekafs/ict/junyiouy/map-anything/configs/dataset/waymo_toy_wai/train/default.yaml`` and ``/wekafs/ict/junyiouy/map-anything/configs/dataset/waymo_toy_wai/val/default.yaml``.
