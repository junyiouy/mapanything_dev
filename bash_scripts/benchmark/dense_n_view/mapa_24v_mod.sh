#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "10 2 benchmark_518_snpp"
    "10 4 benchmark_518_snpp"
    "10 8 benchmark_518_snpp"
    "5 16 benchmark_518_snpp"
    "4 24 benchmark_518_snpp"
    "2 32 benchmark_518_snpp"
    "1 50 benchmark_518_snpp"
    "1 100 benchmark_518_snpp"
)

# Loop through each combination
for combo in "${batch_sizes_and_views[@]}"; do
    # Split the string into batch_size and num_views
    read -r batch_size num_views dataset <<< "$combo"

    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=dgx02 \
        dataset=$dataset \
        dataset.num_workers=12 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=mapanything_prechunk_fusion_2_mean_pool_s2_e24 \
        model/task=images_only \
        model.encoder.uses_torch_hub=false \
        model.pretrained='/wekafs/ict/junyiouy/map-anything/experiments/mapanything/training/prechunk_pre_2_layers_topk_half_mod/checkpoint-best.pth' \
        hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/dense_'"${num_views}"'_view/mapa_mod'

    echo "Finished running $dataset with batch_size=$batch_size and num_views=$num_views"
done
