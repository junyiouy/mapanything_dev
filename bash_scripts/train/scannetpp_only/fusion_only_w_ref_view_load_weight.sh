#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

NUM_GPUS=$1


export OMP_NUM_THREADS=24


torchrun --nproc_per_node ${NUM_GPUS} --master_port 29506 \
    scripts/train.py \
    machine=dgx02 \
    dataset=mine_scannetpp_only.yaml dataset.num_workers=12 \
    dataset.num_views=24 \
    dataset.train.variable_num_views=false \
    loss=overall_pairwise_loss_weigh_pm_higher_chunk_poseNptc_only_inter_chunk \
    model=mapanything_chunked_aat_w_ref_view \
    model/task=images_only \
    model.encoder.gradient_checkpointing=false \
    model.pred_head.gradient_checkpointing=false \
    model.info_sharing.module_args.gradient_checkpointing=false \
    model.pretrained='/wekafs/ict/junyiouy/map-anything/checkpoints/facebook_map-anything.pth' \
    train_params=only_fusion \
    train_params.epochs=20 \
    train_params.resume=true \
    train_params.warmup_epochs=1 \
    train_params.keep_freq=20 \
    train_params.eval_freq=2 \
    train_params.max_num_of_imgs_per_gpu=48 \
    train_params.accum_iter=16 \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/fusion_only_tune_w_ref_view_load_weight_scannetpp'
