#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

NUM_GPUS=$1


export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node ${NUM_GPUS} --master_port 29501 \
    scripts/train.py \
    machine=dgx02 \
    dataset=mine_1d_518_many_ar_48ipg_64g_overfit dataset.num_workers=12 \
    dataset.num_views=24 \
    dataset.train.variable_num_views=false \
    loss=overall_pairwise_loss_weigh_pm_higher_chunk_pose_only \
    model=mapanything_chunked \
    model/task=aug_training \
    model.encoder.gradient_checkpointing=true \
    model.pred_head.gradient_checkpointing=true \
    model.info_sharing.module_args.gradient_checkpointing=true \
    model.pretrained='${root_experiments_dir}/mapanything/training/mapa_mine_4v_2d_12ipg_8g/checkpoint-last.pth' \
    train_params=finetune_with_only_fusion_modules \
    train_params.epochs=10 \
    train_params.resume=true \
    train_params.warmup_epochs=0 \
    train_params.keep_freq=20 \
    train_params.max_num_of_imgs_per_gpu=48 \
    train_params.accum_iter=4 \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/mapa_mine_24v_1d_12ipg_8g_rel_chunk_overfit'

# model.pretrained_checkpoint_path='/wekafs/ict/junyiouy/map-anything/checkpoints/facebook_map-anything.pth' \
