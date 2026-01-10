#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

NUM_GPUS=$1


torchrun --nproc_per_node ${NUM_GPUS} --master_port 29593 \
    scripts/train.py \
    machine=dgx02 \
    dataset=mine_scannetpp_only dataset.num_workers=12 \
    dataset.num_views=24 \
    loss=overall_pairwise_loss_weigh_pm_higher_point_lower \
    model=mapanything_prechunk_fusion_16_mean_pool \
    model/task=aug_training \
    model.encoder.gradient_checkpointing=true \
    model.pred_head.gradient_checkpointing=true \
    model.info_sharing.module_args.gradient_checkpointing=true \
    train_params=lower_encoder_lr \
    train_params.epochs=160 \
    train_params.resume=true \
    train_params.warmup_epochs=2 \
    train_params.keep_freq=20 \
    train_params.max_num_of_imgs_per_gpu=24 \
    train_params.accum_iter=4 \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/prechunk_pre_16_layers_topk_2' \
    model.pretrained='/wekafs/ict/junyiouy/map-anything/checkpoints/facebook_map-anything.pth' 

# 这个在上一个的基础上把kv的归一化加上，并且全量训练