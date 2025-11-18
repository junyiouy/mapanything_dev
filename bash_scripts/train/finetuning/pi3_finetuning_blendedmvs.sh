#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

NUM_GPUS=$1

# Logging Configs
# export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=INFO

# Single-Machine Multi-GPU Configs
export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=dgx02 \
    dataset=blendedmvs_518_24ipg_8g dataset.num_workers=12 \
    dataset.num_views=4 \
    dataset.principal_point_centered=true \
    loss=pi3_loss \
    model=pi3 \
    train_params=pi3_finetune \
    train_params.epochs=10 \
    train_params.warmup_epochs=1 \
    train_params.keep_freq=20 \
    train_params.max_num_of_imgs_per_gpu=12 \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/pi3_finetuning_blendedmvs'
