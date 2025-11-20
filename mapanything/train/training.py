# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Training Code for MapAnything.

References:
DUSt3R: https://github.com/naver/dust3r
"""

import datetime
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sized

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import mapanything.utils.train_tools as train_tools
from mapanything.datasets import get_test_data_loader, get_train_data_loader
from mapanything.models import init_model
from mapanything.train.losses import *  # noqa
from mapanything.utils.inference import loss_of_one_batch_multi_view
from mapanything.utils.train_tools import NativeScalerWithGradNormCount as NativeScaler

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


def train(args):
    """
    Main training function that handles the entire training process.

    This function initializes the distributed training environment, sets up datasets,
    initializes the model, optimizer, and loss functions, and manages the training
    and evaluation loop across multiple epochs.

    In this training, an epoch is just a chunk of the entire dataset.

    Args:
        args: Configuration object containing all training parameters including
              dataset configs, model configs, training parameters, and loss functions.
    """
    # Initialize distributed training if required
    train_tools.init_distributed_mode(args.distributed)
    global_rank = train_tools.get_rank()
    world_size = train_tools.get_world_size()  # noqa

    # Init output directory and device
    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Fix the seed
    seed = args.train_params.seed + train_tools.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.train_params.disable_cudnn_benchmark

    # Datasets and Dataloaders
    print("Building train dataset {:s}".format(args.dataset.train_dataset))
    data_loader_train = build_dataset(
        dataset=args.dataset.train_dataset,
        num_workers=args.dataset.num_workers,
        test=False,
        max_num_of_imgs_per_gpu=args.train_params.max_num_of_imgs_per_gpu,
    )
    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    test_batch_size = 2 * (
        args.train_params.max_num_of_imgs_per_gpu // args.dataset.num_views
    )  # Since we don't have any backward overhead
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset=dataset,
            num_workers=args.dataset.num_workers,
            test=True,
            batch_size=test_batch_size,
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # Load Model
    if global_rank == 0:
        model = init_model(
            args.model.model_str,
            args.model.model_config,
            torch_hub_force_reload=args.model.torch_hub_force_reload,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # Make sure the model is initialized before proceeding
    if global_rank != 0:
        model = init_model(
            args.model.model_str, args.model.model_config, torch_hub_force_reload=False
        )
    model.to(device)  # Move model to device
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # Criterion
    print(f">> Creating train criterion = {args.loss.train_criterion}")
    train_criterion = eval(args.loss.train_criterion).to(device)
    print(
        f">> Creating test criterion = {args.loss.test_criterion or args.loss.train_criterion}"
    )
    test_criterion = eval(args.loss.test_criterion or args.loss.train_criterion).to(
        device
    )

    # Load pretrained model if provided
    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(
            args.model.pretrained, map_location=device, weights_only=False
        )
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt  # in case it occupies memory

    # Init model for DDP training
    if args.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.distributed.gpu],
            find_unused_parameters=True,
            static_graph=False,
        )
        model_without_ddp = model.module

    # Optimizer and loss scaler for gradient accumulation
    # Following timm: set wd as 0 for bias and norm layers
    param_groups, param_groups_name_to_idx_map, param_groups_idx_to_name_map = (
        train_tools.get_parameter_groups(
            model_without_ddp,
            args.train_params.lr,
            args.train_params.weight_decay,
            submodule_configs=args.train_params.submodule_configs,
            warn_not_in_submodule=args.train_params.warn_not_in_submodule,
        )
    )
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.train_params.lr, betas=(0.9, 0.95)
    )
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        """
        Writes training and testing statistics to log files and TensorBoard.

        Args:
            epoch: Current epoch number.
            train_stats: Dictionary containing training metrics.
            test_stats: Dictionary containing testing metrics for each test dataset.
        """
        if train_tools.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        """
        Saves model checkpoint to disk.

        Args:
            epoch: Current epoch number.
            fname: Filename or identifier for the checkpoint.
            best_so_far: Best validation metric achieved so far.
        """
        train_tools.save_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    # Resume from a checkpoint if needed
    last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
    if args.train_params.resume and os.path.isfile(last_ckpt_fname):
        args.train_params.resume_ckpt = last_ckpt_fname
    else:
        args.train_params.resume_ckpt = None
    best_so_far = train_tools.load_model(
        train_args=args.train_params,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    if best_so_far is None:
        best_so_far = float("inf")

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.train_params.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.train_params.start_epoch, args.train_params.epochs + 1):
        # Save immediately the last checkpoint
        if epoch > args.train_params.start_epoch:
            if (
                args.train_params.save_freq
                and epoch % args.train_params.save_freq == 0
                or epoch == args.train_params.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        test_stats = {}
        if (
            args.train_params.eval_freq > 0
            and epoch % args.train_params.eval_freq == 0
            and epoch > 0
        ):
            for test_name, testset in data_loader_test.items():
                print(f"Testing on {test_name} ...")
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

            # Calculate average test loss median
            avg_test_loss_med = np.mean(
                [stats["loss_med"] for stats in test_stats.values()]
            )
            test_stats["Average Test Loss Median"] = avg_test_loss_med
            # Save best
            if avg_test_loss_med < best_so_far:
                best_so_far = avg_test_loss_med
                new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.train_params.start_epoch:
            if args.train_params.keep_freq and epoch % args.train_params.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.train_params.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            param_groups_name_to_idx_map=param_groups_name_to_idx_map,
            param_groups_idx_to_name_map=param_groups_idx_to_name_map,
            model_without_ddp=model_without_ddp,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    save_final_model(
        args, args.train_params.epochs, model_without_ddp, best_so_far=best_so_far
    )


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    """
    Saves the final model checkpoint after training completion.

    Args:
        args: Configuration object containing output directory information.
        epoch: Current epoch number.
        model_without_ddp: Model state dictionary or model instance without DistributedDataParallel wrapper.
        best_so_far: Optional; Best validation metric achieved during training.
    """
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": model_without_ddp
        if isinstance(model_without_ddp, dict)
        else model_without_ddp.cpu().state_dict(),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    train_tools.save_on_master(to_save, checkpoint_path)


def build_dataset(
    dataset, num_workers, test, batch_size=None, max_num_of_imgs_per_gpu=None
):
    """
    Builds data loaders for training or testing.

    Args:
        dataset: Dataset specification string.
        num_workers: Number of worker processes for data loading.
        test: Boolean flag indicating whether this is a test dataset.
        batch_size: Number of samples per batch. Defaults to None. Used only for testing.
        max_num_of_imgs_per_gpu: Maximum number of images per GPU. Defaults to None. Used only for training.

    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset.
    """
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: ", dataset)
    if test:
        assert batch_size is not None, (
            "batch_size must be specified for testing dataloader"
        )
        loader = get_test_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=False,
            drop_last=False,
        )
    else:
        assert max_num_of_imgs_per_gpu is not None, (
            "max_num_of_imgs_per_gpu must be specified for training dataloader"
        )
        loader = get_train_data_loader(
            dataset=dataset,
            max_num_of_imgs_per_gpu=max_num_of_imgs_per_gpu,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=True,
            drop_last=True,
        )

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    param_groups_name_to_idx_map=None,
    param_groups_idx_to_name_map=None,
    model_without_ddp=None,
):
    """
    Trains the model for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function handles the training loop for a single epoch, including forward/backward passes,
    gradient accumulation, learning rate scheduling, and logging metrics.

    Args:
        model: The neural network model to train.
        criterion: Loss function to optimize.
        data_loader: DataLoader providing the training data.
        optimizer: Optimizer for updating model parameters.
        device: Device to run training on (CPU or GPU).
        epoch: Current epoch number.
        loss_scaler: Scaler for gradient accumulation and mixed precision training.
        args: Configuration object containing training parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        param_groups_name_to_idx_map: Mapping from parameter group names to indices.
        param_groups_idx_to_name_map: Mapping from parameter group indices to names.
        model_without_ddp: Model without DistributedDataParallel wrapper for debugging.

    Returns:
        dict: Dictionary containing training metrics averaged over the epoch.
    """
    model.train(True)
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    for submodule_name in param_groups_name_to_idx_map:
        lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
        metric_logger.add_meter(
            lr_name, train_tools.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.train_params.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        epoch_f = epoch + data_iter_step / len(data_loader)

        # We use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            train_tools.adjust_learning_rate(
                optimizer,
                epoch_f,
                args.train_params,
                param_groups_idx_to_name_map,
                args.train_params.submodule_configs,
            )

        loss_tuple = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            use_amp=bool(args.train_params.amp),
            amp_dtype=args.train_params.amp_dtype,
            ret=None,  # Get full result including predictions
        )
        loss_result = loss_tuple["loss"]  # This is (loss, loss_details) tuple from criterion
        if isinstance(loss_result, tuple):
            loss, loss_details = loss_result
        else:
            loss = loss_result
            loss_details = {}
        preds = loss_tuple  # Full result dict containing predictions
        if n_views > 2:
            loss = loss * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        loss_value = float(loss)

        # Debug: Save predictions and GT point clouds as PLY files periodically
        if train_tools.is_main_process() and args.output_dir and data_iter_step % 1000 == 0:  # Save every 1000sh iterations
            debug_dir = os.path.join(args.output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)

            def save_pointcloud_as_ply(points, colors, filename):
                """Save point cloud as PLY file with colors."""
                if points is None or points.numel() == 0:
                    return

                # Flatten to (N, 3) if needed
                if points.dim() == 4:  # (B, H, W, 3)
                    B, H, W, C = points.shape
                    points = points.view(B * H * W, C)
                    if colors is not None and colors.dim() == 4:
                        colors = colors.view(B * H * W, -1)

                # Convert to numpy
                points_np = points.detach().cpu().numpy()
                if colors is not None:
                    colors_np = colors.detach().cpu().numpy()
                    if colors_np.shape[-1] == 1:  # grayscale
                        colors_np = np.repeat(colors_np, 3, axis=-1)
                    elif colors_np.shape[-1] == 3:  # already RGB
                        pass
                    else:
                        colors_np = np.ones_like(points_np) * 128  # default gray
                else:
                    colors_np = np.ones_like(points_np) * 128  # default gray

                # Ensure colors are in 0-255 range
                colors_np = np.clip(colors_np, 0, 255).astype(np.uint8)

                # Write PLY file
                with open(filename, 'w') as f:
                    # Header
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(points_np)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                    f.write("end_header\n")

                    # Data
                    for point, color in zip(points_np, colors_np):
                        f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")

            # Check if chunk_idx is available
            has_chunk_idx = any(preds.get(f"pred{view_idx + 1}", {}).get("chunk_idx") is not None for view_idx in range(n_views))

            if has_chunk_idx:
                # Group by chunk_idx
                chunk_pred_points = {}
                chunk_pred_colors = {}
                chunk_gt_points = {}
                chunk_gt_colors = {}

                for view_idx in range(n_views):
                    pred_key = f"pred{view_idx + 1}"
                    gt_key = f"view{view_idx + 1}"

                    if pred_key in preds and gt_key in preds:
                        chunk_idx = preds[pred_key].get("chunk_idx")
                        if chunk_idx is None:
                            continue

                        # Initialize chunk lists if not exists
                        if chunk_idx not in chunk_pred_points:
                            chunk_pred_points[chunk_idx] = []
                            chunk_pred_colors[chunk_idx] = []
                            chunk_gt_points[chunk_idx] = []
                            chunk_gt_colors[chunk_idx] = []

                        # Get predicted point clouds
                        pred_pts3d = preds[pred_key].get("pts3d")

                        # Get GT point clouds
                        gt_pts3d = preds[gt_key].get("pts3d")
                        gt_valid_mask = preds[gt_key].get("valid_mask")

                        # Get RGB colors from input images using the project's rgb function
                        from mapanything.utils.image import rgb
                        img_tensor = batch[view_idx]["img"]  # (B, 3, H, W) normalized
                        data_norm_type = batch[view_idx]["data_norm_type"][0]
                        # Use rgb function to properly denormalize (returns 0-1 range)
                        img_rgb_01 = rgb(img_tensor, data_norm_type)  # (B, H, W, 3) in 0-1 range
                        # Convert to 0-255 range uint8 for PLY
                        img_rgb = torch.from_numpy((img_rgb_01 * 255).astype(np.uint8)).to(device)

                        # Collect predicted point clouds
                        if pred_pts3d is not None:
                            # Flatten colors to match points
                            if pred_pts3d.dim() == 4:  # (B, H, W, 3)
                                B, H, W, _ = pred_pts3d.shape
                                pred_colors = img_rgb.reshape(B * H * W, 3)
                                pred_pts3d_flat = pred_pts3d.reshape(B * H * W, 3)
                            else:
                                pred_colors = img_rgb.reshape(-1, 3)
                                pred_pts3d_flat = pred_pts3d.reshape(-1, 3)

                            chunk_pred_points[chunk_idx].append(pred_pts3d_flat)
                            chunk_pred_colors[chunk_idx].append(pred_colors)

                        # Collect GT point clouds
                        if gt_pts3d is not None and gt_valid_mask is not None:
                            # Apply valid mask to points and colors
                            if gt_valid_mask.dim() == 4:  # (B, H, W, 1)
                                gt_valid_mask = gt_valid_mask.squeeze(-1)

                            gt_pts3d_masked = gt_pts3d[gt_valid_mask]

                            # Apply mask to colors
                            if gt_valid_mask.dim() == 3:  # (B, H, W)
                                B, H, W = gt_valid_mask.shape
                                gt_colors_masked = img_rgb.reshape(B * H * W, 3)[gt_valid_mask.reshape(-1)]
                            else:
                                gt_colors_masked = img_rgb.reshape(-1, 3)[gt_valid_mask.reshape(-1)]

                            chunk_gt_points[chunk_idx].append(gt_pts3d_masked)
                            chunk_gt_colors[chunk_idx].append(gt_colors_masked)

                # Save point clouds for each chunk
                for chunk_idx in chunk_pred_points.keys():
                    # Save predictions for this chunk
                    if chunk_pred_points[chunk_idx]:
                        combined_pred_points = torch.cat(chunk_pred_points[chunk_idx], dim=0)
                        combined_pred_colors = torch.cat(chunk_pred_colors[chunk_idx], dim=0)
                        pred_filename = f"chunk_{chunk_idx}_pred_epoch_{epoch}_iter_{data_iter_step}.ply"
                        pred_path = os.path.join(debug_dir, pred_filename)
                        save_pointcloud_as_ply(combined_pred_points, combined_pred_colors, pred_path)
                        print(f"Saved chunk {chunk_idx} predicted point clouds to {pred_path}")

                    # Save GT for this chunk
                    if chunk_gt_points[chunk_idx]:
                        combined_gt_points = torch.cat(chunk_gt_points[chunk_idx], dim=0)
                        combined_gt_colors = torch.cat(chunk_gt_colors[chunk_idx], dim=0)
                        gt_filename = f"chunk_{chunk_idx}_gt_epoch_{epoch}_iter_{data_iter_step}.ply"
                        gt_path = os.path.join(debug_dir, gt_filename)
                        save_pointcloud_as_ply(combined_gt_points, combined_gt_colors, gt_path)
                        print(f"Saved chunk {chunk_idx} GT point clouds to {gt_path}")
            else:
                # Fallback: save all views together if no chunk_idx
                all_pred_points = []
                all_pred_colors = []
                all_gt_points = []
                all_gt_colors = []

                for view_idx in range(n_views):
                    pred_key = f"pred{view_idx + 1}"
                    gt_key = f"view{view_idx + 1}"

                    if pred_key in preds and gt_key in preds:
                        # Get predicted point clouds
                        pred_pts3d = preds[pred_key].get("pts3d")

                        # Get GT point clouds
                        gt_pts3d = preds[gt_key].get("pts3d")
                        gt_valid_mask = preds[gt_key].get("valid_mask")

                        # Get RGB colors from input images using the project's rgb function
                        from mapanything.utils.image import rgb
                        img_tensor = batch[view_idx]["img"]  # (B, 3, H, W) normalized
                        data_norm_type = batch[view_idx]["data_norm_type"][0]
                        # Use rgb function to properly denormalize (returns 0-1 range)
                        img_rgb_01 = rgb(img_tensor, data_norm_type)  # (B, H, W, 3) in 0-1 range
                        # Convert to 0-255 range uint8 for PLY
                        img_rgb = torch.from_numpy((img_rgb_01 * 255).astype(np.uint8)).to(device)

                        # Collect predicted point clouds
                        if pred_pts3d is not None:
                            # Flatten colors to match points
                            if pred_pts3d.dim() == 4:  # (B, H, W, 3)
                                B, H, W, _ = pred_pts3d.shape
                                pred_colors = img_rgb.view(B * H * W, 3)
                                pred_pts3d_flat = pred_pts3d.view(B * H * W, 3)
                            else:
                                pred_colors = img_rgb.view(-1, 3)
                                pred_pts3d_flat = pred_pts3d.view(-1, 3)

                            all_pred_points.append(pred_pts3d_flat)
                            all_pred_colors.append(pred_colors)

                        # Collect GT point clouds
                        if gt_pts3d is not None and gt_valid_mask is not None:
                            # Apply valid mask to points and colors
                            if gt_valid_mask.dim() == 4:  # (B, H, W, 1)
                                gt_valid_mask = gt_valid_mask.squeeze(-1)

                            gt_pts3d_masked = gt_pts3d[gt_valid_mask]

                            # Apply mask to colors
                            if gt_valid_mask.dim() == 3:  # (B, H, W)
                                B, H, W = gt_valid_mask.shape
                                gt_colors_masked = img_rgb.view(B * H * W, 3)[gt_valid_mask.view(-1)]
                            else:
                                gt_colors_masked = img_rgb.view(-1, 3)[gt_valid_mask.view(-1)]

                            all_gt_points.append(gt_pts3d_masked)
                            all_gt_colors.append(gt_colors_masked)

                # Save all predictions in one PLY file
                if all_pred_points:
                    combined_pred_points = torch.cat(all_pred_points, dim=0)
                    combined_pred_colors = torch.cat(all_pred_colors, dim=0)
                    pred_filename = f"all_pred_epoch_{epoch}_iter_{data_iter_step}.ply"
                    pred_path = os.path.join(debug_dir, pred_filename)
                    save_pointcloud_as_ply(combined_pred_points, combined_pred_colors, pred_path)
                    print(f"Saved all predicted point clouds to {pred_path}")

                # Save all GT in one PLY file
                if all_gt_points:
                    combined_gt_points = torch.cat(all_gt_points, dim=0)
                    combined_gt_colors = torch.cat(all_gt_colors, dim=0)
                    gt_filename = f"all_gt_epoch_{epoch}_iter_{data_iter_step}.ply"
                    gt_path = os.path.join(debug_dir, gt_filename)
                    save_pointcloud_as_ply(combined_gt_points, combined_gt_colors, gt_path)
                    print(f"Saved all GT point clouds to {gt_path}")

            # Save metadata as text file
            metadata_filename = f"metadata_epoch_{epoch}_iter_{data_iter_step}.txt"
            metadata_path = os.path.join(debug_dir, metadata_filename)
            with open(metadata_path, 'w') as f:
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Iteration: {data_iter_step}\n")
                f.write(f"Loss: {loss_value}\n")
                f.write(f"Loss Details: {loss_details}\n")
                f.write(f"Number of views: {n_views}\n")
            print(f"Saved metadata to {metadata_path}")

        if not math.isfinite(loss_value) or (loss_value > 1000):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            print(f"Loss Details: {loss_details}", force=True)
            print(f"Epoch: {epoch}, Data Iteration: {data_iter_step}", force=True)
            # Save the current batch to the output folder for further inspection
            for view_idx, view in enumerate(batch):
                view_cpu = {}
                for k, v in view.items():
                    view_cpu[k] = v.cpu() if isinstance(v, torch.Tensor) else v
                with open(
                    os.path.join(args.output_dir, f"batch_view_{view_idx}.pkl"), "wb"
                ) as f:
                    pickle.dump(view_cpu, f)
            # Save the model to the output folder for further inspection
            checkpoint_debug_path = os.path.join(
                args.output_dir, "checkpoint-debug.pth"
            )
            to_save_debug = {
                "args": args,
                "model": (
                    model_without_ddp
                    if isinstance(model_without_ddp, dict)
                    else model_without_ddp.cpu().state_dict()
                ),
                "epoch": epoch,
                "data_iter_step": data_iter_step,
            }
            torch.save(to_save_debug, checkpoint_debug_path)
            print(f"Saved debugging material to {args.output_dir}", force=True)
            sys.exit(1)

        # Scale the loss by the number of gradient accumulation iterations
        loss /= accum_iter

        # Compute the scaled gradients (also clip the gradients to max norm of 1)
        gradient_norm = loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=1.0,
        )

        # Zero out the gradients to prepare for the next iteration of gradient descent
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        metric_logger.update(epoch=epoch_f)
        for submodule_name in param_groups_name_to_idx_map:
            lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
            log_lr = optimizer.param_groups[
                param_groups_name_to_idx_map[submodule_name][0]
            ]["lr"]
            metric_logger.meters[lr_name].update(log_lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and (
            (data_iter_step + 1) % (accum_iter * args.train_params.print_freq)
        ) == 0:
            loss_value_reduce = train_tools.all_reduce_mean(
                loss_value
            )  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            if gradient_norm is not None:
                log_writer.add_scalar("train_grad_norm", gradient_norm, epoch_1000x)
            for submodule_name in param_groups_name_to_idx_map:
                lr_name = (
                    f"train_lr_{submodule_name}"
                    if submodule_name != "default"
                    else "train_lr"
                )
                log_lr = optimizer.param_groups[
                    param_groups_name_to_idx_map[submodule_name][0]
                ]["lr"]
                log_writer.add_scalar(lr_name, log_lr, epoch_1000x)
            log_writer.add_scalar("train_iter", epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar("train_" + name, val, epoch_1000x)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):
    """
    Evaluates the model on a test dataset for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function runs evaluation on the test dataset without computing gradients,
    and collects metrics for model performance assessment.

    Args:
        model: The neural network model to evaluate.
        criterion: Loss function for evaluation.
        data_loader: DataLoader providing the test data.
        device: Device to run evaluation on (CPU or GPU).
        epoch: Current epoch number.
        args: Configuration object containing evaluation parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        prefix: String prefix for logging metrics.

    Returns:
        dict: Dictionary containing evaluation metrics (average and median values).
    """
    model.eval()
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(
        lambda: train_tools.SmoothedValue(window_size=9**9)
    )
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if args.train_params.freeze_val_samples_across_all_epochs:
        dataloader_epoch = 0
    else:
        dataloader_epoch = epoch
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(dataloader_epoch)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        loss_tuple = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            use_amp=bool(args.train_params.amp),
            amp_dtype=args.train_params.amp_dtype,
            ret="loss",
        )
        loss_value, loss_details = loss_tuple  # criterion returns two values
        if n_views > 2:
            loss_value = loss_value * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        metric_logger.update(loss=float(loss_value), **loss_details)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

    return results
