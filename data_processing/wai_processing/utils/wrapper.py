# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import traceback
from pathlib import Path
from shutil import rmtree
from typing import Callable

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from data_processing.wai_processing.utils.parallel import parallel_threads
from data_processing.wai_processing.utils.state import set_processing_state
from mapanything.utils.wai.scene_frame import get_scene_names

logger = logging.getLogger(__name__)


def get_original_scene_names(cfg):
    """
    Get all original scene ids that should be converted:
    1. Get all available scenes from the source / original dataset
    2. Filter these scenes with the scene filters set in the config
        yaml, usually filter out scenes with conversion status
        'finished'.
    """
    scene_names = get_scene_names(cfg, root=cfg.original_root)
    scene_names = get_scene_names(cfg, scene_names=scene_names)
    return scene_names


def process_single_scene(args):
    """Process a single scene for parallel execution."""
    scene_name, converter_func, cfg, kwargs = args

    scene_outpath = Path(cfg.root) / scene_name
    overwrite = cfg.get("overwrite", False)

    # Handle existing output path
    if scene_outpath.exists():
        if overwrite:
            rmtree(scene_outpath)
        else:
            raise FileExistsError(
                f"Output path already exists: {scene_outpath} - set overwrite=True if this is safe"
            )

    scene_outpath.mkdir(parents=True, exist_ok=True)
    set_processing_state(scene_outpath, "conversion", "running")

    try:
        return_state = converter_func(cfg, scene_name, **kwargs)
        if (
            (return_state is not None)
            and (return_state != "finished")
            and (return_state[0] != "finished")
        ):
            # Handle non-standard return states
            if type(return_state) is str:
                set_processing_state(scene_outpath, return_state)
            else:
                assert len(return_state) == 2, (
                    "Expected return state to either be a string, "
                    "or a tuple of length two with (<module_state>, <message>)"
                )
                set_processing_state(
                    scene_outpath, return_state[0], return_state[1]
                )
            logger.warning(f"Scene {scene_name} finished with state: {return_state}")
            return scene_name, "success", return_state
        else:
            set_processing_state(scene_outpath, "conversion", "finished")
            for processing_key, processing_val in cfg.get(
                "additional_processing_states", []
            ):
                set_processing_state(scene_outpath, processing_key, processing_val)
            return scene_name, "success", "finished"
    except Exception:
        trace_message = traceback.format_exc()
        logger.warning(
            f"\nConversion failed on scene: {scene_name}"
            f"\nError message: \n{trace_message}\n"
        )
        set_processing_state(
            scene_outpath, "conversion", "failed", message=trace_message
        )
        return scene_name, "failed", trace_message


def convert_scenes_wrapper(
    converter_func: Callable,
    cfg: DictConfig,
    get_original_scene_names_func: Callable | None = None,
    # arbitrary **kwargs forwarded to the converter_func, example in dl3dv conversion
    **kwargs,
):
    """
    Wrapper function for converting scenes that handles folder creation etc.
    Supports parallel processing using multiple threads.

    Args:
        converter_func (callable): Function to use for the per scene conversion.
        cfg (dict): Configuration dictionary.
        get_original_scene_names_func: Optional function to get scene names.
        **kwargs: Additional keyword arguments to pass to the converter function.
    """
    if get_original_scene_names_func is None:
        get_original_scene_names_func = get_original_scene_names

    scene_names = get_original_scene_names_func(cfg)
    logger.info(f"Processing: {len(scene_names)} scenes")

    # Get parallel processing configuration
    num_workers = cfg.get("num_workers", 4)  # Default to 4 workers
    front_num = cfg.get("front_num", 1)      # Process first scene sequentially for debugging

    logger.info(f"Using {num_workers} parallel workers, front_num={front_num}")

    overwrite = cfg.get("overwrite", False)
    if overwrite:
        logger.warning("Careful: Overwrite enabled")

    # Prepare arguments for parallel processing
    scene_args = [(scene_name, converter_func, cfg, kwargs) for scene_name in sorted(scene_names)]

    # Run parallel processing
    results = parallel_threads(
        process_single_scene,
        scene_args,
        workers=num_workers,
        front_num=front_num,
        desc="Converting scenes"
    )

    # Process results
    succ_scenes = []
    failed_scenes = []

    for result in results:
        scene_name, status, details = result
        if status == "success":
            succ_scenes.append(scene_name)
        else:
            failed_scenes.append(scene_name)

    logger.info(f"Finished converting {len(succ_scenes)} / {len(scene_names)} scenes successfully.")
    if failed_scenes:
        logger.warning(f"Failed scenes: {failed_scenes}")
