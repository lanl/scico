# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for checkpointing Flax models."""
from pathlib import Path
from typing import Union

import jax

import orbax

from flax.training import orbax_utils

from .state import TrainState
from .typed_dict import ConfigDict


def checkpoint_restore(
    state: TrainState, workdir: Union[str, Path], ok_no_ckpt: bool = False
) -> TrainState:
    """Load model and optimiser state.

    Args:
        state: Flax train state which includes model and optimiser
            parameters.
        workdir: Checkpoint file or directory of checkpoints to restore
            from.
        ok_no_ckpt: Flag to indicate if a checkpoint is expected. Default:
                    False, a checkpoint is expected and an error is generated.

    Returns:
        A restored Flax train state updated from checkpoint file is returned.
        If no checkpoint files are present and checkpoints are not strictly
        expected it returns the passed-in `state` unchanged.

    Raises:
        FileNotFoundError: If a checkpoint is expected and is not found.
    """
    # Check if workdir is Path or convert to Path
    workdir_ = workdir
    if isinstance(workdir_, str):
        workdir_ = Path(workdir_)
    if workdir_.exists():
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_manager = orbax.checkpoint.CheckpointManager(workdir_, orbax_checkpointer)
        step = checkpoint_manager.latest_step()
        if step is not None:
            target = {"state": state, "config": {}}
            ckpt = checkpoint_manager.restore(step, items=target)
            state = ckpt["state"]
    elif not ok_no_ckpt:
        raise FileNotFoundError("Could not read from checkpoint: " + str(workdir))

    return state


def checkpoint_save(state: TrainState, config: ConfigDict, workdir: Union[str, Path]):
    """Store model, model configuration, and optimiser state.

    Note that naming is slightly different to distinguish from Flax
    functions.

    Args:
        state: Flax train state which includes model and optimiser
            parameters.
        config: Python dictionary including model train configuration.
        workdir: str or pathlib-like path to store checkpoint files in.
    """
    if jax.process_index() == 0:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # Bundle config and model parameters together
        ckpt = {"state": state, "config": config}
        save_args = orbax_utils.save_args_from_target(ckpt)
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=True)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            workdir, orbax_checkpointer, options
        )
        step = int(state.step)
        checkpoint_manager.save(step, ckpt, save_kwargs={"save_args": save_args})
