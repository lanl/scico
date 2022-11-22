# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for checkpointing Flax models."""
import os
from typing import Union

import jax

try:
    from tensorflow.io import gfile  # noqa: F401
except ImportError:
    have_tf = False
else:
    have_tf = True

if have_tf:
    from flax.training import checkpoints

from .state import TrainState


# Flax checkpoints
def restore_checkpoint(
    state: TrainState, workdir: Union[str, os.PathLike]
) -> TrainState:  # pragma: no cover
    """Load model and optimiser state.

    Args:
        state: Flax train state which includes model and optimiser
            parameters.
        workdir: checkpoint file or directory of checkpoints to restore
            from.

    Returns:
        Restored `state` updated from checkpoint file, or if no
        checkpoint files present, returns the passed-in `state` unchanged.
    """
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state: TrainState, workdir: Union[str, os.PathLike]):  # pragma: no cover
    """Store model and optimiser state.

    Args:
        state: Flax train state which includes model and optimiser
            parameters.
        workdir: str or pathlib-like path to store checkpoint files in.
    """
    if jax.process_index() == 0:
        # get train state from first replica
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)
