# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for checkpointing Flax models."""


from pathlib import Path
from typing import Union

try:
    import orbax.checkpoint as ocp

    have_orbax = True
    if not hasattr(ocp, "CheckpointManager"):
        have_orbax = False
except ImportError:
    have_orbax = False

if have_orbax:
    import logging

    from orbax.checkpoint.checkpoint_managers import LatestN

    logging.getLogger("absl").addFilter(logging.Filter("could not be identified as a temporary"))

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
        ok_no_ckpt: Flag to indicate if a checkpoint is expected. If
            ``False``, an error is generated if a checkpoint is not
            found.

    Returns:
        A restored Flax train state updated from checkpoint file is
        returned. If no checkpoint files are present and checkpoints are
        not strictly expected it returns the passed-in `state` unchanged.

    Raises:
        FileNotFoundError: If a checkpoint is expected and is not found.
    """
    if not have_orbax:
        raise RuntimeError("Package orbax.checkpoint is required for use of this function.")
    # Check if workdir is Path or convert to Path
    workdir_ = workdir
    if isinstance(workdir_, str):
        workdir_ = Path(workdir_)
    if workdir_.exists():
        mngr = ocp.CheckpointManager(
            workdir_,
        )
        step = mngr.latest_step()
        if step is not None:
            restored = mngr.restore(
                step, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
            )
            mngr.wait_until_finished()
            mngr.close()
            state = restored.state
    elif not ok_no_ckpt:
        raise FileNotFoundError("Could not read from checkpoint: " + str(workdir) + ".")

    return state


def checkpoint_save(state: TrainState, config: ConfigDict, workdir: Union[str, Path]):
    """Store model, model configuration, and optimiser state.

    Note that naming is slightly different to distinguish from Flax
    functions.

    Args:
        state: Flax train state which includes model and optimiser
            parameters.
        config: Python dictionary including model train configuration.
        workdir: Path in which to store checkpoint files.
    """
    if not have_orbax:
        raise RuntimeError("Package orbax.checkpoint is required for use of this function.")
    # Check if workdir is Path or convert to Path
    workdir_ = workdir
    if isinstance(workdir_, str):
        workdir_ = Path(workdir_)
    options = ocp.CheckpointManagerOptions(preservation_policy=LatestN(3), create=True)
    mngr = ocp.CheckpointManager(
        workdir_,
        options=options,
    )
    step = int(state.step)
    # Remove non-serializable partial functools in post_lst if it exists
    config_ = config.copy()
    if "post_lst" in config_:
        config_.pop("post_lst", None)  # type: ignore
    mngr.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            config=ocp.args.JsonSave(config_),
        ),
    )
    mngr.wait_until_finished()
    mngr.close()
