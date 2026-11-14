# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for checkpointing Flax models."""

import importlib
import json
from pathlib import Path

import jax

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

from .typed_dict import ConfigDict


def checkpoint_restore(state, workdir: str | Path, ok_no_ckpt: bool = False):
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
        returned. Among all the checkpoints available in the specified
        path, this function looks for the latest step, i.e. the
        checkpoint at the largest number of training iterations,
        and reads the corresponding checkpoint. The actual step index
        is also returned. If no checkpoint files are present and checkpoints
        are not strictly expected, it returns the passed-in `state` unchanged
        and step as ``None``.

    Raises:
        FileNotFoundError: If a checkpoint is expected and is not found.
    """
    if not have_orbax:
        raise RuntimeError("Package orbax.checkpoint is required for use of this function.")
    step = None
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
            print(f"Model read from checkpoint id {step}\n\n")
    elif not ok_no_ckpt:
        raise FileNotFoundError("Could not read from checkpoint: " + str(workdir) + ".")

    return state, step


def checkpoint_save(state, epoch: int, config: ConfigDict, workdir: str | Path):
    """Store model, model configuration, and optimiser state.

    Note that naming is slightly different to distinguish from Flax
    functions.

    Args:
        state: Flax train state which includes model and optimiser
            parameters.
        epoch: Epoch where the model is being saved.
        config: Python dictionary including model train configuration.
        workdir: Path in which to store checkpoint files.
    """
    if not have_orbax:
        raise RuntimeError("Package orbax.checkpoint is required for use of this function.")
    if jax.process_index() == 0:
        # Check if workdir is Path or convert to Path
        workdir_ = workdir
        if isinstance(workdir_, str):
            workdir_ = Path(workdir_)
        options = ocp.CheckpointManagerOptions(preservation_policy=LatestN(3), create=True)
        mngr = ocp.CheckpointManager(
            workdir_,
            options=options,
        )
        step = epoch
        ## Remove non-serializable partial functools in post_lst if it exists
        # config_ = config.copy()
        # if "post_lst" in config_:
        #    config_.pop("post_lst", None)  # type: ignore
        mngr.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                # config=ocp.args.JsonSave(config_),
                config=ocp.args.JsonSave(serialize_metadata(config)),
            ),
        )
        mngr.wait_until_finished()
        mngr.close()


def checkpoint_restore_with_config(state, workdir: str | Path, ok_no_ckpt: bool = False):
    """Load model, optimiser state and trainer configuration.

    Args:
        state: Flax train state which includes model and optimiser
            parameters.
        workdir: Checkpoint file or directory of checkpoints to restore
            from.
        ok_no_ckpt: Flag to indicate if a checkpoint is expected. If
            ``False``, an error is generated if a checkpoint is not
            found.

    Returns:
        A restored Flax train state and training configuration updated from
        checkpoint file are returned. If no checkpoint files are present and checkpoints are
        not strictly expected it returns the passed-in `state` and `config` unchanged.

    Raises:
        FileNotFoundError: If a checkpoint is expected and is not found.
    """
    if not have_orbax:
        raise RuntimeError("Package orbax.checkpoint is required for use of this function.")
    step = None
    config = None
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
                step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(state), config=ocp.args.JsonRestore()
                ),
            )
            mngr.wait_until_finished()
            mngr.close()
            state = restored.state
            config = deserialize_metadata(restored.config)
            print(f"Model read from checkpoint id {step}\n\n")
    elif not ok_no_ckpt:
        raise FileNotFoundError("Could not read from checkpoint: " + str(workdir) + ".")

    return state, config, step


def serialize_metadata(obj):
    """Function to preprocess metadata that is not serializable
    before saving to file.

    Args:
        obj: Object to serialize.

    Returns:
        A serialized instance.
    """
    if isinstance(obj, dict):
        return {str(k): serialize_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_metadata(v) for v in obj]
    elif callable(obj) and hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
        # Store full import path so it can be reconstructed later
        return {"__callable__": f"{obj.__module__}.{obj.__qualname__}"}
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)


def deserialize_metadata(obj):
    """Function to preprocess metadata to convert it to the
    original object after reading from file.

    Args:
        obj: Object to deserialize.

    Returns:
        A deserialized instance.
    """
    if isinstance(obj, dict):
        # Check if this dict represents a serialized class or function
        if set(obj.keys()) == {"__callable__"}:
            module_path, _, attr_path = obj["__callable__"].rpartition(".")
            module = importlib.import_module(module_path)
            target = module
            for attr in attr_path.split("."):
                target = getattr(target, attr)
            return target
        return {k: deserialize_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_metadata(v) for v in obj]
    return obj
