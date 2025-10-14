# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for loading and saving Flax models under NNX interface."""

from typing import Callable

import orbax.checkpoint as orbax

from flax import nnx


def save_model(model: Callable, path: str):
    """Save Flax NNX model.

    Args:
        model: Model to save.
        path: Path to store model.
    """
    state = nnx.state(model)
    # Save model parameters
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(f"{path}/state", state)


def load_model(path: str, model: Callable) -> Callable:
    """Load stored Flax NNX model.

    Args:
        path: Path to read model.
        model: Model to load. The model structure is
            needed to read its parameters.

    Returns:
        Model and its parameters read from given file.
    """
    state = nnx.state(model)
    # Load the parameters
    checkpointer = orbax.PyTreeCheckpointer()
    state = checkpointer.restore(f"{path}/state", item=state)
    # Update the model with the loaded state
    nnx.update(model, state)
    return model
