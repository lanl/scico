# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for loading and saving Flax models under NNX interface."""

import pickle
from io import BufferedReader, BufferedWriter
from typing import Callable, Union

from flax import nnx
from flax.serialization import from_state_dict, to_state_dict


def save_model(model: Callable, file: Union[str, BufferedWriter]):
    """Save Flax model.

    Save a Flax NNX neural network model.

    Args:
        model: Flax model to save.
        file: Filename or file stream object.
    """
    # Get the model state (e.g. parameter values)
    state = nnx.state(model)

    # Convert to a pure dictionary. This removes VariableState objects
    # and leaves only Arrays
    pure_dict = to_state_dict(state)

    if isinstance(file, str):
        with open(path, "wb") as f:
            pickle.dump(pure_dict, f)
    else:
        pickle.dump(pure_dict, file)


def load_model(model: Callable, file: Union[str, BufferedReader]) -> Callable:
    """Load Flax model.

    Load a Flax NNX neural network model.

    Args:
        model: Flax model to load.
        file: Filename or file stream object.

    Returns:
        Model with restored data.
    """
    # Use model instance to serve as the target structure
    state_target = nnx.state(model)

    # Load the saved pure dictionary
    if isinstance(file, str):
        with open(file, "rb") as f:
            pure_dict = pickle.load(f)
    else:
        pure_dict = pickle.load(file)

    # Restore the state into the target structure
    restored_state = from_state_dict(state_target, pure_dict)

    # Update the model with the loaded state
    nnx.update(model, restored_state)

    return model
