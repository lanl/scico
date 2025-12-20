# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for loading and saving Flax models under NNX interface."""

from typing import Callable

from pathlib import Path
import pickle

from flax import nnx
from flax.serialization import to_state_dict, from_state_dict


def save_model(model: Callable, file_path: str, file_name: str):
    """Save Flax model.
    
    Function for saving a Flax NNX neural network model. 

    Args:
        model: Flax model to save.
        file_path: Absolute path where model is to be saved.
        file_name: Filename to save to.
    """
    # Create path
    path = Path(file_path + "/" + file_name + ".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the model state (e.g. parameter values)
    state = nnx.state(model)

    # Convert to a pure dictionary
    # This removes VariableState objects and leaves only Arrays
    pure_dict = to_state_dict(state)

    with open(path, "wb") as pickle_file:
        pickle.dump(pure_dict, pickle_file)


def load_model(model: Callable, file_path: str, file_name: str):
    """Load Flax model.
    
    Function for loading a Flax NNX neural network model. 

    Args:
        model: Flax model to load.
        file_path: Absolute path where model is saved.
        file_name: Filename to load from.
    """
    # Use model instance to serve as the target structure
    state_target = nnx.state(model)

    # Load the saved pure dictionary
    path = Path(file_path + "/" + file_name + ".pkl")
    with open(path, "rb") as pickle_file:
        loaded_pure_dict = pickle.load(pickle_file)

    #print(f"Loaded dict: {loaded_pure_dict}")

    # Restore the state into the target structure
    restored_state = from_state_dict(state_target, loaded_pure_dict)

    # Update the model with the loaded state
    nnx.update(model, restored_state)
    
    return model
