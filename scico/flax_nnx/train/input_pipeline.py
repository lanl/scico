# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Construction of data iterator for training script."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from .typed_dict import DataSetDict


def iterate_xy_dataset(
    ds: DataSetDict,
    batch_size: int,
    shuffle: bool = False,
    seed: Optional[int] = None,
    shuffle_buffer_size: int = 10000,
    drop_last: bool = False,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Yield chunks of dataset for training/evaluating ML model.

    Yield the whole dataset in chunks of size `batch_size`. It
    uses NumPy for shuffling (more memory efficient). Avoids materializing
    large arrays on device. Uses a shuffle buffer instead of shuffling
    the entire dataset.

    Args:
        ds: Data set to iterate. It is a dictionary where `image` keyword
            defines the input (feature) data and `label` keyword defines
            the output data.
        batch_size: Number of samples in each chunk.
        shuffle: If ``True``, the data is randomly ordered. Otherwise,
            the data is returned with the ordering of the original dataset.
        seed: Seed for NumPy random generation.
        shuffle_buffer_size: Size of buffer for shuffling data indices.
        drop_last: Drop the last batch if the set is not exactly divisible
            by the batch size.

    Returns:
        Input and output arrays.
    """
    ndata = ds["image"].shape[0]

    if shuffle:
        if seed is None:
            seed = 0
        # Use NumPy for shuffling (more efficient for large datasets)
        rng = np.random.default_rng(seed)
        # Use reservoir sampling for large datasets
        indices = np.arange(ndata)
        for i in range(0, ndata, shuffle_buffer_size):
            buffer_end = min(i + shuffle_buffer_size, ndata)
            rng.shuffle(indices[i:buffer_end])
    else:
        indices = np.arange(ndata)

    # Calculate actual number of batches
    num_batches = ndata // batch_size
    if not drop_last and ndata % batch_size != 0:
        num_batches += 1

    # Iterate and transfer only needed batches to device
    for i in range(0, ndata, batch_size):
        batch_indices = indices[i : i + batch_size]

        # Index using NumPy arrays to avoid device memory issues
        x = np.asarray(ds["image"][batch_indices])
        y = np.asarray(ds["label"][batch_indices])
        yield x, y


def iterate_x_dataset(
    ds: DataSetDict,
    batch_size: int,
    shuffle: bool = False,
    seed: Optional[int] = None,
    shuffle_buffer_size: int = 10000,
    drop_last: bool = False,
) -> npt.NDArray[np.float32]:
    """Yield chunks of dataset for training/evaluating ML model.

    Yield the whole dataset in chunks of size `batch_size`. It
    uses NumPy for shuffling (more memory efficient). Avoids materializing
    large arrays on device. Uses a shuffle buffer instead of shuffling
    the entire dataset.

    Args:
        ds: Data set to iterate. It is a dictionary where `image` keyword
            defines the input (feature) data and `label` keyword defines
            the output data.
        batch_size: Number of samples in each chunk.
        shuffle: If ``True``, the data is randomly ordered. Otherwise,
            the data is returned with the ordering of the original dataset.
        seed: Seed for NumPy random generation.
        shuffle_buffer_size: Size of buffer for shuffling data indices.
        drop_last: Drop the last batch if the set is not exactly divisible
            by the batch size.

    Returns:
        Input arrays.
    """
    ndata = ds["image"].shape[0]

    if shuffle:
        if seed is None:
            seed = 0
        # Use NumPy for shuffling (more efficient for large datasets)
        rng = np.random.default_rng(seed)
        # Use reservoir sampling for large datasets
        indices = np.arange(ndata)
        for i in range(0, ndata, shuffle_buffer_size):
            buffer_end = min(i + shuffle_buffer_size, ndata)
            rng.shuffle(indices[i:buffer_end])
    else:
        indices = np.arange(ndata)

    # Calculate actual number of batches
    num_batches = ndata // batch_size
    if not drop_last and ndata % batch_size != 0:
        num_batches += 1

    # Iterate and transfer only needed batches to device
    for i in range(0, ndata, batch_size):
        batch_indices = indices[i : i + batch_size]

        # Index using NumPy arrays to avoid device memory issues
        x = np.asarray(ds["image"][batch_indices])
        yield x
