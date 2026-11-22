# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Construction of data iterator for training script."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from collections.abc import Generator

import numpy as np
import numpy.typing as npt

from .typed_dict import DataSetDict


def iterate_xy_dataset(
    ds: DataSetDict,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    shuffle_buffer_size: int = 10000,
    drop_last: bool = False,
) -> Generator[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], None, None]:
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
        drop_last: If ``True``, drop the final batch when the number of
            samples is not an exact multiple of the batch size.

    Yields:
        Input and output arrays, of at most `batch_size` samples each.
    """
    ndata = ds["image"].shape[0]

    if shuffle:
        if seed is None:
            seed = 0
        # Use NumPy for shuffling (more efficient for large datasets)
        rng = np.random.default_rng(seed)
        # Permute within each buffer-sized block of indices rather than
        # across the whole data set
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
    for i in range(0, num_batches * batch_size, batch_size):
        batch_indices = indices[i : i + batch_size]

        # Index using NumPy arrays to avoid device memory issues
        x = np.asarray(ds["image"][batch_indices])
        y = np.asarray(ds["label"][batch_indices])
        yield x, y


def iterate_x_dataset(
    ds: DataSetDict,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    shuffle_buffer_size: int = 10000,
    drop_last: bool = False,
) -> Generator[npt.NDArray[np.float32], None, None]:
    """Yield chunks of dataset for training/evaluating ML model.

    Yield the whole dataset in chunks of size `batch_size`. It
    uses NumPy for shuffling (more memory efficient). Avoids materializing
    large arrays on device. Uses a shuffle buffer instead of shuffling
    the entire dataset.

    Args:
        ds: Data set to iterate. It is a dictionary where the `image`
            keyword defines the input (feature) data. Any `label` entry is
            ignored.
        batch_size: Number of samples in each chunk.
        shuffle: If ``True``, the data is randomly ordered. Otherwise,
            the data is returned with the ordering of the original dataset.
        seed: Seed for NumPy random generation.
        shuffle_buffer_size: Size of buffer for shuffling data indices.
        drop_last: If ``True``, drop the final batch when the number of
            samples is not an exact multiple of the batch size.

    Yields:
        Input arrays, of at most `batch_size` samples each.
    """
    ndata = ds["image"].shape[0]

    if shuffle:
        if seed is None:
            seed = 0
        # Use NumPy for shuffling (more efficient for large datasets)
        rng = np.random.default_rng(seed)
        # Permute within each buffer-sized block of indices rather than
        # across the whole data set
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
    for i in range(0, num_batches * batch_size, batch_size):
        batch_indices = indices[i : i + batch_size]

        # Index using NumPy arrays to avoid device memory issues
        x = np.asarray(ds["image"][batch_indices])
        yield x
