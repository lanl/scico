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

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .typed_dict import DataSetDict


def iterate_xy_dataset(
    ds: DataSetDict,
    steps: int,
    batch_size: int,
    subkey: Optional[ArrayLike] = None,
    shuffle: bool = False,
) -> Tuple[ArrayLike, ArrayLike]:
    """Yield chunks of dataset for training/evaluating ML model.

    Yield a number of `steps` chunks of the dataset each of size `batch_size`.

    Args:
        ds: Data set to iterate. It is a dictionary where `image` keyword
            defines the input (feature) data and `label` keyword defines
            the output data.
        steps: Number of data chunks to collect.
        batch_size: Number of samples in each chunk.
        subkey: JAX random generation.
        shuffle: If ``True``, the data is randomly ordered. Otherwise,
            the data is returned with the ordering of the original dataset.

    Returns:
        Input and output arrays.
    """
    ndata = ds["image"].shape[0]

    if shuffle:
        if subkey is None:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, ndata)
    else:
        perms = jnp.arange(ndata)

    for i in range(steps):
        x = ds["image"][perms[i * batch_size : (i + 1) * batch_size]]
        y = ds["label"][perms[i * batch_size : (i + 1) * batch_size]]
        yield x, y


def iterate_x_dataset(
    ds: DataSetDict,
    steps: int,
    batch_size: int,
    subkey: Optional[ArrayLike] = None,
    shuffle: bool = False,
) -> ArrayLike:
    """Yield chunks of dataset for training/evaluating ML model.

    Yield a number of `steps` chunks of the dataset each of size `batch_size`.
    Only input data (i.e. no labels) are yielded.

    Args:
        ds: Data set to iterate. It is a dictionary where `image` keyword
            defines the input (feature) data and `label` keyword defines
            the output data.
        steps: Number of data chunks to collect.
        batch_size: Number of samples in each chunk.
        subkey: JAX random generation.
        shuffle: If ``True``, the data is randomly ordered. Otherwise,
            the data is returned with the ordering of the original dataset.

    Returns:
        Input arrays.
    """
    ndata = ds["image"].shape[0]

    if shuffle:
        if subkey is None:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, ndata)
    else:
        perms = jnp.arange(ndata)

    for i in range(steps):
        x = ds["image"][perms[i * batch_size : (i + 1) * batch_size]]
        yield x
