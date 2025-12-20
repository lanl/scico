# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Generalized data handling for training script.

Includes construction of data iterator and
instantiation for parallel processing.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Any, Union

import jax
import jax.numpy as jnp

from flax import jax_utils
from scico.numpy import Array

from .typed_dict import DataSetDict

DType = Any
KeyArray = Union[Array, jax.Array]


class IterateData:
    """Class to load data for training and testing.

    It uses the generator pattern to obtain an iterable object.
    """

    def __init__(self, dt: DataSetDict, batch_size: int, train: bool = True, key: KeyArray = None):
        r"""Initialize a :class:`IterateData` object.

        Args:
            dt: Dictionary of data for supervised training including
               images and labels.
            batch_size: Size of batch for iterating through the data.
            train: Flag indicating use of iterator for training. Iterator
                for training is infinite, iterator for testing passes
                once through the data. Default: ``True``.
            key: A PRNGKey used as the random key. Default: ``None``.
        """
        self.dt = dt
        self.batch_size = batch_size
        self.train = train
        self.n = dt["image"].shape[0]
        self.key = key
        if key is None:
            self.key = jax.random.key(0)
        self.steps_per_epoch = self.n // batch_size
        self.reset()

    def reset(self):
        """Re-shuffle data in training."""
        if self.train:
            self.key, subkey = jax.random.split(self.key)
            self.perms = jax.random.permutation(subkey, self.n)
        else:
            self.perms = jnp.arange(self.n)

        self.perms = self.perms[: self.steps_per_epoch * self.batch_size]  # skips incomplete batch
        self.perms = self.perms.reshape((self.steps_per_epoch, self.batch_size))
        self.ns = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch.

        During training it reshuffles the batches when the data is
        exhausted."""
        if self.ns >= self.steps_per_epoch:
            if self.train:
                self.reset()
            else:
                self.ns = 0
        batch = {k: v[self.perms[self.ns], ...] for k, v in self.dt.items()}
        self.ns += 1
        return batch


def prepare_data(xs: Array) -> Any:
    """Reshape input batch for parallel training."""
    local_device_count = jax.local_device_count()

    def _prepare(x: Array) -> Array:
        # reshape (host_batch_size, height, width, channels) to
        # (local_devices, device_batch_size, height, width, channels)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(
    key: KeyArray,
    dataset: DataSetDict,
    batch_size: int,
    size_device_prefetch: int = 2,
    dtype: DType = jnp.float32,
    train: bool = True,
) -> Any:
    """Create data iterator for training.

    Create data iterator for training by sharding and prefetching batches
    on device.

    Args:
        key: A PRNGKey used for random data permutations.
        dataset: Dictionary of data for supervised training including
            images and labels.
        batch_size: Size of batch for iterating through the data.
        size_device_prefetch: Size of prefetch buffer. Default: 2.
        dtype: Type of data to handle. Default: :attr:`~numpy.float32`.
        train: Flag indicating the type of iterator to construct and use.
            The iterator for training permutes data on each epoch while
            the iterator for testing passes through the data without
            permuting it. Default: ``True``.

    Returns:
        Array-like data sharded to specific devices coming from an
        iterator built from the provided dataset.
    """
    ds = IterateData(dataset, batch_size, train, key)
    it = map(prepare_data, ds)
    it = jax_utils.prefetch_to_device(it, size_device_prefetch)
    return it
