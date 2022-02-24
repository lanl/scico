#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generalized data handling for training script. Includes construction of data iterator and instantiation for parallel processing.
"""

from typing import Any, TypedDict, Union

from flax import jax_utils
import jax
import jax.numpy as jnp

from scico.typing import Array

DType = Any
KeyArray = Union[Array, jax._src.prng.PRNGKeyArray]


class DataSet(TypedDict):
    """Definition of the dictionary structure
    expected for the data sets for training."""

    image: Array  # Input
    label: Array  # Output


class IterateData:
    """Class to prepare image data for training and
    testing. It uses the generator pattern to obtain
    an iterable object.
    """

    def __init__(self, dt: DataSet, batch_size: int, train: bool = True, key: KeyArray = None):
        r"""Initialize a :class:`IterateData` object.

        Args:
            dt : dictionary of data for supervised training including images and labels.
            batch_size : size of batch for iterating through the data.
            train : Flag indicating use of iterator for training.  Iterator for training is infinite, iterator for testing passes once through the data.  Default: False.
            key : a PRNGKey used as the random key.  Default: None.
        """
        self.dt = dt
        self.train = train
        self.n = dt["image"].shape[0]
        self.batch_size = batch_size
        self.key = key
        if key is None:
            self.key = jax.random.PRNGKey(0)
        self.steps_per_epoch = self.n // batch_size
        self.reset()

    def reset(self):
        """Re-shuffles data in training"""
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
        """Gets next batch.
        During training it reshuffles the batches when
        the data is exhausted.
        """
        if self.ns >= self.steps_per_epoch:
            if self.train:
                self.reset()
        batch = {k: v[self.perms[self.ns], ...] for k, v in self.dt.items()}
        self.ns += 1
        return batch


def prepare_data(xs: Array) -> Any:
    """Reshape input batch for parallel training."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # reshape (host_batch_size, height, width, channels) to
        # (local_devices, device_batch_size, height, width, channels)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def create_input_iter(
    key: KeyArray,
    dataset: DataSet,
    batch_size: int,
    size_device_prefetch: int = 2,
    dtype: DType = jnp.float32,
    train: bool = True,
) -> Any:
    """Create split for training data.

    Args:
        key : a PRNGKey used for random data permutations.
        dataset : dictionary of data for supervised training including images and labels.
        batch_size : size of batch for iterating through the data.
        size_device_prefetch : size of prefetch buffer. Default: 2.
        dtype : class of data to handle. Default: `jnp.float32`.
        train : Flag indicating use of iterator for training.  Iterator for training is infinite, iterator for testing passes once through the data.  Default: True.
    """
    ds = IterateData(dataset, batch_size, train, key)
    it = map(prepare_data, ds)
    it = jax_utils.prefetch_to_device(it, size_device_prefetch)
    return it
