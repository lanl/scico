# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Convolutional neural network models implemented in Flax."""

from functools import partial
from typing import Any, Callable, Tuple

import jax.numpy as jnp

from flax import serialization
from flax.core import Scope  # noqa
from flax.linen.module import Module, _Sentinel  # noqa

from scico.blockarray import BlockArray
from scico.typing import Array

# The imports of Scope and _Sentinel (above) and the definition of Module
# (below) are required to silence "cannot resolve forward reference"
# warnings when building sphinx api docs.


ModuleDef = Any
PyTree = Any


def load_weights(filename: str):
    """Load trained model weights.

    Args:
        filename: Name of file containing parameters for trained model.
    """
    with open(filename, "rb") as data_file:
        bytes_input = data_file.read()

    variables = serialization.msgpack_restore(bytes_input)

    return variables


class FlaxMap:
    r"""A trained flax model."""

    def __init__(self, model: Callable[..., Module], variables: PyTree):
        r"""Initialize a :class:`FlaxMap` object.

        Args:
            model: Flax model to apply.
            variables: Parameters and batch stats of trained model.
        """
        self.model = model
        self.variables = variables
        super().__init__()

    def __call__(self, x: Array) -> Array:
        r"""Apply trained flax model.

        Args:
            x: Input array.

        Returns:
            Output of flax model.
        """
        if isinstance(x, BlockArray):
            raise NotImplementedError

        # Add singleton to input as necessary:
        #   scico typically works with (H x W) or (H x Wx C) arrays
        #   flax expects (K x H x W x C) arrays
        #   H: spatial height  W: spatial width
        #   K: batch size  C: channel size
        x_shape = x.shape
        if x.ndim == 2:
            x = x.reshape((1,) + x.shape + (1,))
        elif x.ndim == 3:
            x = x.reshape((1,) + x.shape)
        y = self.model.apply(self.variables, x, train=False, mutable=False)
        return y.reshape(x_shape)
