# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Convolutional neural network models implemented in Flax."""

import warnings
from typing import Any, Optional

warnings.simplefilter(action="ignore", category=FutureWarning)

from flax import serialization
from flax.linen.module import Module
from scico.numpy import Array, BlockArray
from scico.typing import Shape

PyTree = Any


def load_weights(filename: str) -> PyTree:
    """Load trained model weights.

    Args:
        filename: Name of file containing parameters for trained model.

    Returns:
        A tree-like structure containing the values of the parameters of
        the model.
    """
    with open(filename, "rb") as data_file:
        bytes_input = data_file.read()

    variables = serialization.msgpack_restore(bytes_input)

    var_in = {"params": variables["params"], "batch_stats": variables["batch_stats"]}

    return var_in


def save_weights(variables: PyTree, filename: str):
    """Save trained model weights.

    Args:
        filename: Name of file to save parameters of trained model.
        variables: Parameters of model to save.
    """
    bytes_output = serialization.msgpack_serialize(variables)

    with open(filename, "wb") as data_file:
        data_file.write(bytes_output)


class FlaxMap:
    r"""A trained flax model."""

    def __init__(self, model: Module, variables: PyTree):
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
        #   scico typically works with (H x W) or (H x W x C) arrays
        #   flax expects (K x H x W x C) arrays
        #   H: spatial height  W: spatial width
        #   K: batch size  C: channel size
        xndim = x.ndim
        axsqueeze: Optional[Shape] = None
        if xndim == 2:
            x = x.reshape((1,) + x.shape + (1,))
            axsqueeze = (0, 3)
        elif xndim == 3:
            x = x.reshape((1,) + x.shape)
            axsqueeze = (0,)
        y = self.model.apply(self.variables, x, train=False, mutable=False)
        if y.ndim != xndim:
            return y.squeeze(axis=axsqueeze)
        return y
