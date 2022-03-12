"""Convolutional neural network models implemented in Flax."""

from functools import partial
from typing import Any, Callable, Tuple

import jax.numpy as jnp

from flax import serialization
from flax.core import Scope  # noqa
from flax.linen.module import Module, compact, _Sentinel

from scico.blockarray import BlockArray
from scico.typing import Array, Shape

# The imports of Scope and _Sentinel (above)
# are required to silence "cannot resolve forward reference"
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
        axsqz: Shape = None  # for mypy
        if xndim == 2:
            x = x.reshape((1,) + x.shape + (1,))
            axsqz = (0, 3)
        elif xndim == 3:
            x = x.reshape((1,) + x.shape)
            axsqz = (0,)
        y = self.model.apply(self.variables, x, train=False, mutable=False)
        if y.ndim != xndim:
            return y.squeeze(axis=axsqz)
        return y
