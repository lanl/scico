# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Function base class."""

from typing import Callable, Optional, Sequence, Union

import jax

from scico.numpy import BlockArray
from scico.typing import BlockShape, JaxArray, Shape


class Function:
    r"""Base class for functions.

    A :class:`Function` maps multiple :code:`array-like` arguments to
    another :code:`array-like`. It is more general than both
    :class:`.Functional`, which is a mapping to a scalar, and
    :class:`.Operator`, which takes a single argument.
    """

    def __init__(
        self,
        input_shapes: Sequence[Union[Shape, BlockShape]],
        output_shape: Union[Shape, BlockShape],
        eval_fn: Optional[Callable] = None,
        jit: bool = False,
    ):
        """
        Args:
            input_shapes: Shapes of input arrays.
            output_shape: Shape of output array.
            eval_fn: Function used in evaluating this :class:`Function`.
                Defaults to ``None``. Required unless `__init__` is being
                called from a derived class with an `_eval` method.
            jit: If ``True``,  jit the evaluation function.
        """
        self.input_shapes = input_shapes
        self.output_shape = output_shape
        if eval_fn is not None:
            self._eval = jax.jit(eval_fn) if jit else eval_fn
        elif not hasattr(self, "_eval"):
            raise NotImplementedError(
                "Function is an abstract base class when the eval_fn parameter is not specified."
            )

    def __repr__(self):
        return f"""{type(self)}
input_shapes   : {self.input_shapes}
output_shape   : {self.output_shape}
        """

    def __call__(self, *args: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        r"""Evaluate this function with the specified parameters.

        Args:
           *args: Parameters at which to evaluate the function.

        Returns:
           Value of function with specified parameters.
        """
        return self._eval(*args)
