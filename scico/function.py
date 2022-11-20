# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Function base class."""

from typing import Callable, Optional, Sequence, Union

import jax

import scico.numpy as snp
from scico.linop import LinearOperator, jacobian
from scico.numpy import BlockArray
from scico.operator import Operator
from scico.typing import BlockShape, DType, JaxArray, Shape


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
        output_shape: Optional[Union[Shape, BlockShape]] = None,
        eval_fn: Optional[Callable] = None,
        input_dtypes: Union[DType, Sequence[DType]] = snp.float32,
        output_dtype: Optional[DType] = None,
        jit: bool = False,
    ):
        """
        Args:
            input_shapes: Shapes of input arrays.
            output_shape: Shape of output array. Defaults to ``None``.
                If ``None``, `output_shape` is determined by evaluating
                `self.__call__` on input arrays of zeros.
            eval_fn: Function used in evaluating this :class:`Function`.
                Defaults to ``None``. Required unless `__init__` is being
                called from a derived class with an `_eval` method.
            input_dtypes: `dtype` for input argument. If a single `dtype`
                is specified, it implies a common `dtype` for all inputs,
                otherwise a list or tuple of values should be provided,
                one per input. Defaults to ``float32``.
            output_dtype: `dtype` for output argument. Defaults to
                ``None``. If ``None``, `output_dtype` is determined by
                evaluating `self.__call__` on an input arrays of zeros.
            jit: If ``True``,  jit the evaluation function.
        """
        self.jit = jit
        self.input_shapes = input_shapes
        if isinstance(input_dtypes, (list, tuple)):
            self.input_dtypes = input_dtypes
        else:
            self.input_dtypes = (input_dtypes,) * len(input_shapes)

        if eval_fn is not None:
            self._eval = jax.jit(eval_fn) if jit else eval_fn
        elif not hasattr(self, "_eval"):
            raise NotImplementedError(
                "Function is an abstract base class when the eval_fn parameter is not specified."
            )

        # If the output shape or dtype isn't specified, it can be
        # inferred by calling the evaluation function.
        if output_shape is None or output_dtype is None:
            zeros = [
                snp.zeros(shape, dtype=dtype)
                for (shape, dtype) in zip(self.input_shapes, self.input_dtypes)
            ]
            tmp = self._eval(*zeros)
        if output_shape is None:
            self.output_shape = tmp.shape  # type: ignore
        else:
            self.output_shape = output_shape
        if output_dtype is None:
            self.output_dtype = tmp.dtype
        else:
            self.output_dtype = output_dtype

    def __repr__(self):
        return f"""{type(self)}
input_shapes   : {self.input_shapes}
input_dtypes   : {self.input_dtypes}
output_shape   : {self.output_shape}
output_dtype   : {self.output_dtype}
        """

    def __call__(self, *args: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        """Evaluate this function with the specified parameters.

        Args:
           *args: Parameters at which to evaluate the function.

        Returns:
           Value of function with specified parameters.
        """
        return self._eval(*args)

    def slice(self, index: int, *fix_args: Union[JaxArray, BlockArray]) -> Operator:
        """Fix all but one parameter, returning a :class:`.Operator`.

        Args:
           index: Index of parameter that remains free.
           *fix_args: Fixed values for remaining parameters.

        Returns:
           An :class:`.Operator` taking the free parameter of the
           :class:`Function` as its input.
        """

        def pfunc(var_arg):
            args = fix_args[0:index] + (var_arg,) + fix_args[index:]
            return self._eval(*args)

        return Operator(
            self.input_shapes[index],
            output_shape=self.output_shape,
            eval_fn=pfunc,
            input_dtype=self.input_dtypes[index],
            output_dtype=self.output_dtype,
            jit=self.jit,
        )

    def jacobian(
        self, index: int, *args: Union[JaxArray, BlockArray], include_eval: Optional[bool] = False
    ) -> LinearOperator:
        """Construct Jacobian linear operator for the function.

        Construct a Jacobian :class:`.LinearOperator` that computes the
        Jacobian with respect to a specified variable of the function.

        Args:
           index: Index of parameter with respect to which the Jacobian
              is to be computed.
           *args: Values of function parameters at which Jacobian is to
              be computed.
           include_eval: Flag indicating whether the result of evaluating
              the :class:`.Operator` should be included (as the first
              component of a :class:`.BlockArray`) in the output of the
              Jacobian :class:`.LinearOperator` constructed by this
              function.

        Returns:
           A :class:`.LinearOperator` capable of computing Jacobian-vector
           products.
        """
        var_arg = args[index]
        fix_args = args[0:index] + args[(index + 1) :]
        F = self.slice(index, *fix_args)
        return jacobian(F, var_arg, include_eval=include_eval)
