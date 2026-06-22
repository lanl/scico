# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utils for spectral normalization of convolutional layers in Flax models."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Any, Callable, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import scipy
from flax.core import freeze, unfreeze
from flax.linen import Conv
from flax.linen.module import Module, compact
from scico.numpy import Array
from scico.typing import Shape

from .traversals import ModelParamTraversal

PyTree = Any


# From https://github.com/deepmind/dm-haiku/issues/71
def _l2_normalize(x: Array, eps: float = 1e-12) -> Array:
    r"""Normalize array by its :math:`\el_2` norm.

    Args:
        x: Array to be normalized.
        eps: Small value to prevent divide by zero. Default: 1e-12.

    Returns:
        Normalized array.
    """
    return x * lax.rsqrt((x**2).sum() + eps)


# From https://nbviewer.org/gist/shoyer/fa9a29fd0880e2e033d7696585978bfc
def estimate_spectral_norm(
    f: Callable, input_shape: Shape, seed: float = 0, n_steps: int = 10, eps: float = 1e-12
):
    """Estimate spectral norm of operator.

    This function estimates the spectral norm of an operator by
    estimating the singular vectors of the operator via the power
    iteration method and the transpose operator enabled by nested
    autodiff in JAX.

    Args:
        f: Operator to compute spectral norm.
        input_shape: Shape of input to operator.
        seed: Value to seed the random generation. Default: 0.
        n_steps: Number of power iterations to compute. Default: 10.
        eps: Small value to prevent divide by zero. Default: 1e-12.

    Returns:
        Spectral norm.
    """
    rng = jax.random.key(seed)
    u0 = jax.random.normal(rng, input_shape)
    v0 = jnp.zeros_like(f(u0))

    def fun(carry, _):
        u, v = carry
        v, f_vjp = jax.vjp(f, u)
        v = _l2_normalize(v, eps)
        (u,) = f_vjp(v)
        u = _l2_normalize(u, eps)
        return (u, v), None

    (u, v), _ = lax.scan(fun, (u0, v0), xs=None, length=n_steps)
    return jnp.vdot(v, f(u))


class CNN(Module):
    """Evaluation of convolution operator via Flax convolutional layer.

    Evaluation of convolution operator via Flax implementation of a
    convolutional layer. This is form of convolution is used only for the
    estimation of the spectral norm of the operator. Therefore, the value
    of the kernel is provided too.

    Attributes:
        kernel_size: Size of the convolution filter.
        kernel0: Convolution filter.
        dtype: Output type.
    """

    kernel_size: Sequence[int]
    kernel0: Array
    dtype: Any

    @compact
    def __call__(self, x):
        """Apply CNN layer.

        Args:
            x: The array to be convolved.

        Returns:
            The result of the convolution with `kernel0`.
        """

        def kinit_wrap(rng, shape, dtype=self.dtype):
            return jnp.array(self.kernel0, dtype)

        return Conv(
            features=self.kernel_size[3],
            kernel_size=self.kernel_size[:2],
            use_bias=False,
            padding="CIRCULAR",
            kernel_init=kinit_wrap,
        )(x)


def conv(inputs: Array, kernel: Array) -> Array:
    """Compute convolution betwen input and kernel.

    The convolution is evaluated via a CNN Flax model.

    Args:
        inputs: Array to compute convolution.
        kernel: Filter of the convolutional operator.

    Returns:
        Result of convolution of input with kernel.
    """

    dtype = kernel.dtype
    inputs = jnp.asarray(inputs, dtype)
    kernel = jnp.asarray(kernel, dtype)

    rng = jax.random.key(0)  # not used
    model = CNN(kernel_size=kernel.shape, kernel0=kernel, dtype=dtype)
    variables = model.init(rng, np.zeros(inputs.shape))
    y = model.apply(variables, inputs)

    return y


def spectral_normalization_conv(
    params: PyTree, traversal: ModelParamTraversal, xshape: Shape, n_steps: int = 10
) -> PyTree:
    """Normalize parameters of convolutional layer by its spectral norm.

    Args:
        params: Current model parameters.
        traversal: Utility to select model parameters.
        xshape: Shape of input.
        n_steps: Number of power iterations to compute. Default: 10.
    """
    params_out = traversal.update(
        lambda kernel: kernel
        / (
            estimate_spectral_norm(
                lambda x: conv(x, kernel), (1, xshape[1], xshape[2], kernel.shape[2]), n_steps
            )
            * 1.02
        ),
        unfreeze(params),
    )

    return freeze(params_out)


# From https://nbviewer.org/gist/shoyer/fa9a29fd0880e2e033d7696585978bfc
def exact_spectral_norm(f, input_shape):
    """Compute spectral norm of operator.

    This function computes the spectral norm of an operator via autodiff
    in JAX.

    Args:
        f: Operator to compute spectral norm.
        input_shape: Shape of input to operator.

    Returns:
        Spectral norm.
    """
    dummy_input = jnp.zeros(input_shape)
    jacobian = jax.jacfwd(f)(dummy_input)
    shape = (np.prod(jacobian.shape[: -dummy_input.ndim]), np.prod(input_shape))
    return scipy.linalg.svdvals(jacobian.reshape(shape)).max()
