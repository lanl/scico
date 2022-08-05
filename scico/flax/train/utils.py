#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utils for spectral normalization of convolutional layers in Flax models.
"""

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

from flax.core import freeze, unfreeze
from flax.linen.linear import _conv_dimension_numbers
from flax.traverse_util import ModelParamTraversal
from scico.typing import Array, Shape

PyTree = Any

PRNGKey = jax.random.PRNGKey(10)

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


def _l2_norm(x: Array) -> float:
    r"""Compute :math:`\el_2` norm of array.

    Args:
        x: Array to compute norm.
    """
    return jnp.sqrt((x**2).sum())


def _power_iteration(A: Callable, u: Array, n_steps: int = 10) -> Array:
    """Update an estimate of the first right-singular vector of A().

    Args:
        A: Operator to compute spectral norm.
        u: Current estimate of the first right-singular vector of A.
        n_steps: Number of iterations in power method.

    Returns:
        Updated estimate.
    """

    def fun(u, _):
        v, A_transpose = jax.vjp(A, u)
        (u,) = A_transpose(v)
        u = _l2_normalize(u)
        return u, None

    u, _ = lax.scan(fun, u, xs=None, length=n_steps)
    return u


def spectral_norm_conv(
    unfrw: Array, xshape: Shape, seed: float = 0, maxit: int = 10, eps: float = 1e-12
):
    """Estimate spectral norm of convolution operator.

    This function estimates the spectral norm of a convolution operator
    by estimating the singular vectors of the operator via the
    power iteration method.

    Args:
        unfrw: Unfrozen parameters of convolutional layer (i.e. convolution filter).
        xshape: Shape of input to convolution operator.
        seed: Value to seed the random generation. Default: 0.
        maxit: Number of power iterations to compute. Default: 10.
        eps: Small value to prevent divide by zero. Default: 1e-12.

    Returns:
        Spectral norm.
    """

    ishape = (1, xshape[1], xshape[2], unfrw.shape[2])
    rng = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(rng, ishape)
    f = partial(conv, kernel=unfrw)
    u = _power_iteration(f, u0, maxit)
    sigma = _l2_norm(f(u))
    return sigma


def conv(inputs: Array, kernel: Array) -> Array:
    """Compute convolution betwen input
       and kernel.

    Args:
        inputs: Array to compute convolution.
        kernel: Filter of the convolutional operator.

    Returns:
        Singular vectors.
    """
    strides = (1, 1)
    padding = "SAME"
    input_dilation = (1, 1)
    kernel_dilation = (1, 1)
    feature_group_count = 1
    dtype = jnp.float32

    inputs = jnp.asarray(inputs, dtype)

    kernel = jnp.asarray(kernel, dtype)

    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    y = lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
    )

    return y


def spectral_normalization(params: PyTree, traversal: ModelParamTraversal, xshape: Shape) -> PyTree:
    """Normalize parameters of convolutional layer by its spectral norm.

    Args:
        params: Current model parameters.
        traversal: Utility to select model parameters.
        xshape: Shape of input.
    """
    params_out = traversal.update(
        lambda x: x / (spectral_norm_conv(x, xshape) * 1.02), unfreeze(params)
    )

    return freeze(params_out)
