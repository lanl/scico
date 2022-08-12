#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utils for spectral normalization of convolutional layers in Flax models.
"""

from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import scipy
from flax.core import freeze, unfreeze
from flax.linen.linear import _conv_dimension_numbers
from flax.traverse_util import ModelParamTraversal
from scico.typing import Array, Shape

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

    This function estimates the spectral norm of an operator
    by estimating the singular vectors of the operator via the
    power iteration method and the transpose operator enabled by nested autodiff in JAX.

    Args:
        f: Operator to compute spectral norm.
        input_shape: Shape of input to operator.
        seed: Value to seed the random generation. Default: 0.
        n_steps: Number of power iterations to compute. Default: 10.
        eps: Small value to prevent divide by zero. Default: 1e-12.

    Returns:
        Spectral norm.
    """
    rng = jax.random.PRNGKey(seed)
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


def spectral_normalization_conv(
    params: PyTree, traversal: ModelParamTraversal, xshape: Shape
) -> PyTree:
    """Normalize parameters of convolutional layer by its spectral norm.

    Args:
        params: Current model parameters.
        traversal: Utility to select model parameters.
        xshape: Shape of input.
    """
    params_out = traversal.update(
        lambda kernel: kernel
        / (
            estimate_spectral_norm(
                lambda x: conv(x, kernel), (1, xshape[1], xshape[2], kernel.shape[2])
            )
            * 1.02
        ),
        unfreeze(params),
    )

    return freeze(params_out)


# From https://nbviewer.org/gist/shoyer/fa9a29fd0880e2e033d7696585978bfc
def exact_spectral_norm(f, input_shape):
    """Compute spectral norm of operator.

    This function computes the spectral norm of an operator via autodiff in JAX.

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
