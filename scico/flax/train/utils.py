#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utils for spectral normalization of convolutional layers in Flax models.
"""
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
from flax.traverse_util import _get_params_dict, flatten_dict, unflatten_dict
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


class CNN(Module):
    """Evaluation of convolution operator via Flax implementation of a convolutional layer.

    This is form of convolution is used only for the
    estimation of the spectral norm of the operator.
    Therefore, the value of the kernel is provided too.

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
            x: The nd-array to be convolved.

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
    """Compute convolution betwen input
       and kernel.

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

    rng = jax.random.PRNGKey(0)  # not used
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


# From https://flax.readthedocs.io/en/latest/_modules/flax/traverse_util.html#Traversal
# This is marked as deprecated in Flax since v0.4.1. Copied here to keep functionality
class ModelParamTraversal:
    """Select model parameters using a name filter.

    This traversal operates on a nested dictionary of parameters and selects a
    subset based on the `filter_fn` argument.

    """

    def __init__(self, filter_fn):
        """Constructor a new ModelParamTraversal.

        Args:
          filter_fn: a function that takes a parameter's full name and its value and
            returns whether this parameter should be selected or not. The name of a
            parameter is determined by the module hierarchy and the parameter name
            (for example: '/module/sub_module/parameter_name').
        """
        self._filter_fn = filter_fn

    def iterate(self, inputs):
        """Iterate over the values selected by this `Traversal`.

        Args:
            inputs: the object that should be traversed.
        Returns:
            An iterator over the traversed values.
        """

        params = _get_params_dict(inputs)
        flat_dict = flatten_dict(params)
        for key, value in _sorted_items(flat_dict):
            path = "/" + "/".join(key)
            if self._filter_fn(path, value):
                yield value

    def update(self, fn, inputs):
        """Update the focused items.

        Args:
            fn: the callback function that maps each traversed item to its updated value.
            inputs: the object that should be traversed.
        Returns:
            A new object with the updated values.
        """
        params = _get_params_dict(inputs)
        flat_dict = flatten_dict(params, keep_empty_nodes=True)
        new_dict = {}
        for key, value in _sorted_items(flat_dict):
            # empty_node is not an actual leave. It's just a stub for empty nodes
            # in the nested dict.
            if value is not empty_node:
                path = "/" + "/".join(key)
                if self._filter_fn(path, value):
                    value = fn(value)
            new_dict[key] = value
        new_params = unflatten_dict(new_dict)
        if isinstance(inputs, flax.core.FrozenDict):
            return flax.core.FrozenDict(new_params)
        else:
            return new_params
