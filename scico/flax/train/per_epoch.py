#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flax functionality needed to run per epoch while training.
"""

from typing import Any, Callable, Iterable, Optional

import jax
import jax.numpy as jnp
from jax import lax

import flax
from flax.core import freeze, unfreeze
from flax.linen.linear import _conv_dimension_numbers


Shape = Iterable[int]
Dtype = Any
ModuleDef = Any
Array = Any

PRNGKey = jax.random.PRNGKey(0)


def compute_spectral_normalization(params, xshp):
    # Unfreeze params to normal dict.
    prms = unfreeze(params)
    # Get flattened-key: value list.
    flat_prms = {'/'.join(k): v for k, v in flax.traverse_util.flatten_dict(prms).items()}
    # Normalize kernel by spectral norm
    for k in flat_prms.keys():
        k_tstr = k.split('/')
        prm_str = k_tstr[-1]
        if prm_str == 'kernel':
            sgm_v = spectral_norm_conv2d(flat_prms[k], xshp)
            flat_prms[k] /= (sgm_v * 1.02)
    # Unflatten.
    unflat_prms = flax.traverse_util.unflatten_dict({tuple(k.split('/')): v for k, v in flat_prms.items()})
    # Refreeze.
    unflat_prms = freeze(unflat_prms)

    return unflat_prms


def _l2_normalize(x, eps=1e-12):
    return x * lax.rsqrt((x ** 2).sum() + eps)


def _l2_norm(x):
    return jnp.sqrt((x ** 2).sum())


def conv(inputs, kernel):
    strides = (1, 1)
    padding = 'SAME'
    input_dilation = (1, 1)
    kernel_dilation = (1, 1)
    feature_group_count = 1
    dtype = jnp.float32

    inputs = jnp.asarray(inputs, dtype)
    # single input
    inputs = jnp.expand_dims(inputs, axis=0)

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
        feature_group_count=feature_group_count,)

    # single input
    y = jnp.squeeze(y, axis=0)
    return y


def spectral_norm_conv2d(unfrw, xshp, power_iterations:int=1, eps:float=1e-12):


    xshp_in = xshp[:-1] + (unfrw.shape[2],)
    xshp_out = xshp[:-1] + (unfrw.shape[-1],)
    global PRNGKey
    new_key, subkey = jax.random.split(PRNGKey)
    u0 = jax.random.truncated_normal(subkey, lower=0., upper=1., shape=xshp_out)
    wT = jnp.transpose(unfrw, (0, 1, 3, 2))
    u, v = _power_iteration_conv2d(u0, xshp_in, unfrw, wT, power_iterations, eps)
    sigma = jnp.sum(u * conv(v, unfrw))
    sigma = jnp.clip(sigma, a_min=eps)

    PRNGKey = new_key

    return sigma


def _power_iteration_conv2d(u, xshape, w, wT, n_steps=10, eps=1e-12):
    """Update an estimate of the first right-singular vector and
       first left-singular vector of conv."""
    def fun(carry, _):
        u, _ = carry
        # Compute conv transposed
        # check flip
        # remove conv application as well as conv parameters
        #v = conv(jnp.flip(u, (2, 3)), wT)
        v = conv(jnp.flip(u, (0, 1)), wT)
        v = _l2_normalize(v, eps)
        #v = jnp.flip(v, (2, 3))
        v = jnp.flip(v, (0, 1))
        u = conv(v, w)
        u = _l2_normalize(u, eps)
        return (u,v), None

    carry = (u, jnp.zeros(xshape))
    carry, _ = lax.scan(fun, carry, xs=None, length=n_steps)
    return carry[0], carry[1]
