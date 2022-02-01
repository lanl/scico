# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Random number generation.

This module provides convenient wrappers around several `jax.random
<https://jax.readthedocs.io/en/stable/jax.random.html>`_ routines to
handle the generation and splitting of PRNG keys, as well as the
generation of random :class:`.BlockArray`:

::

   # Calls to scico.random functions always return a PRNG key
   # If no key is passed to the function, a new key is generated
   x, key = scico.random.randn((2,))
   print(x)   # [ 0.19307713 -0.52678305]

   # scico.random functions automatically split the PRNGkey and return
   # an updated key
   y, key = scico.random.randn((2,), key=key)
   print(y) # [ 0.00870693 -0.04888531]

The user is responsible for passing the PRNG key to :mod:`scico.random`
functions. If no key is passed, repeated calls to :mod:`scico.random`
functions will return the same random numbers:

::

   x, key = scico.random.randn((2,))
   print(x)   # [ 0.19307713 -0.52678305]

   # No key passed, will return the same random numbers!
   y, key = scico.random.randn((2,))
   print(y)   # [ 0.19307713 -0.52678305]


If the desired shape is a tuple containing tuples, a :class:`.BlockArray`
is returned:

::

   x, key = scico.random.randn( ((1, 1), (2,)), key=key)
   print(x)  # scico.blockarray.BlockArray:
             # DeviceArray([ 1.1378784 , -1.220955  , -0.59153646], dtype=float32)

"""


import functools
import inspect
import sys
from typing import Optional, Tuple, Union

import numpy as np

import jax

from scico.array import is_nested
from scico.blockarray import BlockArray, block_sizes
from scico.typing import BlockShape, DType, JaxArray, PRNGKey, Shape


def _add_seed(fun):
    """
    Modify a :mod:`jax.random` function to add a `seed` argument.

    Args:
        fun: function to be modified, e.g., :func:`jax.random.normal`.
        Expects `key` to be the first argument.

    Returns:
        fun_alt: a version of `fun` supporting an optional `seed`
           argument that is used to create a :func:`jax.random.PRNGKey`
           that is passed along as the `key`. The `key` argument may
           still be used, but is moved to be second-to-last. By default,
           `seed=0`. The `seed` argument is added last. Other arguments
           are unchanged.
    """

    # find number of arguments to fun
    num_params = len(inspect.signature(fun).parameters)

    def fun_alt(*args, key=None, seed=None, **kwargs):

        # key and seed may be in *args, look for them
        if len(args) >= num_params:  # they passed all position args including key
            key = args[num_params - 1]
        if len(args) > num_params:  # they passed all position args including key and seed
            seed = args[num_params]

        if key is not None and seed is not None:
            raise ValueError("Key and seed cannot both be specified")

        if key is None:
            if seed is None:
                seed = 0
            key = jax.random.PRNGKey(seed)

        result = fun(key, *args[: num_params - 1], **kwargs)

        key, subkey = jax.random.split(key, 2)
        return result, key

    lines = fun.__doc__.split("\n\n")
    fun_alt.__doc__ = "\n\n".join(
        lines[0:1]
        + [
            f"  Wrapped version of `jax.random.{fun.__name__} <https://jax.readthedocs.io/en/stable/jax.random.html#jax.random.{fun.__name__}>`_. "
            "The SCICO version of this function moves the `key` argument to the end of the argument list, "
            "adds an additional `seed` argument after that, and allows the `shape` argument "
            "to accept a nested list, in which case a `BlockArray` is returned. "
            "Always returns a `(result, key)` tuple.",
            "  Original docstring below.",
        ]
        + lines[1:]
    )

    return fun_alt


def _allow_block_shape(fun):
    """
    Decorate a jax.random function so that the `shape` argument may be a BlockShape.
    """

    # use inspect to find which argument number is `shape`
    shape_ind = list(inspect.signature(fun).parameters.keys()).index("shape")

    @functools.wraps(fun)
    def fun_alt(*args, **kwargs):

        # get the shape argument if it was passed
        if len(args) > shape_ind:
            shape = args[shape_ind]
        elif "shape" in kwargs:
            shape = kwargs["shape"]
        else:  # shape was not passed, call fun as normal
            return fun(*args, **kwargs)

        # if shape is not nested, call fun as normal
        if not is_nested(shape):
            return fun(*args, **kwargs)
        # shape is nested, so make a BlockArray!

        # call the wrapped fun with an shape=(size,)
        subargs = list(args)
        subkwargs = kwargs.copy()
        size = np.sum(block_sizes(shape))

        if len(subargs) > shape_ind:
            subargs[shape_ind] = (size,)
        else:  # shape must be a kwarg if not a positional arg
            subkwargs["shape"] = (size,)

        result_flat = fun(*subargs, **subkwargs)

        return BlockArray.array_from_flattened(result_flat, shape)

    return fun_alt


def _wrap(fun):
    fun_wrapped = _add_seed(_allow_block_shape(fun))
    fun_wrapped.__module__ = __name__  # so it appears in docs
    return fun_wrapped


def _is_wrappable(fun):
    params = inspect.signature(getattr(jax.random, fun)).parameters
    prmkey = list(params.keys())
    return prmkey and (prmkey[0] == "key") and ("shape" in params.keys())


wrappable_func_names = [
    t[0] for t in inspect.getmembers(jax.random, inspect.isfunction) if _is_wrappable(t[0])
]

for name in wrappable_func_names:
    setattr(sys.modules[__name__], name, _wrap(getattr(jax.random, name)))


def randn(
    shape: Union[Shape, BlockShape],
    dtype: DType = np.float32,
    key: Optional[PRNGKey] = None,
    seed: Optional[int] = None,
) -> Tuple[Union[JaxArray, BlockArray], PRNGKey]:
    """Return an array drawn from the standard normal distribution.

    Alias for :func:`scico.random.normal`.

    Args:
        shape: Shape of output array. If shape is a tuple, a
            DeviceArray is returned. If shape is a tuple of tuples, a
            :class:`.BlockArray` is returned.
        key: JAX PRNGKey. Defaults to None, in which case a new key
            is created using the seed arg.
        seed: Seed for new PRNGKey. Default: 0.
        dtype: dtype for returned value. Default to ``np.float32``.
            If ``np.complex64``, generates an array sampled from complex
            normal distribution.

    Returns:
        tuple: A tuple (x, key) containing:

           - **x** : (DeviceArray):  Generated random array.
           - **key** : Updated random PRNGKey.
    """
    return normal(shape, dtype, key, seed)
