# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

""":class:`scico.blockarray.BlockArray`-compatible
versions of :mod:`jax.numpy` functions.

This modules consists of functions from :mod:`jax.numpy` wrapped to
support compatibility with :class:`scico.blockarray.BlockArray`. This
module is a work in progress and therefore not all functions are
wrapped. Functions that have not been wrapped yet have WARNING text in
their documentation, below.
"""


from functools import wraps
from inspect import Parameter, signature
from types import FunctionType, ModuleType

import jax.numpy as jnp

from jaxlib.xla_extension import CompiledFunction

from scico.blockarray import BlockArray


def _copy_attributes(to_dict, from_dict, modules_to_recurse=None, reductions=None):
    """Add attributes in `from_dict` to `to_dict`.

    Underscore methods are ignored. Functions are wrapped to allow for
    `BlockArray` inputs. Modules are ignored, except those listed in
    `modules_to_recurse`, which are added recursively. Functions with
    names listed in `reductions` are given the `block_axis` argument,
    allowing reduction down the block axis.

    """

    if modules_to_recurse is None:
        modules_to_recurse = ()

    if reductions is None:
        reductions = ()

    for name, obj in from_dict.items():
        if name[0] == "_":
            continue
        elif isinstance(obj, ModuleType) and name in modules_to_recurse:
            to_dict[name] = ModuleType(name)
            to_dict[name].__package__ = to_dict["__name__"]
            to_dict[name].__doc__ = obj.__doc__
            _copy_attributes(to_dict[name].__dict__, obj.__dict__)
        elif isinstance(obj, (FunctionType, CompiledFunction)):
            obj = _map_func_over_ba(obj)
            to_dict[name] = obj
        else:
            to_dict[name] = obj


def _map_func_over_ba(func):
    """Create a version of `func` that maps over all of its `BlockArray`
    arguments.
    """

    @wraps(func)
    def mapped(*args, **kwargs):
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)

        ba_args = {}
        for k, v in list(bound_args.arguments.items()):
            if isinstance(v, BlockArray):
                ba_args[k] = bound_args.arguments.pop(k)

        ravel_blocks = "axis" not in bound_args.arguments and "axis" in sig.parameters

        if len(ba_args) and ravel_blocks:
            ba_args = {k: v.full_ravel() for k, v in list(ba_args.items())}
            print(ba_args)
            return func(*bound_args.args, **bound_args.kwargs, **ba_args)

        if len(ba_args):  # if any BlockArray arguments,
            return BlockArray(
                map(  # map over
                    lambda *args: (  # lambda x_1, x_2, ..., x_N
                        func(
                            *bound_args.args,
                            **bound_args.kwargs,  # ... nonBlockArray args
                            **dict(zip(ba_args.keys(), args)),
                        )  # plus dict of block args
                    ),
                    *ba_args.values(),  # map(f, ba_1, ba_2, ..., ba_N)
                )
            )

        return func(*args, **kwargs)

    return mapped


_copy_attributes(
    vars(),
    jnp.__dict__,
    modules_to_recurse=("linalg",),
    reductions=("sum",),
)
