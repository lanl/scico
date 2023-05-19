# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.


"""Utilities for wrapping jnp functions to handle BlockArray inputs."""

import sys
import warnings
from functools import wraps
from inspect import signature
from types import ModuleType
from typing import Callable, Iterable, Optional

import jax.numpy as jnp

from ._blockarray import BlockArray


def add_attributes(
    to_dict: dict,
    from_dict: dict,
    modules_to_recurse: Optional[Iterable[str]] = None,
):
    """Add attributes in `from_dict` to `to_dict`.

    Underscore attributes are ignored. Modules are ignored, except those
    listed in `modules_to_recurse`, which are added recursively. All
    others are added.
    """

    if modules_to_recurse is None:
        modules_to_recurse = ()

    for name, obj in from_dict.items():
        if name[0] == "_":
            continue
        if isinstance(obj, ModuleType):
            if name in modules_to_recurse:
                qualname = to_dict["__name__"] + "." + name
                to_dict[name] = ModuleType(name, doc=obj.__doc__)
                to_dict[name].__package__ = to_dict["__name__"]
                # enable `import scico.numpy.linalg` and `from scico.numpy.linalg import norm`
                sys.modules[qualname] = to_dict[name]
                sys.modules[qualname].__name__ = qualname
                add_attributes(to_dict[name].__dict__, obj.__dict__)
        else:
            to_dict[name] = obj


def wrap_recursively(
    target_dict: dict,
    names: Iterable[str],
    wrap: Callable,
):
    """Call wrap functions in `target_dict`, correctly handling names like `"linalg.norm"`."""

    for name in names:
        if "." in name:
            module, rest = name.split(".", maxsplit=1)
            wrap_recursively(target_dict[module].__dict__, [rest], wrap)
        else:
            if name in target_dict:
                target_dict[name] = wrap(target_dict[name])
            else:
                warnings.warn(f"In call to wrap_recursively, name {name} is not in target_dict")


def map_func_over_tuple_of_tuples(func: Callable, map_arg_name: Optional[str] = "shape"):
    """Wrap a function so that it automatically maps over a tuple of tuples
    argument, returning a BlockArray.
    """

    @wraps(func)
    def mapped(*args, **kwargs):
        bound_args = signature(func).bind(*args, **kwargs)

        if map_arg_name not in bound_args.arguments:  # no shape arg
            return func(*args, **kwargs)  # no mapping

        map_arg_val = bound_args.arguments.pop(map_arg_name)

        if not isinstance(map_arg_val, tuple) or not all(
            isinstance(x, tuple) for x in map_arg_val
        ):  # not nested tuple
            return func(*args, **kwargs)  # no mapping

        # map
        return BlockArray(
            func(*bound_args.args, **bound_args.kwargs, **{map_arg_name: x}) for x in map_arg_val
        )

    return mapped


def map_func_over_blocks(func):
    """Wrap a function so that it maps over all of its BlockArray
    arguments.
    """

    @wraps(func)
    def mapped(*args, **kwargs):

        first_ba_arg = next((arg for arg in args if isinstance(arg, BlockArray)), None)
        if first_ba_arg is None:
            first_ba_kwarg = next((v for k, v in kwargs.items() if isinstance(v, BlockArray)), None)
            if first_ba_kwarg is None:
                return func(*args, **kwargs)  # no BlockArray arguments, so no mapping
            num_blocks = len(first_ba_kwarg)
        else:
            num_blocks = len(first_ba_arg)

        # build a list of new args and kwargs, one for each block
        new_args_list = []
        new_kwargs_list = []
        for i in range(num_blocks):
            new_args_list.append([arg[i] if isinstance(arg, BlockArray) else arg for arg in args])
            new_kwargs_list.append(
                {k: (v[i] if isinstance(v, BlockArray) else v) for k, v in kwargs.items()}
            )

        # run the function num_blocks times, return results in a BlockArray
        return BlockArray(func(*new_args_list[i], **new_kwargs_list[i]) for i in range(num_blocks))

    return mapped


def add_full_reduction(func: Callable, axis_arg_name: Optional[str] = "axis"):
    """Wrap a function so that it can fully reduce a BlockArray.

    Wrap a function so that it can fully reduce a :class:`.BlockArray`. If
    nothing is passed for the axis argument and the function is called
    on a :class:`.BlockArray`, it is fully ravelled before the function is
    called.

    Should be outside :func:`map_func_over_blocks`.
    """
    sig = signature(func)
    if axis_arg_name not in sig.parameters:
        raise ValueError(
            f"Cannot wrap {func} as a reduction because it has no {axis_arg_name} argument."
        )

    @wraps(func)
    def wrapped(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)

        ba_args = {}
        for k, v in list(bound_args.arguments.items()):
            if isinstance(v, BlockArray):
                ba_args[k] = bound_args.arguments.pop(k)

        if "axis" in bound_args.arguments:
            return func(*bound_args.args, **bound_args.kwargs, **ba_args)  # call func as normal

        if len(ba_args) > 1:
            raise ValueError("Cannot perform a full reduction with multiple BlockArray arguments.")

        # fully ravel the ba argument
        ba_args = {k: jnp.concatenate(v.ravel()) for k, v in ba_args.items()}
        return func(*bound_args.args, **bound_args.kwargs, **ba_args)

    return wrapped
