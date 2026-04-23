# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.


"""Utilities for wrapping jnp functions to handle BlockArray inputs."""

import sys
import warnings
from functools import wraps
from inspect import Parameter, signature
from types import ModuleType
from typing import Callable, Iterable, Optional

import jax.numpy as jnp

import scico.numpy as snp

from ._blockarray import TransparentTuple


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


def map_func_over_args(
    func: Callable,
    map_if_nested_args: Optional[list[str]] = [],
    map_if_list_args: Optional[list[str]] = [],
    is_void: Optional[bool] = False,
):
    """
    Wrap a function so that it automatically maps over its arguments,
    returning a BlockArray.

    BlockArray arguments always trigger mapping. Other arguments trigger
    mapping if they meet specified criteria.
    """
    # check inputs
    func_signature = signature(func)
    for arg in map_if_nested_args + map_if_list_args:
        if arg not in func_signature.parameters:
            raise ValueError(f"`{arg}` is not an argument of {func.__name__}")

    # define wrapped function
    @wraps(func)
    def wrapped(*args, **kwargs):
        arg_names = [
            k
            for k, v in func_signature.parameters.items()
            if v.kind
            in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        # look in args for mapping triggers
        arg_is_mapping = []
        for arg_num, arg_val in enumerate(args):
            if (
                isinstance(arg_val, TransparentTuple)
                or (
                    snp.util.is_nested(arg_val)
                    and arg_num < len(arg_names)
                    and arg_names[arg_num] in map_if_nested_args
                )
                or (
                    isinstance(arg_val, (list, tuple))
                    and arg_num < len(arg_names)
                    and arg_names[arg_num] in map_if_list_args
                )
            ):
                arg_is_mapping.append(True)
            else:
                arg_is_mapping.append(False)

        # look in kwargs for mapping triggers
        kwarg_is_mapping = {}
        for arg_name, arg_val in kwargs.items():
            if (
                isinstance(arg_val, TransparentTuple)
                or (arg_name in map_if_nested_args and snp.util.is_nested(arg_val))
                or (arg_name in map_if_list_args and isinstance(arg_val, (list, tuple)))
            ):
                kwarg_is_mapping[arg_name] = True
            else:
                kwarg_is_mapping[arg_name] = False

        # no arguments that trigger mapping? call as usual
        if sum(arg_is_mapping) == 0 and sum(kwarg_is_mapping.values()) == 0:
            return func(*args, **kwargs)

        # count number of blocks
        num_blocks = (
            len(
                args[
                    [index for index, mapping_flag in enumerate(arg_is_mapping) if mapping_flag][0]
                ]
            )  # first mapping arg
            if sum(arg_is_mapping)
            else len(
                kwargs[[k for k, mapping_flag in kwarg_is_mapping.items() if mapping_flag][0]]
            )  # first mapping kwarg
        )

        # map func over the mapping args
        results = []
        for block_ind in range(num_blocks):
            result = func(
                *[
                    arg[block_ind] if is_mapping else arg
                    for arg, is_mapping in zip(args, arg_is_mapping)
                ],
                **{
                    k: kwargs[k][block_ind] if is_mapping else kwargs[k]
                    for k, is_mapping in kwarg_is_mapping.items()
                },
            )
            results.append(result)
        if is_void:
            return

        return TransparentTuple(results)

    return wrapped


def add_full_reduction(func: Callable, axis_arg_name: Optional[str] = "axis"):
    """Wrap a function so that it can fully reduce a BlockArray.

    Wrap a function so that it can fully reduce a :class:`.BlockArray`. If
    nothing is passed for the axis argument and the function is called
    on a :class:`.BlockArray`, it is fully ravelled before the function is
    called.

    Should be outside :func:`map_func_over_args`.
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
            if isinstance(v, TransparentTuple):
                ba_args[k] = bound_args.arguments.pop(k)

        if "axis" in bound_args.arguments:
            return func(*bound_args.args, **bound_args.kwargs, **ba_args)  # call func as normal

        if len(ba_args) > 1:
            raise ValueError("Cannot perform a full reduction with multiple BlockArray arguments.")

        # fully ravel the ba argument
        ba_args = {k: jnp.concatenate(v.ravel()) for k, v in ba_args.items()}
        return func(*bound_args.args, **bound_args.kwargs, **ba_args)

    return wrapped
