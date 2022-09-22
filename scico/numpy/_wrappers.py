"""
Utilities for wrapping jnp functions to handle BlockArray inputs.
"""

import sys
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
        if isinstance(obj, ModuleType) and name in modules_to_recurse:
            to_dict[name] = ModuleType(name)
            to_dict[name].__package__ = to_dict["__name__"]
            to_dict[name].__doc__ = obj.__doc__
            # enable `import scico.numpy.linalg` and `from scico.numpy.linalg import norm`
            sys.modules[to_dict["__name__"] + "." + name] = to_dict[name]
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
            target_dict[name] = wrap(target_dict[name])


def map_func_over_tuple_of_tuples(func: Callable, map_arg_name: Optional[str] = "shape"):
    """Wrap a function so that it automatically maps over a tuple of tuples
    argument, returning a `BlockArray`.
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


def map_func_over_blocks(func, is_reduction=False):
    """Wrap a function so that it maps over all of its `BlockArray`
    arguments.

    is_reduction: function is handled in a special way in order to allow
    full reductions of `BlockArray`s.  If the axis parameter exists but
    is not specified, the function is called on a fully ravelled version
    of all `BlockArray` inputs.
    """
    sig = signature(func)

    @wraps(func)
    def mapped(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)

        ba_args = {}
        for k, v in list(bound_args.arguments.items()):
            if isinstance(v, BlockArray):
                ba_args[k] = bound_args.arguments.pop(k)

        if not ba_args:  # no BlockArray arguments
            return func(*args, **kwargs)  # no mapping

        num_blocks = len(list(ba_args.values())[0])

        return BlockArray(
            func(*bound_args.args, **bound_args.kwargs, **{k: v[i] for k, v in ba_args.items()})
            for i in range(num_blocks)
        )

    return mapped


def add_full_reduction(func: Callable, axis_arg_name: Optional[str] = "axis"):
    """Wrap a function so that it can fully reduce a `BlockArray`. If
    nothing is passed for the axis argument and the function is called
    on a `BlockArray`, it is fully ravelled before the function is
    called.

    Should be outside `map_func_over_blocks`.
    """
    sig = signature(func)
    if axis_arg_name not in sig.parameters:
        raise ValueError(
            f"Cannot wrap {func} as a reduction because it has no {axis_arg_name} argument"
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
