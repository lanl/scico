"""
Utilities for wrapping jnp functions to handle BlockArray inputs.
"""

import sys
from functools import wraps
from inspect import signature
from types import ModuleType
from typing import Callable, Iterable, Optional

import jax.numpy as jnp

from scico.array import is_nested

from .blockarray import BlockArray


def wrap_attributes(
    to_dict: dict,
    from_dict: dict,
    modules_to_recurse: Optional[Iterable[str]] = None,
    reductions: Optional[Iterable[str]] = None,
):
    """Add attributes in `from_dict` to `to_dict`.

    Underscore attributes are ignored. Functions are wrapped to allow for
    `BlockArray` inputs. Modules are ignored, except those listed in
    `modules_to_recurse`, which are added recursively. All others are
    passed through unwrapped.

    """

    if modules_to_recurse is None:
        modules_to_recurse = ()

    if reductions is None:
        reductions = ()

    for name, obj in from_dict.items():
        if name[0] == "_":
            continue
        if isinstance(obj, ModuleType) and name in modules_to_recurse:
            to_dict[name] = ModuleType(name)
            to_dict[name].__package__ = to_dict["__name__"]
            to_dict[name].__doc__ = obj.__doc__
            # enable `import scico.numpy.linalg` and `from scico.numpy.linalg import norm`
            sys.modules[to_dict["__name__"] + "." + name] = to_dict[name]
            wrap_attributes(to_dict[name].__dict__, obj.__dict__)
        elif isinstance(obj, Callable):
            obj = map_func_over_ba(obj, is_reduction=name in reductions)
            to_dict[name] = obj
        else:
            to_dict[name] = obj


def map_func_over_ba(func, is_reduction=False):
    """Create a version of `func` that maps over all of its `BlockArray`
    arguments.

    is_reduction: function is handled in a special way in order to allow
    full reductions of `BlockArray`s.  If the axis parameter exists but
    is not specified, the function is mapped over the blocks, then
    called again on the stacked result.
    """

    @wraps(func)
    def mapped(*args, **kwargs):
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)

        ba_args = {}
        for k, v in list(bound_args.arguments.items()):
            if isinstance(v, BlockArray) or is_nested(v):
                ba_args[k] = bound_args.arguments.pop(k)

        if not len(ba_args):  # no BlockArray arguments
            return func(*args, **kwargs)

        result = tuple(
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

        if is_reduction and "axis" not in bound_args.arguments:
            if len(ba_args) > 1:
                raise ValueError(
                    "Cannot perform a full reduction with multiple BlockArray arguments."
                )
            return func(
                *bound_args.args,
                **bound_args.kwargs,
                **{list(ba_args.keys())[0]: jnp.stack(result)},
            )

        if isinstance(result[0], jnp.ndarray):  # True for abstract arrays, too
            return BlockArray(result)

        return result

    return mapped
