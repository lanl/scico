# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Call tracing of scico functions and methods."""


from __future__ import annotations

import inspect
import sys
import types
import warnings
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional, Sequence

import numpy as np

import jax

from jaxlib.xla_extension import PjitFunction

try:
    import colorama

    have_colorama = True
except ImportError:
    have_colorama = False


if have_colorama:
    clr_main = colorama.Fore.LIGHTRED_EX
    clr_array = colorama.Fore.LIGHTBLUE_EX
    clr_reset = colorama.Fore.RESET
else:
    clr_main, clr_array, clr_reset = "", "", ""


def _get_hash(val: Any) -> Optional[int]:
    """Get a hash representing an object.

    Args:
        val: An object for which the hash is required.

    Returns:
        A hash value of ``None`` if a hash cannot be computed.
    """
    if hasattr(val, "__hash__") and callable(val.__hash__):
        try:
            hash = val.__hash__()
        except TypeError:
            hash = None
    else:
        hash = None
    return hash


def _trace_arg_repr(val: Any) -> str:
    """Compute string representation of function arguments.

    Args:
        val: Argument value

    Returns:
        A string representation of the argument.
    """
    if val is None:
        return "None"
    elif np.isscalar(val):
        return str(val)
    elif isinstance(val, np.dtype):
        return f"numpy.{val}"
    elif isinstance(val, tuple) and len(val) < 6 and all([np.isscalar(s) for s in val]):
        return f"{val}"
    elif isinstance(val, type):
        return f"{val.__module__}.{val.__qualname__}"
    elif isinstance(val, (np.ndarray, jax.Array)):
        return f"{clr_array}Array{val.shape}{clr_main}"
    else:
        if _get_hash(val) in call_trace.instance_hash:
            return f"{call_trace.instance_hash[val.__hash__()]}"
        else:
            return f"[{type(val).__name__}]"


def register_variable(var: Any, name: str):
    """Register a variable name for call tracing.

    Args:
        var: The variable to be registered.
        name: The name to be associated with the variable.
    """
    hash = _get_hash(var)
    if hash is None:
        raise ValueError(f"Can't get hash for variable {var}.")
    call_trace.instance_hash[hash] = name


def call_trace(func: Callable) -> Callable:  # pragma: no cover
    """Print log of calls to `func`.

    Decorator for printing a log of calls to the wrapped function. A
    record of call levels is maintained so that call nesting is indicated
    by call log indentation.
    """
    try:
        method_class = inspect._findclass(func)  # type: ignore
    except AttributeError:
        method_class = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        name = f"{func.__module__}.{func.__qualname__}"
        argidx = 0
        if (
            args
            and hasattr(args[0], "__hash__")
            and callable(args[0].__hash__)
            and method_class
            and isinstance(args[0], method_class)
        ):
            argidx = 1
            if args[0].__hash__() in call_trace.instance_hash:
                name = f"{call_trace.instance_hash[args[0].__hash__()]}.{func.__name__}"
            elif hasattr(args[0], "__class__"):
                name = (
                    f"{args[0].__class__.__module__}.{args[0].__class__.__name__}.{func.__name__}"
                )
        argsrep = [_trace_arg_repr(val) for val in args[argidx:]]
        kwargrep = [f"{key}={_trace_arg_repr(val)}" for key, val in kwargs.items()]
        argstr = ", ".join(argsrep + kwargrep)
        print(
            f"{clr_main}>> {' ' * 2 * call_trace.trace_level}{name}({argstr}){clr_reset}",
            file=sys.stderr,
        )
        call_trace.trace_level += 1
        ret = func(*args, **kwargs)
        call_trace.trace_level -= 1
        return ret

    # Set flag indicating that function is already wrapped
    wrapper._call_trace_wrap = True  # type: ignore
    # Avoid multiple wrapper layers
    if hasattr(func, "_call_trace_wrap"):
        return func
    else:
        return wrapper


# call level counter for call_trace decorator
call_trace.trace_level = 0  # type: ignore
# hash dict allowing association of objects with variable names
call_trace.instance_hash = {}  # type: ignore


def apply_decorator(
    module: types.ModuleType,
    decorator: Callable,
    recursive: bool = True,
    skip: Optional[Sequence] = None,
    seen: Optional[defaultdict[str, int]] = None,
    verbose: bool = False,
    level: int = 0,
) -> defaultdict[str, int]:
    """Apply a decorator function to all functions in a scico module.

    Apply a decorator function to all functions in a scico module,
    including methods of classes in that module.

    Args:
        module: The module containing the functions/methods to be
          decorated.
        decorator: The decorator function to apply to each module
          function/method.
        recursive: Flag indicating whether to recurse into submodules
          of the specified module. (Hidden modules with a name starting
          with an underscore are ignored.)
        skip: A list of class/function/method names to be skipped.
        seen: A :class:`defaultdict` providing a count of the number of
          times each function/method was seen.
        verbose: Flag indicating whether to print a log of functions
          as they are encountered.
        level: Counter for recursive call levels.

    Returns:
        A :class:`defaultdict` providing a count of the number of times
        each function/method was seen.
    """
    indent = " " * 4 * level
    if skip is None:
        skip = []
    if seen is None:
        seen = defaultdict(int)
    # Iterate over objects in module
    for obj_name in dir(module):
        if obj_name in skip:
            continue
        obj = getattr(module, obj_name)
        if hasattr(obj, "__module__") and obj.__module__[0:5] == "scico":
            qualname = obj.__module__ + "." + obj.__qualname__
            if isinstance(obj, (types.FunctionType, PjitFunction)):
                if not seen[qualname]:  # avoid multiple applications of decorator
                    setattr(module, obj_name, decorator(obj))
                seen[qualname] += 1
                if verbose:
                    print(f"{indent}Function: {qualname}")
            elif isinstance(obj, type):
                if verbose:
                    print(f"{indent}Class: {qualname}")
                # Iterate over class attributes
                for attr_name in dir(obj):
                    if attr_name in skip:
                        continue
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, (types.FunctionType, PjitFunction)):
                        qualname = attr.__module__ + "." + attr.__qualname__  # type: ignore
                        if not seen[qualname]:  # avoid multiple applications of decorator
                            setattr(obj, attr_name, decorator(attr))
                        seen[qualname] += 1
                        if verbose:
                            print(f"{indent + '    '}Method: {qualname}")
        elif isinstance(obj, types.ModuleType):
            if (
                len(obj.__name__) > len(module.__name__)
                and obj.__name__[0 : len(module.__name__)] == module.__name__
            ):
                short_name = obj.__name__[len(module.__name__) + 1 :]
            else:
                short_name = ""
            if (
                len(obj.__name__) >= 5
                and obj.__name__[0:5] == "scico"
                and len(short_name) > 0
                and short_name[0] != "_"
            ):
                if verbose:
                    print(f"{indent}Module: {obj.__name__}")
                if recursive:
                    seen = apply_decorator(
                        obj,
                        decorator,
                        recursive=recursive,
                        skip=skip,
                        seen=seen,
                        verbose=verbose,
                        level=level + 1,
                    )
    return seen


def trace_scico_calls():  # pragma: no cover
    """Enable tracing of calls to all significant scico functions/methods.

    Enable tracing of calls to all significant scico functions and
    methods. Note that JIT should be disabled to ensure correct
    functioning of the tracing mechanism.
    """
    if not jax.config.jax_disable_jit:
        warnings.warn(
            "Call tracing requested but jit is not disabled. Disable jit"
            " by setting the environment variable JAX_DISABLE_JIT=1, or use"
            " jax.config.update('jax_disable_jit', True)."
        )
    from scico import (
        function,
        functional,
        linop,
        loss,
        metric,
        operator,
        optimize,
        solver,
    )

    for module in (functional, linop, loss, operator, optimize, function, metric, solver):
        apply_decorator(module, call_trace, skip=["__repr__"])
    # Currently unclear why applying this separately is required
    optimize.Optimizer.solve = call_trace(optimize.Optimizer.solve)
