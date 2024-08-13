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
    clr_func = colorama.Fore.RED
    clr_args = colorama.Fore.LIGHTBLUE_EX
    clr_retv = colorama.Fore.LIGHTBLUE_EX
    clr_reset = colorama.Fore.RESET
else:
    clr_main, clr_array, clr_retv, clr_reset = "", "", "", ""


def _get_hash(val: Any) -> Optional[int]:
    """Get a hash representing an object.

    Args:
        val: An object for which the hash is required.

    Returns:
        A hash value of ``None`` if a hash cannot be computed.
    """
    if isinstance(val, np.ndarray):
        hash = val.ctypes.data  # for an ndarray, hash is the memory address
    elif hasattr(val, "__hash__") and callable(val.__hash__):
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
    elif np.isscalar(val):  # a scalar value
        return str(val)
    elif isinstance(val, tuple) and len(val) < 6 and all([np.isscalar(s) for s in val]):
        return f"{val}"  # a short sequence of scalars
    elif isinstance(val, np.dtype):  # a numpy dtype
        return f"numpy.{val}"
    elif isinstance(val, type):  # a class name
        return f"{val.__module__}.{val.__qualname__}"
    elif isinstance(val, np.ndarray) and _get_hash(val) in call_trace.instance_hash:
        return f"{call_trace.instance_hash[_get_hash(val)]}"
    elif isinstance(val, (np.ndarray, jax.Array)):  # a jax or numpy array
        if val.shape == ():
            return str(val)
        else:
            return f"Array{val.shape}"
    else:
        if _get_hash(val) in call_trace.instance_hash:
            return f"{call_trace.instance_hash[val.__hash__()]}"
        else:
            return f"[{type(val).__name__}]"


def register_variable(var: Any, name: str):
    """Register a variable name for call tracing.

    Any hashable object (or numpy arrays, with the memory address
    used as a hash) may be registered. JAX arrays may not be registered
    since they are not hashable and there is no clear mechanism for
    associating them with a unique memory address.

    Args:
        var: The variable to be registered.
        name: The name to be associated with the variable.
    """
    hash = _get_hash(var)
    if hash is None:
        raise ValueError(f"Can't get hash for variable {name}.")
    call_trace.instance_hash[hash] = name


def _call_wrapped_function(func: Callable, *args, **kwargs) -> Any:
    """Call a wrapped function within the wrapper.

    Handle different call mechanisms required for static and class
    methods.

    Args:
        func: Wrapped function
        *args: Positional arguments
        **kwargs: Named arguments

    Returns:
        Return value of wrapped function.
    """
    if isinstance(func, staticmethod):
        ret = func(*args[1:], **kwargs)
    elif isinstance(func, classmethod):
        ret = func.__func__(*args, **kwargs)
    else:
        ret = func(*args, **kwargs)
    return ret


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
        arg_idx = 0
        if (
            args
            and hasattr(args[0], "__hash__")
            and callable(args[0].__hash__)
            and method_class
            and isinstance(args[0], method_class)
        ):  # first argument is self for a method call
            arg_idx = 1  # skip self in handling arguments
            if args[0].__hash__() in call_trace.instance_hash:
                # self object registered using register_variable
                name = f"{call_trace.instance_hash[args[0].__hash__()]}.{clr_func}{func.__name__}"
            elif hasattr(args[0], "__class__"):
                # func is being called as an inherited method of a derived class
                name = (
                    f"{args[0].__class__.__module__}.{args[0].__class__.__name__}."
                    f"{clr_func}{func.__name__}"
                )
        args_repr = [_trace_arg_repr(val) for val in args[arg_idx:]]
        kwargs_repr = [f"{key}={_trace_arg_repr(val)}" for key, val in kwargs.items()]
        args_str = clr_args + ", ".join(args_repr + kwargs_repr) + clr_main
        print(
            f"{clr_main}>> {' ' * 2 * call_trace.trace_level}{name}"
            f"({args_str}{clr_func}){clr_reset}",
            file=sys.stderr,
        )
        # call wrapped function
        call_trace.trace_level += 1
        ret = _call_wrapped_function(func, *args, **kwargs)
        call_trace.trace_level -= 1
        # print representation of return value
        if ret is not None and call_trace.show_return_value:
            print(
                f"{clr_main}>> {' ' * 2 * call_trace.trace_level}{clr_retv}"
                f"{_trace_arg_repr(ret)}{clr_reset}",
                file=sys.stderr,
            )
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
# flag indicating whether to show function return value
call_trace.show_return_value = True


def _submodule_name(module, obj):
    if (
        len(obj.__name__) > len(module.__name__)
        and obj.__name__[0 : len(module.__name__)] == module.__name__
    ):
        short_name = obj.__name__[len(module.__name__) + 1 :]
    else:
        short_name = ""
    return short_name


def OLD_apply_decorator(
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
                    # can't use plain getattr here since it interferes with identification of
                    # static methods
                    attr = inspect.getattr_static(obj, attr_name)
                    if isinstance(attr, (types.FunctionType, PjitFunction)):
                        qualname = attr.__module__ + "." + attr.__qualname__  # type: ignore
                        if not seen[qualname]:  # avoid multiple applications of decorator
                            setattr(obj, attr_name, decorator(attr))
                        seen[qualname] += 1
                        if verbose:
                            print(f"{indent + '    '}Method: {qualname}")
        elif isinstance(obj, types.ModuleType):
            short_name = _submodule_name(module, obj)
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



def _is_scico_object(obj):
    return hasattr(obj, "__module__") and obj.__module__[0:5] == "scico"


def _is_scico_module(mod):
    return hasattr(mod, "__name__") and mod.__name__[0:5] == "scico"

def _is_submodule(mod, submod):
    return submod.__name__[0 : len(mod.__name__)] == mod.__name__


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

    # Iterate over functions in module
    for name, func in inspect.getmembers(
        module,
        lambda obj: isinstance(obj, (types.FunctionType, PjitFunction)) and _is_scico_object(obj),
    ):
        if name in skip:
            continue
        qualname = func.__module__ + "." + func.__qualname__
        if not seen[qualname]:  # avoid multiple applications of decorator
            setattr(module, name, decorator(func))
            seen[qualname] += 1
            if verbose:
                print(f"{indent}Function: {qualname}")

    # Iterate over classes in module
    for name, cls in inspect.getmembers(
        module, lambda obj: inspect.isclass(obj) and _is_scico_object(obj)
    ):
        qualname = cls.__module__ + "." + cls.__qualname__  # type: ignore
        if verbose:
            print(f"{indent}Class: {qualname}")

        # Iterate over methods in class
        for name, func in inspect.getmembers(
            cls,
            lambda obj: isinstance(obj, (types.FunctionType, PjitFunction))
        ):
            if name in skip:
                continue
            qualname = func.__module__ + "." + func.__qualname__  # type: ignore
            if not seen[qualname]:  # avoid multiple applications of decorator
                # Can't use cls returned by inspect.getmembers because it uses plain
                # getattr internally, which interferes with identification of static
                # methods. From Python 3.11 onwards one could use
                # inspect.getmembers_static instead of inspect.getmembers, but that
                # would imply incompatibility with earlier Python versions.
                func = inspect.getattr_static(cls, name)
                setattr(cls, name, decorator(func))
                seen[qualname] += 1
                if verbose:
                    print(f"{indent + '    '}Method: {qualname}")

    # Iterate over submodules of module
    if recursive:
        for name, mod in inspect.getmembers(
                module, lambda obj: inspect.ismodule(obj) and _is_submodule(module, obj)
        ):
            if name[0:1] == "_":
                continue
            qualname = mod.__name__
            if verbose:
                qualname = mod.__name__
                print(f"{indent}Module: {qualname}")
            seen = apply_decorator(
                mod,
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

    seen = None
    for module in (functional, linop, loss, operator, optimize, function, metric, solver):
        seen = apply_decorator(module, call_trace, skip=["__repr__"], seen=seen, verbose=True)
    # unclear why applying this separately is required
    optimize.Optimizer.solve = call_trace(optimize.Optimizer.solve)
