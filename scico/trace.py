# -*- coding: utf-8 -*-
# Copyright (C) 2024-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Call tracing of scico functions and methods.

JIT must be disabled for tracing to function correctly (set environment
variable :code:`JAX_DISABLE_JIT=1`, or call
:code:`jax.config.update('jax_disable_jit', True)` before importing `jax`
or `scico`). Call :code:`trace_scico_calls` to initialize tracing, and
call :code:`register_variable` to associate a name with a variable so
that it can be referenced by name in the call trace.

The call trace is color-code as follows if
`colorama <https://github.com/tartley/colorama>`_ is installed:

- `module and class names`: light red
- `function and method names`: dark red
- `arguments and return values`: light blue
- `names of registered variables`: light yellow

When a method defined in a class is called for an object of a derived
class type, the class of that object is displayed in light magenta, in
square brackets. Function names and return values are distinguished by
initial ``>>`` and ``<<`` characters respectively.

A usage example is provided in the script :code:`trace_example.py`.
"""


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

try:
    from jaxlib.xla_extension import PjitFunction
except ImportError:
    from jaxlib._jax import PjitFunction  # jax >= 0.6.1


try:
    import colorama

    have_colorama = True
except ImportError:
    have_colorama = False


if have_colorama:
    clr_main = colorama.Fore.LIGHTRED_EX  # main trace information
    clr_rvar = colorama.Fore.LIGHTYELLOW_EX  # registered variable names
    clr_self = colorama.Fore.LIGHTMAGENTA_EX  # type of object for which method is called
    clr_func = colorama.Fore.RED  # function/method name
    clr_args = colorama.Fore.LIGHTBLUE_EX  # function/method arguments
    clr_retv = colorama.Fore.LIGHTBLUE_EX  # function/method return values
    clr_devc = colorama.Fore.CYAN  # JAX array device and sharding
    clr_reset = colorama.Fore.RESET  # reset color
else:
    clr_main, clr_rvar, clr_self, clr_func = "", "", "", ""
    clr_args, clr_retv, clr_devc, clr_reset = "", "", "", ""


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
    elif isinstance(val, np.ndarray) and _get_hash(val) in call_trace.instance_hash:  # type: ignore
        return f"{clr_rvar}{call_trace.instance_hash[_get_hash(val)]}{clr_args}"  # type: ignore
    elif isinstance(val, (np.ndarray, jax.Array)):  # a jax or numpy array
        if val.shape == ():
            return str(val)
        else:
            dev_str, shard_str = "", ""
            if isinstance(val, jax.Array) and not isinstance(
                val, jax._src.interpreters.partial_eval.JaxprTracer
            ):
                if call_trace.show_jax_device:  # type: ignore
                    platform = list(val.devices())[0].platform  # assume all of same type
                    devices = ",".join(map(str, sorted([d.id for d in val.devices()])))
                    dev_str = f"{clr_devc}{{dev={platform}({devices})}}{clr_args}"
                if call_trace.show_jax_sharding and isinstance(  # type: ignore
                    val.sharding, jax._src.sharding_impls.PositionalSharding
                ):
                    shard_str = f"{clr_devc}{{shard={val.sharding.shape}}}{clr_args}"
            return f"Array{val.shape}{dev_str}{shard_str}"
    else:
        if _get_hash(val) in call_trace.instance_hash:  # type: ignore
            return f"{clr_rvar}{call_trace.instance_hash[val.__hash__()]}{clr_args}"  # type: ignore
        else:
            return f"[{type(val).__name__}]"


def register_variable(var: Any, name: str):
    """Register a variable name for call tracing.

    Any hashable object (or numpy array, with the memory address
    used as a hash) may be registered. JAX arrays may not be registered
    since they are not hashable and there is no clear mechanism for
    associating them with a unique memory address.

    Args:
        var: The variable to be registered.
        name: The name to be associated with the variable.
    """
    hash = _get_hash(var)
    if hash is None:
        raise ValueError(f"Can't get hash for variable '{name}'.")
    call_trace.instance_hash[hash] = name  # type: ignore


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
        # If the type of the first argument is the same as the class to
        # which the static method belongs, assume that it was called as
        # <object>.<staticmethod>(<args>), which requires that the first
        # argument be stripped before calling the method. This is
        # somewhat heuristic, and may fail, but there is no obvious
        # mechanism for reliably determining how the method was called in
        # the calling scope.
        if inspect._findclass(func) == type(args[0]):  # type: ignore
            call_args = args[1:]
        else:
            call_args = args
        ret = func(*call_args, **kwargs)
    elif isinstance(func, classmethod):
        ret = func.__func__(*args, **kwargs)
    else:
        ret = func(*args, **kwargs)
    return ret


def call_trace(func: Callable) -> Callable:
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
        name = f"{func.__module__}.{clr_func}{func.__qualname__}"
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
                name = (
                    f"{clr_rvar}{call_trace.instance_hash[args[0].__hash__()]}."
                    f"{clr_func}{func.__name__}"
                )
            else:
                # self object not registered
                func_class = method_class.__name__
                self_class = args[0].__class__.__name__
                # If the class in which this method is defined is same as that
                # of the self object for which it's called, just display the
                # class name. Otherwise, display the name of the name defining
                # class followed by the name of the self object class in
                # square brackets.
                if func_class == self_class:
                    class_name = func_class
                else:
                    class_name = f"{func_class}{clr_self}[{self_class}]{clr_main}"
                name = f"{func.__module__}.{class_name}.{clr_func}{func.__name__}"
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
                f"{clr_main}<< {' ' * 2 * call_trace.trace_level}{clr_retv}"
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
call_trace.show_return_value = True  # type: ignore
# flag indicating whether to show JAX array devices
call_trace.show_jax_device = False  # type: ignore
# flag indicating whether to show JAX array sharding shape
call_trace.show_jax_sharding = False  # type: ignore


def _submodule_name(module, obj):
    if (
        len(obj.__name__) > len(module.__name__)
        and obj.__name__[0 : len(module.__name__)] == module.__name__
    ):
        short_name = obj.__name__[len(module.__name__) + 1 :]
    else:
        short_name = ""
    return short_name


def _is_scico_object(obj: Any) -> bool:
    """Determine whether an object is defined in a scico submodule.

    Args:
        obj: Object to check.

    Returns:
        A boolean value indicating whether `obj` is defined in a scico
        submodule.
    """
    return hasattr(obj, "__module__") and obj.__module__[0:5] == "scico"


def _is_scico_module(mod: types.ModuleType) -> bool:
    """Determine whether a module is a scico submodule.

    Args:
        mod: Module to check.

    Returns:
        A boolean value indicating whether `mod` is a scico submodule.
    """
    return hasattr(mod, "__name__") and mod.__name__[0:5] == "scico"


def _in_module(mod: types.ModuleType, obj: Any) -> bool:
    """Determine whether an object is defined in a module.

    Args:
        mod: Module of interest.
        obj: Object to check.

    Returns:
        A boolean value indicating whether `obj` is defined in `mod`.
    """
    return obj.__module__ == mod.__name__


def _is_submodule(mod: types.ModuleType, submod: types.ModuleType) -> bool:
    """Determine whether a module is a submodule of another module.

    Args:
        mod: Parent module of interest.
        submod: Possible submodule to check.

    Returns:
        A boolean value indicating whether `submod` is defined in `mod`.
    """
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
    if verbose:
        print(f"{indent}Module: {module.__name__}")
    indent += " " * 4

    # Iterate over functions in module
    for name, func in inspect.getmembers(
        module,
        lambda obj: isinstance(obj, (types.FunctionType, PjitFunction)) and _in_module(module, obj),
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
        module, lambda obj: inspect.isclass(obj) and _in_module(module, obj)
    ):
        qualname = cls.__module__ + "." + cls.__qualname__  # type: ignore
        if verbose:
            print(f"{indent}Class: {qualname}")

        # Iterate over methods in class
        for name, func in inspect.getmembers(
            cls, lambda obj: isinstance(obj, (types.FunctionType, PjitFunction))
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


def trace_scico_calls(verbose: bool = False):
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
        seen = apply_decorator(module, call_trace, skip=["__repr__"], seen=seen, verbose=verbose)
