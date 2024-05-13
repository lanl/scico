# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""General utility functions."""


from __future__ import annotations

import inspect
import io
import socket
import sys
import types
import urllib.error as urlerror
import urllib.request as urlrequest
import warnings
from collections import defaultdict
from functools import reduce, wraps
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

import jax
from jax.interpreters.batching import BatchTracer
from jax.interpreters.partial_eval import DynamicJaxprTracer

from jaxlib.xla_extension import PjitFunction

try:
    import colorama

    have_colorama = True
except ImportError:
    have_colorama = False


def rgetattr(obj: object, name: str, default: Optional[Any] = None) -> Any:
    """Recursive version of :func:`getattr`.

    Args:
        obj: Object with the attribute to be accessed.
        name: Path to object in with components delimited by a "."
           character.
        default: Default value to be returned if the attribute does not
           exist.

    Returns:
        Attribute value of default if attribute does not exist.
    """

    try:
        return reduce(getattr, name.split("."), obj)
    except AttributeError as e:
        if default is not None:
            return default
        else:
            raise e


def rsetattr(obj: object, name: str, value: Any):
    """Recursive version of :func:`setattr`.

    Args:
        obj: Object with the attribute to be set.
        name: Path to object in with components delimited by a "."
           character.
        value: Value to which the attribute is to be set.
    """

    # See goo.gl/BVJ7MN
    path = name.split(".")
    setattr(reduce(getattr, path[:-1], obj), path[-1], value)


def partial(func: Callable, indices: Sequence, *fixargs: Any, **fixkwargs: Any) -> Callable:
    """Flexible partial function creation.

    This function is similar to :func:`functools.partial`, but allows
    fixing of arbitrary positional arguments rather than just some number
    of trailing positional arguments.

    Args:
        func: Function from which partial function is to be derived.
        indices: Tuple of indices of positional args of `func` that are
           to be fixed to the values specified in `fixargs`.
        *fixargs: Fixed values for specified positional arguments.
        **fixkwargs: Fixed values for keyword arguments.

    Returns:
       The partial function with fixed arguments.
    """

    def pfunc(*freeargs, **freekwargs):
        numargs = len(fixargs) + len(freeargs)
        args = [
            None,
        ] * numargs
        kfix = 0
        kfree = 0
        for k in range(numargs):
            if k in indices:
                args[k] = fixargs[kfix]
                kfix += 1
            else:
                args[k] = freeargs[kfree]
                kfree += 1
        kwargs = freekwargs.copy()
        kwargs.update(fixkwargs)
        return func(*args, **kwargs)

    posdoc = ""
    if indices:
        posdoc = f"positional arguments {','.join(map(str, indices))}"
    kwdoc = ""
    if fixkwargs:
        kwdoc = f"keyword arguments {','.join(fixkwargs.keys())}"
    pfunc.__doc__ = f"Partial function derived from function {func.__name__}"
    if posdoc or kwdoc:
        pfunc.__doc__ += " by fixing " + (" and ".join(filter(None, (posdoc, kwdoc))))
    return pfunc


def device_info(devid: int = 0) -> str:  # pragma: no cover
    """Get a string describing the specified device.

    Args:
        devid: ID number of device.

    Returns:
        Device description string.
    """
    numdev = jax.device_count()
    if devid >= numdev:
        raise RuntimeError(f"Requested information for device {devid} but only {numdev} present.")
    dev = jax.devices()[devid]
    if dev.platform == "cpu":
        info = "CPU"
    else:
        info = f"{dev.platform.upper()} ({dev.device_kind})"
    return info


def check_for_tracer(func: Callable) -> Callable:
    """Check if positional arguments to `func` are jax tracers.

    This is intended to be used as a decorator for functions that call
    external code from within SCICO. At present, external functions
    cannot be jit-ed or vmap/pmaped. This decorator checks for signs of
    jit/vmap/pmap and raises an appropriate exception.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if any([isinstance(x, DynamicJaxprTracer) for x in args]):
            raise TypeError(
                f"DynamicJaxprTracer found in {func.__name__};  did you jit this function?"
            )
        if any([isinstance(x, BatchTracer) for x in args]):
            raise TypeError(
                f"BatchTracer found in {func.__name__};  did you vmap/pmap this function?"
            )
        return func(*args, **kwargs)

    return wrapper


def _val_repr(val):
    if val is None:
        return "None"
    elif np.isscalar(val):
        return str(val)
    elif isinstance(val, np.dtype):
        return f"numpy.{val}"
    elif isinstance(val, tuple) and len(val) < 6:
        return f"{val}"
    elif isinstance(val, type):
        return f"{val.__module__}.{val.__qualname__}"
    elif isinstance(val, (np.ndarray, jax.Array)):
        return f"Array{val.shape}"
    else:
        if (
            hasattr(val, "__hash__")
            and callable(val.__hash__)
            and val.__hash__() in call_trace.instance_hash
        ):
            return f"{call_trace.instance_hash[val.__hash__()]}"
        else:
            return f"[{type(val).__name__}]"


def call_trace(func: Callable) -> Callable:  # pragma: no cover
    """Print log of calls to `func`.

    Decorator for printing a log of calls to the wrapped function. A
    record of call levels is maintained so that call nesting is indicated
    by call log indentation.
    """
    if have_colorama:
        pre, pst = colorama.Fore.LIGHTRED_EX, colorama.Fore.RESET
    else:
        pre, pst = "", ""

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
        argsrep = [_val_repr(val) for val in args[argidx:]]
        kwargrep = [f"{key}={_val_repr(val)}" for key, val in kwargs.items()]
        argstr = ", ".join(argsrep + kwargrep)
        print(
            f"{pre}>> {' ' * 3 * call_trace.trace_level}{name}({argstr}){pst}",
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


# Call level counter for call_trace decorator
call_trace.trace_level = 0  # type: ignore

call_trace.instance_hash = {}  # type: ignore


def apply_decorator(
    module: types.ModuleType,
    decorator: Callable,
    recursive: bool = True,
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
    if seen is None:
        seen = defaultdict(int)
    # Iterate over objects in module
    for obj_name in dir(module):
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
        apply_decorator(module, call_trace)
    # Currently unclear why applying this separately is required
    optimize.Optimizer.solve = call_trace(optimize.Optimizer.solve)


def url_get(url: str, maxtry: int = 3, timeout: int = 10) -> io.BytesIO:  # pragma: no cover
    """Get content of a file via a URL.

    Args:
        url: URL of the file to be downloaded.
        maxtry: Maximum number of download retries.
        timeout: Timeout in seconds for blocking operations.

    Returns:
        Buffered I/O stream.

    Raises:
        ValueError: If the maxtry parameter is not greater than zero.
        urllib.error.URLError: If the file cannot be downloaded.
    """

    if maxtry <= 0:
        raise ValueError("Parameter maxtry should be greater than zero.")
    for ntry in range(maxtry):
        try:
            rspns = urlrequest.urlopen(url, timeout=timeout)
            cntnt = rspns.read()
            break
        except urlerror.URLError as e:
            if not isinstance(e.reason, socket.timeout):
                raise

    return io.BytesIO(cntnt)


# Timer classes are copied from https://github.com/bwohlberg/sporco


class Timer:
    """Timer class supporting multiple independent labeled timers.

    The timer is based on the relative time returned by
    :func:`timeit.default_timer`.
    """

    def __init__(
        self,
        labels: Optional[Union[str, List[str]]] = None,
        default_label: str = "main",
        all_label: str = "all",
    ):
        """
        Args:
            labels: Label(s) of the timer(s) to be initialised to zero.
            default_label: Default timer label to be used when methods
                are called without specifying a label.
            all_label: Label string that will be used to denote all
                timer labels.
        """

        # Initialise current and accumulated time dictionaries
        self.t0: Dict[str, Optional[float]] = {}
        self.td: Dict[str, float] = {}
        # Record default label and string indicating all labels
        self.default_label = default_label
        self.all_label = all_label
        # Initialise dictionary entries for labels to be created
        # immediately
        if labels is not None:
            if not isinstance(labels, (list, tuple)):
                labels = [
                    labels,
                ]
            for lbl in labels:
                self.td[lbl] = 0.0
                self.t0[lbl] = None

    def start(self, labels: Optional[Union[str, List[str]]] = None):
        """Start specified timer(s).

        Args:
            labels: Label(s) of the timer(s) to be started. If it is
               ``None``, start the default timer with label specified by
               the `default_label` parameter of :meth:`__init__`.
        """

        # Default label is self.default_label
        if labels is None:
            labels = self.default_label
        # If label is not a list or tuple, create a singleton list
        # containing it
        if not isinstance(labels, (list, tuple)):
            labels = [
                labels,
            ]
        # Iterate over specified label(s)
        t = timer()
        for lbl in labels:
            # On first call to start for a label, set its accumulator to zero
            if lbl not in self.td:
                self.td[lbl] = 0.0
                self.t0[lbl] = None
            # Record the time at which start was called for this lbl if
            # it isn't already running
            if self.t0[lbl] is None:
                self.t0[lbl] = t

    def stop(self, labels: Optional[Union[str, List[str]]] = None):
        """Stop specified timer(s).

        Args:
            labels: Label(s) of the timer(s) to be stopped. If it is
               ``None``, stop the default timer with label specified by
               the `default_label` parameter of :meth:`__init__`. If it
               is equal to the string specified by the `all_label`
               parameter of :meth:`__init__`, stop all timers.
        """

        # Get current time
        t = timer()
        # Default label is self.default_label
        if labels is None:
            labels = self.default_label
        # All timers are affected if label is equal to self.all_label,
        # otherwise only the timer(s) specified by label
        if labels == self.all_label:
            labels = list(self.t0.keys())
        elif not isinstance(labels, (list, tuple)):
            labels = [
                labels,
            ]
        # Iterate over specified label(s)
        for lbl in labels:
            if lbl not in self.t0:
                raise KeyError(f"Unrecognized timer key {lbl}.")
            # If self.t0[lbl] is None, the corresponding timer is
            # already stopped, so no action is required
            if self.t0[lbl] is not None:
                # Increment time accumulator from the elapsed time
                # since most recent start call
                self.td[lbl] += t - self.t0[lbl]  # type: ignore
                # Set start time to None to indicate timer is not running
                self.t0[lbl] = None

    def reset(self, labels: Optional[Union[str, List[str]]] = None):
        """Reset specified timer(s).

        Args:
            labels: Label(s) of the timer(s) to be stopped. If it is
                ``None``, stop the default timer with label specified by
                the `default_label` parameter of :meth:`__init__`. If it
                is equal to the string specified by the `all_label`
                parameter of :meth:`__init__`, stop all timers.
        """

        # Default label is self.default_label
        if labels is None:
            labels = self.default_label
        # All timers are affected if label is equal to self.all_label,
        # otherwise only the timer(s) specified by label
        if labels == self.all_label:
            labels = list(self.t0.keys())
        elif not isinstance(labels, (list, tuple)):
            labels = [
                labels,
            ]
        # Iterate over specified label(s)
        for lbl in labels:
            if lbl not in self.t0:
                raise KeyError(f"Unrecognized timer key {lbl}.")
            # Set start time to None to indicate timer is not running
            self.t0[lbl] = None
            # Set time accumulator to zero
            self.td[lbl] = 0.0

    def elapsed(self, label: Optional[str] = None, total: bool = True) -> float:
        """Get elapsed time since timer start.

        Args:
           label: Label of the timer for which the elapsed time is
               required. If it is ``None``, the default timer with label
               specified by the `default_label` parameter of
               :meth:`__init__` is selected.
           total: If ``True`` return the total elapsed time since the
               first call of :meth:`start` for the selected timer,
               otherwise return the elapsed time since the most recent
               call of :meth:`start` for which there has not been a
               corresponding call to :meth:`stop`.

        Returns:
           Elapsed time.
        """

        # Get current time
        t = timer()
        # Default label is self.default_label
        if label is None:
            label = self.default_label
            # Return 0.0 if default timer selected and it is not initialised
            if label not in self.t0:
                return 0.0
        # Raise exception if timer with specified label does not exist
        if label not in self.t0:
            raise KeyError(f"Unrecognized timer key {label}.")
        # If total flag is True return sum of accumulated time from
        # previous start/stop calls and current start call, otherwise
        # return just the time since the current start call
        te = 0.0
        if self.t0[label] is not None:
            te = t - self.t0[label]  # type: ignore
        if total:
            te += self.td[label]

        return te

    def labels(self) -> List[str]:
        """Get a list of timer labels.

        Returns:
          List of timer labels.
        """

        return list(self.t0.keys())

    def __str__(self) -> str:
        """Return string representation of object.

        The representation consists of a table with the following columns:

          * Timer label.
          * Accumulated time from past start/stop calls.
          * Time since current start call, or 'Stopped' if timer is not
            currently running.
        """

        # Get current time
        t = timer()
        # Length of label field, calculated from max label length
        fldlen = [len(lbl) for lbl in self.t0] + [
            len(self.default_label),
        ]
        lfldln = max(fldlen) + 2
        # Header string for table of timers
        s = f"{'Label':{lfldln}s}  Accum.       Current\n"
        s += "-" * (lfldln + 25) + "\n"
        # Construct table of timer details
        for lbl in sorted(self.t0):
            td = self.td[lbl]
            if self.t0[lbl] is None:
                ts = " Stopped"
            else:
                ts = f" {(t - self.t0[lbl]):.2e} s" % (t - self.t0[lbl])  # type: ignore
            s += f"{lbl:{lfldln}s}  {td:.2e} s  {ts}\n"

        return s


class ContextTimer:
    """A wrapper class for :class:`Timer` that enables its use as a
    context manager.

    For example, instead of

    >>> t = Timer()
    >>> t.start()
    >>> x = sum(range(1000))
    >>> t.stop()
    >>> elapsed = t.elapsed()

    one can use

    >>> t = Timer()
    >>> with ContextTimer(t):
    ...   x = sum(range(1000))
    >>> elapsed = t.elapsed()
    """

    def __init__(
        self,
        timer: Optional[Timer] = None,
        label: Optional[str] = None,
        action: str = "StartStop",
    ):
        """
        Args:
           timer: Timer object to be used as a context manager. If
              ``None``, a new class:`Timer` object is constructed.
           label: Label of the timer to be used. If it is ``None``, start
              the default timer.
           action: Actions to be taken on context entry and exit. If the
              value is 'StartStop', start the timer on entry and stop on
              exit; if it is 'StopStart', stop the timer on entry and
              start it on exit.
        """

        if action not in ["StartStop", "StopStart"]:
            raise ValueError(f"Unrecognized action {action}.")
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer
        self.label = label
        self.action = action

    def __enter__(self):
        """Start the timer and return this ContextTimer instance."""

        if self.action == "StartStop":
            self.timer.start(self.label)
        else:
            self.timer.stop(self.label)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the timer and return ``True`` if no exception was raised
        within the `with` block, otherwise return ``False``.
        """

        if self.action == "StartStop":
            self.timer.stop(self.label)
        else:
            self.timer.start(self.label)
        return not exc_type

    def elapsed(self, total: bool = True) -> float:
        """Return the elapsed time for the timer.

        Args:
          total: If ``True`` return the total elapsed time since the
             first call of :meth:`start` for the selected timer,
             otherwise return the elapsed time since the most recent call
             of :meth:`start` for which there has not been a
             corresponding call to :meth:`stop`.

        Returns:
          Elapsed time.
        """

        return self.timer.elapsed(self.label, total=total)
