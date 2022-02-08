# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Tools to construct wrapped versions of :mod:`jax.numpy` functions."""

import re
import types
from functools import wraps

import numpy as np

from jaxlib.xla_extension import CompiledFunction

# wrapper for not-implemented jax.numpy functions
# stripped down version of jax._src.lax_numpy._not_implemented and jax.utils._wraps

_NOT_IMPLEMENTED_DESC = """
**WARNING**: This function is not yet implemented by :mod:`scico.numpy` and
may raise an error when operating on :class:`scico.blockarray.BlockArray`.
"""


def _not_implemented(fun):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        return fun(*args, **kwargs)

    if not hasattr(fun, "__doc__") or fun.__doc__ is None:
        return wrapped

    # wrapped.__doc__ = fun.__doc__ + "\n\n" + _NOT_IMPLEMENTED_DESC
    wrapped.__doc__ = re.sub(
        r"^\*Original docstring below\.\*",
        _NOT_IMPLEMENTED_DESC + r"\n\n" + "*Original docstring below.*",
        wrapped.__doc__,
        flags=re.M,
    )
    return wrapped


def _attach_wrapped_func(funclist, wrapper, module_name, fix_mod_name=False):
    # funclist is either a function, or a tuple (name-in-this-module, function)
    # wrapper is a function that is applied to each function in funclist, with
    #  the output being assigned as an attribute of the module `module_name`
    for func in funclist:
        # Test required because func.__name__ isn't always the name we want
        # e.g. jnp.abs.__name__ resolves to 'absolute', not 'abs'
        if isinstance(func, tuple):
            fname = func[0]
            fref = func[1]
        else:
            fname = func.__name__
            fref = func
        # Set wrapped function as an attribute in module_name
        setattr(module_name, fname, wrapper(fref))
        # Set __module__ attribute of wrapped function to
        # module_name.__name__ (i.e., scico.numpy) so that it does not
        # appear to autodoc be an imported function
        if fix_mod_name:
            getattr(module_name, fname).__module__ = module_name.__name__


def _get_module_functions(module):
    """Finds functions in module.

    This function is a slightly modified version of
    :func:`jax._src.util.get_module_functions`. Unlike the JAX version,
    this version will also return any
    :class:`jaxlib.xla_extension.CompiledFunction`s that exist in the
    module.

    Args:
        module: A Python module.
    Returns:
        module_fns: A dict of names mapped to functions, builtins or
        ufuncs in `module`.
    """
    module_fns = {}
    for key in dir(module):
        # Omitting module level __getattr__, __dir__ which was added in Python 3.7
        # https://www.python.org/dev/peps/pep-0562/
        if key in ("__getattr__", "__dir__"):
            continue
        attr = getattr(module, key)
        if isinstance(
            attr, (types.BuiltinFunctionType, types.FunctionType, np.ufunc, CompiledFunction)
        ):
            module_fns[key] = attr
    return module_fns
