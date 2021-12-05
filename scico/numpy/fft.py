# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Construct wrapped versions of :mod:`jax.numpy.fft` functions.

This modules consists of functions from :mod:`jax.numpy.fft`. Some of
these functions are wrapped to support compatibility with
:class:`scico.blockarray.BlockArray` and are documented here.
The remaining functions are imported directly from :mod:`jax.numpy.fft`.
While they can be imported from the :mod:`scico.numpy.fft` namespace,
they are not documented here; please consult the documentation for the
source module :mod:`jax.numpy.fft`.
"""
import sys

import jax.numpy.fft

from ._util import _attach_wrapped_func, _not_implemented

_not_implemented_functions = []
for name, func in jax._src.util.get_module_functions(jax.numpy.fft).items():
    if name not in globals():
        _not_implemented_functions.append((name, func))

_attach_wrapped_func(
    _not_implemented_functions, _not_implemented, module_name=sys.modules[__name__]
)
