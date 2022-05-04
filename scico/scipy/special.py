# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

""":class:`~scico.numpy.BlockArray`-compatible :mod:`jax.scipy.special`
functions.

This modules is a wrapper for :mod:`jax.scipy.special` where some
functions have been extended to automatically map over block array
blocks as described in :ref:`numpy_functions_blockarray`.
"""
import jax.scipy.special as js

from scico.numpy import _wrappers

# add most everything in jax.scipy.special to this module
_wrappers.add_attributes(
    to_dict=vars(),
    from_dict=js.__dict__,
)

# wrap select functions
functions = (
    "betainc",
    "entr",
    "erf",
    "erfc",
    "erfinv",
    "expit",
    "gammainc",
    "gammaincc",
    "gammaln",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "log_ndtr",
    "logit",
    "logsumexp",
    "multigammaln",
    "ndtr",
    "ndtri",
    "polygamma",
    "sph_harm",
    "xlog1py",
    "xlogy",
    "zeta",
    "digamma",
)
_wrappers.wrap_recursively(vars(), functions, _wrappers.map_func_over_blocks)

# clean up
del js, _wrappers
