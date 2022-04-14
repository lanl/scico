# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Wrapped versions of :mod:`jax.scipy.special` functions.

This modules consists of functions from :mod:`jax.scipy.special`. Some of
these functions are wrapped to support compatibility with
:class:`scico.numpy.BlockArray` and are documented here. The
remaining functions are imported directly from :mod:`jax.numpy`. While
they can be imported from the :mod:`scico.numpy` namespace, they are not
documented here; please consult the documentation for the source module
:mod:`jax.scipy.special`.
"""
import jax.scipy.special as js

from scico.numpy import _util

_util.add_attributes(
    vars(),
    js.__dict__,
)

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


_util.wrap_recursively(vars(), functions, _util.map_func_over_blocks)
