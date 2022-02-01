# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Wrapped versions of :mod:`jax.scipy.special` functions.

This modules consists of functions from :mod:`jax.scipy.special`. Some of
these functions are wrapped to support compatibility with
:class:`scico.blockarray.BlockArray` and are documented here. The
remaining functions are imported directly from :mod:`jax.numpy`. While
they can be imported from the :mod:`scico.numpy` namespace, they are not
documented here; please consult the documentation for the source module
:mod:`jax.scipy.special`.
"""


import sys

import jax
import jax.scipy.special as js

from scico.blockarray import _block_array_ufunc_wrapper
from scico.numpy._util import _attach_wrapped_func, _not_implemented

_ufunc_functions = [
    js.betainc,
    js.entr,
    js.erf,
    js.erfc,
    js.erfinv,
    js.expit,
    js.gammainc,
    js.gammaincc,
    js.gammaln,
    js.i0,
    js.i0e,
    js.i1,
    js.i1e,
    js.log_ndtr,
    js.logit,
    js.logsumexp,
    js.multigammaln,
    js.ndtr,
    js.ndtri,
    js.polygamma,
    js.sph_harm,
    js.xlog1py,
    js.xlogy,
    js.zeta,
]

_attach_wrapped_func(
    _ufunc_functions,
    _block_array_ufunc_wrapper,
    module_name=sys.modules[__name__],
    fix_mod_name=True,
)

psi = _block_array_ufunc_wrapper(js.digamma)
digamma = _block_array_ufunc_wrapper(js.digamma)

_not_implemented_functions = []
for name, func in jax._src.util.get_module_functions(js).items():
    if name not in globals():
        _not_implemented_functions.append((name, func))

_attach_wrapped_func(
    _not_implemented_functions, _not_implemented, module_name=sys.modules[__name__]
)
