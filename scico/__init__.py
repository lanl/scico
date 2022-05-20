# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Scientific Computational Imaging COde (SCICO) is a Python package for
solving the inverse problems that arise in scientific imaging applications.
"""

__version__ = "0.0.3.dev0"

import sys

# Python 3.7 compatibility
try:
    from math import prod
    from typing import Literal

    del Literal, prod
except ImportError:
    import typing

    typing.Literal = typing.Any  # type: ignore

    import math
    from functools import reduce
    from operator import mul

    # use a double lambda to bind reduce and mul right now
    math.prod = (lambda fun, op: lambda iterable: fun(op, iterable, 1))(reduce, mul)  # type: ignore

    del typing, math, mul, reduce


# isort: off
from ._autograd import grad, jacrev, linear_adjoint, value_and_grad

import jax, jaxlib

jax_ver_req = "0.3.0"
jaxlib_ver_req = "0.3.0"
if jax.__version__ < jax_ver_req:
    raise Exception(
        f"SCICO {__version__} requires jax>={jax_ver_req}; got {jax.__version__}; "
        "please upgrade jax."
    )
if jaxlib.__version__ < jaxlib_ver_req:
    raise Exception(
        f"SCICO {__version__} requires jaxlib>={jaxlib_ver_req}; got {jaxlib.__version__}; "
        "please upgrade jaxlib."
    )

from jax import custom_jvp, custom_vjp, jacfwd, jvp, linearize, vjp, hessian

from . import numpy

__all__ = [
    "grad",
    "value_and_grad",
    "linear_adjoint",
    "vjp",
    "jvp",
    "jacfwd",
    "jacrev",
    "linearize",
    "hessian",
    "custom_jvp",
    "custom_vjp",
]

# Imported items in __all__ appear to originate in top-level functional module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
