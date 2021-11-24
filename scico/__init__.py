# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""SCICO -- a Python package for solving the inverse problems that arise in scientific imaging applications."""

__version__ = "0.0.2a1"

import sys

# isort: off
from ._autograd import grad, jacrev, linear_adjoint, value_and_grad

# TODO remove this check as we get closer to release?
import jax, jaxlib

if jax.__version__ < "0.2.19":
    raise Exception(
        f"""SCICO {__version__} requires jax>0.2.19; got {jax.__version__}; please upgrade jax."""
    )
if jaxlib.__version__ < "0.1.70":
    raise Exception(
        f"""SCICO {__version__} requires jaxlib>0.1.70; got {jaxlib.__version__}; please upgrade jaxlib."""
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
