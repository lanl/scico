# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Scientific Computational Imaging COde (SCICO) is a Python package for
solving the inverse problems that arise in scientific imaging applications.
"""

__version__ = "0.0.4.dev0"

import sys

from . import _python37  # python 3.7 compatibility

# isort: off
from ._autograd import grad, jacrev, linear_adjoint, value_and_grad, cvjp

import jax, jaxlib

from jax import custom_jvp, custom_vjp, jacfwd, jvp, linearize, vjp, hessian

from . import numpy

__all__ = [
    "grad",
    "value_and_grad",
    "linear_adjoint",
    "vjp",
    "cvjp",
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
