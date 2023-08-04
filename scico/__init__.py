# Copyright (C) 2021-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Scientific Computational Imaging COde (SCICO) is a Python package for
solving the inverse problems that arise in scientific imaging applications.
"""

__version__ = "0.0.4"

import logging
import sys

# isort: off

# Suppress jax device warning. See https://github.com/google/jax/issues/6805
# This only works for jax>0.3.23; for earlier versions, the getLogger
# argument should be "absl". Two filters are included here due to a change
# in jax between versions 0.4.2 and 0.4.8, both of which are supported by
# scico.
logging.getLogger("jax._src.lib.xla_bridge").addFilter(  # jax 0.4.2
    logging.Filter("No GPU/TPU found, falling back to CPU.")
)
logging.getLogger("jax._src.xla_bridge").addFilter(  # jax 0.4.8
    logging.Filter("No GPU/TPU found, falling back to CPU.")
)

# isort: on

import jax
from jax import custom_jvp, custom_vjp, hessian, jacfwd, jvp, linearize, vjp

import jaxlib

from . import numpy
from ._autograd import cvjp, grad, jacrev, linear_adjoint, value_and_grad

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
