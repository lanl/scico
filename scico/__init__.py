__version__ = "0.0.1a4"

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
