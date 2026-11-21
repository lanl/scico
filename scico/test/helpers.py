"""Utilities shared by multiple modules of the test suite.

These were previously defined in ``scico/test/linop/test_linop.py`` and
imported from there by other test modules, which made that module both a
collection of tests and a library. They are gathered here instead so that
importing them does not import a test module.
"""

from typing import Optional

from jax import config

# Enable 64-bit mode for output dtype checks. This is required here, and not
# only in the test modules that check dtypes, because importing the utilities
# below previously imported test_linop, which enables it. Test modules that
# depend on 64-bit mode being enabled therefore continue to get it.
config.update("jax_enable_x64", True)

import scico.numpy as snp
from scico import linop
from scico.typing import PRNGKey

# Scalars used in parametrized tests of operator/scalar arithmetic: a Python
# int, a Python float, and a zero-dimensional jax array.
SCALARS = (2, 1e0, snp.array(1.0))


def adjoint_test(
    A: linop.LinearOperator,
    key: Optional[PRNGKey] = None,
    rtol: float = 1e-4,
    x: Optional[snp.Array] = None,
    y: Optional[snp.Array] = None,
):
    r"""Check the validity of A.conj().T as the adjoint for a LinearOperator A.

    Args:
        A: LinearOperator to test.
        key: PRNGKey for generating `x`.
        rtol: Relative tolerance.
        x: If not the default ``None``, use the specified array instead
           of a random array as test vector :math:`\mb{x}`.
        y: If not the default ``None``, use the specified array instead
           of a random array as test vector :math:`\mb{y}`.
    """

    assert linop.valid_adjoint(A, A.H, key=key, eps=rtol, x=x, y=y)


class AbsMatOp(linop.LinearOperator):
    """Simple LinearOperator subclass for testing purposes.

    Similar to linop.MatrixOperator, but does not use the specialized
    MatrixOperator methods (.T, adj, etc). Used to verify the
    LinearOperator interface.
    """

    def __init__(self, A, adj_fn=None):
        self.A = A
        super().__init__(
            input_shape=A.shape[1], output_shape=A.shape[0], input_dtype=A.dtype, adj_fn=adj_fn
        )

    def _eval(self, x):
        return self.A @ x
