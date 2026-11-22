"""Utilities shared by multiple modules of the test suite.

These were previously defined in ``scico/test/linop/test_linop.py`` and
imported from there by other test modules, which made that module both a
collection of tests and a library. They are gathered here instead so that
importing them does not import a test module.
"""

import operator as op
from typing import Optional

import numpy as np

import scico.numpy as snp
from scico import linop
from scico.typing import PRNGKey

# Scalars used in parametrized tests of operator/scalar arithmetic: a Python
# int, a Python float, and a zero-dimensional jax array.
SCALARS = (2, 1e0, snp.array(1.0))


def scalar_test(
    A,
    x,
    scalar,
    operator=op.mul,
    scalar_first: bool = False,
    rtol: float = 5e-5,
    check_dtype: bool = True,
):
    r"""Check that combining an operator with a scalar combines its output.

    For an operator :math:`A`, a scalar :math:`s` and the binary operation
    :math:`\circ`, check that :math:`(A \circ s)(x) = A(x) \circ s`, which
    is the defining property of scalar arithmetic on operators.

    Note that the argument order is specified by `scalar_first` rather than
    by the words "left" and "right", which have been used inconsistently in
    this test suite.

    Args:
        A: Operator or LinearOperator to test.
        x: Array at which to evaluate the operators.
        scalar: Scalar to combine with `A`.
        operator: Binary operation combining `A` and `scalar`, typically
           one of those in the standard :mod:`operator` module.
        scalar_first: If ``True``, apply the operation as
           `operator(scalar, A)`; if ``False``, as `operator(A, scalar)`.
        rtol: Relative tolerance.
        check_dtype: If ``True``, also check that the combined operator
           declares the output dtype that it in fact produces.
    """

    if scalar_first:
        combined, expected = operator(scalar, A), operator(scalar, A(x))
    else:
        combined, expected = operator(A, scalar), operator(A(x), scalar)

    if check_dtype:
        assert combined.output_dtype == expected.dtype

    np.testing.assert_allclose(combined(x), expected, rtol=rtol)


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
