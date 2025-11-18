import operator as op

import numpy as np

from jax import config

import pytest

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)
from typing import Optional

import jax

import scico.numpy as snp
from scico import linop
from scico.random import randn
from scico.typing import PRNGKey

SCALARS = (2, 1e0, snp.array(1.0))


def adjoint_test(
    A: linop.LinearOperator,
    key: Optional[PRNGKey] = None,
    rtol: float = 1e-4,
    x: Optional[snp.Array] = None,
    y: Optional[snp.Array] = None,
):
    """Check the validity of A.conj().T as the adjoint for a LinearOperator A.

    Args:
        A: LinearOperator to test.
        key: PRNGKey for generating `x`.
        rtol: Relative tolerance.
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


class LinearOperatorTestObj:
    def __init__(self, dtype):
        M, N = (8, 16)
        key = jax.random.PRNGKey(12345)
        self.dtype = dtype

        self.A, key = randn((M, N), dtype=dtype, key=key)
        self.B, key = randn((M, N), dtype=dtype, key=key)
        self.C, key = randn((N, M), dtype=dtype, key=key)
        self.D, key = randn((M, N - 1), dtype=dtype, key=key)

        self.x, key = randn((N,), dtype=dtype, key=key)
        self.y, key = randn((M,), dtype=dtype, key=key)

        self.Ao = AbsMatOp(self.A)
        self.Bo = AbsMatOp(self.B)
        self.Co = AbsMatOp(self.C)
        self.Do = AbsMatOp(self.D)


@pytest.fixture(scope="module", params=[np.float32, np.float64, np.complex64, np.complex128])
def testobj(request):
    yield LinearOperatorTestObj(request.param)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_binary_op(testobj, operator):
    # Our AbsMatOp class does not override the __add__, etc
    # so AbsMatOp + AbsMatOp -> LinearOperator
    # So to verify results, we evaluate the new LinearOperator on a random input

    comp_mat = operator(testobj.A, testobj.B)  # composite matrix
    comp_op = operator(testobj.Ao, testobj.Bo)  # composite linop

    assert isinstance(comp_op, linop.LinearOperator)  # Ensure we don't get a Map
    assert comp_op.input_dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat @ testobj.x, comp_op @ testobj.x, rtol=0, atol=1e-5)

    # linops of different sizes
    with pytest.raises(ValueError):
        operator(testobj.Ao, testobj.Co)
    with pytest.raises(ValueError):
        operator(testobj.Ao, testobj.Do)


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
@pytest.mark.parametrize("scalar", SCALARS)
def test_scalar_left(testobj, operator, scalar):
    comp_mat = operator(testobj.A, scalar)
    comp_op = operator(testobj.Ao, scalar)
    assert isinstance(comp_op, linop.LinearOperator)  # Ensure we don't get a Map
    assert comp_op.input_dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat @ testobj.x, comp_op @ testobj.x, rtol=5e-5)
    np.testing.assert_allclose(comp_mat.conj().T @ testobj.y, comp_op.adj(testobj.y), rtol=2e-4)


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
@pytest.mark.parametrize("scalar", SCALARS)
def test_scalar_right(testobj, operator, scalar):
    if operator == op.truediv:
        pytest.xfail("scalar / LinearOperator is not supported")
    comp_mat = operator(scalar, testobj.A)
    comp_op = operator(scalar, testobj.Ao)
    assert comp_op.input_dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat @ testobj.x, comp_op @ testobj.x, rtol=5e-5)


def test_negation(testobj):
    comp_mat = -testobj.A
    comp_op = -testobj.Ao
    assert comp_op.input_dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat @ testobj.x, comp_op @ testobj.x, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_invalid_add_sub_array(testobj, operator):
    # Try to add or subtract an ndarray with AbsMatOp
    with pytest.raises(TypeError):
        operator(testobj.A, testobj.Ao)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_invalid_add_sub_scalar(testobj, operator):
    # Try to add or subtract a scalar with AbsMatOp
    with pytest.raises(TypeError):
        operator(1.0, testobj.Ao)


def test_matmul_left(testobj):
    comp_mat = testobj.A @ testobj.C
    comp_op = testobj.Ao @ testobj.Co
    assert comp_op.input_dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat @ testobj.y, comp_op @ testobj.y, rtol=5e-5)


def test_matmul_right(testobj):
    comp_mat = testobj.C @ testobj.A
    comp_op = testobj.Co @ testobj.Ao
    assert comp_op.input_dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat @ testobj.x, comp_op @ testobj.x, rtol=5e-5)


def test_matvec_left(testobj):
    comp_mat = testobj.A @ testobj.x
    comp_op = testobj.Ao @ testobj.x
    assert comp_op.dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat, comp_op, rtol=5e-5)


def test_matvec_right(testobj):
    comp_mat = testobj.C @ testobj.y
    comp_op = testobj.Co @ testobj.y
    assert comp_op.dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat, comp_op, rtol=5e-5)


def test_gram(testobj):
    Ao = testobj.Ao
    a = Ao.gram(testobj.x)
    b = Ao.conj().T @ Ao @ testobj.x
    c = Ao.gram_op @ testobj.x

    comp_mat = testobj.A.conj().T @ testobj.A @ testobj.x

    np.testing.assert_allclose(a, comp_mat, rtol=5e-5)
    np.testing.assert_allclose(b, comp_mat, rtol=5e-5)
    np.testing.assert_allclose(c, comp_mat, rtol=5e-5)


def test_matvec_call(testobj):
    # A @ x and A(x) should return same
    np.testing.assert_allclose(testobj.Ao @ testobj.x, testobj.Ao(testobj.x), rtol=5e-5)


def test_adj_composition(testobj):
    Ao = testobj.Ao
    Bo = testobj.Bo
    A = testobj.A
    B = testobj.B
    x = testobj.x

    comp_mat = A.conj().T @ B
    a = Ao.conj().T @ Bo
    b = Ao.adj(Bo)
    assert a.input_dtype == testobj.A.dtype
    assert b.input_dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat @ x, a @ x, rtol=5e-5)
    np.testing.assert_allclose(comp_mat @ x, b @ x, rtol=5e-5)


def test_transpose_matvec(testobj):
    Ao = testobj.Ao
    y = testobj.y

    a = Ao.T @ y
    b = y.T @ Ao

    comp_mat = testobj.A.T @ y

    assert a.dtype == testobj.A.dtype
    assert b.dtype == testobj.A.dtype
    np.testing.assert_allclose(a, comp_mat, rtol=1e-4)
    np.testing.assert_allclose(a, b, rtol=5e-5)


def test_transpose_matmul(testobj):
    Ao = testobj.Ao
    Bo = testobj.Bo
    x = testobj.x
    comp_op = Ao.T @ Bo
    comp_mat = testobj.A.T @ testobj.B
    assert comp_op.input_dtype == testobj.A.dtype
    np.testing.assert_allclose(comp_mat @ x, comp_op @ x, rtol=5e-5)


def test_conj_transpose_matmul(testobj):
    Ao = testobj.Ao
    Bo = testobj.Bo
    x = testobj.x
    comp_op = Ao.conj().T @ Bo
    comp_mat = testobj.A.conj().T @ testobj.B
    assert comp_mat == testobj.A.dtype
    np.testing.assert_allclose(comp_mat @ x, comp_op @ x, rtol=5e-5)


def test_conj_matvec(testobj):
    Ao = testobj.Ao
    x = testobj.x
    a = Ao.conj() @ x
    comp_mat = testobj.A.conj() @ x
    assert a.dtype == testobj.A.dtype
    np.testing.assert_allclose(a, comp_mat, rtol=5e-5)


def test_adjoint_matvec(testobj):
    Ao = testobj.Ao
    y = testobj.y

    a = Ao.adj(y)
    b = Ao.conj().T @ y
    c = (y.conj().T @ Ao).conj()

    comp_mat = testobj.A.conj().T @ y

    assert a.dtype == testobj.A.dtype
    assert b.dtype == testobj.A.dtype
    assert c.dtype == testobj.A.dtype
    np.testing.assert_allclose(a, comp_mat, rtol=1e-4)
    np.testing.assert_allclose(a, b, rtol=5e-5)
    np.testing.assert_allclose(a, c, rtol=5e-5)


def test_adjoint_matmul(testobj):
    # shape mismatch
    Ao = testobj.Ao
    Co = testobj.Co

    with pytest.raises(ValueError):
        Ao.adj(Co)


def test_hermitian(testobj):
    Ao = testobj.Ao
    y = testobj.y

    np.testing.assert_allclose(Ao.conj().T @ y, Ao.H @ y)


def test_shape(testobj):
    Ao = testobj.Ao
    x = testobj.x
    y = testobj.y

    with pytest.raises(ValueError):
        _ = Ao @ y

    with pytest.raises(ValueError):
        _ = Ao(y)

    with pytest.raises(ValueError):
        _ = Ao.T @ x

    with pytest.raises(ValueError):
        _ = Ao.adj(x)


def test_adj_lazy():
    dtype = np.float32
    M, N = (8, 16)
    A, key = randn((M, N), dtype=np.float32, key=None)
    y, key = randn((M,), dtype=np.float32, key=key)
    Ao = AbsMatOp(A, adj_fn=None)  # defer setting the linop

    assert Ao._adj is None
    a = Ao.adj(y)  # Adjoint is set when .adj() is called
    b = A.T @ y
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_jit_adj_lazy():
    dtype = np.float32
    M, N = (8, 16)
    A, key = randn((M, N), dtype=np.float32, key=None)
    y, key = randn((M,), dtype=np.float32, key=key)
    Ao = AbsMatOp(A, adj_fn=None)  # defer setting the linop
    assert Ao._adj is None
    Ao.jit()  # Adjoint set here
    assert Ao._adj is not None
    a = Ao.adj(y)
    b = A.T @ y
    np.testing.assert_allclose(a, b, rtol=1e-5)
