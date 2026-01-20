import numpy as np

from jax import config

import pytest

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)

import jax

import scico.numpy as snp
from scico import linop
from scico.operator import Operator
from scico.random import randn
from scico.test.linop.test_linop import AbsMatOp


def test_valid_adjoint():
    diagonal, key = randn((10,), dtype=np.float32)
    D = linop.Diagonal(diagonal=diagonal)
    assert linop.valid_adjoint(D, D.T, key=key, eps=None) < 1e-4
    x, key = randn((5,), dtype=np.float32)
    y, key = randn((5,), dtype=np.float32)
    with pytest.raises(ValueError):
        linop.valid_adjoint(D, D.T, key=key, x=x)
    with pytest.raises(ValueError):
        linop.valid_adjoint(D, D.T, key=key, y=y)


class PowerIterTestObj:
    def __init__(self, dtype):
        M, N = (4, 4)
        key = jax.random.key(12345)
        self.dtype = dtype

        A, key = randn((M, N), dtype=dtype, key=key)
        self.A = A.conj().T @ A  # ensure symmetric

        self.Ao = linop.MatrixOperator(self.A)
        self.Bo = AbsMatOp(self.A)

        self.key = key
        self.ev = snp.linalg.norm(
            self.A, 2
        )  # The largest eigenvalue of A is the spectral norm of A


@pytest.fixture(scope="module", params=[np.float32, np.complex64])
def pitestobj(request):
    yield PowerIterTestObj(request.param)


def test_power_iteration(pitestobj):
    """Verify that power iteration calculates largest eigenvalue for real and complex
    symmetric matrices.
    """
    # Test using the LinearOperator MatrixOperator
    mu, v = linop.power_iteration(A=pitestobj.Ao, maxiter=100, key=pitestobj.key)
    assert np.abs(mu - pitestobj.ev) < 1e-4

    # Test using the AbsMatOp for test_linop.py
    mu, v = linop.power_iteration(A=pitestobj.Bo, maxiter=100, key=pitestobj.key)
    assert np.abs(mu - pitestobj.ev) < 1e-4


def test_operator_norm():
    Iop = linop.Identity(8)
    Inorm = linop.operator_norm(Iop)
    assert np.abs(Inorm - 1.0) < 1e-5
    key = jax.random.key(12345)
    for dtype in [np.float32, np.complex64]:
        d, key = randn((16,), dtype=dtype, key=key)
        D = linop.Diagonal(d)
        Dnorm = linop.operator_norm(D)
        assert np.abs(Dnorm - snp.abs(d).max()) < 1e-5
    Zop = linop.MatrixOperator(snp.zeros((3, 3)))
    Znorm = linop.operator_norm(Zop)
    assert np.abs(Znorm) < 1e-6


@pytest.mark.parametrize("dtype", [snp.float32, snp.complex64])
@pytest.mark.parametrize("inc_eval", [True, False])
def test_jacobian(dtype, inc_eval):
    N = 7
    M = 8
    key = None
    fmx, key = randn((M, N), key=key, dtype=dtype)
    F = Operator(
        (N, 1),
        output_shape=(M, 1),
        eval_fn=lambda x: fmx @ x,
        input_dtype=dtype,
        output_dtype=dtype,
    )
    u, key = randn((N, 1), key=key, dtype=dtype)
    v, key = randn((N, 1), key=key, dtype=dtype)
    w, key = randn((M, 1), key=key, dtype=dtype)

    J = linop.jacobian(F, u, include_eval=inc_eval)
    Jv = J(v)
    JHw = J.H(w)

    if inc_eval:
        np.testing.assert_allclose(Jv[0], F(u))
        np.testing.assert_allclose(Jv[1], F.jvp(u, v)[1])
        np.testing.assert_allclose(JHw[0], F(u))
        np.testing.assert_allclose(JHw[1], F.vjp(u)[1](w))
    else:
        np.testing.assert_allclose(Jv, F.jvp(u, v)[1])
        np.testing.assert_allclose(JHw, F.vjp(u)[1](w))
