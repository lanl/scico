import numpy as np

from jax.config import config

import pytest

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)

import jax

import scico.numpy as snp
from scico import linop
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
        key = jax.random.PRNGKey(12345)
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
    I = linop.Identity(8)
    Inorm = linop.operator_norm(I)
    assert np.abs(Inorm - 1.0) < 1e-5
    key = jax.random.PRNGKey(12345)
    for dtype in [np.float32, np.complex64]:
        d, key = randn((16,), dtype=dtype, key=key)
        D = linop.Diagonal(d)
        Dnorm = linop.operator_norm(D)
        assert np.abs(Dnorm - snp.abs(d).max()) < 1e-5
