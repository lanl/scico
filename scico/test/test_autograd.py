import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
from scico.random import randn


class GradTestObj:
    def __init__(self, dtype):
        M, N = (3, 4)
        key = jax.random.PRNGKey(12345)
        self.dtype = dtype

        self.A, key = randn((M, N), dtype=dtype, key=key)
        self.x, key = randn((N,), dtype=dtype, key=key)
        self.y, key = randn((M,), dtype=dtype, key=key)

        self.f = lambda x: 0.5 * snp.sum(snp.abs(self.y - self.A @ x) ** 2)


@pytest.fixture(scope="module", params=[np.float32, np.complex64])
def testobj(request):
    yield GradTestObj(request.param)


def test_grad(testobj):
    A = testobj.A
    x = testobj.x
    y = testobj.y
    f = testobj.f

    sgrad = scico.grad(f)(x)
    an_grad = A.conj().T @ (A @ x - y)

    np.testing.assert_allclose(sgrad, an_grad, rtol=1e-4)


def test_grad_aux(testobj):
    A = testobj.A
    x = testobj.x
    y = testobj.y

    def g(x):
        return testobj.f(x), True

    sgrad, aux = scico.grad(g, has_aux=True)(x)
    an_grad = A.conj().T @ (A @ x - y)

    assert aux == True
    np.testing.assert_allclose(sgrad, an_grad, rtol=1e-4)


def test_value_and_grad(testobj):
    A = testobj.A
    x = testobj.x
    y = testobj.y
    f = testobj.f

    svalue, sgrad = scico.value_and_grad(f)(x)

    an_val = f(x)
    an_grad = A.conj().T @ (A @ x - y)

    np.testing.assert_allclose(svalue, an_val, rtol=1e-4)
    np.testing.assert_allclose(sgrad, an_grad, rtol=1e-4)


def test_value_and_grad_aux(testobj):
    A = testobj.A
    x = testobj.x
    y = testobj.y

    def g(x):
        return testobj.f(x), True

    (svalue, aux), sgrad = scico.value_and_grad(g, has_aux=True)(x)

    an_val, aux_ = g(x)
    an_grad = A.conj().T @ (A @ x - y)

    assert aux == aux_
    np.testing.assert_allclose(svalue, an_val, rtol=1e-4)
    np.testing.assert_allclose(sgrad, an_grad, rtol=1e-4)


def test_linear_adjoint(testobj):
    # Verify that linear_adjoint returns a function that
    # implements f(y) = A.conj().T @ y
    A = testobj.A
    x = testobj.x
    y = testobj.y

    f = lambda x: A @ x

    A_adj = scico.linear_adjoint(f, x)
    np.testing.assert_allclose(A.conj().T @ y, A_adj(testobj.y)[0], rtol=1e-4)

    # Test a function with with multiple inputs
    # Same as np.array([0.5, -0.5j])
    f = lambda x, y: 0.5 * x - 0.5j * y

    f_transpose = scico.linear_adjoint(f, 1.0j, 1.0j)
    a, b = f_transpose(1.0 + 0.0j)
    assert a == 0.5
    assert b == 0.5j


def test_linear_adjoint_r_to_c():
    f = snp.fft.rfft
    x, key = randn((32,))
    adj = scico.linear_adjoint(f, x)

    a = snp.sum(x * adj(f(x))[0])
    b = snp.linalg.norm(f(x)) ** 2

    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_linear_adjoint_c_to_r():
    f = snp.fft.irfft
    x, key = randn((32,), dtype=np.complex64)
    adj = scico.linear_adjoint(f, x)

    a = snp.sum(x.conj() * adj(f(x))[0])
    b = snp.linalg.norm(f(x)) ** 2

    np.testing.assert_allclose(a.real, b.real, rtol=1e-4)
    np.testing.assert_allclose(a.imag, 0, atol=1e-2)
