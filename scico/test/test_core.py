import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
from scico.random import randn


class GradTestObj:
    def __init__(self, dtype):
        M, N = (3, 4)
        key = jax.random.key(12345)
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


@pytest.mark.parametrize("shape", [(2, 3), ((2, 3), (4,))])
def test_linear_transpose(shape):
    fun = lambda x: snp.pad(x, 2)
    za = snp.zeros(shape, dtype=snp.float32)
    fza = fun(za)
    dts = jax.ShapeDtypeStruct(shape, dtype=snp.float32)
    lt_za = scico.linear_transpose(fun, za)
    lt_dts = scico.linear_transpose(fun, dts)
    lt_za_fza = lt_za(fza)[0]
    lt_dts_fza = lt_dts(fza)[0]
    assert lt_za_fza.shape == lt_dts_fza.shape
    assert lt_za_fza.dtype == lt_dts_fza.dtype


@pytest.mark.parametrize("shape", [(2, 3), ((2, 3), (4,))])
def test_linear_adjoint_shape(shape):
    fun = lambda x: snp.pad(x, 2)
    za = snp.zeros(shape, dtype=snp.float32)
    fza = fun(za)
    dts = jax.ShapeDtypeStruct(shape, dtype=snp.float32)
    lt_za = scico.linear_adjoint(fun, za)
    lt_dts = scico.linear_adjoint(fun, dts)
    lt_za_fza = lt_za(fza)[0]
    lt_dts_fza = lt_dts(fza)[0]
    assert lt_za_fza.shape == lt_dts_fza.shape
    assert lt_za_fza.dtype == lt_dts_fza.dtype


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


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_cvjp(dtype):
    A, key = randn((3, 3), dtype=dtype)
    B, key = randn((3, 4), dtype=dtype, key=key)
    xp, key = randn((3,), dtype=dtype, key=key)
    yp, key = randn((4,), dtype=dtype, key=key)

    def fun(x, y):
        return A @ x + B @ y

    px, jfnx = scico.cvjp(fun, xp, yp, jidx=0)
    py, jfny = scico.cvjp(fun, xp, yp, jidx=1)

    for k in range(3):
        v = np.zeros((3,), dtype=dtype)
        v[k] = 1.0
        np.testing.assert_allclose(jfnx(v)[0], A[k].conj())
        np.testing.assert_allclose(jfny(v)[0], B[k].conj())


@pytest.mark.parametrize(
    "argskwargs",
    [
        [(snp.ones((3,)), snp.ones((3,)), 1.0), {}],
        [(1.1 * snp.ones((3,)), snp.ones((3,))), {"z": snp.zeros((3,))}],
        [(snp.ones(((2,), (3, 2))), 1.0, 1.0), {}],
        [
            (snp.ones(((2,), (3, 2))), snp.blockarray(((2,), (3, 2)))),
            {"z": 2.0 * snp.ones(((2,), (3, 2)))},
        ],
    ],
)
def test_eval_shape_1(argskwargs):
    def _fun(x, y, z):
        """Test function"""
        return x + y * z

    def _conv(arg):
        """Convert array to jax.ShapeDtypeStruct."""
        if hasattr(arg, "shape"):
            return jax.ShapeDtypeStruct(arg.shape, dtype=arg.dtype)
        else:
            return arg

    args, kwargs = argskwargs
    # Reference shape computed for array objects
    ref_shape = jax.eval_shape(_fun, *args, **kwargs)
    map_args = [_conv(v) for v in args]
    map_kwargs = {k: _conv(v) for k, v in kwargs.items()}
    # Test shape computed for jax.ShapeDtypeStruct objects
    tst_shape = scico.eval_shape(_fun, *map_args, **map_kwargs)
    assert tst_shape.shape == ref_shape.shape


@pytest.mark.parametrize(
    "arrdts",
    [
        [snp.ones((3, 2), dtype=snp.float32), jax.ShapeDtypeStruct((3, 2), dtype=snp.float32)],
        [
            snp.ones(((3,), (2, 3)), dtype=snp.float32),
            jax.ShapeDtypeStruct(((3,), (2, 3)), dtype=snp.float32),
        ],
    ],
)
def test_eval_shape_2(arrdts):
    _fun = lambda x: snp.pad(x, 2)
    arr, dts = arrdts
    # Reference shape computed for array
    ref_shape = jax.eval_shape(_fun, arr)
    # Test shape computed for jax.ShapeDtypeStruct
    tst_shape = scico.eval_shape(_fun, dts)
    assert tst_shape.shape == ref_shape.shape
