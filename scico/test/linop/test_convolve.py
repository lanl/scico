import operator as op
import warnings

import numpy as np

import jax
import jax.scipy.signal as signal

import pytest

from scico.linop import Convolve, ConvolveByX, LinearOperator
from scico.random import randn
from scico.test.linop.test_linop import AbsMatOp, adjoint_test


class TestConvolve:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(32,), (32, 48)])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("jit", [False, True])
    def test_eval(self, input_shape, input_dtype, mode, jit):
        ndim = len(input_shape)

        filter_shape = (3, 4)[:ndim]

        x, key = randn(input_shape, dtype=input_dtype, key=self.key)
        psf, key = randn(filter_shape, dtype=input_dtype, key=key)
        A = Convolve(h=psf, input_shape=input_shape, input_dtype=input_dtype, mode=mode, jit=jit)
        Ax = A @ x
        y = signal.convolve(x, psf, mode=mode)
        np.testing.assert_allclose(Ax.ravel(), y.ravel(), rtol=1e-4)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(32,), (32, 48)])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("jit", [False, True])
    def test_adjoint(self, input_shape, mode, jit, input_dtype):

        ndim = len(input_shape)
        filter_shape = (3, 4)[:ndim]
        x, key = randn(input_shape, dtype=input_dtype, key=self.key)
        psf, key = randn(filter_shape, dtype=input_dtype, key=key)

        A = Convolve(h=psf, input_shape=input_shape, input_dtype=input_dtype, mode=mode, jit=jit)

        adjoint_test(A, self.key)


class ConvolveTestObj:
    def __init__(self):
        dtype = np.float32
        key = jax.random.PRNGKey(12345)

        self.psf_A, key = randn((3,), dtype=dtype, key=key)
        self.psf_B, key = randn((3,), dtype=dtype, key=key)
        self.psf_C, key = randn((5,), dtype=dtype, key=key)

        self.A = Convolve(input_shape=(32,), h=self.psf_A)
        self.B = Convolve(input_shape=(32,), h=self.psf_B)
        self.C = Convolve(input_shape=(32,), h=self.psf_C)

        # Matrix for a 'generic linop'
        m = self.A.output_shape[0]
        n = self.A.input_shape[0]
        G_mat, key = randn((m, n), dtype=dtype, key=key)
        self.G = AbsMatOp(G_mat)

        self.x, key = randn((32,), dtype=dtype, key=key)

        self.scalar = 3.141


@pytest.fixture
def testobj(request):
    yield ConvolveTestObj()


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
def test_scalar_left(testobj, operator):
    A = operator(testobj.A, testobj.scalar)
    x = testobj.x
    B = Convolve(input_shape=(32,), h=operator(testobj.psf_A, testobj.scalar))
    np.testing.assert_allclose(A @ x, B @ x, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
def test_scalar_right(testobj, operator):
    if operator == op.truediv:
        pytest.xfail("scalar / LinearOperator is not supported")
    A = operator(testobj.scalar, testobj.A)
    x = testobj.x
    B = Convolve(input_shape=(32,), h=operator(testobj.scalar, testobj.psf_A))
    np.testing.assert_allclose(A @ x, B @ x, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_convolve_add_sub(testobj, operator):
    A = testobj.A
    B = testobj.B
    C = testobj.C
    x = testobj.x

    # Two operators of same size
    AB = operator(A, B)
    ABx = AB @ x
    AxBx = operator(A @ x, B @ x)
    np.testing.assert_allclose(ABx, AxBx, rtol=5e-5)

    # Two operators of different size
    with pytest.raises(ValueError):
        operator(A, C)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_add_sub_different_mode(testobj, operator):
    # These tests get caught inside of the _wrap_add_sub input/output shape checks,
    # not the explicit mode check inside of the wrapped __add__ method
    B_same = Convolve(input_shape=(32,), h=testobj.psf_B, mode="same")
    with pytest.raises(ValueError):
        operator(testobj.A, B_same)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_add_sum_generic_linop(testobj, operator):
    # Combine a AbsMatOp and Convolve, get a generic LinearOperator
    AG = operator(testobj.A, testobj.G)
    assert isinstance(AG, LinearOperator)

    # Check evaluation
    a = AG @ testobj.x
    b = operator(testobj.A @ testobj.x, testobj.G @ testobj.x)
    np.testing.assert_allclose(a, b, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_add_sum_conv(testobj, operator):
    # Combine a AbsMatOp and Convolve, get a generic LinearOperator
    AA = operator(testobj.A, testobj.A)
    assert isinstance(AA, Convolve)

    # Check evaluation
    a = AA @ testobj.x
    b = operator(testobj.A @ testobj.x, testobj.A @ testobj.x)
    np.testing.assert_allclose(a, b, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
def test_mul_div_generic_linop(testobj, operator):
    # not defined between Convolve and AbsMatOp
    with pytest.raises(TypeError):
        operator(testobj.A, testobj.G)


def test_invalid_mode(testobj):
    # mode that doesn't exist
    with pytest.raises(ValueError):
        Convolve(input_shape=(32,), h=testobj.psf_A, mode="foo")


def test_dimension_mismatch(testobj):
    with pytest.raises(ValueError):
        # 2-dim input shape, 1-dim filter
        Convolve(input_shape=(32, 32), h=testobj.psf_A)


def test_ndarray_h():
    # Used to restore the warnings after the context is used
    with warnings.catch_warnings():
        # Ignores warning raised by ensure_on_device
        warnings.filterwarnings(action="ignore", category=UserWarning)

        h = np.random.randn(3, 3).astype(np.float32)
        A = Convolve(input_shape=(32, 32), h=h)
        assert isinstance(A.h, jax.interpreters.xla.DeviceArray)


class TestConvolveByX:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(32,), (32, 48)])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("jit", [False, True])
    def test_eval(self, input_shape, input_dtype, mode, jit):
        ndim = len(input_shape)

        x_shape = (3, 4)[:ndim]

        h, key = randn(input_shape, dtype=input_dtype, key=self.key)
        x, key = randn(x_shape, dtype=input_dtype, key=key)

        A = ConvolveByX(x=x, input_shape=input_shape, input_dtype=input_dtype, mode=mode, jit=jit)
        Ax = A @ h
        y = signal.convolve(x, h, mode=mode)
        np.testing.assert_allclose(Ax.ravel(), y.ravel(), rtol=1e-4)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(32,), (32, 48)])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("jit", [False, True])
    def test_adjoint(self, input_shape, mode, jit, input_dtype):

        ndim = len(input_shape)
        x_shape = (3, 4)[:ndim]
        x, key = randn(input_shape, dtype=input_dtype, key=self.key)
        x, key = randn(x_shape, dtype=input_dtype, key=key)

        A = ConvolveByX(x=x, input_shape=input_shape, input_dtype=input_dtype, mode=mode, jit=jit)

        adjoint_test(A, self.key)


class ConvolveByXTestObj:
    def __init__(self):
        dtype = np.float32
        key = jax.random.PRNGKey(12345)

        self.x_A, key = randn((3,), dtype=dtype, key=key)
        self.x_B, key = randn((3,), dtype=dtype, key=key)
        self.x_C, key = randn((5,), dtype=dtype, key=key)

        self.A = ConvolveByX(input_shape=(32,), x=self.x_A)
        self.B = ConvolveByX(input_shape=(32,), x=self.x_B)
        self.C = ConvolveByX(input_shape=(32,), x=self.x_C)

        # Matrix for a 'generic linop'
        m = self.A.output_shape[0]
        n = self.A.input_shape[0]
        G_mat, key = randn((m, n), dtype=dtype, key=key)
        self.G = AbsMatOp(G_mat)

        self.h, key = randn((32,), dtype=dtype, key=key)

        self.scalar = 3.141


@pytest.fixture
def cbx_testobj(request):
    yield ConvolveByXTestObj()


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
def test_cbx_scalar_left(cbx_testobj, operator):
    A = operator(cbx_testobj.A, cbx_testobj.scalar)
    h = cbx_testobj.h
    B = ConvolveByX(input_shape=(32,), x=operator(cbx_testobj.x_A, cbx_testobj.scalar))
    np.testing.assert_allclose(A @ h, B @ h, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
def test_cbx_scalar_right(cbx_testobj, operator):
    if operator == op.truediv:
        pytest.xfail("scalar / LinearOperator is not supported")
    A = operator(cbx_testobj.scalar, cbx_testobj.A)
    h = cbx_testobj.h
    B = ConvolveByX(input_shape=(32,), x=operator(cbx_testobj.scalar, cbx_testobj.x_A))
    np.testing.assert_allclose(A @ h, B @ h, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_convolve_add_sub(cbx_testobj, operator):
    A = cbx_testobj.A
    B = cbx_testobj.B
    C = cbx_testobj.C
    h = cbx_testobj.h

    # Two operators of same size
    AB = operator(A, B)
    ABh = AB @ h
    AfiltBh = operator(A @ h, B @ h)
    np.testing.assert_allclose(ABh, AfiltBh, rtol=5e-5)

    # Two operators of different size
    with pytest.raises(ValueError):
        operator(A, C)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_add_sub_different_mode(cbx_testobj, operator):
    # These tests get caught inside of the _wrap_add_sub input/output shape checks,
    # not the explicit mode check inside of the wrapped __add__ method
    B_same = ConvolveByX(input_shape=(32,), x=cbx_testobj.x_B, mode="same")
    with pytest.raises(ValueError):
        operator(cbx_testobj.A, B_same)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_add_sum_generic_linop(cbx_testobj, operator):
    # Combine a AbsMatOp and ConvolveByX, get a generic LinearOperator
    AG = operator(cbx_testobj.A, cbx_testobj.G)
    assert isinstance(AG, LinearOperator)

    # Check evaluation
    a = AG @ cbx_testobj.h
    b = operator(cbx_testobj.A @ cbx_testobj.h, cbx_testobj.G @ cbx_testobj.h)
    np.testing.assert_allclose(a, b, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_add_sum_conv(cbx_testobj, operator):
    # Combine a AbsMatOp and ConvolveByX, get a generic LinearOperator
    AA = operator(cbx_testobj.A, cbx_testobj.A)
    assert isinstance(AA, ConvolveByX)

    # Check evaluation
    a = AA @ cbx_testobj.h
    b = operator(cbx_testobj.A @ cbx_testobj.h, cbx_testobj.A @ cbx_testobj.h)
    np.testing.assert_allclose(a, b, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
def test_mul_div_generic_linop(cbx_testobj, operator):
    # not defined between ConvolveByX and AbsMatOp
    with pytest.raises(TypeError):
        operator(cbx_testobj.A, cbx_testobj.G)


def test_invalid_mode(cbx_testobj):
    # mode that doesn't exist
    with pytest.raises(ValueError):
        ConvolveByX(input_shape=(32,), x=cbx_testobj.x_A, mode="foo")


def test_dimension_mismatch(cbx_testobj):
    with pytest.raises(ValueError):
        # 2-dim input shape, 1-dim xer
        ConvolveByX(input_shape=(32, 32), x=cbx_testobj.x_A)


def test_ndarray_x():
    # Used to restore the warnings after the context is used
    with warnings.catch_warnings():
        # Ignores warning raised by ensure_on_device
        warnings.filterwarnings(action="ignore", category=UserWarning)

        x = np.random.randn(3, 3).astype(np.float32)
        A = ConvolveByX(input_shape=(32, 32), x=x)
        assert isinstance(A.x, jax.interpreters.xla.DeviceArray)
