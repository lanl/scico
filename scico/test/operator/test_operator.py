import operator as op

import numpy as np

from jax import config

import pytest

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)

import jax

import scico.numpy as snp
from scico.operator import Abs, Angle, Exp, Operator, operator_from_function
from scico.random import randn

SCALARS = (2, 1e0, snp.array(1.0))


class AbsOperator(Operator):
    def _eval(self, x):
        return snp.sum(snp.abs(x))


class SquareOperator(Operator):
    def _eval(self, x):
        return x**2


class SumSquareOperator(Operator):
    def _eval(self, x):
        return snp.sum(x**2)


class OperatorTestObj:
    def __init__(self, dtype):
        M, N = (32, 64)
        key = jax.random.PRNGKey(12345)
        self.dtype = dtype

        self.A = AbsOperator(input_shape=(N,), input_dtype=dtype)
        self.B = SquareOperator(input_shape=(N,), input_dtype=dtype)
        self.S = SumSquareOperator(input_shape=(N,), input_dtype=dtype)

        self.mat = randn(self.A.input_shape, dtype=dtype, key=key)
        self.x, key = randn((N,), dtype=dtype, key=key)

        self.z, key = randn((2 * N,), dtype=dtype, key=key)


@pytest.fixture(scope="module", params=[np.float32, np.float64, np.complex64, np.complex128])
def testobj(request):
    yield OperatorTestObj(request.param)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_binary_op(testobj, operator):
    # Our AbsOperator class does not override the __add__, etc
    # so AbsOperator + AbsMatOp -> Operator

    x = testobj.x
    # Composite operator
    comp_op = operator(testobj.A, testobj.S)

    # evaluate Operators separately, then add/sub
    res = operator(testobj.A(x), testobj.S(x))

    assert comp_op.output_dtype == res.dtype
    np.testing.assert_allclose(comp_op(x), res, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_binary_op_same(testobj, operator):
    x = testobj.x
    # Composite operator
    comp_op = operator(testobj.A, testobj.A)

    # evaluate Operators separately, then add/sub
    res = operator(testobj.A(x), testobj.A(x))

    assert isinstance(comp_op, Operator)
    assert comp_op.output_dtype == res.dtype
    np.testing.assert_allclose(comp_op(x), res, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
@pytest.mark.parametrize("scalar", SCALARS)
def test_scalar_left(testobj, operator, scalar):
    x = testobj.x
    comp_op = operator(testobj.A, scalar)
    res = operator(testobj.A(x), scalar)
    assert comp_op.output_dtype == res.dtype
    np.testing.assert_allclose(comp_op(x), res, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
@pytest.mark.parametrize("scalar", SCALARS)
def test_scalar_right(testobj, operator, scalar):
    if operator == op.truediv:
        pytest.xfail("scalar / Operator is not supported")
    x = testobj.x
    comp_op = operator(scalar, testobj.A)
    res = operator(scalar, testobj.A(x))
    assert comp_op.output_dtype == res.dtype
    np.testing.assert_allclose(comp_op(x), res, rtol=5e-5)


def test_negation(testobj):
    x = testobj.x
    comp_op = -testobj.A
    res = -(testobj.A(x))
    assert comp_op.input_dtype == testobj.A.input_dtype
    np.testing.assert_allclose(comp_op(x), res, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_invalid_add_sub_array(testobj, operator):
    # Try to add or subtract an ndarray with Operator
    with pytest.raises(TypeError):
        operator(testobj.A, testobj.mat)


@pytest.mark.parametrize("operator", [op.add, op.sub])
def test_invalid_add_sub_scalar(testobj, operator):
    # Try to add or subtract a scalar with AbsMatOp
    with pytest.raises(TypeError):
        operator(1.0, testobj.A)


def test_call_operator_operator(testobj):
    x = testobj.x
    A = testobj.A
    B = testobj.B
    np.testing.assert_allclose(A(B)(x), A(B(x)))

    with pytest.raises(ValueError):
        # incompatible shapes
        A(testobj.S)


def test_shape_call_vec(testobj):
    # evaluate operator on an array of incompatible size
    with pytest.raises(ValueError):
        testobj.A(testobj.z)


def test_scale_vmap(testobj):
    A = testobj.A
    x = testobj.x

    def foo(c):
        return (c * A)(x)

    c_list = [1.0, 2.0, 3.0]
    non_vmap = np.array([foo(c) for c in c_list])
    vmapped = jax.vmap(foo)(snp.array(c_list))
    np.testing.assert_allclose(non_vmap, vmapped)


def test_scale_pmap(testobj):
    A = testobj.A
    x = testobj.x

    def foo(c):
        return (c * A)(x)

    c_list = np.random.randn(jax.device_count())
    non_pmap = np.array([foo(c) for c in c_list])
    pmapped = jax.pmap(foo)(c_list)
    np.testing.assert_allclose(non_pmap, pmapped, rtol=1e-6)


def test_freeze_3arg():
    A = Operator(
        input_shape=((1, 3, 4), (2, 1, 4), (2, 3, 1)), eval_fn=lambda x: x[0] * x[1] * x[2]
    )

    a, _ = randn((1, 3, 4))
    b, _ = randn((2, 1, 4))
    c, _ = randn((2, 3, 1))

    x = snp.blockarray([a, b, c])
    Abc = A.freeze(0, a)  # A as a function of b, c
    Aac = A.freeze(1, b)  # A as a function of a, c
    Aab = A.freeze(2, c)  # A as a function of a, b

    assert Abc.input_shape == ((2, 1, 4), (2, 3, 1))
    assert Aac.input_shape == ((1, 3, 4), (2, 3, 1))
    assert Aab.input_shape == ((1, 3, 4), (2, 1, 4))

    bc = snp.blockarray([b, c])
    ac = snp.blockarray([a, c])
    ab = snp.blockarray([a, b])
    np.testing.assert_allclose(A(x), Abc(bc), rtol=5e-4)
    np.testing.assert_allclose(A(x), Aac(ac), rtol=5e-4)
    np.testing.assert_allclose(A(x), Aab(ab), rtol=5e-4)


def test_freeze_2arg():
    A = Operator(input_shape=((1, 3, 4), (2, 1, 4)), eval_fn=lambda x: x[0] * x[1])

    a, _ = randn((1, 3, 4))
    b, _ = randn((2, 1, 4))

    x = snp.blockarray([a, b])
    Ab = A.freeze(0, a)  # A as a function of 'b' only
    Aa = A.freeze(1, b)  # A as a function of 'a' only

    assert Ab.input_shape == (2, 1, 4)
    assert Aa.input_shape == (1, 3, 4)

    np.testing.assert_allclose(A(x), Ab(b), rtol=5e-4)
    np.testing.assert_allclose(A(x), Aa(a), rtol=5e-4)


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("op_fn", [(Abs, snp.abs), (Angle, snp.angle), (Exp, snp.exp)])
def test_func_op(op_fn, dtype):
    op = op_fn[0]
    fn = op_fn[1]
    shape = (2, 3)
    x, _ = randn(shape, dtype=dtype)
    H = op(input_shape=shape, input_dtype=dtype)
    np.testing.assert_array_equal(H(x), fn(x))


def test_make_func_op():
    AbsVal = operator_from_function(snp.abs, "AbsVal")
    shape = (2,)
    x, _ = randn(shape, dtype=np.float32)
    H = AbsVal(input_shape=shape, input_dtype=np.float32)
    np.testing.assert_array_equal(H(x), snp.abs(x))


def test_make_func_op_ext_init():
    AbsVal = operator_from_function(snp.abs, "AbsVal")
    shape = (2,)
    x, _ = randn(shape, dtype=np.float32)
    H = AbsVal(
        input_shape=shape, output_shape=shape, input_dtype=np.float32, output_dtype=np.float32
    )
    np.testing.assert_array_equal(H(x), snp.abs(x))


class TestJacobianProdReal:
    def setup_method(self):
        N = 7
        M = 8
        key = None
        dtype = snp.float32
        self.fmx, key = randn((M, N), key=key, dtype=dtype)
        self.F = Operator(
            (N, 1),
            output_shape=(M, 1),
            eval_fn=lambda x: self.fmx @ x,
            input_dtype=dtype,
            output_dtype=dtype,
        )
        self.u, key = randn((N, 1), key=key, dtype=dtype)
        self.v, key = randn((N, 1), key=key, dtype=dtype)
        self.w, key = randn((M, 1), key=key, dtype=dtype)

    def test_jvp(self):
        Fu, JFuv = self.F.jvp(self.u, self.v)
        np.testing.assert_allclose(Fu, self.F(self.u))
        np.testing.assert_allclose(JFuv, self.fmx @ self.v, rtol=1e-6)

    def test_vjp_conj(self):
        Fu, G = self.F.vjp(self.u, conjugate=True)
        JFTw = G(self.w)
        np.testing.assert_allclose(Fu, self.F(self.u))
        np.testing.assert_allclose(JFTw, self.fmx.T @ self.w, rtol=1e-6)

    def test_vjp_noconj(self):
        Fu, G = self.F.vjp(self.u, conjugate=False)
        JFTw = G(self.w)
        np.testing.assert_allclose(Fu, self.F(self.u))
        np.testing.assert_allclose(JFTw, self.fmx.T @ self.w, rtol=1e-6)


class TestJacobianProdComplex:
    def setup_method(self):
        N = 7
        M = 8
        key = None
        dtype = snp.complex64
        self.fmx, key = randn((M, N), key=key, dtype=dtype)
        self.F = Operator(
            (N, 1),
            output_shape=(M, 1),
            eval_fn=lambda x: self.fmx @ x,
            input_dtype=dtype,
            output_dtype=dtype,
        )
        self.u, key = randn((N, 1), key=key, dtype=dtype)
        self.v, key = randn((N, 1), key=key, dtype=dtype)
        self.w, key = randn((M, 1), key=key, dtype=dtype)

    def test_jvp(self):
        Fu, JFuv = self.F.jvp(self.u, self.v)
        np.testing.assert_allclose(Fu, self.F(self.u))
        np.testing.assert_allclose(JFuv, self.fmx @ self.v, rtol=1e-6)

    def test_vjp_conj(self):
        Fu, G = self.F.vjp(self.u, conjugate=True)
        JFTw = G(self.w)
        np.testing.assert_allclose(Fu, self.F(self.u))
        np.testing.assert_allclose(JFTw, self.fmx.T.conj() @ self.w, rtol=1e-6)

    def test_vjp_noconj(self):
        Fu, G = self.F.vjp(self.u, conjugate=False)
        JFTw = G(self.w)
        np.testing.assert_allclose(Fu, self.F(self.u))
        np.testing.assert_allclose(JFTw, self.fmx.T @ self.w, rtol=1e-6)
