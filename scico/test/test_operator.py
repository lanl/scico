import operator as op

import numpy as np

from jax.config import config

import pytest

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)

import jax

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.operator import Operator
from scico.random import randn


class AbsOperator(Operator):
    def _eval(self, x):
        return snp.sum(snp.abs(x))


class SquareOperator(Operator):
    def _eval(self, x):
        return x ** 2


class SumSquareOperator(Operator):
    def _eval(self, x):
        return snp.sum(x ** 2)


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
        scalar, key = randn((1,), dtype=dtype, key=key)
        self.scalar = scalar.copy().ravel()[0]

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
def test_scalar_left(testobj, operator):
    x = testobj.x
    comp_op = operator(testobj.A, testobj.scalar)
    res = operator(testobj.A(x), testobj.scalar)
    assert comp_op.output_dtype == res.dtype
    np.testing.assert_allclose(comp_op(x), res, rtol=5e-5)


@pytest.mark.parametrize("operator", [op.mul, op.truediv])
def test_scalar_right(testobj, operator):
    if operator == op.truediv:
        pytest.xfail("scalar / Operator is not supported")
    x = testobj.x
    comp_op = operator(testobj.scalar, testobj.A)
    res = operator(testobj.scalar, testobj.A(x))
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

    a = np.random.randn(1, 3, 4)
    b = np.random.randn(2, 1, 4)
    c = np.random.randn(2, 3, 1)

    x = BlockArray.array([a, b, c])
    Abc = A.freeze(0, a)  # A as a function of b, c
    Aac = A.freeze(1, b)  # A as a function of a, c
    Aab = A.freeze(2, c)  # A as a function of a, b

    assert Abc.input_shape == ((2, 1, 4), (2, 3, 1))
    assert Aac.input_shape == ((1, 3, 4), (2, 3, 1))
    assert Aab.input_shape == ((1, 3, 4), (2, 1, 4))

    bc = BlockArray.array([b, c])
    ac = BlockArray.array([a, c])
    ab = BlockArray.array([a, b])
    np.testing.assert_allclose(A(x), Abc(bc), rtol=5e-4)
    np.testing.assert_allclose(A(x), Aac(ac), rtol=5e-4)
    np.testing.assert_allclose(A(x), Aab(ab), rtol=5e-4)


def test_freeze_2arg():

    A = Operator(input_shape=((1, 3, 4), (2, 1, 4)), eval_fn=lambda x: x[0] * x[1])

    a = np.random.randn(1, 3, 4)
    b = np.random.randn(2, 1, 4)

    x = BlockArray.array([a, b])
    Ab = A.freeze(0, a)  # A as a function of 'b' only
    Aa = A.freeze(1, b)  # A as a function of 'a' only

    assert Ab.input_shape == (2, 1, 4)
    assert Aa.input_shape == (1, 3, 4)

    np.testing.assert_allclose(A(x), Ab(b), rtol=5e-4)
    np.testing.assert_allclose(A(x), Aa(a), rtol=5e-4)
