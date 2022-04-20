import itertools
import operator as op

import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.numpy import BlockArray
from scico.random import randn

math_ops = [op.add, op.sub, op.mul, op.truediv, op.pow]  # op.floordiv doesn't work on complex
comp_ops = [op.le, op.lt, op.ge, op.gt, op.eq]


class OperatorsTestObj:
    operators = math_ops + comp_ops

    def __init__(self, dtype):
        key = None
        scalar, key = randn(shape=(1,), dtype=dtype, key=key)
        self.scalar = scalar.item()  # convert to float

        self.a0, key = randn(shape=(2, 3), dtype=dtype, key=key)
        self.a1, key = randn(shape=(2, 3, 4), dtype=dtype, key=key)
        self.a = BlockArray((self.a0, self.a1))

        self.b0, key = randn(shape=(2, 3), dtype=dtype, key=key)
        self.b1, key = randn(shape=(2, 3, 4), dtype=dtype, key=key)
        self.b = BlockArray((self.b0, self.b1))

        self.d0, key = randn(shape=(3, 2), dtype=dtype, key=key)
        self.d1, key = randn(shape=(2, 4, 3), dtype=dtype, key=key)
        self.d = BlockArray((self.d0, self.d1))

        c0, key = randn(shape=(2, 3), dtype=dtype, key=key)
        self.c = BlockArray((c0,))

        # A flat device array with same size as self.a & self.b
        self.flat_da, key = randn(shape=self.a.size, dtype=dtype, key=key)
        self.flat_nd = np.array(self.flat_da)

        # A device array with length == self.a.num_blocks
        self.block_da, key = randn(shape=(len(self.a),), dtype=dtype, key=key)

        # block_da but as a numpy array
        self.block_nd = np.array(self.block_da)

        self.key = key


@pytest.fixture(scope="module", params=[np.float32, np.complex64])
def test_operator_obj(request):
    yield OperatorsTestObj(request.param)


# Operations between a blockarray and scalar
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_operator_left(test_operator_obj, operator):
    scalar = test_operator_obj.scalar
    a = test_operator_obj.a
    x = operator(scalar, a)
    y = BlockArray(operator(scalar, a_i) for a_i in a)
    snp.testing.assert_allclose(x, y)


@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_operator_right(test_operator_obj, operator):
    scalar = test_operator_obj.scalar
    a = test_operator_obj.a
    x = operator(a, scalar)
    y = BlockArray(operator(a_i, scalar) for a_i in a)
    snp.testing.assert_allclose(x, y)


# Operations between two blockarrays of same size
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ba_ba_operator(test_operator_obj, operator):
    a = test_operator_obj.a
    b = test_operator_obj.b
    x = operator(a, b)
    y = BlockArray(operator(a_i, b_i) for a_i, b_i in zip(a, b))
    snp.testing.assert_allclose(x, y)


# Testing the @ interface for blockarrays of same size, and a blockarray and flattened ndarray/devicearray
def test_ba_ba_matmul(test_operator_obj):
    a = test_operator_obj.a
    b = test_operator_obj.d
    c = test_operator_obj.c

    a0 = test_operator_obj.a0
    a1 = test_operator_obj.a1
    d0 = test_operator_obj.d0
    d1 = test_operator_obj.d1

    x = a @ b

    y = BlockArray([a0 @ d0, a1 @ d1])
    assert x.shape == y.shape
    snp.testing.assert_allclose(x, y)

    with pytest.raises(TypeError):
        z = a @ c


def test_conj(test_operator_obj):
    a = test_operator_obj.a
    ac = a.conj()

    assert a.shape == ac.shape
    snp.testing.assert_allclose(BlockArray(a_i.conj() for a_i in a), ac)


def test_real(test_operator_obj):
    a = test_operator_obj.a
    ac = a.real

    snp.testing.assert_allclose(BlockArray(a_i.real for a_i in a), ac)


def test_imag(test_operator_obj):
    a = test_operator_obj.a
    ac = a.imag

    snp.testing.assert_allclose(BlockArray(a_i.imag for a_i in a), ac)


def test_ndim(test_operator_obj):
    assert test_operator_obj.a.ndim == (2, 3)
    assert test_operator_obj.c.ndim == (2,)


def test_getitem(test_operator_obj):
    # Make a length-4 blockarray
    a0 = test_operator_obj.a0
    a1 = test_operator_obj.a1
    b0 = test_operator_obj.b0
    b1 = test_operator_obj.b1
    x = BlockArray([a0, a1, b0, b1])

    # Positive indexing
    np.testing.assert_allclose(x[0], a0)
    np.testing.assert_allclose(x[1], a1)
    np.testing.assert_allclose(x[2], b0)
    np.testing.assert_allclose(x[3], b1)

    # Negative indexing
    np.testing.assert_allclose(x[-4], a0)
    np.testing.assert_allclose(x[-3], a1)
    np.testing.assert_allclose(x[-2], b0)
    np.testing.assert_allclose(x[-1], b1)


def test_split(test_operator_obj):
    a = test_operator_obj.a
    np.testing.assert_allclose(a[0], test_operator_obj.a0)
    np.testing.assert_allclose(a[1], test_operator_obj.a1)


@pytest.mark.skip()
# currently creation is exactly like a tuple,
# so BlockArray(np.jnp.zeros((3,6))) makes a block array
# with 3 length-6 blocks
# TODO replace with test of new behavior
def test_blockarray_from_one_array():
    with pytest.raises(TypeError):
        BlockArray(np.random.randn(32, 32))


@pytest.mark.parametrize("axis", [None, 1])
@pytest.mark.parametrize("keepdims", [True, False])
def test_sum_method(test_operator_obj, axis, keepdims):
    a = test_operator_obj.a

    method_result = a.sum(axis=axis, keepdims=keepdims)
    snp_result = snp.sum(a, axis=axis, keepdims=keepdims)

    snp.testing.assert_allclose(method_result, snp_result)


@pytest.mark.parametrize("operator", [snp.dot, snp.matmul])
def test_ba_ba_dot(test_operator_obj, operator):
    a = test_operator_obj.a
    d = test_operator_obj.d
    a0 = test_operator_obj.a0
    a1 = test_operator_obj.a1
    d0 = test_operator_obj.d0
    d1 = test_operator_obj.d1

    x = operator(a, d)
    y = BlockArray([operator(a0, d0), operator(a1, d1)])
    snp.testing.assert_allclose(x, y)


###############################################################################
# Reduction tests
###############################################################################
reduction_funcs = [
    snp.sum,
    snp.linalg.norm,
]

real_reduction_funcs = []


class BlockArrayReductionObj:
    def __init__(self, dtype):
        key = None

        a0, key = randn(shape=(2, 3), dtype=dtype, key=key)
        a1, key = randn(shape=(2, 3, 4), dtype=dtype, key=key)
        b0, key = randn(shape=(2, 3), dtype=dtype, key=key)
        b1, key = randn(shape=(2, 3), dtype=dtype, key=key)
        c0, key = randn(shape=(2, 3), dtype=dtype, key=key)
        c1, key = randn(shape=(3,), dtype=dtype, key=key)

        self.a = BlockArray((a0, a1))
        self.b = BlockArray((b0, b1))
        self.c = BlockArray((c0, c1))


@pytest.fixture(scope="module")  # so that random objects are cached
def reduction_obj(request):
    yield BlockArrayReductionObj(request.param)


REDUCTION_PARAMS = dict(
    argnames="reduction_obj, func",
    argvalues=(
        list(zip(itertools.repeat(np.float32), reduction_funcs))
        + list(zip(itertools.repeat(np.complex64), reduction_funcs))
        + list(zip(itertools.repeat(np.float32), real_reduction_funcs))
    ),
    indirect=["reduction_obj"],
)


@pytest.mark.parametrize(**REDUCTION_PARAMS)
def test_reduce(reduction_obj, func):
    x = func(reduction_obj.a)
    x_jit = jax.jit(func)(reduction_obj.a)
    y = func(snp.concatenate(snp.ravel(reduction_obj.a)))
    np.testing.assert_allclose(x, x_jit, rtol=1e-6)  # test jitted function
    np.testing.assert_allclose(x, y, rtol=1e-6)  # test for correctness


@pytest.mark.parametrize(**REDUCTION_PARAMS)
@pytest.mark.parametrize("axis", (0, 1))
def test_reduce_axis(reduction_obj, func, axis):
    f = lambda x: func(x, axis=axis)
    x = f(reduction_obj.a)
    x_jit = jax.jit(f)(reduction_obj.a)

    snp.testing.assert_allclose(x, x_jit, rtol=1e-4)  # test jitted function

    # test for correctness
    y0 = func(reduction_obj.a[0], axis=axis)
    y1 = func(reduction_obj.a[1], axis=axis)
    y = BlockArray((y0, y1))
    snp.testing.assert_allclose(x, y)


@pytest.mark.parametrize(**REDUCTION_PARAMS)
def test_reduce_singleton(reduction_obj, func):
    # Case where one block is reduced to a singleton
    f = lambda x: func(x, axis=0)
    x = f(reduction_obj.c)
    x_jit = jax.jit(f)(reduction_obj.c)

    snp.testing.assert_allclose(x, x_jit, rtol=1e-4)  # test jitted function

    y0 = func(reduction_obj.c[0], axis=0)
    y1 = func(reduction_obj.c[1], axis=0)[None]  # Ensure size (1,)
    y = BlockArray((y0, y1))
    snp.testing.assert_allclose(x, y)


class TestCreators:
    def setup_method(self, method):
        np.random.seed(12345)
        self.a_shape = (2, 3)
        self.b_shape = (2, 4, 3)
        self.c_shape = (1,)
        self.shape = (self.a_shape, self.b_shape, self.c_shape)
        self.size = np.prod(self.a_shape) + np.prod(self.b_shape) + np.prod(self.c_shape)

    def test_zeros(self):
        x = snp.zeros(self.shape, dtype=np.float32)
        assert x.shape == self.shape
        assert snp.all(x == 0)

    def test_empty(self):
        x = snp.empty(self.shape, dtype=np.float32)
        assert x.shape == self.shape
        assert snp.all(x == 0)

    def test_ones(self):
        x = snp.ones(self.shape, dtype=np.float32)
        assert x.shape == self.shape
        assert snp.all(x == 1)

    def test_full(self):
        fill_value = np.float32(np.random.randn())
        x = snp.full(self.shape, fill_value=fill_value, dtype=np.float32)
        assert x.shape == self.shape
        assert x.dtype == np.float32
        assert snp.all(x == fill_value)

    def test_full_nodtype(self):
        fill_value = np.float32(np.random.randn())
        x = snp.full(self.shape, fill_value=fill_value, dtype=None)
        assert x.shape == self.shape
        assert x.dtype == fill_value.dtype
        assert snp.all(x == fill_value)


@pytest.mark.skip
# indexing now works just like a list of DeviceArrays:
# x[1] = x[1].at[:].set(0)
# TODO: some of these are new syntax?
class TestBlockArrayIndex:
    def setup_method(self):
        key = None

        self.A, key = randn(shape=((4, 4), (3,)), key=key)
        self.B, key = randn(shape=((3, 3), (4, 2, 3)), key=key)
        self.C, key = randn(shape=((3, 3), (4, 2), (4, 4)), key=key)

    def test_set_block(self):
        # Test assignment of an entire block
        A2 = self.A[0].at[:].set(1)
        np.testing.assert_allclose(A2[0], snp.ones_like(A2[0]), rtol=5e-5)
        np.testing.assert_allclose(A2[1], A2[1], rtol=5e-5)

    def test_set(self):
        # Test assignment using (bkidx, idx) format
        A2 = self.A[0].at[2:, :-2].set(1.45)
        tmp = A2[2:, :-2]
        np.testing.assert_allclose(A2[0][2:, :-2], 1.45 * snp.ones_like(tmp), rtol=5e-5)
        np.testing.assert_allclose(A2[1].full_ravel(), A2[1], rtol=5e-5)

    def test_add(self):
        A2 = self.A.at[0, 2:, :-2].add(1.45)
        tmp = np.array(self.A[0])
        tmp[2:, :-2] += 1.45
        y = BlockArray([tmp, self.A[1]])
        np.testing.assert_allclose(A2.full_ravel(), y.full_ravel(), rtol=5e-5)

        D2 = self.D.at[1].add(1.45)
        y = BlockArray([self.D[0], self.D[1] + 1.45])
        np.testing.assert_allclose(D2.full_ravel(), y.full_ravel(), rtol=5e-5)

    def test_multiply(self):
        A2 = self.A.at[0, 2:, :-2].multiply(1.45)
        tmp = np.array(self.A[0])
        tmp[2:, :-2] *= 1.45
        y = BlockArray([tmp, self.A[1]])
        np.testing.assert_allclose(A2.full_ravel(), y.full_ravel(), rtol=5e-5)

        D2 = self.D.at[1].multiply(1.45)
        y = BlockArray([self.D[0], self.D[1] * 1.45])
        np.testing.assert_allclose(D2.full_ravel(), y.full_ravel(), rtol=5e-5)

    def test_divide(self):
        A2 = self.A.at[0, 2:, :-2].divide(1.45)
        tmp = np.array(self.A[0])
        tmp[2:, :-2] /= 1.45
        y = BlockArray([tmp, self.A[1]])
        np.testing.assert_allclose(A2.full_ravel(), y.full_ravel(), rtol=5e-5)

        D2 = self.D.at[1].divide(1.45)
        y = BlockArray([self.D[0], self.D[1] / 1.45])
        np.testing.assert_allclose(D2.full_ravel(), y.full_ravel(), rtol=5e-5)

    def test_power(self):
        A2 = self.A.at[0, 2:, :-2].power(2)
        tmp = np.array(self.A[0])
        tmp[2:, :-2] **= 2
        y = BlockArray([tmp, self.A[1]])
        np.testing.assert_allclose(A2.full_ravel(), y.full_ravel(), rtol=5e-5)

        D2 = self.D.at[1].power(1.45)
        y = BlockArray([self.D[0], self.D[1] ** 1.45])
        np.testing.assert_allclose(D2.full_ravel(), y.full_ravel(), rtol=5e-5)

    def test_set_slice(self):
        with pytest.raises(TypeError):
            C2 = self.C.at[::2, 0].set(0)
