import itertools
import operator as op

import numpy as np

import jax
import jax.numpy as jnp
from jax.interpreters.xla import DeviceArray

import pytest

import scico.blockarray as ba
import scico.numpy as snp
from scico.random import randn

math_ops = [op.add, op.sub, op.mul, op.truediv, op.pow, op.floordiv]
comp_ops = [op.le, op.lt, op.ge, op.gt, op.eq]


class OperatorsTestObj:
    operators = math_ops + comp_ops

    def __init__(self, dtype):
        key = None
        scalar, key = randn(shape=(1,), dtype=dtype, key=key)
        self.scalar = scalar.copy().ravel()[0]  # convert to float

        self.a0, key = randn(shape=(3, 4), dtype=dtype, key=key)
        self.a1, key = randn(shape=(4, 5, 6), dtype=dtype, key=key)
        self.a = ba.BlockArray.array((self.a0, self.a1), dtype=dtype)

        self.b0, key = randn(shape=(3, 4), dtype=dtype, key=key)
        self.b1, key = randn(shape=(4, 5, 6), dtype=dtype, key=key)
        self.b = ba.BlockArray.array((self.b0, self.b1), dtype=dtype)

        self.d0, key = randn(shape=(4, 3), dtype=dtype, key=key)
        self.d1, key = randn(shape=(4, 6, 5), dtype=dtype, key=key)
        self.d = ba.BlockArray.array((self.d0, self.d1), dtype=dtype)

        c0, key = randn(shape=(3, 4), dtype=dtype, key=key)
        self.c = ba.BlockArray.array((c0,), dtype=dtype)

        # A flat device array with same size as self.a & self.b
        self.flat_da, key = randn(shape=(self.a.size,), dtype=dtype, key=key)
        self.flat_nd = self.flat_da.copy()

        # A device array with length == self.a.num_blocks
        self.block_da, key = randn(shape=(self.a.num_blocks,), dtype=dtype, key=key)

        # block_da but as a numpy array
        self.block_nd = self.block_da.copy()

        self.key = key


@pytest.fixture(scope="module", params=[np.float32, np.complex64])
def test_operator_obj(request):
    yield OperatorsTestObj(request.param)


# Operations between a blockarray and scalar
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_operator_left(test_operator_obj, operator):
    scalar = test_operator_obj.scalar
    a = test_operator_obj.a
    x = operator(scalar, a).ravel()
    y = operator(scalar, a.ravel())
    np.testing.assert_allclose(x, y)


@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_operator_right(test_operator_obj, operator):
    scalar = test_operator_obj.scalar
    a = test_operator_obj.a
    x = operator(a, scalar).ravel()
    y = operator(a.ravel(), scalar)
    np.testing.assert_allclose(x, y)


# Operations between a blockarray and a flat DeviceArray
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ba_da_left(test_operator_obj, operator):
    flat_da = test_operator_obj.flat_da
    a = test_operator_obj.a
    x = operator(flat_da, a).ravel()
    y = operator(flat_da, a.ravel())
    np.testing.assert_allclose(x, y, rtol=5e-5)


@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ba_da_right(test_operator_obj, operator):
    flat_da = test_operator_obj.flat_da
    a = test_operator_obj.a
    x = operator(a, flat_da).ravel()
    y = operator(a.ravel(), flat_da)
    np.testing.assert_allclose(x, y)


# Blockwise comparison between a BlockArray and Ndarray
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ndarray_left(test_operator_obj, operator):
    a = test_operator_obj.a
    block_nd = test_operator_obj.block_nd

    x = operator(a, block_nd).ravel()
    y = ba.BlockArray.array([operator(a[i], block_nd[i]) for i in range(a.num_blocks)]).ravel()
    np.testing.assert_allclose(x, y)


@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ndarray_right(test_operator_obj, operator):
    a = test_operator_obj.a
    block_nd = test_operator_obj.block_nd

    x = operator(block_nd, a).ravel()
    y = ba.BlockArray.array([operator(block_nd[i], a[i]) for i in range(a.num_blocks)]).ravel()
    np.testing.assert_allclose(x, y)


# Blockwise comparison between a BlockArray and DeviceArray
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_devicearray_left(test_operator_obj, operator):
    a = test_operator_obj.a
    block_da = test_operator_obj.block_da

    x = operator(a, block_da).ravel()
    y = ba.BlockArray.array([operator(a[i], block_da[i]) for i in range(a.num_blocks)]).ravel()
    np.testing.assert_allclose(x, y)


@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_devicearray_right(test_operator_obj, operator):
    a = test_operator_obj.a
    block_da = test_operator_obj.block_da

    x = operator(block_da, a).ravel()
    y = ba.BlockArray.array([operator(block_da[i], a[i]) for i in range(a.num_blocks)]).ravel()
    np.testing.assert_allclose(x, y)


# Operations between two blockarrays of same size
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ba_ba_operator(test_operator_obj, operator):
    a = test_operator_obj.a
    b = test_operator_obj.b
    x = operator(a, b).ravel()
    y = operator(a.ravel(), b.ravel())
    np.testing.assert_allclose(x, y)


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

    y = ba.BlockArray.array([a0 @ d0, a1 @ d1])
    assert x.shape == y.shape
    np.testing.assert_allclose(x.ravel(), y.ravel())

    with pytest.raises(TypeError):
        z = a @ c


def test_conj(test_operator_obj):
    a = test_operator_obj.a
    ac = a.conj()

    assert a.shape == ac.shape
    np.testing.assert_allclose(a.ravel().conj(), ac.ravel())


def test_real(test_operator_obj):
    a = test_operator_obj.a
    ac = a.real

    assert a.shape == ac.shape
    np.testing.assert_allclose(a.ravel().real, ac.ravel())


def test_imag(test_operator_obj):
    a = test_operator_obj.a
    ac = a.imag

    assert a.shape == ac.shape
    np.testing.assert_allclose(a.ravel().imag, ac.ravel())


def test_ndim(test_operator_obj):
    assert test_operator_obj.a.ndim == (2, 3)
    assert test_operator_obj.c.ndim == (2,)


def test_getitem(test_operator_obj):
    # Make a length-4 blockarray
    a0 = test_operator_obj.a0
    a1 = test_operator_obj.a1
    b0 = test_operator_obj.b0
    b1 = test_operator_obj.b1
    x = ba.BlockArray.array([a0, a1, b0, b1])

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


@pytest.mark.parametrize("index", (np.s_[0, 0], np.s_[0, 1:3], np.s_[0, :, 0:2], np.s_[0, ..., 2:]))
def test_getitem_tuple(test_operator_obj, index):
    a = test_operator_obj.a
    a0 = test_operator_obj.a0
    np.testing.assert_allclose(a[index], a0[index[1:]])


def test_blockidx(test_operator_obj):
    a = test_operator_obj.a
    a0 = test_operator_obj.a0
    a1 = test_operator_obj.a1

    # use the blockidx to index the flattened data
    x0 = a.ravel()[a.blockidx(0)]
    x1 = a.ravel()[a.blockidx(1)]
    np.testing.assert_allclose(x0, a0.ravel())
    np.testing.assert_allclose(x1, a1.ravel())


def test_split(test_operator_obj):
    a = test_operator_obj.a
    a_split = a.split
    np.testing.assert_allclose(a_split[0], test_operator_obj.a0)
    np.testing.assert_allclose(a_split[1], test_operator_obj.a1)


def test_blockarray_from_one_array():
    with pytest.raises(TypeError):
        ba.BlockArray.array(np.random.randn(32, 32))


@pytest.mark.parametrize("axis", [None, 1])
@pytest.mark.parametrize("keepdims", [True, False])
def test_sum_method(test_operator_obj, axis, keepdims):
    a = test_operator_obj.a

    method_result = a.sum(axis=axis, keepdims=keepdims).ravel()
    snp_result = snp.sum(a, axis=axis, keepdims=keepdims).ravel()

    assert method_result.shape == snp_result.shape
    np.testing.assert_allclose(method_result, snp_result)


def test_ba_ba_vdot(test_operator_obj):
    a = test_operator_obj.a
    d = test_operator_obj.d
    a0 = test_operator_obj.a0
    a1 = test_operator_obj.a1
    d0 = test_operator_obj.d0
    d1 = test_operator_obj.d1

    x = snp.vdot(a, d)
    y = jnp.vdot(a.ravel(), d.ravel())
    np.testing.assert_allclose(x, y)


@pytest.mark.parametrize("operator", [snp.dot, snp.matmul])
def test_ba_ba_dot(test_operator_obj, operator):
    a = test_operator_obj.a
    d = test_operator_obj.d
    a0 = test_operator_obj.a0
    a1 = test_operator_obj.a1
    d0 = test_operator_obj.d0
    d1 = test_operator_obj.d1

    x = operator(a, d)
    y = ba.BlockArray.array([operator(a0, d0), operator(a1, d1)])
    np.testing.assert_allclose(x.ravel(), y.ravel())


###############################################################################
# Reduction tests
###############################################################################
reduction_funcs = [
    snp.count_nonzero,
    snp.sum,
    snp.linalg.norm,
    snp.mean,
    snp.var,
    snp.max,
    snp.min,
    snp.amin,
    snp.amax,
    snp.all,
    snp.any,
]

real_reduction_funcs = [
    snp.median,
]


class BlockArrayReductionObj:
    def __init__(self, dtype):
        key = None

        a0, key = randn(shape=(3, 4), dtype=dtype, key=key)
        a1, key = randn(shape=(3, 5, 6), dtype=dtype, key=key)
        b0, key = randn(shape=(3, 4), dtype=dtype, key=key)
        b1, key = randn(shape=(3, 4), dtype=dtype, key=key)
        c0, key = randn(shape=(3, 4), dtype=dtype, key=key)
        c1, key = randn(shape=(3,), dtype=dtype, key=key)

        self.a = ba.BlockArray.array((a0, a1), dtype=dtype)
        self.b = ba.BlockArray.array((b0, b1), dtype=dtype)
        self.c = ba.BlockArray.array((c0, c1), dtype=dtype)


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
    y = func(reduction_obj.a.ravel())
    np.testing.assert_allclose(x, x_jit, rtol=1e-4)  # test jitted function
    np.testing.assert_allclose(x, y)  # test for correctness


@pytest.mark.parametrize(**REDUCTION_PARAMS)
def test_reduce_axis0(reduction_obj, func):
    f = lambda x: func(x, axis=0)
    x = f(reduction_obj.b)
    x_jit = jax.jit(f)(reduction_obj.b)

    np.testing.assert_allclose(x, x_jit, rtol=1e-4)  # test jitted function

    # test for correctness
    # stack into a (2, 3, 4) array, call func
    y = func(np.stack(list(reduction_obj.b)), axis=0)
    np.testing.assert_allclose(x, y)

    with pytest.raises(ValueError):
        # Reduction along axis=0 only works if all blocks are same shape
        func(reduction_obj.a, axis=0)


@pytest.mark.parametrize(**REDUCTION_PARAMS)
def test_reduce_axis1(reduction_obj, func):
    """this is _not_ duplicated from test_reduce_axis0"""
    f = lambda x: func(x, axis=1).ravel()
    x = f(reduction_obj.a)
    x_jit = jax.jit(f)(reduction_obj.a)

    np.testing.assert_allclose(x, x_jit, rtol=1e-4)  # test jitted function

    # test for correctness
    y0 = func(reduction_obj.a[0], axis=0)
    y1 = func(reduction_obj.a[1], axis=0)
    y = ba.BlockArray.array((y0, y1), dtype=reduction_obj.a[0].dtype).ravel()
    np.testing.assert_allclose(x, y)


@pytest.mark.parametrize(**REDUCTION_PARAMS)
def test_reduce_axis2(reduction_obj, func):
    """this is _not_ duplicated from test_reduce_axis0"""
    f = lambda x: func(x, axis=2).ravel()
    x = f(reduction_obj.a)
    x_jit = jax.jit(f)(reduction_obj.a)

    np.testing.assert_allclose(x, x_jit, rtol=1e-4)  # test jitted function

    y0 = func(reduction_obj.a[0], axis=1)
    y1 = func(reduction_obj.a[1], axis=1)
    y = ba.BlockArray.array((y0, y1), dtype=reduction_obj.a[0].dtype).ravel()
    np.testing.assert_allclose(x, y)


@pytest.mark.parametrize(**REDUCTION_PARAMS)
def test_reduce_axis3(reduction_obj, func):
    """this is _not_ duplicated from test_reduce_axis0"""
    f = lambda x: func(x, axis=3).ravel()
    x = f(reduction_obj.a)
    x_jit = jax.jit(f)(reduction_obj.a)

    np.testing.assert_allclose(x, x_jit, rtol=1e-4)  # test jitted function

    y0 = reduction_obj.a[0]
    y1 = func(reduction_obj.a[1], axis=2)
    y = ba.BlockArray.array((y0, y1), dtype=reduction_obj.a[0].dtype).ravel()
    np.testing.assert_allclose(x.ravel(), y)


@pytest.mark.parametrize(**REDUCTION_PARAMS)
def test_reduce_singleton(reduction_obj, func):
    # Case where a block is reduced to a singleton
    f = lambda x: func(x, axis=1).ravel()
    x = f(reduction_obj.c)
    x_jit = jax.jit(f)(reduction_obj.c)

    np.testing.assert_allclose(x, x_jit, rtol=1e-4)  # test jitted function

    y0 = func(reduction_obj.c[0], axis=0)
    y1 = func(reduction_obj.c[1], axis=0)[None]  # Ensure size (1,)
    y = ba.BlockArray.array((y0, y1), dtype=reduction_obj.a[0].dtype).ravel()
    np.testing.assert_allclose(x, y)


class TestCreators:
    def setup_method(self, method):
        np.random.seed(12345)
        self.a_shape = (3, 4)
        self.b_shape = (4, 5, 3)
        self.c_shape = (1,)
        self.shape = (self.a_shape, self.b_shape, self.c_shape)
        self.size = np.prod(self.a_shape) + np.prod(self.b_shape) + np.prod(self.c_shape)

    def test_zeros(self):
        x = ba.BlockArray.zeros(self.shape, dtype=np.float32)
        assert x.shape == self.shape
        assert snp.all(x == 0)

    def test_empty(self):
        x = ba.BlockArray.empty(self.shape, dtype=np.float32)
        assert x.shape == self.shape
        assert snp.all(x == 0)

    def test_ones(self):
        x = ba.BlockArray.ones(self.shape, dtype=np.float32)
        assert x.shape == self.shape
        assert snp.all(x == 1)

    def test_full(self):
        fill_value = np.float32(np.random.randn())
        x = ba.BlockArray.full(self.shape, fill_value=fill_value, dtype=np.float32)
        assert x.shape == self.shape
        assert x.dtype == np.float32
        assert snp.all(x == fill_value)

    def test_full_nodtype(self):
        fill_value = np.float32(np.random.randn())
        x = ba.BlockArray.full(self.shape, fill_value=fill_value, dtype=None)
        assert x.shape == self.shape
        assert x.dtype == fill_value.dtype
        assert snp.all(x == fill_value)


def test_incompatible_shapes():
    # Verify that array_from_flattened raises exception when
    # len(data_ravel) != size determined by shape_tuple
    shape_tuple = ((32, 32), (16,))  # len == 1040
    data_ravel = np.ones(1030)
    with pytest.raises(ValueError):
        ba.BlockArray.array_from_flattened(data_ravel=data_ravel, shape_tuple=shape_tuple)


class NestedTestObj:
    operators = math_ops + comp_ops

    def __init__(self, dtype):
        key = None
        scalar, key = randn(shape=(1,), dtype=dtype, key=key)
        self.scalar = scalar.copy().ravel()[0]  # convert to float

        self.a00, key = randn(shape=(2, 2, 2), dtype=dtype, key=key)
        self.a01, key = randn(shape=(3, 2, 4), dtype=dtype, key=key)
        self.a1, key = randn(shape=(2, 4), dtype=dtype, key=key)

        self.a = ba.BlockArray.array(((self.a00, self.a01), self.a1))


@pytest.fixture(scope="module")
def nested_obj(request):
    yield NestedTestObj(request.param)


@pytest.mark.parametrize("nested_obj", [np.float32, np.complex64], indirect=True)
def test_nested_shape(nested_obj):
    a = nested_obj.a

    a00 = nested_obj.a00
    a01 = nested_obj.a01
    a1 = nested_obj.a1

    assert a.shape == (((2, 2, 2), (3, 2, 4)), (2, 4))
    assert a.size == 2 * 2 * 2 + 3 * 2 * 4 + 2 * 4

    assert a[0].shape == ((2, 2, 2), (3, 2, 4))
    assert a[1].shape == (2, 4)

    np.testing.assert_allclose(a[0][0].ravel(), a00.ravel())
    np.testing.assert_allclose(a[0][1].ravel(), a01.ravel())
    np.testing.assert_allclose(a[1].ravel(), a1.ravel())

    # basic test for block_sizes
    assert ba.block_sizes(a.shape) == (a[0].size, a[1].size)


NESTED_REDUCTION_PARAMS = dict(
    argnames="nested_obj, func",
    argvalues=(
        list(zip(itertools.repeat(np.float32), reduction_funcs))
        + list(zip(itertools.repeat(np.complex64), reduction_funcs))
        + list(zip(itertools.repeat(np.float32), real_reduction_funcs))
    ),
    indirect=["nested_obj"],
)


@pytest.mark.parametrize(**NESTED_REDUCTION_PARAMS)
def test_nested_reduce_singleton(nested_obj, func):
    a = nested_obj.a
    x = func(a)
    y = func(a.ravel())
    np.testing.assert_allclose(x, y, rtol=5e-5)


@pytest.mark.parametrize(**NESTED_REDUCTION_PARAMS)
def test_nested_reduce_axis1(nested_obj, func):
    a = nested_obj.a

    with pytest.raises(ValueError):
        # Blocks don't conform!
        x = func(a, axis=1)


@pytest.mark.parametrize(**NESTED_REDUCTION_PARAMS)
def test_nested_reduce_axis2(nested_obj, func):
    a = nested_obj.a

    x = func(a, axis=2)
    assert x.shape == (((2, 2), (2, 4)), (2,))

    y = ba.BlockArray.array((func(a[0], axis=1), func(a[1], axis=1)))
    assert x.shape == y.shape

    np.testing.assert_allclose(x.ravel(), y.ravel(), rtol=5e-5)


@pytest.mark.parametrize(**NESTED_REDUCTION_PARAMS)
def test_nested_reduce_axis3(nested_obj, func):
    a = nested_obj.a

    x = func(a, axis=3)
    assert x.shape == (((2, 2), (3, 4)), (2, 4))

    y = ba.BlockArray.array((func(a[0], axis=2), a[1]))
    assert x.shape == y.shape

    np.testing.assert_allclose(x.ravel(), y.ravel(), rtol=5e-5)


def test_array_from_flattened():
    x = np.random.randn(19)
    x_b = ba.BlockArray.array_from_flattened(x, shape_tuple=((4, 4), (3,)))
    assert isinstance(x_b._data, DeviceArray)


class TestBlockArrayIndex:
    def setup_method(self):
        key = None

        self.A, key = randn(shape=((4, 4), (3,)), key=key)
        self.B, key = randn(shape=((3, 3), (4, 2, 3)), key=key)
        self.C, key = randn(shape=((3, 3), (4, 2), (4, 4)), key=key)

        # nested
        self.D, key = randn(shape=((self.A.shape, self.B.shape)), key=key)

    def test_set_block(self):
        # Test assignment of an entire block
        A2 = self.A.at[0].set(1)
        np.testing.assert_allclose(A2[0], snp.ones_like(A2[0]), rtol=5e-5)
        np.testing.assert_allclose(A2[1], A2[1], rtol=5e-5)

        D2 = self.D.at[1].set(1.45)
        np.testing.assert_allclose(D2[0].ravel(), self.D[0].ravel(), rtol=5e-5)
        np.testing.assert_allclose(
            D2[1].ravel(), 1.45 * snp.ones_like(self.D[1]).ravel(), rtol=5e-5
        )

    def test_set(self):
        # Test assignment using (bkidx, idx) format
        A2 = self.A.at[0, 2:, :-2].set(1.45)
        tmp = A2[0][2:, :-2]
        np.testing.assert_allclose(A2[0][2:, :-2], 1.45 * snp.ones_like(tmp), rtol=5e-5)
        np.testing.assert_allclose(A2[1].ravel(), A2[1], rtol=5e-5)

    def test_add(self):
        A2 = self.A.at[0, 2:, :-2].add(1.45)
        tmp = self.A[0].copy().copy()
        tmp[2:, :-2] += 1.45
        y = ba.BlockArray.array([tmp, self.A[1]])
        np.testing.assert_allclose(A2.ravel(), y.ravel(), rtol=5e-5)

        D2 = self.D.at[1].add(1.45)
        y = ba.BlockArray.array([self.D[0], self.D[1] + 1.45])
        np.testing.assert_allclose(D2.ravel(), y.ravel(), rtol=5e-5)

    def test_multiply(self):
        A2 = self.A.at[0, 2:, :-2].multiply(1.45)
        tmp = self.A[0].copy().copy()
        tmp[2:, :-2] *= 1.45
        y = ba.BlockArray.array([tmp, self.A[1]])
        np.testing.assert_allclose(A2.ravel(), y.ravel(), rtol=5e-5)

        D2 = self.D.at[1].multiply(1.45)
        y = ba.BlockArray.array([self.D[0], self.D[1] * 1.45])
        np.testing.assert_allclose(D2.ravel(), y.ravel(), rtol=5e-5)

    def test_divide(self):
        A2 = self.A.at[0, 2:, :-2].divide(1.45)
        tmp = self.A[0].copy().copy()
        tmp[2:, :-2] /= 1.45
        y = ba.BlockArray.array([tmp, self.A[1]])
        np.testing.assert_allclose(A2.ravel(), y.ravel(), rtol=5e-5)

        D2 = self.D.at[1].divide(1.45)
        y = ba.BlockArray.array([self.D[0], self.D[1] / 1.45])
        np.testing.assert_allclose(D2.ravel(), y.ravel(), rtol=5e-5)

    def test_power(self):
        A2 = self.A.at[0, 2:, :-2].power(2)
        tmp = self.A[0].copy().copy()
        tmp[2:, :-2] **= 2
        y = ba.BlockArray.array([tmp, self.A[1]])
        np.testing.assert_allclose(A2.ravel(), y.ravel(), rtol=5e-5)

        D2 = self.D.at[1].power(1.45)
        y = ba.BlockArray.array([self.D[0], self.D[1] ** 1.45])
        np.testing.assert_allclose(D2.ravel(), y.ravel(), rtol=5e-5)

    def test_set_slice(self):
        with pytest.raises(TypeError):
            C2 = self.C.at[::2, 0].set(0)
