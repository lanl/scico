import itertools
import operator as op

import numpy as np

import jax
import jax.numpy as jnp
from jax.interpreters.xla import DeviceArray

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


# Operations between a blockarray and a flat DeviceArray
@pytest.mark.skip  # do we want to allow ((3,4), (4, 5, 6)) + (132,) ?
# argument against: numpy doesn't allow (3, 4) + (12,)
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ba_da_left(test_operator_obj, operator):
    flat_da = test_operator_obj.flat_da
    a = test_operator_obj.a
    x = operator(flat_da, a)
    y = BlockArray(operator(flat_da, a_i) for a_i in a)
    snp.testing.assert_allclose(x, y, rtol=5e-5)


@pytest.mark.skip  # see previous
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ba_da_right(test_operator_obj, operator):
    flat_da = test_operator_obj.flat_da
    a = test_operator_obj.a
    x = operator(a, flat_da)
    y = BlockArray(operator(a_i, flat_da) for a_i in a)
    np.testing.assert_allclose(x, y)


# Blockwise comparison between a BlockArray and Ndarray
@pytest.mark.skip  # do we want to allow ((3,4), (4, 5, 6)) + (2,) ?
# argument against numpy doesn't allow (3, 4) + (3,), though leading dims match
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ndarray_left(test_operator_obj, operator):
    a = test_operator_obj.a
    block_nd = test_operator_obj.block_nd

    x = operator(a, block_nd)
    y = BlockArray([operator(a[i], block_nd[i]) for i in range(len(a))])
    snp.testing.assert_allclose(x, y)


@pytest.mark.skip  # see previous
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_ndarray_right(test_operator_obj, operator):
    a = test_operator_obj.a
    block_nd = test_operator_obj.block_nd

    x = operator(block_nd, a)
    y = BlockArray([operator(block_nd[i], a[i]) for i in range(len(a))])
    snp.testing.assert_allclose(x, y)


# Blockwise comparison between a BlockArray and DeviceArray
@pytest.mark.skip  # see previous
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_devicearray_left(test_operator_obj, operator):
    a = test_operator_obj.a
    block_da = test_operator_obj.block_da

    x = operator(a, block_da)
    y = BlockArray([operator(a[i], block_da[i]) for i in range(len(a))])
    snp.testing.assert_allclose(x, y)


@pytest.mark.skip  # see previous
@pytest.mark.parametrize("operator", math_ops + comp_ops)
def test_devicearray_right(test_operator_obj, operator):
    a = test_operator_obj.a
    block_da = test_operator_obj.block_da

    x = operator(block_da, a)
    y = BlockArray([operator(block_da[i], a[i]) for i in range(len(a))])
    snp.testing.assert_allclose(x, y, atol=1e-7, rtol=0)


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


@pytest.mark.skip()
# this is indexing block dimension and internal dimensions simultaneously
# supporting it adds complexity, are we okay with just x[0][1:3] instead of x[0, 1:3]?
@pytest.mark.parametrize("index", (np.s_[0, 0], np.s_[0, 1:3], np.s_[0, :, 0:2], np.s_[0, ..., 2:]))
def test_getitem_tuple(test_operator_obj, index):
    a = test_operator_obj.a
    a0 = test_operator_obj.a0
    np.testing.assert_allclose(a[index], a0[index[1:]])


@pytest.mark.skip()
# `.blockidx` was an index into the underlying 1D array that no longer exists
def test_blockidx(test_operator_obj):
    a = test_operator_obj.a
    a0 = test_operator_obj.a0
    a1 = test_operator_obj.a1

    # use the blockidx to index the flattened data
    x0 = a.full_ravel()[a.blockidx(0)]
    x1 = a.full_ravel()[a.blockidx(1)]
    np.testing.assert_allclose(x0, a0.full_ravel())
    np.testing.assert_allclose(x1, a1.full_ravel())


def test_split(test_operator_obj):
    a = test_operator_obj.a
    np.testing.assert_allclose(a[0], test_operator_obj.a0)
    np.testing.assert_allclose(a[1], test_operator_obj.a1)


@pytest.mark.skip()
# currently creation is exactly like a tuple,
# so BlockArray(np.jnp.zeros((32,32))) makes a block array
# with 32 1d blocks
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


@pytest.mark.skip()
# previously vdot returned a scalar,
# in this proposal, it acts blockwize
def test_ba_ba_vdot(test_operator_obj):
    a = test_operator_obj.a
    d = test_operator_obj.d
    a0 = test_operator_obj.a0
    a1 = test_operator_obj.a1
    d0 = test_operator_obj.d0
    d1 = test_operator_obj.d1

    x = snp.vdot(a, d)
    y = jnp.vdot(a.full_ravel(), d.full_ravel())
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


@pytest.mark.skip
# this is reduction along the block axis, which (in the old version)
# requires all blocks to be the same shape. If you know all blocks are the same shape,
# why use a block array?
@pytest.mark.parametrize(**REDUCTION_PARAMS)
def test_reduce_axis0_old(reduction_obj, func):
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
# it no longer makes sense to make a BlockArray from a flattened array
def test_incompatible_shapes():
    # Verify that array_from_flattened raises exception when
    # len(data_ravel) != size determined by shape_tuple
    shape_tuple = ((32, 32), (16,))  # len == 1040
    data_ravel = np.ones(1030)
    with pytest.raises(ValueError):
        BlockArray.array_from_flattened(data_ravel=data_ravel, shape_tuple=shape_tuple)


class NestedTestObj:
    operators = math_ops + comp_ops

    def __init__(self, dtype):
        key = None
        scalar, key = randn(shape=(1,), dtype=dtype, key=key)
        self.scalar = scalar.item()  # convert to float

        self.a00, key = randn(shape=(2, 2, 2), dtype=dtype, key=key)
        self.a01, key = randn(shape=(3, 2, 4), dtype=dtype, key=key)
        self.a1, key = randn(shape=(2, 4), dtype=dtype, key=key)

        self.a = BlockArray(((self.a00, self.a01), self.a1))


@pytest.fixture(scope="module")
def nested_obj(request):
    yield NestedTestObj(request.param)


@pytest.mark.skip  # deeply nested shapes no longer allowed
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

    snp.testing.assert_allclose(a[0][0], a00)
    snp.testing.assert_allclose(a[0][1], a01)
    snp.testing.assert_allclose(a[1], a1)

    # basic test for block_sizes
    assert a.shape == (a[0].size, a[1].size)


NESTED_REDUCTION_PARAMS = dict(
    argnames="nested_obj, func",
    argvalues=(
        list(zip(itertools.repeat(np.float32), reduction_funcs))
        + list(zip(itertools.repeat(np.complex64), reduction_funcs))
        + list(zip(itertools.repeat(np.float32), real_reduction_funcs))
    ),
    indirect=["nested_obj"],
)


@pytest.mark.skip  # deeply nested shapes no longer allowed
@pytest.mark.parametrize(**NESTED_REDUCTION_PARAMS)
def test_nested_reduce_singleton(nested_obj, func):
    a = nested_obj.a
    x = func(a)
    y = func(a.full_ravel())
    np.testing.assert_allclose(x, y, rtol=5e-5)


@pytest.mark.skip  # deeply nested shapes no longer allowed
@pytest.mark.parametrize(**NESTED_REDUCTION_PARAMS)
def test_nested_reduce_axis1(nested_obj, func):
    a = nested_obj.a

    with pytest.raises(ValueError):
        # Blocks don't conform!
        x = func(a, axis=1)


@pytest.mark.skip  # deeply nested shapes no longer allowed
@pytest.mark.parametrize(**NESTED_REDUCTION_PARAMS)
def test_nested_reduce_axis2(nested_obj, func):
    a = nested_obj.a

    x = func(a, axis=2)
    assert x.shape == (((2, 2), (2, 4)), (2,))

    y = BlockArray((func(a[0], axis=1), func(a[1], axis=1)))
    assert x.shape == y.shape

    np.testing.assert_allclose(x.full_ravel(), y.full_ravel(), rtol=5e-5)


@pytest.mark.skip  # deeply nested shapes no longer allowed
@pytest.mark.parametrize(**NESTED_REDUCTION_PARAMS)
def test_nested_reduce_axis3(nested_obj, func):
    a = nested_obj.a

    x = func(a, axis=3)
    assert x.shape == (((2, 2), (3, 4)), (2, 4))

    y = BlockArray((func(a[0], axis=2), a[1]))
    assert x.shape == y.shape

    np.testing.assert_allclose(x.full_ravel(), y.full_ravel(), rtol=5e-5)


@pytest.mark.skip
# no longer makes sense to make BlockArray from 1d array
def test_array_from_flattened():
    x = np.random.randn(19)
    x_b = ba.BlockArray.array_from_flattened(x, shape_tuple=((4, 4), (3,)))
    assert isinstance(x_b._data, DeviceArray)


@pytest.mark.skip
# indexing now works just like a list of DeviceArrays:
# x[1] = x[1].at[:].set(0)
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
