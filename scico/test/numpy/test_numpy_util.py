import collections

import numpy as np

import pytest

import scico.numpy as snp
from scico.numpy.util import (
    array_to_namedtuple,
    complex_dtype,
    indexed_shape,
    is_blockable,
    is_collapsible,
    is_complex_dtype,
    is_nested,
    is_real_dtype,
    is_scalar_equiv,
    jax_indexed_shape,
    namedtuple_to_array,
    no_nan_divide,
    normalize_axes,
    real_dtype,
    slice_length,
    transpose_list_of_ntpl,
    transpose_ntpl_of_list,
)
from scico.random import randn


def test_ntpl_list_transpose():
    nt = collections.namedtuple("NT", ("a", "b", "c"))
    ntlist0 = [nt(0, 1, 2), nt(3, 4, 5)]
    listnt = transpose_list_of_ntpl(ntlist0)
    ntlist1 = transpose_ntpl_of_list(listnt)
    assert ntlist0[0] == ntlist1[0]
    assert ntlist0[1] == ntlist1[1]


def test_namedtuple_to_array():
    nt = collections.namedtuple("NT", ("A", "B", "C"))
    t0 = nt(0, 1, 2)
    t0a = namedtuple_to_array(t0)
    t1 = array_to_namedtuple(t0a)
    assert t0 == t1


def test_no_nan_divide_array():
    x, key = randn((4,), dtype=np.float32)
    y, key = randn(x.shape, dtype=np.float32, key=key)
    y = y.at[0].set(0)

    res = no_nan_divide(x, y)

    assert res[0] == 0
    idx = y != 0
    np.testing.assert_allclose(res[idx], x[idx] / y[idx])


def test_no_nan_divide_blockarray():
    x, key = randn(((3, 3), (4,)), dtype=np.float32)

    y, key = randn(x.shape, dtype=np.float32, key=key)
    y[1] = y[1].at[:].set(0 * y[1])

    res = no_nan_divide(x, y)

    assert snp.all(res[1] == 0.0)
    np.testing.assert_allclose(res[0], x[0] / y[0])


def test_normalize_axes():
    axes = None
    np.testing.assert_raises(ValueError, normalize_axes, axes)

    axes = None
    assert normalize_axes(axes, np.shape([[1, 1], [1, 1]])) == (0, 1)

    axes = None
    assert normalize_axes(axes, np.shape([[1, 1], [1, 1]]), default=[0]) == [0]

    axes = [1, 2]
    assert normalize_axes(axes) == axes

    axes = 1
    assert normalize_axes(axes) == (1,)

    axes = (-1,)
    assert normalize_axes(axes, shape=(1, 2)) == (1,)

    axes = (0, 2, 1)
    assert normalize_axes(axes, shape=(2, 3, 4), sort=True) == (0, 1, 2)

    axes = "axes"
    np.testing.assert_raises(ValueError, normalize_axes, axes)

    axes = 2
    np.testing.assert_raises(ValueError, normalize_axes, axes, np.shape([1]))

    axes = (1, 2, 2)
    np.testing.assert_raises(ValueError, normalize_axes, axes)


@pytest.mark.parametrize("length", (4, 5, 8, 16, 17))
@pytest.mark.parametrize("start", (None, 0, 1, 2, 3))
@pytest.mark.parametrize("stop", (None, 0, 1, 2, -2, -1))
@pytest.mark.parametrize("stride", (None, 1, 2, 3))
def test_slice_length(length, start, stop, stride):
    x = np.zeros(length)
    slc = slice(start, stop, stride)
    assert x[slc].size == slice_length(length, slc)


@pytest.mark.parametrize("length", (4, 5))
@pytest.mark.parametrize("slc", (0, 1, -4, Ellipsis))
def test_slice_length_other(length, slc):
    x = np.zeros(length)
    if isinstance(slc, int):
        assert slice_length(length, slc) is None
    else:
        assert x[slc].size == slice_length(length, slc)


@pytest.mark.parametrize("shape", ((8, 8, 1), (7, 1, 6, 5)))
@pytest.mark.parametrize(
    "slc",
    (
        np.s_[0],
        np.s_[0:5],
        np.s_[:, 0:4],
        np.s_[2:, :, :-2],
        np.s_[..., 2:],
        np.s_[..., 2:, :],
        np.s_[1:, ..., 2:],
        np.s_[np.newaxis],
        np.s_[:, np.newaxis],
        np.s_[np.newaxis, :, np.newaxis],
        np.s_[np.newaxis, ..., 0:2, :],
    ),
)
def test_indexed_shape(shape, slc):
    x = np.zeros(shape)
    assert x[slc].shape == indexed_shape(shape, slc)
    assert x[slc].shape == jax_indexed_shape(shape, slc)


def test_is_nested():
    # list
    assert is_nested([1, 2, 3]) == False

    # tuple
    assert is_nested((1, 2, 3)) == False

    # list of lists
    assert is_nested([[1, 2], [4, 5], [3]]) == True

    # list of lists + scalar
    assert is_nested([[1, 2], 3]) == True

    # list of tuple + scalar
    assert is_nested([(1, 2), 3]) == True

    # tuple of tuple + scalar
    assert is_nested(((1, 2), 3)) == True

    # tuple of lists + scalar
    assert is_nested(([1, 2], 3)) == True


def test_is_collapsible():
    shape1 = ((1, 2, 3), (1, 2, 3), (1, 3, 3))
    shape2 = ((1, 2, 3), (1, 2, 3), (1, 2, 3))
    assert not is_collapsible(shape1)
    assert is_collapsible(shape2)


def test_is_blockable():
    shape1 = ((1, 2, 3), (1, 2, 3), (1, 2, 3))
    shape2 = ((1, 2, 3), ((1, 2, 3), (1, 2, 3)))
    assert is_blockable(shape1)
    assert not is_blockable(shape2)


def test_is_real_dtype():
    assert not is_real_dtype(snp.complex64)
    assert is_real_dtype(snp.float32)


def test_is_complex_dtype():
    assert is_complex_dtype(snp.complex64)
    assert not is_complex_dtype(snp.float32)


def test_real_dtype():
    assert real_dtype(snp.complex64) == snp.float32


def test_complex_dtype():
    assert complex_dtype(snp.float32) == snp.complex64


def test_broadcast_nested_shapes():
    # unnested should work as usual
    assert snp.util.broadcast_nested_shapes((1, 3, 4, 7), (3, 1, 7)) == (1, 3, 4, 7)

    # nested + unested
    assert snp.util.broadcast_nested_shapes(((2, 3), (1, 1, 3)), (2, 3)) == ((2, 3), (1, 2, 3))

    # unested + nested
    assert snp.util.broadcast_nested_shapes((1, 1, 3), ((2, 3), (7, 3))) == ((1, 2, 3), (1, 7, 3))

    # nested + nested
    snp.util.broadcast_nested_shapes(((1, 1, 3), (1, 7, 1, 3)), ((2, 3), (7, 4, 3))) == (
        (1, 2, 3),
        (1, 7, 4, 3),
    )


def test_is_scalar_equiv():
    assert is_scalar_equiv(1e0)
    assert is_scalar_equiv(snp.array(1e0))
    assert is_scalar_equiv(snp.sum(snp.zeros(1)))
    assert not is_scalar_equiv(snp.array([1e0]))
    assert not is_scalar_equiv(snp.array([1e0, 2e0]))
