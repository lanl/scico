import warnings

import numpy as np

from jax.interpreters.xla import DeviceArray

import pytest

import scico.numpy as snp
from scico.array import (
    complex_dtype,
    ensure_on_device,
    indexed_shape,
    is_complex_dtype,
    is_nested,
    is_real_dtype,
    no_nan_divide,
    parse_axes,
    real_dtype,
    slice_length,
)
from scico.blockarray import BlockArray
from scico.random import randn


def test_ensure_on_device():
    # Used to restore the warnings after the context is used
    with warnings.catch_warnings():
        # Ignores warning raised by ensure_on_device
        warnings.filterwarnings(action="ignore", category=UserWarning)

        NP = np.ones(2)
        SNP = snp.ones(2)
        BA = BlockArray.array([NP, SNP])
        NP_, SNP_, BA_ = ensure_on_device(NP, SNP, BA)

        assert isinstance(NP_, DeviceArray)

        assert isinstance(SNP_, DeviceArray)
        assert SNP.unsafe_buffer_pointer() == SNP_.unsafe_buffer_pointer()

        assert isinstance(BA_, BlockArray)
        assert BA._data.unsafe_buffer_pointer() == BA_._data.unsafe_buffer_pointer()

        np.testing.assert_raises(TypeError, ensure_on_device, [1, 1, 1])

        NP_ = ensure_on_device(NP)
        assert isinstance(NP_, DeviceArray)


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
    y = y.at[1].set(0 * y[1])

    res = no_nan_divide(x, y)

    assert snp.all(res[1] == 0.0)
    np.testing.assert_allclose(res[0], x[0] / y[0])


def test_parse_axes():
    axes = None
    np.testing.assert_raises(ValueError, parse_axes, axes)

    axes = None
    assert parse_axes(axes, np.shape([[1, 1], [1, 1]])) == [0, 1]

    axes = None
    assert parse_axes(axes, np.shape([[1, 1], [1, 1]]), default=[0]) == [0]

    axes = [1, 2]
    assert parse_axes(axes) == axes

    axes = 1
    assert parse_axes(axes) == (1,)

    axes = "axes"
    np.testing.assert_raises(ValueError, parse_axes, axes)

    axes = 2
    np.testing.assert_raises(ValueError, parse_axes, axes, np.shape([1]))

    axes = (1, 2, 2)
    np.testing.assert_raises(ValueError, parse_axes, axes)


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
    ),
)
def test_indexed_shape(shape, slc):
    x = np.zeros(shape)
    assert x[slc].shape == indexed_shape(shape, slc)


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
