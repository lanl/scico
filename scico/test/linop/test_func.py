import numpy as np

import pytest

import scico.numpy as snp
from scico import linop
from scico.random import randn
from scico.test.linop.test_linop import adjoint_test


def test_transpose():
    shape = (1, 2, 3, 4)
    perm = (1, 0, 3, 2)
    x, _ = randn(shape)
    H = linop.Transpose(shape, perm)
    np.testing.assert_array_equal(H @ x, x.transpose(perm))

    # transpose transpose is transpose inverse
    np.testing.assert_array_equal(H.T @ H @ x, x)


def test_reshape():
    shape = (1, 2, 3, 4)
    newshape = (2, 12)
    x, _ = randn(shape)
    H = linop.Reshape(shape, newshape)
    np.testing.assert_array_equal(H @ x, x.reshape(newshape))

    # reshape reshape is reshape inverse
    np.testing.assert_array_equal(H.T @ H @ x, x)


def test_pad():
    shape = (2, 3, 4)
    pad = 1
    x, _ = randn(shape)
    H = linop.Pad(shape, pad)

    pad_shape = tuple(n + 2 * pad for n in shape)
    y = snp.zeros(pad_shape)
    y = y.at[pad:-pad, pad:-pad, pad:-pad].set(x)
    np.testing.assert_array_equal(H @ x, y)

    # pad transpose is crop
    y, _ = randn(pad_shape)
    np.testing.assert_array_equal(H.T @ y, y[pad:-pad, pad:-pad, pad:-pad])


def test_crop():
    shape = (7, 9)
    crop = (1, 2)
    x, _ = randn(shape)
    H = linop.Crop(crop, shape)

    y = x[crop[0] : -crop[1], crop[0] : -crop[1]]
    np.testing.assert_array_equal(H @ x, y)


@pytest.mark.parametrize("pad", [1, (1, 2), ((1, 0), (0, 1)), ((1, 1), (2, 2))])
def test_crop_pad_adjoint(pad):
    shape = (9, 10)
    H = linop.Pad(shape, pad)
    G = linop.Crop(pad, H.output_shape)
    assert linop.valid_adjoint(H, G, eps=1e-6)


class SliceTestObj:
    def __init__(self, dtype):
        self.x = snp.zeros((4, 5, 6, 7), dtype=dtype)


@pytest.fixture(scope="module", params=[np.float32, np.complex64])
def slicetestobj(request):
    yield SliceTestObj(request.param)


slice_examples = [
    np.s_[1:],
    np.s_[:, 2:],
    np.s_[..., 3:],
    np.s_[1:, :-3],
    np.s_[1:, :, :3],
    np.s_[1:, ..., 2:],
]


@pytest.mark.parametrize("idx", slice_examples)
def test_slice_eval(slicetestobj, idx):
    x = slicetestobj.x
    A = linop.Slice(idx=idx, input_shape=x.shape, input_dtype=x.dtype)
    assert (A @ x).shape == x[idx].shape


@pytest.mark.parametrize("idx", slice_examples)
def test_slice_adj(slicetestobj, idx):
    x = slicetestobj.x
    A = linop.Slice(idx=idx, input_shape=x.shape, input_dtype=x.dtype)
    adjoint_test(A)


block_slice_examples = [
    1,
    np.s_[0:1],
    np.s_[:1],
]


@pytest.mark.parametrize("idx", block_slice_examples)
def test_slice_blockarray(idx):
    x = snp.BlockArray((snp.zeros((3, 4)), snp.ones((3, 4, 5, 6))))
    A = linop.Slice(idx=idx, input_shape=x.shape, input_dtype=x.dtype)
    assert (A @ x).shape == x[idx].shape
