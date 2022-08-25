import numpy as np

import scico.numpy as snp
from scico import linop
from scico.random import randn


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
