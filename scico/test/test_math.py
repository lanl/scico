import numpy as np

import scico.numpy as snp
from scico.math import (
    complex_dtype,
    is_complex_dtype,
    is_real_dtype,
    real_dtype,
    rel_res,
    safe_divide,
)
from scico.random import randn


def test_safe_divide_array():
    x, key = randn((4,), dtype=np.float32)
    y, key = randn(x.shape, dtype=np.float32, key=key)
    y = y.at[0].set(0)

    res = safe_divide(x, y)

    assert res[0] == 0
    idx = y != 0
    np.testing.assert_allclose(res[idx], x[idx] / y[idx])


def test_safe_divide_blockarray():
    x, key = randn(((3, 3), (4,)), dtype=np.float32)

    y, key = randn(x.shape, dtype=np.float32, key=key)
    y = y.at[1].set(0 * y[1])

    res = safe_divide(x, y)

    assert snp.all(res[1] == 0.0)
    np.testing.assert_allclose(res[0], x[0] / y[0])


def test_rel_res():
    A = snp.array([[2, -1], [1, 0], [-1, 1]], dtype=snp.float32)
    x = snp.array([[3], [-2]], dtype=snp.float32)
    Ax = snp.matmul(A, x)
    b = snp.array([[8], [3], [-5]], dtype=snp.float32)
    assert 0.0 == rel_res(Ax, b)

    A = snp.array([[2, -1], [1, 0], [-1, 1]], dtype=snp.float32)
    x = snp.array([[0], [0]], dtype=snp.float32)
    Ax = snp.matmul(A, x)
    b = snp.array([[0], [0], [0]], dtype=snp.float32)
    assert 0.0 == rel_res(Ax, b)


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
