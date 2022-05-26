import numpy as np

import pytest

import scico.numpy as snp
from scico.linop import (
    DFT,
    Diagonal,
    Identity,
    MatrixOperator,
    Slice,
    Sum,
    Transpose,
    operator_norm,
)


def norm_test(A):
    𝛼 = np.sqrt(2)
    x = A.norm()
    y = operator_norm(A, maxiter=100)
    np.testing.assert_allclose(x, y, rtol=1e-4)
    if A.input_dtype == np.float32:
        B = A.T
    else:
        B = A.H
    x = B.norm()
    y = operator_norm(B, maxiter=100)
    np.testing.assert_allclose(x, y, rtol=1e-4)
    B = 𝛼 * A
    x = B.norm()
    y = operator_norm(B, maxiter=100)
    np.testing.assert_allclose(x, y, rtol=1e-4)
    B = A * 𝛼
    x = B.norm()
    y = operator_norm(B, maxiter=100)
    np.testing.assert_allclose(x, y, rtol=1e-4)
    B = A / 𝛼
    x = B.norm()
    y = operator_norm(B, maxiter=100)
    np.testing.assert_allclose(x, y, rtol=1e-4)


@pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("input_shape", [(7,), (8, 9)])
def test_identity(input_shape, input_dtype):
    A = Identity(input_shape=input_shape, input_dtype=input_dtype)
    norm_test(A)


@pytest.mark.parametrize(
    "diag", [snp.arange(0, 8.0), snp.arange(0, 8.0) + 1j * snp.arange(8.0, 0, -1)]
)
def test_diagonal(diag):
    A = Diagonal(diag)
    norm_test(A)


@pytest.mark.parametrize(
    "matrix",
    [
        snp.arange(0, 9.0).reshape(3, 3),
        snp.arange(-5, 7.0).reshape(3, 4),
        snp.arange(0, 9.0).reshape(3, 3) + 1j * snp.arange(9.0, 0, -1).reshape(3, 3),
    ],
)
def test_matrixoperator(matrix):
    A = MatrixOperator(matrix)
    norm_test(A)


@pytest.mark.parametrize("shape", [[(7,), None], [(7,), (12,)], [(8, 9), None], [(8, 9), (8, 16)]])
def test_identity(shape):
    input_shape = shape[0]
    output_shape = shape[1]
    A = DFT(input_shape=input_shape, output_shape=output_shape)
    norm_test(A)


@pytest.mark.parametrize(
    "shape_and_idx",
    [[(8,), np.s_[1:4]], [(11, 3), np.s_[2:]], [(2, 12), np.s_[:, 2:]], [(5, 12), np.s_[1:3, 4:]]],
)
def test_slice(shape_and_idx):
    shape = shape_and_idx[0]
    idx = shape_and_idx[1]
    A = Slice(idx=idx, input_shape=shape)
    norm_test(A)


@pytest.mark.parametrize(
    "shape_and_axes",
    [
        [
            (
                3,
                4,
            ),
            (1, 0),
        ],
        [(3, 4, 5), (1, 2, 0)],
        [(5, 4, 2), (2, 1, 0)],
    ],
)
def test_transpose(shape_and_axes):
    shape = shape_and_axes[0]
    axes = shape_and_axes[1]
    A = Transpose(input_shape=shape, axes=axes)
    norm_test(A)


@pytest.mark.parametrize(
    "shape_and_axis",
    [
        [(7,), None],
        [(7,), 0],
        [(7, 1), 1],
        [(5, 6), (1,)],
        [(5, 4, 2), (0, 2)],
    ],
)
def test_sum(shape_and_axis):
    shape = shape_and_axis[0]
    axis = shape_and_axis[1]
    A = Sum(input_shape=shape, axis=axis)
    norm_test(A)


def test_sum_bound():
    N = 8
    A = Diagonal(snp.arange(0.0, N) - N / 2.0)
    B = DFT(input_shape=(N,))
    C = A + B
    Cnorm = operator_norm(C, maxiter=100)
    assert Cnorm <= C.norm() + 1e-4
    assert C.norm_is_bound


def test_sum_product():
    N = 8
    A = Diagonal(snp.arange(0.0, N) - N / 2.0, input_dtype=snp.complex64)
    B = DFT(input_shape=(N,))
    C = A @ B
    Cnorm = operator_norm(C, maxiter=100)
    assert Cnorm <= C.norm() + 1e-4
    assert C.norm_is_bound
