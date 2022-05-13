import numpy as np

import pytest

import scico.numpy as snp
from scico.linop import DFT, Diagonal, Identity, MatrixOperator, operator_norm


def norm_test(A):
    x = A.norm()
    y = operator_norm(A, maxiter=100)
    np.testing.assert_allclose(x, y, rtol=1e-4)
    if A.input_dtype == np.float32:
        x = A.T.norm()
        y = operator_norm(A.T, maxiter=100)
        np.testing.assert_allclose(x, y, rtol=1e-4)
    else:
        x = A.H.norm()
        y = operator_norm(A.H, maxiter=100)
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
