"""
Test methods that make one kind of Operator out of another.
"""
import numpy as np

import pytest

from scico.linop import CircularConvolve, FiniteDifference
from scico.random import randn


@pytest.mark.parametrize(
    "shape_axes",
    [
        ((3, 4), None),  # 2d
        ((3, 4, 5), None),  # 3d
        # ((3, 4, 5), [0, 2]),  # 3d specific axes -- not supported
    ],
)
@pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("jit_old", [False, True])
@pytest.mark.parametrize("jit_new", [False, True])
def testCircularConvolve_from_FiniteDifference(shape_axes, input_dtype, jit_old, jit_new):
    input_shape, axes = shape_axes
    x, _ = randn(input_shape, dtype=input_dtype)

    # make a CircularConvolve from a FiniteDifference
    A = FiniteDifference(
        input_shape=input_shape, input_dtype=input_dtype, axes=axes, circular=True, jit=jit_old
    )

    B = CircularConvolve.from_operator(A, ndims=x.ndim, jit=jit_new)
    np.testing.assert_allclose(A @ x, B @ x, atol=1e-5)

    # try the same on the FiniteDifference Gram
    ATA = A.gram_op

    B = CircularConvolve.from_operator(ATA, ndims=x.ndim, jit=jit_new)
    np.testing.assert_allclose(ATA @ x, B @ x, atol=1e-5)
