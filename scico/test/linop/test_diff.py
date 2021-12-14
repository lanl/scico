import numpy as np

import pytest

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.linop import FiniteDifference
from scico.random import randn
from scico.test.linop.test_linop import adjoint_test


@pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("input_shape", [(32,), (32, 48)])
@pytest.mark.parametrize("axes", [0, 1, (0,), (1,), None])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("append", [None, 0.0])
def test_eval(input_shape, input_dtype, axes, jit, append):

    ndim = len(input_shape)
    x, _ = randn(input_shape, dtype=input_dtype)

    if axes in [1, (1,)] and ndim == 1:
        with pytest.raises(ValueError):
            A = FiniteDifference(
                input_shape=input_shape, input_dtype=input_dtype, axes=axes, append=append
            )
    else:
        A = FiniteDifference(
            input_shape=input_shape, input_dtype=input_dtype, axes=axes, jit=jit, append=append
        )
        Ax = A @ x
        assert A.input_dtype == input_dtype

        # construct expected output
        if axes is None:
            if ndim == 1:
                y = snp.diff(x, append=append)
            else:
                y = BlockArray.array(
                    [snp.diff(x, axis=0, append=append), snp.diff(x, axis=1, append=append)]
                )
        elif np.isscalar(axes):
            y = snp.diff(x, axis=axes, append=append)
        elif len(axes) == 1:
            y = snp.diff(x, axis=axes[0], append=append)

        np.testing.assert_allclose(Ax.ravel(), y.ravel(), rtol=1e-4)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(32,), (32, 48)])
    @pytest.mark.parametrize("axes", [0, 1, (0,), (1,), None])
    @pytest.mark.parametrize("jit", [False, True])
    def test_adjoint(self, input_shape, input_dtype, axes, jit):
        ndim = len(input_shape)
        if axes in [1, (1,)] and ndim == 1:
            pass
        else:
            A = FiniteDifference(
                input_shape=input_shape, input_dtype=input_dtype, axes=axes, jit=jit
            )
            adjoint_test(A)


@pytest.mark.parametrize(
    "shape_axes",
    [
        ((3, 4), None),  # 2d
        ((3, 4), 0),  # 2d specific axis
        ((3, 4, 5), None),  # 3d
        ((3, 4, 5), [0, 2]),  # 3d specific axes
    ],
)
@pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("jit", [False, True])
def test_eval_circular(shape_axes, input_dtype, jit):
    input_shape, axes = shape_axes
    x, _ = randn(input_shape, dtype=input_dtype)
    A = FiniteDifference(
        input_shape=input_shape, input_dtype=input_dtype, axes=axes, circular=True, jit=jit
    )
    Ax = A @ x

    # check that correct differences are returned
    for ax in A.axes:
        np.testing.assert_allclose(np.roll(x, -1, ax) - x, Ax[ax], atol=1e-5, rtol=0)

    # check that the all results match noncircular results except at the last pixel
    B = FiniteDifference(input_shape=input_shape, input_dtype=input_dtype, axes=axes, jit=jit)
    Bx = B @ x

    for ax_ind, ax in enumerate(A.axes):
        np.testing.assert_allclose(
            Ax[
                (ax_ind,)
                + tuple(slice(0, -1) if i == ax else slice(None) for i in range(len(input_shape)))
            ],
            Bx[ax_ind],
            atol=1e-5,
            rtol=0,
        )
