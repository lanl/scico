import numpy as np

import pytest

import scico.numpy as snp
from scico.linop import FiniteDifference, SingleAxisFiniteDifference
from scico.random import randn
from scico.test.linop.test_linop import adjoint_test


def test_eval():
    with pytest.raises(ValueError):  # axis 3 does not exist
        A = FiniteDifference(input_shape=(3, 4, 5), axes=(0, 3))

    A = FiniteDifference(input_shape=(2, 3), append=1)

    x = snp.array([[1, 0, 1], [1, 1, 0]], dtype=snp.float32)

    Ax = A @ x

    snp.testing.assert_allclose(
        Ax[0],  # down columns x[1] - x[0], ..., append - x[N-1]
        snp.array([[0, 1, -1], [-1, -1, 0]]),
    )
    snp.testing.assert_allclose(Ax[1], snp.array([[-1, 1, -1], [0, -1, 0]]))  # along rows


def test_except():
    with pytest.raises(TypeError):  # axis is not an int
        A = SingleAxisFiniteDifference(input_shape=(3,), axis=2.5)

    with pytest.raises(ValueError):  # invalid parameter combination
        A = SingleAxisFiniteDifference(input_shape=(3,), prepend=0, circular=True)

    with pytest.raises(ValueError):  # invalid prepend value
        A = SingleAxisFiniteDifference(input_shape=(3,), prepend=2)

    with pytest.raises(ValueError):  # invalid append value
        A = SingleAxisFiniteDifference(input_shape=(3,), append="a")


def test_eval_prepend():
    x = snp.arange(1, 6)
    A = SingleAxisFiniteDifference(input_shape=(5,), prepend=0)
    snp.testing.assert_allclose(A @ x, snp.array([0, 1, 1, 1, 1]))
    A = SingleAxisFiniteDifference(input_shape=(5,), prepend=1)
    snp.testing.assert_allclose(A @ x, snp.array([1, 1, 1, 1, 1]))


def test_eval_append():
    x = snp.arange(1, 6)
    A = SingleAxisFiniteDifference(input_shape=(5,), append=0)
    snp.testing.assert_allclose(A @ x, snp.array([1, 1, 1, 1, 0]))
    A = SingleAxisFiniteDifference(input_shape=(5,), append=1)
    snp.testing.assert_allclose(A @ x, snp.array([1, 1, 1, 1, -5]))


@pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("input_shape", [(16,), (16, 24)])
@pytest.mark.parametrize("axes", [0, 1, (0,), (1,), None])
@pytest.mark.parametrize("jit", [False, True])
def test_adjoint(input_shape, input_dtype, axes, jit):
    ndim = len(input_shape)
    if axes in [1, (1,)] and ndim == 1:
        return

    A = FiniteDifference(input_shape=input_shape, input_dtype=input_dtype, axes=axes, jit=jit)
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
