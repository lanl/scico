import numpy as np

import jax
import jax.numpy as jnp

import pytest

import scico
from scico.linop.xray import XRayTransform2D, XRayTransform3D
from scico.metric import psnr


@pytest.mark.filterwarnings("error")
def test_init():
    input_shape = (3, 3)

    # no warning with default settings, even at 45 degrees
    H = XRayTransform2D(input_shape, jnp.array([jnp.pi / 4]))

    # no warning if we project orthogonally with oversized pixels
    H = XRayTransform2D(input_shape, jnp.array([0]), dx=jnp.array([1, 1]))

    # warning if the projection angle changes
    with pytest.warns(UserWarning):
        H = XRayTransform2D(input_shape, jnp.array([0.1]), dx=jnp.array([1.1, 1.1]))

    # warning if the pixels get any larger
    with pytest.warns(UserWarning):
        H = XRayTransform2D(input_shape, jnp.array([0]), dx=jnp.array([1.1, 1.1]))


def test_apply():
    im_shape = (12, 13)
    num_angles = 10
    x = jnp.ones(im_shape)

    angles = jnp.linspace(0, jnp.pi, num=num_angles, endpoint=False)

    # general projection
    H = XRayTransform2D(x.shape, angles)
    y = H @ x
    assert y.shape[0] == (num_angles)

    # fixed det_count
    det_count = 14
    H = XRayTransform2D(x.shape, angles, det_count=det_count)
    y = H @ x
    assert y.shape[1] == det_count


def test_apply_adjoint():
    im_shape = (12, 13)
    num_angles = 10
    x = jnp.ones(im_shape, dtype=jnp.float32)

    angles = jnp.linspace(0, jnp.pi, num=num_angles, endpoint=False)

    # general projection
    H = XRayTransform2D(x.shape, angles)
    y = H @ x
    assert y.shape[0] == (num_angles)

    # adjoint
    bp = H.T @ y
    assert scico.linop.valid_adjoint(
        H, H.T, eps=1e-4
    )  # associative reductions might cause small errors, hence 1e-5

    # fixed det_length
    det_count = 14
    H = XRayTransform2D(x.shape, angles, det_count=det_count)
    y = H @ x
    assert y.shape[1] == det_count


@pytest.mark.parametrize("dx", [0.5, 1.0 / np.sqrt(2)])
@pytest.mark.parametrize("det_count_factor", [1.02 / np.sqrt(2.0), 1.0])
def test_fbp(dx, det_count_factor):
    N = 256
    x_gt = np.zeros((N, N), dtype=np.float32)
    N4 = N // 4
    x_gt[N4:-N4, N4:-N4] = 1.0

    det_count = int(det_count_factor * N)
    n_proj = 360
    angles = np.linspace(0, np.pi, n_proj, endpoint=False)
    A = XRayTransform2D(x_gt.shape, angles, det_count=det_count, dx=dx)
    y = A(x_gt)
    x_fbp = A.fbp(y)
    assert psnr(x_gt, x_fbp) > 28


def test_fbp_jit():
    N = 64
    x_gt = np.ones((N, N), dtype=np.float32)

    det_count = N
    n_proj = 90
    angles = np.linspace(0, np.pi, n_proj, endpoint=False)
    A = XRayTransform2D(x_gt.shape, angles, det_count=det_count)
    y = A(x_gt)
    fbp = jax.jit(A.fbp)
    x_fbp = fbp(y)


def test_3d_scaling():
    x = jnp.zeros((4, 4, 1))
    x = x.at[1:3, 1:3, 0].set(1.0)

    input_shape = x.shape
    output_shape = x.shape[:2]

    # default spacing
    M = XRayTransform3D.matrices_from_euler_angles(input_shape, output_shape, "X", [0.0])
    H = XRayTransform3D(input_shape, matrices=M, det_shape=output_shape)

    # fmt: off
    truth = jnp.array(
        [[[0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 1.0, 0.0],
          [0.0, 1.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0]]]
    )  # fmt: on
    np.testing.assert_allclose(H @ x, truth)

    # bigger voxels in the x (first index) direction
    M = XRayTransform3D.matrices_from_euler_angles(
        input_shape, output_shape, "X", [0.0], voxel_spacing=[2.0, 1.0, 1.0]
    )
    H = XRayTransform3D(input_shape, matrices=M, det_shape=output_shape)
    # fmt: off
    truth = jnp.array(
        [[[0. , 0.5, 0.5, 0. ],
          [0. , 0.5, 0.5, 0. ],
          [0. , 0.5, 0.5, 0. ],
          [0. , 0.5, 0.5, 0. ]]]
    )  # fmt: on
    np.testing.assert_allclose(H @ x, truth)

    # bigger detector pixels in the x (first index) direction
    M = XRayTransform3D.matrices_from_euler_angles(
        input_shape, output_shape, "X", [0.0], det_spacing=[2.0, 1.0]
    )
    H = XRayTransform3D(input_shape, matrices=M, det_shape=output_shape)
    # fmt: off
    truth = None  # fmt: on  # TODO: Check this case more closely.
    # np.testing.assert_allclose(H @ x, truth)
