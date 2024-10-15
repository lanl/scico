import numpy as np

import jax.numpy as jnp

import pytest

import scico
import scico.linop
from scico.linop.xray import XRayTransform2D


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


def test_matched_adjoint():
    """See https://github.com/lanl/scico/issues/560."""
    N = 16
    det_count = int(N * 1.05 / np.sqrt(2.0))
    dx = 1.0 / np.sqrt(2)
    n_projection = 3
    angles = np.linspace(0, np.pi, n_projection, endpoint=False)
    A = XRayTransform2D((N, N), angles, det_count=det_count, dx=dx)
    assert scico.linop.valid_adjoint(A, A.T, eps=1e-5)
