import jax.numpy as jnp

import pytest

import scico
from scico.linop import Parallel2dProjector, XRayTransform


@pytest.mark.filterwarnings("error")
def test_init():
    input_shape = (3, 3)

    # no warning with default settings, even at 45 degrees
    H = XRayTransform(Parallel2dProjector(input_shape, jnp.array([jnp.pi / 4])))

    # no warning if we project orthogonally with oversized pixels
    H = XRayTransform(Parallel2dProjector(input_shape, jnp.array([0]), dx=jnp.array([1, 1])))

    # warning if the projection angle changes
    with pytest.warns(UserWarning):
        H = XRayTransform(
            Parallel2dProjector(input_shape, jnp.array([0.1]), dx=jnp.array([1.1, 1.1]))
        )

    # warning if the pixels get any larger
    with pytest.warns(UserWarning):
        H = XRayTransform(
            Parallel2dProjector(input_shape, jnp.array([0]), dx=jnp.array([1.1, 1.1]))
        )


def test_apply():
    im_shape = (12, 13)
    num_angles = 10
    x = jnp.ones(im_shape)

    angles = jnp.linspace(0, jnp.pi, num=num_angles, endpoint=False)

    # general projection
    H = XRayTransform(Parallel2dProjector(x.shape, angles))
    y = H @ x
    assert y.shape[0] == (num_angles)

    # fixed det_count
    det_count = 14
    H = XRayTransform(Parallel2dProjector(x.shape, angles, det_count=det_count))
    y = H @ x
    assert y.shape[1] == det_count


def test_apply_adjoint():
    im_shape = (12, 13)
    num_angles = 10
    x = jnp.ones(im_shape)

    angles = jnp.linspace(0, jnp.pi, num=num_angles, endpoint=False)

    # general projection
    H = XRayTransform(Parallel2dProjector(x.shape, angles))
    y = H @ x
    assert y.shape[0] == (num_angles)

    # adjoint
    bp = H.T @ y
    assert scico.linop.valid_adjoint(
        H, H.T, eps=1e-4
    )  # associative reductions might cause small errors, hence 1e-5

    # fixed det_length
    det_count = 14
    H = XRayTransform(Parallel2dProjector(x.shape, angles, det_count=det_count))
    y = H @ x
    assert y.shape[1] == det_count
