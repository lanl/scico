import jax.numpy as jnp

import scico
from scico.linop import Parallel2dProjector, XRayTransform


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
        H, H.T, eps=1e-5
    )  # associative reductions might cause small errors, hence 1e-5

    # fixed det_length
    det_count = 14
    H = XRayTransform(Parallel2dProjector(x.shape, angles, det_count=det_count))
    y = H @ x
    assert y.shape[1] == det_count
