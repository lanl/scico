import jax.numpy as jnp

from scico.linop import ParallelFixedAxis2dProjector, XRayTransform


def test_apply():
    im_shape = (12, 13)
    num_angles = 10
    x = jnp.ones(im_shape)

    angles = jnp.linspace(0, jnp.pi, num=num_angles, endpoint=False)

    # general projection
    H = XRayTransform(ParallelFixedAxis2dProjector(x.shape, angles))
    y = H @ x
    assert y.shape[0] == (num_angles)

    # fixed det_length
    det_length = 14
    H = XRayTransform(ParallelFixedAxis2dProjector(x.shape, angles, det_length=det_length))
    y = H @ x
    assert y.shape[1] == det_length

    # dither off
    H = XRayTransform(ParallelFixedAxis2dProjector(x.shape, angles, do_dithering=False))
    y = H @ x
