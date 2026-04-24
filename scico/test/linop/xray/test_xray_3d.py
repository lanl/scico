import numpy as np

import jax.numpy as jnp

import scico.linop
from scico.linop.xray import XRayTransform3D


def test_matched_adjoint():
    """See https://github.com/lanl/scico/issues/560."""
    N = 16
    det_count = int(N * 1.05 / np.sqrt(2.0))
    n_projection = 3

    input_shape = (N, N, N)
    det_shape = (det_count, det_count)

    M = XRayTransform3D.matrices_from_euler_angles(
        input_shape,
        det_shape,
        "X",
        np.linspace(0, np.pi, n_projection, endpoint=False)[:, None],  # make (n_projection, 1)
    )
    H = XRayTransform3D(input_shape, matrices=M, det_shape=det_shape)

    assert scico.linop.valid_adjoint(H, H.T, eps=1e-5)


def test_scaling():
    x = jnp.zeros((4, 4, 1))
    x = x.at[1:3, 1:3, 0].set(1.0)

    input_shape = x.shape
    det_shape = x.shape[:2]

    # default spacing
    M = XRayTransform3D.matrices_from_euler_angles(input_shape, det_shape, "X", [[0.0]])
    H = XRayTransform3D(input_shape, matrices=M, det_shape=det_shape)
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
        input_shape, det_shape, "X", [[0.0]], voxel_spacing=[2.0, 1.0, 1.0]
    )
    H = XRayTransform3D(input_shape, matrices=M, det_shape=det_shape)
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
        input_shape, det_shape, "X", [[0.0]], det_spacing=[2.0, 1.0]
    )
    H = XRayTransform3D(input_shape, matrices=M, det_shape=det_shape)
    # fmt: off
    truth = None  # fmt: on  # TODO: Check this case more closely.
    # np.testing.assert_allclose(H @ x, truth)
