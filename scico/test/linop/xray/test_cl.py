import numpy as np

import jax

import pytest

from scico.linop.xray import XRayTransform3D as scicoXRayTransform3D
from scico.linop.xray._cl import _filter_projection, cl_angles_to_vecs, cl_fbp

try:
    import astra  # noqa

    from scico.linop.xray.astra import XRayTransform3D as astraXRayTransform3D
    from scico.linop.xray.astra import convert_to_scico_geometry

    have_astra = True
except ModuleNotFoundError as e:
    if e.name == "astra":
        have_astra = False
    else:
        raise e

have_gpu = True if jax.devices()[0].platform == "gpu" else False


def test_cl_angles_to_vecs():
    angles = np.array([0.0, np.pi / 2])
    vecs = cl_angles_to_vecs(angles)
    assert vecs.shape == (2, 12)


def test_filter_projection():
    alpha = 60.0 * np.pi / 180.0
    y = np.zeros((1, 1, 5), dtype=np.float32)
    y[..., 2] = 1.0
    yf = _filter_projection(y, alpha)[0, 0]
    assert np.argmax(np.abs(yf)) == 2
    np.testing.assert_allclose(yf, np.flipud(yf), atol=1e-7)


@pytest.mark.skipif(not have_astra, reason="astra not installed")
def test_cl_fbp_scico():
    alpha = 60.0 * np.pi / 180.0
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False, dtype=np.float32)
    vectors = cl_angles_to_vecs(angles, alpha)
    vol = np.zeros((9, 33, 33), dtype=np.float32)
    vol[4, 16, 16] = 1
    det_shape = (24, 48)
    matrices = convert_to_scico_geometry(
        input_shape=vol.shape, det_count=det_shape, vectors=vectors
    )
    X = scicoXRayTransform3D(vol.shape, matrices, det_shape)
    y = X @ vol
    xfbp = cl_fbp(y, alpha, X)
    assert xfbp[4, 16, 16] == xfbp.max()


@pytest.mark.skipif(
    not have_astra or not have_gpu, reason="astra not installed or GPU not available"
)
def test_cl_fbp_astra():
    alpha = 60.0 * np.pi / 180.0
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False, dtype=np.float32)
    vectors = cl_angles_to_vecs(angles, alpha)
    vol = np.zeros((9, 33, 33), dtype=np.float32)
    vol[4, 16, 16] = 1
    det_shape = (24, 48)
    X = astraXRayTransform3D(
        vol.shape,
        det_count=det_shape,
        vectors=vectors,
    )
    y = X @ vol
    xfbp = cl_fbp(y, alpha, X)
    assert xfbp[4, 16, 16] == xfbp.max()
