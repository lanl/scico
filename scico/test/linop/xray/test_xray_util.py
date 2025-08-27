import numpy as np

from jax.scipy.spatial.transform import Rotation

import scipy.ndimage
from scico.linop.xray import (
    center_image,
    image_alignment_rotation,
    image_centroid,
    rotate_volume,
)


def test_image_centroid():
    v = np.zeros((4, 5))
    v[1:-1, 1:-1] = 1
    assert image_centroid(v) == (1.5, 2.0)
    image_centroid(v, center_offset=True) == (0.0, 0.0)


def test_center_image():
    u = np.zeros((4, 5))
    u[0:-2, 0:-2] = 1
    v = center_image(u)
    np.testing.assert_allclose(image_centroid(v, center_offset=True), (0.0, 0.0), atol=1e-7)
    v = center_image(u, axes=(0,))
    np.testing.assert_allclose(image_centroid(v, center_offset=True), (0.0, -1.0), atol=1e-7)


def test_rotate_volume():
    vol = np.arange(27).reshape((3, 3, 3))
    rot = Rotation.from_euler("XY", [90, 90], degrees=True)
    vol_rot = rotate_volume(vol, rot)
    np.testing.assert_allclose(vol.transpose((1, 2, 0)), vol_rot, rtol=1e-7)


def test_image_alignment():
    u = np.zeros((256, 256), dtype=np.float32)
    u[:, 8::16] = 1
    u[:, 9::16] = 1
    angle = image_alignment_rotation(u)
    assert np.abs(angle) < 1e-3
    ur = scipy.ndimage.rotate(u, 0.75)
    angle = image_alignment_rotation(ur)
    assert np.abs(angle - 0.75) < 1e-3
