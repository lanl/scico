import numpy as np

from jax.scipy.spatial.transform import Rotation

import scipy.ndimage
from scico.linop.xray import (
    center_image,
    image_alignment_rotation,
    image_centroid,
    rotate_volume,
    volume_alignment_rotation,
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


def test_volume_alignment():
    u = np.zeros((256, 256, 32), dtype=np.float32)
    u[8::16, :, 2::6] = 1
    u[9::16, :, 2::6] = 1
    u[:, 8::16, 2::6] = 1
    u[:, 9::16, 2::6] = 1
    u[8::16, :, 3::6] = 1
    u[9::16, :, 3::6] = 1
    u[:, 8::16, 3::6] = 1
    u[:, 9::16, 3::6] = 1
    rot = volume_alignment_rotation(u)
    assert rot.magnitude() < 1e-5
    ref_rot = Rotation.from_euler("XY", (1.6, -0.9), degrees=True)
    ur = rotate_volume(u, ref_rot)
    rot = volume_alignment_rotation(ur)
    assert (
        np.abs(ref_rot.as_euler("XYZ", degrees=True) - rot.as_euler("XYZ", degrees=True)).max()
        < 1e-1
    )
