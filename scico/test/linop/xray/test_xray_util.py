import numpy as np

from jax.scipy.spatial.transform import Rotation

from scico.linop.xray import rotate_volume


def test_rotate_volume():
    vol = np.arange(27).reshape((3, 3, 3))
    rot = Rotation.from_euler("XY", [90, 90], degrees=True)
    vol_rot = rotate_volume(vol, rot)
    np.testing.assert_allclose(vol.transpose((1, 2, 0)), vol_rot, rtol=1e-7)
