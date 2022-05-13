import numpy as np

import jax

import pytest

from scico.denoiser import DnCNN, bm3d, bm4d, have_bm3d, have_bm4d
from scico.random import randn
from scico.test.osver import osx_ver_geq_than


# bm3d is known to be broken on OSX 11.6.5. It may be broken on earlier versions too,
# but this has not been confirmed
@pytest.mark.skipif(osx_ver_geq_than("11.6.5"), reason="bm3d broken on this platform")
@pytest.mark.skipif(not have_bm3d, reason="bm3d package not installed")
class TestBM3D:
    def setup(self):
        key = None
        self.x_gry, key = randn((32, 33), key=key, dtype=np.float32)
        self.x_rgb, key = randn((33, 34, 3), key=key, dtype=np.float32)

    def test_shape(self):
        assert bm3d(self.x_gry, 1.0).shape == self.x_gry.shape
        assert bm3d(self.x_rgb, 1.0, is_rgb=True).shape == self.x_rgb.shape

    def test_gry(self):
        no_jit = bm3d(self.x_gry, 1.0)
        jitted = jax.jit(bm3d)(self.x_gry, 1.0)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

    def test_rgb(self):
        no_jit = bm3d(self.x_rgb, 1.0)
        jitted = jax.jit(bm3d)(self.x_rgb, 1.0, is_rgb=True)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

    def test_bad_inputs(self):
        x, key = randn((32,), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            bm3d(x, 1.0)
        x, key = randn((12, 12, 4, 3), key=key, dtype=np.float32)
        with pytest.raises(ValueError):
            bm3d(x, 1.0)
        x, key = randn(((2, 3), (3, 4, 5)), key=key, dtype=np.float32)
        with pytest.raises(ValueError):
            bm3d(x, 1.0)
        x, key = randn((5, 9), key=key, dtype=np.float32)
        with pytest.raises(ValueError):
            bm3d(x, 1.0)
        z, key = randn((32, 32), key=key, dtype=np.complex64)
        with pytest.raises(TypeError):
            bm3d(z, 1.0)


# bm4d is known to be broken on OSX 11.6.5. It may be broken on earlier versions too,
# but this has not been confirmed
@pytest.mark.skipif(osx_ver_geq_than("11.6.5"), reason="bm4d broken on this platform")
@pytest.mark.skipif(not have_bm4d, reason="bm4d package not installed")
class TestBM4D:
    def setup(self):
        key = None
        self.x1, key = randn((16, 17, 18), key=key, dtype=np.float32)
        self.x2, key = randn((16, 17, 8), key=key, dtype=np.float32)
        self.x3, key = randn((16, 17, 9, 1, 1), key=key, dtype=np.float32)

    def test_shape(self):
        assert bm4d(self.x1, 1.0).shape == self.x1.shape
        assert bm4d(self.x2, 1.0).shape == self.x2.shape
        assert bm4d(self.x3, 1.0).shape == self.x3.shape

    def test_jit(self):
        no_jit = bm4d(self.x1, 1.0)
        jitted = jax.jit(bm4d)(self.x1, 1.0)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

        no_jit = bm4d(self.x2, 1.0)
        jitted = jax.jit(bm4d)(self.x2, 1.0)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

    def test_bad_inputs(self):
        x, key = randn((32,), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            bm4d(x, 1.0)
        x, key = randn((12, 12, 4, 3), key=key, dtype=np.float32)
        with pytest.raises(ValueError):
            bm4d(x, 1.0)
        x, key = randn(((2, 3), (3, 4, 5)), key=key, dtype=np.float32)
        with pytest.raises(ValueError):
            bm4d(x, 1.0)
        x, key = randn((5, 9), key=key, dtype=np.float32)
        with pytest.raises(ValueError):
            bm4d(x, 1.0)
        z, key = randn((32, 32), key=key, dtype=np.complex64)
        with pytest.raises(TypeError):
            bm4d(z, 1.0)


class TestDnCNN:
    def setup(self):
        key = None
        self.x_sngchn, key = randn((32, 33), key=key, dtype=np.float32)
        self.x_mltchn, key = randn((33, 34, 5), key=key, dtype=np.float32)
        self.dncnn = DnCNN()

    def test_single_channel(self):
        no_jit = self.dncnn(self.x_sngchn)
        jitted = jax.jit(self.dncnn)(self.x_sngchn)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

    def test_multi_channel(self):
        no_jit = self.dncnn(self.x_mltchn)
        jitted = jax.jit(self.dncnn)(self.x_mltchn)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

    def test_init(self):
        dncnn = DnCNN(variant="6L")
        x = dncnn(self.x_sngchn)
        dncnn = DnCNN(variant="17H")
        x = dncnn(self.x_mltchn)
        with pytest.raises(ValueError):
            dncnn = DnCNN(variant="3A")

    def test_bad_inputs(self):
        x, key = randn((32,), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            self.dncnn(x)
        x, key = randn((12, 12, 4, 3), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            self.dncnn(x)
        x, key = randn(((2, 3), (3, 4, 5)), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            self.dncnn(x)
        z, key = randn((32, 32), key=None, dtype=np.complex64)
        with pytest.raises(TypeError):
            self.dncnn(z)
