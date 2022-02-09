import numpy as np

import jax

import pytest

from scico.denoiser import DnCNN, bm3d
from scico.random import randn


class TestBM3D:
    def setup(self):
        key = None
        self.x_gry, key = randn((32, 33), key=key, dtype=np.float32)
        self.x_rgb, key = randn((33, 34, 3), key=key, dtype=np.float32)

    def test_shape(self):
        assert bm3d(self.x_gry, 1.0).shape == self.x_gry.shape
        assert bm3d(self.x_rgb, 1.0).shape == self.x_rgb.shape

    def test_gry(self):
        no_jit = bm3d(self.x_gry, 1.0)
        jitted = jax.jit(bm3d)(self.x_gry, 1.0)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

    def test_rgb(self):
        no_jit = bm3d(self.x_rgb, 1.0)
        jitted = jax.jit(bm3d)(self.x_rgb, 1.0)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

    def test_bad_inputs(self):
        x, key = randn((32,), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            bm3d(x, 1.0)

        x, key = randn((12, 12, 4, 3), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            bm3d(x, 1.0)

        x_b, key = randn(((2, 3), (3, 4, 5)), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            bm3d(x, 1.0)

        z, key = randn((32, 32), key=None, dtype=np.complex64)
        with pytest.raises(TypeError):
            bm3d(z, 1.0)


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

    def test_bad_inputs(self):
        x, key = randn((32,), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            self.dncnn(x)

        x, key = randn((12, 12, 4, 3), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            self.dncnn(x)

        x_b, key = randn(((2, 3), (3, 4, 5)), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            self.dncnn(x)

        z, key = randn((32, 32), key=None, dtype=np.complex64)
        with pytest.raises(TypeError):
            self.dncnn(z)
