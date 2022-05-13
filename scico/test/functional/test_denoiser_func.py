import numpy as np

import pytest

from scico import denoiser, functional
from scico.denoiser import have_bm3d, have_bm4d
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
        self.f_gry = functional.BM3D()
        self.f_rgb = functional.BM3D(is_rgb=True)

    def test_gry(self):
        y0 = self.f_gry.prox(self.x_gry, 1.0)
        y1 = denoiser.bm3d(self.x_gry, 1.0)
        np.testing.assert_allclose(y0, y1, rtol=1e-5)

    def test_rgb(self):
        y0 = self.f_rgb.prox(self.x_rgb, 1.0)
        y1 = denoiser.bm3d(self.x_rgb, 1.0, is_rgb=True)
        np.testing.assert_allclose(y0, y1, rtol=1e-5)


# bm4d is known to be broken on OSX 11.6.5. It may be broken on earlier versions too,
# but this has not been confirmed
@pytest.mark.skipif(osx_ver_geq_than("11.6.5"), reason="bm4d broken on this platform")
@pytest.mark.skipif(not have_bm4d, reason="bm4d package not installed")
class TestBM4D:
    def setup(self):
        key = None
        self.x, key = randn((16, 17, 14), key=key, dtype=np.float32)
        self.f = functional.BM4D()

    def test(self):
        y0 = self.f.prox(self.x, 1.0)
        y1 = denoiser.bm4d(self.x, 1.0)
        np.testing.assert_allclose(y0, y1, rtol=1e-5)


class TestDnCNN:
    def setup(self):
        key = None
        self.x_sngchn, key = randn((32, 33), key=key, dtype=np.float32)
        self.x_mltchn, key = randn((33, 34, 5), key=key, dtype=np.float32)
        self.dncnn = denoiser.DnCNN()
        self.f = functional.DnCNN()

    def test_sngchn(self):
        y0 = self.f.prox(self.x_sngchn, 1.0)
        y1 = self.dncnn(self.x_sngchn)
        np.testing.assert_allclose(y0, y1, rtol=1e-5)

    def test_mltchn(self):
        y0 = self.f.prox(self.x_mltchn, 1.0)
        y1 = self.dncnn(self.x_mltchn)
        np.testing.assert_allclose(y0, y1, rtol=1e-5)
