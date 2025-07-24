import numpy as np

import pytest

from scico import metric
from scico.examples import create_circular_phantom
from scico.linop.xray.abelc import AxiallySymmetricVolume, _volume_by_axial_symmetry
from scipy.ndimage import gaussian_filter


class TestAxialSymm:
    def setup_method(self, method):
        N = 64
        self.N = N
        self.x2d = create_circular_phantom((N, N), [0.4 * N, 0.2 * N, 0.1 * N], [1, 0, 0.5])
        self.x3d = create_circular_phantom((N, N, N), [0.4 * N, 0.2 * N, 0.1 * N], [1, 0, 0.5])
        self.x2d = gaussian_filter(self.x2d, 1.0)
        self.x3d = gaussian_filter(self.x3d, 1.0)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_vbas(self, axis):
        v0 = _volume_by_axial_symmetry(self.x2d, axis=axis)
        assert metric.rel_res(self.x3d, v0) < 5e-2

        offset = -3
        x2dr = np.roll(self.x2d, offset, axis=1 - axis)
        Nh = (self.N + 1) / 2 - 1
        v1 = _volume_by_axial_symmetry(x2dr, axis=axis, center=Nh + offset)
        assert metric.rel_res(v0, v1) < 1e-5

        zrange = np.arange(-Nh, 0)
        v2 = _volume_by_axial_symmetry(self.x2d, axis=axis, zrange=zrange)
        assert metric.rel_res(self.x3d[..., 0 : self.N // 2], v2) < 5e-2

        A = AxiallySymmetricVolume((self.N, self.N), axis=axis)
        vl = A(self.x2d)
        assert metric.rel_res(v0, vl) < 1e-7
