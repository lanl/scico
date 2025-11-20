import numpy as np

import pytest

from scico import metric
from scico.examples import create_circular_phantom
from scico.linop.xray.symcone import (
    AxiallySymmetricVolume,
    SymConeXRayTransform,
    _volume_by_axial_symmetry,
)
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


class TestAbelCone:
    def setup_method(self, method):
        N = 64
        self.N = N
        self.x2d = create_circular_phantom((N, N), [0.4 * N, 0.2 * N, 0.1 * N], [1, 0, 0.5])
        self.x3d = create_circular_phantom((N, N, N), [0.4 * N, 0.2 * N, 0.1 * N], [1, 0, 0.5])
        self.x2d = gaussian_filter(self.x2d, 1.0)
        self.x3d = gaussian_filter(self.x3d, 1.0)

    @pytest.mark.parametrize("num_slabs", [1, 2, 3])
    def test_2d(self, num_slabs):
        A = SymConeXRayTransform(self.x2d.shape, 1e8, 1e8 + 1, num_slabs=num_slabs)
        ya = A(self.x2d)
        x2ds = _volume_by_axial_symmetry(self.x2d, axis=0)
        ys = np.sum(x2ds, axis=1)
        assert metric.rel_res(ys, ya) < 1e-6

    @pytest.mark.parametrize("num_slabs", [1, 2, 3])
    def test_2d_unequal(self, num_slabs):
        x2dc = self.x2d[1:-1]
        A = SymConeXRayTransform(x2dc.shape, 1e8, 1e8 + 1, num_slabs=num_slabs)
        ya = A(x2dc)
        x2ds = _volume_by_axial_symmetry(x2dc, axis=0)
        ys = np.sum(x2ds, axis=1)
        assert metric.rel_res(ys, ya) < 1e-6

    @pytest.mark.parametrize("num_slabs", [1, 2, 3])
    def test_3d(self, num_slabs):
        A = SymConeXRayTransform(self.x3d.shape, 1e8, 1e8 + 1, num_slabs=num_slabs)
        ya = A(self.x3d)
        ys = np.sum(self.x3d, axis=1)
        assert metric.rel_res(ys, ya) < 1e-6

    @pytest.mark.parametrize("num_slabs", [1, 2, 3])
    def test_3d_unequal(self, num_slabs):
        x3dc = self.x3d[1:-1, 2:-2]
        A = SymConeXRayTransform(x3dc.shape, 1e8, 1e8 + 1, num_slabs=num_slabs)
        ya = A(x3dc)
        ys = np.sum(x3dc, axis=1)
        assert metric.rel_res(ys, ya) < 1e-6

    @pytest.mark.parametrize("num_slabs", [1, 2, 3])
    def test_2d3d_unequal(self, num_slabs):
        A2d = SymConeXRayTransform(self.x2d.shape, 5e1, 6e1, num_slabs=num_slabs)
        A3d = SymConeXRayTransform(self.x3d.shape, 5e1, 6e1, num_slabs=num_slabs)
        y2d = A2d(self.x2d)
        y3d = A3d(self.x3d)
        assert metric.rel_res(y3d, y2d) < 2e-2

    @pytest.mark.parametrize("axis", [0, 1])
    def test_proj_axis(self, axis):
        N = self.N
        N2 = N // 2
        N4 = N // 4
        x = np.zeros((N, N))
        if axis == 0:
            x[N2 - 1 : N2 + 1, N4 - 1 : N4 + 1] = 1
        else:
            x[N4 - 1 : N4 + 1, N2 - 1 : N2 + 1] = 1
        A = SymConeXRayTransform(x.shape, 1e2, 2e2, axis=axis, num_slabs=1)
        y = A(x)
        if axis == 0:
            assert np.sum(np.sum(y, axis=1) > 0) <= 4
            assert np.sum(np.sum(y, axis=0) > 0) >= N2
        else:
            assert np.sum(np.sum(y, axis=0) > 0) <= 4
            assert np.sum(np.sum(y, axis=1) > 0) >= N2

    @pytest.mark.parametrize("axis", [0, 1])
    def test_fdk(self, axis):
        A = SymConeXRayTransform(self.x3d.shape, 1e2, 2e2, axis=axis, num_slabs=1)
        y = A(self.x3d)
        z = A.fdk(y)
        assert metric.rel_res(self.x2d, z) < 0.2
