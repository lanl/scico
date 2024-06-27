from itertools import combinations

import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.linop import CylindricalGradient, PolarGradient, SphericalGradient
from scico.numpy import Array, BlockArray
from scico.random import randn


class TestPolarGradient:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("jit", [True, False])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("outflags", [(True, True), (True, False), (False, True)])
    @pytest.mark.parametrize("center", [None, (-2, 3), (1.2, -3.5)])
    @pytest.mark.parametrize(
        "shape_axes",
        [
            ((20, 20), None),
            ((20, 21), (0, 1)),
            ((16, 17, 3), (0, 1)),
            ((2, 17, 16), (1, 2)),
            ((2, 17, 16, 3), (2, 1)),
        ],
    )
    @pytest.mark.parametrize("cdiff", [True, False])
    def test_eval(self, cdiff, shape_axes, center, outflags, input_dtype, jit):

        input_shape, axes = shape_axes
        if axes is None:
            testaxes = (0, 1)
        else:
            testaxes = axes
        if center is not None:
            axes_shape = [input_shape[ax] for ax in testaxes]
            center = (snp.array(axes_shape) - 1) / 2 + snp.array(center)
        angular, radial = outflags
        x, key = randn(input_shape, dtype=input_dtype, key=self.key)
        A = PolarGradient(
            input_shape,
            axes=axes,
            center=center,
            angular=angular,
            radial=radial,
            cdiff=cdiff,
            input_dtype=input_dtype,
            jit=jit,
        )
        Ax = A @ x
        if angular and radial:
            assert isinstance(Ax, BlockArray)
            assert len(Ax.shape) == 2
            assert Ax[0].shape == input_shape
            assert Ax[1].shape == input_shape
        else:
            assert isinstance(Ax, Array)
            assert Ax.shape == input_shape
        assert Ax.dtype == input_dtype

        # Test orthogonality of coordinate axes
        coord = A.coord
        for n0, n1 in combinations(range(len(coord)), 2):
            c0 = coord[n0]
            c1 = coord[n1]
            assert snp.abs(snp.sum(c0 * c1)) < 1e-5


class TestCylindricalGradient:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("jit", [True, False])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize(
        "outflags",
        [
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (True, False, False),
            (False, True, True),
            (False, True, False),
            (False, False, True),
        ],
    )
    @pytest.mark.parametrize("center", [None, (-2, 3, 0), (1.2, -3.5, 1.5)])
    @pytest.mark.parametrize(
        "shape_axes",
        [
            ((20, 20, 20), None),
            ((17, 18, 19), (0, 1, 2)),
            ((16, 17, 18, 3), (0, 1, 2)),
            ((2, 17, 16, 15), (1, 2, 3)),
            ((17, 2, 16, 15), (0, 2, 3)),
            ((17, 2, 16, 15), (3, 2, 0)),
        ],
    )
    def test_eval(self, shape_axes, center, outflags, input_dtype, jit):

        input_shape, axes = shape_axes
        if axes is None:
            testaxes = (0, 1, 2)
        else:
            testaxes = axes
        if center is not None:
            axes_shape = [input_shape[ax] for ax in testaxes]
            center = (snp.array(axes_shape) - 1) / 2 + snp.array(center)
        angular, radial, axial = outflags
        x, key = randn(input_shape, dtype=input_dtype, key=self.key)
        A = CylindricalGradient(
            input_shape,
            axes=axes,
            center=center,
            angular=angular,
            radial=radial,
            axial=axial,
            input_dtype=input_dtype,
            jit=jit,
        )
        Ax = A @ x
        Nc = sum([angular, radial, axial])
        if Nc > 1:
            assert isinstance(Ax, BlockArray)
            assert len(Ax) == Nc
            for n in range(Nc):
                assert Ax[n].shape == input_shape
        else:
            assert isinstance(Ax, Array)
            assert Ax.shape == input_shape
        assert Ax.dtype == input_dtype

        # Test orthogonality of coordinate axes
        coord = A.coord
        for n0, n1 in combinations(range(len(coord)), 2):
            c0 = coord[n0]
            c1 = coord[n1]
            s = sum([c0[m] * c1[m] for m in range(len(c0))]).sum()
            assert snp.abs(s) < 1e-5


class TestSphericalGradient:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("jit", [True, False])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize(
        "outflags",
        [
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (True, False, False),
            (False, True, True),
            (False, True, False),
            (False, False, True),
        ],
    )
    @pytest.mark.parametrize("center", [None, (-2, 3, 0), (1.2, -3.5, 1.5)])
    @pytest.mark.parametrize(
        "shape_axes",
        [
            ((20, 20, 20), None),
            ((17, 18, 19), (0, 1, 2)),
            ((16, 17, 18, 3), (0, 1, 2)),
            ((2, 17, 16, 15), (1, 2, 3)),
            ((17, 2, 16, 15), (0, 2, 3)),
            ((17, 2, 16, 15), (3, 2, 0)),
        ],
    )
    def test_eval(self, shape_axes, center, outflags, input_dtype, jit):

        input_shape, axes = shape_axes
        if axes is None:
            testaxes = (0, 1, 2)
        else:
            testaxes = axes
        if center is not None:
            axes_shape = [input_shape[ax] for ax in testaxes]
            center = (snp.array(axes_shape) - 1) / 2 + snp.array(center)
        azimuthal, polar, radial = outflags
        x, key = randn(input_shape, dtype=input_dtype, key=self.key)
        A = SphericalGradient(
            input_shape,
            axes=axes,
            center=center,
            azimuthal=azimuthal,
            polar=polar,
            radial=radial,
            input_dtype=input_dtype,
            jit=jit,
        )
        Ax = A @ x
        Nc = sum([azimuthal, polar, radial])
        if Nc > 1:
            assert isinstance(Ax, BlockArray)
            assert len(Ax) == Nc
            for n in range(Nc):
                assert Ax[n].shape == input_shape
        else:
            assert isinstance(Ax, Array)
            assert Ax.shape == input_shape
        assert Ax.dtype == input_dtype

        # Test orthogonality of coordinate axes
        coord = A.coord
        for n0, n1 in combinations(range(len(coord)), 2):
            c0 = coord[n0]
            c1 = coord[n1]
            s = sum([c0[m] * c1[m] for m in range(len(c0))]).sum()
            assert snp.abs(s) < 1e-5
