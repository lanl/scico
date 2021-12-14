import numpy as np

import jax

import pytest

from scico.linop.optics import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    FresnelPropagator,
    radial_transverse_frequency,
)
from scico.random import randn
from scico.test.linop.test_linop import adjoint_test

prop_list = [AngularSpectrumPropagator, FresnelPropagator, FraunhoferPropagator]


class TestPropagator:
    def setup_method(self, method):
        key = jax.random.PRNGKey(12345)
        self.N = 128
        self.dx = 1
        self.k0 = 1
        self.z = 1
        self.key = key

    @pytest.mark.parametrize("ndim", [1, 2])
    @pytest.mark.parametrize("prop", prop_list)
    def test_prop_adjoint(self, prop, ndim):
        A = prop(input_shape=(self.N,) * ndim, dx=self.dx, k0=self.k0, z=self.z)
        adjoint_test(A, self.key)

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_AS_inverse(self, ndim):
        A = AngularSpectrumPropagator(
            input_shape=(self.N,) * ndim, dx=self.dx, k0=self.k0, z=self.z
        )
        x, key = randn(A.input_shape, dtype=np.complex64, key=self.key)
        Ax = A @ x
        AiAx = A.pinv(Ax)
        np.testing.assert_allclose(x, AiAx, rtol=5e-4)

    @pytest.mark.parametrize("prop", prop_list)
    def test_3d_invalid(self, prop):
        with pytest.raises(ValueError):
            prop(input_shape=(self.N, self.N, self.N), dx=self.dx, k0=self.k0, z=self.z)

    @pytest.mark.parametrize("prop", prop_list)
    def test_shape_dx_mismatch(self, prop):
        with pytest.raises(ValueError):
            prop(input_shape=(self.N,), dx=(self.dx, self.dx), k0=self.k0, z=self.z)

    def test_3d_invalid_radial(self):
        with pytest.raises(ValueError):
            radial_transverse_frequency(input_shape=(self.N, self.N, self.N), dx=self.dx)

    def test_shape_dx_mismatch(self):
        with pytest.raises(ValueError):
            radial_transverse_frequency(input_shape=(self.N,), dx=(self.dx, self.dx))


@pytest.mark.parametrize("ndim", [1, 2])
def test_asp_sampling(ndim):
    N = 128
    dx = 1
    z = 1
    A = AngularSpectrumPropagator(input_shape=(N,) * ndim, dx=dx, k0=1, z=z)
    assert not A.adequate_sampling()
    A = AngularSpectrumPropagator(input_shape=(N,) * ndim, dx=dx, k0=100, z=z)
    assert A.adequate_sampling()


@pytest.mark.parametrize("ndim", [1, 2])
def test_fresnel_sampling(ndim):
    N = 128
    dx = 1
    k0 = 1
    A = FresnelPropagator(input_shape=(N,) * ndim, dx=dx, k0=k0, z=N ** 2)
    assert not A.adequate_sampling()
    A = FresnelPropagator(input_shape=(N,) * ndim, dx=dx, k0=k0, z=1)
    assert A.adequate_sampling()


@pytest.mark.parametrize("ndim", [1, 2])
def test_fraunhofer_sampling(ndim):
    N = 128
    dx = 1
    k0 = 1
    A = FraunhoferPropagator(input_shape=(N,) * ndim, dx=dx, k0=k0, z=N ** 2)
    assert not A.adequate_sampling()
    A = FraunhoferPropagator(input_shape=(N,) * ndim, dx=dx, k0=k0, z=1)
    assert A.adequate_sampling()
