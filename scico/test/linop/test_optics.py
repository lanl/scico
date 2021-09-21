import numpy as np

import jax

from scico.linop.optics import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    FresnelPropagator,
)
from scico.random import randn
from scico.test.linop.test_linop import adjoint_AAt_test, adjoint_AtA_test


class TestPropagator:
    def setup_method(self, method):
        key = jax.random.PRNGKey(12345)
        self.N = 128
        self.dx = 1
        self.k0 = 1
        self.z = 1
        self.x, key = randn((self.N, self.N), dtype=np.complex64, key=key)
        self.key = key

    def test_AS_adjoint(self):
        A = AngularSpectrumPropagator(
            input_shape=(self.N, self.N), dx=self.dx, k0=self.k0, z=self.z
        )
        adjoint_AtA_test(A, self.key)
        adjoint_AAt_test(A, self.key)

    def test_Fresnel_adjoint(self):
        A = FresnelPropagator(input_shape=(self.N, self.N), dx=self.dx, k0=self.k0, z=self.z)
        adjoint_AtA_test(A, self.key)
        adjoint_AAt_test(A, self.key)

    def test_Fraunhofer_adjoint(self):
        A = FraunhoferPropagator(input_shape=(self.N, self.N), dx=self.dx, k0=self.k0, z=self.z)
        adjoint_AtA_test(A, self.key)
        adjoint_AAt_test(A, self.key)

    def test_AS_inverse(self):
        A = AngularSpectrumPropagator(
            input_shape=(self.N, self.N), dx=self.dx, k0=self.k0, z=self.z
        )
        Ax = A @ self.x
        AiAx = A.pinv(Ax)
        np.testing.assert_allclose(self.x, AiAx, rtol=5e-4)
