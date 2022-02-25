import numpy as np

from jax.config import config

import pytest

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)


from prox import prox_test

import scico.numpy as snp
from scico import functional, linop, loss
from scico.random import randn


class TestLoss:
    def setup_method(self):
        n = 4
        dtype = np.float64
        A, key = randn((n, n), key=None, dtype=dtype, seed=1234)
        D, key = randn((n,), key=key, dtype=dtype)
        W, key = randn((n,), key=key, dtype=dtype)
        W = 0.1 * W + 1.0
        self.Ao = linop.MatrixOperator(A)
        self.Ao_abs = linop.MatrixOperator(snp.abs(A))
        self.Do = linop.Diagonal(D)
        self.W = linop.Diagonal(W)
        self.y, key = randn((n,), key=key, dtype=dtype)
        self.v, key = randn((n,), key=key, dtype=dtype)  # point for prox eval
        scalar, key = randn((1,), key=key, dtype=dtype)
        self.key = key
        self.scalar = scalar.copy().ravel()[0]

    def test_generic_squared_l2(self):
        A = linop.Identity(input_shape=self.y.shape)
        f = functional.SquaredL2Norm()
        L0 = loss.Loss(self.y, A=A, f=f, scale=0.5)
        L1 = loss.SquaredL2Loss(y=self.y, A=A)
        np.testing.assert_allclose(L0(self.v), L1(self.v))
        np.testing.assert_allclose(L0.prox(self.v, self.scalar), L1.prox(self.v, self.scalar))

    def test_generic_exception(self):
        A = linop.Diagonal(self.v)
        L = loss.Loss(self.y, A=A, scale=0.5)
        with pytest.raises(NotImplementedError):
            L(self.v)
        f = functional.L1Norm()
        L = loss.Loss(self.y, A=A, f=f, scale=0.5)
        assert not L.has_prox
        with pytest.raises(NotImplementedError):
            L.prox(self.v, self.scalar)

    def test_squared_l2(self):
        L = loss.SquaredL2Loss(y=self.y, A=self.Ao)
        assert L.has_eval
        assert L.has_prox

        # test eval
        np.testing.assert_allclose(L(self.v), 0.5 * ((self.Ao @ self.v - self.y) ** 2).sum())

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale
        assert cL(self.v) == self.scalar * L(self.v)

        # SquaredL2 with Diagonal linop has a prox
        L_d = loss.SquaredL2Loss(y=self.y, A=self.Do)

        # test eval
        np.testing.assert_allclose(L_d(self.v), 0.5 * ((self.Do @ self.v - self.y) ** 2).sum())

        assert L_d.has_eval
        assert L_d.has_prox

        cL = self.scalar * L_d
        assert L_d.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L_d.scale
        assert cL(self.v) == self.scalar * L_d(self.v)

        pf = prox_test(self.v, L_d, L_d.prox, 0.75)
        pf = prox_test(self.v, L, L.prox, 0.75)

    def test_weighted_squared_l2(self):
        L = loss.WeightedSquaredL2Loss(y=self.y, A=self.Ao, W=self.W)
        assert L.has_eval
        assert L.has_prox

        # test eval
        np.testing.assert_allclose(
            L(self.v), 0.5 * (self.W @ (self.Ao @ self.v - self.y) ** 2).sum()
        )

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale
        assert cL(self.v) == self.scalar * L(self.v)

        # SquaredL2 with Diagonal linop has a prox
        L_d = loss.WeightedSquaredL2Loss(y=self.y, A=self.Do, W=self.W)

        assert L_d.has_eval
        assert L_d.has_prox

        # test eval
        np.testing.assert_allclose(
            L_d(self.v), 0.5 * (self.W @ (self.Do @ self.v - self.y) ** 2).sum()
        )

        cL = self.scalar * L_d
        assert L_d.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L_d.scale
        assert cL(self.v) == self.scalar * L_d(self.v)

        pf = prox_test(self.v, L_d, L_d.prox, 0.75)
        pf = prox_test(self.v, L, L.prox, 0.75)

    def test_poisson(self):
        L = loss.PoissonLoss(y=self.y, A=self.Ao_abs)
        assert L.has_eval
        assert not L.has_prox

        # test eval
        v = snp.abs(self.v)
        Av = self.Ao_abs @ v
        np.testing.assert_allclose(L(v), 0.5 * snp.sum(Av - self.y * snp.log(Av) + L.const))

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale
        assert cL(v) == self.scalar * L(v)

    def test_weighted_squared_l2_abs(self):
        L = loss.WeightedSquaredL2AbsLoss(y=self.y, A=self.Ao, W=self.W)
        assert L.has_eval
        assert not L.has_prox

        # test eval
        np.testing.assert_allclose(
            L(self.v), 0.5 * (self.W @ (snp.abs(self.Ao @ self.v) - self.y) ** 2).sum()
        )

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale
        assert cL(self.v) == self.scalar * L(self.v)

        # Loss has a prox with Identity linop
        y = snp.abs(self.y)
        L_d = loss.WeightedSquaredL2AbsLoss(y=y, A=None, W=self.W)

        assert L_d.has_eval
        assert L_d.has_prox

        # test eval
        np.testing.assert_allclose(L_d(self.v), 0.5 * (self.W @ (snp.abs(self.v) - y) ** 2).sum())

        cL = self.scalar * L_d
        assert L_d.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L_d.scale
        assert cL(self.v) == self.scalar * L_d(self.v)

        v, key = randn(y.shape, key=self.key, dtype=np.complex128)
        pf = prox_test(self.v, L_d, L_d.prox, 0.75)
        pf = prox_test(v, L_d, L_d.prox, 0.75)
        with pytest.raises(NotImplementedError):
            pf = prox_test(self.v, L, L.prox, 0.75)
