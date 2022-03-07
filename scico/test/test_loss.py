import numpy as np

from jax.config import config

import pytest

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)


from prox import prox_test

import scico.numpy as snp
from scico import functional, linop, loss
from scico.array import complex_dtype
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

        # squared l2 loss with diagonal linop has a prox
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
        L = loss.SquaredL2Loss(y=self.y, A=self.Ao, W=self.W)
        assert L.has_eval
        assert L.has_prox
        np.testing.assert_allclose(
            L(self.v), 0.5 * (self.W @ (self.Ao @ self.v - self.y) ** 2).sum()
        )
        pf = prox_test(self.v, L, L.prox, 0.75)

        # weighted l2 loss with diagonal linop has a prox
        L_d = loss.SquaredL2Loss(y=self.y, A=self.Do, W=self.W)
        assert L_d.has_eval
        assert L_d.has_prox
        np.testing.assert_allclose(
            L_d(self.v), 0.5 * (self.W @ (self.Do @ self.v - self.y) ** 2).sum()
        )
        pf = prox_test(self.v, L_d, L_d.prox, 0.75)

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


class TestAbsLoss:

    abs_loss = (
        (loss.SquaredL2AbsLoss, snp.abs),
        (loss.SquaredL2SquaredAbsLoss, lambda x: snp.abs(x) ** 2),
    )

    def setup_method(self):
        n = 4
        dtype = np.float64
        A, key = randn((n, n), key=None, dtype=dtype, seed=1234)
        W, key = randn((n,), key=key, dtype=dtype)
        W = 0.1 * W + 1.0
        self.Ao = linop.MatrixOperator(A)
        self.Ao_abs = linop.MatrixOperator(snp.abs(A))
        self.W = linop.Diagonal(W)
        self.x, key = randn((n,), key=key, dtype=complex_dtype(dtype))
        self.v, key = randn((n,), key=key, dtype=complex_dtype(dtype))  # point for prox eval
        scalar, key = randn((1,), key=key, dtype=dtype)
        self.scalar = scalar.copy().ravel()[0]

    @pytest.mark.parametrize("loss_tuple", abs_loss)
    def test_properties(self, loss_tuple):
        loss_class = loss_tuple[0]
        loss_func = loss_tuple[1]

        y = loss_func(self.Ao(self.x))
        L = loss_class(y=y, A=self.Ao, W=self.W)
        assert L.has_eval
        assert not L.has_prox

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale
        assert cL(self.v) == self.scalar * L(self.v)

        with pytest.raises(NotImplementedError):
            px = L.prox(self.v, 0.75)

        np.testing.assert_allclose(L(self.x), 0)

        y = loss_func(self.x)
        L = loss_class(y=y, A=None, W=None)
        assert L.has_eval
        assert L.has_prox

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale
        assert cL(self.v) == self.scalar * L(self.v)

        np.testing.assert_allclose(L(self.x), 0)

        W = -1 * self.W
        with pytest.raises(ValueError):
            L = loss_class(y=y, W=W)

        with pytest.raises(TypeError):
            L = loss_class(y=y, W=linop.Sum(input_shape=W.input_shape))

    @pytest.mark.parametrize("loss_tuple", abs_loss)
    def test_prox(self, loss_tuple):
        loss_class = loss_tuple[0]
        loss_func = loss_tuple[1]

        y = loss_func(self.x)
        L = loss_class(y=y, A=None, W=self.W)

        pf = prox_test(self.v.real, L, L.prox, 0.5)  # real v

        pf = prox_test(self.v, L, L.prox, 0.0)  # complex v
        pf = prox_test(self.v, L, L.prox, 0.1)  # complex v
        pf = prox_test(self.v, L, L.prox, 2.0)  # complex v

        pf = prox_test((1 + 1j) * snp.zeros(self.v.shape), L, L.prox, 0.0)  # complex zero v
        pf = prox_test((1 + 1j) * snp.zeros(self.v.shape), L, L.prox, 1.0)  # complex zero v
        pf = prox_test((1 + 1j) * snp.zeros(self.v.shape), L, L.prox, 2.0)  # complex zero v

        # zero y
        y = snp.zeros(self.x.shape)
        L = loss_class(y=y, A=None, W=self.W)

        pf = prox_test(self.v.real, L, L.prox, 0.5)  # real v

        pf = prox_test(self.v, L, L.prox, 0.0)  # complex v
        pf = prox_test(self.v, L, L.prox, 0.1)  # complex v

        pf = prox_test((1 + 1j) * snp.zeros(self.v.shape), L, L.prox, 0.0)  # complex zero v
        pf = prox_test((1 + 1j) * snp.zeros(self.v.shape), L, L.prox, 1.0)  # complex zero v
