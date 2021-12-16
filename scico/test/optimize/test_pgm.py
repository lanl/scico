import numpy as np

import jax

from scico import functional, linop, loss, random
from scico.optimize import PGM, AcceleratedPGM
from scico.optimize.pgm import (
    AdaptiveBBStepSize,
    BBStepSize,
    LineSearchStepSize,
    RobustLineSearchStepSize,
)


class TestSet:
    def setup_method(self, method):
        np.random.seed(12345)
        M = 9
        N = 8
        # Set up arrays for problem argmin (1/2) ||A x - y||_2^2 + (λ/2) ||B x||_2^2
        Amx = np.random.randn(M, N)
        Bmx = np.identity(N)
        y = jax.device_put(np.random.randn(M))
        λ = 1e0
        self.Amx = Amx
        self.y = y
        self.λ = λ
        # Solution of problem is given by linear system (A^T A + λ B^T B) x = A^T y
        self.grdA = lambda x: (Amx.T @ Amx + λ * Bmx.T @ Bmx) @ x
        self.grdb = Amx.T @ y

    def test_pgm(self):
        maxiter = 200

        A = linop.MatrixOperator(self.Amx)
        L0 = 1.05 * linop.power_iteration(A.T @ A)[0]
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        pgm_ = PGM(f=loss_, g=g, L0=L0, maxiter=maxiter, x0=A.adj(self.y))
        x = pgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_accelerated_pgm(self):
        maxiter = 200

        A = linop.MatrixOperator(self.Amx)
        L0 = 1.05 * linop.power_iteration(A.T @ A)[0]
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        apgm_ = AcceleratedPGM(f=loss_, g=g, L0=L0, maxiter=maxiter, x0=A.adj(self.y))
        x = apgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_pgm_BB_step_size(self):
        maxiter = 200
        A = linop.MatrixOperator(self.Amx)
        L0 = 1.05 * linop.power_iteration(A.T @ A)[0] / 5.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        pgm_ = PGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=BBStepSize(),
            maxiter=maxiter,
        )
        x = pgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_pgm_adaptive_BB_step_size(self):
        maxiter = 200
        A = linop.MatrixOperator(self.Amx)
        L0 = 1.05 * linop.power_iteration(A.T @ A)[0] / 5.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        pgm_ = PGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=AdaptiveBBStepSize(),
            maxiter=maxiter,
        )
        x = pgm_.solve()

    def test_accelerated_pgm_line_search(self):
        maxiter = 150
        A = linop.MatrixOperator(self.Amx)
        L0 = 1.05 * linop.power_iteration(A.T @ A)[0] / 5.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        apgm_ = AcceleratedPGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=LineSearchStepSize(gamma_u=1.03, maxiter=55),
            maxiter=maxiter,
        )
        x = apgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_accelerated_pgm_robust_line_search(self):
        maxiter = 100
        A = linop.MatrixOperator(self.Amx)
        L0 = 1.05 * linop.power_iteration(A.T @ A)[0] / 5.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        apgm_ = AcceleratedPGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=RobustLineSearchStepSize(gamma_d=0.95, gamma_u=1.05, maxiter=80),
            maxiter=maxiter,
        )
        x = apgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_pgm_BB_step_size_jit(self):
        maxiter = 200
        A = linop.MatrixOperator(self.Amx)
        L0 = 1.05 * linop.power_iteration(A.T @ A)[0] / 5.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        pgm_ = PGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=BBStepSize(),
            maxiter=maxiter,
        )
        x = pgm_.x
        try:
            update_step = jax.jit(pgm_.step_size.update)
            L = update_step(x)
        except Exception as e:
            print(e)
            assert 0

    def test_accelerated_pgm_adaptive_BB_step_size_jit(self):
        maxiter = 200
        A = linop.MatrixOperator(self.Amx)
        L0 = 1.05 * linop.power_iteration(A.T @ A)[0] / 5.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        apgm_ = AcceleratedPGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=AdaptiveBBStepSize(),
            maxiter=maxiter,
        )
        x = apgm_.x
        try:
            update_step = jax.jit(apgm_.step_size.update)
            L = update_step(x)
        except Exception as e:
            print(e)
            assert 0


class TestComplex:
    def setup_method(self, method):
        M = 9
        N = 8
        # Set up arrays for problem argmin (1/2) ||A x - y||_2^2 + (λ/2) ||x||_2^2
        Amx, key = random.randn((M, N), dtype=np.complex64, key=None)
        Bmx = np.identity(N)
        y = jax.device_put(np.random.randn(M))
        λ = 1e0
        self.Amx = Amx
        self.Bmx = Bmx
        self.y = y
        self.λ = λ
        # Solution of problem is given by linear system (A^T A + λ B^T B) x = A^T y
        self.grdA = lambda x: (Amx.conj().T @ Amx + λ * Bmx.T @ Bmx) @ x
        self.grdb = Amx.conj().T @ y

    def test_pgm(self):
        maxiter = 200
        A = linop.MatrixOperator(self.Amx)
        L0 = 50.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        pgm_ = PGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            maxiter=maxiter,
        )
        x = pgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_accelerated_pgm(self):
        maxiter = 200
        A = linop.MatrixOperator(self.Amx)
        L0 = 50.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        apgm_ = AcceleratedPGM(f=loss_, g=g, L0=L0, x0=A.adj(self.y), maxiter=maxiter)
        x = apgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_pgm_BB_step_size(self):
        maxiter = 200
        A = linop.MatrixOperator(self.Amx)
        L0 = 10.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        pgm_ = PGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=BBStepSize(),
            maxiter=maxiter,
        )
        x = pgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_pgm_adaptive_BB_step_size(self):
        maxiter = 200
        A = linop.MatrixOperator(self.Amx)
        L0 = 10.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        pgm_ = PGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=AdaptiveBBStepSize(),
            maxiter=maxiter,
        )
        x = pgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_accelerated_pgm_line_search(self):
        maxiter = 200
        A = linop.MatrixOperator(self.Amx)
        L0 = 10.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        apgm_ = AcceleratedPGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=LineSearchStepSize(gamma_u=1.03, maxiter=55),
            maxiter=maxiter,
        )
        x = apgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)

    def test_accelerated_pgm_robust_line_search(self):
        maxiter = 100
        A = linop.MatrixOperator(self.Amx)
        L0 = 10.0
        loss_ = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2.0) * functional.SquaredL2Norm()
        apgm_ = AcceleratedPGM(
            f=loss_,
            g=g,
            L0=L0,
            x0=A.adj(self.y),
            step_size=RobustLineSearchStepSize(gamma_d=0.95, gamma_u=1.05, maxiter=80),
            maxiter=maxiter,
        )
        x = apgm_.solve()
        np.testing.assert_allclose(self.grdA(x), self.grdb, rtol=5e-3)
