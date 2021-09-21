import numpy as np

import scico.numpy as snp
from scico import functional, linop, loss, metric, random
from scico.admm import (
    ADMM,
    CircularConvolveSolver,
    GenericSubproblemSolver,
    LinearSubproblemSolver,
)


class TestReal:
    def setup_method(self, method):
        np.random.seed(12345)
        MA = 9
        MB = 10
        N = 8
        # Set up arrays for problem argmin (1/2) ||A x - y||_2^2 + (λ/2) ||B x||_2^2
        Amx = np.random.randn(MA, N)
        Bmx = np.random.randn(MB, N)
        y = np.random.randn(MA)
        λ = 1e0
        self.Amx = Amx
        self.Bmx = Bmx
        self.y = y
        self.λ = λ
        # Solution of problem is given by linear system (A^T A + λ B^T B) x = A^T y
        self.grdA = lambda x: (Amx.T @ Amx + λ * Bmx.T @ Bmx) @ x
        self.grdb = Amx.T @ y

    def test_admm_generic(self):
        maxiter = 100
        ρ = 1e-1
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            verbose=False,
            x0=A.adj(self.y),
            subproblem_solver=GenericSubproblemSolver(
                minimize_kwargs={"options": {"maxiter": 100}}
            ),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4

    def test_admm_quadratic_scico(self):
        maxiter = 50
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            verbose=False,
            x0=A.adj(self.y),
            subproblem_solver=LinearSubproblemSolver(cg_function="scico"),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-5

    def test_admm_quadratic_jax(self):
        maxiter = 50
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            verbose=False,
            x0=A.adj(self.y),
            subproblem_solver=LinearSubproblemSolver(cg_function="jax"),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-5


class TestComplex:
    def setup_method(self, method):
        MA = 9
        MB = 10
        N = 8
        # Set up arrays for problem argmin (1/2) ||A x - y||_2^2 + (λ/2) ||B x||_2^2
        Amx, key = random.randn((MA, N), dtype=np.complex64, key=None)
        Bmx, key = random.randn((MB, N), dtype=np.complex64, key=key)
        y = np.random.randn(MA)
        λ = 1e0
        self.Amx = Amx
        self.Bmx = Bmx
        self.y = y
        self.λ = λ
        # Solution of problem is given by linear system (A^T A + λ B^T B) x = A^T y
        self.grdA = lambda x: (Amx.conj().T @ Amx + λ * Bmx.conj().T @ Bmx) @ x
        self.grdb = Amx.conj().T @ y

    def test_admm_generic(self):
        maxiter = 100
        ρ = 1e-1
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            verbose=False,
            x0=A.adj(self.y),
            subproblem_solver=GenericSubproblemSolver(
                minimize_kwargs={"options": {"maxiter": 100}}
            ),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4

    def test_admm_quadratic(self):
        maxiter = 50
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            verbose=False,
            x0=A.adj(self.y),
            subproblem_solver=LinearSubproblemSolver(),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-5


class TestCircularConvolveSolve:
    def setup_method(self, method):
        np.random.seed(12345)
        Nx = 8
        x = np.pad(np.ones((Nx, Nx)), Nx)
        Npsf = 3
        psf = snp.ones((Npsf, Npsf)) / (Npsf ** 2)
        self.A = linop.CircularConvolve(
            h=psf,
            input_shape=x.shape,
        )
        self.y = self.A(x)
        λ = 1e-2
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g_list = [λ * functional.L1Norm()]
        self.C_list = [linop.FiniteDifference(input_shape=x.shape, circular=True)]

    def test_admm(self):
        maxiter = 50
        ρ = 1e-1
        rho_list = [ρ]
        admm_lin = ADMM(
            f=self.f,
            g_list=self.g_list,
            C_list=self.C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            verbose=False,
            x0=self.A.adj(self.y),
            subproblem_solver=LinearSubproblemSolver(),
        )
        x_lin = admm_lin.solve()
        admm_dft = ADMM(
            f=self.f,
            g_list=self.g_list,
            C_list=self.C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            verbose=False,
            x0=self.A.adj(self.y),
            subproblem_solver=CircularConvolveSolver(),
        )
        x_dft = admm_dft.solve()
        np.testing.assert_allclose(x_dft, x_lin, atol=1e-4, rtol=0)
        assert metric.mse(x_lin, x_dft) < 1e-9
