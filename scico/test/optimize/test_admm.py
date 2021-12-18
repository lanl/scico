import numpy as np

import jax

import scico.numpy as snp
from scico import functional, linop, loss, metric, random
from scico.optimize import ADMM
from scico.optimize.admm import (
    CircularConvolveSolver,
    GenericSubproblemSolver,
    LinearSubproblemSolver,
)


class TestMisc:
    def setup_method(self, method):
        np.random.seed(12345)
        self.y = jax.device_put(np.random.randn(32, 33).astype(np.float32))
        self.λ = 1e0

    def test_admm(self):
        maxiter = 2
        ρ = 1e-1
        A = linop.Identity(self.y.shape)
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.BM3D()
        C = linop.Identity(self.y.shape)

        itstat_fields = {"Iter": "%d", "Time": "%8.2e"}

        def itstat_func(obj):
            return (obj.itnum, obj.timer.elapsed())

        admm_ = ADMM(
            f=f,
            g_list=[g],
            C_list=[C],
            rho_list=[ρ],
            maxiter=maxiter,
            itstat_options={"display": False},
        )
        assert len(admm_.itstat_object.fieldname) == 6
        assert snp.sum(admm_.x) == 0.0
        admm_ = ADMM(
            f=f,
            g_list=[g],
            C_list=[C],
            rho_list=[ρ],
            maxiter=maxiter,
            itstat_options={"fields": itstat_fields, "itstat_func": itstat_func, "display": False},
        )
        assert len(admm_.itstat_object.fieldname) == 2

        admm_.test_flag = False

        def callback(obj):
            obj.test_flag = True

        x = admm_.solve(callback=callback)
        assert admm_.test_flag


class TestReal:
    def setup_method(self, method):
        np.random.seed(12345)
        MA = 9
        MB = 10
        N = 8
        # Set up arrays for problem argmin (𝛼/2) ||A x - y||_2^2 + (λ/2) ||B x||_2^2
        Amx = np.random.randn(MA, N)
        Bmx = np.random.randn(MB, N)
        y = np.random.randn(MA)
        𝛼 = np.pi  # sort of random number chosen to test non-default scale factor
        λ = 1e0
        self.Amx = Amx
        self.Bmx = Bmx
        self.y = jax.device_put(y)
        self.𝛼 = 𝛼
        self.λ = λ
        # Solution of problem is given by linear system (𝛼 A^T A + λ B^T B) x = 𝛼 A^T y
        self.grdA = lambda x: (𝛼 * Amx.T @ Amx + λ * Bmx.T @ Bmx) @ x
        self.grdb = 𝛼 * Amx.T @ y

    def test_admm_generic(self):
        maxiter = 100
        ρ = 2e-1
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A, scale=self.𝛼 / 2.0)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            x0=A.adj(self.y),
            subproblem_solver=GenericSubproblemSolver(
                minimize_kwargs={"options": {"maxiter": 100}}
            ),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4

    def test_admm_quadratic_scico(self):
        maxiter = 50
        ρ = 4e-1
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A, scale=self.𝛼 / 2.0)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            x0=A.adj(self.y),
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4}, cg_function="scico"),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4

    def test_admm_quadratic_jax(self):
        maxiter = 50
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A, scale=self.𝛼 / 2.0)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            x0=A.adj(self.y),
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4}, cg_function="jax"),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4

    def test_admm_quadratic_relax(self):
        maxiter = 50
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A, scale=self.𝛼 / 2.0)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            alpha=1.6,
            maxiter=maxiter,
            itstat_options={"display": False},
            x0=A.adj(self.y),
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4}, cg_function="jax"),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4


class TestRealWeighted:
    def setup_method(self, method):
        np.random.seed(12345)
        MA = 9
        MB = 10
        N = 8
        # Set up arrays for problem argmin (𝛼/2) ||A x - y||_W^2 + (λ/2) ||B x||_2^2
        Amx = np.random.randn(MA, N)
        W = np.abs(np.random.randn(MA, 1))
        Bmx = np.random.randn(MB, N)
        y = np.random.randn(MA)
        𝛼 = np.pi  # sort of random number chosen to test non-default scale factor
        λ = np.e
        self.Amx = Amx
        self.W = jax.device_put(W)
        self.Bmx = Bmx
        self.y = jax.device_put(y)
        self.𝛼 = 𝛼
        self.λ = λ
        # Solution of problem is given by linear system
        #   (𝛼 A^T W A + λ B^T B) x = 𝛼 A^T W y
        self.grdA = lambda x: (𝛼 * Amx.T @ (W * Amx) + λ * Bmx.T @ Bmx) @ x
        self.grdb = 𝛼 * Amx.T @ (W[:, 0] * y)

    def test_admm_quadratic(self):
        maxiter = 100
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.WeightedSquaredL2Loss(
            y=self.y, A=A, W=linop.Diagonal(self.W[:, 0]), scale=self.𝛼 / 2.0
        )
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            x0=A.adj(self.y),
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4}, cg_function="scico"),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4


class TestComplex:
    def setup_method(self, method):
        MA = 9
        MB = 10
        N = 8
        # Set up arrays for problem argmin (𝛼/2) ||A x - y||_2^2 + (λ/2) ||B x||_2^2
        Amx, key = random.randn((MA, N), dtype=np.complex64, key=None)
        Bmx, key = random.randn((MB, N), dtype=np.complex64, key=key)
        y = np.random.randn(MA)
        𝛼 = 1.0 / 3.0
        λ = 1e0
        self.Amx = Amx
        self.Bmx = Bmx
        self.y = jax.device_put(y)
        self.𝛼 = 𝛼
        self.λ = λ
        # Solution of problem is given by linear system (𝛼 A^T A + λ B^T B) x = A^T y
        self.grdA = lambda x: (𝛼 * Amx.conj().T @ Amx + λ * Bmx.conj().T @ Bmx) @ x
        self.grdb = 𝛼 * Amx.conj().T @ y

    def test_admm_generic(self):
        maxiter = 100
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A, scale=self.𝛼 / 2.0)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            x0=A.adj(self.y),
            subproblem_solver=GenericSubproblemSolver(
                minimize_kwargs={"options": {"maxiter": 100}}
            ),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-3

    def test_admm_quadratic(self):
        maxiter = 50
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A, scale=self.𝛼 / 2.0)
        g_list = [(self.λ / 2) * functional.SquaredL2Norm()]
        C_list = [linop.MatrixOperator(self.Bmx)]
        rho_list = [ρ]
        admm_ = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            x0=A.adj(self.y),
            subproblem_solver=LinearSubproblemSolver(
                cg_kwargs={"tol": 1e-4},
            ),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4


class TestCircularConvolveSolve:
    def setup_method(self, method):
        np.random.seed(12345)
        Nx = 8
        x = np.pad(np.ones((Nx, Nx), dtype=np.float32), Nx)
        Npsf = 3
        psf = snp.ones((Npsf, Npsf), dtype=np.float32) / (Npsf ** 2)
        self.A = linop.CircularConvolve(
            h=psf,
            input_shape=x.shape,
            input_dtype=np.float32,
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
            itstat_options={"display": False},
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
            itstat_options={"display": False},
            x0=self.A.adj(self.y),
            subproblem_solver=CircularConvolveSolver(),
        )
        x_dft = admm_dft.solve()
        np.testing.assert_allclose(x_dft, x_lin, atol=1e-4, rtol=0)
        assert metric.mse(x_lin, x_dft) < 1e-9
