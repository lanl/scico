import os
import tempfile

import numpy as np

import pytest

import scico.numpy as snp
from scico import functional, linop, loss, metric, operator, random
from scico.optimize import ADMM
from scico.optimize.admm import (
    CircularConvolve3DSolver,
    CircularConvolveSolver,
    FBlockCircularConvolveSolver,
    G0BlockCircularConvolveSolver,
    GenericSubproblemSolver,
    LinearSubproblemSolver,
    MatrixSubproblemSolver,
)


class TestMisc:
    def setup_method(self, method):
        np.random.seed(12345)
        self.y = snp.array(np.random.randn(16, 17).astype(np.float32))

    def test_admm(self):
        maxiter = 2
        ρ = 1e-1
        A = linop.Identity(self.y.shape)
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = functional.DnCNN()
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

        with pytest.raises(TypeError):
            admm_ = ADMM(f=f, g_list=[g], C_list=[C], rho_list=[ρ], invalid_keyword_arg=None)

        admm_ = ADMM(f=f, g_list=[g], C_list=[C], rho_list=[ρ], maxiter=maxiter, nanstop=True)
        admm_.step()
        admm_.x = admm_.x.at[0].set(np.nan)
        with pytest.raises(ValueError):
            admm_.solve()

    @pytest.mark.parametrize(
        "solver", [LinearSubproblemSolver, MatrixSubproblemSolver, CircularConvolveSolver]
    )
    def test_admm_aux(self, solver):
        maxiter = 2
        ρ = 1e-1
        A = operator.Abs(self.y.shape)
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = functional.DnCNN()
        C = linop.Identity(self.y.shape)

        with pytest.raises(TypeError):
            admm_ = ADMM(
                f=f,
                g_list=[g],
                C_list=[C],
                rho_list=[ρ],
                maxiter=maxiter,
                subproblem_solver=solver(),
            )

        with pytest.raises(TypeError):
            admm_ = ADMM(
                f=g,
                g_list=[g],
                C_list=[C],
                rho_list=[ρ],
                maxiter=maxiter,
                subproblem_solver=solver(),
            )


class TestReal:
    def setup_method(self, method):
        np.random.seed(12345)
        MA = 4
        MB = 5
        N = 6
        # Set up arrays for problem argmin (𝛼/2) ||A x - y||_2^2 + (λ/2) ||B x||_2^2
        Amx = np.random.randn(MA, N).astype(np.float32)
        Bmx = np.random.randn(MB, N).astype(np.float32)
        y = np.random.randn(MA).astype(np.float32)
        𝛼 = np.pi  # sort of random number chosen to test non-default scale factor
        λ = 1e0
        self.Amx = Amx
        self.Bmx = Bmx
        self.y = snp.array(y)
        self.𝛼 = 𝛼
        self.λ = λ
        # Solution of problem is given by linear system (𝛼 A^T A + λ B^T B) x = 𝛼 A^T y
        self.grdA = lambda x: (𝛼 * Amx.T @ Amx + λ * Bmx.T @ Bmx) @ x
        self.grdb = 𝛼 * Amx.T @ y

    def test_admm_generic(self):
        maxiter = 25
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
            subproblem_solver=GenericSubproblemSolver(minimize_kwargs={"options": {"maxiter": 50}}),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-3

    def test_admm_saveload(self):
        maxiter = 5
        x_ref = np.ones((16, 16), dtype=np.float32)
        x_ref[4:-4, 4:-4] = 1.0
        n = 3
        psf = snp.ones((n, n), dtype=np.float32) / (n * n)
        A = linop.CircularConvolve(h=psf, input_shape=x_ref.shape)
        y = A(x_ref)
        λ = 2e-2
        ρ = 5e-1
        f = loss.SquaredL2Loss(y=y, A=A)
        g = λ * functional.L21Norm()
        C = linop.FiniteDifference(x_ref.shape, circular=True)
        admm0 = ADMM(
            f=f,
            g_list=[g],
            C_list=[C],
            rho_list=[ρ],
            x0=A.adj(y),
            maxiter=maxiter,
            subproblem_solver=CircularConvolveSolver(),
        )
        admm0.solve()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "admm.npz")
            admm0.save_state(path)
            admm0.solve()
            h0 = admm0.history()
            admm1 = ADMM(
                f=f,
                g_list=[g],
                C_list=[C],
                rho_list=[ρ],
                x0=A.adj(y),
                maxiter=maxiter,
                subproblem_solver=CircularConvolveSolver(),
            )
            admm1.load_state(path)
            admm1.solve()
            h1 = admm1.history()
            np.testing.assert_allclose(admm0.minimizer(), admm1.minimizer(), atol=1e-7)
            assert np.abs(h0[-1].Objective - h1[-1].Objective) < 1e-6

    def test_admm_quadratic_scico(self):
        maxiter = 25
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
        maxiter = 25
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
        maxiter = 25
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
        MA = 4
        MB = 5
        N = 6
        # Set up arrays for problem argmin (𝛼/2) ||A x - y||_W^2 + (λ/2) ||B x||_2^2
        Amx = np.random.randn(MA, N).astype(np.float32)
        W = np.abs(np.random.randn(MA, 1).astype(np.float32))
        Bmx = np.random.randn(MB, N).astype(np.float32)
        y = np.random.randn(MA).astype(np.float32)
        𝛼 = np.pi  # sort of random number chosen to test non-default scale factor
        λ = np.e
        self.Amx = Amx
        self.W = snp.array(W)
        self.Bmx = Bmx
        self.y = snp.array(y)
        self.𝛼 = 𝛼
        self.λ = λ
        # Solution of problem is given by linear system
        #   (𝛼 A^T W A + λ B^T B) x = 𝛼 A^T W y
        self.grdA = lambda x: (𝛼 * Amx.T @ (W * Amx) + λ * Bmx.T @ Bmx) @ x
        self.grdb = 𝛼 * Amx.T @ (W[:, 0] * y)

    def test_admm_quadratic_linear(self):
        maxiter = 100
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A, W=linop.Diagonal(self.W[:, 0]), scale=self.𝛼 / 2.0)
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

    def test_admm_quadratic_matrix(self):
        maxiter = 50
        ρ = 1e0
        A = linop.MatrixOperator(self.Amx)
        f = loss.SquaredL2Loss(y=self.y, A=A, W=linop.Diagonal(self.W[:, 0]), scale=self.𝛼 / 2.0)
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
            subproblem_solver=MatrixSubproblemSolver(),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-5


class TestComplex:
    def setup_method(self, method):
        MA = 4
        MB = 5
        N = 6
        # Set up arrays for problem argmin (𝛼/2) ||A x - y||_2^2 + (λ/2) ||B x||_2^2
        Amx, key = random.randn((MA, N), dtype=np.complex64, key=None)
        Bmx, key = random.randn((MB, N), dtype=np.complex64, key=key)
        y, key = random.randn((MA,), dtype=np.complex64, key=key)
        𝛼 = 1.0 / 3.0
        λ = 1e0
        self.Amx = Amx
        self.Bmx = Bmx
        self.y = y
        self.𝛼 = 𝛼
        self.λ = λ
        # Solution of problem is given by linear system (𝛼 A^T A + λ B^T B) x = A^T y
        self.grdA = lambda x: (𝛼 * Amx.conj().T @ Amx + λ * Bmx.conj().T @ Bmx) @ x
        self.grdb = 𝛼 * Amx.conj().T @ y

    def test_admm_generic(self):
        maxiter = 30
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
            subproblem_solver=GenericSubproblemSolver(minimize_kwargs={"options": {"maxiter": 50}}),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-3

    def test_admm_quadratic_linear(self):
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

    def test_admm_quadratic_matrix(self):
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
            subproblem_solver=MatrixSubproblemSolver(),
        )
        x = admm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-5


@pytest.mark.parametrize("extra_axis", (False, True))
@pytest.mark.parametrize("center", (None, [-1.0, 2.5]))
class TestCircularConvolveSolve:

    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(self, extra_axis, center):
        np.random.seed(12345)
        Nx = 8
        x = snp.pad(snp.ones((Nx, Nx), dtype=np.float32), Nx)
        Npsf = 3
        psf = snp.ones((Npsf, Npsf), dtype=np.float32) / (Npsf**2)
        if extra_axis:
            x = x[np.newaxis]
            psf = psf[np.newaxis]
        self.A = linop.CircularConvolve(
            h=psf, input_shape=x.shape, ndims=2, input_dtype=np.float32, h_center=center
        )
        self.y = self.A(x)
        λ = 1e-2
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g_list = [λ * functional.L1Norm()]
        self.C_list = [linop.FiniteDifference(input_shape=x.shape, circular=True)]
        yield

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
        assert admm_dft.subproblem_solver.A_lhs.ndims == 2
        x_dft = admm_dft.solve()
        np.testing.assert_allclose(x_dft, x_lin, atol=1e-4, rtol=0)
        assert metric.mse(x_lin, x_dft) < 1e-9


@pytest.mark.parametrize("center", (None, [-1.0, 2.5, 0.0]))
class TestCircularConvolve3DSolve:

    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(self, center):
        np.random.seed(12345)
        Nx = 8
        x = snp.pad(snp.ones((Nx, Nx, 1), dtype=np.float32), Nx)
        Npsf = 3
        psf = snp.ones((Npsf, Npsf, 1), dtype=np.float32) / (Npsf**2)
        self.A = linop.CircularConvolve3D(
            h=psf, input_shape=x.shape, input_dtype=np.float32, h_center=center
        )
        self.y = self.A(x)
        λ = 1e-2
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g_list = [λ * functional.L1Norm()]
        self.C_list = [linop.FiniteDifference(input_shape=x.shape, circular=True)]
        yield

    def test_admm(self):
        maxiter = 100
        ρ = 8e-2
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
            subproblem_solver=CircularConvolve3DSolver(),
        )
        assert admm_dft.subproblem_solver.A_lhs.ndims == 3
        x_dft = admm_dft.solve()
        np.testing.assert_allclose(x_dft, x_lin, atol=5e-4, rtol=0)
        assert metric.mse(x_lin, x_dft) < 1e-9


@pytest.mark.parametrize("with_cconv", (False, True))
class TestSpecialCaseCircularConvolveSolve:

    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(self, with_cconv):
        np.random.seed(12345)
        Nx = 8
        x = snp.pad(snp.ones((1, Nx, Nx), dtype=np.float32), Nx)
        if with_cconv:
            Npsf = 3
            psf = snp.ones((1, Npsf, Npsf), dtype=np.float32) / (Npsf**2)
            C0 = linop.CircularConvolve(h=psf, input_shape=x.shape, ndims=2, input_dtype=np.float32)
        else:
            C0 = linop.FiniteDifference(input_shape=x.shape, axes=(1, 2), circular=True)
        C1 = linop.Identity(input_shape=x.shape)
        self.y = C0(x)
        self.g_list = [loss.SquaredL2Loss(y=self.y), functional.L2Norm()]
        self.C_list = [C0, C1]
        self.with_cconv = with_cconv
        yield

    def test_admm(self):
        maxiter = 50
        ρ = 1e-1
        rho_list = [ρ, ρ]
        admm_lin = ADMM(
            f=None,
            g_list=self.g_list,
            C_list=self.C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            x0=self.C_list[0].adj(self.y),
            subproblem_solver=LinearSubproblemSolver(),
        )
        x_lin = admm_lin.solve()
        ndims = None if self.with_cconv else 2
        admm_dft = ADMM(
            f=None,
            g_list=self.g_list,
            C_list=self.C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            x0=self.C_list[0].adj(self.y),
            subproblem_solver=CircularConvolveSolver(ndims=ndims),
        )
        assert admm_dft.subproblem_solver.A_lhs.ndims == 2
        x_dft = admm_dft.solve()
        np.testing.assert_allclose(x_dft, x_lin, atol=1e-4, rtol=0)
        assert metric.mse(x_lin, x_dft) < 1e-9


class TestBlockCircularConvolveSolve:
    def setup_method(self, method):
        np.random.seed(12345)
        Nx = 8
        x = np.zeros((2, Nx, Nx), dtype=np.float32)
        x[0, 2, 2] = 1.0
        x[1, 3, 3] = 1.0
        Npsf = 3
        psf = np.zeros((2, Npsf, Npsf), dtype=np.float32)
        psf[0, 1] = 1.0
        psf[1, :, 1] = 1.0
        C = linop.CircularConvolve(h=psf, input_shape=x.shape, input_dtype=np.float32, ndims=2)
        S = linop.Sum(input_shape=x.shape, axis=0)
        self.A = S @ C
        self.y = self.A(x)
        λ = 1e-1
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g_list = [λ * functional.L1Norm()]
        self.C_list = [linop.Identity(input_shape=x.shape)]

    def test_fblock_init(self):
        with pytest.raises(ValueError):
            slvr = ADMM(
                f=None,
                g_list=self.g_list,
                C_list=self.C_list,
                rho_list=[1.0],
                itstat_options={"display": False},
                subproblem_solver=FBlockCircularConvolveSolver(),
            )
        with pytest.raises(TypeError):
            slvr = ADMM(
                f=loss.PoissonLoss(y=self.y),
                g_list=self.g_list,
                C_list=self.C_list,
                rho_list=[1.0],
                itstat_options={"display": False},
                subproblem_solver=FBlockCircularConvolveSolver(),
            )
        with pytest.raises(TypeError):
            slvr = ADMM(
                f=loss.SquaredL2Loss(y=self.y, A=self.A.A),
                g_list=self.g_list,
                C_list=self.C_list,
                rho_list=[1.0],
                itstat_options={"display": False},
                subproblem_solver=FBlockCircularConvolveSolver(),
            )

    def test_g0block_init(self):
        with pytest.raises(ValueError):
            slvr = ADMM(
                f=self.f,
                g_list=self.g_list,
                C_list=self.C_list,
                rho_list=[1.0],
                itstat_options={"display": False},
                subproblem_solver=G0BlockCircularConvolveSolver(),
            )
        with pytest.raises(TypeError):
            slvr = ADMM(
                f=functional.ZeroFunctional(),
                g_list=[loss.PoissonLoss(y=self.y)],
                C_list=self.C_list,
                rho_list=[1.0],
                itstat_options={"display": False},
                subproblem_solver=G0BlockCircularConvolveSolver(),
            )
        with pytest.raises(TypeError):
            slvr = ADMM(
                f=functional.ZeroFunctional(),
                g_list=[loss.SquaredL2Loss(y=self.y)] + self.g_list,
                C_list=[self.A.A] + self.C_list,
                rho_list=[1.0, 1.0],
                itstat_options={"display": False},
                subproblem_solver=G0BlockCircularConvolveSolver(),
            )

    def test_solve(self):
        maxiter = 50
        ρ = 1e1
        rho_list = [ρ]
        admm_lin = ADMM(
            f=self.f,
            g_list=self.g_list,
            C_list=self.C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            subproblem_solver=LinearSubproblemSolver(),
        )
        x_lin = admm_lin.solve()

        admm_dft1 = ADMM(
            f=self.f,
            g_list=self.g_list,
            C_list=self.C_list,
            rho_list=rho_list,
            maxiter=maxiter,
            itstat_options={"display": False},
            subproblem_solver=FBlockCircularConvolveSolver(check_solve=True),
        )
        x_dft1 = admm_dft1.solve()
        np.testing.assert_allclose(x_dft1, x_lin, atol=1e-4, rtol=0)
        assert metric.mse(x_lin, x_dft1) < 1e-9
        assert admm_dft1.subproblem_solver.accuracy <= 1e-6

        admm_dft2 = ADMM(
            f=functional.ZeroFunctional(),
            g_list=[loss.SquaredL2Loss(y=self.y)] + self.g_list,
            C_list=[self.A] + self.C_list,
            rho_list=[1.0, ρ],
            maxiter=maxiter,
            itstat_options={"display": False},
            subproblem_solver=G0BlockCircularConvolveSolver(check_solve=True),
        )
        admm_dft2.z_list[0] = self.y  # significantly improves convergence
        x_dft2 = admm_dft2.solve()
        np.testing.assert_allclose(x_dft2, x_lin, atol=1e-4, rtol=0)
        assert metric.mse(x_lin, x_dft2) < 1e-9
        assert admm_dft2.subproblem_solver.accuracy <= 1e-6
