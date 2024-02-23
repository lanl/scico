import numpy as np

import pytest

import scico.random
from scico import functional, linop, loss, metric
from scico.examples import create_circular_phantom
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.optimize.pgm import AcceleratedPGM


def test_aniso_1d():
    N = 128
    g = np.linspace(0, 2 * np.pi, N, dtype=np.float32)
    x_gt = np.sin(2 * g)
    x_gt[x_gt > 0.5] = 0.5
    x_gt[x_gt < -0.5] = -0.5
    σ = 0.02
    noise, key = scico.random.randn(x_gt.shape, seed=0)
    y = x_gt + σ * noise

    λ = 5e-2
    f = loss.SquaredL2Loss(y=y)

    C = linop.FiniteDifference(input_shape=x_gt.shape, circular=True)
    g = λ * functional.L1Norm()
    solver = ADMM(
        f=f,
        g_list=[g],
        C_list=[C],
        rho_list=[1e1],
        x0=y,
        maxiter=50,
        subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-3, "maxiter": 20}),
    )
    x_tvdn = solver.solve()

    h = λ * functional.AnisotropicTVNorm()
    solver = AcceleratedPGM(f=f, g=h, L0=2e2, x0=y, maxiter=50)
    x_approx = solver.solve()

    assert metric.snr(x_tvdn, x_approx) > 50
    assert metric.rel_res(g(C(x_tvdn)), h(x_tvdn)) < 1e-6


class Test2D:
    def setup_method(self):
        N = 32
        x_gt = create_circular_phantom((N, N), [0.4 * N, 0.2 * N, 0.1 * N], [1, 0, 0.5])
        σ = 0.02
        noise, key = scico.random.randn(x_gt.shape, seed=0)
        y = x_gt + σ * noise
        self.x_gt = x_gt
        self.y = y

    @pytest.mark.parametrize("tvtype", ["aniso", "iso"])
    def test_2d(self, tvtype):
        x_gt = self.x_gt
        y = self.y

        λ = 5e-2
        f = loss.SquaredL2Loss(y=y)
        if tvtype == "aniso":
            g = λ * functional.L1Norm()
        else:
            g = λ * functional.L21Norm()
        C = linop.FiniteDifference(input_shape=x_gt.shape, circular=True)

        solver = ADMM(
            f=f,
            g_list=[g],
            C_list=[C],
            rho_list=[1e1],
            x0=y,
            maxiter=100,
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 25}),
        )
        x_tvdn = solver.solve()

        if tvtype == "aniso":
            h = λ * functional.AnisotropicTVNorm(circular=True)
        else:
            h = λ * functional.IsotropicTVNorm(circular=True)
        solver = AcceleratedPGM(
            f=f,
            g=h,
            L0=1e3,
            x0=y,
            maxiter=300,
        )
        x_aprx = solver.solve()

        assert metric.snr(x_tvdn, x_aprx) > 50
        assert metric.rel_res(g(C(x_tvdn)), h(x_tvdn)) < 1e-6
