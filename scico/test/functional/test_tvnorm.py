import numpy as np

import pytest

try:
    from xdesign import SiemensStar, discrete_phantom

    have_xdesign = True
except ImportError:
    have_xdesign = False

import scico.numpy as snp
import scico.random
from scico import functional, linop, loss, metric
from scico.optimize import AcceleratedPGM, ProximalADMM
from scico.optimize.admm import ADMM, LinearSubproblemSolver


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

    assert metric.snr(x_tvdn, x_approx) > 45


@pytest.mark.skipif(not have_xdesign, reason="xdesign package not installed")
class Test2D:
    def setup_method(self):
        N = 128
        σ = 0.25
        phantom = SiemensStar(16)
        x_gt = snp.pad(discrete_phantom(phantom, N - 16), 8)
        x_gt = x_gt / x_gt.max()
        noise, key = scico.random.randn(x_gt.shape, seed=0)
        y = x_gt + σ * noise

        self.x_gt = x_gt
        self.y = y

    def test_aniso(self):
        x_gt = self.x_gt
        y = self.y

        λ = 2e-1
        f = loss.SquaredL2Loss(y=y)
        g = λ * functional.L1Norm()
        C = linop.FiniteDifference(input_shape=x_gt.shape, circular=True)

        mu, nu = ProximalADMM.estimate_parameters(C)
        solver = ProximalADMM(
            f=f,
            g=g,
            A=C,
            rho=1e0,
            mu=mu,
            nu=nu,
            x0=y,
            maxiter=200,
        )
        x = solver.solve()

        h = λ * functional.AnisotropicTVNorm()
        solver = AcceleratedPGM(
            f=f,
            g=h,
            L0=1e3,
            x0=y,
            maxiter=300,
        )
        x_aprx = solver.solve()

        assert metric.snr(x, x_aprx) > 30.0

    def test_iso(self):
        x_gt = self.x_gt
        y = self.y

        λ = 2e-1
        f = loss.SquaredL2Loss(y=y)
        g = λ * functional.L21Norm()
        C = linop.FiniteDifference(input_shape=x_gt.shape, circular=True)

        mu, nu = ProximalADMM.estimate_parameters(C)
        solver = ProximalADMM(
            f=f,
            g=g,
            A=C,
            rho=1e0,
            mu=mu,
            nu=nu,
            x0=y,
            maxiter=200,
        )
        x = solver.solve()

        h = λ * functional.IsotropicTVNorm()
        solver = AcceleratedPGM(
            f=f,
            g=h,
            L0=1e3,
            x0=y,
            maxiter=300,
        )
        x_aprx = solver.solve()

        assert metric.snr(x, x_aprx) > 20.0
