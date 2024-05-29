import numpy as np

import pytest

import scico.random
from scico import functional, linop, loss, metric
from scico.examples import create_circular_phantom
from scico.functional._tvnorm import HaarTransform, SingleAxisHaarTransform
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.optimize.pgm import AcceleratedPGM


@pytest.mark.parametrize("axis", [0, 1])
def test_single_axis_haar_transform(axis):
    x, key = scico.random.randn((3, 4), seed=1234)
    HT = SingleAxisHaarTransform(x.shape, axis=axis)
    np.testing.assert_allclose(x, HT.T(HT(x)), rtol=1e-6)


def test_haar_transform():
    x, key = scico.random.randn((3, 4), seed=1234)
    HT = HaarTransform(x.shape)
    np.testing.assert_allclose(2 * x, HT.T(HT(x)), rtol=1e-6)


@pytest.mark.parametrize("circular", [True, False])
def test_aniso_1d(circular):
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

    C = linop.FiniteDifference(
        input_shape=x_gt.shape, circular=circular, append=None if circular else 0
    )
    g = λ * functional.L1Norm()
    solver = ADMM(
        f=f,
        g_list=[g],
        C_list=[C],
        rho_list=[1e1],
        x0=y,
        maxiter=50,
        subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 25}),
    )
    x_tvdn = solver.solve()

    h = λ * functional.AnisotropicTVNorm(circular=circular, input_shape=y.shape)
    solver = AcceleratedPGM(f=f, g=h, L0=5e2, x0=y, maxiter=100)
    x_approx = solver.solve()

    assert metric.snr(x_tvdn, x_approx) > 50
    assert metric.rel_res(g(C(x_tvdn)), h(x_tvdn)) < 1e-6


class Test2D:
    def setup_method(self):
        N = 32
        x_gt = create_circular_phantom(
            (N, N), [0.6 * N, 0.4 * N, 0.2 * N, 0.1 * N], [0.25, 1, 0, 0.5]
        ).astype(np.float32)
        gr, gc = np.ogrid[0:N, 0:N]
        x_gt += ((gr + gc) / (4 * N)).astype(np.float32)
        σ = 0.02
        noise, key = scico.random.randn(x_gt.shape, seed=0, dtype=np.float32)
        y = x_gt + σ * noise
        self.x_gt = x_gt
        self.y = y

    @pytest.mark.parametrize("circular", [True, False])
    @pytest.mark.parametrize("tvtype", ["aniso", "iso"])
    def test_2d(self, tvtype, circular):
        x_gt = self.x_gt
        y = self.y

        λ = 5e-2
        f = loss.SquaredL2Loss(y=y)
        if tvtype == "aniso":
            g = λ * functional.L1Norm()
        else:
            g = λ * functional.L21Norm()
        C = linop.FiniteDifference(
            input_shape=x_gt.shape, circular=circular, append=None if circular else 0
        )

        solver = ADMM(
            f=f,
            g_list=[g],
            C_list=[C],
            rho_list=[1e1],
            x0=y,
            maxiter=150,
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 25}),
        )
        x_tvdn = solver.solve()

        if tvtype == "aniso":
            h = λ * functional.AnisotropicTVNorm(circular=circular, input_shape=y.shape)
        else:
            h = λ * functional.IsotropicTVNorm(circular=circular, input_shape=y.shape)

        solver = AcceleratedPGM(
            f=f,
            g=h,
            L0=1e3,
            x0=y,
            maxiter=400,
        )
        x_aprx = solver.solve()

        assert metric.snr(x_tvdn, x_aprx) > 50
        assert metric.rel_res(g(C(x_tvdn)), h(x_tvdn)) < 1e-6


class Test3D:
    def setup_method(self):
        N = 32
        x2d = create_circular_phantom(
            (N, N), [0.6 * N, 0.4 * N, 0.2 * N, 0.1 * N], [0.25, 1, 0, 0.5]
        ).astype(np.float32)
        gr, gc = np.ogrid[0:N, 0:N]
        x2d += ((gr + gc) / (4 * N)).astype(np.float32)
        x_gt = np.stack((0.9 * x2d, np.zeros(x2d.shape), 1.1 * x2d), dtype=np.float32)
        σ = 0.02
        noise, key = scico.random.randn(x_gt.shape, seed=0, dtype=np.float32)
        y = x_gt + σ * noise
        self.x_gt = x_gt
        self.y = y

    @pytest.mark.parametrize("circular", [False])
    @pytest.mark.parametrize("tvtype", ["iso"])
    def test_3d(self, tvtype, circular):
        x_gt = self.x_gt
        y = self.y

        λ = 5e-2
        f = loss.SquaredL2Loss(y=y)
        if tvtype == "aniso":
            g = λ * functional.L1Norm()
        else:
            g = λ * functional.L21Norm()
        C = linop.FiniteDifference(
            input_shape=x_gt.shape, axes=(1, 2), circular=circular, append=None if circular else 0
        )

        solver = ADMM(
            f=f,
            g_list=[g],
            C_list=[C],
            rho_list=[5e0],
            x0=y,
            maxiter=150,
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 25}),
        )
        x_tvdn = solver.solve()

        if tvtype == "aniso":
            h = λ * functional.AnisotropicTVNorm(
                circular=circular, axes=(1, 2), input_shape=y.shape
            )
        else:
            h = λ * functional.IsotropicTVNorm(circular=circular, axes=(1, 2), input_shape=y.shape)

        solver = AcceleratedPGM(
            f=f,
            g=h,
            L0=1e3,
            x0=y,
            maxiter=400,
        )
        x_aprx = solver.solve()

        assert metric.snr(x_tvdn, x_aprx) > 50
        assert metric.rel_res(g(C(x_tvdn)), h(x_tvdn)) < 1e-6
