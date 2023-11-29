import numpy as np

import scico.random
from scico import functional, linop, loss, metric
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.optimize.pgm import AcceleratedPGM


def test_tvnorm():

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
