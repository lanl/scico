import numpy as np

import pytest

import scico.numpy as snp
from scico import functional, linop, loss, metric
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.optimize.pgm import AcceleratedPGM


def test_proxavg_init():
    g0 = functional.L1Norm()
    g1 = functional.L2Norm()

    with pytest.raises(ValueError):
        h = functional.ProximalAverage(
            [g0, g1],
            alpha_list=[
                0.1,
            ],
        )

    h = functional.ProximalAverage([g0, g1], alpha_list=[0.1, 0.1])
    assert sum(h.alpha_list) == 1.0

    g1.has_prox = False
    with pytest.raises(ValueError):
        h = functional.ProximalAverage([g0, g1])


def test_proxavg():
    N = 128
    g = np.linspace(0, 2 * np.pi, N, dtype=np.float32)
    y = np.sin(2 * g)
    y[y > 0.5] = 0.5
    y[y < -0.5] = -0.5
    y *= 2
    y = snp.array(y)

    λ0 = 6e-1
    λ1 = 6e-1
    f = loss.SquaredL2Loss(y=y)
    g0 = λ0 * functional.L1Norm()
    g1 = λ1 * functional.L2Norm()

    solver = ADMM(
        f=f,
        g_list=[0.5 * g0, 0.5 * g1],
        C_list=[linop.Identity(y.shape), linop.Identity(y.shape)],
        rho_list=[1e1, 1e1],
        x0=y,
        maxiter=100,
        subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-5, "maxiter": 20}),
    )
    x_admm = solver.solve()

    h = functional.ProximalAverage([λ0 * functional.L1Norm(), λ1 * functional.L2Norm()])
    solver = AcceleratedPGM(f=f, g=h, L0=3.4e2, x0=y, maxiter=250)
    x_prxavg = solver.solve()

    assert metric.snr(x_admm, x_prxavg) > 50
