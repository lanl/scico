import numpy as np

import scico.numpy as snp
from scico.solver import minimize


def prox_func(x, v, f, alpha):
    """Evaluate functional of which the proximal operator is the argmin."""
    return 0.5 * snp.sum(snp.abs(x.reshape(v.shape) - v) ** 2) + alpha * snp.array(
        f(x.reshape(v.shape)), dtype=snp.float64
    )


def prox_solve(v, v0, f, alpha):
    """Evaluate the alpha-scaled proximal operator of f at v, using v0 as an
    initial point for the optimization."""
    fnc = lambda x: prox_func(x, v, f, alpha)
    fmn = minimize(
        fnc,
        v0,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-9, "fatol": 1e-9},
    )

    return fmn.x.reshape(v.shape), fmn.fun


def prox_test(v, nrm, prx, alpha, x0=None, rtol=1e-6):
    """Test the alpha-scaled proximal operator function prx of norm functional nrm
    at point v."""
    # Evaluate the proximal operator at v
    px = snp.array(prx(v, alpha, v0=x0))
    # Proximal operator functional value (i.e. Moreau envelope) at v
    pf = prox_func(px, v, nrm, alpha)
    # Brute-force solve of the proximal operator at v
    mx, mf = prox_solve(v, px, nrm, alpha)

    # Compare prox functional value with brute-force solution
    if pf < mf:
        return  # prox gave a lower cost than brute force, so it passes

    np.testing.assert_allclose(pf, mf, rtol=rtol)
