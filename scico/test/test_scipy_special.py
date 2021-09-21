import numpy as np

import pytest

import scico.scipy.special as ss
from scico.random import randn

# these are functions that take only a single ndarray as input
one_arg_funcs = [
    ss.digamma,
    ss.entr,
    ss.erf,
    ss.erfc,
    ss.erfinv,
    ss.expit,
    ss.gammaln,
    ss.i0,
    ss.i0e,
    ss.i1,
    ss.i1e,
    ss.ndtr,
    ss.log_ndtr,
    ss.logit,
    ss.ndtri,
]


@pytest.mark.parametrize("func", one_arg_funcs)
def test_one_arg_funcs(func):

    # blockarray array
    x, key = randn(((8, 8), (4,)), key=None)

    Fx = func(x)

    fx0 = func(x[0])
    fx1 = func(x[1])
    np.testing.assert_allclose(Fx[0].ravel(), fx0.ravel(), rtol=1e-4)
    np.testing.assert_allclose(Fx[1].ravel(), fx1.ravel(), rtol=1e-4)


def test_betainc():
    a, key = randn(((8, 8), (4,)), key=None)
    b, key = randn(((8, 8), (4,)), key=key)
    x, key = randn(((8, 8), (4,)), key=key)

    Fx = ss.betainc(a, b, x)
    fx0 = ss.betainc(a[0], b[0], x[0])
    fx1 = ss.betainc(a[1], b[1], x[1])
    np.testing.assert_allclose(Fx[0].ravel(), fx0.ravel(), rtol=1e-4)
    np.testing.assert_allclose(Fx[1].ravel(), fx1.ravel(), rtol=1e-4)


@pytest.mark.parametrize("func", [ss.gammainc, ss.gammaincc])
def test_gammainc(func):
    a, key = randn(((8, 8), (4,)), key=None)
    b, key = randn(((8, 8), (4,)), key=key)
    x, key = randn(((8, 8), (4,)), key=key)

    Fx = ss.betainc(a, b, x)
    fx0 = ss.betainc(a[0], b[0], x[0])
    fx1 = ss.betainc(a[1], b[1], x[1])
    np.testing.assert_allclose(Fx[0].ravel(), fx0.ravel(), rtol=1e-4)
    np.testing.assert_allclose(Fx[1].ravel(), fx1.ravel(), rtol=1e-4)


def test_multigammaln():
    x, key = randn(((8, 8), (4,)), key=None)
    d = 2

    Fx = ss.multigammaln(x, d)
    fx0 = ss.multigammaln(x[0], d)
    fx1 = ss.multigammaln(x[1], d)
    np.testing.assert_allclose(Fx[0].ravel(), fx0.ravel(), rtol=1e-4)
    np.testing.assert_allclose(Fx[1].ravel(), fx1.ravel(), rtol=1e-4)


@pytest.mark.parametrize("func", [ss.xlog1py, ss.xlogy])
def test_logs(func):
    x, key = randn(((8, 8), (4,)), key=None)
    y, key = randn(((8, 8), (4,)), key=key)

    Fx = func(x, y)
    fx0 = func(x[0], y[0])
    fx1 = func(x[1], y[1])
    np.testing.assert_allclose(Fx[0].ravel(), fx0.ravel(), rtol=1e-4)
    np.testing.assert_allclose(Fx[1].ravel(), fx1.ravel(), rtol=1e-4)


def test_zeta():
    x, key = randn(((8, 8), (4,)), key=None)
    y, key = randn(((8, 8), (4,)), key=None)

    Fx = ss.zeta(x, y)
    fx0 = ss.zeta(x[0], y[0])
    fx1 = ss.zeta(x[1], y[1])
    np.testing.assert_allclose(Fx[0].ravel(), fx0.ravel(), rtol=1e-4)
    np.testing.assert_allclose(Fx[1].ravel(), fx1.ravel(), rtol=1e-4)
