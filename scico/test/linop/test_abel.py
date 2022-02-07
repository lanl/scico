import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.linop.abel import AbelProjector
from scico.test.linop.test_linop import adjoint_test

BIG_INPUT = (128, 128)
SMALL_INPUT = (4, 5)


def make_im(Nx, Ny):
    x, y = snp.meshgrid(snp.linspace(-1, 1, Nx), snp.linspace(-1, 1, Ny))

    im = snp.where(x ** 2 + y ** 2 < 0.3, 1.0, 0.0)

    return im


@pytest.mark.parametrize("Nx, Ny", (BIG_INPUT, SMALL_INPUT))
def test_inverse(Nx, Ny):
    im = make_im(Nx, Ny)
    A = AbelProjector(im.shape)

    Ax = A @ im
    im_hat = A.inverse(Ax)
    np.testing.assert_allclose(im_hat, im, rtol=5e-5)


@pytest.mark.parametrize("Nx, Ny", (BIG_INPUT, SMALL_INPUT))
def test_adjoint(Nx, Ny):
    im = make_im(Nx, Ny)
    A = AbelProjector(im.shape)
    adjoint_test(A)


@pytest.mark.parametrize("Nx, Ny", (BIG_INPUT, SMALL_INPUT))
def test_ATA(Nx, Ny):
    x = make_im(Nx, Ny)
    A = AbelProjector(x.shape)
    Ax = A(x)
    ATAx = A.adj(Ax)
    np.testing.assert_allclose(np.sum(x * ATAx), np.linalg.norm(Ax) ** 2, rtol=5e-5)


@pytest.mark.parametrize("Nx, Ny", (BIG_INPUT, SMALL_INPUT))
def test_grad(Nx, Ny):
    # ensure that we can take grad on a function using our projector
    # grad || A(x) ||_2^2 == 2 A.T @ A x
    x = make_im(Nx, Ny)
    A = AbelProjector(x.shape)
    g = lambda x: jax.numpy.linalg.norm(A(x)) ** 2
    np.testing.assert_allclose(jax.grad(g)(x), 2 * A.adj(A(x)), rtol=5e-5)


@pytest.mark.parametrize("Nx, Ny", (BIG_INPUT, SMALL_INPUT))
def test_adjoint_grad(Nx, Ny):
    x = make_im(Nx, Ny)
    A = AbelProjector(x.shape)
    Ax = A @ x
    f = lambda y: jax.numpy.linalg.norm(A.T(y)) ** 2
    np.testing.assert_allclose(jax.grad(f)(Ax), 2 * A(A.adj(Ax)), rtol=5e-5)
