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


@pytest.mark.parametrize("Nx, Ny", (BIG_INPUT,))
def test_grad(Nx, Ny):
    im = make_im(Nx, Ny)

    A = AbelProjector(im.shape)

    def f(im):
        return snp.sum(A._eval(im) ** 2)

    val_1 = jax.grad(f)(im)
    # val_2 = 2 * A.adj(A(im))
    val_2 = 2 * A._adj(A(im))
    # val_2 = 2 * A.T @ A @ im

    np.testing.assert_allclose(val_1, val_2, rtol=5e-5)


@pytest.mark.parametrize("Nx, Ny", (BIG_INPUT,))
def test_adjoint(Nx, Ny):
    im = make_im(Nx, Ny)
    A = AbelProjector(im.shape)

    adjoint_test(A)


# @pytest.mark.parametrize("Nx, Ny", (BIG_INPUT,))
# def test_adjoint_grad(Nx, Ny):
#     x = make_im(Nx, Ny)
#     A = AbelProjector(x.shape)
#     Ax = A @ x

#     f = lambda y: jax.numpy.linalg.norm(A.T(y)) ** 2
#     # print(scico.grad(f)(Ax))
#     # print(2 * A(A.adj(Ax)))
#     np.testing.assert_allclose(scico.grad(f)(Ax), 2 * A(A._adj(Ax)), rtol=5e-5)
