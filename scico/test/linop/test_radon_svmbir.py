import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
from scico.test.linop.test_linop import adjoint_AAt_test, adjoint_AtA_test
from scico.test.test_functional import prox_test

try:
    from scico.linop.radon_svmbir import (
        ParallelBeamProjector,
        SvmbirWeightedSquaredL2Loss,
    )
except ImportError as e:
    pytest.skip("svmbir not installed", allow_module_level=True)


def make_im(Nx, Ny):
    x, y = snp.meshgrid(snp.linspace(-1, 1, Nx), snp.linspace(-1, 1, Ny))

    im = snp.where((x - 0.25) ** 2 / 3 + y ** 2 < 0.1, 1.0, 0.0)
    im = im[snp.newaxis, :, :]
    im = im.astype(snp.float32)

    return im


@pytest.fixture
def im():
    # make everything different sizes to catch dimension swap bugs
    Nx, Ny = 128, 129

    return make_im(Nx, Ny)


@pytest.fixture
def im_small():
    # make everything different sizes to catch dimension swap bugs
    Nx, Ny = 4, 5

    return make_im(Nx, Ny)


def make_A(im, num_angles, num_channels):
    angles = snp.linspace(0, snp.pi, num_angles, dtype=snp.float32)
    A = ParallelBeamProjector(im.shape, angles, num_channels)

    return A


@pytest.fixture
def A(im):
    num_angles = 200
    num_channels = 201

    return make_A(im, num_angles, num_channels)


@pytest.fixture
def A_small(im_small):
    num_angles = 7
    num_channels = 8

    return make_A(im_small, num_angles, num_channels)


def test_grad(A, im):
    def f(im):
        return snp.sum(A._eval(im) ** 2)

    val_1 = jax.grad(f)(im)
    val_2 = 2 * A.adj(A(im))

    np.testing.assert_allclose(val_1, val_2)


def test_adjoint(A):
    adjoint_AtA_test(A)
    adjoint_AAt_test(A)


def test_prox(im_small, A_small):
    A, im = A_small, im_small
    sino = A @ im

    v, _ = scico.random.normal(im.shape, dtype=im.dtype)
    f = SvmbirWeightedSquaredL2Loss(y=sino, A=A)
    prox_test(v, f, f.prox, alpha=0.25)


def test_prox_weights(im_small, A_small):
    A, im = A_small, im_small
    sino = A @ im

    v, _ = scico.random.normal(im.shape, dtype=im.dtype)

    # test with weights
    weights, _ = scico.random.uniform(sino.shape, dtype=im.dtype)
    D = scico.linop.Diagonal(weights)
    f = SvmbirWeightedSquaredL2Loss(y=sino, A=A, weight_op=D)
    prox_test(v, f, f.prox, alpha=0.25)
