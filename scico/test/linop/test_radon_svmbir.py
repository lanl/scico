import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
from scico.linop import Diagonal
from scico.loss import SquaredL2Loss
from scico.test.functional.prox import prox_test
from scico.test.linop.test_linop import adjoint_test

try:
    import svmbir

    from scico.linop.radon_svmbir import (
        SVMBIRExtendedLoss,
        SVMBIRSquaredL2Loss,
        TomographicProjector,
    )
except ImportError as e:
    pytest.skip("svmbir not installed", allow_module_level=True)


BIG_INPUT = (32, 33, 50, 51, 125, 1.2)
SMALL_INPUT = (4, 5, 7, 8, 16, 1.2)

BIG_INPUT_OFFSET_RANGE = (0, 3)
SMALL_INPUT_OFFSET_RANGE = (0, 0.1)

device = jax.devices()[0]


def make_im(Nx, Ny, is_3d=True):
    x, y = snp.meshgrid(snp.linspace(-1, 1, Nx), snp.linspace(-1, 1, Ny))

    im = snp.where((x - 0.25) ** 2 / 3 + y**2 < 0.1, 1.0, 0.0)
    if is_3d:
        im = im[snp.newaxis, :, :]
    im = im.astype(snp.float32)

    return im


def make_angles(num_angles):
    return snp.linspace(0, snp.pi, num_angles, dtype=snp.float32)


def make_A(
    im,
    num_angles,
    num_channels,
    center_offset,
    is_masked,
    geometry="parallel",
    dist_source_detector=None,
    magnification=None,
    delta_channel=None,
    delta_pixel=None,
):

    angles = make_angles(num_angles)
    A = TomographicProjector(
        im.shape,
        angles,
        num_channels,
        center_offset=center_offset,
        is_masked=is_masked,
        geometry=geometry,
        dist_source_detector=dist_source_detector,
        magnification=magnification,
    )

    return A


@pytest.mark.parametrize(
    "Nx, Ny, num_angles, num_channels, dist_source_detector, magnification", (BIG_INPUT,)
)
@pytest.mark.parametrize("is_3d", (True, False))
@pytest.mark.parametrize("center_offset", BIG_INPUT_OFFSET_RANGE)
@pytest.mark.parametrize("is_masked", (True, False))
@pytest.mark.parametrize("geometry", ("parallel", "fan-curved", "fan-flat"))
def test_grad(
    Nx,
    Ny,
    num_angles,
    num_channels,
    is_3d,
    center_offset,
    is_masked,
    geometry,
    dist_source_detector,
    magnification,
):
    im = make_im(Nx, Ny, is_3d)
    A = make_A(
        im,
        num_angles,
        num_channels,
        center_offset,
        is_masked,
        geometry=geometry,
        dist_source_detector=dist_source_detector,
        magnification=magnification,
    )

    def f(im):
        return snp.sum(A._eval(im) ** 2)

    val_1 = jax.grad(f)(im)
    val_2 = 2 * A.adj(A(im))

    np.testing.assert_allclose(val_1, val_2)


@pytest.mark.parametrize(
    "Nx, Ny, num_angles, num_channels, dist_source_detector, magnification", (BIG_INPUT,)
)
@pytest.mark.parametrize("is_3d", (True, False))
@pytest.mark.parametrize("center_offset", BIG_INPUT_OFFSET_RANGE)
@pytest.mark.parametrize("is_masked", (True, False))
@pytest.mark.parametrize("geometry", ("parallel", "fan-curved", "fan-flat"))
def test_adjoint(
    Nx,
    Ny,
    num_angles,
    num_channels,
    is_3d,
    center_offset,
    is_masked,
    geometry,
    dist_source_detector,
    magnification,
):
    im = make_im(Nx, Ny, is_3d)
    A = make_A(
        im,
        num_angles,
        num_channels,
        center_offset,
        is_masked,
        geometry=geometry,
        dist_source_detector=dist_source_detector,
        magnification=magnification,
    )

    adjoint_test(A)


@pytest.mark.skipif(device.platform != "cpu", reason="test hangs on gpu")
@pytest.mark.parametrize(
    "Nx, Ny, num_angles, num_channels, dist_source_detector, magnification", (SMALL_INPUT,)
)
@pytest.mark.parametrize("is_3d", (True, False))
@pytest.mark.parametrize("center_offset", SMALL_INPUT_OFFSET_RANGE)
@pytest.mark.parametrize("is_masked", (True, False))
@pytest.mark.parametrize("geometry", ("parallel", "fan-curved", "fan-flat"))
def test_prox(
    Nx,
    Ny,
    num_angles,
    num_channels,
    is_3d,
    center_offset,
    is_masked,
    geometry,
    dist_source_detector,
    magnification,
):

    im = make_im(Nx, Ny, is_3d)
    A = make_A(
        im,
        num_angles,
        num_channels,
        center_offset,
        is_masked,
        geometry=geometry,
        dist_source_detector=dist_source_detector,
        magnification=magnification,
    )

    sino = A @ im

    v, _ = scico.random.normal(im.shape, dtype=im.dtype)

    if is_masked:
        f = SVMBIRExtendedLoss(y=sino, A=A, positivity=False, prox_kwargs={"maxiter": 5})
    else:
        f = SVMBIRSquaredL2Loss(y=sino, A=A, prox_kwargs={"maxiter": 5})

    prox_test(v, f, f.prox, alpha=0.25, rtol=5e-4)


@pytest.mark.skipif(device.platform != "cpu", reason="test hangs on gpu")
@pytest.mark.parametrize(
    "Nx, Ny, num_angles, num_channels, dist_source_detector, magnification", (SMALL_INPUT,)
)
@pytest.mark.parametrize("is_3d", (True, False))
@pytest.mark.parametrize("center_offset", SMALL_INPUT_OFFSET_RANGE)
@pytest.mark.parametrize("is_masked", (True, False))
@pytest.mark.parametrize("geometry", ("parallel", "fan-curved", "fan-flat"))
def test_prox_weights(
    Nx,
    Ny,
    num_angles,
    num_channels,
    is_3d,
    center_offset,
    is_masked,
    geometry,
    dist_source_detector,
    magnification,
):

    im = make_im(Nx, Ny, is_3d)
    A = make_A(
        im,
        num_angles,
        num_channels,
        center_offset,
        is_masked,
        geometry=geometry,
        dist_source_detector=dist_source_detector,
        magnification=magnification,
    )

    sino = A @ im

    v, _ = scico.random.normal(im.shape, dtype=im.dtype)

    # test with weights
    weights, _ = scico.random.uniform(sino.shape, dtype=im.dtype)
    W = scico.linop.Diagonal(weights)

    if is_masked:
        f = SVMBIRExtendedLoss(y=sino, A=A, W=W, positivity=False, prox_kwargs={"maxiter": 5})
    else:
        f = SVMBIRSquaredL2Loss(y=sino, A=A, W=W, prox_kwargs={"maxiter": 5})

    prox_test(v, f, f.prox, alpha=0.25, rtol=5e-5)


@pytest.mark.parametrize(
    "Nx, Ny, num_angles, num_channels, dist_source_detector, magnification", (SMALL_INPUT,)
)
@pytest.mark.parametrize("is_3d", (True, False))
@pytest.mark.parametrize("weight_type", ("transmission", "unweighted"))
@pytest.mark.parametrize("center_offset", SMALL_INPUT_OFFSET_RANGE)
@pytest.mark.parametrize("is_masked", (True, False))
@pytest.mark.parametrize("geometry", ("parallel", "fan-curved", "fan-flat"))
def test_prox_cg(
    Nx,
    Ny,
    num_angles,
    num_channels,
    is_3d,
    weight_type,
    center_offset,
    is_masked,
    geometry,
    dist_source_detector,
    magnification,
):

    im = make_im(Nx, Ny, is_3d=is_3d) / Nx * 10
    A = make_A(
        im,
        num_angles,
        num_channels,
        center_offset,
        is_masked=is_masked,
        geometry=geometry,
        dist_source_detector=dist_source_detector,
        magnification=magnification,
    )
    y = A @ im

    A_colsum = A.H @ snp.ones(
        y.shape, dtype=snp.float32
    )  # backproject ones to get sum over cols of A
    if is_masked:
        mask = np.asarray(A_colsum) > 0  # cols of A which are not all zeros
    else:
        mask = np.ones(im.shape) > 0

    W = svmbir.calc_weights(y, weight_type=weight_type).astype("float32")
    W = jax.device_put(W)
    λ = 0.01

    if is_masked:
        f_sv = SVMBIRExtendedLoss(
            y=y, A=A, W=Diagonal(W), positivity=False, prox_kwargs={"maxiter": 5}
        )
    else:
        f_sv = SVMBIRSquaredL2Loss(y=y, A=A, W=Diagonal(W), prox_kwargs={"maxiter": 5})

    f_wg = SquaredL2Loss(y=y, A=A, W=Diagonal(W), prox_kwargs={"tol": 5e-4})

    v, _ = scico.random.normal(im.shape, dtype=im.dtype)
    v *= im.max() * 0.5

    xprox_sv = f_sv.prox(v, λ)
    xprox_cg = f_wg.prox(v, λ)  # this uses cg

    assert snp.linalg.norm(xprox_sv[mask] - xprox_cg[mask]) / snp.linalg.norm(xprox_sv[mask]) < 5e-4


@pytest.mark.parametrize(
    "Nx, Ny, num_angles, num_channels, dist_source_detector, magnification", (SMALL_INPUT,)
)
@pytest.mark.parametrize("is_3d", (True, False))
@pytest.mark.parametrize("weight_type", ("transmission", "unweighted"))
@pytest.mark.parametrize("center_offset", SMALL_INPUT_OFFSET_RANGE)
@pytest.mark.parametrize("is_masked", (True, False))
@pytest.mark.parametrize("positivity", (True, False))
@pytest.mark.parametrize("geometry", ("parallel", "fan-curved", "fan-flat"))
def test_approx_prox(
    Nx,
    Ny,
    num_angles,
    num_channels,
    is_3d,
    weight_type,
    center_offset,
    is_masked,
    positivity,
    geometry,
    dist_source_detector,
    magnification,
):
    im = make_im(Nx, Ny, is_3d)
    A = make_A(
        im,
        num_angles,
        num_channels,
        center_offset,
        is_masked,
        geometry=geometry,
        dist_source_detector=dist_source_detector,
        magnification=magnification,
    )

    y = A @ im

    W = svmbir.calc_weights(y, weight_type=weight_type).astype("float32")
    W = jax.device_put(W)
    λ = 0.01

    v, _ = scico.random.normal(im.shape, dtype=im.dtype)
    if is_masked or positivity:
        f = SVMBIRExtendedLoss(
            y=y, A=A, W=Diagonal(W), positivity=positivity, prox_kwargs={"maxiter": 5}
        )
    else:
        f = SVMBIRSquaredL2Loss(y=y, A=A, W=Diagonal(W), prox_kwargs={"maxiter": 5})

    xprox = snp.array(f.prox(v, lam=λ))

    if is_masked or positivity:
        f_approx = SVMBIRExtendedLoss(
            y=y, A=A, W=Diagonal(W), prox_kwargs={"maxiter": 2}, positivity=positivity
        )
    else:
        f_approx = SVMBIRSquaredL2Loss(y=y, A=A, W=Diagonal(W), prox_kwargs={"maxiter": 2})

    xprox_approx = snp.array(f_approx.prox(v, lam=λ, v0=xprox))

    assert snp.linalg.norm(xprox - xprox_approx) / snp.linalg.norm(xprox) < 4e-6
