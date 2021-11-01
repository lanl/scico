import numpy as np

import jax

import pytest

from scico.test.linop.test_linop import adjoint_test

try:
    from scico.linop.radon_astra import ParallelBeamProjector
except ImportError as e:
    pytest.skip("astra not installed", allow_module_level=True)

import scico

N = 128


def get_tol():
    if jax.devices()[0].device_kind == "cpu":
        rtol = 5e-5
    else:
        rtol = 7e-2
    return rtol


class ParallelBeamProjectorTest:
    def __init__(self, volume_geometry):
        N_proj = 180  # number of projection angles
        N_det = 384
        detector_spacing = 1
        angles = np.linspace(0, np.pi, N_proj, False)

        np.random.seed(1234)
        self.x = np.random.randn(N, N).astype(np.float32)
        self.y = np.random.randn(N_proj, N_det).astype(np.float32)
        self.A = ParallelBeamProjector(
            input_shape=(N, N),
            volume_geometry=volume_geometry,
            detector_spacing=1,
            det_count=384,
            angles=np.linspace(0, np.pi, 180, False),
        )


@pytest.fixture(params=[None, [-N / 2, N / 2, -N / 2, N / 2]])
def testobj(request):
    yield ParallelBeamProjectorTest(request.param)


def test_ATA_call(testobj):
    # Test for the call-based interface
    Ax = testobj.A(testobj.x)
    ATAx = testobj.A.adj(Ax)
    np.testing.assert_allclose(np.sum(testobj.x * ATAx), np.linalg.norm(Ax) ** 2, rtol=get_tol())


def test_ATA_matmul(testobj):
    # Test for the matmul interface
    Ax = testobj.A @ testobj.x
    ATAx = testobj.A.T @ Ax
    np.testing.assert_allclose(np.sum(testobj.x * ATAx), np.linalg.norm(Ax) ** 2, rtol=get_tol())


def test_AAT_call(testobj):
    # Test for the call-based interface
    ATy = testobj.A.adj(testobj.y)
    AATy = testobj.A(ATy)
    np.testing.assert_allclose(np.sum(testobj.y * AATy), np.linalg.norm(ATy) ** 2, rtol=get_tol())


def test_AAT_matmul(testobj):
    # Test for the matmul interface
    ATy = testobj.A.T @ testobj.y
    AATy = testobj.A @ ATy
    np.testing.assert_allclose(np.sum(testobj.y * AATy), np.linalg.norm(ATy) ** 2, rtol=get_tol())


def test_grad(testobj):
    # ensure that we can take grad on a function using our projector
    # grad || A(x) ||_2^2 == 2 A.T @ A x
    A = testobj.A
    x = testobj.x
    g = lambda x: jax.numpy.linalg.norm(A(x)) ** 2
    np.testing.assert_allclose(scico.grad(g)(x), 2 * A.adj(A(x)), rtol=get_tol())


def test_adjoint_grad(testobj):
    A = testobj.A
    x = testobj.x
    Ax = A @ x
    f = lambda y: jax.numpy.linalg.norm(A.T(y)) ** 2
    np.testing.assert_allclose(scico.grad(f)(Ax), 2 * A(A.adj(Ax)), rtol=get_tol())


def test_adjoint(testobj):
    A = testobj.A
    adjoint_test(A, rtol=get_tol())
