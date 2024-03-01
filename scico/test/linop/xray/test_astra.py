import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
from scico.linop import DiagonalStack
from scico.test.linop.test_linop import adjoint_test
from scico.test.linop.xray.test_svmbir import make_im

try:
    from scico.linop.xray.astra import XRayTransform2D, XRayTransform3D, angle_to_vector
except ModuleNotFoundError as e:
    if e.name == "astra":
        pytest.skip("astra not installed", allow_module_level=True)
    else:
        raise e


N = 128
RTOL_CPU = 5e-5
RTOL_GPU = 7e-2
RTOL_GPU_RANDOM_INPUT = 1.0


def get_tol():
    if jax.devices()[0].device_kind == "cpu":
        rtol = RTOL_CPU
    else:
        rtol = RTOL_GPU  # astra inaccurate in GPU
    return rtol


def get_tol_random_input():
    if jax.devices()[0].device_kind == "cpu":
        rtol = RTOL_CPU
    else:
        rtol = RTOL_GPU_RANDOM_INPUT  # astra more inaccurate in GPU for random inputs
    return rtol


class XRayTransform2DTest:
    def __init__(self, volume_geometry):
        N_proj = 180  # number of projection angles
        N_det = 384
        det_spacing = 1
        angles = np.linspace(0, np.pi, N_proj, False)

        np.random.seed(1234)
        self.x = np.random.randn(N, N).astype(np.float32)
        self.y = np.random.randn(N_proj, N_det).astype(np.float32)
        self.A = XRayTransform2D(
            input_shape=(N, N),
            det_count=N_det,
            det_spacing=det_spacing,
            angles=angles,
            volume_geometry=volume_geometry,
        )


@pytest.fixture(params=[None, [-N / 2, N / 2, -N / 2, N / 2]])
def testobj(request):
    yield XRayTransform2DTest(request.param)


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


def test_adjoint_random(testobj):
    A = testobj.A
    adjoint_test(A, rtol=get_tol_random_input())


def test_adjoint_typical_input(testobj):
    A = testobj.A
    x = make_im(A.input_shape[0], A.input_shape[1], is_3d=False)

    adjoint_test(A, x=x, rtol=get_tol())


def test_jit_in_DiagonalStack():
    """See https://github.com/lanl/scico/issues/331"""
    N = 10
    H = DiagonalStack([XRayTransform2D((N, N), N, 1.0, snp.linspace(0, snp.pi, N))])
    H.T @ snp.zeros(H.output_shape, dtype=snp.float32)


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="checking GPU behavior")
def test_3D_on_GPU():
    x = snp.zeros((4, 5, 6))
    A = XRayTransform3D(
        x.shape, det_count=[6, 6], det_spacing=[1.0, 1.0], angles=snp.linspace(0, snp.pi, 10)
    )

    assert A.num_dims == 3
    y = A @ x
    ATy = A.T @ y


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for test")
def test_3D_api_equiv():
    x = np.random.randn(4, 5, 6).astype(np.float32)
    det_count = [7, 8]
    det_spacing = [1.0, 1.5]
    angles = snp.linspace(0, snp.pi, 10)
    A = XRayTransform3D(x.shape, det_count=det_count, det_spacing=det_spacing, angles=angles)
    vectors = angle_to_vector(det_spacing, angles)
    B = XRayTransform3D(x.shape, det_count=det_count, vectors=vectors)
    ya = A @ x
    yb = B @ x
    np.testing.assert_allclose(ya, yb, rtol=get_tol())


def test_angle_to_vector():
    angles = snp.linspace(0, snp.pi, 5)
    det_spacing = [0.9, 1.5]
    vectors = angle_to_vector(det_spacing, angles)
    assert vectors.shape == (angles.size, 12)
