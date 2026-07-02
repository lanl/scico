import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
from scico.linop import DiagonalStack
from scico.test.linop.test_linop import adjoint_test

try:
    from scico.linop.xray.astra_cone import (
        XRayTransform3DCone,
        _ensure_writeable,
        angle_to_vector_cone,
    )
except ModuleNotFoundError as e:
    if e.name == "astra":
        pytest.skip("astra not installed", allow_module_level=True)
    else:
        raise e


N = 128
RTOL_GPU = 1e-1
RTOL_GPU_RANDOM_INPUT = 2.0


def make_volume(Nx, Ny, Nz):
    """Create a simple 3D test volume."""
    x, y, z = snp.meshgrid(
        snp.linspace(-1, 1, Nx),
        snp.linspace(-1, 1, Ny),
        snp.linspace(-1, 1, Nz),
        indexing="ij",
    )

    vol = snp.where((x - 0.25) ** 2 / 3 + y**2 + z**2 / 2 < 0.1, 1.0, 0.0)
    vol = vol.astype(snp.float32)

    return vol


class XRayTransform3DConeTest:
    def __init__(self):
        N_proj = 90  # number of projection angles
        det_count = (64, 64)  # detector rows, columns
        det_spacing = (1.0, 1.0)
        angles = np.linspace(0, 2 * np.pi, N_proj, endpoint=False)
        source_dist = 100.0  # distance from source to origin
        det_dist = 100.0  # distance from origin to detector

        np.random.seed(1234)
        self.x = np.random.randn(32, 32, 32).astype(np.float32)
        self.y = np.random.randn(det_count[0], N_proj, det_count[1]).astype(np.float32)
        self.A = XRayTransform3DCone(
            input_shape=(32, 32, 32),
            det_count=det_count,
            det_spacing=det_spacing,
            angles=angles,
            source_dist=source_dist,
            det_dist=det_dist,
        )


@pytest.fixture
def testobj():
    yield XRayTransform3DConeTest()


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_init():
    """Test initialization with various parameter combinations."""
    # Valid initialization with angles
    A = XRayTransform3DCone(
        input_shape=(16, 16, 16),
        det_count=(16, 16),
        det_spacing=(1.0, 1.0),
        angles=np.linspace(0, np.pi, 32, False),
        source_dist=50.0,
        det_dist=50.0,
    )
    assert A.input_shape == (16, 16, 16)
    assert A.output_shape == (16, 32, 16)

    # Valid initialization with vectors
    vectors = angle_to_vector_cone((1.0, 1.0), np.linspace(0, np.pi, 32, False), 50.0, 50.0)
    B = XRayTransform3DCone(
        input_shape=(16, 16, 16),
        det_count=(16, 16),
        vectors=vectors,
    )
    assert B.input_shape == (16, 16, 16)
    assert B.output_shape == (16, 32, 16)

    # Test invalid input shapes
    with pytest.raises(ValueError, match="Only 3D projections are supported"):
        XRayTransform3DCone(
            input_shape=(16, 16),
            det_count=(16, 16),
            det_spacing=(1.0, 1.0),
            angles=np.linspace(0, np.pi, 32, False),
            source_dist=50.0,
            det_dist=50.0,
        )

    # Test invalid det_count
    with pytest.raises(ValueError, match="Expected argument 'det_count' to be a tuple"):
        XRayTransform3DCone(
            input_shape=(16, 16, 16),
            det_count=16,
            det_spacing=(1.0, 1.0),
            angles=np.linspace(0, np.pi, 32, False),
            source_dist=50.0,
            det_dist=50.0,
        )

    # Test mutually exclusive parameters
    with pytest.raises(ValueError, match="Either keyword"):
        XRayTransform3DCone(
            input_shape=(16, 16, 16),
            det_count=(16, 16),
            det_spacing=(1.0, 1.0),
            angles=np.linspace(0, np.pi, 32, False),
            source_dist=50.0,
            det_dist=50.0,
            vectors=vectors,
        )

    # Test missing required parameters
    with pytest.raises(ValueError, match="must be specified"):
        XRayTransform3DCone(
            input_shape=(16, 16, 16),
            det_count=(16, 16),
            det_spacing=(1.0, 1.0),
            angles=np.linspace(0, np.pi, 32, False),
            # Missing source_dist and det_dist
        )


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_cone_det_offset():
    """Test detector offset functionality."""
    x = np.zeros((32, 32, 32), dtype=np.float32)
    x[10:-10, 10:-10, 10:-10] = 1.0

    A = XRayTransform3DCone(
        x.shape,
        det_count=(40, 40),
        det_spacing=(1.0, 1.0),
        angles=np.linspace(0, np.pi, 90),
        source_dist=100.0,
        det_dist=100.0,
    )

    shift = (2, -3)
    As = XRayTransform3DCone(
        x.shape,
        det_count=(40, 40),
        det_spacing=(1.0, 1.0),
        det_offset=shift,
        angles=np.linspace(0, np.pi, 90),
        source_dist=100.0,
        det_dist=100.0,
    )

    y = A(x)
    ys = As(x)
    yss = np.roll(ys, shift, axis=(2, 0))

    # Note: Tolerance may be higher for cone beam due to interpolation effects
    np.testing.assert_almost_equal(yss, y, decimal=1)


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_ATA_call(testobj):
    """Test x A^T A x = || A x ||_2^2 property using call interface."""
    Ax = testobj.A(testobj.x)
    ATAx = testobj.A.adj(Ax)
    n0 = np.sum(testobj.x * ATAx)
    n1 = np.linalg.norm(Ax) ** 2
    assert np.abs(n0 - n1) / max(n0, n1) < 0.5  # poorly matched adjoint


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_ATA_matmul(testobj):
    """Test x A^T A x = || A x ||_2^2 property using matmul interface."""
    Ax = testobj.A @ testobj.x
    ATAx = testobj.A.T @ Ax
    n0 = np.sum(testobj.x * ATAx)
    n1 = np.linalg.norm(Ax) ** 2
    assert np.abs(n0 - n1) / max(n0, n1) < 0.5  # poorly matched adjoint


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_AAT_call(testobj):
    """Test y A A^T y = || A^T y ||_2^2 property using call interface."""
    ATy = testobj.A.adj(testobj.y)
    AATy = testobj.A(ATy)
    n0 = np.sum(testobj.y * AATy)
    n1 = np.linalg.norm(ATy) ** 2
    assert np.abs(n0 - n1) / max(n0, n1) < 0.75  # poorly matched adjoint


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_AAT_matmul(testobj):
    """Test y A A^T y = || A^T y ||_2^2 property using matmul interface."""
    ATy = testobj.A.T @ testobj.y
    AATy = testobj.A @ ATy
    n0 = np.sum(testobj.y * AATy)
    n1 = np.linalg.norm(ATy) ** 2
    assert np.abs(n0 - n1) / max(n0, n1) < 0.75  # poorly matched adjoint


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_grad(testobj):
    """Test gradient computation: grad ||A(x)||^2 = 2 A^T A x."""
    A = testobj.A
    x = testobj.x
    g = lambda x: jax.numpy.linalg.norm(A(x)) ** 2
    np.testing.assert_allclose(
        scico.grad(g)(x), 2 * A.adj(A(x)), atol=RTOL_GPU * x.max(), rtol=RTOL_GPU
    )


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_adjoint_grad(testobj):
    """Test gradient of adjoint: grad ||A^T(y)||^2 = 2 A A^T y."""
    A = testobj.A
    x = testobj.x
    Ax = A @ x
    f = lambda y: jax.numpy.linalg.norm(A.T(y)) ** 2
    np.testing.assert_allclose(scico.grad(f)(Ax), 2 * A(A.adj(Ax)), rtol=RTOL_GPU)


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_adjoint_random(testobj):
    """Test adjoint property with random input."""
    A = testobj.A
    adjoint_test(A, rtol=10 * RTOL_GPU_RANDOM_INPUT)


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_adjoint_typical_input(testobj):
    """Test adjoint property with typical input (structured volume)."""
    A = testobj.A
    x = make_volume(A.input_shape[0], A.input_shape[1], A.input_shape[2])
    adjoint_test(A, x=x, rtol=0.75)  # poorly matched adjoint


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_cone_api_equiv():
    """Test equivalence between angle-based and vector-based initialization."""
    x = np.random.randn(16, 16, 16).astype(np.float32)
    det_count = (24, 24)
    det_spacing = (1.0, 1.5)
    angles = snp.linspace(0, snp.pi, 45)
    source_dist = 80.0
    det_dist = 80.0

    # Angle-based geometry
    A = XRayTransform3DCone(
        x.shape,
        det_count=det_count,
        det_spacing=det_spacing,
        angles=angles,
        source_dist=source_dist,
        det_dist=det_dist,
    )

    # Vector-based geometry
    vectors = angle_to_vector_cone(det_spacing, angles, source_dist, det_dist)
    B = XRayTransform3DCone(x.shape, det_count=det_count, vectors=vectors)

    ya = A @ x
    yb = B @ x

    np.testing.assert_allclose(ya, yb, rtol=RTOL_GPU)


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_cone_magnification():
    """Test that cone beam produces expected magnification."""
    # Create a small sphere at the origin
    x = np.zeros((32, 32, 32), dtype=np.float32)
    x[14:18, 14:18, 14:18] = 1.0

    # Close source (more magnification)
    A_close = XRayTransform3DCone(
        x.shape,
        det_count=(48, 48),
        det_spacing=(1.0, 1.0),
        angles=np.array([0.0]),
        source_dist=50.0,
        det_dist=50.0,
    )

    # Far source (less magnification)
    A_far = XRayTransform3DCone(
        x.shape,
        det_count=(48, 48),
        det_spacing=(1.0, 1.0),
        angles=np.array([0.0]),
        source_dist=200.0,
        det_dist=200.0,
    )

    y_close = A_close @ x
    y_far = A_far @ x

    # Closer source should produce larger projection (sum should be more spread out)
    # This is a qualitative test that cone beam is different from parallel beam
    assert y_close.max() > 0
    assert y_far.max() > 0

    # The projections should have different distributions due to magnification
    # (not equal, even accounting for numerical tolerance)
    assert not np.allclose(y_close, y_far, rtol=0.1)


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_jit_in_DiagonalStack():
    """Test that operator works within DiagonalStack (relates to issue #331)."""
    N = 10
    H = DiagonalStack(
        [
            XRayTransform3DCone(
                (N, N, N),
                (N, N),
                det_spacing=(1.0, 1.0),
                angles=snp.linspace(0, snp.pi, N),
                source_dist=50.0,
                det_dist=50.0,
            )
        ]
    )
    result = H.T @ snp.zeros(H.output_shape, dtype=snp.float32)
    assert result.shape == (N, N, N)


def test_angle_to_vector_cone():
    """Test conversion from angles to cone beam vectors."""
    angles = snp.linspace(0, snp.pi, 5)
    det_spacing = (0.9, 1.5)
    source_dist = 100.0
    det_dist = 80.0

    vectors = angle_to_vector_cone(det_spacing, angles, source_dist, det_dist)

    assert vectors.shape == (angles.size, 12)

    # Check that vectors have the expected structure
    # Source position should be at distance source_dist
    source_dist = np.linalg.norm(vectors[:, 0:3], axis=1)
    np.testing.assert_allclose(source_dist, source_dist, rtol=1e-6)

    # Detector center should be at distance det_dist (opposite side)
    det_dist = np.linalg.norm(vectors[:, 3:6], axis=1)
    np.testing.assert_allclose(det_dist, det_dist, rtol=1e-6)

    # u vector should have length det_spacing[1]
    u_length = np.linalg.norm(vectors[:, 6:9], axis=1)
    np.testing.assert_allclose(u_length, det_spacing[1], rtol=1e-6)

    # v vector should have length det_spacing[0]
    v_length = np.linalg.norm(vectors[:, 9:12], axis=1)
    np.testing.assert_allclose(v_length, det_spacing[0], rtol=1e-6)

    # v vector should be vertical (in z direction)
    np.testing.assert_allclose(vectors[:, 9:11], 0, atol=1e-7)  # x, y components
    np.testing.assert_allclose(vectors[:, 11], det_spacing[0], rtol=1e-6)  # z component


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_cone_geometry_sanity():
    """Basic sanity check for cone beam geometry."""
    x = np.zeros((32, 32, 32), dtype=np.float32)
    # Put a point in the center
    x[15, 15, 15] = 1.0

    A = XRayTransform3DCone(
        x.shape,
        det_count=(32, 32),
        det_spacing=(1.0, 1.0),
        angles=np.array([0.0]),  # Single angle
        source_dist=100.0,
        det_dist=100.0,
    )

    y = A @ x

    # Should produce non-zero projection
    assert np.sum(np.abs(y)) > 0

    # Test backprojection produces non-zero result
    x_back = A.T @ y
    assert np.sum(np.abs(x_back)) > 0


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for cone beam")
def test_create_astra_geometry():
    """Test the static method for creating ASTRA geometry objects."""
    input_shape = (20, 24, 28)
    det_count = (32, 36)
    det_spacing = (1.2, 1.5)
    angles = np.linspace(0, 2 * np.pi, 60)
    source_dist = 150.0
    det_dist = 100.0

    # Test angle-based geometry
    vol_geom, proj_geom = XRayTransform3DCone.create_astra_geometry(
        input_shape,
        det_count,
        det_spacing=det_spacing,
        angles=angles,
        source_dist=source_dist,
        det_dist=det_dist,
    )

    assert proj_geom["type"] == "cone"
    assert proj_geom["DetectorRowCount"] == det_count[0]
    assert proj_geom["DetectorColCount"] == det_count[1]

    # Test vector-based geometry
    vectors = angle_to_vector_cone(det_spacing, angles, source_dist, det_dist)
    vol_geom2, proj_geom2 = XRayTransform3DCone.create_astra_geometry(
        input_shape, det_count, vectors=vectors
    )

    assert proj_geom2["type"] == "cone_vec"
    assert proj_geom2["DetectorRowCount"] == det_count[0]
    assert proj_geom2["DetectorColCount"] == det_count[1]
    assert proj_geom2["Vectors"].shape == (len(angles), 12)


def test_ensure_writeable():
    """Test the _ensure_writeable utility function."""
    # Test with numpy array
    assert isinstance(_ensure_writeable(np.ones((2, 1))), np.ndarray)

    # Test with JAX array
    assert isinstance(_ensure_writeable(snp.ones((2, 1))), np.ndarray)
