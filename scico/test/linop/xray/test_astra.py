import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
from scico.linop import DiagonalStack
from scico.test.linop.test_linop import adjoint_test
from scipy.spatial.transform import Rotation

try:
    from scico.linop.xray.astra import (
        XRayTransform2D,
        XRayTransform3D,
        _ensure_writeable,
        angle_to_vector,
        rotate_vectors,
    )
except ModuleNotFoundError as e:
    if e.name == "astra":
        pytest.skip("astra not installed", allow_module_level=True)
    else:
        raise e


N = 128
RTOL_CPU = 1e-4
RTOL_GPU = 1e-1
RTOL_GPU_RANDOM_INPUT = 2.0


def make_im(Nx, Ny, is_3d=True):
    x, y = snp.meshgrid(snp.linspace(-1, 1, Nx), snp.linspace(-1, 1, Ny), indexing="ij")

    im = snp.where((x - 0.25) ** 2 / 3 + y**2 < 0.1, 1.0, 0.0)
    if is_3d:
        im = im[snp.newaxis, :, :]
    im = im.astype(snp.float32)

    return im


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


def test_init(testobj):
    with pytest.raises(ValueError):
        A = XRayTransform2D(
            input_shape=(16, 16, 16),
            det_count=16,
            det_spacing=1.0,
            angles=np.linspace(0, np.pi, 32, False),
        )
    with pytest.raises(ValueError):
        A = XRayTransform2D(
            input_shape=(16, 16),
            det_count=16.3,
            det_spacing=1.0,
            angles=np.linspace(0, np.pi, 32, False),
        )
    with pytest.raises(ValueError):
        A = XRayTransform2D(
            input_shape=(16, 16),
            det_count=16,
            det_spacing=1.0,
            angles=np.linspace(0, np.pi, 32, False),
            device="invalid",
        )


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
    np.testing.assert_allclose(
        scico.grad(g)(x), 2 * A.adj(A(x)), atol=get_tol() * x.max(), rtol=np.inf
    )


def test_adjoint_grad(testobj):
    A = testobj.A
    x = testobj.x
    Ax = A @ x
    f = lambda y: jax.numpy.linalg.norm(A.T(y)) ** 2
    np.testing.assert_allclose(scico.grad(f)(Ax), 2 * A(A.adj(Ax)), rtol=get_tol())


def test_adjoint_random(testobj):
    A = testobj.A
    adjoint_test(A, rtol=10 * get_tol_random_input())


def test_adjoint_typical_input(testobj):
    A = testobj.A
    x = make_im(A.input_shape[0], A.input_shape[1], is_3d=False)

    adjoint_test(A, x=x, rtol=get_tol())


def test_fbp(testobj):
    x = testobj.A.fbp(testobj.y)
    # Test for a bug (related to calling the Astra CPU FBP implementation
    # when using a FPU device) that resulted in a constant zero output.
    assert np.sum(np.abs(x)) > 0.0


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


def test_rotate_vectors():
    v0 = angle_to_vector([1.0, 1.0], np.linspace(0, np.pi / 2, 4, endpoint=False))
    v1 = angle_to_vector([1.0, 1.0], np.linspace(np.pi / 2, np.pi, 4, endpoint=False))
    r = Rotation.from_euler("z", np.pi / 2)
    v0r = rotate_vectors(v0, r)
    np.testing.assert_allclose(v1, v0r, atol=1e-7)


## conversion functions
@pytest.fixture(scope="module")
def test_geometry():
    """
    In this geometry, if vol[i, j, k]==1, we expect proj[j-2, k-1]==1.

    Because:
    - We project along z, i.e. `ray=(0,0,1)`, i.e., we remove axis=0.
    - We set `v=(0, 1, 0)`, so detector rows go with y axis, axis=1.
    - We set `u=(1, 0, 0)`, so detector columns go with x axis, axis=2.
    - We shift the detector by (x=1, y=2, z=3) <-> i-3, j-2, k-1
    """
    in_shape = (30, 31, 32)
    # in ASTRA terminology:
    n_rows = in_shape[1]  # y
    n_cols = in_shape[2]  # x
    n_slices = in_shape[0]  # z
    vol_geom = scico.linop.xray.astra.astra.create_vol_geom(n_rows, n_cols, n_slices)

    assert vol_geom["option"]["WindowMinX"] == -n_cols / 2
    assert vol_geom["option"]["WindowMinY"] == -n_rows / 2
    assert vol_geom["option"]["WindowMinZ"] == -n_slices / 2

    # project along z, axis=0
    det_row_count = n_rows
    det_col_count = n_cols
    ray = (0, 0, 1)
    d = (1, 2, 3)  # axis=2 offset by 1, axis=1 offset by 2, axis=0 offset by 3
    u = (1, 0, 0)  # increments columns, goes with X
    v = (0, 1, 0)  # increments rows, goes with Y
    vectors = np.array(ray + d + u + v)[np.newaxis, :]
    proj_geom = scico.linop.xray.astra.astra.create_proj_geom(
        "parallel3d_vec", det_row_count, det_col_count, vectors
    )

    return vol_geom, proj_geom


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for test")
def test_projection_convention(test_geometry):
    """
    If vol[i, j, k]==1, test that astra puts proj[j-2, k-1]==1.

    See `test_geometry` for the setup.
    """
    vol_geom, proj_geom = test_geometry
    in_shape = scico.linop.xray.astra.astra.functions.geom_size(vol_geom)
    vol = np.zeros(in_shape)

    i, j, k = [np.random.randint(0, s) for s in in_shape]
    vol[i, j, k] = 1.0

    proj_id, proj = scico.linop.xray.astra.astra.create_sino3d_gpu(vol, proj_geom, vol_geom)
    scico.linop.xray.astra.astra.data3d.delete(proj_id)
    proj = proj[:, 0, :]  # get first view
    assert len(np.unique(proj) == 2)

    idx_proj_i, idx_proj_j = np.nonzero(proj)
    np.testing.assert_array_equal(idx_proj_i, j - 2)
    np.testing.assert_array_equal(idx_proj_j, k - 1)


def test_project_coords(test_geometry):
    """
    If vol[i, j, k]==1, test that we predict proj[j-2, k-1]==1.

    See `test_geometry` for the setup and `test_projection_convention`
    for proof ASTRA works this way.
    """
    vol_geom, proj_geom = test_geometry
    in_shape = scico.linop.xray.astra.astra.functions.geom_size(vol_geom)
    x_vol = np.array([np.random.randint(0, s) for s in in_shape])
    x_proj_gt = np.array(
        [[x_vol[1] - 2, x_vol[2] - 1]]
    )  # projection along slices removes first index
    x_proj = scico.linop.xray.astra._project_coords(x_vol, vol_geom, proj_geom)
    np.testing.assert_array_equal(x_proj_gt, x_proj)


def test_convert_to_scico_geometry(test_geometry):
    """
    Basic regression test, `test_project_coords` tests the logic.
    """
    vol_geom, proj_geom = test_geometry
    matrices_truth = scico.linop.xray.astra._astra_to_scico_geometry(vol_geom, proj_geom)
    truth = np.array([[[0.0, 1.0, 0.0, -2.0], [0.0, 0.0, 1.0, -1.0]]])
    np.testing.assert_allclose(matrices_truth, truth)


def test_convert_from_scico_geometry(test_geometry):
    """
    Basic regression test, `test_project_coords` tests the logic.
    """
    in_shape = (30, 31, 32)
    matrices = np.array([[[0.0, 1.0, 0.0, -2.0], [0.0, 0.0, 1.0, -1.0]]])
    det_shape = (31, 32)
    vectors = scico.linop.xray.astra.convert_from_scico_geometry(in_shape, matrices, det_shape)

    _, proj_geom_truth = test_geometry
    # skip testing element 5, as it is detector center along the ray and doesn't matter
    np.testing.assert_allclose(vectors[0, :5], proj_geom_truth["Vectors"][0, :5])
    np.testing.assert_allclose(vectors[0, 6:], proj_geom_truth["Vectors"][0, 6:])


def test_vol_coord_to_world_coord():
    vol_geom = scico.linop.xray.astra.astra.create_vol_geom(16, 16)
    vc = np.array([[0.0, 0.0], [1.0, 1.0]])
    wc = scico.linop.xray.astra.volume_coords_to_world_coords(vc, vol_geom)
    assert wc.shape == (2, 2)


def test_ensure_writeable():
    assert isinstance(_ensure_writeable(np.ones((2, 1))), np.ndarray)
    assert isinstance(_ensure_writeable(snp.ones((2, 1))), np.ndarray)
