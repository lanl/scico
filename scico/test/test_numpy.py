import numpy as np

import jax
from jax.interpreters.xla import DeviceArray

import pytest

import scico.numpy as snp
import scico.numpy._create as snc
import scico.numpy.linalg as sla
from scico.blockarray import BlockArray
from scico.linop import MatrixOperator


def on_cpu():
    return jax.devices()[0].device_kind == "cpu"


def check_results(jout, sout):
    if isinstance(jout, (tuple, list)) and isinstance(sout, (tuple, list)):
        # multiple outputs from the function
        for x, y in zip(jout, sout):
            np.testing.assert_allclose(x, y, rtol=1e-4)
    elif isinstance(jout, DeviceArray) and isinstance(sout, DeviceArray):
        # single array output from the function
        np.testing.assert_allclose(sout, jout, rtol=1e-4)
    elif jout.shape == () and sout.shape == ():
        # single scalar output from the function
        np.testing.assert_allclose(jout, sout, rtol=1e-4)
    else:
        # some type of output that isn't being captured?
        raise Exception


def test_reshape_array():
    a = np.random.randn(4, 4)
    np.testing.assert_allclose(snp.reshape(a.ravel(), (4, 4)), a)


def test_reshape_array():
    a = np.random.randn(13)
    b = snp.reshape(a, ((3, 3), (4,)))

    c = BlockArray.array_from_flattened(a, ((3, 3), (4,)))

    assert isinstance(b, BlockArray)
    assert b.shape == c.shape
    np.testing.assert_allclose(b.ravel(), c.ravel())


@pytest.mark.parametrize("compute_uv", [True, False])
@pytest.mark.parametrize("full_matrices", [True, False])
@pytest.mark.parametrize("shape", [(8, 8), (4, 8), (8, 4)])
def test_svd(compute_uv, full_matrices, shape):
    A = jax.device_put(np.random.randn(*shape))
    Ao = MatrixOperator(A)
    f = lambda x: sla.svd(x, compute_uv=compute_uv, full_matrices=full_matrices)
    check_results(f(A), f(Ao))


def test_cond():
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = sla.cond
    check_results(f(A), f(Ao))


def test_det():
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = sla.det
    check_results(f(A), f(Ao))


@pytest.mark.skipif(
    on_cpu() == False, reason="nonsymmetric eigendecompositions only supported on cpu"
)
def test_eig():
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = sla.eig
    check_results(f(A), f(Ao))


@pytest.mark.parametrize("symmetrize", [True, False])
@pytest.mark.parametrize("UPLO", [None, "L", "U"])
def test_eigh(UPLO, symmetrize):
    A = jax.device_put(np.random.randn(8, 8))
    A = A.T @ A
    Ao = MatrixOperator(A)
    f = lambda x: sla.eigh(x, UPLO=UPLO, symmetrize_input=symmetrize)
    check_results(f(A), f(Ao))


@pytest.mark.skipif(
    on_cpu() == False, reason="nonsymmetric eigendecompositions only supported on cpu"
)
def test_eigvals():
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = sla.eigvals
    check_results(f(A), f(Ao))


@pytest.mark.parametrize("UPLO", [None, "L", "U"])
def test_eigvalsh(UPLO):
    A = jax.device_put(np.random.randn(8, 8))
    A = A.T @ A
    Ao = MatrixOperator(A)
    f = lambda x: sla.eigvalsh(x, UPLO=UPLO)
    check_results(f(A), f(Ao))


def test_inv():
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = sla.inv
    check_results(f(A), f(Ao))


def test_lstsq():
    A = jax.device_put(np.random.randn(8, 8))
    b = jax.device_put(np.random.randn(8))
    Ao = MatrixOperator(A)
    f = lambda A: sla.lstsq(A, b)
    check_results(f(A), f(Ao))


def test_matrix_power():
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = lambda A: sla.matrix_power(A, 3)
    check_results(f(A), f(Ao))


def test_matrixrank():
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = lambda A: sla.matrix_rank(A, 3)
    check_results(f(A), f(Ao))


@pytest.mark.parametrize("rcond", [None, 1e-3])
def test_pinv(rcond):
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = sla.pinv
    check_results(f(A), f(Ao))


@pytest.mark.parametrize("rcond", [None, 1e-3])
def test_pinv(rcond):
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = sla.pinv
    check_results(f(A), f(Ao))


@pytest.mark.parametrize("shape", [(8, 8), (4, 8), (8, 4)])
@pytest.mark.parametrize("mode", ["reduced", "complete", "r"])
def test_qr(shape, mode):
    A = jax.device_put(np.random.randn(*shape))
    Ao = MatrixOperator(A)
    f = lambda A: sla.qr(A, mode)
    check_results(f(A), f(Ao))


def test_slogdet():
    A = jax.device_put(np.random.randn(8, 8))
    Ao = MatrixOperator(A)
    f = sla.slogdet
    check_results(f(A), f(Ao))


def test_solve():
    A = jax.device_put(np.random.randn(8, 8))
    b = jax.device_put(np.random.randn(8))
    Ao = MatrixOperator(A)
    f = lambda A: sla.solve(A, b)
    check_results(f(A), f(Ao))


def test_multi_dot():
    A = jax.device_put(np.random.randn(8, 8))
    B = jax.device_put(np.random.randn(8, 4))
    Ao = MatrixOperator(A)
    Bo = MatrixOperator(B)
    f = sla.multi_dot
    check_results(f([A, B]), f([Ao, Bo]))


def test_ufunc_abs():
    A = snp.array([-1, 2, 5])
    res = snp.array([1, 2, 5])
    np.testing.assert_allclose(snp.abs(A), res)

    A = snp.array([-1, -1, -1])
    res = snp.array([1, 1, 1])
    np.testing.assert_allclose(snp.abs(A), res)

    Ba = BlockArray.array((snp.array([-1, 2, 5]),))
    res = BlockArray.array((snp.array([1, 2, 5]),))
    np.testing.assert_allclose(snp.abs(Ba).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([-1, -1, -1]),))
    res = BlockArray.array((snp.array([1, 1, 1]),))
    np.testing.assert_allclose(snp.abs(Ba).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([-1, 2, -3]), snp.array([1, -2, 3])))
    res = BlockArray.array((snp.array([1, 2, 3]), snp.array([1, 2, 3])))
    np.testing.assert_allclose(snp.abs(Ba).ravel(), res.ravel())


def test_ufunc_maximum():
    A = snp.array([1, 2, 5])
    B = snp.array([2, 3, 4])
    res = snp.array([2, 3, 5])
    np.testing.assert_allclose(snp.maximum(A, B), res)
    np.testing.assert_allclose(snp.maximum(B, A), res)

    A = snp.array([1, 1, 1])
    B = snp.array([2, 2, 2])
    res = snp.array([2, 2, 2])
    np.testing.assert_allclose(snp.maximum(A, B), res)
    np.testing.assert_allclose(snp.maximum(B, A), res)

    A = 4
    B = snp.array([3, 5, 2])
    res = snp.array([4, 5, 4])
    np.testing.assert_allclose(snp.maximum(A, B), res)
    np.testing.assert_allclose(snp.maximum(B, A), res)

    A = 5
    B = 6
    res = 6
    np.testing.assert_allclose(snp.maximum(A, B), res)
    np.testing.assert_allclose(snp.maximum(B, A), res)

    A = snp.array([1, 2, 3])
    B = snp.array([2, 3, 4])
    C = snp.array([5, 6])
    D = snp.array([2, 7])
    Ba = BlockArray.array((A, C))
    Bb = BlockArray.array((B, D))
    res = BlockArray.array((snp.array([2, 3, 4]), snp.array([5, 7])))
    Bmax = snp.maximum(Ba, Bb)
    assert Bmax.shape == res.shape
    np.testing.assert_allclose(Bmax.ravel(), res.ravel())

    A = snp.array([1, 6, 3])
    B = snp.array([6, 3, 8])
    C = 5
    Ba = BlockArray.array((A, B))
    res = BlockArray.array((snp.array([5, 6, 5]), snp.array([6, 5, 8])))
    Bmax = snp.maximum(Ba, C)
    assert Bmax.shape == res.shape
    np.testing.assert_allclose(Bmax.ravel(), res.ravel())


def test_ufunc_sign():
    A = snp.array([10, -5, 0])
    res = snp.array([1, -1, 0])
    np.testing.assert_allclose(snp.sign(A), res)

    Ba = BlockArray.array((snp.array([10, -5, 0]),))
    res = BlockArray.array((snp.array([1, -1, 0]),))
    np.testing.assert_allclose(snp.sign(Ba).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([10, -5, 0]), snp.array([0, 5, -6])))
    res = BlockArray.array((snp.array([1, -1, 0]), snp.array([0, 1, -1])))
    np.testing.assert_allclose(snp.sign(Ba).ravel(), res.ravel())


def test_ufunc_where():
    A = snp.array([1, 2, 4, 5])
    B = snp.array([-1, -1, -1, -1])
    cond = snp.array([False, False, True, True])
    res = snp.array([-1, -1, 4, 5])
    np.testing.assert_allclose(snp.where(cond, A, B), res)

    Ba = BlockArray.array((snp.array([1, 2, 4, 5]),))
    Bb = BlockArray.array((snp.array([-1, -1, -1, -1]),))
    Bcond = BlockArray.array((snp.array([False, False, True, True]),))
    Bres = BlockArray.array((snp.array([-1, -1, 4, 5]),))
    assert snp.where(Bcond, Ba, Bb).shape == Bres.shape
    np.testing.assert_allclose(snp.where(Bcond, Ba, Bb).ravel(), Bres.ravel())

    Ba = BlockArray.array((snp.array([1, 2, 4, 5]), snp.array([1, 2, 4, 5])))
    Bb = BlockArray.array((snp.array([-1, -1, -1, -1]), snp.array([-1, -1, -1, -1])))
    Bcond = BlockArray.array(
        (snp.array([False, False, True, True]), snp.array([True, True, False, False]))
    )
    Bres = BlockArray.array((snp.array([-1, -1, 4, 5]), snp.array([1, 2, -1, -1])))
    assert snp.where(Bcond, Ba, Bb).shape == Bres.shape
    np.testing.assert_allclose(snp.where(Bcond, Ba, Bb).ravel(), Bres.ravel())


def test_ufunc_true_divide():
    A = snp.array([1, 2, 3])
    B = snp.array([3, 3, 3])
    res = snp.array([0.33333333, 0.66666667, 1.0])
    np.testing.assert_allclose(snp.true_divide(A, B), res)

    A = snp.array([1, 2, 3])
    B = 3
    res = snp.array([0.33333333, 0.66666667, 1.0])
    np.testing.assert_allclose(snp.true_divide(A, B), res)

    Ba = BlockArray.array((snp.array([1, 2, 3]),))
    Bb = BlockArray.array((snp.array([3, 3, 3]),))
    res = BlockArray.array((snp.array([0.33333333, 0.66666667, 1.0]),))
    np.testing.assert_allclose(snp.true_divide(Ba, Bb).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([1, 2, 3]), snp.array([1, 2])))
    Bb = BlockArray.array((snp.array([3, 3, 3]), snp.array([2, 2])))
    res = BlockArray.array((snp.array([0.33333333, 0.66666667, 1.0]), snp.array([0.5, 1.0])))
    np.testing.assert_allclose(snp.true_divide(Ba, Bb).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([1, 2, 3]), snp.array([1, 2])))
    A = 2
    res = BlockArray.array((snp.array([0.5, 1.0, 1.5]), snp.array([0.5, 1.0])))
    np.testing.assert_allclose(snp.true_divide(Ba, A).ravel(), res.ravel())


def test_ufunc_floor_divide():
    A = snp.array([1, 2, 3])
    B = snp.array([3, 3, 3])
    res = snp.array([0, 0, 1.0])
    np.testing.assert_allclose(snp.floor_divide(A, B), res)

    A = snp.array([4, 2, 3])
    B = 3
    res = snp.array([1.0, 0, 1.0])
    np.testing.assert_allclose(snp.floor_divide(A, B), res)

    Ba = BlockArray.array((snp.array([1, 2, 3]),))
    Bb = BlockArray.array((snp.array([3, 3, 3]),))
    res = BlockArray.array((snp.array([0, 0, 1.0]),))
    np.testing.assert_allclose(snp.floor_divide(Ba, Bb).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([1, 7, 3]), snp.array([1, 2])))
    Bb = BlockArray.array((snp.array([3, 3, 3]), snp.array([2, 2])))
    res = BlockArray.array((snp.array([0, 2, 1.0]), snp.array([0, 1.0])))
    np.testing.assert_allclose(snp.floor_divide(Ba, Bb).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([1, 2, 3]), snp.array([1, 2])))
    A = 2
    res = BlockArray.array((snp.array([0, 1.0, 1.0]), snp.array([0, 1.0])))
    np.testing.assert_allclose(snp.floor_divide(Ba, A).ravel(), res.ravel())


def test_ufunc_real():
    A = snp.array([1 + 3j])
    res = snp.array([1])
    np.testing.assert_allclose(snp.real(A), res)

    A = snp.array([1 + 3j, 4.0 + 2j])
    res = snp.array([1, 4.0])
    np.testing.assert_allclose(snp.real(A), res)

    Ba = BlockArray.array((snp.array([1 + 3j]),))
    res = BlockArray.array((snp.array([1]),))
    np.testing.assert_allclose(snp.real(Ba).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([1 + 3j]), snp.array([1 + 3j, 4.0])))
    res = BlockArray.array((snp.array([1]), snp.array([1, 4.0])))
    np.testing.assert_allclose(snp.real(Ba).ravel(), res.ravel())


def test_ufunc_imag():
    A = snp.array([1 + 3j])
    res = snp.array([3])
    np.testing.assert_allclose(snp.imag(A), res)

    A = snp.array([1 + 3j, 4.0 + 2j])
    res = snp.array([3, 2])
    np.testing.assert_allclose(snp.imag(A), res)

    Ba = BlockArray.array((snp.array([1 + 3j]),))
    res = BlockArray.array((snp.array([3]),))
    np.testing.assert_allclose(snp.imag(Ba).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([1 + 3j]), snp.array([1 + 3j, 4.0])))
    res = BlockArray.array((snp.array([3]), snp.array([3, 0])))
    np.testing.assert_allclose(snp.imag(Ba).ravel(), res.ravel())


def test_ufunc_conj():
    A = snp.array([1 + 3j])
    res = snp.array([1 - 3j])
    np.testing.assert_allclose(snp.conj(A), res)

    A = snp.array([1 + 3j, 4.0 + 2j])
    res = snp.array([1 - 3j, 4.0 - 2j])
    np.testing.assert_allclose(snp.conj(A), res)

    Ba = BlockArray.array((snp.array([1 + 3j]),))
    res = BlockArray.array((snp.array([1 - 3j]),))
    np.testing.assert_allclose(snp.conj(Ba).ravel(), res.ravel())

    Ba = BlockArray.array((snp.array([1 + 3j]), snp.array([1 + 3j, 4.0])))
    res = BlockArray.array((snp.array([1 - 3j]), snp.array([1 - 3j, 4.0 - 0j])))
    np.testing.assert_allclose(snp.conj(Ba).ravel(), res.ravel())


def test_create_zeros():
    A = snc.zeros(2)
    assert np.all(A == 0)

    A = snc.zeros([(2,), (2,)])
    assert np.all(A.ravel() == 0)


def test_create_ones():
    A = snc.ones(2, dtype=np.float32)
    assert np.all(A == 1)

    A = snc.ones([(2,), (2,)])
    assert np.all(A.ravel() == 1)


def test_create_zeros():
    A = snc.empty(2)
    assert np.all(A == 0)

    A = snc.empty([(2,), (2,)])
    assert np.all(A.ravel() == 0)


def test_create_full():
    A = snc.full((2,), 1)
    assert np.all(A == 1)

    A = snc.full((2,), 1, dtype=np.float32)
    assert np.all(A == 1)

    A = snc.full([(2,), (2,)], 1)
    assert np.all(A.ravel() == 1)


def test_create_zeros_like():
    A = snc.ones(2, dtype=np.float32)
    B = snc.zeros_like(A)
    assert np.all(B == 0) and A.shape == B.shape and A.dtype == B.dtype

    A = snc.ones(2, dtype=np.float32)
    B = snc.zeros_like(A)
    assert np.all(B == 0) and A.shape == B.shape and A.dtype == B.dtype

    A = snc.ones([(2,), (2,)], dtype=np.float32)
    B = snc.zeros_like(A)
    assert np.all(B.ravel() == 0) and A.shape == B.shape and A.dtype == B.dtype


def test_create_empty_like():
    A = snc.ones(2, dtype=np.float32)
    B = snc.empty_like(A)
    assert np.all(B == 0) and A.shape == B.shape and A.dtype == B.dtype

    A = snc.ones(2, dtype=np.float32)
    B = snc.empty_like(A)
    assert np.all(B == 0) and A.shape == B.shape and A.dtype == B.dtype

    A = snc.ones([(2,), (2,)], dtype=np.float32)
    B = snc.empty_like(A)
    assert np.all(B.ravel() == 0) and A.shape == B.shape and A.dtype == B.dtype


def test_create_ones_like():
    A = snc.zeros(2, dtype=np.float32)
    B = snc.ones_like(A)
    assert np.all(B == 1) and A.shape == B.shape and A.dtype == B.dtype

    A = snc.zeros(2, dtype=np.float32)
    B = snc.ones_like(A)
    assert np.all(B == 1) and A.shape == B.shape and A.dtype == B.dtype

    A = snc.zeros([(2,), (2,)], dtype=np.float32)
    B = snc.ones_like(A)
    assert np.all(B.ravel() == 1) and A.shape == B.shape and A.dtype == B.dtype


def test_create_full_like():
    A = snc.zeros(2, dtype=np.float32)
    B = snc.full_like(A, 1.0)
    assert np.all(B == 1) and (A.shape == B.shape) and (A.dtype == B.dtype)

    A = snc.zeros(2, dtype=np.float32)
    B = snc.full_like(A, 1)
    assert np.all(B == 1) and (A.shape == B.shape) and (A.dtype == B.dtype)

    A = snc.zeros([(2,), (2,)], dtype=np.float32)
    B = snc.full_like(A, 1)
    assert np.all(B.ravel() == 1) and (A.shape == B.shape) and (A.dtype == B.dtype)
