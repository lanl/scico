import numpy as np

import jax
from jax.interpreters.xla import DeviceArray

import scico.numpy as snp


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


def test_ufunc_abs():
    A = snp.array([-1, 2, 5])
    res = snp.array([1, 2, 5])
    np.testing.assert_allclose(snp.abs(A), res)

    A = snp.array([-1, -1, -1])
    res = snp.array([1, 1, 1])
    np.testing.assert_allclose(snp.abs(A), res)

    Ba = snp.blockarray((snp.array([-1, 2, 5]),))
    res = snp.blockarray((snp.array([1, 2, 5]),))
    np.testing.assert_allclose(snp.abs(Ba).ravel(), res.ravel())

    Ba = snp.blockarray((snp.array([-1, -1, -1]),))
    res = snp.blockarray((snp.array([1, 1, 1]),))
    np.testing.assert_allclose(snp.abs(Ba).ravel(), res.ravel())

    Ba = snp.blockarray((snp.array([-1, 2, -3]), snp.array([1, -2, 3])))
    res = snp.blockarray((snp.array([1, 2, 3]), snp.array([1, 2, 3])))
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
    Ba = snp.blockarray((A, C))
    Bb = snp.blockarray((B, D))
    res = snp.blockarray((snp.array([2, 3, 4]), snp.array([5, 7])))
    Bmax = snp.maximum(Ba, Bb)
    snp.testing.assert_allclose(Bmax, res)

    A = snp.array([1, 6, 3])
    B = snp.array([6, 3, 8])
    C = 5
    Ba = snp.blockarray((A, B))
    res = snp.blockarray((snp.array([5, 6, 5]), snp.array([6, 5, 8])))
    Bmax = snp.maximum(Ba, C)
    snp.testing.assert_allclose(Bmax, res)


def test_ufunc_sign():
    A = snp.array([10, -5, 0])
    res = snp.array([1, -1, 0])
    np.testing.assert_allclose(snp.sign(A), res)

    Ba = snp.blockarray((snp.array([10, -5, 0]),))
    res = snp.blockarray((snp.array([1, -1, 0]),))
    snp.testing.assert_allclose(snp.sign(Ba), res)

    Ba = snp.blockarray((snp.array([10, -5, 0]), snp.array([0, 5, -6])))
    res = snp.blockarray((snp.array([1, -1, 0]), snp.array([0, 1, -1])))
    snp.testing.assert_allclose(snp.sign(Ba), res)


def test_ufunc_where():
    A = snp.array([1, 2, 4, 5])
    B = snp.array([-1, -1, -1, -1])
    cond = snp.array([False, False, True, True])
    res = snp.array([-1, -1, 4, 5])
    np.testing.assert_allclose(snp.where(cond, A, B), res)

    Ba = snp.blockarray((snp.array([1, 2, 4, 5]),))
    Bb = snp.blockarray((snp.array([-1, -1, -1, -1]),))
    Bcond = snp.blockarray((snp.array([False, False, True, True]),))
    Bres = snp.blockarray((snp.array([-1, -1, 4, 5]),))
    assert snp.where(Bcond, Ba, Bb).shape == Bres.shape
    np.testing.assert_allclose(snp.where(Bcond, Ba, Bb).ravel(), Bres.ravel())

    Ba = snp.blockarray((snp.array([1, 2, 4, 5]), snp.array([1, 2, 4, 5])))
    Bb = snp.blockarray((snp.array([-1, -1, -1, -1]), snp.array([-1, -1, -1, -1])))
    Bcond = snp.blockarray(
        (snp.array([False, False, True, True]), snp.array([True, True, False, False]))
    )
    Bres = snp.blockarray((snp.array([-1, -1, 4, 5]), snp.array([1, 2, -1, -1])))
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

    Ba = snp.blockarray((snp.array([1, 2, 3]),))
    Bb = snp.blockarray((snp.array([3, 3, 3]),))
    res = snp.blockarray((snp.array([0.33333333, 0.66666667, 1.0]),))
    snp.testing.assert_allclose(snp.true_divide(Ba, Bb), res)

    Ba = snp.blockarray((snp.array([1, 2, 3]), snp.array([1, 2])))
    Bb = snp.blockarray((snp.array([3, 3, 3]), snp.array([2, 2])))
    res = snp.blockarray((snp.array([0.33333333, 0.66666667, 1.0]), snp.array([0.5, 1.0])))
    snp.testing.assert_allclose(snp.true_divide(Ba, Bb), res)

    Ba = snp.blockarray((snp.array([1, 2, 3]), snp.array([1, 2])))
    A = 2
    res = snp.blockarray((snp.array([0.5, 1.0, 1.5]), snp.array([0.5, 1.0])))
    snp.testing.assert_allclose(snp.true_divide(Ba, A), res)


def test_ufunc_floor_divide():
    A = snp.array([1, 2, 3])
    B = snp.array([3, 3, 3])
    res = snp.array([0, 0, 1.0])
    np.testing.assert_allclose(snp.floor_divide(A, B), res)

    A = snp.array([4, 2, 3])
    B = 3
    res = snp.array([1.0, 0, 1.0])
    np.testing.assert_allclose(snp.floor_divide(A, B), res)

    Ba = snp.blockarray((snp.array([1, 2, 3]),))
    Bb = snp.blockarray((snp.array([3, 3, 3]),))
    res = snp.blockarray((snp.array([0, 0, 1.0]),))
    snp.testing.assert_allclose(snp.floor_divide(Ba, Bb), res)

    Ba = snp.blockarray((snp.array([1, 7, 3]), snp.array([1, 2])))
    Bb = snp.blockarray((snp.array([3, 3, 3]), snp.array([2, 2])))
    res = snp.blockarray((snp.array([0, 2, 1.0]), snp.array([0, 1.0])))
    snp.testing.assert_allclose(snp.floor_divide(Ba, Bb), res)

    Ba = snp.blockarray((snp.array([1, 2, 3]), snp.array([1, 2])))
    A = 2
    res = snp.blockarray((snp.array([0, 1.0, 1.0]), snp.array([0, 1.0])))
    snp.testing.assert_allclose(snp.floor_divide(Ba, A), res)


def test_ufunc_real():
    A = snp.array([1 + 3j])
    res = snp.array([1])
    np.testing.assert_allclose(snp.real(A), res)

    A = snp.array([1 + 3j, 4.0 + 2j])
    res = snp.array([1, 4.0])
    np.testing.assert_allclose(snp.real(A), res)

    Ba = snp.blockarray((snp.array([1 + 3j]),))
    res = snp.blockarray((snp.array([1]),))
    snp.testing.assert_allclose(snp.real(Ba), res)

    Ba = snp.blockarray((snp.array([1.0 + 3j]), snp.array([1 + 3j, 4.0])))
    res = snp.blockarray((snp.array([1.0]), snp.array([1, 4.0])))
    snp.testing.assert_allclose(snp.real(Ba), res)


def test_ufunc_imag():
    A = snp.array([1 + 3j])
    res = snp.array([3])
    np.testing.assert_allclose(snp.imag(A), res)

    A = snp.array([1 + 3j, 4.0 + 2j])
    res = snp.array([3, 2])
    np.testing.assert_allclose(snp.imag(A), res)

    Ba = snp.blockarray((snp.array([1 + 3j]),))
    res = snp.blockarray((snp.array([3]),))
    snp.testing.assert_allclose(snp.imag(Ba), res)

    Ba = snp.blockarray((snp.array([1 + 3j]), snp.array([1 + 3j, 4.0])))
    res = snp.blockarray((snp.array([3]), snp.array([3, 0])))
    snp.testing.assert_allclose(snp.imag(Ba), res)


def test_ufunc_conj():
    A = snp.array([1 + 3j])
    res = snp.array([1 - 3j])
    np.testing.assert_allclose(snp.conj(A), res)

    A = snp.array([1 + 3j, 4.0 + 2j])
    res = snp.array([1 - 3j, 4.0 - 2j])
    np.testing.assert_allclose(snp.conj(A), res)

    Ba = snp.blockarray((snp.array([1 + 3j]),))
    res = snp.blockarray((snp.array([1 - 3j]),))
    snp.testing.assert_allclose(snp.conj(Ba), res)

    Ba = snp.blockarray((snp.array([1 + 3j]), snp.array([1 + 3j, 4.0])))
    res = snp.blockarray((snp.array([1 - 3j]), snp.array([1 - 3j, 4.0 - 0j])))
    snp.testing.assert_allclose(snp.conj(Ba), res)


def test_create_zeros():
    A = snp.zeros(2)
    assert np.all(A == 0)

    A = snp.zeros(((2,), (2,)))
    assert all(snp.all(A == 0))


def test_create_ones():
    A = snp.ones(2, dtype=np.float32)
    assert np.all(A == 1)

    A = snp.ones(((2,), (2,)))
    assert all(snp.all(A == 1))


def test_create_zeros():
    A = snp.empty(2)
    assert np.all(A == 0)

    A = snp.empty(((2,), (2,)))
    assert all(snp.all(A == 0))


def test_create_full():
    A = snp.full((2,), 1)
    assert np.all(A == 1)

    A = snp.full((2,), 1, dtype=np.float32)
    assert np.all(A == 1)

    A = snp.full(((2,), (2,)), 1)
    assert all(snp.all(A == 1))


def test_create_zeros_like():
    A = snp.ones(2, dtype=np.float32)
    B = snp.zeros_like(A)
    assert np.all(B == 0) and A.shape == B.shape and A.dtype == B.dtype

    A = snp.ones(2, dtype=np.float32)
    B = snp.zeros_like(A)
    assert np.all(B == 0) and A.shape == B.shape and A.dtype == B.dtype

    A = snp.ones(((2,), (2,)), dtype=np.float32)
    B = snp.zeros_like(A)
    assert all(snp.all(B == 0))
    assert A.shape == B.shape
    assert A.dtype == B.dtype


def test_create_empty_like():
    A = snp.ones(2, dtype=np.float32)
    B = snp.empty_like(A)
    assert np.all(B == 0) and A.shape == B.shape and A.dtype == B.dtype

    A = snp.ones(2, dtype=np.float32)
    B = snp.empty_like(A)
    assert np.all(B == 0) and A.shape == B.shape and A.dtype == B.dtype

    A = snp.ones(((2,), (2,)), dtype=np.float32)
    B = snp.empty_like(A)
    assert all(snp.all(B == 0)) and A.shape == B.shape and A.dtype == B.dtype


def test_create_ones_like():
    A = snp.zeros(2, dtype=np.float32)
    B = snp.ones_like(A)
    assert np.all(B == 1) and A.shape == B.shape and A.dtype == B.dtype

    A = snp.zeros(2, dtype=np.float32)
    B = snp.ones_like(A)
    assert np.all(B == 1) and A.shape == B.shape and A.dtype == B.dtype

    A = snp.zeros(((2,), (2,)), dtype=np.float32)
    B = snp.ones_like(A)
    assert all(snp.all(B == 1)) and A.shape == B.shape and A.dtype == B.dtype


def test_create_full_like():
    A = snp.zeros(2, dtype=np.float32)
    B = snp.full_like(A, 1.0)
    assert np.all(B == 1) and (A.shape == B.shape) and (A.dtype == B.dtype)

    A = snp.zeros(2, dtype=np.float32)
    B = snp.full_like(A, 1)
    assert np.all(B == 1) and (A.shape == B.shape) and (A.dtype == B.dtype)

    A = snp.zeros(((2,), (2,)), dtype=np.float32)
    B = snp.full_like(A, 1)
    assert all(snp.all(B == 1)) and (A.shape == B.shape) and (A.dtype == B.dtype)
