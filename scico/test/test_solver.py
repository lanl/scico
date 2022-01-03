import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico import random, solver
from scico.blockarray import BlockArray


class TestSet:
    def setup_method(self, method):
        np.random.seed(12345)

    def test_wrap_func_and_grad(self):
        N = 8
        A = jax.device_put(np.random.randn(N, N))
        x = jax.device_put(np.random.randn(N))

        f = lambda x: 0.5 * snp.linalg.norm(A @ x) ** 2

        func_and_grad = solver._wrap_func_and_grad(f, shape=(N,), dtype=x.dtype)
        fx, grad = func_and_grad(x)

        np.testing.assert_allclose(fx, f(x), rtol=5e-5)
        np.testing.assert_allclose(grad, A.T @ A @ x, rtol=5e-5)

    def test_cg_std(self):
        N = 64
        Ac = np.random.randn(N, N)
        Am = Ac.dot(Ac.T)
        A = Am.dot
        x = np.random.randn(N)
        b = Am.dot(x)
        x0 = np.zeros((N,))
        tol = 1e-12
        try:
            xcg, info = solver.cg(A, b, x0, tol=tol)
        except Exception as e:
            print(e)
            assert 0
        assert np.linalg.norm(A(xcg) - b) / np.linalg.norm(b) < 1e-6

    def test_cg_info(self):
        N = 64
        Ac = np.random.randn(N, N)
        Am = Ac.dot(Ac.T)
        A = Am.dot
        x = np.random.randn(N)
        b = Am.dot(x)
        x0 = np.zeros((N,))
        tol = 1e-12
        try:
            xcg, info = solver.cg(A, b, x0, tol=tol, info=True)
        except Exception as e:
            print(e)
            assert 0
        assert info["rel_res"] <= tol
        assert np.linalg.norm(A(xcg) - b) / np.linalg.norm(b) < 1e-6

    def test_cg_complex(self):
        N = 64
        Ac = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        Am = Ac.dot(Ac.conj().T)
        A = Am.dot
        x = np.random.randn(N) + 1j * np.random.randn(N)
        b = Am.dot(x)
        x0 = np.zeros_like(x)
        tol = 1e-12
        try:
            xcg, info = solver.cg(A, b, x0, tol=tol)
        except Exception as e:
            print(e)
            assert 0
        assert np.linalg.norm(A(xcg) - b) / np.linalg.norm(b) < 1e-6

    def test_preconditioned_cg(self):
        N = 64
        D = np.diag(np.linspace(0.1, 20, N))
        Ac = D @ np.random.randn(
            N, N
        )  # Poorly scaled matrix; good fit for diagonal preconditioning
        Am = Ac.dot(Ac.conj().T)

        A = Am.dot

        Mm = np.diag(1 / np.diag(Am))  # inverse of diagonal of Am
        M = Mm.dot

        x = np.random.randn(N) + 1j * np.random.randn(N)
        b = Am.dot(x)
        x0 = np.zeros_like(x)
        tol = 1e-12
        x_cg, cg_info = solver.cg(A, b, x0, tol=tol, info=True, M=None, maxiter=3)
        x_pcg, pcg_info = solver.cg(A, b, x0, tol=tol, info=True, M=M, maxiter=3)

        # Assert that PCG converges faster in a few iterations
        assert cg_info["rel_res"] > 3 * pcg_info["rel_res"]


class TestOptimizeScalar:
    # Adopted from SciPy minimize_scalar tests
    # https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/scipy/optimize/tests/test_optimize.py#L1364
    def setup_method(self):
        self.solution = 1.5
        self.rtol = 1e-3

    def fun(self, x, a=1.5):
        """Objective function"""
        # Jax version of (x - a)**2 - 0.8; will return a devicearray
        return snp.square(x - a) - 0.8

    def test_minimize_scalar(self):
        # combine all tests above for the minimize_scalar wrapper
        x = solver.minimize_scalar(self.fun).x
        np.testing.assert_allclose(x, self.solution, rtol=self.rtol)

        x = solver.minimize_scalar(self.fun, method="Brent")
        np.testing.assert_(x.success)

        x = solver.minimize_scalar(self.fun, method="Brent", options=dict(maxiter=3))
        np.testing.assert_(not x.success)

        x = solver.minimize_scalar(self.fun, bracket=(-3, -2), args=(1.5,), method="Brent").x
        np.testing.assert_allclose(x, self.solution, rtol=self.rtol)

        x = solver.minimize_scalar(self.fun, method="Brent", args=(1.5,)).x
        np.testing.assert_allclose(x, self.solution, rtol=self.rtol)

        x = solver.minimize_scalar(self.fun, bracket=(-15, -1, 15), args=(1.5,), method="Brent").x
        np.testing.assert_allclose(x, self.solution, rtol=self.rtol)

        x = solver.minimize_scalar(self.fun, bracket=(-3, -2), args=(1.5,), method="golden").x
        np.testing.assert_allclose(x, self.solution, rtol=self.rtol)

        x = solver.minimize_scalar(self.fun, method="golden", args=(1.5,)).x
        np.testing.assert_allclose(x, self.solution, rtol=self.rtol)

        x = solver.minimize_scalar(self.fun, bracket=(-15, -1, 15), args=(1.5,), method="golden").x
        np.testing.assert_allclose(x, self.solution, rtol=self.rtol)

        x = solver.minimize_scalar(self.fun, bounds=(0, 1), args=(1.5,), method="Bounded").x
        np.testing.assert_allclose(x, 1, rtol=1e-4)

        x = solver.minimize_scalar(self.fun, bounds=(1, 5), args=(1.5,), method="bounded").x
        np.testing.assert_allclose(x, self.solution, rtol=self.rtol)

        x = solver.minimize_scalar(
            self.fun,
            bounds=(np.array([1]), np.array([5])),
            args=(np.array([1.5]),),
            method="bounded",
        ).x
        np.testing.assert_allclose(x, self.solution, rtol=self.rtol)


@pytest.mark.parametrize("dtype", [snp.float32, snp.complex64])
@pytest.mark.parametrize("method", ["CG", "L-BFGS-B"])
def test_minimize(dtype, method):
    from scipy.linalg import block_diag

    B, M, N = (4, 3, 2)

    # Models a 12x8 block-diagonal matrix with 4x3 blocks
    A, key = random.randn((B, M, N), dtype=dtype)
    x, key = random.randn((B, N), dtype=dtype)
    y = snp.sum(A * x[:, None], axis=2)  # contract along the N axis

    # result by directly inverting the dense matrix
    A_mat = block_diag(*A)
    expected = np.linalg.pinv(A_mat) @ y.ravel()

    def f(x):
        return 0.5 * snp.linalg.norm(y - snp.sum(A * x[:, None], axis=2)) ** 2

    out = solver.minimize(f, x0=snp.zeros_like(x), method=method)

    assert out.x.shape == x.shape
    np.testing.assert_allclose(out.x.ravel(), expected, rtol=5e-4)


def test_split_join_array():
    x, key = random.randn((4, 4), dtype=np.complex64)
    x_s = solver._split_real_imag(x)
    assert x_s.shape == (2, 4, 4)
    np.testing.assert_allclose(x_s[0], snp.real(x))
    np.testing.assert_allclose(x_s[1], snp.imag(x))

    x_j = solver._join_real_imag(x_s)
    np.testing.assert_allclose(x_j, x, rtol=1e-4)


def test_split_join_blockarray():
    x, key = random.randn(((4, 4), (3,)), dtype=np.complex64)
    x_s = solver._split_real_imag(x)
    assert x_s.shape == ((2, 4, 4), (2, 3))

    real_block = BlockArray.array((x_s[0][0], x_s[1][0]))
    imag_block = BlockArray.array((x_s[0][1], x_s[1][1]))
    np.testing.assert_allclose(real_block.ravel(), snp.real(x).ravel(), rtol=1e-4)
    np.testing.assert_allclose(imag_block.ravel(), snp.imag(x).ravel(), rtol=1e-4)

    x_j = solver._join_real_imag(x_s)
    assert x_j.shape == x.shape
    np.testing.assert_allclose(x_j.ravel(), x.ravel(), rtol=1e-4)
