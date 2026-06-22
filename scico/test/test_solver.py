import numpy as np

from jax.scipy.linalg import block_diag

import pytest

import scico.numpy as snp
from scico import linop, metric, random, solver


class TestSet:
    def setup_method(self, method):
        np.random.seed(12345)

    def test_wrap_func_and_grad(self):
        N = 8
        A = snp.array(np.random.randn(N, N))
        x = snp.array(np.random.randn(N))

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
        assert info["rel_res"].ndim == 0
        assert np.linalg.norm(A(xcg) - b) / np.linalg.norm(b) < 1e-6

    def test_cg_op(self):
        N = 32
        Ac = np.random.randn(N, N).astype(np.float32)
        Am = Ac.dot(Ac.T)
        A = Am.dot
        x = np.random.randn(N).astype(np.float32)
        b = Am.dot(x)
        tol = 1e-12
        try:
            xcg, info = solver.cg(linop.MatrixOperator(Am), b, tol=tol)
        except Exception as e:
            print(e)
            assert 0
        assert info["rel_res"].ndim == 0
        assert np.linalg.norm(A(xcg) - b) / np.linalg.norm(b) < 1e-6

    def test_cg_no_info(self):
        N = 64
        Ac = np.random.randn(N, N)
        Am = Ac.dot(Ac.T)
        A = Am.dot
        x = np.random.randn(N)
        b = Am.dot(x)
        x0 = np.zeros((N,))
        tol = 1e-12
        try:
            xcg = solver.cg(A, b, x0, tol=tol, info=False)
        except Exception as e:
            print(e)
            assert 0
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

    def test_lstsq_func(self):
        N = 24
        M = 32
        Ac = snp.array(np.random.randn(N, M).astype(np.float32))
        Am = Ac.dot(Ac.T)
        A = Am.dot
        x = snp.array(np.random.randn(N).astype(np.float32))
        b = Am.dot(x)
        x0 = snp.zeros((N,), dtype=np.float32)
        tol = 1e-6
        try:
            xlsq = solver.lstsq(A, b, x0=x0, tol=tol)
        except Exception as e:
            print(e)
            assert 0
        assert np.linalg.norm(A(xlsq) - b) / np.linalg.norm(b) < 5e-6

    def test_lstsq_op(self):
        N = 32
        M = 24
        Ac = snp.array(np.random.randn(N, M).astype(np.float32))
        A = linop.MatrixOperator(Ac)
        x = snp.array(np.random.randn(M).astype(np.float32))
        b = Ac.dot(x)
        tol = 1e-7
        try:
            xlsq = solver.lstsq(A, b, tol=tol)
        except Exception as e:
            print(e)
            assert 0
        assert np.linalg.norm(A(xlsq) - b) / np.linalg.norm(b) < 1e-6


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
def test_minimize_vector(dtype, method):
    B, M, N = (4, 3, 2)

    # model a 12x8 block-diagonal matrix with 3x2 blocks
    A, key = random.randn((B, M, N), dtype=dtype)
    x, key = random.randn((B, N), dtype=dtype, key=key)
    y = snp.sum(A * x[:, None], axis=2)  # contract along the N axis

    # result by directly inverting the dense matrix
    A_mat = block_diag(*A)
    expected = snp.linalg.pinv(A_mat) @ y.ravel()

    def f(x):
        return 0.5 * snp.linalg.norm(y - snp.sum(A * x[:, None], axis=2)) ** 2

    out = solver.minimize(f, x0=snp.zeros_like(x), method=method)

    assert out.x.shape == x.shape
    np.testing.assert_allclose(out.x.ravel(), expected, rtol=5e-4)


@pytest.mark.parametrize("dtype", [snp.float32])
@pytest.mark.parametrize("method", ["CG"])
def test_minimize_blockarray(dtype, method):
    # model a 6x8 block-diagonal matrix with 3x4 blocks
    A, key = random.randn(((3, 4), (3, 4)), dtype=dtype)
    x, key = random.randn(((4,), (4,)), dtype=dtype, key=key)
    y = A @ x

    # result by directly inverting the dense matrix
    A_mat = block_diag(*A)
    expected = snp.linalg.pinv(A_mat) @ y.stack(axis=0).ravel()

    def f(x):
        return 0.5 * snp.linalg.norm(y - A @ x) ** 2

    out = solver.minimize(f, x0=snp.zeros_like(x), method=method)

    assert out.x.shape == x.shape
    np.testing.assert_allclose(solver._ravel(out.x), expected, rtol=5e-4)


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

    real_block = snp.blockarray((x_s[0][0], x_s[1][0]))
    imag_block = snp.blockarray((x_s[0][1], x_s[1][1]))
    snp.testing.assert_allclose(real_block, snp.real(x), rtol=1e-4)
    snp.testing.assert_allclose(imag_block, snp.imag(x), rtol=1e-4)

    x_j = solver._join_real_imag(x_s)
    snp.testing.assert_allclose(x_j, x, rtol=1e-4)


def test_bisect():
    f = lambda x: x**3
    x, info = solver.bisect(f, -snp.ones((5, 1)), snp.ones((5, 1)), full_output=True)
    assert snp.sum(snp.abs(x)) == 0.0
    assert info["iter"] == 0
    x = solver.bisect(f, -2.0 * snp.ones((5, 3)), snp.ones((5, 3)), xtol=1e-5, ftol=1e-5)
    assert snp.max(snp.abs(x)) <= 1e-5
    assert snp.max(snp.abs(f(x))) <= 1e-5
    c, key = random.randn((5, 1), dtype=np.float32)
    f = lambda x, c: x**3 - c**3
    x = solver.bisect(f, -snp.abs(c) - 1, snp.abs(c) + 1, args=(c,), xtol=1e-5, ftol=1e-5)
    assert snp.max(snp.abs(x - c)) <= 1e-5
    assert snp.max(snp.abs(f(x, c))) <= 1e-5


def test_golden():
    f = lambda x: x**2
    x, info = solver.golden(f, -snp.ones((5, 1)), snp.ones((5, 1)), full_output=True)
    assert snp.max(snp.abs(x)) <= 1e-7
    x = solver.golden(f, -2.0 * snp.ones((5, 3)), snp.ones((5, 3)), xtol=1e-5)
    assert snp.max(snp.abs(x)) <= 1e-5
    c, key = random.randn((5, 1), dtype=np.float32)
    f = lambda x, c: (x - c) ** 2
    x = solver.golden(f, -snp.abs(c) - 1, snp.abs(c) + 1, args=(c,), xtol=1e-5)
    assert snp.max(snp.abs(x - c)) <= 1e-5


@pytest.mark.parametrize("cho_factor", [True, False])
@pytest.mark.parametrize("wide", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("alpha", [1e-1, 1e1])
def test_solve_atai(cho_factor, wide, weighted, alpha):
    A, key = random.randn((5, 8), dtype=snp.float32)
    if wide:
        x0, key = random.randn((8,), key=key)
    else:
        A = A.T
        x0, key = random.randn((5,), key=key)

    if weighted:
        W, key = random.randn((A.shape[0],), key=key)
        W = snp.abs(W)
        Wa = W[:, snp.newaxis]
    else:
        W = None
        Wa = snp.array([1.0])[:, snp.newaxis]

    D = alpha * snp.ones((A.shape[1],))
    ATAD = A.T @ (Wa * A) + alpha * snp.identity(A.shape[1])
    b = ATAD @ x0
    slv = solver.MatrixATADSolver(A, D, W=W, cho_factor=cho_factor)
    x1 = slv.solve(b)
    assert metric.rel_res(x0, x1) < 5e-5


@pytest.mark.parametrize("cho_factor", [True, False])
@pytest.mark.parametrize("wide", [True, False])
@pytest.mark.parametrize("alpha", [1e-1, 1e1])
def test_solve_aati(cho_factor, wide, alpha):
    A, key = random.randn((5, 8), dtype=snp.float32)
    if wide:
        x0, key = random.randn((5,), key=key)
    else:
        A = A.T
        x0, key = random.randn((8,), key=key)

    D = alpha * snp.ones((A.shape[0],))
    AATD = A @ A.T + alpha * snp.identity(A.shape[0])
    b = AATD @ x0
    slv = solver.MatrixATADSolver(A.T, D)
    x1 = slv.solve(b)
    assert metric.rel_res(x0, x1) < 5e-5


@pytest.mark.parametrize("cho_factor", [True, False])
@pytest.mark.parametrize("wide", [True, False])
@pytest.mark.parametrize("vector", [True, False])
def test_solve_atad(cho_factor, wide, vector):
    A, key = random.randn((5, 8), dtype=snp.float32)
    if wide:
        D, key = random.randn((8,), key=key)
        if vector:
            x0, key = random.randn((8,), key=key)
        else:
            x0, key = random.randn((8, 3), key=key)
    else:
        A = A.T
        D, key = random.randn((5,), key=key)
        if vector:
            x0, key = random.randn((5,), key=key)
        else:
            x0, key = random.randn((5, 3), key=key)

    D = snp.abs(D)  # only required for Cholesky, but improved accuracy for LU
    ATAD = A.T @ A + snp.diag(D)
    b = ATAD @ x0
    slv = solver.MatrixATADSolver(A, D, cho_factor=cho_factor)
    x1 = slv.solve(b)
    assert metric.rel_res(x0, x1) < 5e-5
    assert slv.accuracy(x1, b) < 5e-5
