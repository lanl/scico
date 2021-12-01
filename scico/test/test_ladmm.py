import numpy as np

import jax

import scico.numpy as snp
from scico import functional, linop, loss, random
from scico.ladmm import LinearizedADMM

flag = False


class TestMisc:
    def setup_method(self, method):
        np.random.seed(12345)
        self.y = jax.device_put(np.random.randn(32, 33).astype(np.float32))
        self.λ = 1e0

    def test_ladmm(self):
        maxiter = 2
        μ = 1e-1
        ν = 1e-1
        A = linop.Identity(self.y.shape)
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.BM3D()
        C = linop.Identity(self.y.shape)

        itstat_dict = {"Iter": "%d", "Time": "%8.2e"}

        def itstat_func(obj):
            return (obj.itnum, obj.timer.elapsed())

        ladmm_ = LinearizedADMM(
            f=f,
            g=g,
            C=C,
            mu=μ,
            nu=ν,
            maxiter=maxiter,
            verbose=False,
        )
        assert len(ladmm_.itstat_object.fieldname) == 4
        assert snp.sum(ladmm_.x) == 0.0
        ladmm_ = LinearizedADMM(
            f=f,
            g=g,
            C=C,
            mu=μ,
            nu=ν,
            maxiter=maxiter,
            verbose=False,
            itstat=(itstat_dict, itstat_func),
        )
        assert len(ladmm_.itstat_object.fieldname) == 2

        def callback(obj):
            global flag
            flag = True

        x = ladmm_.solve(callback=callback)
        assert flag


class TestReal:
    def setup_method(self, method):
        np.random.seed(12345)
        N = 8
        MB = 10
        # Set up arrays for problem argmin (1/2) ||A x - y||_2^2 + (λ/2) ||B x||_2^2
        Amx = np.diag(np.random.randn(N).astype(np.float32))
        Bmx = np.random.randn(MB, N).astype(np.float32)
        y = np.random.randn(N).astype(np.float32)
        λ = 1e0
        self.Amx = Amx
        self.Bmx = Bmx
        self.y = jax.device_put(y)
        self.λ = λ
        # Solution of problem is given by linear system (A^T A + λ B^T B) x = A^T y
        self.grdA = lambda x: (Amx.T @ Amx + λ * Bmx.T @ Bmx) @ x
        self.grdb = Amx.T @ y

    def test_ladmm(self):
        maxiter = 400
        μ = 1e-2
        ν = 2e-1
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        ladmm_ = LinearizedADMM(
            f=f,
            g=g,
            C=C,
            mu=μ,
            nu=ν,
            maxiter=maxiter,
            verbose=False,
            x0=A.adj(self.y),
        )
        x = ladmm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4


class TestComplex:
    def setup_method(self, method):
        N = 8
        MB = 10
        # Set up arrays for problem argmin (1/2) ||A x - y||_2^2 + (λ/2) ||B x||_2^2
        Amx, key = random.randn((N,), dtype=np.complex64, key=None)
        Amx = snp.diag(Amx)
        Bmx, key = random.randn((MB, N), dtype=np.complex64, key=key)
        y, key = random.randn((N,), dtype=np.complex64, key=key)
        λ = 1e0
        self.Amx = Amx
        self.Bmx = Bmx
        self.y = jax.device_put(y)
        self.λ = λ
        # Solution of problem is given by linear system (A^T A + λ B^T B) x = A^T y
        self.grdA = lambda x: (Amx.conj().T @ Amx + λ * Bmx.conj().T @ Bmx) @ x
        self.grdb = Amx.conj().T @ y

    def test_ladmm(self):
        maxiter = 500
        μ = 1e-2
        ν = 2e-1
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        ladmm_ = LinearizedADMM(
            f=f,
            g=g,
            C=C,
            mu=μ,
            nu=ν,
            maxiter=maxiter,
            verbose=False,
            x0=A.adj(self.y),
        )
        x = ladmm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 5e-4
