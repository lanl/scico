import os
import tempfile

import numpy as np

import pytest

import scico.numpy as snp
from scico import functional, linop, loss, random
from scico.numpy import BlockArray
from scico.optimize import LinearizedADMM


class TestMisc:
    def setup_method(self, method):
        np.random.seed(12345)
        self.y = snp.array(np.random.randn(32, 33).astype(np.float32))
        self.maxiter = 2
        self.μ = 1e-1
        self.ν = 1e-1
        self.A = linop.Identity(self.y.shape)
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g = functional.DnCNN()
        self.C = linop.Identity(self.y.shape)

    def test_itstat(self):
        itstat_fields = {"Iter": "%d", "Time": "%8.2e"}

        def itstat_func(obj):
            return (obj.itnum, obj.timer.elapsed())

        ladmm_ = LinearizedADMM(
            f=self.f,
            g=self.g,
            C=self.C,
            mu=self.μ,
            nu=self.ν,
            maxiter=self.maxiter,
        )
        assert len(ladmm_.itstat_object.fieldname) == 4
        assert snp.sum(ladmm_.x) == 0.0

        ladmm_ = LinearizedADMM(
            f=self.f,
            g=self.g,
            C=self.C,
            mu=self.μ,
            nu=self.ν,
            maxiter=self.maxiter,
            itstat_options={"fields": itstat_fields, "itstat_func": itstat_func, "display": False},
        )
        assert len(ladmm_.itstat_object.fieldname) == 2

    def test_callback(self):
        ladmm_ = LinearizedADMM(
            f=self.f,
            g=self.g,
            C=self.C,
            mu=self.μ,
            nu=self.ν,
            maxiter=self.maxiter,
        )
        ladmm_.test_flag = False

        def callback(obj):
            obj.test_flag = True

        x = ladmm_.solve(callback=callback)
        assert ladmm_.test_flag

    def test_finite_check(self):
        ladmm_ = LinearizedADMM(
            f=self.f, g=self.g, C=self.C, mu=self.μ, nu=self.ν, maxiter=self.maxiter, nanstop=True
        )
        ladmm_.step()
        ladmm_.x = ladmm_.x.at[0].set(np.nan)
        with pytest.raises(ValueError):
            ladmm_.solve()


class TestBlockArray:
    def setup_method(self, method):
        np.random.seed(12345)
        self.y = snp.blockarray(
            (
                np.random.randn(32, 33).astype(np.float32),
                np.random.randn(
                    17,
                ).astype(np.float32),
            )
        )
        self.λ = 1e0
        self.maxiter = 1
        self.μ = 1e-1
        self.ν = 1e-1
        self.A = linop.Identity(self.y.shape)
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g = (self.λ / 2) * functional.L2Norm()
        self.C = linop.Identity(self.y.shape)

    def test_blockarray(self):
        ladmm_ = LinearizedADMM(
            f=self.f,
            g=self.g,
            C=self.C,
            mu=self.μ,
            nu=self.ν,
            maxiter=self.maxiter,
        )
        x = ladmm_.solve()
        assert isinstance(x, BlockArray)


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
        self.y = snp.array(y)
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
            x0=A.adj(self.y),
        )
        x = ladmm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4

    def test_ladmm_saveload(self):
        maxiter = 5
        μ = 1e-2
        ν = 2e-1
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        ladmm0 = LinearizedADMM(
            f=f,
            g=g,
            C=C,
            mu=μ,
            nu=ν,
            maxiter=maxiter,
            x0=A.adj(self.y),
        )
        ladmm0.solve()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ladmm.npz")
            ladmm0.save_state(path)
            ladmm0.solve()
            h0 = ladmm0.history()
            ladmm1 = LinearizedADMM(
                f=f,
                g=g,
                C=C,
                mu=μ,
                nu=ν,
                maxiter=maxiter,
                x0=A.adj(self.y),
            )
            ladmm1.load_state(path)
            ladmm1.solve()
            h1 = ladmm1.history()
            np.testing.assert_allclose(ladmm0.minimizer(), ladmm1.minimizer(), rtol=1e-6)
            assert np.abs(h0[-1].Objective - h1[-1].Objective) < 1e-6


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
        self.y = snp.array(y)
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
            x0=A.adj(self.y),
        )
        x = ladmm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 5e-4
