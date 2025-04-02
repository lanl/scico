import os
import tempfile

import numpy as np

import pytest

import scico.numpy as snp
from scico import functional, linop, loss, operator, random
from scico.numpy import BlockArray
from scico.optimize import PDHG


class TestMisc:
    def setup_method(self, method):
        np.random.seed(12345)
        self.y = snp.array(np.random.randn(32, 33).astype(np.float32))
        self.maxiter = 2
        self.τ = 1e-1
        self.σ = 1e-1
        self.A = linop.Identity(self.y.shape)
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g = functional.DnCNN()
        self.C = linop.Identity(self.y.shape)

    def test_itstat(self):
        itstat_fields = {"Iter": "%d", "Time": "%8.2e"}

        def itstat_func(obj):
            return (obj.itnum, obj.timer.elapsed())

        pdhg_ = PDHG(
            f=self.f,
            g=self.g,
            C=self.C,
            tau=self.τ,
            sigma=self.σ,
            maxiter=self.maxiter,
        )
        assert len(pdhg_.itstat_object.fieldname) == 4
        assert snp.sum(pdhg_.x) == 0.0

        pdhg_ = PDHG(
            f=self.f,
            g=self.g,
            C=self.C,
            tau=self.τ,
            sigma=self.σ,
            maxiter=self.maxiter,
            itstat_options={"fields": itstat_fields, "itstat_func": itstat_func, "display": False},
        )
        assert len(pdhg_.itstat_object.fieldname) == 2

    def test_callback(self):
        pdhg_ = PDHG(
            f=self.f,
            g=self.g,
            C=self.C,
            tau=self.τ,
            sigma=self.σ,
            maxiter=self.maxiter,
        )
        pdhg_.test_flag = False

        def callback(obj):
            obj.test_flag = True

        x = pdhg_.solve(callback=callback)
        assert pdhg_.test_flag

    def test_finite_check(self):
        pdhg_ = PDHG(
            f=self.f,
            g=self.g,
            C=self.C,
            tau=self.τ,
            sigma=self.σ,
            maxiter=self.maxiter,
            nanstop=True,
        )
        pdhg_.step()
        pdhg_.x = pdhg_.x.at[0].set(np.nan)
        with pytest.raises(ValueError):
            pdhg_.solve()


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
        self.τ = 1e-1
        self.σ = 1e-1
        self.A = linop.Identity(self.y.shape)
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g = (self.λ / 2) * functional.L2Norm()
        self.C = linop.Identity(self.y.shape)

    def test_blockarray(self):
        pdhg_ = PDHG(
            f=self.f,
            g=self.g,
            C=self.C,
            tau=self.τ,
            sigma=self.σ,
            maxiter=self.maxiter,
        )
        x = pdhg_.solve()
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

    def test_pdhg(self):
        maxiter = 300
        τ = 2e-1
        σ = 2e-1
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        pdhg_ = PDHG(
            f=f,
            g=g,
            C=C,
            tau=τ,
            sigma=σ,
            maxiter=maxiter,
            x0=A.adj(self.y),
        )
        x = pdhg_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4

    def test_pdhg_saveload(self):
        maxiter = 5
        τ = 2e-1
        σ = 2e-1
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        pdhg0 = PDHG(
            f=f,
            g=g,
            C=C,
            tau=τ,
            sigma=σ,
            maxiter=maxiter,
            x0=A.adj(self.y),
        )
        pdhg0.solve()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pdhg.npz")
            pdhg0.save_state(path)
            pdhg0.solve()
            h0 = pdhg0.history()
            pdhg1 = PDHG(
                f=f,
                g=g,
                C=C,
                tau=τ,
                sigma=σ,
                maxiter=maxiter,
                x0=A.adj(self.y),
            )
            pdhg1.load_state(path)
            pdhg1.solve()
            h1 = pdhg1.history()
            np.testing.assert_allclose(pdhg0.minimizer(), pdhg1.minimizer(), atol=1e-7)
            assert np.abs(h0[-1].Objective - h1[-1].Objective) < 1e-6

    def test_nlpdhg(self):
        maxiter = 300
        τ = 2e-1
        σ = 2e-1
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        cfn = lambda x: self.Bmx @ x
        Cop = operator.operator_from_function(cfn, "Cop")
        C = Cop(input_shape=self.Bmx.shape[1:])
        pdhg_ = PDHG(
            f=f,
            g=g,
            C=C,
            tau=τ,
            sigma=σ,
            maxiter=maxiter,
            x0=A.adj(self.y),
        )
        x = pdhg_.solve()
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
        self.y = snp.array(y)
        self.λ = λ
        # Solution of problem is given by linear system (A^T A + λ B^T B) x = A^T y
        self.grdA = lambda x: (Amx.conj().T @ Amx + λ * Bmx.conj().T @ Bmx) @ x
        self.grdb = Amx.conj().T @ y

    def test_pdhg(self):
        maxiter = 300
        τ = 2e-1
        σ = 2e-1
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        pdhg_ = PDHG(
            f=f,
            g=g,
            C=C,
            tau=τ,
            sigma=σ,
            maxiter=maxiter,
            x0=A.adj(self.y),
        )
        x = pdhg_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 5e-4


class TestEstimateParameters:
    def setup_method(self):
        shape = (32, 33)
        A = linop.Identity(shape, input_dtype=np.float32)
        B = linop.Identity(shape, input_dtype=np.complex64)
        opcls = operator.operator_from_function(lambda x: snp.abs(x), "op")
        C = opcls(input_shape=shape, input_dtype=np.float32)
        D = opcls(input_shape=shape, input_dtype=np.complex64)
        self.operators = [A, B, C, D]

    def test_operators_dlft(self):
        for op in self.operators[0:2]:
            tau, sigma = PDHG.estimate_parameters(op, factor=1.0)
            assert snp.abs(tau - sigma) < 1e-6
            assert snp.abs(tau - 1.0) < 1e-6

    def test_operators(self):
        for op in self.operators:
            x = snp.ones(op.input_shape, op.input_dtype)
            tau, sigma = PDHG.estimate_parameters(op, x=x, factor=None)
            assert snp.abs(tau - sigma) < 1e-6
            assert snp.abs(tau - 1.0) < 1e-6

    def test_ratio(self):
        op = self.operators[0]
        tau, sigma = PDHG.estimate_parameters(op, factor=1.0, ratio=10.0)
        assert snp.abs(tau * sigma - 1.0) < 1e-6
        assert snp.abs(sigma - 10.0 * tau) < 1e-6
