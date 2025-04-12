import os
import tempfile

import numpy as np

import pytest

import scico.numpy as snp
from scico import function, functional, linop, loss, random
from scico.numpy import BlockArray
from scico.optimize import NonLinearPADMM, ProximalADMM


class TestMisc:
    def setup_method(self, method):
        np.random.seed(12345)
        self.y = snp.array(np.random.randn(32, 33).astype(np.float32))
        self.maxiter = 2
        self.ρ = 1e0
        self.μ = 1e0
        self.ν = 1e0
        self.A = linop.Identity(self.y.shape)
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g = functional.DnCNN()
        self.H = function.Function(
            (self.A.input_shape, self.A.input_shape),
            output_shape=self.A.input_shape,
            eval_fn=lambda x, z: x - z,
            input_dtypes=np.float32,
            output_dtype=np.float32,
        )
        self.x0 = snp.zeros(self.A.input_shape, dtype=snp.float32)

    def test_itstat_padmm(self):
        itstat_fields = {"Iter": "%d", "Time": "%8.2e"}

        def itstat_func(obj):
            return (obj.itnum, obj.timer.elapsed())

        padmm_ = ProximalADMM(
            f=self.f,
            g=self.g,
            A=self.A,
            rho=self.ρ,
            mu=self.μ,
            nu=self.ν,
            x0=self.x0,
            z0=self.x0,
            u0=self.x0,
            maxiter=self.maxiter,
        )
        assert len(padmm_.itstat_object.fieldname) == 4
        assert snp.sum(padmm_.x) == 0.0

        padmm_ = ProximalADMM(
            f=self.f,
            g=self.g,
            A=self.A,
            rho=self.ρ,
            mu=self.μ,
            nu=self.ν,
            B=None,
            maxiter=self.maxiter,
            itstat_options={"fields": itstat_fields, "itstat_func": itstat_func, "display": False},
        )
        assert len(padmm_.itstat_object.fieldname) == 2

    def test_itstat_nlpadmm(self):
        itstat_fields = {"Iter": "%d", "Time": "%8.2e"}

        def itstat_func(obj):
            return (obj.itnum, obj.timer.elapsed())

        nlpadmm_ = NonLinearPADMM(
            f=self.f,
            g=self.g,
            H=self.H,
            rho=self.ρ,
            mu=self.μ,
            nu=self.ν,
            x0=self.x0,
            z0=self.x0,
            u0=self.x0,
            maxiter=self.maxiter,
        )
        assert len(nlpadmm_.itstat_object.fieldname) == 4
        assert snp.sum(nlpadmm_.x) == 0.0

        nlpadmm_ = NonLinearPADMM(
            f=self.f,
            g=self.g,
            H=self.H,
            rho=self.ρ,
            mu=self.μ,
            nu=self.ν,
            maxiter=self.maxiter,
            itstat_options={"fields": itstat_fields, "itstat_func": itstat_func, "display": False},
        )
        assert len(nlpadmm_.itstat_object.fieldname) == 2

    def test_callback(self):
        padmm_ = ProximalADMM(
            f=self.f,
            g=self.g,
            A=self.A,
            rho=self.ρ,
            mu=self.μ,
            nu=self.ν,
            maxiter=self.maxiter,
        )
        padmm_.test_flag = False

        def callback(obj):
            obj.test_flag = True

        x = padmm_.solve(callback=callback)
        assert padmm_.test_flag

    def test_finite_check(self):
        padmm_ = ProximalADMM(
            f=self.f,
            g=self.g,
            A=self.A,
            rho=self.ρ,
            mu=self.μ,
            nu=self.ν,
            maxiter=self.maxiter,
            nanstop=True,
        )
        padmm_.step()
        padmm_.x = padmm_.x.at[0].set(np.nan)
        with pytest.raises(ValueError):
            padmm_.solve()


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
        self.ρ = 1e0
        self.μ = 1e0
        self.ν = 1e0
        self.A = linop.Identity(self.y.shape)
        self.f = loss.SquaredL2Loss(y=self.y, A=self.A)
        self.g = (self.λ / 2) * functional.L2Norm()
        self.H = function.Function(
            (self.A.input_shape, self.A.input_shape),
            output_shape=self.A.input_shape,
            eval_fn=lambda x, z: x - z,
            input_dtypes=np.float32,
            output_dtype=np.float32,
        )
        self.x0 = snp.zeros(self.A.input_shape, dtype=snp.float32)

    def test_blockarray_padmm(self):
        padmm_ = ProximalADMM(
            f=self.f,
            g=self.g,
            A=self.A,
            rho=self.ρ,
            mu=self.μ,
            nu=self.ν,
            maxiter=self.maxiter,
        )
        x = padmm_.solve()
        assert isinstance(x, BlockArray)

    def test_blockarray_nlpadmm(self):
        nlpadmm_ = NonLinearPADMM(
            f=self.f,
            g=self.g,
            H=self.H,
            rho=self.ρ,
            mu=self.μ,
            nu=self.ν,
            maxiter=self.maxiter,
        )
        x = nlpadmm_.solve()
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

    def test_padmm(self):
        maxiter = 200
        ρ = 1e0
        μ = 5e1
        ν = 1e0
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        padmm_ = ProximalADMM(
            f=f,
            g=g,
            A=C,
            rho=ρ,
            mu=μ,
            nu=ν,
            maxiter=maxiter,
        )
        x = padmm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4

    def test_padmm_saveload(self):
        maxiter = 5
        ρ = 1e0
        μ = 5e1
        ν = 1e0
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        padmm0 = ProximalADMM(
            f=f,
            g=g,
            A=C,
            rho=ρ,
            mu=μ,
            nu=ν,
            maxiter=maxiter,
        )
        padmm0.solve()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "padmm.npz")
            padmm0.save_state(path)
            padmm0.solve()
            h0 = padmm0.history()
            padmm1 = ProximalADMM(
                f=f,
                g=g,
                A=C,
                rho=ρ,
                mu=μ,
                nu=ν,
                maxiter=maxiter,
            )
            padmm1.load_state(path)
            padmm1.solve()
            h1 = padmm1.history()
            np.testing.assert_allclose(padmm0.minimizer(), padmm1.minimizer(), rtol=1e-6)
            assert np.abs(h0[-1].Objective - h1[-1].Objective) < 1e-6

    def test_nlpadmm(self):
        maxiter = 200
        ρ = 1e0
        μ = 5e1
        ν = 1e0
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        H = function.Function(
            (C.input_shape, C.output_shape),
            output_shape=C.output_shape,
            eval_fn=lambda x, z: C(x) - z,
            input_dtypes=snp.float32,
            output_dtype=snp.float32,
        )
        nlpadmm_ = NonLinearPADMM(
            f=f,
            g=g,
            H=H,
            rho=ρ,
            mu=μ,
            nu=ν,
            maxiter=maxiter,
        )
        x = nlpadmm_.solve()
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

    def test_nlpadmm(self):
        maxiter = 300
        ρ = 1e0
        μ = 3e1
        ν = 1e0
        A = linop.Diagonal(snp.diag(self.Amx))
        f = loss.SquaredL2Loss(y=self.y, A=A)
        g = (self.λ / 2) * functional.SquaredL2Norm()
        C = linop.MatrixOperator(self.Bmx)
        H = function.Function(
            (C.input_shape, C.output_shape),
            output_shape=C.output_shape,
            eval_fn=lambda x, z: C(x) - z,
            input_dtypes=snp.complex64,
            output_dtype=snp.complex64,
        )
        nlpadmm_ = NonLinearPADMM(
            f=f,
            g=g,
            H=H,
            rho=ρ,
            mu=μ,
            nu=ν,
            maxiter=maxiter,
        )
        x = nlpadmm_.solve()
        assert (snp.linalg.norm(self.grdA(x) - self.grdb) / snp.linalg.norm(self.grdb)) < 1e-4


class TestEstimateParameters:
    def setup_method(self):
        shape = (32, 33)
        self.A = linop.Identity(shape)
        self.Hr = function.Function(
            (shape, shape),
            output_shape=shape,
            eval_fn=lambda x, z: x - z,
            input_dtypes=np.float32,
            output_dtype=np.float32,
        )
        self.Hc = function.Function(
            (shape, shape),
            output_shape=shape,
            eval_fn=lambda x, z: x - z,
            input_dtypes=np.complex64,
            output_dtype=np.complex64,
        )

    def test_padmm(self):
        mu, nu = ProximalADMM.estimate_parameters(self.A, factor=1.0)
        assert snp.abs(mu - 1.0) < 1e-6
        assert snp.abs(nu - 1.0) < 1e-6

    def test_real(self):
        mu, nu = NonLinearPADMM.estimate_parameters(self.Hr, factor=1.0)
        assert snp.abs(mu - 1.0) < 1e-6
        assert snp.abs(nu - 1.0) < 1e-6

    def test_complex(self):
        mu, nu = NonLinearPADMM.estimate_parameters(self.Hc, factor=1.0)
        assert snp.abs(mu - 1.0) < 1e-6
        assert snp.abs(nu - 1.0) < 1e-6
