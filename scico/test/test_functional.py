import numpy as np

from jax.config import config

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)

import warnings

import jax

import pytest

import scico.numpy as snp
from scico import denoiser, functional, linop, loss
from scico.blockarray import BlockArray
from scico.random import randn
from scico.solver import minimize

NO_BLOCK_ARRAY = [functional.L21Norm, functional.NuclearNorm]
NO_COMPLEX = [
    functional.NonNegativeIndicator,
]


def prox_func(x, v, f, alpha):
    """Evaluate functional of which the proximal operator is the argmin."""
    return 0.5 * snp.sum(snp.abs(x.reshape(v.shape) - v) ** 2) + alpha * snp.array(
        f(x.reshape(v.shape)), dtype=snp.float64
    )


def prox_solve(v, v0, f, alpha):
    """Evaluate the alpha-scaled proximal operator of f at v, using v0 as an
    initial point for the optimization."""
    fnc = lambda x: prox_func(x, v, f, alpha)
    fmn = minimize(
        fnc,
        v0,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-9, "fatol": 1e-9},
    )

    return fmn.x.reshape(v.shape), fmn.fun


def prox_test(v, nrm, prx, alpha, x0=None):
    """Test the alpha-scaled proximal operator function prx of norm functional nrm
    at point v."""
    # Evaluate the proximal operator at v
    px = snp.array(prx(v, alpha, v0=x0))
    # Proximal operator functional value (i.e. Moreau envelope) at v
    pf = prox_func(px, v, nrm, alpha)
    # Brute-force solve of the proximal operator at v
    mx, mf = prox_solve(v, px, nrm, alpha)

    # Compare prox functional value with brute-force solution
    if pf < mf:
        return  # prox gave a lower cost than brute force, so it passes

    np.testing.assert_allclose(pf, mf, rtol=1e-6)


class ProxTestObj:
    def __init__(self, dtype):
        key = None
        self.v, key = randn(shape=(32, 1), dtype=dtype, key=key, seed=3)
        self.vb, key = randn(shape=((8, 8), (4,)), dtype=dtype, key=key)
        self.scalar = np.pi

        self.vz = snp.zeros((8, 8), dtype=dtype)


@pytest.fixture(params=[np.float32, np.complex64, np.float64, np.complex128])
def test_prox_obj(request):
    return ProxTestObj(request.param)


class SeparableTestObject:
    def __init__(self, dtype):
        self.f = functional.L1Norm()
        self.g = functional.SquaredL2Norm()
        self.fg = functional.SeparableFunctional([self.f, self.g])

        n = 4
        m = 6
        key = None

        self.v1, key = randn((n,), key=key, dtype=dtype)  # point for prox eval
        self.v2, key = randn((m,), key=key, dtype=dtype)  # point for prox eval
        self.vb = BlockArray.array([self.v1, self.v2])


@pytest.fixture(params=[np.float32, np.complex64, np.float64, np.complex128])
def test_separable_obj(request):
    return SeparableTestObject(request.param)


def test_separable_eval(test_separable_obj):
    fv1 = test_separable_obj.f(test_separable_obj.v1)
    gv2 = test_separable_obj.g(test_separable_obj.v2)
    fgv = test_separable_obj.fg(test_separable_obj.vb)
    np.testing.assert_allclose(fv1 + gv2, fgv, rtol=5e-2)


def test_separable_prox(test_separable_obj):
    alpha = 0.1
    fv1 = test_separable_obj.f.prox(test_separable_obj.v1, alpha)
    gv2 = test_separable_obj.g.prox(test_separable_obj.v2, alpha)
    fgv = test_separable_obj.fg.prox(test_separable_obj.vb, alpha)
    out = BlockArray.array((fv1, gv2)).ravel()
    np.testing.assert_allclose(out, fgv.ravel(), rtol=5e-2)


def test_separable_grad(test_separable_obj):
    # Used to restore the warnings after the context is used
    with warnings.catch_warnings():
        # Ignores warning raised by ensure_on_device
        warnings.filterwarnings(action="ignore", category=UserWarning)

        # Verifies that there is a warning on f.grad and fg.grad
        np.testing.assert_warns(test_separable_obj.f.grad(test_separable_obj.v1))
        np.testing.assert_warns(test_separable_obj.fg.grad(test_separable_obj.vb))

        # Tests the separable grad with warnings being supressed
        fv1 = test_separable_obj.f.grad(test_separable_obj.v1)
        gv2 = test_separable_obj.g.grad(test_separable_obj.v2)
        fgv = test_separable_obj.fg.grad(test_separable_obj.vb)
        out = BlockArray.array((fv1, gv2)).ravel()
        np.testing.assert_allclose(out, fgv.ravel(), rtol=5e-2)


class TestNormProx:

    alphalist = [1e-2, 1e-1, 1e0, 1e1]
    normlist = [
        functional.L0Norm,
        functional.L1Norm,
        functional.SquaredL2Norm,
        functional.L2Norm,
        functional.L21Norm,
        functional.NuclearNorm,
        functional.ZeroFunctional,
    ]

    normlist_blockarray_ready = set(normlist.copy()) - set(NO_BLOCK_ARRAY)

    @pytest.mark.parametrize("norm", normlist)
    @pytest.mark.parametrize("alpha", alphalist)
    def test_prox(self, norm, alpha, test_prox_obj):
        nrmobj = norm()
        nrm = nrmobj.__call__
        prx = nrmobj.prox
        pf = prox_test(test_prox_obj.v, nrm, prx, alpha)

    @pytest.mark.parametrize("norm", normlist)
    @pytest.mark.parametrize("alpha", alphalist)
    def test_conj_prox(self, norm, alpha, test_prox_obj):
        nrmobj = norm()
        v = test_prox_obj.v
        # Test checks extended Moreau decomposition at a random vector
        lhs = nrmobj.prox(v, alpha) + alpha * nrmobj.conj_prox(v / alpha, 1.0 / alpha)
        rhs = v
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=0.0)

    @pytest.mark.parametrize("norm", normlist_blockarray_ready)
    @pytest.mark.parametrize("alpha", alphalist)
    def test_prox_blockarray(self, norm, alpha, test_prox_obj):
        nrmobj = norm()
        nrm = nrmobj.__call__
        prx = nrmobj.prox
        pf = nrmobj.prox(test_prox_obj.vb.ravel(), alpha)
        pf_b = nrmobj.prox(test_prox_obj.vb, alpha)
        np.testing.assert_allclose(pf, pf_b.ravel())

    @pytest.mark.parametrize("norm", normlist)
    def test_prox_zeros(self, norm, test_prox_obj):
        nrmobj = norm()
        nrm = nrmobj.__call__
        prx = nrmobj.prox
        pf = prox_test(test_prox_obj.vz, nrm, prx, alpha=1.0)

    @pytest.mark.parametrize("norm", normlist)
    def test_scaled_attrs(self, norm, test_prox_obj):
        alpha = np.sqrt(2)
        unscaled = norm()
        scaled = test_prox_obj.scalar * norm()

        assert scaled.has_eval == unscaled.has_eval
        assert scaled.has_prox == unscaled.has_prox
        assert scaled.scale == test_prox_obj.scalar

    @pytest.mark.parametrize("norm", normlist)
    @pytest.mark.parametrize("alpha", alphalist)
    def test_scaled_eval(self, norm, alpha, test_prox_obj):

        unscaled = norm()
        scaled = test_prox_obj.scalar * norm()

        a = test_prox_obj.scalar * unscaled(test_prox_obj.v)
        b = scaled(test_prox_obj.v)
        np.testing.assert_allclose(a, b)

    @pytest.mark.parametrize("norm", normlist)
    @pytest.mark.parametrize("alpha", alphalist)
    def test_scaled_prox(self, norm, alpha, test_prox_obj):
        # Test prox
        unscaled = norm()
        scaled = test_prox_obj.scalar * norm()
        a = unscaled.prox(test_prox_obj.v, alpha * test_prox_obj.scalar)
        b = scaled.prox(test_prox_obj.v, alpha)
        np.testing.assert_allclose(a, b)


class TestBlockArrayEval:
    # Ensures that functionals evaluate properly on a blockarray
    # By convention, should be the same as evaluating on the flattened array

    # Generate a list of all functionals in scico.functionals that we will check
    ignore = [functional.Functional, functional.ScaledFunctional]
    to_check = []
    for name, cls in functional.__dict__.items():
        if isinstance(cls, type):
            if issubclass(cls, functional.Functional):
                if cls not in ignore and cls.has_eval is True:
                    to_check.append(cls)

    to_check = set(to_check) - set(NO_BLOCK_ARRAY)

    @pytest.mark.parametrize("cls", to_check)
    def test_eval(self, cls, test_prox_obj):
        func = cls()  # instantiate the functional we are testing

        if cls in NO_COMPLEX and snp.iscomplexobj(test_prox_obj.vb):
            with pytest.raises(ValueError):
                x = func(test_prox_obj.vb)
            return

        x = func(test_prox_obj.vb)
        y = func(test_prox_obj.vb.ravel())
        np.testing.assert_allclose(x, y)


# only check double precision on projections
@pytest.fixture(params=[np.float64, np.complex128])
def test_proj_obj(request):
    return ProxTestObj(request.param)


class TestProj:

    cnstrlist = [functional.NonNegativeIndicator, functional.L2BallIndicator]

    @pytest.mark.parametrize("cnstr", cnstrlist)
    def test_prox(self, cnstr, test_proj_obj):
        alpha = 1
        cnsobj = cnstr()
        cns = cnsobj.__call__
        prx = cnsobj.prox

        if cnstr in NO_COMPLEX and snp.iscomplexobj(test_proj_obj.v):
            with pytest.raises(ValueError):
                prox_test(test_proj_obj.v, cns, prx, alpha)
            return

        prox_test(test_proj_obj.v, cns, prx, alpha)

    @pytest.mark.parametrize("cnstr", cnstrlist)
    def test_prox_scale_invariance(self, cnstr, test_proj_obj):
        alpha1 = 1e-2
        alpha2 = 1e0
        cnsobj = cnstr()
        u1 = cnsobj.prox(test_proj_obj.v, alpha1)
        u2 = cnsobj.prox(test_proj_obj.v, alpha2)
        assert np.linalg.norm(u1 - u2) / np.linalg.norm(u1) <= 1e-7


class TestCheckAttrs:
    # Ensure that the has_eval, has_prox attrs are overridden
    # and set to True/False in the Functional subclasses.

    # Generate a list of all functionals in scico.functionals that we will check
    ignore = [functional.Functional, functional.ScaledFunctional, functional.SeparableFunctional]
    to_check = []
    for name, cls in functional.__dict__.items():
        if isinstance(cls, type):
            if issubclass(cls, functional.Functional):
                if cls not in ignore:
                    to_check.append(cls)

    @pytest.mark.parametrize("cls", to_check)
    def test_has_eval(self, cls):
        assert isinstance(cls.has_eval, bool)

    @pytest.mark.parametrize("cls", to_check)
    def test_has_prox(self, cls):
        assert isinstance(cls.has_prox, bool)


def test_scalar_vmap():
    x = np.random.randn(4, 4)
    f = functional.L1Norm()

    def foo(c):
        return (c * f)(x)

    c_list = [1.0, 2.0, 3.0]
    non_vmap = np.array([foo(c) for c in c_list])

    vmapped = jax.vmap(foo)(snp.array(c_list))
    np.testing.assert_allclose(non_vmap, vmapped)


def test_scalar_pmap():
    x = np.random.randn(4, 4)
    f = functional.L1Norm()

    def foo(c):
        return (c * f)(x)

    c_list = np.random.randn(jax.device_count())
    non_pmap = np.array([foo(c) for c in c_list])
    pmapped = jax.pmap(foo)(c_list)
    np.testing.assert_allclose(non_pmap, pmapped)


class TestLoss:
    def setup_method(self):
        n = 4
        dtype = np.float64
        A, key = randn((n, n), key=None, dtype=dtype, seed=1234)
        D, key = randn((n,), key=key, dtype=dtype)
        W, key = randn((n,), key=key, dtype=dtype)
        W = 0.1 * W + 1.0
        self.Ao = linop.MatrixOperator(A)
        self.Ao_abs = linop.MatrixOperator(snp.abs(A))
        self.Do = linop.Diagonal(D)
        self.W = linop.Diagonal(W)
        self.y, key = randn((n,), key=key, dtype=dtype)
        self.v, key = randn((n,), key=key, dtype=dtype)  # point for prox eval
        scalar, key = randn((1,), key=key, dtype=dtype)
        self.scalar = scalar.copy().ravel()[0]

    def test_squared_l2(self):
        L = loss.SquaredL2Loss(y=self.y, A=self.Ao)
        assert L.has_eval == True
        assert L.has_prox == True

        # test eval
        np.testing.assert_allclose(L(self.v), 0.5 * ((self.Ao @ self.v - self.y) ** 2).sum())

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale
        assert cL(self.v) == self.scalar * L(self.v)

        # SquaredL2 with Diagonal linop has a prox
        L_d = loss.SquaredL2Loss(y=self.y, A=self.Do)

        # test eval
        np.testing.assert_allclose(L_d(self.v), 0.5 * ((self.Do @ self.v - self.y) ** 2).sum())

        assert L_d.has_eval == True
        assert L_d.has_prox == True

        cL = self.scalar * L_d
        assert L_d.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L_d.scale
        assert cL(self.v) == self.scalar * L_d(self.v)

        pf = prox_test(self.v, L_d, L_d.prox, 0.75)

        pf = prox_test(self.v, L, L.prox, 0.75)

    def test_weighted_squared_l2(self):
        L = loss.WeightedSquaredL2Loss(y=self.y, A=self.Ao, W=self.W)
        assert L.has_eval == True
        assert L.has_prox == True

        # test eval
        np.testing.assert_allclose(
            L(self.v), 0.5 * (self.W @ (self.Ao @ self.v - self.y) ** 2).sum()
        )

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale
        assert cL(self.v) == self.scalar * L(self.v)

        # SquaredL2 with Diagonal linop has a prox
        L_d = loss.WeightedSquaredL2Loss(y=self.y, A=self.Do, W=self.W)

        assert L_d.has_eval == True
        assert L_d.has_prox == True

        # test eval
        np.testing.assert_allclose(
            L_d(self.v), 0.5 * (self.W @ (self.Do @ self.v - self.y) ** 2).sum()
        )

        cL = self.scalar * L_d
        assert L_d.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L_d.scale
        assert cL(self.v) == self.scalar * L_d(self.v)

        pf = prox_test(self.v, L_d, L_d.prox, 0.75)

        pf = prox_test(self.v, L, L.prox, 0.75)

    def test_poisson(self):
        L = loss.PoissonLoss(y=self.y, A=self.Ao_abs)
        assert L.has_eval == True
        assert L.has_prox == False

        # test eval
        v = snp.abs(self.v)
        Av = self.Ao_abs @ v
        np.testing.assert_allclose(L(v), 0.5 * snp.sum(Av - self.y * snp.log(Av) + L.const))

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale
        assert cL(v) == self.scalar * L(v)


class TestBM3D:
    def setup(self):
        key = None
        self.x_gry, key = randn((32, 33), key=key, dtype=np.float32)
        self.x_rgb, key = randn((33, 34, 3), key=key, dtype=np.float32)
        self.f_gry = functional.BM3D()
        self.f_rgb = functional.BM3D(is_rgb=True)

    def test_gry(self):
        y0 = self.f_gry.prox(self.x_gry, 1.0)
        y1 = denoiser.bm3d(self.x_gry, 1.0)
        np.testing.assert_allclose(y0, y1, rtol=1e-5)

    def test_rgb(self):
        y0 = self.f_rgb.prox(self.x_rgb, 1.0)
        y1 = denoiser.bm3d(self.x_rgb, 1.0, is_rgb=True)
        np.testing.assert_allclose(y0, y1, rtol=1e-5)


class TestDnCNN:
    def setup(self):
        key = None
        self.x_sngchn, key = randn((32, 33), key=key, dtype=np.float32)
        self.x_mltchn, key = randn((33, 34, 5), key=key, dtype=np.float32)
        self.dncnn = denoiser.DnCNN()
        self.f = functional.DnCNN()

    def test_sngchn(self):
        y0 = self.f.prox(self.x_sngchn, 1.0)
        y1 = self.dncnn(self.x_sngchn)
        np.testing.assert_allclose(y0, y1, rtol=1e-5)

    def test_mltchn(self):
        y0 = self.f.prox(self.x_mltchn, 1.0)
        y1 = self.dncnn(self.x_mltchn)
        np.testing.assert_allclose(y0, y1, rtol=1e-5)
