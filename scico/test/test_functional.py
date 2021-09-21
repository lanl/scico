import numpy as np

from jax.config import config

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)

import jax

import pytest

import scico.numpy as snp
from scico import functional, linop, loss
from scico.blockarray import BlockArray
from scico.random import randn
from scico.solver import minimize

NO_BLOCK_ARRAY = [
    functional.L21Norm,
]


def prox_func(x, v, f, alpha):
    """Evaluate functional of which the proximal operator is the argmin."""
    return 0.5 * snp.sum(snp.abs(x.reshape(v.shape) - v) ** 2) + alpha * snp.array(
        f(x.reshape(v.shape)), dtype=np.float64
    )


def prox_solve(v, v0, f, alpha):
    """Evaluate the alpha-scaled proximal operator of f at v, using v0 as an
    initial point for the optimization."""
    fnc = lambda x: prox_func(x, v, f, alpha)
    fmn = minimize(
        fnc,
        v0.ravel(),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-9, "fatol": 1e-9},
    )

    return fmn.x.reshape(v.shape), fmn.fun


def prox_test(v, nrm, prx, alpha):
    """Test the alpha-scaled proximal operator function prx of norm functional nrm
    at point v."""
    # Evaluate the proximal operator at v
    px = snp.array(prx(v, alpha))
    # Proximal operator functional value (i.e. Moreau envelope) at v
    pf = prox_func(px, v, nrm, alpha)
    # Brute-force solve of the proximal operator at v
    mx, mf = prox_solve(v, px, nrm, alpha)
    # Compare prox functional value with brute-force solution
    assert mf >= pf or (pf - mf) / pf <= 1e-3
    # Compare prox solution with brute-force solution
    np.testing.assert_allclose(np.linalg.norm(mx), np.linalg.norm(px), rtol=5e-2)


class ProxTestObj:
    def __init__(self, dtype):
        key = None
        self.v, key = randn(shape=(32, 1), dtype=dtype, key=key)
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
    # TODO: should tweak test to verify that there is a warning on f.grad and fg.grad
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

        assert scaled.is_smooth == unscaled.is_smooth
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
        alpha = 0.1
        cnsobj = cnstr()
        cns = cnsobj.__call__
        prx = cnsobj.prox
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
    # Ensure that the has_eval, has_prox, is_smooth attrs are overridden
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

    @pytest.mark.parametrize("cls", to_check)
    def test_is_smooth(self, cls):
        assert isinstance(cls.is_smooth, bool)


def test_scalar_vmap():
    x = np.random.randn(4, 4)
    f = functional.L1Norm()

    def foo(c):
        return (c * f)(x)

    c_list = snp.array([1.0, 2.0, 3.0])
    non_vmap = np.array([foo(c) for c in c_list])
    vmapped = jax.vmap(foo)(c_list)
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
        W = 0.1 * np.random.rand(n) + 1.0
        self.Ao = linop.MatrixOperator(A)
        self.Do = linop.Diagonal(D)
        self.Wo = linop.Diagonal(W)
        self.y, key = randn((n,), key=key, dtype=dtype)
        self.v, key = randn((n,), key=key, dtype=dtype)  # point for prox eval
        scalar, key = randn((1,), key=key, dtype=dtype)
        self.scalar = scalar.copy().ravel()[0]

    def test_squared_l2(self):
        L = loss.SquaredL2Loss(y=self.y, A=self.Ao)
        assert L.is_smooth == True
        assert L.has_eval == True
        assert L.has_prox == False  # not diagonal

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale

        # SquaredL2 with Diagonal linop has a prox
        L_d = loss.SquaredL2Loss(y=self.y, A=self.Do)

        assert L_d.is_smooth == True
        assert L_d.has_eval == True
        assert L_d.has_prox == True

        cL = self.scalar * L_d
        assert L_d.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L_d.scale

        pf = prox_test(self.v, L_d, L_d.prox, 0.75)

    def test_weighted_squared_l2(self):
        L = loss.WeightedSquaredL2Loss(y=self.y, A=self.Ao, weight_op=self.Wo)
        assert L.is_smooth == True
        assert L.has_eval == True
        assert L.has_prox == False  # not diagonal

        cL = self.scalar * L
        assert L.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L.scale

        # SquaredL2 with Diagonal linop has a prox
        Wo = self.Wo
        L_d = loss.WeightedSquaredL2Loss(y=self.y, A=self.Do, weight_op=Wo)

        assert L_d.is_smooth == True
        assert L_d.has_eval == True
        assert L_d.has_prox == True

        cL = self.scalar * L_d
        assert L_d.scale == 0.5  # hasn't changed
        assert cL.scale == self.scalar * L_d.scale

        pf = prox_test(self.v, L_d, L_d.prox, 0.75)


class TestBM3D:
    def setup(self):
        key = None
        N = 32
        self.x, key = randn((N, N), key=key, dtype=np.float32)
        self.x_rgb, key = randn((N, N, 3), key=key, dtype=np.float32)

        self.f = functional.BM3D()
        self.f_rgb = functional.BM3D(is_rgb=True)

    def test_prox(self):
        no_jit = self.f.prox(self.x, 1.0)
        jitted = jax.jit(self.f.prox)(self.x, 1.0)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

    def test_prox_rgb(self):
        no_jit = self.f_rgb.prox(self.x_rgb, 1.0)
        jitted = jax.jit(self.f_rgb.prox)(self.x_rgb, 1.0)
        np.testing.assert_allclose(no_jit, jitted, rtol=1e-3)
        assert no_jit.dtype == np.float32
        assert jitted.dtype == np.float32

    def test_prox_bad_inputs(self):

        x, key = randn((32,), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            self.f.prox(x, 1.0)

        x, key = randn((12, 12, 4, 3), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            self.f.prox(x, 1.0)

        x_b, key = randn(((2, 3), (3, 4, 5)), key=None, dtype=np.float32)
        with pytest.raises(ValueError):
            self.f.prox(x, 1.0)

        z, key = randn((32, 32), key=None, dtype=np.complex64)
        with pytest.raises(TypeError):
            self.f.prox(z, 1.0)
