import numpy as np

from jax.config import config

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)

import warnings

import jax

import pytest
from prox import prox_test

import scico.numpy as snp
from scico import denoiser, functional
from scico.blockarray import BlockArray
from scico.denoiser import have_bm3d
from scico.random import randn

NO_BLOCK_ARRAY = [functional.L21Norm, functional.NuclearNorm]
NO_COMPLEX = [
    functional.NonNegativeIndicator,
]


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


@pytest.mark.skipif(not have_bm3d, reason="bm3d package not installed")
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
