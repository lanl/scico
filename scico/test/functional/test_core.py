import numpy as np

from jax.config import config

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)


import pytest
from prox import prox_test

import scico.numpy as snp
from scico import functional
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


class TestNormProx:

    alphalist = [1e-2, 1e-1, 1e0, 1e1]
    normlist = [
        functional.L0Norm,
        functional.L1Norm,
        functional.SquaredL2Norm,
        functional.L2Norm,
        functional.L21Norm,
        functional.HuberNorm,
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
    ignore = [
        functional.Functional,
        functional.ScaledFunctional,
        functional.SetDistance,
        functional.SquaredSetDistance,
    ]
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
    sdistlist = [functional.SetDistance, functional.SquaredSetDistance]
    alphalist = [1e-2, 1e-1, 1e0, 1e1]

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

    @pytest.mark.parametrize("sdist", sdistlist)
    @pytest.mark.parametrize("cnstr", cnstrlist)
    @pytest.mark.parametrize("alpha", alphalist)
    def test_setdistance(self, sdist, cnstr, alpha, test_proj_obj):
        if cnstr in NO_COMPLEX and snp.iscomplexobj(test_proj_obj.v):
            return
        cnsobj = cnstr()
        proj = cnsobj.prox
        sdobj = sdist(proj)
        call = sdobj.__call__
        prox = sdobj.prox
        prox_test(test_proj_obj.v, call, prox, alpha)
