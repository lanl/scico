import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico import functional


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


def test_functional_sum():
    x = np.random.randn(4, 4)
    f0 = functional.L1Norm()
    f1 = 2.0 * functional.L2Norm()
    f = f0 + f1
    assert f(x) == f0(x) + f1(x)
    with pytest.raises(TypeError):
        f = f0 + 2.0


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


@pytest.mark.parametrize("axis", [0, 1, (0, 2)])
def test_l21norm(axis):
    x = np.ones((3, 4, 5))
    if isinstance(axis, int):
        l2axis = (axis,)
    else:
        l2axis = axis
    l2shape = [x.shape[k] for k in l2axis]
    l1axis = tuple(set(range(len(x))) - set(l2axis))
    l1shape = [x.shape[k] for k in l1axis]

    l21ana = np.sqrt(np.prod(l2shape)) * np.prod(l1shape)
    F = functional.L21Norm(l2_axis=axis)
    l21num = F(x)
    np.testing.assert_allclose(l21ana, l21num, rtol=1e-5)

    l2ana = np.sqrt(np.prod(l2shape))
    prxana = (l2ana - 1.0) / l2ana * x
    prxnum = F.prox(x, 1.0)
    np.testing.assert_allclose(prxana, prxnum, rtol=1e-5)
