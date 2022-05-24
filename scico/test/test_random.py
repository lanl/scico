import numpy as np

import jax

import pytest

import scico.random


@pytest.mark.parametrize("seed", [None, 42])
def test_wrapped_funcs(seed):
    fun = jax.random.normal
    fun_wrapped = scico.random.normal

    # test seed argument
    if seed is None:
        key = jax.random.PRNGKey(0)
    else:
        key = jax.random.PRNGKey(seed)

    np.testing.assert_array_equal(fun(key), fun_wrapped(seed=seed)[0])

    # test blockarray
    shape = ((7,), (3, 2), (2, 4, 1))
    seed = 42
    key = jax.random.PRNGKey(seed)

    result, _ = fun_wrapped(shape, seed=seed)


def test_add_seed_adapter():
    fun = jax.random.normal

    fun_alt = scico.random._add_seed(fun)

    # specify a seed instead of a key
    assert fun(jax.random.PRNGKey(42)) == fun_alt(seed=42)[0]

    # seed defaults to zero
    assert fun(jax.random.PRNGKey(0)) == fun_alt()[0]

    # other parameters still work...
    key = jax.random.PRNGKey(0)
    sz = (10, 3)
    dtype = np.float64

    # ...positional
    np.testing.assert_array_equal(fun(key, sz), fun_alt(sz)[0])
    np.testing.assert_array_equal(fun(key, sz, dtype), fun_alt(sz, dtype)[0])
    np.testing.assert_array_equal(fun(key, sz, dtype), fun_alt(sz, dtype, key)[0])
    np.testing.assert_array_equal(fun(key, sz, dtype), fun_alt(sz, dtype, None, 0)[0])

    # ... keyword
    np.testing.assert_array_equal(fun(shape=sz, key=key), fun_alt(shape=sz)[0])
    np.testing.assert_array_equal(
        fun(shape=sz, key=key, dtype=dtype), fun_alt(dtype=dtype, shape=sz)[0]
    )

    # ... mixed
    np.testing.assert_array_equal(
        fun(key, dtype=dtype, shape=sz), fun_alt(dtype=dtype, shape=sz)[0]
    )

    # get back the split key
    _, key_a = fun_alt(seed=42)
    key_b, _ = jax.random.split(jax.random.PRNGKey(42), 2)
    np.testing.assert_array_equal(key_a, key_b)

    # error when key and seed are specified
    with pytest.raises(Exception):
        _ = fun_alt(key=jax.random.PRNGKey(0), seed=42)[0]
