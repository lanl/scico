from functools import partial

import numpy as np

import jax

import pytest

from scico import flax as sflax
from scico import linop
from scico.flax.train.train import construct_traversal
from scico.flax.train.utils import (
    _l2_normalize,
    conv,
    estimate_spectral_norm,
    exact_spectral_norm,
    spectral_normalization_conv,
)
from scico.random import randn


def test_l2_normalize():
    N = 256
    x, key = randn((N, N), seed=135)

    eps = 1e-6
    l2_jnp = jax.numpy.sqrt((x**2).sum())
    l2n_jnp = x / (l2_jnp + eps)
    l2n_util = _l2_normalize(x, eps)
    np.testing.assert_allclose(l2n_jnp, l2n_util, rtol=eps)


@pytest.mark.parametrize("kernel_shape", [(3, 3, 1, 1), (11, 11, 1, 1)])
def test_conv(kernel_shape):
    key = jax.random.PRNGKey(97531)
    kernel, key = randn(kernel_shape, dtype=np.float32, key=key)

    input_shape = (1, 128, 128, 1)
    x, key = randn(input_shape, dtype=np.float32, key=key)

    y = jax.scipy.signal.convolve(x.squeeze(), jax.numpy.flip(kernel).squeeze(), mode="same")

    y_util = conv(x, kernel).squeeze()

    np.testing.assert_allclose(y, y_util)


@pytest.mark.parametrize("input_shape", [(8,), (128,)])
def test_spectral_norm(input_shape):
    key = jax.random.PRNGKey(1357)
    diagonal, key = randn(input_shape, dtype=np.float32, key=key)

    mu = np.linalg.norm(np.diag(diagonal), 2)

    D = linop.Diagonal(diagonal=diagonal)
    x, key = randn(input_shape, dtype=np.float32, key=key)
    mu_util = estimate_spectral_norm(lambda x: D @ x, x.shape, n_steps=200)

    np.testing.assert_allclose(mu, mu_util)


@pytest.mark.parametrize("kernel_shape", [(3, 3, 1, 1), (7, 7, 1, 1)])
def test_spectral_norm_conv(kernel_shape):

    key = jax.random.PRNGKey(2468)
    kernel, key = randn(kernel_shape, dtype=np.float32, key=key)

    input_shape = (1, 32, 32, 1)
    x, key = randn(input_shape, dtype=np.float32, key=key)

    sn = exact_spectral_norm(lambda x: conv(x, kernel), x.shape)

    sn_util = estimate_spectral_norm(lambda x: conv(x, kernel), x.shape, n_steps=200)

    np.testing.assert_allclose(sn, sn_util, rtol=5e-5)


def test_train_spectral_norm():
    depth = 3
    channels = 1
    num_filters = 16
    model = sflax.DnCNNNet(depth, channels, num_filters)

    dconf: sflax.ConfigDict = {
        "seed": 0,
        "opt_type": "ADAM",
        "batch_size": 16,
        "num_epochs": 1,
        "base_learning_rate": 1e-3,
        "lr_decay_rate": 0.95,
        "warmup_epochs": 0,
        "num_train_steps": -1,
        "steps_per_eval": -1,
        "steps_per_epoch": 1,
        "log_every_steps": 1000,
    }

    N = 64
    xtr, key = randn((dconf["batch_size"], N, N, channels), seed=4321)
    xtt, key = randn((dconf["batch_size"], N, N, channels), key=key)
    train_ds = {"image": xtr, "label": xtr}
    test_ds = {"image": xtt, "label": xtt}

    try:
        xshape = (1,) + train_ds["label"][0].shape
        convtrav = construct_traversal("kernel")
        kernelnorm = partial(
            spectral_normalization_conv,
            traversal=convtrav,
            xshape=xshape,
        )
        modvar, _ = sflax.train_and_evaluate(
            dconf,
            "./",
            model,
            train_ds,
            test_ds,
            post_lst=[kernelnorm],
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        knlsn = np.array(
            [
                estimate_spectral_norm(
                    lambda x: conv(x, kernel), (1, xshape[1], xshape[2], kernel.shape[2])
                )
                for kernel in convtrav.iterate(modvar["params"])
            ]
        )
        np.testing.assert_array_less(knlsn, np.ones(knlsn.shape))
