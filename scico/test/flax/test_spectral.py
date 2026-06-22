from functools import partial
from typing import Any, Tuple

import numpy as np

import jax

import pytest

from flax.linen import Conv
from flax.linen.module import Module, compact
from scico import flax as sflax
from scico import linop
from scico.flax.train.spectral import (
    _l2_normalize,
    conv,
    estimate_spectral_norm,
    exact_spectral_norm,
    spectral_normalization_conv,
)
from scico.flax.train.traversals import construct_traversal
from scico.random import randn


def test_l2_normalize():
    N = 256
    x, key = randn((N, N), seed=135)

    eps = 1e-6
    l2_jnp = jax.numpy.sqrt((x**2).sum())
    l2n_jnp = x / (l2_jnp + eps)
    l2n_util = _l2_normalize(x, eps)
    np.testing.assert_allclose(l2n_jnp, l2n_util, rtol=eps)


@pytest.mark.parametrize("kernel_size", [(3, 3, 1, 1), (11, 11, 1, 1)])
def test_conv(kernel_size):
    key = jax.random.key(97531)
    kernel, key = randn(kernel_size, dtype=np.float32, key=key)

    input_size = (1, 128, 128, 1)
    x, key = randn(input_size, dtype=np.float32, key=key)

    pads = (
        [(0, 0)]
        + [(kernel_size[0] // 2, kernel_size[0] // 2)]
        + [(kernel_size[1] // 2, kernel_size[1] // 2)]
        + [(0, 0)]
    )
    xext = np.pad(x, pads, mode="wrap")

    y = jax.scipy.signal.convolve(xext.squeeze(), jax.numpy.flip(kernel).squeeze(), mode="valid")

    y_util = conv(x, kernel).squeeze()

    np.testing.assert_allclose(y, y_util)


class CNN(Module):
    kernel_size: Tuple[int, int]
    kernel0: Any

    @compact
    def __call__(self, x):
        def kinit_wrap(rng, shape, dtype=np.float32):
            return np.array(self.kernel0, dtype)

        return Conv(
            features=1,
            kernel_size=self.kernel_size,
            use_bias=False,
            padding="CIRCULAR",
            kernel_init=kinit_wrap,
        )(x)


@pytest.mark.parametrize("kernel_size", [(3, 3, 1, 1), (11, 11, 1, 1)])
def test_conv_layer(kernel_size):
    key = jax.random.key(12345)
    kernel, key = randn(kernel_size, dtype=np.float32, key=key)

    input_size = (1, 128, 128, 1)
    x, key = randn(input_size, dtype=np.float32, key=key)

    rng = jax.random.key(42)
    model = CNN(kernel_size=kernel_size[:2], kernel0=kernel)
    variables = model.init(rng, np.zeros(x.shape))
    prms = variables["params"]
    np.testing.assert_allclose(prms["Conv_0"]["kernel"], kernel)

    y_layer = model.apply(variables, x)
    y_util = conv(x, kernel)

    np.testing.assert_allclose(y_layer, y_util)


@pytest.mark.parametrize("input_shape", [(8,), (128,)])
def test_spectral_norm(input_shape):
    key = jax.random.key(1357)
    diagonal, key = randn(input_shape, dtype=np.float32, key=key)

    mu = np.linalg.norm(np.diag(diagonal), 2)

    D = linop.Diagonal(diagonal=diagonal)
    x, key = randn(input_shape, dtype=np.float32, key=key)
    mu_util = estimate_spectral_norm(lambda x: D @ x, x.shape, n_steps=200)

    np.testing.assert_allclose(mu, mu_util, rtol=1e-6)


@pytest.mark.parametrize("kernel_shape", [(3, 3, 1, 1), (7, 7, 1, 1)])
def test_spectral_norm_conv(kernel_shape):

    key = jax.random.key(2468)
    kernel, key = randn(kernel_shape, dtype=np.float32, key=key)

    input_shape = (1, 32, 32, 1)
    x, key = randn(input_shape, dtype=np.float32, key=key)

    sn = exact_spectral_norm(lambda x: conv(x, kernel), x.shape)

    sn_util = estimate_spectral_norm(lambda x: conv(x, kernel), x.shape, n_steps=100)

    np.testing.assert_allclose(sn, sn_util, rtol=1e-3, atol=1e-2)


def test_train_spectral_norm():
    depth = 3
    channels = 1
    num_filters = 16
    model = sflax.DnCNNNet(depth, channels, num_filters)

    train_conf: sflax.ConfigDict = {
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
    xtr, key = randn((train_conf["batch_size"], N, N, channels), seed=4321)
    xtt, key = randn((train_conf["batch_size"], N, N, channels), key=key)
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
        train_conf["post_lst"] = [kernelnorm]
        trainer = sflax.BasicFlaxTrainer(
            train_conf,
            model,
            train_ds,
            test_ds,
        )
        modvar, _ = trainer.train()
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
