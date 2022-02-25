import pytest

import numpy as np

import jax
from scico import random
from scico.flax import create_input_iter, compute_metrics
from scico.flax.train.input_pipeline import prepare_data


class DatasetTest:
    def __init__(self):
        datain = np.arange(80)
        datain_test = np.arange(80, 120)
        dataout = np.zeros(80)
        dataout[:40] = 1
        dataout_test = np.zeros(40)
        dataout_test[:20] = 1

        self.train_ds = {"image": datain, "label": dataout}
        self.test_ds = {"image": datain_test, "label": dataout_test}


@pytest.fixture(scope="module")
def testobj():
    yield DatasetTest()


@pytest.mark.parametrize("local_batch", [2, 4, 8])
def test_dataset_train_iter(testobj, local_batch):

    key = jax.random.PRNGKey(seed=1234)

    train_iter = create_input_iter(
        key,
        testobj.train_ds,
        local_batch,
    )

    nproc = jax.device_count()
    ll = []
    num_steps = 40
    for step, batch in zip(range(num_steps), train_iter):
        for j in range(nproc):
            ll.append(batch["image"][j])

    ll_ = np.array(jax.device_get(ll)).flatten()
    ll_ar = np.array(list(set(np.sort(ll_))))

    np.testing.assert_allclose(ll_ar, np.arange(80))


@pytest.mark.parametrize("local_batch", [2, 4, 8])
def test_dataset_test_iter(testobj, local_batch):

    key = jax.random.PRNGKey(seed=1234)

    train_iter = create_input_iter(key,
        testobj.test_ds,
        local_batch,
        train=False)

    nproc = jax.device_count()
    ll = []
    num_steps = 20
    for step, batch in zip(range(num_steps), train_iter):
        for j in range(nproc):
            ll.append(batch["image"][j])

    ll_ = np.array(jax.device_get(ll)).flatten()
    ll_ar = np.array(list(set(np.sort(ll_))))

    np.testing.assert_allclose(ll_ar, np.arange(80, 120))


def test_train_metrics():
    N = 128 # Signal size
    chn = 1 # Number of channels
    bsize = 10 # Batch size
    x, key = random.randn((bsize, N, N, chn), seed=1234)

    xbtch = prepare_data(x)

    xbtch = xbtch / jax.numpy.sqrt(jax.numpy.var(xbtch, axis=(1,2,3,4)))
    ybtch = xbtch + 1

    p_eval = jax.pmap(compute_metrics, axis_name='batch')
    mtrcs = p_eval(ybtch, xbtch)
    assert np.abs(mtrcs['loss']) < 0.51
    assert mtrcs['snr'] < 1e-6

