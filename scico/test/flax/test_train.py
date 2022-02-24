import pytest

import numpy as np

import jax
from scico.flax import create_input_iter


class DatasetTest:
    def __init__(self):
        datain = np.arange(40)
        datain_test = np.arange(40, 60)
        dataout = np.zeros(40)
        dataout[:20] = 1
        dataout_test = np.zeros(20)
        dataout_test[:10] = 1

        self.train_ds = {"image": datain, "label": dataout}
        self.test_ds = {"image": datain_test, "label": dataout_test}


@pytest.fixture(scope="module")
def testobj():
    yield DatasetTest()


def test_dataset_train_iter(testobj):

    key = jax.random.PRNGKey(seed=1234)
    local_batch_size = 4

    train_iter = create_input_iter(
        key,
        testobj.train_ds,
        local_batch_size,
    )

    nproc = jax.device_count()
    ll = []
    num_steps = 20
    for step, batch in zip(range(num_steps), train_iter):
        for j in range(nproc):
            ll.append(batch["image"][j])

    ll_ = np.array(jax.device_get(ll)).flatten()
    ll_ar = np.array(list(set(np.sort(ll_))))

    np.testing.assert_allclose(ll_ar, np.arange(40))


def test_dataset_test_iter(testobj):

    key = jax.random.PRNGKey(seed=1234)
    local_batch_size = 4

    train_iter = create_input_iter(key, testobj.test_ds, local_batch_size, train=False)

    nproc = jax.device_count()
    ll = []
    num_steps = 10
    for step, batch in zip(range(num_steps), train_iter):
        for j in range(nproc):
            ll.append(batch["image"][j])

    ll_ = np.array(jax.device_get(ll)).flatten()
    ll_ar = np.array(list(set(np.sort(ll_))))

    np.testing.assert_allclose(ll_ar, np.arange(40, 60))
