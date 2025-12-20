import numpy as np

import jax

import pytest
from test_trainer import SetupTest

from scico import flax as sflax
from scico import random
from scico.flax.train.clu_utils import flatten_dict
from scico.flax.train.diagnostics import ArgumentStruct, compute_metrics, stats_obj
from scico.flax.train.input_pipeline import IterateData, prepare_data
from scico.flax.train.learning_rate import (
    create_cnst_lr_schedule,
    create_cosine_lr_schedule,
    create_exp_lr_schedule,
)
from scico.flax.train.losses import mse_loss
from scico.flax.train.state import create_basic_train_state, initialize


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_mse_loss():
    N = 256
    x, key = random.randn((N, N), seed=4321)
    y, key = random.randn((N, N), key=key)
    # Optax uses a 0.5 factor.
    mse_jnp = 0.5 * jax.numpy.mean((x - y) ** 2)
    mse_optx = mse_loss(y, x)
    np.testing.assert_allclose(mse_jnp, mse_optx)


@pytest.mark.parametrize("batch_size", [4, 8, 16])
def test_data_iterator(testobj, batch_size):
    ds = IterateData(testobj.test_ds_simple, batch_size, train=False)
    N = testobj.test_ds_simple["image"].shape[0]
    assert ds.steps_per_epoch == N // batch_size
    assert ds.key is not None


@pytest.mark.parametrize("local_batch", [8, 16, 24])
def test_dstrain(testobj, local_batch):

    key = jax.random.key(seed=1234)

    train_iter = sflax.create_input_iter(
        key,
        testobj.train_ds_simple,
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


@pytest.mark.parametrize("local_batch", [8, 16, 32])
def test_dstest(testobj, local_batch):

    key = jax.random.key(seed=1234)

    train_iter = sflax.create_input_iter(key, testobj.test_ds_simple, local_batch, train=False)

    nproc = jax.device_count()
    ll = []
    num_steps = 20
    for step, batch in zip(range(num_steps), train_iter):
        for j in range(nproc):
            ll.append(batch["image"][j])

    ll_ = np.array(jax.device_get(ll)).flatten()
    ll_ar = np.array(list(set(np.sort(ll_))))

    np.testing.assert_allclose(ll_ar, np.arange(80, 112))


def test_prepare_data(testobj):
    xbtch = prepare_data(testobj.x)
    local_device_count = jax.local_device_count()
    shrdsz = testobj.x.shape[0] // local_device_count
    assert xbtch.shape == (local_device_count, shrdsz, testobj.N, testobj.N, testobj.chn)


def test_compute_metrics(testobj):
    xbtch = prepare_data(testobj.x)

    xbtch = xbtch / jax.numpy.sqrt(jax.numpy.var(xbtch, axis=(1, 2, 3, 4)))
    ybtch = xbtch + 1

    p_eval = jax.pmap(compute_metrics, axis_name="batch")
    eval_metrics = p_eval(ybtch, xbtch)
    mtrcs = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
    assert np.abs(mtrcs["loss"]) < 0.51
    assert mtrcs["snr"] < 5e-4


def test_cnst_learning_rate(testobj):
    step = 1
    cnst_sch = create_cnst_lr_schedule(testobj.train_conf)
    lr = cnst_sch(step)
    assert lr == testobj.train_conf["base_learning_rate"]


def test_cos_learning_rate(testobj):
    step = 1
    len_train = testobj.train_ds["label"].shape[0]
    train_conf = dict(testobj.train_conf)
    train_conf["steps_per_epoch"] = len_train // testobj.train_conf["batch_size"]
    sch = create_cosine_lr_schedule(train_conf)
    lr = sch(step)
    decay_steps = (train_conf["num_epochs"] - train_conf["warmup_epochs"]) * train_conf[
        "steps_per_epoch"
    ]
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
    np.testing.assert_allclose(lr, train_conf["base_learning_rate"] * cosine_decay, rtol=1e-06)


def test_exp_learning_rate(testobj):
    step = 1
    len_train = testobj.train_ds["label"].shape[0]
    train_conf = dict(testobj.train_conf)
    train_conf["steps_per_epoch"] = len_train // testobj.train_conf["batch_size"]
    steps = train_conf["steps_per_epoch"] * train_conf["num_epochs"]
    sch = create_exp_lr_schedule(train_conf)
    lr = sch(step)
    exp_decay = train_conf["lr_decay_rate"] ** float(step / steps)

    np.testing.assert_allclose(lr, train_conf["base_learning_rate"] * exp_decay, rtol=1e-06)


def test_train_initialize_function(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    key = jax.random.key(seed=4444)
    input_shape = (1, testobj.N, testobj.N, testobj.chn)

    # Via initialize function
    dparams1, dbstats1 = initialize(key, model, input_shape[1:3])
    flat_params1 = flatten_dict(dparams1)
    flat_bstats1 = flatten_dict(dbstats1)
    params1 = [t[1] for t in sorted(flat_params1.items())]
    bstats1 = [t[1] for t in sorted(flat_bstats1.items())]

    # Via model initialization
    variables2 = model.init({"params": key}, np.ones(input_shape, model.dtype))
    flat_params2 = flatten_dict(variables2["params"])
    flat_bstats2 = flatten_dict(variables2["batch_stats"])
    params2 = [t[1] for t in sorted(flat_params2.items())]
    bstats2 = [t[1] for t in sorted(flat_bstats2.items())]

    for i in range(len(params1)):
        np.testing.assert_allclose(params1[i], params2[i], rtol=1e-5)
    for i in range(len(bstats1)):
        np.testing.assert_allclose(bstats1[i], bstats2[i], rtol=1e-5)


def test_create_basic_train_state_default(testobj):
    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    key = jax.random.key(seed=432)
    input_shape = (1, testobj.N, testobj.N, testobj.chn)

    # Model initialization
    variables1 = model.init({"params": key}, np.ones(input_shape, model.dtype))
    flat_params1 = flatten_dict(variables1["params"])
    flat_bstats1 = flatten_dict(variables1["batch_stats"])
    params1 = [t[1] for t in sorted(flat_params1.items())]
    bstats1 = [t[1] for t in sorted(flat_bstats1.items())]

    learning_rate = create_cnst_lr_schedule(testobj.train_conf)

    try:
        # State initialization
        state = create_basic_train_state(
            key, testobj.train_conf, model, input_shape[1:3], learning_rate
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        flat_params2 = flatten_dict(state.params)
        flat_bstats2 = flatten_dict(state.batch_stats)
        params2 = [t[1] for t in sorted(flat_params2.items())]
        bstats2 = [t[1] for t in sorted(flat_bstats2.items())]

        for i in range(len(params1)):
            np.testing.assert_allclose(params1[i], params2[i], rtol=1e-5)
        for i in range(len(bstats1)):
            np.testing.assert_allclose(bstats1[i], bstats2[i], rtol=1e-5)


def test_create_basic_train_state(testobj):
    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    key = jax.random.key(seed=432)
    input_shape = (1, testobj.N, testobj.N, testobj.chn)

    # Model initialization
    variables1 = model.init({"params": key}, np.ones(input_shape, model.dtype))
    flat_params1 = flatten_dict(variables1["params"])
    flat_bstats1 = flatten_dict(variables1["batch_stats"])
    params1 = [t[1] for t in sorted(flat_params1.items())]
    bstats1 = [t[1] for t in sorted(flat_bstats1.items())]

    learning_rate = create_cnst_lr_schedule(testobj.train_conf)

    try:
        # State initialization
        state = create_basic_train_state(
            key, testobj.train_conf, model, input_shape[1:3], learning_rate, variables1
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        flat_params2 = flatten_dict(state.params)
        flat_bstats2 = flatten_dict(state.batch_stats)
        params2 = [t[1] for t in sorted(flat_params2.items())]
        bstats2 = [t[1] for t in sorted(flat_bstats2.items())]

        for i in range(len(params1)):
            np.testing.assert_allclose(params1[i], params2[i], rtol=1e-5)
        for i in range(len(bstats1)):
            np.testing.assert_allclose(bstats1[i], bstats2[i], rtol=1e-5)


def test_sgd_train_state(testobj):
    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    key = jax.random.key(seed=432)
    input_shape = (1, testobj.N, testobj.N, testobj.chn)

    # Model initialization
    variables = model.init({"params": key}, np.ones(input_shape, model.dtype))
    learning_rate = create_cnst_lr_schedule(testobj.train_conf)

    train_conf = dict(testobj.train_conf)
    train_conf["opt_type"] = "SGD"

    try:
        # State initialization
        state = create_basic_train_state(
            key, train_conf, model, input_shape[1:3], learning_rate, variables
        )
    except Exception as e:
        print(e)
        assert 0


def test_sgd_no_momentum_train_state(testobj):
    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    key = jax.random.key(seed=432)
    input_shape = (1, testobj.N, testobj.N, testobj.chn)

    # Model initialization
    variables = model.init({"params": key}, np.ones(input_shape, model.dtype))
    learning_rate = create_cnst_lr_schedule(testobj.train_conf)

    train_conf = dict(testobj.train_conf)
    train_conf["opt_type"] = "SGD"
    train_conf.pop("momentum")

    try:
        # State initialization
        state = create_basic_train_state(
            key, train_conf, model, input_shape[1:3], learning_rate, variables
        )
    except Exception as e:
        print(e)
        assert 0


def test_argument_struct():
    dictaux = {"epochs": 5, "num_steps": 10, "seed": 0}
    try:
        dictstruct = ArgumentStruct(**dictaux)
    except Exception as e:
        print(e)
        assert 0
    else:
        assert hasattr(dictstruct, "epochs")
        assert hasattr(dictstruct, "num_steps")
        assert hasattr(dictstruct, "seed")


def test_complete_stats_obj():
    try:
        itstat_object, itstat_insert_func = stats_obj()
    except Exception as e:
        print(e)
        assert 0
    else:
        summary = {
            "epoch": 3,
            "time": 231.0,
            "train_learning_rate": 1e-2,
            "train_loss": 1.4e-2,
            "train_snr": 3,
            "loss": 1.6e-2,
            "snr": 2.4,
        }
        try:
            itstat_object.insert(itstat_insert_func(ArgumentStruct(**summary)))
        except Exception as e:
            print(e)
            assert 0


def test_except_incomplete_stats_obj():

    itstat_object, itstat_insert_func = stats_obj()
    summary = {
        "epoch": 3,
        "time": 231.0,
        "train_learning_rate": 1e-2,
        "train_loss": 1.4e-2,
        "train_snr": 3,
        "loss": 1.6e-2,
        "snr": 2.4,
    }
    itstat_object.insert(itstat_insert_func(ArgumentStruct(**summary)))
    summary2 = {
        "epoch": 3,
        "time": 231.0,
        "train_learning_rate": 1e-2,
        "train_loss": 1.4e-2,
        "train_snr": 3,
    }
    with pytest.raises(AttributeError):
        itstat_object.insert(itstat_insert_func(ArgumentStruct(**summary2)))


def test_patch_incomplete_stats_obj():

    itstat_object, itstat_insert_func = stats_obj()
    summary = {
        "epoch": 3,
        "time": 231.0,
        "train_learning_rate": 1e-2,
        "train_loss": 1.4e-2,
        "train_snr": 3,
        "loss": 1.6e-2,
        "snr": 2.4,
    }
    itstat_object.insert(itstat_insert_func(ArgumentStruct(**summary)))
    summary2 = {
        "epoch": 3,
        "time": 231.0,
        "train_learning_rate": 1e-2,
        "train_loss": 1.4e-2,
        "train_snr": 3,
    }

    try:
        summary2["loss"] = -1
        summary2["snr"] = -1
        itstat_object.insert(itstat_insert_func(ArgumentStruct(**summary2)))
    except Exception as e:
        print(e)
        assert 0
