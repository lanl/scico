import functools
import os

import numpy as np

import jax

import optax
import pytest

from flax import jax_utils
from scico import flax as sflax
from scico import random
from scico.flax.train.apply import _apply_fn
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
from scico.flax.train.steps import _eval_step, _train_step, _train_step_post
from scico.flax.train.trainer import sync_batch_stats
from scico.flax.train.traversals import clip_positive, clip_range, construct_traversal

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def test_mse_loss():
    N = 256
    x, key = random.randn((N, N), seed=4321)
    y, key = random.randn((N, N), key=key)
    # Optax uses a 0.5 factor.
    mse_jnp = 0.5 * jax.numpy.mean((x - y) ** 2)
    mse_optx = mse_loss(y, x)
    np.testing.assert_allclose(mse_jnp, mse_optx)


class SetupTest:
    def __init__(self):
        datain = np.arange(80)
        datain_test = np.arange(80, 112)
        dataout = np.zeros(80)
        dataout[:40] = 1
        dataout_test = np.zeros(40)
        dataout_test[:20] = 1

        self.train_ds_simple = {"image": datain, "label": dataout}
        self.test_ds_simple = {"image": datain_test, "label": dataout_test}

        # More complex data structure
        self.N = 128  # signal size
        self.chn = 1  # number of channels
        self.bsize = 16  # batch size
        self.x, key = random.randn((4 * self.bsize, self.N, self.N, self.chn), seed=4321)

        xt, key = random.randn((32, self.N, self.N, self.chn), key=key)

        self.train_ds = {"image": self.x, "label": self.x}
        self.test_ds = {"image": xt, "label": xt}

        # Model configuration
        self.model_conf = {
            "depth": 2,
            "num_filters": 16,
            "block_depth": 2,
        }

        # Training configuration
        self.train_conf: sflax.ConfigDict = {
            "seed": 0,
            "opt_type": "ADAM",
            "momentum": 0.9,
            "batch_size": self.bsize,
            "num_epochs": 1,
            "base_learning_rate": 1e-3,
            "lr_decay_rate": 0.95,
            "warmup_epochs": 0,
            "log_every_steps": 1000,
        }


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


@pytest.mark.parametrize("batch_size", [4, 8, 16])
def test_data_iterator(testobj, batch_size):
    ds = IterateData(testobj.test_ds_simple, batch_size, train=False)
    N = testobj.test_ds_simple["image"].shape[0]
    assert ds.steps_per_epoch == N // batch_size
    assert ds.key is not None


@pytest.mark.parametrize("local_batch", [8, 16, 24])
def test_dstrain(testobj, local_batch):

    key = jax.random.PRNGKey(seed=1234)

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

    key = jax.random.PRNGKey(seed=1234)

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


@pytest.mark.parametrize("opt_type", ["SGD", "ADAM", "ADAMW"])
def test_optimizers(testobj, opt_type):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    train_conf = dict(testobj.train_conf)
    train_conf["opt_type"] = opt_type
    try:
        trainer = sflax.BasicFlaxTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
        modvar, _ = trainer.train()
    except Exception as e:
        print(e)
        assert 0


def test_optimizers_exception(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    train_conf = dict(testobj.train_conf)
    train_conf["opt_type"] = ""
    with pytest.raises(NotImplementedError):
        sflax.BasicFlaxTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )


def test_train_initialize_function(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    key = jax.random.PRNGKey(seed=4444)
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

    key = jax.random.PRNGKey(seed=432)
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

    key = jax.random.PRNGKey(seed=432)
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


def test_basic_train_step(testobj):
    key = jax.random.PRNGKey(seed=531)
    key1, key2 = jax.random.split(key)

    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    input_shape = (1, testobj.N, testobj.N, testobj.chn)
    learning_rate = create_cnst_lr_schedule(testobj.train_conf)
    state = create_basic_train_state(key1, testobj.train_conf, model, input_shape, learning_rate)
    criterion = mse_loss

    local_batch_size = testobj.train_conf["batch_size"] // jax.process_count()
    size_device_prefetch = 2
    train_dt_iter = sflax.create_input_iter(
        key2,
        testobj.train_ds,
        local_batch_size,
        size_device_prefetch,
        model.dtype,
        train=True,
    )
    # Training is configured as parallel operation
    state = jax_utils.replicate(state)
    p_train_step = jax.pmap(
        functools.partial(
            _train_step,
            learning_rate_fn=learning_rate,
            criterion=criterion,
            metrics_fn=compute_metrics,
        ),
        axis_name="batch",
    )

    try:
        batch = next(train_dt_iter)
        p_train_step(state, batch)
    except Exception as e:
        print(e)
        assert 0


@pytest.mark.parametrize("pname", ["kernel", "bias", "scale"])
def test_construct_traversal(testobj, pname):

    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    ndim = 1
    if pname == "kernel":
        ndim = 4

    key = jax.random.PRNGKey(seed=432)
    input_shape = (1, testobj.N, testobj.N, testobj.chn)
    variables = model.init({"params": key}, np.ones(input_shape, model.dtype))

    ptrav = construct_traversal(pname)
    for pm in ptrav.iterate(variables["params"]):
        assert len(pm.shape) == ndim


def test_post_train_step(testobj):
    key = jax.random.PRNGKey(seed=531)
    key1, key2 = jax.random.split(key)

    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    input_shape = (1, testobj.N, testobj.N, testobj.chn)
    learning_rate = create_cnst_lr_schedule(testobj.train_conf)
    state = create_basic_train_state(key1, testobj.train_conf, model, input_shape, learning_rate)
    criterion = mse_loss

    local_batch_size = testobj.train_conf["batch_size"] // jax.process_count()
    size_device_prefetch = 2
    train_dt_iter = sflax.create_input_iter(
        key2,
        testobj.train_ds,
        local_batch_size,
        size_device_prefetch,
        model.dtype,
        train=True,
    )
    # Dum range requirement over kernel parameters
    ktrav = construct_traversal("kernel")
    krange = functools.partial(clip_range, traversal=ktrav, minval=1e-5, maxval=1e1)
    # Training is configured as parallel operation
    state = jax_utils.replicate(state)
    p_train_step = jax.pmap(
        functools.partial(
            _train_step_post,
            learning_rate_fn=learning_rate,
            criterion=criterion,
            train_step_fn=_train_step,
            metrics_fn=compute_metrics,
            post_lst=[krange],
        ),
        axis_name="batch",
    )

    try:
        batch = next(train_dt_iter)
        p_train_step(state, batch)
    except Exception as e:
        print(e)
        assert 0


def test_basic_eval_step(testobj):
    key = jax.random.PRNGKey(seed=531)
    key1, key2 = jax.random.split(key)

    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    input_shape = (1, testobj.N, testobj.N, testobj.chn)
    learning_rate = create_cnst_lr_schedule(testobj.train_conf)
    state = create_basic_train_state(key1, testobj.train_conf, model, input_shape, learning_rate)
    criterion = mse_loss

    local_batch_size = testobj.train_conf["batch_size"] // jax.process_count()
    size_device_prefetch = 2
    eval_dt_iter = sflax.create_input_iter(
        key2,
        testobj.test_ds,
        local_batch_size,
        size_device_prefetch,
        model.dtype,
        train=False,
    )
    # Evaluation is configured as parallel operation
    state = jax_utils.replicate(state)
    p_eval_step = jax.pmap(
        functools.partial(_eval_step, criterion=criterion, metrics_fn=compute_metrics),
        axis_name="batch",
    )

    try:
        batch = next(eval_dt_iter)
        p_eval_step(state, batch)
    except Exception as e:
        print(e)
        assert 0


def test_sync_batch_stats(testobj):
    key = jax.random.PRNGKey(seed=12345)
    key1, key2 = jax.random.split(key)

    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    input_shape = (1, testobj.N, testobj.N, testobj.chn)
    learning_rate = create_cnst_lr_schedule(testobj.train_conf)
    state0 = create_basic_train_state(key1, testobj.train_conf, model, input_shape, learning_rate)

    # For parallel training
    state = jax_utils.replicate(state0)
    state = sync_batch_stats(state)
    state1 = jax_utils.unreplicate(state)

    flat_bstats0 = flatten_dict(state0.batch_stats)
    bstats0 = [t[1] for t in sorted(flat_bstats0.items())]

    flat_bstats1 = flatten_dict(state1.batch_stats)
    bstats1 = [t[1] for t in sorted(flat_bstats1.items())]

    for i in range(len(bstats0)):
        np.testing.assert_allclose(bstats0[i], bstats1[i], rtol=1e-5)


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
    with pytest.raises(Exception):
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


def test_class_train_default_init(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )
    try:
        trainer = sflax.BasicFlaxTrainer(
            testobj.train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert trainer.itstat_object is None


def test_class_train_default_noseed(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )
    train_conf = dict(testobj.train_conf)
    train_conf.pop("seed", None)
    try:
        trainer = sflax.BasicFlaxTrainer(
            testobj.train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0


def test_class_train_nolog(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    train_conf = dict(testobj.train_conf)
    train_conf["log"] = False
    try:
        trainer = sflax.BasicFlaxTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert trainer.itstat_object is None


def test_class_train_required_steps(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )
    train_conf = dict(testobj.train_conf)
    train_conf.pop("batch_size", None)
    train_conf.pop("num_epochs", None)
    try:
        trainer = sflax.BasicFlaxTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        batch_size = 2 * jax.device_count()
        local_batch_size = batch_size // jax.process_count()
        num_epochs = 10
        num_steps = int(trainer.steps_per_epoch * num_epochs)
        assert trainer.local_batch_size == local_batch_size
        assert trainer.num_steps == num_steps


def test_except_class_train_batch_size(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )
    train_conf = dict(testobj.train_conf)
    train_conf["batch_size"] = 5
    with pytest.raises(ValueError):
        trainer = sflax.BasicFlaxTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )


def test_class_train_set_steps(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )
    train_conf = dict(testobj.train_conf)
    train_conf["steps_per_eval"] = 1
    train_conf["steps_per_checkpoint"] = 1
    train_conf["log_every_steps"] = 3
    try:
        trainer = sflax.BasicFlaxTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert trainer.steps_per_eval == train_conf["steps_per_eval"]
        assert trainer.steps_per_checkpoint == train_conf["steps_per_checkpoint"]
        assert trainer.log_every_steps == train_conf["log_every_steps"]


def test_class_train_set_reporting(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )
    train_conf = dict(testobj.train_conf)
    train_conf["log"] = True
    train_conf["workdir"] = "./out/"
    train_conf["checkpointing"] = False
    train_conf["return_state"] = True
    try:
        trainer = sflax.BasicFlaxTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert trainer.logflag == train_conf["log"]
        assert trainer.workdir == train_conf["workdir"]
        assert trainer.checkpointing == train_conf["checkpointing"]
        assert trainer.return_state == train_conf["return_state"]


def test_class_train_set_functions(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    def huber_loss(output, labels):
        return jax.numpy.mean(optax.huber_loss(output, labels))

    # Dum range requirement over kernel parameters
    ktrav = construct_traversal("kernel")
    krange = functools.partial(clip_range, traversal=ktrav, minval=1e-5, maxval=1e1)

    train_conf = dict(testobj.train_conf)
    train_conf["criterion"] = huber_loss
    train_conf["create_train_state"] = create_basic_train_state
    train_conf["train_step_fn"] = _train_step
    train_conf["eval_step_fn"] = _eval_step
    train_conf["post_lst"] = [krange]
    try:
        trainer = sflax.BasicFlaxTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert trainer.criterion == train_conf["criterion"]
        assert trainer.create_train_state == train_conf["create_train_state"]
        assert trainer.train_step_fn == train_conf["train_step_fn"]
        assert trainer.eval_step_fn == train_conf["eval_step_fn"]
        assert trainer.post_lst[0] == train_conf["post_lst"][0]
        assert hasattr(trainer, "lr_schedule")


def test_class_train_set_iterators(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )
    try:
        trainer = sflax.BasicFlaxTrainer(
            testobj.train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert trainer.ishape == testobj.train_ds["image"].shape[1:3]
        assert hasattr(trainer, "train_dt_iter")
        assert hasattr(trainer, "eval_dt_iter")


@pytest.mark.parametrize("postl", [False, True])
def test_class_train_set_parallel(testobj, postl):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    train_conf = dict(testobj.train_conf)

    train_conf["post_lst"] = []
    if postl:
        # Dum range requirement over kernel parameters
        ktrav = construct_traversal("kernel")
        krange = functools.partial(clip_range, traversal=ktrav, minval=1e-5, maxval=1e1)
        train_conf["post_lst"] = [krange]

    try:
        trainer = sflax.BasicFlaxTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert hasattr(trainer, "p_train_step")
        assert hasattr(trainer, "p_eval_step")


@pytest.mark.parametrize("chkflag", [False, True])
def test_class_train_external_init(testobj, chkflag):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    key = jax.random.PRNGKey(seed=1234)
    input_shape = (1, testobj.N, testobj.N, testobj.chn)

    # Via model initialization
    variables1 = model.init({"params": key}, np.ones(input_shape, model.dtype))
    flat_params1 = flatten_dict(variables1["params"])
    flat_bstats1 = flatten_dict(variables1["batch_stats"])
    params1 = [t[1] for t in sorted(flat_params1.items())]
    bstats1 = [t[1] for t in sorted(flat_bstats1.items())]

    # Via BasicFlaxTrainer object initialization
    train_conf = dict(testobj.train_conf)
    train_conf["checkpointing"] = chkflag
    trainer = sflax.BasicFlaxTrainer(
        train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
        variables0=variables1,
    )
    flat_params2 = flatten_dict(trainer.state.params)
    flat_bstats2 = flatten_dict(trainer.state.batch_stats)
    params2 = [t[1] for t in sorted(flat_params2.items())]
    bstats2 = [t[1] for t in sorted(flat_bstats2.items())]

    for i in range(len(params1)):
        np.testing.assert_allclose(params1[i], params2[i], rtol=1e-5)
    for i in range(len(bstats1)):
        np.testing.assert_allclose(bstats1[i], bstats2[i], rtol=1e-5)


@pytest.mark.parametrize("model_cls", [sflax.DnCNNNet, sflax.ResNet, sflax.ConvBNNet, sflax.UNet])
def test_class_train_train_loop(testobj, model_cls):
    depth = testobj.model_conf["depth"]
    model = model_cls(depth, testobj.chn, testobj.model_conf["num_filters"])
    if isinstance(model, sflax.DnCNNNet):
        depth = 3
        model = sflax.DnCNNNet(depth, testobj.chn, testobj.model_conf["num_filters"])

    # Create training object
    trainer = sflax.BasicFlaxTrainer(
        testobj.train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
    )

    try:
        modvar, _ = trainer.train()
    except Exception as e:
        print(e)
        assert 0
    else:
        assert "params" in modvar
        assert "batch_stats" in modvar


def test_class_train_train_post_loop(testobj):
    depth = testobj.model_conf["depth"]
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    train_conf = dict(testobj.train_conf)

    # Dum positive requirement over kernel parameters
    ktrav = construct_traversal("kernel")
    kpos = functools.partial(clip_positive, traversal=ktrav, minval=1e-5)
    train_conf["post_lst"] = [kpos]

    # Create training object
    trainer = sflax.BasicFlaxTrainer(
        train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
    )

    try:
        modvar, _ = trainer.train()
    except Exception as e:
        print(e)
        assert 0
    else:
        assert "params" in modvar
        assert "batch_stats" in modvar


def test_class_train_return_state(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    train_conf = dict(testobj.train_conf)
    train_conf["return_state"] = True
    trainer = sflax.BasicFlaxTrainer(
        train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
    )
    try:
        state, _ = trainer.train()
    except Exception as e:
        print(e)
        assert 0
    else:
        assert hasattr(state, "params")
        assert hasattr(state, "batch_stats")


def test_class_train_update_metrics(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    train_conf = dict(testobj.train_conf)
    train_conf["log"] = True
    train_conf["log_every_steps"] = 1
    trainer = sflax.BasicFlaxTrainer(
        train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
    )
    total_steps = (testobj.train_ds["label"].shape[0] // testobj.bsize) * train_conf["num_epochs"]
    try:
        state, stats_object = trainer.train()
    except Exception as e:
        print(e)
        assert 0
    else:
        hist = stats_object.history(transpose=True)
        assert len(hist.Train_Loss) == total_steps


def test_class_train_update_metrics_nolog(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    train_conf = dict(testobj.train_conf)
    train_conf["log"] = False
    train_conf["log_every_steps"] = 1
    trainer = sflax.BasicFlaxTrainer(
        train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
    )
    try:
        state, stats_object = trainer.train()
    except Exception as e:
        print(e)
        assert 0
    else:
        assert stats_object is None


def test_apply_fn(testobj):
    key = jax.random.PRNGKey(seed=531)
    key1, key2 = jax.random.split(key)

    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    input_shape = (1, testobj.N, testobj.N, testobj.chn)
    variables = model.init({"params": key1}, np.ones(input_shape, model.dtype))

    ds = IterateData(testobj.test_ds, testobj.bsize, train=False)

    try:
        batch = next(ds)
        output = _apply_fn(model, variables, batch)
    except Exception as e:
        print(e)
        assert 0
    else:
        assert output.shape[1:] == testobj.test_ds["label"].shape[1:]


def test_except_only_apply(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    with pytest.raises(Exception):
        out_ = sflax.only_apply(
            testobj.train_conf,
            model,
            testobj.test_ds,
        )


@pytest.mark.parametrize("model_cls", [sflax.DnCNNNet, sflax.ResNet, sflax.ConvBNNet, sflax.UNet])
def test_eval(testobj, model_cls):
    depth = testobj.model_conf["depth"]
    model = model_cls(depth, testobj.chn, testobj.model_conf["num_filters"])
    if isinstance(model, sflax.DnCNNNet):
        depth = 3
        model = sflax.DnCNNNet(depth, testobj.chn, testobj.model_conf["num_filters"])

    key = jax.random.PRNGKey(123)
    variables = model.init(key, testobj.train_ds["image"])

    # from train script
    out_, _ = sflax.only_apply(
        testobj.train_conf,
        model,
        testobj.test_ds,
        variables=variables,
    )
    # from scico FlaxMap util
    fmap = sflax.FlaxMap(model, variables)
    out_fmap = fmap(testobj.test_ds["image"])

    np.testing.assert_allclose(out_, out_fmap, atol=5e-6)
