import functools

import jax

import pytest
from test_trainer import SetupTest

from flax import jax_utils
from scico import flax as sflax
from scico.flax.train.diagnostics import compute_metrics
from scico.flax.train.learning_rate import create_cnst_lr_schedule
from scico.flax.train.losses import mse_loss
from scico.flax.train.state import create_basic_train_state
from scico.flax.train.steps import eval_step, train_step, train_step_post
from scico.flax.train.traversals import clip_range, construct_traversal


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_basic_train_step(testobj):
    key = jax.random.key(seed=531)
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
            train_step,
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


def test_post_train_step(testobj):
    key = jax.random.key(seed=531)
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
            train_step_post,
            learning_rate_fn=learning_rate,
            criterion=criterion,
            train_step_fn=train_step,
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
    key = jax.random.key(seed=531)
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
        functools.partial(eval_step, criterion=criterion, metrics_fn=compute_metrics),
        axis_name="batch",
    )

    try:
        batch = next(eval_dt_iter)
        p_eval_step(state, batch)
    except Exception as e:
        print(e)
        assert 0
