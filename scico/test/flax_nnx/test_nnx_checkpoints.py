import json
import os
import tempfile

import numpy as np

import jax

import pytest
from test_nnx_trainer import SetupTest

from flax import nnx
from scico.flax_nnx._models import DnCNNNet, ResNet
from scico.flax_nnx.train.checkpoints import (
    checkpoint_restore,
    checkpoint_restore_with_config,
    checkpoint_save,
    have_orbax,
)
from scico.flax_nnx.train.optax_utils import (
    build_optax_optimizer,
    create_exp_lr_schedule,
)
from scico.flax_nnx.train.trainer import BasicFlaxNNXTrainer


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


@pytest.mark.skipif(not have_orbax, reason="orbax.checkpoint package not installed")
def test_checkpoint(testobj):
    depth = 3
    model = DnCNNNet(
        depth,
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(np.random.randint(10)),
    )
    model2 = DnCNNNet(
        depth,
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(4444),
    )

    temp_dir = tempfile.TemporaryDirectory()
    workdir = os.path.join(temp_dir.name, "temp_ckp")

    train_conf = dict(testobj.train_conf)

    # Optimizer creation
    train_conf["lr_schedule"] = create_exp_lr_schedule
    train_conf["lr_decay_rate"] = 0.95
    tx = build_optax_optimizer(train_conf)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # State initialization
    state = nnx.state((model, optimizer))
    epoch = 0

    try:
        checkpoint_save(state, epoch, testobj.train_conf, workdir)
        state_in, step = checkpoint_restore(state, workdir)

    except Exception as e:
        print(e)
        assert 0
    else:
        assert step == epoch

        # Restore model2 from checkpoint
        nnx.update((model2, optimizer), state_in)

        # Verify forward pass and optimizer step count match
        test_input = jax.random.normal(jax.random.key(42), testobj.train_ds["image"][:4].shape)
        np.testing.assert_allclose(model(test_input), model2(test_input))
        assert optimizer.step.get_value() == step


@pytest.mark.skipif(not have_orbax, reason="orbax.checkpoint package not installed")
@pytest.mark.parametrize("model_cls", [DnCNNNet, ResNet])
def test_checkpointing_from_trainer(testobj, model_cls):
    depth = 3
    model = model_cls(
        depth, testobj.chn, testobj.model_conf["num_filters"], rngs=nnx.Rngs(np.random.randint(10))
    )
    model2 = model_cls(depth, testobj.chn, testobj.model_conf["num_filters"], rngs=nnx.Rngs(4444))

    temp_dir = tempfile.TemporaryDirectory()
    workdir = os.path.join(temp_dir.name, "temp_ckp")

    train_conf = dict(testobj.train_conf)
    train_conf["checkpointing"] = True
    train_conf["workdir"] = workdir

    try:
        trainer = BasicFlaxNNXTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
        trainer.train()

    except Exception as e:
        print(e)
        assert 0
    else:
        # Restore model2 from checkpoint
        # Dum optimizer creation
        tx = build_optax_optimizer(train_conf)
        optimizer2 = nnx.Optimizer(model2, tx, wrt=nnx.Param)
        # State initialization
        state2 = nnx.state((model2, optimizer2))
        state_in, step = checkpoint_restore(state2, workdir)
        nnx.update((model2, optimizer2), state_in)
        model2.eval()
        # Verify forward pass and optimizer step count match
        model.eval()
        model2.eval()
        test_input = jax.random.normal(jax.random.key(42), testobj.train_ds["image"][:4].shape)
        np.testing.assert_allclose(model(test_input), model2(test_input))
        assert train_conf["num_epochs"] == step


@pytest.mark.skipif(not have_orbax, reason="orbax.checkpoint package not installed")
def test_checkpoint_exception(testobj):
    depth = 3
    model = DnCNNNet(
        depth, testobj.chn, testobj.model_conf["num_filters"], rngs=nnx.Rngs(np.random.randint(10))
    )

    temp_dir = tempfile.TemporaryDirectory()
    workdir = os.path.join(temp_dir.name, "temp_ckp")

    train_conf = dict(testobj.train_conf)

    # Optimizer creation
    train_conf["lr_schedule"] = create_exp_lr_schedule
    train_conf["lr_decay_rate"] = 0.95
    tx = build_optax_optimizer(train_conf)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # State initialization
    state = nnx.state((model, optimizer))

    with pytest.raises(FileNotFoundError):
        checkpoint_restore(state, workdir)


@pytest.mark.skipif(not have_orbax, reason="orbax.checkpoint package not installed")
def test_checkpoint_with_config(testobj):
    depth = 3
    model = DnCNNNet(
        depth,
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(np.random.randint(10)),
    )
    model2 = DnCNNNet(
        depth,
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(4444),
    )

    temp_dir = tempfile.TemporaryDirectory()
    workdir = os.path.join(temp_dir.name, "temp_ckp")

    train_conf = dict(testobj.train_conf)

    # Optimizer creation
    tx = build_optax_optimizer(train_conf)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # State initialization
    state = nnx.state((model, optimizer))
    epoch = 0

    try:
        checkpoint_save(state, epoch, testobj.train_conf, workdir)
        state_in, config_in, step = checkpoint_restore_with_config(state, workdir)

    except Exception as e:
        print(e)
        assert 0
    else:
        assert step == epoch

        # Restore model2 from checkpoint
        nnx.update((model2, optimizer), state_in)

        # Verify forward pass and optimizer step count match
        test_input = jax.random.normal(jax.random.key(42), testobj.train_ds["image"][:4].shape)
        np.testing.assert_allclose(model(test_input), model2(test_input))
        assert optimizer.step.get_value() == step

        config1 = filter_unserializable_metadata(testobj.train_conf)
        config2 = filter_unserializable_metadata(config_in)

        assert config1 == config2

        # chex.assert_trees_all_close(config1, config2, rtol=1e-6, atol=1e-6)
        # chex.assert_trees_all_equal_comparator(safe_compare, safe_compare, config1, config2)

        # chex.assert_trees_all_equal_comparator(safe_compare, safe_compare, testobj.train_conf, config_in)


def safe_compare(a, b):
    if isinstance(a, str) or isinstance(b, str):
        assert a == b, f"String mismatch: {a!r} != {b!r}"
    else:
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def filter_unserializable_metadata(obj):
    """Recursively remove elements that are not JSON-serializable.

    Args:
        obj: Object to filter.

    Returns:
        A copy of the structure with non-serializable elements removed.
    """
    if isinstance(obj, dict):
        cleaned_dict: dict[str, Any] = {}
        for k, v in obj.items():
            if _is_serializable(v):
                cleaned_dict[str(k)] = filter_unserializable_metadata(v)
        return cleaned_dict

    elif isinstance(obj, (list, tuple)):
        return [filter_unserializable_metadata(v) for v in obj if _is_serializable(v)]

    return obj if _is_serializable(obj) else None


def _is_serializable(obj) -> bool:
    """Check if an object or container has serializable leaf values."""
    if isinstance(obj, dict):
        return True
    if isinstance(obj, (list, tuple)):
        return True
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
