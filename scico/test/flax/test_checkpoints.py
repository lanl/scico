import os
import tempfile

import numpy as np

import jax

import pytest
from test_trainer import SetupTest

from flax.traverse_util import flatten_dict
from scico import flax as sflax
from scico.flax.train.checkpoints import checkpoint_restore, checkpoint_save, have_orbax
from scico.flax.train.learning_rate import create_cnst_lr_schedule
from scico.flax.train.state import create_basic_train_state


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


@pytest.mark.skipif(not have_orbax, reason="orbax.checkpoint package not installed")
def test_checkpoint(testobj):
    depth = 3
    model = sflax.DnCNNNet(depth, testobj.chn, testobj.model_conf["num_filters"])

    key = jax.random.key(123)
    variables = model.init(key, testobj.train_ds["image"])

    temp_dir = tempfile.TemporaryDirectory()
    workdir = os.path.join(temp_dir.name, "temp_ckp")

    # State initialization
    learning_rate = create_cnst_lr_schedule(testobj.train_conf)
    state = create_basic_train_state(
        key, testobj.train_conf, model, (testobj.N, testobj.N), learning_rate
    )
    flat_params1 = flatten_dict(state.params)
    flat_bstats1 = flatten_dict(state.batch_stats)
    params1 = [t[1] for t in sorted(flat_params1.items())]
    bstats1 = [t[1] for t in sorted(flat_bstats1.items())]

    try:
        checkpoint_save(state, testobj.train_conf, workdir)
        state_in = checkpoint_restore(state, workdir)

    except Exception as e:
        print(e)
        assert 0
    else:

        flat_params2 = flatten_dict(state_in.params)
        flat_bstats2 = flatten_dict(state_in.batch_stats)
        params2 = [t[1] for t in sorted(flat_params2.items())]
        bstats2 = [t[1] for t in sorted(flat_bstats2.items())]

        for i in range(len(params1)):
            np.testing.assert_allclose(params1[i], params2[i], rtol=1e-5)
        for i in range(len(bstats1)):
            np.testing.assert_allclose(bstats1[i], bstats2[i], rtol=1e-5)


@pytest.mark.skipif(not have_orbax, reason="orbax.checkpoint package not installed")
@pytest.mark.parametrize("model_cls", [sflax.DnCNNNet, sflax.ResNet])
def test_checkpointing_from_trainer(testobj, model_cls):
    depth = 3
    model = model_cls(depth, testobj.chn, testobj.model_conf["num_filters"])

    temp_dir = tempfile.TemporaryDirectory()
    workdir = os.path.join(temp_dir.name, "temp_ckp")

    train_conf = dict(testobj.train_conf)
    train_conf["checkpointing"] = True
    train_conf["workdir"] = workdir
    train_conf["return_state"] = True

    # Create training object
    trainer = sflax.BasicFlaxTrainer(
        train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
    )
    try:
        state_out, _ = trainer.train()
    except Exception as e:
        print(e)
        assert 0
    else:
        # Model parameters from training
        flat_params1 = flatten_dict(state_out.params)
        params1 = [t[1] for t in sorted(flat_params1.items())]

        # Model parameteres from checkpoint
        state_in = checkpoint_restore(state_out, workdir)
        flat_params2 = flatten_dict(state_in.params)
        params2 = [t[1] for t in sorted(flat_params2.items())]

        for i in range(len(params1)):
            np.testing.assert_allclose(params1[i], params2[i], rtol=1e-5)

        if hasattr(state_out, "batch_stats"):
            # Batch stats from training
            flat_bstats1 = flatten_dict(state_out.batch_stats)
            bstats1 = [t[1] for t in sorted(flat_bstats1.items())]
            # Batch stats from checkpoint
            flat_bstats2 = flatten_dict(state_in.batch_stats)
            bstats2 = [t[1] for t in sorted(flat_bstats2.items())]
            for i in range(len(bstats1)):
                np.testing.assert_allclose(bstats1[i], bstats2[i], rtol=1e-5)


@pytest.mark.skipif(not have_orbax, reason="orbax.checkpoint package not installed")
def test_checkpoint_exception(testobj):
    depth = 3
    model = sflax.DnCNNNet(depth, testobj.chn, testobj.model_conf["num_filters"])

    key = jax.random.key(123)
    variables = model.init(key, testobj.train_ds["image"])

    temp_dir = tempfile.TemporaryDirectory()
    workdir = os.path.join(temp_dir.name, "temp_ckp")

    # State initialization
    learning_rate = create_cnst_lr_schedule(testobj.train_conf)
    state = create_basic_train_state(
        key, testobj.train_conf, model, (testobj.N, testobj.N), learning_rate
    )

    with pytest.raises(FileNotFoundError):
        state_in = checkpoint_restore(state, workdir)
