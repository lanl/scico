import os
import tempfile

import numpy as np

import jax

import pytest
from test_trainer import SetupTest

from flax.traverse_util import flatten_dict
from scico import flax as sflax
from scico.flax.train.checkpoints import checkpoint_restore, checkpoint_save
from scico.flax.train.learning_rate import create_cnst_lr_schedule
from scico.flax.train.state import create_basic_train_state


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_checkpoint(testobj):
    depth = 3
    model = sflax.DnCNNNet(depth, testobj.chn, testobj.model_conf["num_filters"])

    key = jax.random.PRNGKey(123)
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

    # Bundle config and model parameters together
    ckpt = {"state": state, "config": testobj.train_conf}

    try:
        checkpoint_save(ckpt, workdir)
        state_in = checkpoint_restore(workdir)

    except Exception as e:
        print(e)
        assert 0
    else:

        flat_params2 = flatten_dict(state_in["params"])
        flat_bstats2 = flatten_dict(state_in["batch_stats"])
        params2 = [t[1] for t in sorted(flat_params2.items())]
        bstats2 = [t[1] for t in sorted(flat_bstats2.items())]

        for i in range(len(params1)):
            np.testing.assert_allclose(params1[i], params2[i], rtol=1e-5)
        for i in range(len(bstats1)):
            np.testing.assert_allclose(bstats1[i], bstats2[i], rtol=1e-5)


@pytest.mark.parametrize("model_cls", [sflax.DnCNNNet, sflax.ResNet])
def test_checkpointing_from_trainer(testobj, model_cls):
    depth = 3
    model = model_cls(depth, testobj.chn, testobj.model_conf["num_filters"])

    temp_dir = tempfile.TemporaryDirectory()
    workdir = os.path.join(temp_dir.name, "temp_ckp")

    train_conf = dict(testobj.train_conf)
    train_conf["checkpointing"] = True
    train_conf["workdir"] = workdir

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
        # Model parameters from training
        flat_params1 = flatten_dict(modvar["params"])
        params1 = [t[1] for t in sorted(flat_params1.items())]

        # Model parameteres from checkpoint
        state_in = checkpoint_restore(workdir)
        assert state_in is not None
        flat_params2 = flatten_dict(state_in["params"])
        params2 = [t[1] for t in sorted(flat_params2.items())]

        for i in range(len(params1)):
            np.testing.assert_allclose(params1[i], params2[i], rtol=1e-5)

        if "batch_stats" in modvar:
            # Batch stats from training
            flat_bstats1 = flatten_dict(modvar["batch_stats"])
            bstats1 = [t[1] for t in sorted(flat_bstats1.items())]
            # Batch stats from checkpoint
            flat_bstats2 = flatten_dict(state_in["batch_stats"])
            bstats2 = [t[1] for t in sorted(flat_bstats2.items())]
            for i in range(len(bstats1)):
                np.testing.assert_allclose(bstats1[i], bstats2[i], rtol=1e-5)


def test_checkpoint_exception():
    temp_dir = tempfile.TemporaryDirectory()
    workdir = os.path.join(temp_dir.name, "temp_ckp")

    with pytest.raises(FileNotFoundError):
        state_in = checkpoint_restore(workdir)
