import os
import tempfile

import numpy as np

import jax

import pytest
from test_trainer import SetupTest

from flax import jax_utils
from scico import flax as sflax
from scico.flax.train.checkpoints import have_tf
from scico.flax.train.learning_rate import create_cnst_lr_schedule
from scico.flax.train.state import create_basic_train_state

if have_tf:
    from scico.flax.train.checkpoints import checkpoint_restore, checkpoint_save


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


@pytest.mark.skipif(not have_tf, reason="tensorflow package not installed")
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

    # Emulating parallel training
    state = jax_utils.replicate(state)
    try:
        checkpoint_save(state, workdir)
        state_in = checkpoint_restore(model, workdir)

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
