import os
import tempfile

import numpy as np

import jax

import pytest
from test_trainer import SetupTest

from flax.traverse_util import flatten_dict
from scico import flax as sflax
from scico.flax.train.apply import apply_fn
from scico.flax.train.checkpoints import checkpoint_save, have_orbax
from scico.flax.train.input_pipeline import IterateData
from scico.flax.train.learning_rate import create_cnst_lr_schedule
from scico.flax.train.state import create_basic_train_state


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_apply_fn(testobj):
    key = jax.random.key(seed=531)
    key1, key2 = jax.random.split(key)

    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    input_shape = (1, testobj.N, testobj.N, testobj.chn)
    variables = model.init({"params": key1}, np.ones(input_shape, model.dtype))

    ds = IterateData(testobj.test_ds, testobj.bsize, train=False)

    try:
        batch = next(ds)
        output = apply_fn(model, variables, batch)
    except Exception as e:
        print(e)
        assert 0
    else:
        assert output.shape[1:] == testobj.test_ds["label"].shape[1:]


def test_except_only_apply(testobj):
    model = sflax.ResNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    with pytest.raises(RuntimeError):
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

    key = jax.random.key(123)
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


@pytest.mark.skipif(not have_orbax, reason="orbax.checkpoint package not installed")
def test_apply_from_checkpoint(testobj):
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

    train_conf = dict(testobj.train_conf)
    train_conf["checkpointing"] = True
    train_conf["workdir"] = workdir
    checkpoint_save(state, train_conf, workdir)

    try:
        output, variables = sflax.only_apply(
            train_conf,
            model,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        flat_params2 = flatten_dict(variables["params"])
        flat_bstats2 = flatten_dict(variables["batch_stats"])
        params2 = [t[1] for t in sorted(flat_params2.items())]
        bstats2 = [t[1] for t in sorted(flat_bstats2.items())]

        for i in range(len(params1)):
            np.testing.assert_allclose(params1[i], params2[i], rtol=1e-5)
        for i in range(len(bstats1)):
            np.testing.assert_allclose(bstats1[i], bstats2[i], rtol=1e-5)
