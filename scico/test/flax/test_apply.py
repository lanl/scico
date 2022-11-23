import tempfile

import numpy as np

import jax

import pytest
from test_trainer import SetupTest

from scico import flax as sflax
from scico.flax.train.apply import apply_fn
from scico.flax.train.input_pipeline import IterateData
from scico.flax.train.trainer import have_tf

if have_tf:
    from .checkpoints import save_checkpoint


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


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


@pytest.mark.skipif(not have_tf, reason="tensorflow package not installed")
def test_checkpointing(testobj):
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
    save_checkpoint(state, workdir)

    train_conf = dict(testobj.train_conf)
    train_conf["checkpointing"] = True
    train_conf["workdir"] = workdir

    try:
        out, var_in = sflax.only_apply(
            train_conf,
            model,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        flat_bstats0 = flatten_dict(state.batch_stats)
        bstats0 = [t[1] for t in sorted(flat_bstats0.items())]

        flat_bstats1 = flatten_dict(var_in["batch_stats"])
        bstats1 = [t[1] for t in sorted(flat_bstats1.items())]

        for i in range(len(bstats0)):
            np.testing.assert_allclose(bstats0[i], bstats1[i], rtol=1e-5)
