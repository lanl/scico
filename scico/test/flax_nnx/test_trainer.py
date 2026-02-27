import numpy as np

import jax

import optax
import pytest

from flax import nnx
from scico import random

# from scico.flax_nnx.train.clu_utils import flatten_dict
from scico.flax_nnx._models import ConvBNNet, DnCNNNet, ResNet, UNet
from scico.flax_nnx.train.steps import eval_step, train_step
from scico.flax_nnx.train.trainer import BasicFlaxNNXTrainer
from scico.flax_nnx.train.typed_dict import ConfigDict

# from scico.flax.train.traversals import clip_positive, clip_range, construct_traversal


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
        self.train_conf: ConfigDict = {
            "seed": 0,
            "opt_type": "ADAM",
            "momentum": 0.9,
            "batch_size": self.bsize,
            "num_epochs": 1,
            "base_learning_rate": 1e-3,
            "lr_decay_rate": 0.95,
            "warmup_epochs": 0,
            "log_every_epochs": 10,
        }


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


@pytest.mark.parametrize("opt_type", ["SGD", "ADAM", "ADAMW"])
def test_optimizers(testobj, opt_type):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )

    train_conf = dict(testobj.train_conf)
    train_conf["opt_type"] = opt_type
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


def test_optimizers_exception(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )

    train_conf = dict(testobj.train_conf)
    train_conf["opt_type"] = ""
    with pytest.raises(NotImplementedError):
        BasicFlaxNNXTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )


def test_class_train_default_init(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )
    try:
        trainer = BasicFlaxNNXTrainer(
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
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )
    train_conf = dict(testobj.train_conf)
    train_conf.pop("seed", None)
    try:
        trainer = BasicFlaxNNXTrainer(
            testobj.train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0


def test_class_train_nolog(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )

    train_conf = dict(testobj.train_conf)
    train_conf["log"] = False
    try:
        trainer = BasicFlaxNNXTrainer(
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


@pytest.mark.skipif(jax.device_count() == 1, reason="single device present")
def test_except_class_train_batch_size(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )
    train_conf = dict(testobj.train_conf)
    train_conf["batch_size"] = jax.device_count() + 1
    with pytest.raises(ValueError):
        trainer = BasicFlaxNNXTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )


def test_class_train_set_epochs(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )
    train_conf = dict(testobj.train_conf)
    train_conf.pop("num_epochs", None)

    try:
        trainer = BasicFlaxNNXTrainer(
            train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert trainer.train_epochs > 0
        assert trainer.checkpoint_every_epochs > 0
        assert trainer.log_every_epochs > 0


def test_class_train_set_reporting(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )
    train_conf = dict(testobj.train_conf)
    train_conf["log"] = True
    train_conf["workdir"] = "./out/"
    train_conf["checkpointing"] = False
    try:
        trainer = BasicFlaxNNXTrainer(
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


def test_class_train_set_functions(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )

    def huber_loss(output, labels):
        return jax.numpy.mean(optax.huber_loss(output, labels))

    train_conf = dict(testobj.train_conf)
    train_conf["criterion"] = huber_loss
    train_conf["train_step_fn"] = train_step
    train_conf["eval_step_fn"] = eval_step
    try:
        trainer = BasicFlaxNNXTrainer(
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
        assert trainer.train_step_fn == train_conf["train_step_fn"]
        assert trainer.eval_step_fn == train_conf["eval_step_fn"]


def test_class_train_set_iterators(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )
    try:
        trainer = BasicFlaxNNXTrainer(
            testobj.train_conf,
            model,
            testobj.train_ds,
            testobj.test_ds,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert hasattr(trainer, "dt_iterator_fn")


@pytest.mark.parametrize("model_cls", [DnCNNNet, ResNet, ConvBNNet, UNet])
def test_class_train_train_loop(testobj, model_cls):
    depth = testobj.model_conf["depth"]
    model = model_cls(depth, testobj.chn, testobj.model_conf["num_filters"], rngs=nnx.Rngs(0))
    if isinstance(model, DnCNNNet):
        depth = 3
        model = DnCNNNet(depth, testobj.chn, testobj.model_conf["num_filters"], rngs=nnx.Rngs(0))

    train_conf = dict(testobj.train_conf)
    train_conf["log"] = True
    train_conf["log_every_epochs"] = 1

    # Create training object
    trainer = BasicFlaxNNXTrainer(
        train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
    )

    try:
        stats_object = trainer.train()
    except Exception as e:
        print(e)
        assert 0
    else:
        hist = stats_object.history(transpose=True)
        assert len(hist.Train_Loss) == testobj.train_conf["num_epochs"]


def test_class_train_update_metrics(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )

    train_conf = dict(testobj.train_conf)
    train_conf["log"] = True
    train_conf["log_every_epochs"] = 1
    num_epochs = 2
    train_conf["num_epochs"] = num_epochs
    trainer = BasicFlaxNNXTrainer(
        train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
    )
    try:
        stats_object = trainer.train()
    except Exception as e:
        print(e)
        assert 0
    else:
        hist = stats_object.history(transpose=True)
        assert len(hist.Train_Loss) == num_epochs


def test_class_train_update_metrics_nolog(testobj):
    model = ResNet(
        testobj.model_conf["depth"],
        testobj.chn,
        testobj.model_conf["num_filters"],
        rngs=nnx.Rngs(0),
    )

    train_conf = dict(testobj.train_conf)
    train_conf["log"] = False
    train_conf["log_every_epochs"] = 2  # Less than num_epochs (is 1) --> No logging
    trainer = BasicFlaxNNXTrainer(
        train_conf,
        model,
        testobj.train_ds,
        testobj.test_ds,
    )
    try:
        stats_object = trainer.train()
    except Exception as e:
        print(e)
        assert 0
    else:
        assert stats_object is None
