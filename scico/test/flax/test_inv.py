import os
from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import lax

import pytest

from scico import flax as sflax
from scico import random
from scico.examples_flax import construct_blurring_operator
from scico.flax.train.train import clip_positive, construct_traversal, train_step_post
from scico.linop import Identity
from scico.linop.radon_astra import TomographicProjector

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def construct_projector(N, n_projection):
    import numpy as np

    angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
    return TomographicProjector(
        input_shape=(N, N),
        detector_spacing=1,
        det_count=N,
        angles=angles,
    )  # Radon transform operator


class TestSet:
    def setup_method(self, method):
        self.depth = 1  # depth (equivalent to number of blocks) of model
        self.chn = 1  # number of channels
        self.num_filters = 16  # number of filters per layer
        self.block_depth = 2  # number of layers in block
        self.N = 128  # image size

    def test_modlct_default(self):
        nproj = 60  # number of projections
        y, key = random.randn((10, nproj, self.N, self.chn), seed=1234)

        opCT = construct_projector(self.N, nproj)

        modln = sflax.MoDLNet(
            operator=opCT,
            depth=self.depth,
            channels=self.chn,
            num_filters=self.num_filters,
            block_depth=self.block_depth,
        )
        variables = modln.init(key, y)
        # Test for the construction / forward pass.
        mny = modln.apply(variables, y, train=False, mutable=False)
        assert y.dtype == mny.dtype

    def test_odpdn_default(self):
        y, key = random.randn((10, self.N, self.N, self.chn), seed=1234)

        opI = Identity(y.shape)

        odpdn = sflax.ODPNet(
            operator=opI,
            depth=self.depth,
            channels=self.chn,
            num_filters=self.num_filters,
            block_depth=self.block_depth,
        )

        variables = odpdn.init(key, y)
        # Test for the construction / forward pass.
        mny = odpdn.apply(variables, y, train=False, mutable=False)
        assert y.dtype == mny.dtype
        assert y.shape == mny.shape

    def test_odpdblr_default(self):
        y, key = random.randn((10, self.N, self.N, self.chn), seed=1234)

        blur_ksize = (9, 9)
        blur_sigma = 2.24
        output_size = (self.N, self.N)
        opBlur = construct_blurring_operator(output_size, self.chn, blur_ksize, blur_sigma)

        odpdb = sflax.ODPNet(
            operator=opBlur,
            depth=self.depth,
            channels=self.chn,
            num_filters=self.num_filters,
            block_depth=self.block_depth,
            odp_block=sflax.ODPProxDblrBlock,
        )

        variables = odpdb.init(key, y)
        # Test for the construction / forward pass.
        mny = odpdb.apply(variables, y, train=False, mutable=False)
        assert y.dtype == mny.dtype
        assert y.shape == mny.shape

    def test_odpct_default(self):
        nproj = 60  # number of projections
        y, key = random.randn((10, nproj, self.N, self.chn), seed=1234)

        opCT = construct_projector(self.N, nproj)

        odpct = sflax.ODPNet(
            operator=opCT,
            depth=self.depth,
            channels=self.chn,
            num_filters=self.num_filters,
            block_depth=self.block_depth,
            odp_block=sflax.ODPGrDescBlock,
        )

        variables = odpct.init(key, y)
        # Test for the construction / forward pass.
        oy = odpct.apply(variables, y, train=False, mutable=False)
        assert y.dtype == oy.dtype


class SetupTest02:
    def __init__(self):
        self.N = 32  # Signal size
        self.chn = 1  # Number of channels
        self.bsize = 16  # Batch size
        xt, key = random.randn((2 * self.bsize, self.N, self.N, self.chn), seed=4321)

        self.nproj = 60  # number of projections
        self.opCT = construct_projector(self.N, self.nproj)
        a_f = lambda v: jnp.atleast_3d(self.opCT(v.squeeze()))
        y = lax.map(a_f, xt)

        self.train_ds = {"image": y, "label": xt}
        self.test_ds = {"image": y, "label": xt}

        self.dconf: sflax.ConfigDict = {
            "seed": 0,
            "depth": 1,
            "num_filters": 16,
            "block_depth": 2,
            "opt_type": "ADAM",
            "batch_size": self.bsize,
            "num_epochs": 2,
            "base_learning_rate": 1e-3,
            "warmup_epochs": 0,
            "num_train_steps": -1,
            "steps_per_eval": -1,
            "steps_per_epoch": 1,
            "log_every_steps": 1000,
        }


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest02()


def test_train_modl(testobj):
    model = sflax.MoDLNet(
        operator=testobj.opCT,
        depth=testobj.dconf["depth"],
        channels=testobj.chn,
        num_filters=testobj.dconf["num_filters"],
        block_depth=testobj.dconf["block_depth"],
    )
    try:
        minval = 1.1e-2
        lmbdatrav = construct_traversal("lmbda")
        lmbdapos = partial(
            clip_positive,
            traversal=lmbdatrav,
            minval=minval,
        )
        train_step = partial(train_step_post, post_fn=lmbdapos)
        modvar = sflax.train_and_evaluate(
            testobj.dconf,
            "./",
            model,
            testobj.train_ds,
            testobj.test_ds,
            training_step_fn=train_step,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        lmbdaval = np.array([lmb for lmb in lmbdatrav.iterate(modvar["params"])])
        np.testing.assert_array_less(1e-2 * np.ones(lmbdaval.shape), lmbdaval)


def test_train_odpct(testobj):
    model = sflax.ODPNet(
        operator=testobj.opCT,
        depth=testobj.dconf["depth"],
        channels=testobj.chn,
        num_filters=testobj.dconf["num_filters"],
        block_depth=testobj.dconf["block_depth"],
        odp_block=sflax.ODPGrDescBlock,
    )

    try:
        minval = 1.1e-2
        alphatrav = construct_traversal("alpha")
        alphapos = partial(
            clip_positive,
            traversal=alphatrav,
            minval=minval,
        )
        train_step = partial(train_step_post, post_fn=alphapos)
        modvar = sflax.train_and_evaluate(
            testobj.dconf,
            "./",
            model,
            testobj.train_ds,
            testobj.test_ds,
            training_step_fn=train_step,
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        alphaval = np.array([alpha for alpha in alphatrav.iterate(modvar["params"])])
        np.testing.assert_array_less(1e-2 * np.ones(alphaval.shape), alphaval)
