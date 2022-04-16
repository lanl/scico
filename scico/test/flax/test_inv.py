import os
from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import lax

import pytest

from scico import flax as sflax
from scico import random
from scico.flax.examples import construct_blurring_operator, have_astra
from scico.flax.train.train import clip_positive, construct_traversal, train_step_post
from scico.linop import Identity

if have_astra:
    from scico.linop.radon_astra import TomographicProjector


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


class TestSet:
    def setup_method(self, method):
        self.depth = 1  # depth (equivalent to number of blocks) of model
        self.chn = 1  # number of channels
        self.num_filters = 16  # number of filters per layer
        self.block_depth = 2  # number of layers in block
        self.N = 128  # image size

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


@pytest.mark.skipif(not have_astra, reason="astra package not installed")
class TestCT:
    def setup(self):
        self.N = 32  # Signal size
        self.chn = 1  # Number of channels
        self.bsize = 16  # Batch size
        xt, key = random.randn((2 * self.bsize, self.N, self.N, self.chn), seed=4321)

        self.nproj = 60  # number of projections
        angles = np.linspace(0, np.pi, self.nproj)  # evenly spaced projection angles
        self.opCT = TomographicProjector(
            input_shape=(self.N, self.N),
            detector_spacing=1,
            det_count=self.N,
            angles=angles,
        )  # Radon transform operator
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

    def test_odpct_default(self):
        y, key = random.randn((10, self.nproj, self.N, self.chn), seed=1234)

        model = sflax.ODPNet(
            operator=self.opCT,
            depth=self.dconf["depth"],
            channels=self.chn,
            num_filters=self.dconf["num_filters"],
            block_depth=self.dconf["block_depth"],
            odp_block=sflax.ODPGrDescBlock,
        )

        variables = model.init(key, y)
        # Test for the construction / forward pass.
        oy = model.apply(variables, y, train=False, mutable=False)
        assert y.dtype == oy.dtype

    def test_modlct_default(self):
        y, key = random.randn((10, self.nproj, self.N, self.chn), seed=1234)

        model = sflax.MoDLNet(
            operator=self.opCT,
            depth=self.dconf["depth"],
            channels=self.chn,
            num_filters=self.dconf["num_filters"],
            block_depth=self.dconf["block_depth"],
        )

        variables = model.init(key, y)
        # Test for the construction / forward pass.
        mny = model.apply(variables, y, train=False, mutable=False)
        assert y.dtype == mny.dtype

    def test_train_modl(self):
        model = sflax.MoDLNet(
            operator=self.opCT,
            depth=self.dconf["depth"],
            channels=self.chn,
            num_filters=self.dconf["num_filters"],
            block_depth=self.dconf["block_depth"],
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
                self.dconf,
                "./",
                model,
                self.train_ds,
                self.test_ds,
                training_step_fn=train_step,
            )
        except Exception as e:
            print(e)
            assert 0
        else:
            lmbdaval = np.array([lmb for lmb in lmbdatrav.iterate(modvar["params"])])
            np.testing.assert_array_less(1e-2 * np.ones(lmbdaval.shape), lmbdaval)

    def test_train_odpct(self):
        model = sflax.ODPNet(
            operator=self.opCT,
            depth=self.dconf["depth"],
            channels=self.chn,
            num_filters=self.dconf["num_filters"],
            block_depth=self.dconf["block_depth"],
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
                self.dconf,
                "./",
                model,
                self.train_ds,
                self.test_ds,
                training_step_fn=train_step,
            )
        except Exception as e:
            print(e)
            assert 0
        else:
            alphaval = np.array([alpha for alpha in alphatrav.iterate(modvar["params"])])
            np.testing.assert_array_less(1e-2 * np.ones(alphaval.shape), alphaval)
