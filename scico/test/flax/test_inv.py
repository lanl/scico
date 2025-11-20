import os
from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import lax

from scico import flax as sflax
from scico import random
from scico.flax.examples import PaddedCircularConvolve, build_blur_kernel
from scico.flax.train.traversals import clip_positive, clip_range, construct_traversal
from scico.linop import CircularConvolve, Identity
from scico.linop.xray import XRayTransform2D

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

    def test_odpdcnv_default(self):
        y, key = random.randn((10, self.N, self.N, self.chn), seed=1234)

        blur_shape = (9, 9)
        blur_sigma = 2.24
        kernel = build_blur_kernel(blur_shape, blur_sigma)

        ishape = (self.N, self.N)
        opBlur = CircularConvolve(h=kernel, input_shape=ishape)

        odpdb = sflax.ODPNet(
            operator=opBlur,
            depth=self.depth,
            channels=self.chn,
            num_filters=self.num_filters,
            block_depth=self.block_depth,
            odp_block=sflax.inverse.ODPProxDcnvBlock,
        )

        variables = odpdb.init(key, y)
        # Test for the construction / forward pass.
        mny = odpdb.apply(variables, y, train=False, mutable=False)
        assert y.dtype == mny.dtype
        assert y.shape == mny.shape

    def test_odpdcnv_padded(self):
        y, key = random.randn((10, self.N, self.N, self.chn), seed=1234)

        blur_shape = (9, 9)
        blur_sigma = 2.24
        opBlur = PaddedCircularConvolve(self.N, self.chn, blur_shape, blur_sigma)

        odpdb = sflax.ODPNet(
            operator=opBlur,
            depth=self.depth,
            channels=self.chn,
            num_filters=self.num_filters,
            block_depth=self.block_depth,
            odp_block=sflax.inverse.ODPProxDcnvBlock,
        )

        variables = odpdb.init(key, y)
        # Test for the construction / forward pass.
        mny = odpdb.apply(variables, y, train=False, mutable=False)
        assert y.dtype == mny.dtype
        assert y.shape == mny.shape

    def test_train_odpdcnv_default(self):
        xt, key = random.randn((10, self.N, self.N, self.chn), seed=4444)

        blur_shape = (7, 7)
        blur_sigma = 3.3
        kernel = build_blur_kernel(blur_shape, blur_sigma)

        ishape = (self.N, self.N)
        opBlur = CircularConvolve(h=kernel, input_shape=ishape)

        model = sflax.ODPNet(
            operator=opBlur,
            depth=self.depth,
            channels=self.chn,
            num_filters=self.num_filters,
            block_depth=self.block_depth,
            odp_block=sflax.inverse.ODPProxDcnvBlock,
        )

        train_conf: sflax.ConfigDict = {
            "seed": 0,
            "opt_type": "ADAM",
            "batch_size": 8,
            "num_epochs": 2,
            "base_learning_rate": 1e-3,
            "warmup_epochs": 0,
            "num_train_steps": -1,
            "steps_per_eval": -1,
            "steps_per_epoch": 1,
            "log_every_steps": 1000,
        }

        a_f = lambda v: jnp.atleast_3d(opBlur(v.reshape(opBlur.input_shape)))
        y = lax.map(a_f, xt)

        train_ds = {"image": y, "label": xt}
        test_ds = {"image": y, "label": xt}

        try:
            alphatrav = construct_traversal("alpha")
            alphapos = partial(clip_positive, traversal=alphatrav, minval=1e-3)
            train_conf["post_lst"] = [alphapos]
            trainer = sflax.BasicFlaxTrainer(
                train_conf,
                model,
                train_ds,
                test_ds,
            )
            modvar, _ = trainer.train()
        except Exception as e:
            print(e)
            assert 0
        else:
            alphaval = np.array([alpha for alpha in alphatrav.iterate(modvar["params"])])
            np.testing.assert_array_less(1e-2 * np.ones(alphaval.shape), alphaval)


class TestCT:
    def setup_method(self, method):
        self.N = 32  # signal size
        self.chn = 1  # number of channels
        self.bsize = 16  # batch size
        xt, key = random.randn((2 * self.bsize, self.N, self.N, self.chn), seed=4321)

        self.nproj = 60  # number of projections
        angles = np.linspace(0, np.pi, self.nproj, endpoint=False, dtype=np.float32)
        self.opCT = XRayTransform2D(
            input_shape=(self.N, self.N), det_count=self.N, angles=angles, dx=0.9999 / np.sqrt(2.0)
        )  # Radon transform operator
        a_f = lambda v: jnp.atleast_3d(self.opCT(v.squeeze()))
        y = lax.map(a_f, xt)

        self.train_ds = {"image": y, "label": xt}
        self.test_ds = {"image": y, "label": xt}

        # Model configuration
        self.model_conf = {
            "depth": 1,
            "num_filters": 16,
            "block_depth": 2,
        }

        # Training configuration
        self.train_conf: sflax.ConfigDict = {
            "seed": 0,
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
            depth=self.model_conf["depth"],
            channels=self.chn,
            num_filters=self.model_conf["num_filters"],
            block_depth=self.model_conf["block_depth"],
            odp_block=sflax.inverse.ODPGrDescBlock,
        )

        variables = model.init(key, y)
        # Test for the construction / forward pass.
        oy = model.apply(variables, y, train=False, mutable=False)
        assert y.dtype == oy.dtype

    def test_modlct_default(self):
        y, key = random.randn((10, self.nproj, self.N, self.chn), seed=1234)

        model = sflax.MoDLNet(
            operator=self.opCT,
            depth=self.model_conf["depth"],
            channels=self.chn,
            num_filters=self.model_conf["num_filters"],
            block_depth=self.model_conf["block_depth"],
        )

        variables = model.init(key, y)
        # Test for the construction / forward pass.
        mny = model.apply(variables, y, train=False, mutable=False)
        assert y.dtype == mny.dtype

    def test_train_modl(self):
        model = sflax.MoDLNet(
            operator=self.opCT,
            depth=self.model_conf["depth"],
            channels=self.chn,
            num_filters=self.model_conf["num_filters"],
            block_depth=self.model_conf["block_depth"],
        )
        try:
            minval = 1.1e-2
            lmbdatrav = construct_traversal("lmbda")
            lmbdapos = partial(
                clip_positive,
                traversal=lmbdatrav,
                minval=minval,
            )
            train_conf = dict(self.train_conf)
            train_conf["post_lst"] = [lmbdapos]
            trainer = sflax.BasicFlaxTrainer(
                train_conf,
                model,
                self.train_ds,
                self.test_ds,
            )
            modvar, _ = trainer.train()
        except Exception as e:
            print(e)
            assert 0
        else:
            lmbdaval = np.array([lmb for lmb in lmbdatrav.iterate(modvar["params"])])
            np.testing.assert_array_less(1e-2 * np.ones(lmbdaval.shape), lmbdaval)

    def test_train_odpct(self):
        model = sflax.ODPNet(
            operator=self.opCT,
            depth=self.model_conf["depth"],
            channels=self.chn,
            num_filters=self.model_conf["num_filters"],
            block_depth=self.model_conf["block_depth"],
            odp_block=sflax.inverse.ODPGrDescBlock,
        )

        try:
            minval = 1.1e-2
            maxval = 1e2
            alphatrav = construct_traversal("alpha")
            alpharange = partial(clip_range, traversal=alphatrav, minval=minval, maxval=maxval)
            train_conf = dict(self.train_conf)
            train_conf["post_lst"] = [alpharange]
            trainer = sflax.BasicFlaxTrainer(
                train_conf,
                model,
                self.train_ds,
                self.test_ds,
            )
            modvar, _ = trainer.train()
        except Exception as e:
            print(e)
            assert 0
        else:
            alphaval = np.array([alpha for alpha in alphatrav.iterate(modvar["params"])])
            np.testing.assert_array_less(1e-2 * np.ones(alphaval.shape), alphaval)
