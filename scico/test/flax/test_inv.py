import pytest

import jax.numpy as jnp

from scico import flax as sflax
from scico import random
from scico.flax.inverse import MoDLNet, ODPProxDnBlock, ODPProxDblrBlock, ODPGrDescBlock, ODPNet
from scico.linop.radon_astra import ParallelBeamProjector
from scico.linop import Identity, CircularConvolve


def construct_projector(N, n_projection):
    import numpy as np

    angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
    return ParallelBeamProjector(
        input_shape=(N, N),
        detector_spacing=1,
        det_count=N,
        angles=angles,
    )  # Radon transform operator


def create_gaussian_kernel(sigma, kernel_size, n_channels, dtype=jnp.float32):

    kernel = 1
    meshgrids = jnp.meshgrid(*[jnp.arange(size, dtype=dtype) for size in kernel_size])
    for size, mgrid in zip(kernel_size, meshgrids):
        mean = (size - 1) / 2
        kernel *= jnp.exp(-(((mgrid - mean) / sigma) ** 2) / 2)

    # Make sure norm of values in gaussian kernel equals 1.
    kernel = kernel / jnp.sum(kernel)

    # Reshape to depthwise convolutional weight (HWC)
    kernel = jnp.reshape(kernel, kernel.shape + (1,))
    # Repeat to match channels
    kernel = jnp.repeat(kernel, n_channels, axis=2)
    return kernel


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

        modln = MoDLNet(
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

        odpdn = ODPNet(
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

        h = create_gaussian_kernel(2.24, (9, 9), self.chn)

        opBlur = CircularConvolve(h, y.shape, ndims=3)

        odpdb = ODPNet(
            operator=opBlur,
            depth=self.depth,
            channels=self.chn,
            num_filters=self.num_filters,
            block_depth=self.block_depth,
            odp_block=ODPProxDblrBlock,
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

        odpct = ODPNet(
            operator=opCT,
            depth=self.depth,
            channels=self.chn,
            num_filters=self.num_filters,
            block_depth=self.block_depth,
            odp_block=ODPGrDescBlock,
        )

        variables = odpct.init(key, y)
        # Test for the construction / forward pass.
        oy = odpct.apply(variables, y, train=False, mutable=False)
        assert y.dtype == oy.dtype
