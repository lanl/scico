import pytest

from scico import flax as sflax
from scico import random
from scico.flax.inverse import MoDLNet
from scico.linop.radon_astra import ParallelBeamProjector


def construct_projector(N, n_projection):
    import numpy as np

    angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
    return ParallelBeamProjector(
        input_shape=(N, N),
        detector_spacing=1,
        det_count=N,
        angles=angles,
    )  # Radon transform operator


class TestSet:
    def test_modlct_default(self):
        depth = 1  # depth of model
        chn = 1  # number of channels
        num_filters = 16  # number of filters per layer
        block_depth = 2  # number of layers in block
        N = 128  # image size
        nproj = 60  # number of projections
        y, key = random.randn((10, nproj, N, chn), seed=1234)

        opCT = construct_projector(N, nproj)

        modln = MoDLNet(
            depth=depth,
            channels=chn,
            num_filters=num_filters,
            block_depth=block_depth,
            operator=opCT,
        )
        variables = modln.init(key, y)
        # Test for the construction / forward pass.
        mny = modln.apply(variables, y, train=False, mutable=False)
        assert y.dtype == mny.dtype
