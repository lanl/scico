import numpy as np

import jax

import pytest
from test_trainer import SetupTest

from scico import flax as sflax
from scico.flax.train.traversals import construct_traversal


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


@pytest.mark.parametrize("pname", ["kernel", "bias", "scale"])
def test_construct_traversal(testobj, pname):
    model = sflax.ConvBNNet(
        testobj.model_conf["depth"], testobj.chn, testobj.model_conf["num_filters"]
    )

    ndim = 1
    if pname == "kernel":
        ndim = 4

    key = jax.random.key(seed=432)
    input_shape = (1, testobj.N, testobj.N, testobj.chn)
    variables = model.init({"params": key}, np.ones(input_shape, model.dtype))

    ptrav = construct_traversal(pname)
    for pm in ptrav.iterate(variables["params"]):
        assert len(pm.shape) == ndim
