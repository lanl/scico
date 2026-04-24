import numpy as np

import jax

from flax.linen import Conv
from flax.linen.module import Module, compact
from scico import flax as sflax
from scico.flax.train.clu_utils import (
    _default_table_value_formatter,
    get_parameter_overview,
)


def test_count_parameters():
    N = 128  # signal size
    chn = 1  # number of channels

    # Model configuration
    mconf = {
        "depth": 2,
        "num_filters": 16,
    }

    model = sflax.ResNet(mconf["depth"], chn, mconf["num_filters"])

    key = jax.random.key(seed=1234)
    input_shape = (1, N, N, chn)
    variables = model.init({"params": key}, np.ones(input_shape, model.dtype))

    filter_sz = model.kernel_size[0] * model.kernel_size[1]
    # filter parameters output layer
    sum_manual_params = filter_sz * mconf["num_filters"] * chn
    # bias and scale of batch normalization output layer
    sum_manual_params += chn * 2
    # mean and bar of batch normalization output layer
    sum_manual_bst = chn * 2
    chn_prev = 1
    for i in range(mconf["depth"] - 1):
        # filter parameters
        sum_manual_params += filter_sz * mconf["num_filters"] * chn_prev
        # bias and scale of batch normalization
        sum_manual_params += mconf["num_filters"] * 2
        # mean and bar of batch normalization
        sum_manual_bst += mconf["num_filters"] * 2
        chn_prev = mconf["num_filters"]

    total_nvar_params = sflax.count_parameters(variables["params"])
    total_nvar_bst = sflax.count_parameters(variables["batch_stats"])

    assert total_nvar_params == sum_manual_params
    assert total_nvar_bst == sum_manual_bst


def test_count_parameters_empty():
    assert sflax.count_parameters({}) == 0


# From https://github.com/google/CommonLoopUtils/blob/main/clu/parameter_overview_test.py
EMPTY_PARAMETER_OVERVIEW = """+------+-------+------+------+-----+
| Name | Shape | Size | Mean | Std |
+------+-------+------+------+-----+
+------+-------+------+------+-----+
Total weights: 0"""

FLAX_CONV2D_PARAMETER_OVERVIEW = """+-------------+--------------+------+
| Name        | Shape        | Size |
+-------------+--------------+------+
| conv/bias   | (2,)         | 2    |
| conv/kernel | (3, 3, 3, 2) | 54   |
+-------------+--------------+------+
Total weights: 56"""

FLAX_CONV2D_PARAMETER_OVERVIEW_WITH_STATS = """+-------------+--------------+------+------+-----+
| Name        | Shape        | Size | Mean | Std |
+-------------+--------------+------+------+-----+
| conv/bias   | (2,)         | 2    | 1.0  | 0.0 |
| conv/kernel | (3, 3, 3, 2) | 54   | 1.0  | 0.0 |
+-------------+--------------+------+------+-----+
Total weights: 56"""

FLAX_CONV2D_MAPPING_PARAMETER_OVERVIEW_WITH_STATS = """+--------------------+--------------+------+------+-----+
| Name               | Shape        | Size | Mean | Std |
+--------------------+--------------+------+------+-----+
| params/conv/bias   | (2,)         | 2    | 1.0  | 0.0 |
| params/conv/kernel | (3, 3, 3, 2) | 54   | 1.0  | 0.0 |
+--------------------+--------------+------+------+-----+
Total weights: 56"""


# From https://github.com/google/CommonLoopUtils/blob/main/clu/parameter_overview_test.py
def test_get_parameter_overview_empty():
    assert get_parameter_overview({}) == EMPTY_PARAMETER_OVERVIEW


class CNN(Module):
    @compact
    def __call__(self, x):
        return Conv(features=2, kernel_size=(3, 3), name="conv")(x)


# From https://github.com/google/CommonLoopUtils/blob/main/clu/parameter_overview_test.py
def test_get_parameter_overview():
    rng = jax.random.key(42)
    # Weights of a 2D convolution with 2 filters..
    variables = CNN().init(rng, np.zeros((2, 5, 5, 3)))
    variables = jax.tree_util.tree_map(jax.numpy.ones_like, variables)
    assert (
        get_parameter_overview(variables["params"], include_stats=False)
        == FLAX_CONV2D_PARAMETER_OVERVIEW
    )
    assert get_parameter_overview(variables["params"]) == FLAX_CONV2D_PARAMETER_OVERVIEW_WITH_STATS
    assert get_parameter_overview(variables) == FLAX_CONV2D_MAPPING_PARAMETER_OVERVIEW_WITH_STATS


# From https://github.com/google/CommonLoopUtils/blob/main/clu/parameter_overview_test.py
def test_printing_bool():
    assert _default_table_value_formatter(True) == "True"
    assert _default_table_value_formatter(False) == "False"
