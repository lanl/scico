# -*- coding: utf-8 -*-
# Copyright (C) 2021-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Neural network models implemented in `Flax <https://flax.readthedocs.io/en/latest/>`_ and utility functions.

Many of the function and parameter names used in this sub-package are
based on the somewhat non-standard Flax terminology for neural network
components:

`model`
    The model is an abstract representation of the network structure that
    does not include specific weight values.

`parameters`
    The parameters of a model are the weights of the network represented
    by the model.

`variables`
    The variables encompass both the parameters (i.e. network weights)
    and secondary values that are set from training data, such as
    layer-dependent statistics used in batch normalization.

`state`
    The state encompasses both a set of model parameters as well as
    optimizer parameters involved in training of that model. Storing the
    state rather than just the variables enables a warm start for
    additional training.

|
"""

import sys

# isort: off
from ._flax import FlaxMap, load_variables, save_variables
from ._models import ConvBNNet, DnCNNNet, ResNet, UNet
from .inverse import MoDLNet, ODPNet
from .train.input_pipeline import create_input_iter
from .train.typed_dict import ConfigDict
from .train.trainer import BasicFlaxTrainer
from .train.apply import only_apply
from .train.clu_utils import count_parameters

__all__ = [
    "FlaxMap",
    "load_variables",
    "save_variables",
    "ConvBNNet",
    "DnCNNNet",
    "ResNet",
    "UNet",
    "MoDLNet",
    "ODPNet",
    "create_input_iter",
    "ConfigDict",
    "BasicFlaxTrainer",
    "only_apply",
    "count_parameters",
]

# Imported items in __all__ appear to originate in top-level flax module
# except ConfigDict.
for name in __all__:
    if name != "ConfigDict":
        getattr(sys.modules[__name__], name).__module__ = __name__
