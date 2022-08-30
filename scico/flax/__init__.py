# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Neural network models implemented in Flax and utility functions."""

import sys

# isort: off
from ._flax import FlaxMap, load_weights, save_weights
from .blocks import (
    ConvBlock,
    ConvBNBlock,
    ConvBNMultiBlock,
    ConvBNPoolBlock,
    ConvBNUpsampleBlock,
    upscale_nn,
)
from .models import ConvBNNet, DnCNNNet, ResNet, UNet
from .inverse import MoDLNet, ODPGrDescBlock, ODPNet, ODPProxDcnvBlock, ODPProxDnBlock
from .train.input_pipeline import create_input_iter
from .train.train import ConfigDict, BasicFlaxTrainer, only_apply
from .train.clu_utils import count_parameters

__all__ = [
    "FlaxMap",
    "load_weights",
    "save_weigths",
    "ConvBNBlock",
    "ConvBlock",
    "ConvBNPoolBlock",
    "ConvBNUpsampleBlock",
    "ConvBNMultiBlock",
    "upscale_nn",
    "ConvBNNet",
    "DnCNNNet",
    "ResNet",
    "UNet",
    "MoDLNet",
    "ODPProxDnBlock",
    "ODPProxDcnvBlock",
    "ODPGrDescBlock",
    "ODPNet",
    "create_input_iter",
    "ConfigDict",
    "BasicFlaxTrainer",
    "only_apply",
    "count_parameters",
]

# Imported items in __all__ appear to originate in top-level flax module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
