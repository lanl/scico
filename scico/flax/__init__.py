# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Neural network models implemented in Flax and utility functions."""

import sys

# isort: off
from ._flax import FlaxMap, load_weights
from .blocks import (
    ConvBNBlock,
    ConvBlock,
    ConvBNPoolBlock,
    ConvBNUpsampleBlock,
    ConvBNMultiBlock,
    upscale_nn,
)
from .models import DnCNNNet, ResNet, ConvBNNet, UNet

from .inverse import MoDLNet, ODPGrDescBlock, ODPNet

from .train.input_pipeline import create_input_iter
from .train.train import ConfigDict, train_and_evaluate, only_evaluate


__all__ = [
    "FlaxMap",
    "load_weights",
    "ConvBNBlock",
    "ConvBlock",
    "ConvBNPoolBlock",
    "ConvBNUpsampleBlock",
    "ConvBNMultiBlock",
    "upscale_nn",
    "DnCNNNet",
    "ResNet",
    "ConvBNNet",
    "UNet",
    "MoDLNet",
    "ODPGrDescBlock",
    "ODPNet",
    "create_input_iter",
    "ConfigDict",
    "train_and_evaluate",
    "only_evaluate",
]

# Imported items in __all__ appear to originate in top-level flax module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
