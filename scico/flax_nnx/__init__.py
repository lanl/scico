# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Neural network models implemented with Flax NNX.

The models, autoencoders and diffusion models of this sub-package, and
the trainers used to fit them, are re-exported here so that they have a
public import path; the modules they are defined in are an implementation
detail.

|
"""

import sys

# isort: off
from ._models import ConvBNNet, DnCNNNet, ResNet, UNet
from .utils import load_model, save_model
from .autoencoders.autoencoders import Decoder  # noqa: F401
from .autoencoders.autoencoders import (
    AutoEncoder,
    ConvAutoEncoder,
    Encoder,
    MLPAutoEncoder,
)
from .autoencoders.varautoencoders import (
    VAE,
    ConvVarAutoEncoder,
    MLPVarAutoEncoder,
    VarEncoder,
)
from .autoencoders.trainer_vae import FlaxNNXVAETrainer
from .diffusion.models import ConditionalUNet
from .diffusion.trainer_ddpm import FlaxNNXDDPMTrainer
from .diffusion.trainer_score import FlaxNNXScoreTrainer
from .diffusion.trainer_score_ncsn import FlaxNNXNCSNTrainer
from .train.trainer import BasicFlaxNNXTrainer
from .train.typed_dict import ConfigDict

__all__ = [
    "ConvBNNet",
    "DnCNNNet",
    "ResNet",
    "UNet",
    "load_model",
    "save_model",
    "Encoder",
    "AutoEncoder",
    "MLPAutoEncoder",
    "ConvAutoEncoder",
    "VarEncoder",
    "VAE",
    "MLPVarAutoEncoder",
    "ConvVarAutoEncoder",
    "ConditionalUNet",
    "BasicFlaxNNXTrainer",
    "FlaxNNXVAETrainer",
    "FlaxNNXScoreTrainer",
    "FlaxNNXNCSNTrainer",
    "FlaxNNXDDPMTrainer",
    "ConfigDict",
]

# Decoder is imported above, so that it can be used in annotations, but is
# deliberately absent from __all__: it is an alias of Encoder, so listing it
# would have autosummary emit a second page rendering the same class.
#
# Imported items in __all__ appear to originate in the top-level flax_nnx
# module except ConfigDict, which is a TypedDict.
for name in __all__:
    if name != "ConfigDict":
        getattr(sys.modules[__name__], name).__module__ = __name__
