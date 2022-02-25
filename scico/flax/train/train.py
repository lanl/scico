# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for training Flax models."""


from typing import Any, TypedDict
import jax.numpy as jnp
from jax import lax
import optax

from scico.typing import Array
from scico.metric import snr

class ConfigDict(TypedDict):
    """Definition of the dictionary structure
    expected for the data sets for training."""

    seed: float
    model: str
    depth: int
    num_filters: int
    block_depth: int
    spectraln: bool
    opt_type: str
    lr: float
    momentum: float
    batch_size: int
    num_epochs: int

# Loss Function
def mse_loss(output: Array, labels: Array) -> float:
    """
    Compute Mean Squared Error (MSE) loss for training
    via optax.

    Args:
        output : Comparison signal.
        labels : Reference signal.

    Returns:
        MSE between `output` and `labels`.
    """
    mse = optax.l2_loss(output, labels)
    return jnp.mean(mse)


def compute_metrics(output: Array, labels: Array):
    """Compute diagnostic metrics.

    Args:
        output : Comparison signal.
        labels : Reference signal.

    Returns:
        MSE and SNR between `output` and `labels`.
        Assummes batched data.
    """
    loss = mse_loss(output, labels)
    snr_ = snr(labels, output)
    metrics = {
        'loss': loss,
        'snr': snr_,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics



def train_and_evaluate(config: ConfigDict):
    """Execute model training and evaluation loop."""
    return 0
