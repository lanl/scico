# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of loss functions for model optimization."""


from typing import List, Optional

import optax

from scico.numpy import Array


def mse_loss(output: Array, labels: Array, reduce_dims: Optional[List[int]] = None) -> float:
    """Compute Mean Squared Error (MSE) loss for training via Optax.

    Args:
        output: Comparison signal.
        labels: Reference signal.
        reduce_dims: Dimensions to reduce over. If None, all dimensions (except the first,
        assumed as batch) are reduced.

    Returns:
        MSE between `output` and `labels`.
    """
    mse = optax.l2_loss(output, labels)
    if reduce_dims is None:
        reduce_dims = range(1, len(mse.shape))  # Reduce over all dimensions except batch
    return mse.sum(axis=reduce_dims).mean()


def huber_loss(output: Array, labels: Array, reduce_dims: Optional[List[int]] = None) -> float:
    """Compute Hubber loss for training via Optax.

    Args:
        output: Comparison signal.
        labels: Reference signal.
        reduce_dims: Dimensions to reduce over. If None, all dimensions (except the first,
        assumed as batch) are reduced.

    Returns:
        Huber loss between `output` and `labels`.
    """
    hbl = optax.huber_loss(output, labels)
    if reduce_dims is None:
        reduce_dims = range(1, len(hbl.shape))  # Reduce over all dimensions except batch
    return hbl.sum(axis=reduce_dims).mean()
