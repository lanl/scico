# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Image quality metrics and related functions."""

# This module is copied from https://github.com/bwohlberg/sporco

from typing import Optional, Union

import numpy as np

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.typing import JaxArray

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


def mae(reference: Union[JaxArray, BlockArray], comparison: Union[JaxArray, BlockArray]) -> float:
    """Compute Mean Absolute Error (MAE) between two images.

    Args:
        reference: Reference image.
        comparison: Comparison image.

    Returns:
        MAE between `reference` and `comparison`.
    """

    return snp.mean(snp.abs(reference - comparison).ravel())


def mse(reference: Union[JaxArray, BlockArray], comparison: Union[JaxArray, BlockArray]) -> float:
    """Compute Mean Squared Error (MSE) between two images.

    Args:
        reference : Reference image.
        comparison : Comparison image.

    Returns:
        MSE between `reference` and `comparison`.
    """

    return snp.mean(snp.abs(reference - comparison).ravel() ** 2)


def snr(reference: Union[JaxArray, BlockArray], comparison: Union[JaxArray, BlockArray]) -> float:
    """Compute Signal to Noise Ratio (SNR) of two images.

    Args:
        reference: Reference image.
        comparison: Comparison image.

    Returns:
        SNR of `comparison` with respect to `reference`.
    """

    dv = snp.var(reference)
    with np.errstate(divide="ignore"):
        rt = dv / mse(reference, comparison)
    return 10.0 * snp.log10(rt)


def psnr(
    reference: Union[JaxArray, BlockArray],
    comparison: Union[JaxArray, BlockArray],
    signal_range: Optional[Union[int, float]] = None,
) -> float:
    """Compute Peak Signal to Noise Ratio (PSNR) of two images.

    The PSNR calculation defaults to using the less common definition
    in terms of the actual range (i.e. max minus min) of the reference
    signal instead of the maximum possible range for the data type
    (i.e. :math:`2^b-1` for a :math:`b` bit representation).

    Args:
        reference: Reference image.
        comparison: Comparison image.
        signal_range: Signal range, either the value to use (e.g. 255
            for 8 bit samples) or None, in which case the actual range
            of the reference signal is used.

    Returns:
        PSNR of `comparison` with respect to `reference`.
    """

    if signal_range is None:
        signal_range = snp.abs(snp.max(reference) - snp.min(reference))
    with np.errstate(divide="ignore"):
        rt = signal_range ** 2 / mse(reference, comparison)
    return 10.0 * snp.log10(rt)


def isnr(
    reference: Union[JaxArray, BlockArray],
    degraded: Union[JaxArray, BlockArray],
    restored: Union[JaxArray, BlockArray],
) -> float:
    """Compute Improvement Signal to Noise Ratio (ISNR).

    Compute Improvement Signal to Noise Ratio (ISNR) for reference,
    degraded, and restored images.

    Args:
        reference: Reference image.
        degraded: Degraded image.
        restored: Restored image.

    Returns:
        ISNR of `restored` with respect to `reference` and `degraded`.
    """

    msedeg = mse(reference, degraded)
    mserst = mse(reference, restored)
    with np.errstate(divide="ignore"):
        rt = msedeg / mserst
    return 10.0 * snp.log10(rt)


def bsnr(blurry: Union[JaxArray, BlockArray], noisy: Union[JaxArray, BlockArray]) -> float:
    """Compute Blurred Signal to Noise Ratio (BSNR).

    Compute Blurred Signal to Noise Ratio (BSNR) for a blurred and noisy
    image.

    Args:
        blurry: Blurred noise free image.
        noisy: Blurred image with additive noise.

    Returns:
        BSNR of `noisy` with respect to `blurry` and `degraded`.
    """

    blrvar = snp.var(blurry)
    nsevar = snp.var(noisy - blurry)
    with np.errstate(divide="ignore"):
        rt = blrvar / nsevar
    return 10.0 * snp.log10(rt)


def rel_res(ax: Union[BlockArray, JaxArray], b: Union[BlockArray, JaxArray]) -> float:
    r"""Relative residual of the solution to a linear equation.

    The standard relative residual for the linear system
    :math:`A \mathbf{x} = \mathbf{b}` is :math:`\|\mathbf{b} -
    A \mathbf{x}\|_2 / \|\mathbf{b}\|_2`. This function computes a
    variant :math:`\|\mathbf{b} - A \mathbf{x}\|_2 /
    \max(\|A\mathbf{x}\|_2, \|\mathbf{b}\|_2)` that is robust to the case
    :math:`\mathbf{b} = 0`.

    Args:
        ax: Linear component :math:`A \mathbf{x}` of equation.
        b: Constant component :math:`\mathbf{b}` of equation.

    Returns:
        Relative residual value.
    """

    nrm = max(snp.linalg.norm(ax.ravel()), snp.linalg.norm(b.ravel()))
    if nrm == 0.0:
        return 0.0
    return snp.linalg.norm((b - ax).ravel()) / nrm
