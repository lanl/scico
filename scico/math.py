# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Math functions."""


from typing import Union

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.typing import DType, JaxArray

__author__ = """\n""".join(["Luke Pfister <pfister@lanl.gov>", "Brendt Wohlberg <brendt@ieee.org>"])


def safe_divide(
    x: Union[BlockArray, JaxArray], y: Union[BlockArray, JaxArray]
) -> Union[BlockArray, JaxArray]:
    """Return `x/y`, with 0 instead of NaN where `y` is 0.

    Args:
        x:  Numerator
        y:  Denominator

    Returns:
        `x / y` with 0 wherever `y == 0`.
    """

    return snp.where(y != 0, snp.divide(x, snp.where(y != 0, y, 1)), 0)


def rel_res(ax: Union[BlockArray, JaxArray], b: Union[BlockArray, JaxArray]) -> float:
    r"""Relative residual of the solution to a linear equation.

    The standard relative residual for the linear system :math:`A \mathbf{x} = \mathbf{b}`
    is :math:`\|\mathbf{b} - A \mathbf{x}\|_2 / \|\mathbf{b}\|_2`. This function computes
    a variant :math:`\|\mathbf{b} - A \mathbf{x}\|_2 / \max(\|A\mathbf{x}\|_2,
    \|\mathbf{b}\|_2)` that is robust to the case :math:`\mathbf{b} = 0`.

    Args:
        ax: Linear component :math:`A \mathbf{x}` of equation.
        b: Constant component :math:`\mathbf{b}` of equation.

    Returns:
        x: Relative residual value.
    """

    nrm = max(snp.linalg.norm(ax.ravel()), snp.linalg.norm(b.ravel()))
    if nrm == 0.0:
        return 0.0
    else:
        return snp.linalg.norm((b - ax).ravel()) / nrm


def is_real_dtype(dtype: DType) -> bool:
    """Determine whether a dtype is real.

    Args:
        dtype: A numpy or scico.numpy dtype (e.g. np.float32, snp.complex64)

    Returns:
        False if the dtype is complex, otherwise True
    """
    return snp.dtype(dtype).kind != "c"


def is_complex_dtype(dtype: DType) -> bool:
    """Determine whether a dtype is complex.

    Args:
        dtype: A numpy or scico.numpy dtype (e.g. np.float32, snp.complex64)

    Returns:
        True if the dtype is complex, otherwise False
    """
    return snp.dtype(dtype).kind == "c"


def real_dtype(dtype: DType) -> DType:
    """Construct the corresponding real dtype for a given complex dtype.

    Construct the corresponding real dtype for a given complex dtype,
    e.g. the real dtype corresponding to `np.complex64` is
    `np.float32`.

    Args:
        dtype: A complex numpy or scico.numpy  dtype, e.g. np.complex64, np.complex128

    Returns:
        The real dtype corresponding to the input dtype
    """

    return snp.zeros(1, dtype).real.dtype


def complex_dtype(dtype: DType) -> DType:
    """Construct the corresponding complex dtype for a given real dtype.

    Construct the corresponding complex dtype for a given real dtype,
    e.g. the complex dtype corresponding to `np.float32` is
    `np.complex64`.

    Args:
        dtype: A real numpy or scico.numpy dtype, e.g. np.float32, np.float64

    Returns:
        The complex dtype corresponding to the input dtype
    """

    return (snp.zeros(1, dtype) + 1j).dtype
