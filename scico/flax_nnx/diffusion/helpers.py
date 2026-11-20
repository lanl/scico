# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Helper functions for diffusion generative models."""

import warnings
from typing import Any

warnings.simplefilter(action="ignore", category=FutureWarning)

from inspect import isfunction


def exists(x: Any) -> bool:
    """Determine whether `x` is not ``None``.

    Args:
        x: Object to test.

    Returns:
        ``True`` if `x` is not ``None``, otherwise ``False``.
    """
    return x is not None


def default(val: Any, d: Any) -> Any:
    """Return `val` if it is not ``None``, otherwise fall back to `d`.

    Args:
        val: Value under test. Returned unchanged if not ``None``.
        d: Fallback used when `val` is ``None``. If it is a function it is
            called with no arguments and its result returned, otherwise it
            is returned as is.

    Returns:
        `val` if it is not ``None``, otherwise `d` or the result of
        calling it.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num: int, divisor: int) -> list[int]:
    """Split `num` into a list of group sizes of at most `divisor`.

    Args:
        num: Total to split.
        divisor: Maximum size of each group.

    Returns:
        List of `divisor`-sized group sizes, with a final smaller entry
        holding the remainder if `divisor` does not divide `num`.
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
