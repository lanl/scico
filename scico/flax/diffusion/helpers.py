# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Helper functions for diffusion generative models."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from inspect import isfunction


def exists(x):
    """Determine if x is not none."""
    return x is not None


def default(val, d):
    """Return default value if given. Otherwise return object d.
    Args:
        val: Default value.
        d: Function or variable to return if no default value provided.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    """Group setup."""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
