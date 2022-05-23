# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

""" Alter the behavior of (monkey patch) several modules so that SCICO
is compatible with Python 3.7, allowing it to run on Google Colab.
This file should be removed when/if Colab upgrades to Python 3.8.
"""

import sys

if sys.version_info.major == 3 and sys.version_info.minor == 7:
    import typing

    typing.Literal = typing.Any  # type: ignore

    import math
    from functools import reduce
    from operator import mul

    # math.prod = (lambda fun, op: lambda iterable: fun(op, iterable, 1))(reduce, mul)  # type: ignore
    math.prod = lambda iterable: reduce(mul, iterable, 1)  # type: ignore
