# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.


"""
Lists of functions to be wrapped in scico.numpy.

These are intended to be the functions in :mod:`jax.numpy` that should
either
   #. map over the blocks of a block array (for math functions);
   #. map over a tuple of tuples to create a block array (for creation
      functions); or
   #. reduce a block array to a scalar (for reductions).

The links to the numpy docs in the comments are useful for distinguishing
between these three cases, but note that these lists of numpy functions
include extra functions that are not in :mod:`jax.numpy`, and that are
therefore not listed here.
"""


""" BlockArray """
unary_ops = (  # found from dir() on jax array
    "__abs__",
    "__neg__",
    "__pos__",
)

binary_ops = (  # found from dir() on jax array
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__mod__",
    "__rmul__",
    "__matmul__",
    "__rmatmul__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__pow__",
    "__rpow__",
    "__gt__",
    "__ge__",
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
)

""" jax.numpy """

creation_routines = (
    "empty",
    "ones",
    "zeros",
    "full",
)

mathematical_functions = (
    "sin",  # https://numpy.org/doc/stable/reference/routines.math.html
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "hypot",
    "arctan2",
    "degrees",
    "radians",
    "unwrap",
    "deg2rad",
    "rad2deg",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "around",
    "round_",
    "rint",
    "fix",
    "floor",
    "ceil",
    "trunc",
    "prod",
    "sum",
    "nanprod",
    "nansum",
    "cumprod",
    "cumsum",
    "nancumprod",
    "nancumsum",
    "diff",
    "ediff1d",
    "gradient",
    "cross",
    "trapz",
    "exp",
    "expm1",
    "exp2",
    "log",
    "log10",
    "log2",
    "log1p",
    "logaddexp",
    "logaddexp2",
    "i0",
    "sinc",
    "signbit",
    "copysign",
    "frexp",
    "ldexp",
    "nextafter",
    "lcm",
    "gcd",
    "add",
    "reciprocal",
    "positive",
    "negative",
    "multiply",
    "divide",
    "power",
    "subtract",
    "true_divide",
    "floor_divide",
    "float_power",
    "fmod",
    "mod",
    "modf",
    "remainder",
    "divmod",
    "angle",
    "real",
    "imag",
    "conj",
    "conjugate",
    "maximum",
    "fmax",
    "amax",
    "nanmax",
    "minimum",
    "fmin",
    "amin",
    "nanmin",
    "convolve",
    "clip",
    "sqrt",
    "cbrt",
    "square",
    "abs",
    "absolute",
    "fabs",
    "sign",
    "heaviside",
    "nan_to_num",
    "interp",
    "sort",  # https://numpy.org/doc/stable/reference/routines.sort.html
    "lexsort",
    "argsort",
    "sort_complex",
    "partition",
    "argmax",
    "nanargmax",
    "argmin",
    "nanargmin",
    "argwhere",
    "nonzero",
    "flatnonzero",
    "where",
    "searchsorted",
    "extract",
    "count_nonzero",
    "dot",  # https://numpy.org/doc/stable/reference/routines.linalg.html
    "linalg.multi_dot",
    "vdot",
    "inner",
    "outer",
    "matmul",
    "tensordot",
    "einsum",
    "einsum_path",
    "linalg.matrix_power",
    "kron",
    "linalg.cholesky",
    "linalg.qr",
    "linalg.svd",
    "linalg.eig",
    "linalg.eigh",
    "linalg.eigvals",
    "linalg.eigvalsh",
    "linalg.norm",
    "linalg.cond",
    "linalg.det",
    "linalg.matrix_rank",
    "linalg.slogdet",
    "trace",
    "linalg.solve",
    "linalg.tensorsolve",
    "linalg.lstsq",
    "linalg.inv",
    "linalg.pinv",
    "linalg.tensorinv",
    "shape",  # https://numpy.org/doc/stable/reference/routines.array-manipulation.html
    "reshape",
    "ravel",
    "moveaxis",
    "rollaxis",
    "swapaxes",
    "transpose",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "expand_dims",
    "squeeze",
    "asarray",
    "stack",
    "block",
    "vstack",
    "hstack",
    "dstack",
    "column_stack",
    "row_stack",
    "split",
    "array_split",
    "dsplit",
    "hsplit",
    "vsplit",
    "tile",
    "repeat",
    "insert",
    "append",
    "resize",
    "trim_zeros",
    "unique",
    "flip",
    "fliplr",
    "flipud",
    "reshape",
    "roll",
    "rot90",
    "all",
    "any",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "iscomplex",
    "iscomplexobj",
    "isreal",
    "isrealobj",
    "isscalar",
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "allclose",
    "isclose",
    "array_equal",
    "array_equiv",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "empty_like",  # https://numpy.org/doc/stable/reference/routines.array-creation.html
    "ones_like",
    "zeros_like",
    "full_like",
)

reduction_functions = ("sum", "linalg.norm", "count_nonzero", "any")

""" "testing", """

testing_functions = ("testing.assert_allclose", "testing.assert_array_equal")
