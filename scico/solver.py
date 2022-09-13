# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""SciPy optimization algorithms.

.. raw:: html

    <style type='text/css'>
    div.document li {
      list-style: square outside !important;
      margin-left: 1em !important;
    }
    div.document li > p {
       margin-bottom: 4px !important;
    }
    ul {
      margin-bottom: 1em;
    }
    </style>

This module provides scico interface wrappers for functions
from :mod:`scipy.optimize` since jax directly implements only a very
limited subset of these functions (there is limited, experimental support
for `L-BFGS-B <https://github.com/google/jax/pull/6053>`_), but only CG
and BFGS are fully supported. These wrappers are required because the
functions in :mod:`scipy.optimize` only support on 1D, real valued, numpy
arrays. These limitations are addressed by:

- Enabling the use of multi-dimensional arrays by flattening and reshaping
  within the wrapper.
- Enabling the use of jax arrays by automatically converting to and from
  numpy arrays.
- Enabling the use of complex arrays by splitting them into real and
  imaginary parts.

The wrapper also JIT compiles the function and gradient evaluations.

The functions provided in this module have a number of advantages and
disadvantages with respect to those in :mod:`jax.scipy.optimize`:

- This module provides many more algorithms than
  :mod:`jax.scipy.optimize`.
- The functions in this module tend to be faster for small-scale problems
  (presumably due to some overhead in the jax functions).
- The functions in this module are slower for large problems due to the
  frequent host-device copies corresponding to conversion between numpy
  arrays and jax arrays.
- The solvers in this module can't be JIT compiled, and gradients cannot
  be taken through them.

In the future, this module may be replaced with a dependency on
`JAXopt <https://github.com/google/jaxopt>`__.
"""


from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

import jax
import jax.experimental.host_callback as hcb

import scico.numpy as snp
from scico.numpy import BlockArray
from scico.typing import BlockShape, DType, JaxArray, Shape
from scipy import optimize as spopt


def _wrap_func(func: Callable, shape: Union[Shape, BlockShape], dtype: DType) -> Callable:
    """Function evaluation for use in :mod:`scipy.optimize`.

    Compute function evaluation (without gradient) for use in
    :mod:`scipy.optimize` functions. Reshapes the input to `func` to
    have `shape`. Evaluates `func`.

    Args:
        func: The function to minimize.
        shape: Shape of input to `func`.
        dtype: Data type of input to `func`.
    """

    val_func = jax.jit(func)

    @wraps(func)
    def wrapper(x, *args):
        # apply val_grad_func to un-vectorized input
        val = val_func(snp.reshape(x, shape).astype(dtype), *args)

        # Convert val into numpy array, then cast to float
        # Convert 'val' into a scalar, rather than ndarray of shape (1,)
        val = np.array(val).astype(float).item()
        return val

    return wrapper


def _wrap_func_and_grad(func: Callable, shape: Union[Shape, BlockShape], dtype: DType) -> Callable:
    """Function evaluation and gradient for use in :mod:`scipy.optimize`.

    Compute function evaluation and gradient for use in
    :mod:`scipy.optimize` functions. Reshapes the input to `func` to
    have `shape`.  Evaluates `func` and computes gradient. Ensures
    the returned `grad` is an ndarray.

    Args:
        func: The function to minimize.
        shape: Shape of input to `func`.
        dtype: Data type of input to `func`.
    """

    # argnums=0 ensures only differentiate func wrt first argument,
    #   in case func signature is func(x, *args)
    val_grad_func = jax.jit(jax.value_and_grad(func, argnums=0))

    @wraps(func)
    def wrapper(x, *args):
        # apply val_grad_func to un-vectorized input
        val, grad = val_grad_func(snp.reshape(x, shape).astype(dtype), *args)

        # Convert val & grad into numpy arrays, then cast to float
        # Convert 'val' into a scalar, rather than ndarray of shape (1,)
        val = np.array(val).astype(float).item()
        grad = np.array(grad).astype(float).ravel()
        return val, grad

    return wrapper


def _split_real_imag(x: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
    """Split an array of shape (N, M, ...) into real and imaginary parts.

    Args:
        x: Array to split.

    Returns:
        A real ndarray with stacked real/imaginary parts. If `x` has
        shape (M, N, ...), the returned array will have shape
        (2, M, N, ...) where the first slice contains the `x.real` and
        the second contains `x.imag`. If `x` is a BlockArray, this
        function is called on each block and the output is joined into a
        BlockArray.
    """
    if isinstance(x, BlockArray):
        return snp.blockarray([_split_real_imag(_) for _ in x])
    return snp.stack((snp.real(x), snp.imag(x)))


def _join_real_imag(x: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
    """Join a real array of shape (2,N,M,...) into a complex array.

    Join a real array of shape (2,N,M,...) into a complex array of length
    (N,M, ...).

    Args:
        x: Array to join.

    Returns:
        A complex array with real and imaginary parts taken from `x[0]`
        and `x[1]` respectively.
    """
    if isinstance(x, BlockArray):
        return snp.blockarray([_join_real_imag(_) for _ in x])
    return x[0] + 1j * x[1]


def minimize(
    func: Callable,
    x0: Union[JaxArray, BlockArray],
    args: Union[Tuple, Tuple[Any]] = (),
    method: str = "L-BFGS-B",
    hess: Optional[Union[Callable, str]] = None,
    hessp: Optional[Callable] = None,
    bounds: Optional[Union[Sequence, spopt.Bounds]] = None,
    constraints: Union[spopt.LinearConstraint, spopt.NonlinearConstraint, dict] = (),
    tol: Optional[float] = None,
    callback: Optional[Callable] = None,
    options: Optional[dict] = None,
) -> spopt.OptimizeResult:
    """Minimization of scalar function of one or more variables.

    Wrapper around :func:`scipy.optimize.minimize`. This function differs
    from :func:`scipy.optimize.minimize` in three ways:

        - The `jac` options of :func:`scipy.optimize.minimize` are not
          supported. The gradient is calculated using `jax.grad`.
        - Functions mapping from N-dimensional arrays -> float are
          supported.
        - Functions mapping from complex arrays -> float are supported.

    For more detail, including descriptions of the optimization methods
    and custom minimizers, refer to the original docs for
    :func:`scipy.optimize.minimize`.
    """

    if snp.util.is_complex_dtype(x0.dtype):
        # scipy minimize function requires real-valued arrays, so
        # we split x0 into a vector with real/imaginary parts stacked
        # and compose `func` with a `_join_real_imag`
        iscomplex = True
        func_ = lambda x: func(_join_real_imag(x))
        x0 = _split_real_imag(x0)
    else:
        iscomplex = False
        func_ = func

    x0_shape = x0.shape
    x0_dtype = x0.dtype
    x0 = x0.ravel()  # if x0 is a BlockArray it will become a DeviceArray here

    # Run the SciPy minimizer
    if method in (
        "CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, "
        "trust-exact, trust-constr"
    ).split(
        ", "
    ):  # uses gradient info
        min_func = _wrap_func_and_grad(func_, x0_shape, x0_dtype)
        jac = True  # see scipy.minimize docs
    else:  # does not use gradient info
        min_func = _wrap_func(func_, x0_shape, x0_dtype)
        jac = False

    res = spopt.OptimizeResult({"x": None})

    def fun(x0):
        nonlocal res  # To use the external res and update side effect
        res = spopt.minimize(
            min_func,
            x0=x0,
            args=args,
            jac=jac,
            method=method,
            options=options,
        )  # Returns OptimizeResult with x0 as ndarray
        return res.x.astype(x0_dtype)

    # HCB call with side effects to get the OptimizeResult on the same device it was called
    res.x = hcb.call(
        fun,
        arg=x0,
        result_shape=x0,  # From Jax-docs: This can be an object that has .shape and .dtype attributes
    )

    # un-vectorize the output array from spopt.minimize
    res.x = snp.reshape(
        res.x, x0_shape
    )  # if x0 was originally a BlockArray then res.x is converted back to one here

    if iscomplex:
        res.x = _join_real_imag(res.x)

    return res


def minimize_scalar(
    func: Callable,
    bracket: Optional[Union[Sequence[float]]] = None,
    bounds: Optional[Sequence[float]] = None,
    args: Union[Tuple, Tuple[Any]] = (),
    method: str = "brent",
    tol: Optional[float] = None,
    options: Optional[dict] = None,
) -> spopt.OptimizeResult:

    """Minimization of scalar function of one variable.

    Wrapper around :func:`scipy.optimize.minimize_scalar`.

    For more detail, including descriptions of the optimization methods
    and custom minimizers, refer to the original docstring for
    :func:`scipy.optimize.minimize_scalar`.
    """

    def f(x, *args):
        # Wrap jax-based function `func` to return a numpy float
        # rather than a DeviceArray of size (1,)
        return func(x, *args).item()

    res = spopt.minimize_scalar(
        fun=f,
        bracket=bracket,
        bounds=bounds,
        args=args,
        method=method,
        tol=tol,
        options=options,
    )
    return res


def cg(
    A: Callable,
    b: JaxArray,
    x0: JaxArray,
    *,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int = 1000,
    info: bool = False,
    M: Optional[Callable] = None,
) -> Tuple[JaxArray, dict]:
    r"""Conjugate Gradient solver.

    Solve the linear system :math:`A\mb{x} = \mb{b}`, where :math:`A` is
    positive definite, via the conjugate gradient method.

    Args:
        A: Function implementing linear operator :math:`A`, should be
            positive definite.
        b: Input array :math:`\mb{b}`.
        x0: Initial solution.
        tol: Relative residual stopping tolerance. Convergence occurs
           when `norm(residual) <= max(tol * norm(b), atol)`.
        atol: Absolute residual stopping tolerance. Convergence occurs
           when `norm(residual) <= max(tol * norm(b), atol)`.
        maxiter: Maximum iterations. Default: 1000.
        M: Preconditioner for `A`. The preconditioner should approximate
           the inverse of `A`. The default, ``None``, uses no
           preconditioner.

    Returns:
        tuple: A tuple (x, info) containing:

            - **x** : Solution array.
            - **info**: Dictionary containing diagnostic information.
    """

    if M is None:
        M = lambda x: x

    x = x0
    Ax = A(x0)
    bn = snp.linalg.norm(b)
    r = b - Ax
    z = M(r)
    p = z
    num = snp.sum(r.conj() * z)
    ii = 0

    # termination tolerance
    # uses the "non-legacy" form of scicpy.sparse.linalg.cg
    termination_tol_sq = snp.maximum(tol * bn, atol) ** 2

    while (ii < maxiter) and (num > termination_tol_sq):
        Ap = A(p)
        alpha = num / snp.sum(p.conj() * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M(r)
        num_old = num
        num = snp.sum(r.conj() * z)
        beta = num / num_old
        p = z + beta * p
        ii += 1

    return (x, {"num_iter": ii, "rel_res": snp.sqrt(num).real / bn})


def bisect(
    f: Callable,
    a: JaxArray,
    b: JaxArray,
    args: Tuple = (),
    xtol: float = 1e-7,
    ftol: float = 1e-7,
    maxiter: int = 100,
    full_output: bool = False,
    range_check: bool = True,
) -> Union[JaxArray, dict]:
    """Vectorised root finding via bisection method.

    Vectorised root finding via bisection method, supporting
    simultaneous finding of multiple roots on a function defined over a
    multi-dimensional array. When the function is array-valued, each of
    these values is treated as the independent application of a scalar
    function. The initial interval `[a, b]` must bracket the root for all
    scalar functions.

    The interface is similar to that of :func:`scipy.optimize.bisect`,
    which is much faster when `f` is a scalar function and `a` and `b`
    are scalars.

    Args:
        f: Function returning a float or an array of floats.
        a: Lower bound of interval on which to apply bisection.
        b: Upper bound of interval on which to apply bisection.
        args: Additional arguments for function `f`.
        xtol: Stopping tolerance based on maximum bisection interval
            length over array.
        ftol: Stopping tolerance based on maximum absolute function value
            over array.
        maxiter: Maximum number of algorithm iterations.
        full_output: If ``False``, return just the root, otherwise return a
            tuple `(x, info)` where `x` is the root and `info` is a dict
            containing algorithm status information.
        range_check: If ``True``, check to ensure that the initial
            `[a, b]` range brackets the root of `f`.

    Returns:
        tuple: A tuple `(x, info)` containing:

            - **x** : Root array.
            - **info**: Dictionary containing diagnostic information.
    """

    fa = f(*((a,) + args))
    fb = f(*((b,) + args))
    if range_check and snp.any(snp.sign(fa) == snp.sign(fb)):
        raise ValueError("Initial bisection range does not bracket zero")

    for numiter in range(maxiter):
        c = (a + b) / 2.0
        fc = f(*((c,) + args))
        fcs = snp.sign(fc)
        a = snp.where(snp.logical_or(snp.sign(fa) * fcs == 1, fc == 0.0), c, a)
        b = snp.where(snp.logical_or(fcs * snp.sign(fb) == 1, fc == 0.0), c, b)
        fa = f(*((a,) + args))
        fb = f(*((b,) + args))
        xerr = snp.max(snp.abs(b - a))
        ferr = snp.max(snp.abs(fc))
        if xerr <= xtol and ferr <= ftol:
            break

    idx = snp.argmin(snp.stack((snp.abs(fa), snp.abs(fb))), axis=0)
    x = snp.choose(idx, (a, b))
    if full_output:
        r = x, {"iter": numiter, "xerr": xerr, "ferr": ferr, "a": a, "b": b}
    else:
        r = x
    return r


def golden(
    f: Callable,
    a: JaxArray,
    b: JaxArray,
    c: Optional[JaxArray] = None,
    args: Tuple = (),
    xtol: float = 1e-7,
    maxiter: int = 100,
    full_output: bool = False,
) -> Union[JaxArray, dict]:
    """Vectorised scalar minimization via golden section method.

    Vectorised scalar minimization via golden section method, supporting
    simultaneous minimization of a function defined over a
    multi-dimensional array. When the function is array-valued, each of
    these values is treated as the independent application of a scalar
    function. The minimizer must lie within the interval `(a, b)` for all
    scalar functions, and, if specified `c` must be within that interval.


    The interface is more similar to that of :func:`.bisect` than that of
    :func:`scipy.optimize.golden` which is much faster when `f` is a
    scalar function and `a`, `b`, and `c` are scalars.

    Args:
        f: Function returning a float or an array of floats.
        a: Lower bound of interval on which to search.
        b: Upper bound of interval on which to search.
        c: Initial value for first search point interior to bounding
            interval `(a, b)`
        args: Additional arguments for function `f`.
        xtol: Stopping tolerance based on maximum search interval length
            over array.
        maxiter: Maximum number of algorithm iterations.
        full_output: If ``False``, return just the minizer, otherwise
            return a tuple `(x, info)` where `x` is the minimizer and
            `info` is a dict containing algorithm status information.

    Returns:
        tuple: A tuple `(x, info)` containing:

            - **x** : Minimizer array.
            - **info**: Dictionary containing diagnostic information.
    """
    gr = 2 / (snp.sqrt(5) + 1)
    if c is None:
        c = b - gr * (b - a)
    d = a + gr * (b - a)
    for numiter in range(maxiter):
        fc = f(*((c,) + args))
        fd = f(*((d,) + args))
        b = snp.where(fc < fd, d, b)
        a = snp.where(fc >= fd, c, a)
        xerr = snp.amax(snp.abs(b - a))
        if xerr <= xtol:
            break
        c = b - gr * (b - a)
        d = a + gr * (b - a)

    fa = f(*((a,) + args))
    fb = f(*((b,) + args))
    idx = snp.argmin(snp.stack((fa, fb)), axis=0)
    x = snp.choose(idx, (a, b))
    if full_output:
        r = (x, {"iter": numiter, "xerr": xerr})
    else:
        r = x
    return r
