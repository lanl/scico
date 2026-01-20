# -*- coding: utf-8 -*-
# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Solver and optimization algorithms.

This module provides a number of functions for solving linear systems and
optimization problems, some of which are used as subproblem solvers
within the iterations of the proximal algorithms in the
:mod:`scico.optimize` subpackage.

This module also provides scico interface wrappers for functions
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

These wrapper functions have a number of advantages and disadvantages
with respect to those in :mod:`jax.scipy.optimize`:

- This module provides many more algorithms than
  :mod:`jax.scipy.optimize`.
- The functions in this module tend to be faster for small-scale problems
  (presumably due to some overhead in the jax functions).
- The functions in this module are slower for large problems due to the
  frequent host-device copies corresponding to conversion between numpy
  arrays and jax arrays.
- The solvers in this module can't be JIT compiled, and gradients cannot
  be taken through them.

In the future, these wrapper functions may be replaced with a dependency
on `JAXopt <https://github.com/google/jaxopt>`__.
"""

from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

import scico.numpy as snp
from scico.linop import (
    CircularConvolve,
    ComposedLinearOperator,
    Diagonal,
    LinearOperator,
    MatrixOperator,
    Sum,
)
from scico.metric import rel_res
from scico.numpy import Array, BlockArray
from scico.numpy.util import is_complex_dtype, is_nested, is_real_dtype
from scico.typing import BlockShape, DType, Shape
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
        val = val_func(_unravel(x, shape).astype(dtype), *args)

        # Convert val into numpy array, cast to float, convert to scalar
        val = np.array(val).astype(float)
        val = val.item() if val.ndim == 0 else val[0].item()

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
        val, grad = val_grad_func(_unravel(x, shape).astype(dtype), *args)

        # Convert val & grad into numpy arrays, then cast to float
        # Convert 'val' into a scalar, rather than ndarray of shape (1,)
        val = np.array(val).astype(float).item()
        grad = np.array(grad).astype(float).ravel()
        return val, grad

    return wrapper


def _split_real_imag(x: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
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


def _join_real_imag(x: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
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


def _ravel(x: Union[Array, BlockArray]) -> Array:
    """Vectorize an array or blockarray to a 1d array.

    Args:
        x: Array or blockarray to be vectorized.

    Returns:
        Vectorized array.
    """
    if isinstance(x, snp.BlockArray):
        return jnp.hstack(x.ravel().arrays)
    else:
        return x.ravel()


def _unravel(x: Array, shape: Union[Shape, BlockShape]) -> Union[Array, BlockArray]:
    """Return a vectorized array or blockarray to its original shape.

    Args:
        x: Vectorized array representation.
        shape: Shape of original array or blockarray.

    Returns:
        Array or blockarray with original shape.
    """
    if is_nested(shape):
        sizes = [np.prod(e).item() for e in shape]
        indices = np.cumsum(sizes[:-1])
        chunks = jnp.split(x, indices)
        return snp.BlockArray([chunks[k].reshape(cs) for k, cs in enumerate(shape)])
    else:
        return x.reshape(shape)


def minimize(
    func: Callable,
    x0: Union[Array, BlockArray],
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
          supported. The gradient is calculated using :func:`jax.grad`.
        - Functions mapping from N-dimensional arrays -> float are
          supported.
        - Functions mapping from complex arrays -> float are supported.

    For more detail, including descriptions of the optimization methods
    and custom minimizers, refer to the original docs for
    :func:`scipy.optimize.minimize`.
    """

    if is_complex_dtype(x0.dtype):
        # scipy minimize function requires real-valued arrays, so
        # we split x0 into a vector with real/imaginary parts stacked
        # and compose `func` with a `_join_real_imag`
        iscomplex = True
        func_real = lambda x: func(_join_real_imag(x))
        x0 = _split_real_imag(x0)
    else:
        iscomplex = False
        func_real = func

    x0_shape = x0.shape
    x0_dtype = x0.dtype
    x0 = _ravel(x0)

    # Run the SciPy minimizer
    if method in (
        "CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, "
        "trust-exact, trust-constr"
    ).split(
        ", "
    ):  # uses gradient info
        min_func = _wrap_func_and_grad(func_real, x0_shape, x0_dtype)
        jac = True  # see scipy.minimize docs
    else:  # does not use gradient info
        min_func = _wrap_func(func_real, x0_shape, x0_dtype)
        jac = False

    res = spopt.OptimizeResult({"x": None})

    def fun(x0):
        nonlocal res  # To use the external res
        res = spopt.minimize(
            min_func,
            x0=x0,
            args=args,
            jac=jac,
            method=method,
            options=options,
        )  # Return OptimizeResult with x0 as ndarray
        return res.x.astype(x0_dtype)

    res.x = jax.pure_callback(
        fun,
        jax.ShapeDtypeStruct(x0.shape, x0_dtype),
        x0,
    )

    res.x = _unravel(res.x, x0_shape)  # un-vectorize the output array from spopt.minimize
    if iscomplex:
        res.x = _join_real_imag(res.x)

    return res


def minimize_scalar(
    func: Callable,
    bracket: Optional[Sequence[float]] = None,
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
        # Wrap jax-based function `func` to return a numpy float rather
        # than a jax array of size (1,)
        y = func(x, *args)
        return y.item() if y.ndim == 0 else y[0].item()

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
    b: Array,
    x0: Optional[Array] = None,
    *,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int = 1000,
    info: bool = True,
    M: Optional[Callable] = None,
) -> Tuple[Array, dict]:
    r"""Conjugate Gradient solver.

    Solve the linear system :math:`A\mb{x} = \mb{b}`, where :math:`A` is
    positive definite, via the conjugate gradient method.

    Args:
        A: Callable implementing linear operator :math:`A`, which should
           be positive definite.
        b: Input array :math:`\mb{b}`.
        x0: Initial solution. If `A` is a :class:`.LinearOperator`, this
          parameter need not be specified, and defaults to a zero array.
          Otherwise, it is required.
        tol: Relative residual stopping tolerance. Convergence occurs
           when `norm(residual) <= max(tol * norm(b), atol)`.
        atol: Absolute residual stopping tolerance. Convergence occurs
           when `norm(residual) <= max(tol * norm(b), atol)`.
        maxiter: Maximum iterations. Default: 1000.
        info: If ``True`` return a tuple consting of the solution array
           and a dictionary containing diagnostic information, otherwise
           just return the solution.
        M: Preconditioner for `A`. The preconditioner should approximate
           the inverse of `A`. The default, ``None``, uses no
           preconditioner.

    Returns:
        tuple: A tuple (x, info) containing:

            - **x** : Solution array.
            - **info**: Dictionary containing diagnostic information.
    """
    if x0 is None:
        if isinstance(A, LinearOperator):
            x0 = snp.zeros(A.input_shape, b.dtype)
        else:
            raise ValueError(
                "Argument 'x0' must be specified if argument 'A' is not a LinearOperator."
            )

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

    # termination tolerance (uses the "non-legacy" form of scicpy.sparse.linalg.cg)
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

    if info:
        return (x, {"num_iter": ii, "rel_res": snp.sqrt(num).real / bn})
    else:
        return x


def lstsq(
    A: Callable,
    b: Array,
    x0: Optional[Array] = None,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int = 1000,
    info: bool = False,
    M: Optional[Callable] = None,
) -> Tuple[Array, dict]:
    r"""Least squares solver.

    Solve the least squares problem

    .. math::
        \argmin_{\mb{x}} \; (1/2) \norm{ A \mb{x} - \mb{b} }_2^2 \;,

    where :math:`A` is a linear operator and :math:`\mb{b}` is a vector.
    The problem is solved using :func:`cg`.

    Args:
        A: Callable implementing linear operator :math:`A`.
        b: Input array :math:`\mb{b}`.
        x0: Initial solution. If `A` is a :class:`.LinearOperator`, this
          parameter need not be specified, and defaults to a zero array.
          Otherwise, it is required.
        tol: Relative residual stopping tolerance. Convergence occurs
           when `norm(residual) <= max(tol * norm(b), atol)`.
        atol: Absolute residual stopping tolerance. Convergence occurs
           when `norm(residual) <= max(tol * norm(b), atol)`.
        maxiter: Maximum iterations. Default: 1000.
        info: If ``True`` return a tuple consting of the solution array
           and a dictionary containing diagnostic information, otherwise
           just return the solution.
        M: Preconditioner for `A`. The preconditioner should approximate
           the inverse of `A`. The default, ``None``, uses no
           preconditioner.

    Returns:
        tuple: A tuple (x, info) containing:

            - **x** : Solution array.
            - **info**: Dictionary containing diagnostic information.
    """
    if isinstance(A, LinearOperator):
        Aop = A
    else:
        assert x0 is not None
        Aop = LinearOperator(
            input_shape=x0.shape,
            output_shape=b.shape,
            eval_fn=A,
            input_dtype=b.dtype,
            output_dtype=b.dtype,
        )

    ATA = Aop.T @ Aop
    ATb = Aop.T @ b
    return cg(ATA, ATb, x0=x0, tol=tol, atol=atol, maxiter=maxiter, info=info, M=M)


def bisect(
    f: Callable,
    a: Array,
    b: Array,
    args: Tuple = (),
    xtol: float = 1e-7,
    ftol: float = 1e-7,
    maxiter: int = 100,
    full_output: bool = False,
    range_check: bool = True,
) -> Union[Array, dict]:
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
        raise ValueError("Initial bisection range does not bracket zero.")

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
    a: Array,
    b: Array,
    c: Optional[Array] = None,
    args: Tuple = (),
    xtol: float = 1e-7,
    maxiter: int = 100,
    full_output: bool = False,
) -> Union[Array, dict]:
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


class MatrixATADSolver:
    r"""Solver for linear system involving a symmetric product.

    Solve a linear system of the form

    .. math::

       (A^T W A + D) \mb{x} = \mb{b}

    or

    .. math::

       (A^T W A + D) X = B \;,

    where :math:`A \in \mbb{R}^{M \times N}`,
    :math:`W \in \mbb{R}^{M \times M}` and
    :math:`D \in \mbb{R}^{N \times N}`. :math:`A` must be an instance of
    :class:`.MatrixOperator` or an array; :math:`D` must be an instance
    of :class:`.MatrixOperator`, :class:`.Diagonal`, or an array, and
    :math:`W`, if specified, must be an instance of :class:`.Diagonal`
    or an array.


    The solution is computed by factorization of matrix
    :math:`A^T W A + D` and solution via Gaussian elimination. If
    :math:`D` is diagonal and :math:`N < M` (i.e. :math:`A W A^T` is
    smaller than :math:`A^T W A`), then :math:`A W A^T + D` is factorized
    and the original problem is solved via the Woodbury matrix identity

    .. math::

       (E + U C V)^{-1} = E^{-1} - E^{-1} U (C^{-1} + V E^{-1} U)^{-1}
       V E^{-1} \;.

    Setting

    .. math::

       E &= D \\
       U &= A^T \\
       C &= W \\
       V &= A

    we have

    .. math::

       (D + A^T W A)^{-1} = D^{-1} - D^{-1} A^T (W^{-1} + A D^{-1} A^T)^{-1} A
       D^{-1}

    which can be simplified to

    .. math::

       (D + A^T W A)^{-1} = D^{-1} (I - A^T G^{-1} A D^{-1})

    by defining :math:`G = W^{-1} + A D^{-1} A^T`. We therefore have that

    .. math::

       \mb{x} = (D + A^T W A)^{-1} \mb{b} = D^{-1} (I - A^T G^{-1} A
       D^{-1}) \mb{b} \;.

    If we have a Cholesky factorization of :math:`G`, e.g.
    :math:`G = L L^T`, we can define

    .. math::

       \mb{w} = G^{-1} A D^{-1} \mb{b}

    so that

    .. math::

       G \mb{w} &= A D^{-1} \mb{b} \\
       L L^T \mb{w} &= A D^{-1} \mb{b} \;.

    The Cholesky factorization can be exploited by solving for
    :math:`\mb{z}` in

    .. math::

       L \mb{z} = A D^{-1} \mb{b}

    and then for :math:`\mb{w}` in

    .. math::

       L^T \mb{w} = \mb{z} \;,

    so that

    .. math::

       \mb{x} = D^{-1} \mb{b} - D^{-1} A^T \mb{w} \;.

    (Functions :func:`~jax.scipy.linalg.cho_solve` and
    :func:`~jax.scipy.linalg.lu_solve` allow direct solution for
    :math:`\mb{w}` without the two-step procedure described here.) A
    Cholesky factorization should only be used when :math:`G` is
    positive-definite (e.g. :math:`D` is diagonal and positive); if not,
    an LU factorization should be used.

    Complex-valued problems are also supported, in which case the
    transpose :math:`\cdot^T` in the equations above should be taken to
    represent the conjugate transpose.

    To solve problems directly involving a matrix of the form
    :math:`A W A^T + D`, initialize with :code:`A.T` (or
    :code:`A.T.conj()` for complex problems) instead of :code:`A`.
    """

    def __init__(
        self,
        A: Union[MatrixOperator, Array],
        D: Union[MatrixOperator, Diagonal, Array],
        W: Optional[Union[Diagonal, Array]] = None,
        cho_factor: bool = False,
        lower: bool = False,
        check_finite: bool = True,
    ):
        r"""
        Args:
            A: Matrix :math:`A`.
            D: Matrix :math:`D`. If a 2D array or :class:`MatrixOperator`,
                specifies the 2D matrix :math:`D`. If 1D array or
                :class:`Diagonal`, specifies the diagonal elements
                of :math:`D`.
            W: Matrix :math:`W`. Specifies the diagonal elements of
                :math:`W`. Defaults to an array with unit entries.
            cho_factor: Flag indicating whether to use Cholesky
                (``True``) or LU (``False``) factorization.
            lower: Flag indicating whether lower (``True``) or upper
                (``False``) triangular factorization should be computed.
                Only relevant to Cholesky factorization.
            check_finite: Flag indicating whether the input array should
                be checked for ``Inf`` and ``NaN`` values.
        """
        A = jnp.array(A)

        if isinstance(D, Diagonal):
            D = D.diagonal
            if D.ndim != 1:
                raise ValueError("If Diagonal, 'D' should have a 1D diagonal.")
        else:
            D = jnp.array(D)
            if not D.ndim in [1, 2]:
                raise ValueError("If array or MatrixOperator, 'D' should be 1D or 2D.")

        if W is None:
            W = snp.ones(A.shape[0], dtype=A.dtype)
        elif isinstance(W, Diagonal):
            W = W.diagonal
            assert hasattr(W, "ndim")
            if W.ndim != 1:
                raise ValueError("If Diagonal, 'W' should have a 1D diagonal.")
        elif not isinstance(W, Array):
            raise TypeError(
                f"Operator 'W' is required to be None, a Diagonal, or an array; got a {type(W)}."
            )

        self.A = A
        self.D = D
        self.W = W
        self.cho_factor = cho_factor
        self.lower = lower
        self.check_finite = check_finite

        assert isinstance(W, Array)
        N, M = A.shape
        if N < M and D.ndim == 1:
            G = snp.diag(1.0 / W) + A @ (A.T.conj() / D[:, snp.newaxis])
        else:
            if D.ndim == 1:
                G = A.T.conj() @ (W[:, snp.newaxis] * A) + snp.diag(D)
            else:
                G = A.T.conj() @ (W[:, snp.newaxis] * A) + D

        if cho_factor:
            c, lower = jsl.cho_factor(G, lower=lower, check_finite=check_finite)
            self.factor = (c, lower)
        else:
            lu, piv = jsl.lu_factor(G, check_finite=check_finite)
            self.factor = (lu, piv)

    def solve(self, b: Array, check_finite: Optional[bool] = None) -> Array:
        r"""Solve the linear system.

        Solve the linear system with right hand side :math:`\mb{b}` (`b`
        is a vector) or :math:`B` (`b` is a 2d array).

        Args:
           b: Vector :math:`\mathbf{b}` or matrix :math:`B`.
           check_finite: Flag indicating whether the input array should
               be checked for ``Inf`` and ``NaN`` values. If ``None``,
               use the value selected on initialization.

        Returns:
          Solution to the linear system.
        """
        if check_finite is None:
            check_finite = self.check_finite
        if self.cho_factor:
            fact_solve = lambda x: jsl.cho_solve(self.factor, x, check_finite=check_finite)
        else:
            fact_solve = lambda x: jsl.lu_solve(self.factor, x, trans=0, check_finite=check_finite)

        if b.ndim == 1:
            D = self.D
        else:
            D = self.D[:, snp.newaxis]
        N, M = self.A.shape
        if N < M and self.D.ndim == 1:
            w = fact_solve(self.A @ (b / D))
            x = (b - (self.A.T.conj() @ w)) / D
        else:
            x = fact_solve(b)

        return x

    def accuracy(self, x: Array, b: Array) -> float:
        r"""Compute solution relative residual.

        Args:
           x: Array :math:`\mathbf{x}` (solution).
           b: Array :math:`\mathbf{b}` (right hand side of linear system).

        Returns:
           Relative residual of solution.
        """
        if b.ndim == 1:
            D = self.D
        else:
            D = self.D[:, snp.newaxis]
        assert isinstance(self.W, Array)
        return rel_res(self.A.T.conj() @ (self.W[:, snp.newaxis] * self.A) @ x + D * x, b)


class ConvATADSolver:
    r"""Solver for a linear system involving a sum of convolutions.

    Solve a linear system of the form

    .. math::

       (A^H A + D) \mb{x} = \mb{b}

    where :math:`A` is a block-row operator with circulant blocks, i.e. it
    can be written as

    .. math::

       A = \left( \begin{array}{cccc} A_1 & A_2 & \ldots & A_{K}
           \end{array} \right) \;,

    where all of the :math:`A_k` are circular convolution operators, and
    :math:`D` is a circular convolution operator. This problem is most
    easily solved in the DFT transform domain, where the circular
    convolutions become diagonal operators. Denoting the frequency-domain
    versions of variables with a circumflex (e.g. :math:`\hat{\mb{x}}` is
    the frequency-domain version of :math:`\mb{x}`), the the problem can
    be written as

    .. math::

       (\hat{A}^H \hat{A} + \hat{D}) \hat{\mb{x}} = \hat{\mb{b}} \;,

    where

    .. math::

       \hat{A} = \left( \begin{array}{cccc} \hat{A}_1 & \hat{A}_2 &
       \ldots & \hat{A}_{K} \end{array} \right) \;,

    and :math:`\hat{D}` and all the :math:`\hat{A}_k` are diagonal
    operators.

    This linear equation is computational expensive to solve because
    the left hand side includes the term :math:`\hat{A}^H \hat{A}`,
    which corresponds to the outer product of :math:`\hat{A}^H`
    and :math:`\hat{A}`. A computationally efficient solution is possible,
    however, by exploiting the Woodbury matrix identity
    :cite:`wohlberg-2014-efficient`

    .. math::

       (B + U C V)^{-1} = B^{-1} - B^{-1} U (C^{-1} + V B^{-1} U)^{-1}
       V B^{-1} \;.

    Setting

    .. math::

       B &= \hat{D} \\
       U &= \hat{A}^H \\
       C &= I \\
       V &= \hat{A}

    we have

    .. math::

       (\hat{D} + \hat{A}^H \hat{A})^{-1} = \hat{D}^{-1} - \hat{D}^{-1}
       \hat{A}^H (I + \hat{A} \hat{D}^{-1} \hat{A}^H)^{-1} \hat{A}
       \hat{D}^{-1}

    which can be simplified to

    .. math::

       (\hat{D} + \hat{A}^H \hat{A})^{-1} = \hat{D}^{-1} (I - \hat{A}^H
       \hat{E}^{-1} \hat{A} \hat{D}^{-1})

    by defining :math:`\hat{E} = I + \hat{A} \hat{D}^{-1} \hat{A}^H`. The
    right hand side is much cheaper to compute because the only matrix
    inversions involve :math:`\hat{D}`, which is diagonal, and
    :math:`\hat{E}`, which is a weighted inner product of
    :math:`\hat{A}^H` and :math:`\hat{A}`.
    """

    def __init__(self, A: ComposedLinearOperator, D: CircularConvolve):
        r"""
        Args:
            A: Operator :math:`A`.
            D: Operator :math:`D`.
        """
        if not isinstance(A, ComposedLinearOperator):
            raise TypeError(
                f"Operator 'A' is required to be a ComposedLinearOperator; got a {type(A)}."
            )
        if not isinstance(A.A, Sum) or not isinstance(A.B, CircularConvolve):
            raise TypeError(
                "Operator 'A' is required to be a composition of Sum and CircularConvolve"
                f"linear operators; got a composition of {type(A.A)} and {type(A.B)}."
            )

        self.A = A
        self.D = D
        self.sum_axis = A.A.kwargs["axis"]
        if not isinstance(self.sum_axis, int):
            raise ValueError(
                "Sum component of operator 'A' must sum over a single axis of its input."
            )
        self.fft_axes = A.B.x_fft_axes
        self.real_result = is_real_dtype(D.input_dtype)

        Ahat = A.B.h_dft
        Dhat = D.h_dft
        self.AHEinv = Ahat.conj() / (
            1.0 + snp.sum(Ahat * (Ahat.conj() / Dhat), axis=self.sum_axis, keepdims=True)
        )

    def solve(self, b: Array) -> Array:
        r"""Solve the linear system.

        Solve the linear system with right hand side :math:`\mb{b}`.

        Args:
           b: Array :math:`\mathbf{b}`.

        Returns:
          Solution to the linear system.
        """
        assert isinstance(self.A.B, CircularConvolve)

        Ahat = self.A.B.h_dft
        Dhat = self.D.h_dft
        bhat = snp.fft.fftn(b, axes=self.fft_axes)
        xhat = (
            bhat - (self.AHEinv * (snp.sum(Ahat * bhat / Dhat, axis=self.sum_axis, keepdims=True)))
        ) / Dhat
        x = snp.fft.ifftn(xhat, axes=self.fft_axes)
        if self.real_result:
            x = x.real

        return x

    def accuracy(self, x: Array, b: Array) -> float:
        r"""Compute solution relative residual.

        Args:
           x: Array :math:`\mathbf{x}` (solution).
           b: Array :math:`\mathbf{b}` (right hand side of linear system).

        Returns:
           Relative residual of solution.
        """
        return rel_res(self.A.gram_op(x) + self.D(x), b)
