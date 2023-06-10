# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 by SCICO Developers
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

import scico.linop
import scico.numpy as snp
import scipy.linalg as spl
from scico.metric import rel_res
from scico.numpy import Array, BlockArray
from scico.numpy.util import is_real_dtype
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
    x0 = x0.ravel()  # if x0 is a BlockArray it will become a jax array here

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
        # Wrap jax-based function `func` to return a numpy float rather
        # than a jax array of size (1,)
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
        if isinstance(A, scico.linop.LinearOperator):
            x0 = snp.zeros(A.input_shape, b.dtype)
        else:
            raise ValueError("Parameter x0 must be specified if A is not a LinearOperator")

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
        \argmin_{\mb{x}} \; (1/2) \norm{ A \mb{x} - \mb{b}) }_2^2 \;,

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
    if isinstance(A, scico.linop.LinearOperator):
        Aop = A
    else:
        assert x0 is not None
        Aop = scico.linop.LinearOperator(
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


class SolveATAI:
    r"""Solver for linear system involving a matrix :math:`A^T A + \alpha I`        .

    Solve a linear system of the form

    .. math::

       (A^T A + \alpha I) \mb{x} = \mb{b}

    or

    .. math::

       (A^T A + \alpha I) X = B \;,

    where :math:`A \in \mbb{R}^{M \times N}`. The solution is computed by
    factorizing the matrix :math:`A^T A + \alpha I` or
    :math:`A A^T + \alpha I`, depending on which is smaller. If it is the
    latter, the matrix inversion lemma is used to solve the linear system.
    """

    def __init__(
        self,
        A: Union[scico.linop.MatrixOperator, Array],
        alpha: float,
        cho_factor: bool = True,
        lower: bool = False,
        check_finite: bool = True,
    ):
        r"""
        Args:
            A: Matrix :math:`A`.
            alpha: Scalar :math:`\alpha`.
            cho_factor: Flag indicating whether to use Cholesky
                (``True``) or LU (``False``) factorization.
            lower: Flag indicating whether lower (``True``) or upper
                (``False``) triangular factorization should be computed.
                Only relevant to Cholesky factorization.
            check_finite: Flag indicating whether the input array should
                be checked for ``Inf`` and ``NaN`` values.
        """
        if isinstance(A, scico.linop.MatrixOperator):
            A = A.to_array()
        self.A = A
        self.alpha = alpha
        self.cho_factor = cho_factor
        self.lower = lower
        self.check_finite = check_finite

        N, M = A.shape
        # If N < M it is cheaper to factorise A*A^T + alpha*I and then use the
        # matrix inversion lemma to compute the inverse of A^T*A + alpha*I
        if N >= M:
            B = A.T @ A + alpha * np.identity(M, dtype=A.dtype)
        else:
            B = A @ A.T + alpha * np.identity(N, dtype=A.dtype)

        if cho_factor:
            c, lower = spl.cho_factor(B, lower=lower, check_finite=check_finite)
            self.factor = (c, lower)
        else:
            lu, piv = spl.lu_factor(B, check_finite=check_finite)
            self.factor = (lu, piv)

    def solve(self, b: Array, check_finite: bool = None) -> Array:
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
            fact_solve = lambda x, t: spl.cho_solve(self.factor, x, check_finite=check_finite)
        else:
            fact_solve = lambda x, t: spl.lu_solve(
                self.factor, x, trans=t, check_finite=check_finite
            )
        N, M = self.A.shape
        if N >= M:
            x = fact_solve(b, 0)
        else:
            x = (b - self.A.T @ fact_solve(self.A @ b, 1)) / self.alpha
        return x


class SolveConvATAD:
    r"""Solver for sum of convolutions linear system`        .

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

    def __init__(self, A: scico.linop.ComposedLinearOperator, D: scico.linop.CircularConvolve):
        r"""
        Args:
            A: Operator :math:`A`.
            D: Operator :math:`D`.
        """
        if not isinstance(A.A, scico.linop.Sum) or not isinstance(
            A.B, scico.linop.CircularConvolve
        ):
            raise TypeError(
                "Operator A is required to be a composition of Sum and CircularConvolve"
                f"linear operators; got a composition of {type(A.A)} and {type(A.B)}."
            )

        self.A = A
        self.D = D
        self.sum_axis = A.A.kwargs["axis"]
        self.fft_axes = A.B.x_fft_axes
        self.real_result = is_real_dtype(D.input_dtype)
        self.accuracy: Optional[float] = None

        Ahat = A.B.h_dft
        Dhat = D.h_dft
        self.AHEinv = Ahat.conj() / (
            1.0 + snp.sum(Ahat * (Ahat.conj() / Dhat), axis=self.sum_axis, keepdims=True)
        )

    def solve(self, b: Array, check_solve: bool = False) -> Array:
        r"""Solve the linear system.

        Solve the linear system with right hand side :math:`\mb{b}`.

        Args:
           b: Array :math:`\mathbf{b}`.
           check_solve: Flag indicating whether the solution accuracy
               should be computed.

        Returns:
          Solution to the linear system.
        """
        assert isinstance(self.A.B, scico.linop.CircularConvolve)

        Ahat = self.A.B.h_dft
        Dhat = self.D.h_dft
        bhat = snp.fft.fftn(b, axes=self.fft_axes)
        xhat = (
            bhat - (self.AHEinv * (snp.sum(Ahat * bhat / Dhat, axis=self.sum_axis, keepdims=True)))
        ) / Dhat
        x = snp.fft.ifftn(xhat, axes=self.fft_axes)
        if self.real_result:
            x = x.real

        if check_solve:
            lhs = self.A.gram_op(x) + self.D(x)
            self.accuracy = rel_res(lhs, b)
        else:
            self.accuracy = None

        return x
