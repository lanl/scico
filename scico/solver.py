# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Optimization algorithms.

.. todo::
    Add motivation for this module; when to choose over jax optimizers

"""


from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.typing import BlockShape, JaxArray, Shape
from scipy import optimize as spopt

__author__ = """Luke Pfister <pfister@lanl.gov>"""


def _wrap_func_and_grad(func: Callable, shape: Union[Shape, BlockShape]):
    """Computes function evaluation and gradient for use in :mod:`scipy.optimize` functions.

    Reshapes the input to ``func`` to have ``shape``.  Evaluates ``func`` and computes gradient.
    Ensures the returned ``grad`` is an ndarray.

    Args:
        func: The function to minimize.
        shape: Shape of input to ``func``.
    """

    # argnums=0 ensures only differentiate func wrt first argument,
    #   in case func signature is func(x, *args)
    val_grad_func = jax.jit(jax.value_and_grad(func, argnums=0))

    @wraps(func)
    def wrapper(x, *args):
        # apply val_grad_func to un-vectorized input
        val, grad = val_grad_func(snp.reshape(x, shape), *args)

        # Convert val & grad into numpy arrays (.copy()), then cast to float
        # Convert 'val' into a scalar, rather than ndarray of shape (1,)
        # TODO can this be relaxed to float32?
        val = val.copy().astype(float).item()
        grad = grad.copy().astype(float).ravel()
        return val, grad

    return wrapper


def split_real_imag(x: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
    """Splits an array of shape (N,M,...) into real and imaginary parts.

    Args:
        x:  Array to split.

    Returns:
        A real ndarray with stacked real/imaginary parts.  If ``x`` has shape
        (M, N, ...), the returned array will have shape (2, M, N, ...)
        where the first slice contains the ``x.real`` and the second contains
        ``x.imag``.  If `x` is a BlockArray, this function is called on each block
        and the output is joined into a BlockArray.
    """
    if isinstance(x, BlockArray):
        return BlockArray.array([split_real_imag(_) for _ in x])
    else:
        return snp.stack((snp.real(x), snp.imag(x)))


def join_real_imag(x: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
    """Join a real array of shape (2,N,M,...) into a complex array of length (N,M, ...)"""
    if isinstance(x, BlockArray):
        return BlockArray.array([join_real_imag(_) for _ in x])
    else:
        return x[0] + 1j * x[1]


# TODO:  Use jax to compute Hessian-vector products for use in Newton methods
# see https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Hessian-vector-products-using-both-forward--and-reverse-mode
# for examples of constructing Hessians in jax


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
    """Minimization of scalar function of one or more variables. Wrapper around
    :func:`scipy.optimize.minimize`.

    This function differs from :func:`scipy.optimize.minimize` in three ways:

        - The ``jac`` options of  :func:`scipy.optimize.minimize` are not supported. The gradient is calculated using ``jax.grad``.
        - Functions mapping from N-dimensional arrays -> float are supported
        - Functions mapping from complex arrays -> float are supported.

    Docstring for :func:`scipy.optimize.minimize` follows. For descriptions of
    the optimization methods and custom minimizers, refer to the original
    docstring for :func:`scipy.optimize.minimize`.

    Args:
        func:  The objective function to be minimized.

            ``func(x, *args) -> float``

            where ``x`` is an array and ``args`` is a tuple of the fixed parameters
            needed to completely specify the function.  Unlike
            :func:`scipy.optimize.minimize`, ``x`` need not be a 1D array.
        x0: Initial guess.  If ``func`` is a mapping from complex arrays to floats,
            x0 must have a complex data type.
        args: Extra arguments passed to the objective function and `hess`.
        method: Type of solver.  Should be one of:

            - 'Nelder-Mead' `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>`__
            - 'Powell'      `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html>`__
            - 'CG'          `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html>`__
            - 'BFGS'        `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html>`__
            - 'Newton-CG'   `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html>`__
            - 'L-BFGS-B'    `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`__
            - 'TNC'         `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-tnc.html>`__
            - 'COBYLA'      `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html>`__
            - 'SLSQP'       `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html>`__
            - 'trust-constr'`(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`__
            - 'dogleg'      `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-dogleg.html>`__
            - 'trust-ncg'   `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustncg.html>`__
            - 'trust-exact' `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustexact.html>`__
            - 'trust-krylov' `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustkrylov.html>`__
            - custom - a callable object (added in version SciPy 0.14.0), see :func:`scipy.optimize.minmize_scalar`.

            If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
            depending if the problem has constraints or bounds.

        hess: Method for computing the Hessian matrix. Only for Newton-CG, dogleg,
            trust-ncg,  trust-krylov, trust-exact and trust-constr. If it is
            callable, it should return the  Hessian matrix:

                ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

            where x is a (n,) ndarray and `args` is a tuple with the fixed
            parameters. LinearOperator and sparse matrix returns are
            allowed only for 'trust-constr' method. Alternatively, the keywords
            {'2-point', '3-point', 'cs'} select a finite difference scheme
            for numerical estimation. Or, objects implementing
            `HessianUpdateStrategy` interface can be used to approximate
            the Hessian. Available quasi-Newton methods implementing
            this interface are:

                - `BFGS`;
                - `SR1`.

            Whenever the gradient is estimated via finite-differences,
            the Hessian cannot be estimated with options
            {'2-point', '3-point', 'cs'} and needs to be
            estimated using one of the quasi-Newton strategies.
            Finite-difference options {'2-point', '3-point', 'cs'} and
            `HessianUpdateStrategy` are available only for 'trust-constr' method.
            NOTE:  In the future, `hess` may be determined using jax.
        hessp: Hessian of objective function times an arbitrary vector p.
            Only for Newton-CG, trust-ncg, trust-krylov, trust-constr.
            Only one of `hessp` or `hess` needs to be given.  If `hess` is
            provided, then `hessp` will be ignored.  `hessp` must compute the
            Hessian times an arbitrary vector:

                ``hessp(x, p, *args) ->  array``

            where x is a ndarray, p is an arbitrary vector with
            dimension equal to x, and `args` is a tuple with the fixed parameters.
        bounds (None, optional): Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and
            trust-constr methods. There are two ways to specify the bounds:

                1. Instance of `Bounds` class.
                2. Sequence of ``(min, max)`` pairs for each element in `x`. None
                    is used to specify no bound.

        constraints: Constraints definition (only for COBYLA, SLSQP and trust-constr).
            Constraints for 'trust-constr' are defined as a single object or a
            list of objects specifying constraints to the optimization problem.

            Available constraints are:

                - `LinearConstraint`
                - `NonlinearConstraint`

            Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
            Each dictionary with fields:

                type : str
                    Constraint type: 'eq' for equality, 'ineq' for inequality.
                fun : callable
                    The function defining the constraint.
                jac : callable, optional
                    The Jacobian of `fun` (only for SLSQP).
                args : sequence, optional
                    Extra arguments to be passed to the function and Jacobian.

            Equality constraint means that the constraint function result is to
            be zero whereas inequality means that it is to be non-negative.
            Note that COBYLA only supports inequality constraints.

        tol: Tolerance for termination.  For detailed control, use solver-specific options.
        callback:  Called after each iteration. For 'trust-constr' it is a callable with
            the signature:

                ``callback(xk, OptimizeResult state) -> bool``

            where ``xk`` is the current parameter vector. and ``state``
            is an `OptimizeResult` object, with the same fields
            as the ones from the return. If callback returns True
            the algorithm execution is terminated.
            For all the other methods, the signature is:

                ``callback(xk)``

            where ``xk`` is the current parameter vector.
        options: A dictionary of solver options. All methods accept the following
            generic options:

                maxiter : int
                    Maximum number of iterations to perform.
                disp : bool
                    Set to True to print convergence messages.

            See :func:`scipy.optimize.show_options()` for solver-specific options.

    """

    if snp.iscomplexobj(x0):
        # scipy minimize function requires real-valued arrays, so
        # we split x0 into a vector with real/imaginary parts stacked
        # and compose `func` with a `join_real_imag`
        iscomplex = True
        func_ = lambda x: func(join_real_imag(x))
        x0 = split_real_imag(x0)
    else:
        iscomplex = False
        func_ = func

    x0_shape = x0.shape
    x0 = x0.ravel()  # If x0 is a BlockArray it will become a DeviceArray here
    if isinstance(x0, jax.interpreters.xla.DeviceArray):
        dev = x0.device_buffer.device()  # device where x0 resides; used to put result back in place
        x0 = x0.copy().astype(float)
    else:
        dev = None

    # Run the SciPy minimizer
    res = spopt.minimize(
        _wrap_func_and_grad(func_, x0_shape),
        x0=x0,
        args=args,
        jac=True,
        method=method,
        options=options,
    )

    # TODO: need tests for multi-gpu machines
    # un-vectorize the output array, put on device
    res.x = snp.reshape(
        res.x, x0_shape
    )  # If x0 was originally a BlockArray be converted back to one here
    if dev:
        res.x = jax.device_put(res.x, dev)

    if iscomplex:
        res.x = join_real_imag(res.x)
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

    """Minimization of scalar function of one variable. Wrapper around
    :func:`scipy.optimize.minimize_scalar`.

    Docstring for :func:`scipy.optimize.minimize_scalar` follows.
    For descriptions of the optimization methods and custom minimizers, refer to the original
    docstring for :func:`scipy.optimize.minimize_scalar`.

    Args:
        func: Objective function.  Scalar function, must return a scalar.
        bracket: For methods 'brent' and 'golden', `bracket` defines the bracketing
            interval and can either have three items ``(a, b, c)`` so that
            ``a < b < c`` and ``fun(b) < fun(a), fun(c)`` or two items ``a`` and
            ``c`` which are assumed to be a starting interval for a downhill
            bracket search (see `bracket`); it doesn't always mean that the
            obtained solution will satisfy ``a <= x <= c``.
        bounds: For method 'bounded', `bounds` is mandatory and must have two items
            corresponding to the optimization bounds.
        args:  Extra arguments passed to the objective function.
        method: Type of solver.  Should be one of:

            - 'Brent'     `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize_scalar-brent.html>`__
            - 'Bounded'    `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize_scalar-bounded.html>`__
            - 'Golden'     `(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize_scalar-golden.html>`__
            - custom - a callable object (added in SciPy version 0.14.0), see :func:`scipy.optimize.minmize_scalar`.


        tol:  Tolerance for termination. For detailed control, use solver-specific
            options.
        options: A dictionary of solver options.
                maxiter : int
                    Maximum number of iterations to perform.
                disp : bool
                    Set to True to print convergence messages.

            See :func:`scipy.optimize.show_options()` for solver-specific options.

    Returns:
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        :class:`scipy.optimize.OptimizeResult` for a description of other attributes.

    """

    def f(x, *args):
        # Wrap jax-based function ``func`` to return a numpy float
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
) -> Union[JaxArray, dict]:
    r"""Conjugate Gradient solver.

    Solve the linear system :math:`A\mb{x} = \mb{b}` via the conjugate
    gradient method.

    Args:
        A: Function implementing linear operator :math:`A`
        b: Input array :math:`\mb{b}`
        x0: Initial solution
        tol: Relative residual stopping tolerance. Default: 1e-5
           Convergence occurs when ``norm(residual) <= max(tol * norm(b), atol)``.
        atol : Absolute residual stopping tolerance.  Default: 0.0
           Convergence occurs when ``norm(residual) <= max(tol * norm(b), atol)``
        maxiter: Maximum iterations.  Default: 1000
        M: Preconditioner for A.  The preconditioner should approximate the
           inverse of ``A``.  The default, ``None``, uses no preconditioner.

    Returns:
        tuple: A tuple (x, info) containing:

            - **x** : Solution array
            - **info**: Dictionary containing diagnostic information
    """

    if M is None:
        M = lambda x: x

    x = x0
    Ax = A(x0)
    bn = snp.linalg.norm(b)
    r = b - Ax
    z = M(r)
    p = z
    num = r.ravel().conj().T @ z.ravel()
    ii = 0

    # termination tolerance
    # uses the "non-legacy" form of scicpy.sparse.linalg.cg
    termination_tol_sq = snp.maximum(tol * bn, atol) ** 2

    while (ii < maxiter) and (num > termination_tol_sq):
        Ap = A(p)
        alpha = num / (p.ravel().conj().T @ Ap.ravel())
        x = x + alpha * p
        r = r - alpha * Ap
        z = M(r)
        num_old = num
        num = r.ravel().conj().T @ z.ravel()
        beta = num / num_old
        p = z + beta * p
        ii += 1

    return (x, {"num_iter": ii, "rel_res": snp.sqrt(num) / bn})
