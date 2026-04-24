# Copyright (C) 2020-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Automatic differentiation tools."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import scico.util


def _append_jax_docs(fn, jaxfn=None):
    """Append the jax function docs.

    Given wrapper function `fn`, concatenate its docstring with the
    docstring of the wrapped jax function.
    """

    name = fn.__name__
    if jaxfn is None:
        jaxfn = getattr(jax, name)
    doc = "  " + fn.__doc__.replace("\n    ", "\n  ")  # deal with indentation differences
    jaxdoc = "\n".join(jaxfn.__doc__.split("\n")[2:])  # strip initial lines
    return doc + f"\n  Docstring for :func:`jax.{name}`:\n\n" + jaxdoc


def grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable:
    """Create a function that evaluates the gradient of `fun`.

    :func:`scico.grad` differs from :func:`jax.grad` in that the output
    is conjugated.
    """

    jax_grad = jax.grad(
        fun=fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic, allow_int=allow_int
    )

    def conjugated_grad_aux(*args, **kwargs):
        jg, aux = jax_grad(*args, **kwargs)
        return tree_map(jax.numpy.conj, jg), aux

    def conjugated_grad(*args, **kwargs):
        jg = jax_grad(*args, **kwargs)
        return tree_map(jax.numpy.conj, jg)

    return conjugated_grad_aux if has_aux else conjugated_grad


# Append docstring from original jax function
grad.__doc__ = _append_jax_docs(grad)


def value_and_grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable[..., Tuple[Any, Any]]:
    """Create a function that evaluates both `fun` and its gradient.

    :func:`scico.value_and_grad` differs from :func:`jax.value_and_grad`
    in that the gradient is conjugated.
    """
    jax_val_grad = jax.value_and_grad(
        fun=fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic, allow_int=allow_int
    )

    def conjugated_value_and_grad_aux(*args, **kwargs):
        (value, aux), jg = jax_val_grad(*args, **kwargs)
        conj_grad = tree_map(jax.numpy.conj, jg)
        return (value, aux), conj_grad

    def conjugated_value_and_grad(*args, **kwargs):
        value, jax_grad = jax_val_grad(*args, **kwargs)
        conj_grad = tree_map(jax.numpy.conj, jax_grad)
        return value, conj_grad

    return conjugated_value_and_grad_aux if has_aux else conjugated_value_and_grad


# Append docstring from original jax function
value_and_grad.__doc__ = _append_jax_docs(value_and_grad)


def linear_adjoint(fun: Callable, *primals) -> Callable:
    """Conjugate transpose a function that is guaranteed to be linear.

    :func:`scico.linear_adjoint` differs from :func:`jax.linear_transpose`
    for complex inputs in that the conjugate transpose (adjoint) of `fun`
    is returned. :func:`scico.linear_adjoint` is identical to
    :func:`jax.linear_transpose` for real-valued primals.
    """

    def conj_fun(*primals):
        conj_primals = tree_map(jax.numpy.conj, primals)
        return tree_map(jax.numpy.conj, fun(*(conj_primals)))

    if any([jnp.iscomplexobj(_) for _ in primals]):
        # fun is C->R or C->C
        _primals = tree_map(jax.numpy.conj, primals)
        _fun = conj_fun
    elif jnp.iscomplexobj(fun(*primals)):
        # fun is from R -> C
        _primals = primals
        _fun = conj_fun
    else:
        # fun is R->R
        _fun = fun
        _primals = primals
    return jax.linear_transpose(_fun, *_primals)


# Append docstring from original jax function
linear_adjoint.__doc__ = _append_jax_docs(linear_adjoint, jaxfn=jax.linear_transpose)


def jacrev(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable:
    """Jacobian of `fun` evaluated row-by-row using reverse-mode AD.

    :func:`scico.jacrev` differs from :func:`jax.jacrev` in that the
    output is conjugated.
    """

    jax_jacrev = jax.jacrev(fun=fun, argnums=argnums, holomorphic=holomorphic, allow_int=allow_int)

    def conjugated_jacrev(*args, **kwargs):
        tmp = jax_jacrev(*args, **kwargs)
        return tree_map(jax.numpy.conj, tmp)

    return conjugated_jacrev


# Append docstring from original jax function
jacrev.__doc__ = _append_jax_docs(jacrev)


def cvjp(fun: Callable, *primals, jidx: Optional[int] = None) -> Tuple[Tuple[Any, ...], Callable]:
    r"""Compute a vector-Jacobian product with conjugate transpose.

    Compute the product :math:`[J(\mb{x})]^H \mb{v}` where
    :math:`[J(\mb{x})]` is the Jacobian of function `fun` evaluated at
    :math:`\mb{x}`. Instead of directly evaluating the product, a
    function is returned that takes :math:`\mb{v}` as an argument. If
    `fun` has multiple positional parameters, the Jacobian can be taken
    with respect to only one of them by setting the `jidx` parameter of
    this function to the positional index of that parameter.

    Args:
        fun: Function for which the Jacobian is implicitly computed.
        primals: Sequence of values at which the Jacobian is
           evaluated, with length equal to the number of positional
           arguments of `fun`.
        jidx: Index of the positional parameter of `fun` with respect
           to which the Jacobian is taken.

    Returns:
        A pair `(primals_out, conj_vjp)` where `primals_out` is the
        output of `fun` evaluated at `primals`, i.e. `primals_out
        = fun(*primals)`, and `conj_vjp` is a function that computes the
        product of the conjugate (Hermitian) transpose of the Jacobian of
        `fun` and its argument. If the `jidx` parameter is an integer,
        then the Jacobian is only taken with respect to the coresponding
        positional parameter of `fun`.
    """

    if jidx is None:
        primals_out, fun_vjp = jax.vjp(fun, *primals)
    else:
        fixidx = tuple(range(0, jidx)) + tuple(range(jidx + 1, len(primals)))
        fixprm = primals[0:jidx] + primals[jidx + 1 :]
        pfun = scico.util.partial(fun, fixidx, *fixprm)
        primals_out, fun_vjp = jax.vjp(pfun, primals[jidx])

    def conj_vjp(tangent):
        return jax.tree_util.tree_map(jax.numpy.conj, fun_vjp(tangent.conj()))

    return primals_out, conj_vjp
