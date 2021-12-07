# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Automatic differentiation tools."""


from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp


def _append_jax_docs(fn, jaxfn=None):
    """Append the jax function docs.

    Given wrapper function ``fn``, concatenate its docstring with the
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
    """Create a function that evaluates the gradient of ``fun``.

    :func:`scico.grad` differs from :func:`jax.grad` in that the output
    is conjugated.
    """

    jax_grad = jax.grad(
        fun=fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic, allow_int=allow_int
    )

    def conjugated_grad_aux(*args, **kwargs):
        jg, aux = jax_grad(*args, **kwargs)
        return jax.tree_map(jax.numpy.conj, jg), aux

    def conjugated_grad(*args, **kwargs):
        jg = jax_grad(*args, **kwargs)
        return jax.tree_map(jax.numpy.conj, jg)

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
    """Create a function that evaluates both ``fun`` and its gradient.

    :func:`scico.value_and_grad` differs from :func:`jax.value_and_grad`
    in that the gradient is conjugated.
    """
    jax_val_grad = jax.value_and_grad(
        fun=fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic, allow_int=allow_int
    )

    def conjugated_value_and_grad_aux(*args, **kwargs):
        (value, aux), jg = jax_val_grad(*args, **kwargs)
        conj_grad = jax.tree_map(jax.numpy.conj, jg)
        return (value, aux), conj_grad

    def conjugated_value_and_grad(*args, **kwargs):
        value, jax_grad = jax_val_grad(*args, **kwargs)
        conj_grad = jax.tree_map(jax.numpy.conj, jax_grad)
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
        conj_primals = jax.tree_map(jax.numpy.conj, primals)
        return jax.tree_map(jax.numpy.conj, fun(*(conj_primals)))

    if any([jnp.iscomplexobj(_) for _ in primals]):
        # fun is C->R or C->C
        _primals = jax.tree_map(jax.numpy.conj, primals)
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
    """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

    :func:`scico.jacrev` differs from :func:`jax.jacrev` in that the
    output is conjugated.
    """

    jax_jacrev = jax.jacrev(fun=fun, argnums=argnums, holomorphic=holomorphic, allow_int=allow_int)

    def conjugated_jacrev(*args, **kwargs):
        tmp = jax_jacrev(*args, **kwargs)
        return jax.tree_map(jax.numpy.conj, tmp)

    return conjugated_jacrev


# Append docstring from original jax function
jacrev.__doc__ = _append_jax_docs(jacrev)
