from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp


def grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable:
    """Creates a function which evaluates the gradient of ``fun``.

    :func:`scico.grad` differs from :func:`jax.grad` in that the output is conjugated.

    Docstring for :func:`jax.grad`:

    Args:
        fun: Function to be differentiated. Its arguments at positions specified by
            ``argnums`` should be arrays, scalars, or standard Python containers.
            Argument arrays in the positions specified by ``argnums`` must be of
            inexact (i.e., floating-point or complex) type. It
            should return a scalar (which includes arrays with shape ``()`` but not
            arrays with shape ``(1,)`` etc.)
        argnums: Optional, integer or sequence of integers. Specifies which
            positional argument(s) to differentiate with respect to (default 0).
        has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
            first element is considered the output of the mathematical function to be
            differentiated and the second element is auxiliary data. Default False.
        holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
            holomorphic. If True, inputs and outputs must be complex. Default False.
        allow_int: Optional, bool. Whether to allow differentiating with
            respect to integer valued inputs. The gradient of an integer input will
            have a trivial vector-space dtype (float0). Default False.

    Returns:
        A function with the same arguments as ``fun``, that evaluates the gradient
        of ``fun``. If ``argnums`` is an integer then the gradient has the same
        shape and type as the positional argument indicated by that integer. If
        argnums is a tuple of integers, the gradient is a tuple of values with the
        same shapes and types as the corresponding arguments. If ``has_aux`` is True
        then a pair of (gradient, auxiliary_data) is returned.

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


def value_and_grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable[..., Tuple[Any, Any]]:
    """Create a function which evaluates both ``fun`` and the gradient of ``fun``.

    :func:`scico.value_and_grad` differs from :func:`jax.value_and_grad` in that the gradient is conjugated.

    Docstring for :func:`jax.value_and_grad`:


    Args:
      fun: Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, or standard Python containers. It
        should return a scalar (which includes arrays with shape ``()`` but not
        arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
      holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
        holomorphic. If True, inputs and outputs must be complex. Default False.
      allow_int: Optional, bool. Whether to allow differentiating with
        respect to integer valued inputs. The gradient of an integer input will
        have a trivial vector-space dtype (float0). Default False.

    Returns:
      A function with the same arguments as ``fun`` that evaluates both ``fun``
      and the gradient of ``fun`` and returns them as a pair (a two-element
      tuple). If ``argnums`` is an integer then the gradient has the same shape
      and type as the positional argument indicated by that integer. If argnums is
      a sequence of integers, the gradient is a tuple of values with the same
      shapes and types as the corresponding arguments.
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


def linear_adjoint(fun: Callable, *primals) -> Callable:
    """Conjugate transpose a function that is promised to be linear.

    :func:`scico.linear_adjoint` differs from :func:`jax.linear_transpose`
    for complex inputs in that the conjugate transpose (adjoint) of `fun` is returned.
    :func:`scico.linear_adjoint` is identical to :func:`jax.linear_transpose`
    for real-valued primals.

    Docstring for :func:`jax.linear_transpose`:

    For linear functions, this transformation is equivalent to ``vjp``, but
    avoids the overhead of computing the forward pass.

    The outputs of the transposed function will always have the exact same dtypes
    as ``primals``, even if some values are truncated (e.g., from complex to
    float, or from float64 to float32). To avoid truncation, use dtypes in
    ``primals`` that match the full range of desired outputs from the transposed
    function. Integer dtypes are not supported.

    Args:
        fun: the linear function to be transposed.
        *primals: a positional argument tuple of arrays, scalars, or (nested)
            standard Python containers (tuples, lists, dicts, namedtuples, i.e.,
            pytrees) of those types used for evaluating the shape/dtype of
            ``fun(*primals)``. These arguments may be real scalars/ndarrays, but that
            is not required: only the ``shape`` and `dtype` attributes are accessed.
            See below for an example. (Note that the duck-typed objects cannot be
            namedtuples because those are treated as standard Python containers.)

    Returns:
        A callable that calculates the transpose of ``fun``. Valid input into this
        function must have the same shape/dtypes/structure as the result of
        ``fun(*primals)``. Output will be a tuple, with the same
        shape/dtypes/structure as ``primals``.

    >>> import jax
    >>> import types
    >>> import numpy as np
    >>>
    >>> f = lambda x, y: 0.5 * x - 0.5 * y
    >>> scalar = types.SimpleNamespace(shape=(), dtype=np.float32)
    >>> f_transpose = jax.linear_transpose(f, scalar, scalar)
    >>> f_transpose(1.0)
    (DeviceArray(0.5, dtype=float32), DeviceArray(-0.5, dtype=float32))
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


def jacrev(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable:
    """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

    :func:`scico.jacrev` differs from :func:`jax.jacrev` in that the output is conjugated.

    Docstring for :func:`jax.jacrev`:

    Args:
        fun: Function whose Jacobian is to be computed.
        argnums: Optional, integer or sequence of integers. Specifies which
            positional argument(s) to differentiate with respect to (default ``0``).
        holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
            holomorphic. Default False.
        allow_int: Optional, bool. Whether to allow differentiating with
            respect to integer valued inputs. The gradient of an integer input will
            have a trivial vector-space dtype (float0). Default False.

    Returns:
        A function with the same arguments as ``fun``, that evaluates the Jacobian of
        ``fun`` using reverse-mode automatic differentiation.

    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> def f(x):
    ...   return jnp.asarray(
    ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
    ...
    >>> print(jax.jacrev(f)(jnp.array([1., 2., 3.])))  # doctest: +SKIP
    [[ 1.       0.       0.     ]
    [ 0.       0.       5.     ]
    [ 0.      16.      -2.     ]
    [ 1.6209   0.       0.84147]]
    """

    jax_jacrev = jax.jacrev(fun=fun, argnums=argnums, holomorphic=holomorphic, allow_int=allow_int)

    def conjugated_jacrev(*args, **kwargs):
        tmp = jax_jacrev(*args, **kwargs)
        return jax.tree_map(jax.numpy.conj, tmp)

    return conjugated_jacrev
