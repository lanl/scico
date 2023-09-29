*****
Notes
*****

No GPU/TPU Warning
==================

JAX currently issues a warning when used on a platform without a
GPU. To disable this warning, set the environment variable
``JAX_PLATFORM_NAME=cpu`` before running Python. This warning is
suppressed by SCICO for JAX versions after 0.3.23, making use of
the environment variable unnecessary.


Debugging
=========

If difficulties are encountered in debugging jitted functions, jit can
be globally disabled by setting the environment variable
``JAX_DISABLE_JIT=1`` before running Python, as in

::

   JAX_DISABLE_JIT=1 python test_script.py


Double Precision
================

By default, JAX enforces single-precision numbers. Double precision
can be enabled in one of two ways:

1. Setting the environment variable ``JAX_ENABLE_X64=TRUE`` before
   launching Python.
2. Manually setting the ``jax_enable_x64`` flag **at program
   startup**; that is, **before** importing SCICO.

::

   from jax.config import config
   config.update("jax_enable_x64", True)
   import scico # continue as usual


For more information, see the `JAX notes on double precision <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`_.


Random Number Generation
========================

JAX implements an explicit, non-stateful pseudorandom number generator (PRNG).
The user is responsible for generating a PRNG key and mutating it each time a
new random number is generated. We recommend users read the `JAX documentation
<https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers>`_
for information on the design of JAX random number functionality.


In :mod:`scico.random` we provide convenient wrappers around several `jax.random
<https://jax.readthedocs.io/en/stable/jax.random.html>`_ routines to handle
the generation and splitting of PRNG keys.

::

   # Calls to scico.random functions always return a PRNG key
   # If no key is passed to the function, a new key is generated
   x, key = scico.random.randn((2,))
   print(x)   # [ 0.19307713 -0.52678305]

   # scico.random functions automatically split the PRNGkey and return
   # an updated key
   y, key = scico.random.randn((2,), key=key)
   print(y) # [ 0.00870693 -0.04888531]

The user is responsible for passing the PRNG key to
:mod:`scico.random` functions. If no key is passed, repeated calls to
:mod:`scico.random` functions will return the same random numbers:

::

   x, key = scico.random.randn((2,))
   print(x)   # [ 0.19307713 -0.52678305]

   # No key passed, will return the same random numbers!
   y, key = scico.random.randn((2,))
   print(y)   # [ 0.19307713 -0.52678305]



.. _non_jax_dep:

Compiled Dependency Packages
============================

The code acceleration and automatic differentiation features of JAX
are not available for some components of SCICO that are provided via
interfaces to compiled C code. When these components are used on a
platform with GPUs, the remainder of the code will run on a GPU, but
there is potential for a considerable delay due to host-GPU memory
transfers. This issue primarily affects:


Denoisers
---------

The :func:`.bm3d` and :func:`.bm4d` denoisers (and the corresponding
:class:`.BM3D` and :class:`.BM4D` pseudo-functionals) are implemented
via interfaces to the `bm3d <https://pypi.org/project/bm3d/>`__ and
`bm4d <https://pypi.org/project/bm4d/>`__ packages respectively. The
:class:`~.denoiser.DnCNN` denoiser (and the corresponding
:class:`~.functional.DnCNN` pseudo-functional) denoiser should be used
when the full benefits of JAX-based code are required.


Tomographic Projectors/Radon Transforms
---------------------------------------

Note that the tomographic projections that are frequently referred
to as Radon transforms are referred to as X-ray transforms in SCICO.
While the Radon transform is far more well-known than the X-ray
transform, which is the same as the Radon transform for projections
in two dimensions, these two transform differ in higher numbers of
dimensions, and it is the X-ray transform that is the appropriate
mathematical model for beam attenuation based imaging in three or
more dimensions.

SCICO includes three different implementations of X-ray transforms.
Of these, :class:`.linop.XRayTransform` is an integral component of
SCICO, while the other two depend on external packages.
The :class:`.xray.svmbir.XRayTransform` class is implemented
via an interface to the `svmbir
<https://svmbir.readthedocs.io/en/latest/>`__ package. The
:class:`.xray.astra.XRayTransform` class is implemented via an
interface to the `ASTRA toolbox
<https://www.astra-toolbox.com/>`__. This toolbox does provide some
GPU acceleration support, but efficiency is expected to be lower than
JAX-based code due to host-GPU memory transfers.


Automatic Differentiation Caveats
=================================


Complex Functions
-----------------

The JAX-defined gradient of a complex-valued function is a
complex-conjugated version of the usual gradient used in mathematical
optimization and computational imaging. Minimizing a function using
the JAX convention involves taking steps in the direction of the
complex conjugated gradient.

The function :func:`scico.grad` returns the expected gradient, that
is, the conjugate of the JAX gradient. For further discussion, see
this `JAX issue <https://github.com/google/jax/issues/4891>`_.

As a concrete example, consider the function :math:`f(x) =
\frac{1}{2}\norm{\mb{A} \mb{x}}_2^2` where :math:`\mb{A}` is a complex
matrix. The gradient of :math:`f` is usually given :math:`(\nabla
f)(\mb{x}) = \mb{A}^H \mb{A} \mb{x}`, where :math:`\mb{A}^H` is the
conjugate transpose of :math:`\mb{A}`. Applying :func:`jax.grad` to
:math:`f` will yield :math:`(\mb{A}^H \mb{A} \mb{x})^*`, where
:math:`\cdot^*` denotes complex conjugation.

The following code demonstrates the use of :func:`jax.grad` and
:func:`scico.grad`:


::

    m, n = (4, 3)
    A, key = randn((m, n), dtype=np.complex64, key=None)
    x, key = randn((n,), dtype=np.complex64, key=key)

    def f(x):
        return 0.5 * snp.linalg.norm(A @ x)**2

    an_grad = A.conj().T @ A @ x  # The expected gradient

    np.testing.assert_allclose(jax.grad(f)(x), an_grad.conj(), rtol=1e-4)
    np.testing.assert_allclose(scico.grad(f)(x), an_grad, rtol=1e-4)


Non-differentiable Functionals
------------------------------

:func:`scico.grad` can be applied to any function, but has undefined
behavior for non-differentiable functions. For non-differerentiable
functions, :func:`scico.grad` may or may not return a valid
subgradient. As an example, ``scico.grad(snp.abs)(0.) = 0``, which is
a valid subgradient. However, ``scico.grad(snp.linalg.norm)([0., 0.])
= [nan, nan]``.

Differentiable functions that are written as the composition of a
differentiable and non-differentiable function should be avoided. As
an example, :math:`f(x) = \norm{x}_2^2` can be implemented in as ``f =
lambda x: snp.linalg.norm(x)**2``. This involves first calculating the
non-squared :math:`\ell_2` norm, then squaring it. The un-squared
:math:`\ell_2` norm is not differentiable at zero. When evaluating
the gradient of ``f`` at 0, :func:`scico.grad` returns :data:`~numpy.NaN`:

::

   >>> import scico
   >>> import scico.numpy as snp
   >>> f = lambda x: snp.linalg.norm(x)**2
   >>> scico.grad(f)(snp.zeros(2, dtype=snp.float32))  # doctest: +SKIP
   Array([nan, nan], dtype=float32)

This can be fixed (assuming real-valued arrays only) by defining the
squared :math:`\ell_2` norm directly as ``g = lambda x: snp.sum(x**2)``.
The gradient will work as expected:

::

   >>> g = lambda x: snp.sum(x**2)
   >>> scico.grad(g)(snp.zeros(2, dtype=snp.float32))  #doctest: +SKIP
   Array([0., 0.], dtype=float32)

If complex-valued arrays also need to be supported, a minor modification is
necessary:

::

   >>> g = lambda x: snp.sum(snp.abs(x)**2)
   >>> scico.grad(g)(snp.zeros(2, dtype=snp.float32))  #doctest: +SKIP
   Array([0., 0.], dtype=float32)
   >>> scico.grad(g)(snp.zeros(2, dtype=snp.complex64))  #doctest: +SKIP
   Array([0.-0.j, 0.-0.j], dtype=complex64)


An alternative is to define a `custom derivative rule
<https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#enforcing-a-differentiation-convention>`_
to enforce a particular derivative convention at a point.


JAX Arrays
==========

JAX utilizes a new array type :class:`~jax.Array`, which is similar to
NumPy :class:`~numpy.ndarray`, but can be backed by CPU, GPU, or TPU
memory and is immutable.


JAX and NumPy Arrays
--------------------

SCICO and JAX functions can be applied directly to NumPy arrays
without explicit conversion to JAX arrays, but this is not
recommended, as it can result in repeated data transfers from the CPU
to GPU. Consider this toy example on a system with a GPU present:

::

   x = np.random.randn(8)    # Array on host
   A = np.random.randn(8, 8) # Array on host
   y = snp.dot(A, x)         # A, x transfered to GPU
                             # y resides on GPU
   z = y + x                 # x must be transfered to GPU again


The unnecessary transfer can be avoided by first converting ``A`` and ``x`` to
JAX arrays:

::

   x = np.random.randn(8)    # array on host
   A = np.random.randn(8, 8) # array on host
   x = jax.device_put(x)     # transfer to GPU
   A = jax.device_put(A)
   y = snp.dot(A, x)         # no transfer needed
   z = y + x                 # no transfer needed


We recommend that input data be converted to JAX arrays via
:func:`jax.device_put` before calling any SCICO optimizers.

On a multi-GPU system, :func:`jax.device_put` can place data on a specific
GPU. See the `JAX notes on data placement
<https://jax.readthedocs.io/en/latest/faq.html?highlight=data%20placement#controlling-data-and-computation-placement-on-devices>`_.


JAX Arrays are Immutable
------------------------

Unlike standard NumPy arrays, JAX arrays are immutable: once they have
been created, they cannot be changed. This prohibits in-place updating
of JAX arrays. JAX provides special syntax for updating individual
array elements through the `indexed update operators
<https://jax.readthedocs.io/en/latest/jax.ops.html#syntactic-sugar-for-indexed-update-operators>`_.
