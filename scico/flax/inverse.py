# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.
"""Flax implementation of different imaging inversion models."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from functools import partial
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import lax

from flax.core import Scope  # noqa
from flax.linen.module import _Sentinel  # noqa
from flax.linen.module import Module, compact
from scico.flax import ResNet
from scico.linop import operator_norm
from scico.numpy import Array
from scico.typing import DType, PRNGKey, Shape

# The imports of Scope and _Sentinel (above) are required to silence
# "cannot resolve forward reference" warnings when building sphinx api
# docs.


ModuleDef = Any


class MoDLNet(Module):
    """Flax implementation of MoDL :cite:`aggarwal-2019-modl`.

    Flax implementation of the model-based deep learning (MoDL)
    architecture for inverse problems described in :cite:`aggarwal-2019-modl`.

    Args:
        operator: Operator for computing forward and adjoint mappings.
        depth: Depth of MoDL net. Default: 1.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the convolutional layer of the
            block. Corresponds to the number of channels in the output
            tensor.
        block_depth: Number of layers in the computational block.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        lmbda_ini: Initial value of the regularization weight `lambda`.
            Default: 0.5.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
        cg_iter: Number of iterations for cg solver. Default: 10.
    """

    operator: ModuleDef
    depth: int
    channels: int
    num_filters: int
    block_depth: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    lmbda_ini: float = 0.5
    dtype: Any = jnp.float32
    cg_iter: int = 10

    @compact
    def __call__(self, y: Array, train: bool = True) -> Array:
        """Apply MoDL net for inversion.

        Args:
            y: The array with signal to invert.
            train: Flag to differentiate between training and testing
               stages.

        Returns:
            The reconstructed signal.
        """

        def lmbda_init_wrap(rng: PRNGKey, shape: Shape, dtype: DType = self.dtype) -> Array:
            return jnp.ones(shape, dtype) * self.lmbda_ini

        lmbda = self.param("lmbda", lmbda_init_wrap, (1,))

        resnet = ResNet(
            self.block_depth,
            self.channels,
            self.num_filters,
            self.kernel_size,
            self.strides,
            dtype=self.dtype,
        )

        ah_f = lambda v: jnp.atleast_3d(self.operator.adj(v.reshape(self.operator.output_shape)))

        Ahb = lax.map(ah_f, y)
        x = Ahb

        ahaI_f = lambda v: self.operator.adj(self.operator(v)) + lmbda * v

        cgsol = lambda b: jnp.atleast_3d(
            cg_solver(ahaI_f, b.reshape(self.operator.input_shape), maxiter=self.cg_iter)
        )

        for i in range(self.depth):
            z = resnet(x, train)
            # Solve:
            # (AH A + lmbda I) x = Ahb + lmbda * z
            b = Ahb + lmbda * z
            x = lax.map(cgsol, b)
        return x


def cg_solver(A: Callable, b: Array, x0: Array = None, maxiter: int = 50) -> Array:
    r"""Conjugate gradient solver.

    Solve the linear system :math:`A\mb{x} = \mb{b}`, where :math:`A` is
    positive definite, via the conjugate gradient method. This is a light
    version constructed to be differentiable with the autograd
    functionality from jax. Therefore, (i) it uses :meth:`jax.lax.scan`
    to execute a fixed number of iterations and (ii) it assumes that the
    linear operator may use :meth:`jax.experimental.host_callback`. Due
    to the utilization of a while cycle, :meth:`scico.cg` is not
    differentiable by jax and :meth:`jax.scipy.sparse.linalg.cg` does not
    support functions using :meth:`jax.experimental.host_callback`
    explaining why an additional conjugate gradient function is implemented.

    Args:
        A: Function implementing linear operator :math:`A`, should be
            positive definite.
        b: Input array :math:`\mb{b}`.
        x0: Initial solution. Default: ``None``.
        maxiter: Maximum iterations. Default: 50.

    Returns:
        x: Solution array.
    """

    def fun(carry, _):
        """Function implementing one iteration of the conjugate gradient solver."""
        x, r, p, num = carry
        Ap = A(p)
        alpha = num / (p.ravel().conj().T @ Ap.ravel())
        x = x + alpha * p
        r = r - alpha * Ap
        num_old = num
        num = r.ravel().conj().T @ r.ravel()
        beta = num / num_old
        p = r + beta * p

        return (x, r, p, num), None

    if x0 is None:
        x0 = jnp.zeros_like(b)
    r0 = b - A(x0)
    num0 = r0.ravel().conj().T @ r0.ravel()
    carry = (x0, r0, r0, num0)
    carry, _ = lax.scan(fun, carry, xs=None, length=maxiter)
    return carry[0]


class ODPProxDnBlock(Module):
    """Flax implementation of ODP proximal gradient denoise block.

    Flax implementation of the unrolled optimization with deep priors
    (ODP) proximal gradient block for denoising :cite:`diamond-2018-odp`.

    Args:
        operator: Operator for computing forward and adjoint mappings.
            In this case it corresponds to the identity operator and is
            used at the network level.
        depth: Number of layers in block.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the convolutional layer of the
            block. Corresponds to the number of channels in the output
            tensor.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        alpha_ini: Initial value of the fidelity weight `alpha`.
            Default: 0.2.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
    """

    operator: ModuleDef
    depth: int
    channels: int
    num_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    alpha_ini: float = 0.2
    dtype: Any = jnp.float32

    def batch_op_adj(self, y: Array) -> Array:
        """Batch application of adjoint operator."""
        return self.operator.adj(y)

    @compact
    def __call__(self, x: Array, y: Array, train: bool = True) -> Array:
        """Apply denoising block.

        Args:
            x: The array with current stage of denoised signal.
            y: The array with noisy signal.
            train: Flag to differentiate between training and testing
                stages.

        Returns:
            The block output (i.e. next stage of denoised signal).
        """

        def alpha_init_wrap(rng: PRNGKey, shape: Shape, dtype: DType = self.dtype) -> Array:
            return jnp.ones(shape, dtype) * self.alpha_ini

        alpha = self.param("alpha", alpha_init_wrap, (1,))

        resnet = ResNet(
            self.depth,
            self.channels,
            self.num_filters,
            self.kernel_size,
            self.strides,
            dtype=self.dtype,
        )

        x = (resnet(x, train) + y * alpha) / (1.0 + alpha)

        return x


class ODPProxDcnvBlock(Module):
    """Flax implementation of ODP proximal gradient deconvolution block.

    Flax implementation of the unrolled optimization with deep priors
    (ODP) proximal gradient block for deconvolution under Gaussian noise
    :cite:`diamond-2018-odp`.

    Args:
        operator: Operator for computing forward and adjoint mappings.
            In this case it correponds to a circular convolution operator.
        depth: Number of layers in block.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the convolutional layer of the
            block. Corresponds to the number of channels in the output
            tensor.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        alpha_ini: Initial value of the fidelity weight `alpha`.
            Default: 0.99.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
    """

    operator: ModuleDef
    depth: int
    channels: int
    num_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    alpha_ini: float = 0.99
    dtype: Any = jnp.float32

    def setup(self):
        """Computing operator norm and setting operator for batch
        evaluation and defining network layers."""
        self.operator_norm = operator_norm(self.operator)
        self.ah_f = lambda v: jnp.atleast_3d(
            self.operator.adj(v.reshape(self.operator.output_shape))
        )

        self.resnet = ResNet(
            self.depth,
            self.channels,
            self.num_filters,
            self.kernel_size,
            self.strides,
            dtype=self.dtype,
        )

        def alpha_init_wrap(rng: PRNGKey, shape: Shape, dtype: DType = self.dtype) -> Array:
            return jnp.ones(shape, dtype) * self.alpha_ini

        self.alpha = self.param("alpha", alpha_init_wrap, (1,))

    def batch_op_adj(self, y: Array) -> Array:
        """Batch application of adjoint operator."""
        return lax.map(self.ah_f, y)

    def __call__(self, x: Array, y: Array, train: bool = True) -> Array:
        """Apply debluring block.

        Args:
            x: The array with current stage of reconstructed signal.
            y: The array with signal to invert.
            train: Flag to differentiate between training and testing
                stages.

        Returns:
            The block output (i.e. next stage of reconstructed signal).
        """

        # DFT over spatial dimensions
        fft_shape: Shape = x.shape[1:-1]
        fft_axes: Tuple[int, int] = (1, 2)

        scale = 1.0 / (self.alpha * self.operator_norm**2 + 1)

        x = jnp.fft.irfftn(
            jnp.fft.rfftn(
                self.alpha * self.batch_op_adj(y) + self.resnet(x, train),
                s=fft_shape,
                axes=fft_axes,
            )
            / scale,
            s=fft_shape,
            axes=fft_axes,
        )

        return x


class ODPGrDescBlock(Module):
    r"""Flax implementation of ODP gradient descent with :math:`\ell_2` loss block.

    Flax implementation of the unrolled optimization with deep priors
    (ODP) gradient descent block for inversion using :math:`\ell_2` loss
    described in :cite:`diamond-2018-odp`.

    Args:
        operator: Operator for computing forward and adjoint mappings. In
            this case it corresponds to the identity operator and is used
            at the network level.
        depth: Number of layers in block.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the convolutional layer of the
            block. Corresponds to the number of channels in the output
            tensor.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        alpha_ini: Initial value of the fidelity weight `alpha`.
            Default: 0.2.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
    """
    operator: ModuleDef
    depth: int
    channels: int
    num_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    alpha_ini: float = 0.2
    dtype: Any = jnp.float32

    def setup(self):
        """Setting operator for batch evaluation and defining network layers."""
        self.ah_f = lambda v: jnp.atleast_3d(
            self.operator.adj(v.reshape(self.operator.output_shape))
        )
        self.a_f = lambda v: jnp.atleast_3d(self.operator(v.reshape(self.operator.input_shape)))

        self.resnet = ResNet(
            self.depth,
            self.channels,
            self.num_filters,
            self.kernel_size,
            self.strides,
            dtype=self.dtype,
        )

        def alpha_init_wrap(rng: PRNGKey, shape: Shape, dtype: DType = self.dtype) -> Array:
            return jnp.ones(shape, dtype) * self.alpha_ini

        self.alpha = self.param("alpha", alpha_init_wrap, (1,))

    def batch_op_adj(self, y: Array) -> Array:
        """Batch application of adjoint operator."""
        return lax.map(self.ah_f, y)

    def __call__(self, x: Array, y: Array, train: bool = True) -> Array:
        """Apply gradient descent block.

        Args:
            x: The array with current stage of reconstructed signal.
            y: The array with signal to invert.
            train: Flag to differentiate between training and testing
                stages.

        Returns:
            The block output (i.e. next stage of inverted signal).
        """

        x = self.resnet(x, train) - self.alpha * self.batch_op_adj(lax.map(self.a_f, x) - y)

        return x


class ODPNet(Module):
    """Flax implementation of ODP network :cite:`diamond-2018-odp`.

    Flax implementation of the unrolled optimization with deep priors
    (ODP) network for inverse problems described in
    :cite:`diamond-2018-odp`. It can be constructed with proximal gradient
    blocks or gradient descent blocks.

    Args:
        operator: Operator for computing forward and adjoint mappings.
        depth: Depth of MoDL net. Default: 1.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the convolutional layer of the
            block. Corresponds to the number of channels in the output
            tensor.
        block_depth: Number of layers in the computational block.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        alpha_ini: Initial value of the fidelity weight `alpha`.
            Default: 0.5.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
        odp_block: processing block to apply. Default
            :class:`ODPProxDnBlock`.
    """

    operator: ModuleDef
    depth: int
    channels: int
    num_filters: int
    block_depth: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    alpha_ini: float = 0.5
    dtype: Any = jnp.float32
    odp_block: Callable = ODPProxDnBlock

    @compact
    def __call__(self, y: Array, train: bool = True) -> Array:
        """Apply ODP net for inversion.

        Args:
            y: The array with signal to invert.
            train: Flag to differentiate between training and testing
                stages.

        Returns:
            The reconstructed signal.
        """
        block = partial(
            self.odp_block,
            operator=self.operator,
            depth=self.block_depth,
            channels=self.channels,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            dtype=self.dtype,
        )

        # Initial block handles initial inversion.
        # Not all operators are batch-ready.
        alpha0_i = self.alpha_ini
        block0 = block(alpha_ini=alpha0_i)
        x = block0.batch_op_adj(y)
        x = block0(x, y, train)
        alpha0_i /= 2.0

        for i in range(self.depth - 1):
            x = block(alpha_ini=alpha0_i)(x, y, train)
            alpha0_i /= 2.0
        return x
