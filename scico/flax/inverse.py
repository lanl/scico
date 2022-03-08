#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flax implementation of different imaging inversion models.
"""

from functools import partial
from typing import Any, Callable, Tuple

from flax.linen.module import Module, compact, _Sentinel
from flax.core import Scope  # noqa

from jax import lax
import jax.numpy as jnp

from scico.typing import Array
from scico.flax import ResNet


# The imports of Scope and _Sentinel (above)
# are required to silence "cannot resolve forward reference"
# warnings when building sphinx api docs.


ModuleDef = Any


class MoDLNet(Module):
    r"""Flax implementation of MoDL :cite:`aggarwal-2019-modl`.

    Flax implementation of the model-based deep learning (MoDL)
    architecture for inverse problems described in :cite:`aggarwal-2019-modl`.

    Args:
        operator : Operator for computing forward and adjoint mappings.
        depth : Depth of MoDL net. Default = 1.
        channels : Number of channels of input tensor.
        num_filters : Number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        block_depth : Number of layers in the computational block.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        lmbda_ini : Initial value of the regularization weight `lambda`. Default: 0.5.
        dtype : Output type. Default: jnp.float32.
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

    @compact
    def __call__(self, y: Array, train: bool = True) -> Array:
        """Apply MoDL net for inversion.

        Args:
            y: The nd-array with signal to invert.
            train: Flag to differentiate between training and testing stages.

        Returns:
            The reconstructed signal.
        """

        def lmbda_init_wrap(rng, shape, dtype=self.dtype):
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

        ah_f = lambda v: jnp.atleast_3d(self.operator.adj(v.squeeze()))

        Ahb = lax.map(ah_f, y)
        x = Ahb

        ahaI_f = lambda v: self.operator.adj(self.operator(v)) + lmbda * v

        cgsol = lambda b: jnp.atleast_3d(cg_solver(ahaI_f, b.squeeze(), maxiter=10))

        for i in range(self.depth):
            z = resnet(x, train)
            # Solve:
            # (AH A + lmbda I) x = Ahb + lmbda * z
            b = Ahb + lmbda * z
            x = lax.map(cgsol, b)
        return x


def cg_solver(A: Callable, b: Array, x0: Array = None, maxiter: int = 50):
    r"""Conjugate Gradient solver.

    Solve the linear system :math:`A\mb{x} = \mb{b}`, where :math:`A` is
    positive definite, via the conjugate gradient method. This is a light version constructed to be differentiable with the autograd functionality from jax. Therefore, (i) it uses :meth:`jax.lax.scan` to execute a fixed number of iterations and (ii) it assumes that the linear operator may use :meth:`jax.experimental.host_callback`. Due to a while cycle, :meth:`scico.cg` is not differentiable by jax and :meth:`jax.scipy.sparse.linalg.cg` does not support functions using :meth:`jax.experimental.host_callback` explaining why an additional conjugate gradient function is implemented.

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


class ODPDnBlock(Module):
    r"""Flax implementation of ODP denoise block :cite:`diamond-2018-odp`.

    Flax implementation of the unrolled optimization with deep priors (ODP) block for denoising described in :cite:`diamond-2018-odp`.

    Args:
        operator : Operator for computing forward and adjoint mappings. In this case it corresponds to the identity operator and is used at the network level.
        depth : Number of layers in block.
        channels : Number of channels of input tensor.
        num_filters : Number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        alpha_ini : Initial value of the fidelity weight `alpha`. Default: 0.2.
        dtype : Output type. Default: jnp.float32.
    """
    operator: ModuleDef
    depth: int
    channels: int
    num_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    alpha_ini: float = 0.2
    dtype: Any = jnp.float32

    @compact
    def __call__(self, x: Array, y: Array, train: bool = True) -> Array:
        """Apply denoising block.

        Args:
            x: The nd-array with current stage of denoised signal.
            y: The nd-array with noisy signal.
            train: Flag to differentiate between training and testing stages.

        Returns:
            The block output (i.e. next stage of denoised signal).
        """

        def alpha_init_wrap(rng, shape, dtype=self.dtype):
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


class ODPDblrBlock(Module):
    r"""Flax implementation of ODP deblurring block :cite:`diamond-2018-odp`.

    Flax implementation of the unrolled optimization with deep priors (ODP) block for debluring under Gaussian noise described in :cite:`diamond-2018-odp`.

    Args:
        operator : Operator for computing forward and adjoint mappings. In this case it correponds to a circular convolution operator.
        depth : Number of layers in block.
        channels : Number of channels of input tensor.
        num_filters : Number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        alpha_ini : Initial value of the fidelity weight `alpha`. Default: 0.99.
        dtype : Output type. Default: jnp.float32.
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
        self.operator_norm = 1.0  # ToDo

    @compact
    def __call__(self, x: Array, y: Array, train: bool = True) -> Array:
        """Apply debluring block.

        Args:
            x: The nd-array with current stage of reconstructed signal.
            y: The nd-array with signal to invert.
            train: Flag to differentiate between training and testing stages.

        Returns:
            The block output (i.e. next stage of reconstructed signal).
        """

        def alpha_init_wrap(rng, shape, dtype=self.dtype):
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

        # DFT over spatial dimensions
        fft_shape: Shape = x.shape[1:-1]
        fft_axes: Tuple[int, int] = (1, 2)

        scale = 1.0 / (alpha * self.operator_norm**2 + 1)

        x = jnp.fft.irfftn(
            jnp.fft.rfftn(
                alpha * self.operator.adj(y) + resnet(x, train), s=fft_shape, axes=fft_axes
            )
            / scale,
            s=fft_shape,
            axes=fft_axes,
        )

        return x


class ODPProxNet(Module):
    r"""Flax implementation of ODP proximal gradient network :cite:`diamond-2018-odp`.

    Flax implementation of the unrolled optimization with deep priors (ODP) with a proximal gradient network architecture for inverse problems  described in :cite:`diamond-2018-odp`.

    Args:
        operator : Operator for computing forward and adjoint mappings.
        depth : Depth of MoDL net. Default = 1.
        channels : Number of channels of input tensor.
        num_filters : Number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        block_depth : Number of layers in the computational block.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        alpha_ini : Initial value of the fidelity weight `alpha`. Default: 0.5.
        dtype : Output type. Default: jnp.float32.
        odp_block : processing block to apply. Default :class:`ODPDnBlock`.
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
    odp_block: Callable = ODPDnBlock

    @compact
    def __call__(self, y: Array, train: bool = True) -> Array:
        """Apply ODP net for inversion.

        Args:
            y: The nd-array with signal to invert.
            train: Flag to differentiate between training and testing stages.

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
        alpha0_i = self.alpha_ini
        x = self.operator.adj(y)
        for i in range(self.depth):
            x = block(alpha_ini=alpha0_i)(x, y, train)
            alpha0_i /= 2.0
        return x
