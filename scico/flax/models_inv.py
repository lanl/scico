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
from scico.linop.radon_astra import ParallelBeamProjector

# The imports of Scope and _Sentinel (above)
# are required to silence "cannot resolve forward reference"
# warnings when building sphinx api docs.


ModuleDef = Any


def construct_projector(N, n_projection):
    import numpy as np

    angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
    return ParallelBeamProjector(
        input_shape=(N, N),
        detector_spacing=1,
        det_count=N,
        angles=angles,
    )  # Radon transform operator


class MoDLNet(Module):

    r"""Net implementing a Model-Based Deep Learning Model for inversion.

    Flax implementation of the model-based deep learning (MoDL)
    architecture described in cite.

    Args:
        operator : object for computing forward and adjoint mappings.
        depth : depth of MoDL net. Default = 1.
        channels : number of channels of input tensor.
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        block_depth : depth of blocks.
        lmbda : initial value of lmbda in initial layer. Default: 0.5.
        dtype : type of signal to process. Default: jnp.float32.
    """
    operator: Callable[[Array], Array]
    depth: int
    channels: int
    num_filters: int
    block_depth: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    alpha_ini: float = 0.5
    dtype: Any = jnp.float32

    @compact
    def __call__(self, y: Array, train: bool = True) -> Array:
        """Apply MoDL net for inversion.

        Args:
            y: The nd-array with signal to invert.
            train: flag to differentiate between training and testing stages.

        Returns:
            The inverted signal.
        """

        def alpha_init_wrap(rng, shape, dtype=self.dtype):
            return jnp.ones(shape, dtype) * self.alpha_ini

        alpha = self.param("alpha", alpha_init_wrap, (1,))

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

        ahaI_f = lambda v: self.operator.adj(self.operator(v)) + alpha * v

        cgsol = lambda b: jnp.atleast_3d(cg_solver(ahaI_f, b.squeeze(), maxiter=10))

        for i in range(self.depth):
            z = resnet(x, train)
            # Solve:
            # (AH A + alpha I) x = Ahb + alpha * z
            b = Ahb + alpha * z
            x = lax.map(cgsol, b)
        return x


def cg_solver(A, b, x0=None, maxiter=50):
    def fun(carry, _):
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
