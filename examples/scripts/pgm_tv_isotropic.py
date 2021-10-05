#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Isotropic Total Variation (Accelerated PGM)
===========================================

This example demonstrates the use of class [pgm.AcceleratedPGM](../_autosummary/scico.pgm.rst#scico.pgm.AcceleratedPGM) to solve isotropic total variation (TV) regularization. It solves the denoising problem

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - \mathbf{x} \|^2 + \lambda R(\mathbf{x}) \;,$$

where $R$ is the isotropic TV: the sum of the norms of the gradient vectors at each point in the image $\mathbf{x}$. The same reconstruction is performed with anisotropic TV regularization for comparison; the isotropic version shows fewer block-like artifacts.

The solution via PGM is based on :cite:`beck-2009-tv`. This follows a dual approach that constructs a dual for the constrained denoising problem (the constraint given by restricting the solution to the [0,1] range). The PGM solution minimizes the resulting dual. In this case, switching between the two regularizers corresponds to switching between two different projectors.
"""

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from scico import functional, linop, loss, operator, plot
from scico.blockarray import BlockArray
from scico.pgm import AcceleratedPGM, RobustLineSearchStepSize
from scico.typing import JaxArray
from scico.util import ensure_on_device

"""
Create a ground truth image.
"""
N = 256  # Image size


# These steps create a ground truth image by spatially filtering noise
kernel_size = N // 5
key = jax.random.PRNGKey(1)
x_gt = jax.random.uniform(key, shape=(N + kernel_size - 1, N + kernel_size - 1))
x = jnp.linspace(-3, 3, kernel_size)
window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
window = window / window.sum()
x_gt = jsp.signal.convolve(x_gt, window, mode="valid")
x_gt = (x_gt > jnp.percentile(x_gt, 25)).astype(float) + (x_gt > jnp.percentile(x_gt, 75)).astype(
    float
)
x_gt = x_gt / x_gt.max()


"""
Add noise to create a noisy test image.
"""
sigma = 1.0  # noise standard deviation
key, subkey = jax.random.split(key)

n = sigma * jax.random.normal(subkey, shape=x_gt.shape)

y = x_gt + n


"""
Define finite difference operator and adjoint
"""
# the append=0 option appends 0 to the input along the axis
# prior to performing the difference to make the results of
# horizontal and vertical finite differences the same shape
C = linop.FiniteDifference(input_shape=x_gt.shape, append=0)
A = C.adj


"""
Define initial estimate: a zero array
"""
x0 = jnp.zeros(C(y).shape)


"""
Define the dual of the total variation denoising problem
"""


class DualTVLoss(loss.Loss):
    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        lmbda: float = 0.5,
    ):
        y = ensure_on_device(y)
        self.functional = functional.SquaredL2Norm()
        super().__init__(y=y, A=A, scale=1.0)
        self.lmbda = lmbda

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:

        xint = self.y - self.lmbda * self.A(x)
        return -1.0 * self.functional(xint - jnp.clip(xint, 0.0, 1.0)) + self.functional(xint)


"""
Denoise with isotropic total variation. Define projector for isotropic total variation.
"""
# Evaluation of functional set to zero
class IsoProjector(functional.Functional):

    has_eval = True
    has_prox = True
    is_smooth = False

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return 0.0

    def prox(self, x: JaxArray, lam: float) -> JaxArray:
        norm_x_ptp = jnp.sqrt(jnp.sum(jnp.abs(x) ** 2, axis=0))

        x_out = x / jnp.maximum(jnp.ones(x.shape), norm_x_ptp)
        out1 = x[0, :, -1] / jnp.maximum(jnp.ones(x[0, :, -1].shape), jnp.abs(x[0, :, -1]))
        x_out_1 = jax.ops.index_update(x_out, jax.ops.index[0, :, -1], out1)
        out2 = x[1, -1, :] / jnp.maximum(jnp.ones(x[1, -1, :].shape), jnp.abs(x[1, -1, :]))
        x_out = jax.ops.index_update(x_out_1, jax.ops.index[1, -1, :], out2)

        return x_out


"""
Use RobustLineSearchStepSize object and set up AcceleratedPGM solver object. Run the solver.
"""
reg_weight_iso = 2e0
f_iso = DualTVLoss(y=y, A=A, lmbda=reg_weight_iso)
f_iso.is_smooth = True
g_iso = IsoProjector()

solver_iso = AcceleratedPGM(
    f=f_iso,
    g=g_iso,
    L0=16.0 * f_iso.lmbda ** 2,
    x0=x0,
    maxiter=100,
    verbose=True,
    step_size=RobustLineSearchStepSize(),
)

# Run the solver.
x = solver_iso.solve()
hist_iso = solver_iso.itstat_object.history(transpose=True)
# project to constraint set
x_iso = jnp.clip(y - f_iso.lmbda * f_iso.A(x), 0.0, 1.0)


"""
Denoise with anisotropic total variation for comparison. Define projector for anisotropic total variation.
"""
# Evaluation of functional set to zero
class AnisoProjector(functional.Functional):

    has_eval = True
    has_prox = True
    is_smooth = False

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return 0.0

    def prox(self, x: JaxArray, lam: float) -> JaxArray:

        return x / jnp.maximum(jnp.ones(x.shape), jnp.abs(x))


"""
Use RobustLineSearchStepSize object and set up AcceleratedPGM solver object. Weight was tuned to give the same data fidelty as the isotropic case. Run the solver.
"""

reg_weight_aniso = 1.74e0
f = DualTVLoss(y=y, A=A, lmbda=reg_weight_aniso)
f.is_smooth = True
g = AnisoProjector()

solver = AcceleratedPGM(
    f=f,
    g=g,
    L0=16.0 * f.lmbda ** 2,
    x0=x0,
    maxiter=100,
    verbose=True,
    step_size=RobustLineSearchStepSize(),
)

# Run the solver.
x = solver.solve()
# project to constraint set
x_aniso = jnp.clip(y - f.lmbda * f.A(x), 0.0, 1.0)


"""
Compute the data fidelity.
"""
df = hist_iso.Objective[-1]
print(f"Data fidelity for isotropic TV was {df:.2e}")
hist = solver.itstat_object.history(transpose=True)
df = hist.Objective[-1]
print(f"Data fidelity for anisotropic TV was {df:.2e}")


"""
Plot results.
"""
plt_args = dict(norm=plot.matplotlib.colors.Normalize(vmin=0, vmax=1.5))
fig, ax = plot.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(11, 10))
plot.imview(x_gt, title="Ground truth", fig=fig, ax=ax[0, 0], **plt_args)
plot.imview(y, title="Noisy version", fig=fig, ax=ax[0, 1], **plt_args)
plot.imview(x_iso, title="Isotropic TV denoising", fig=fig, ax=ax[1, 0], **plt_args)
plot.imview(x_aniso, title="Anisotropic TV denoising", fig=fig, ax=ax[1, 1], **plt_args)
fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.05, wspace=0.2, hspace=0.01)
fig.colorbar(
    ax[0, 0].get_images()[0], ax=ax, location="right", shrink=0.9, pad=0.05, label="Arbitrary Units"
)
fig.suptitle("Denoising comparison")
fig.show()

# Zoomed version
fig, ax = plot.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(11, 10))
plot.imview(x_gt, title="Ground truth", fig=fig, ax=ax[0, 0], **plt_args)
plot.imview(y, title="Noisy version", fig=fig, ax=ax[0, 1], **plt_args)
plot.imview(x_iso, title="Isotropic TV denoising", fig=fig, ax=ax[1, 0], **plt_args)
plot.imview(x_aniso, title="Anisotropic TV denoising", fig=fig, ax=ax[1, 1], **plt_args)
ax[0, 0].set_xlim(N // 4, N // 4 + N // 2)
ax[0, 0].set_ylim(N // 4, N // 4 + N // 2)
fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.05, wspace=0.2, hspace=0.01)
fig.colorbar(
    ax[0, 0].get_images()[0], ax=ax, location="right", shrink=0.9, pad=0.05, label="Arbitrary Units"
)
fig.suptitle("Denoising comparison (zoomed)")
fig.show()

input("\nWaiting for input to close figures and exit")
