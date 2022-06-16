"""
# Example: CT Reconstruction

This example demonstrates the process of signal reconstruction in SCICO.

## Introduction

You perform a computed tomography (CT) measurement of an object and will want to reconstruct it from
the sinogram, which we'll call $y$.
"""
import scico.numpy as snp
from scico import plot

plot.config_notebook_plotting()

"""
Run the next cell to see $y$.
"""

import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "gray"  # set default colormap

from tutorial_funcs_part2 import load_y_ct

y = load_y_ct()

print(f"The shape of y is {y.shape}")

fig, ax = plt.subplots()
ax.imshow(y)
ax.set_title("$y$")
fig.show()

"""
This image shows the sinogram, the first dimension represents the number of projections, while the second the
size of the object. Although this is not directly specified, we will assume that the the projections are equally
spaced. Let's try to construct a tomographic projector using SCICO. Run the next cell to build a ground truth
signal.
"""

from xdesign import SiemensStar, discrete_phantom

phantom = SiemensStar(32)
N = 512  # image size
x_ss = snp.pad(discrete_phantom(phantom, N - 16), 8)

# Plot signal
fig, ax = plt.subplots()
ax.imshow(x_ss)
ax.set_title("Siemens Star")
fig.show()

"""
SCICO provides CT projectors based on Python libraries such as ASTRA and SVMBIR. In this case we will use the
SVMBIR interface (see https://scico.readthedocs.io/en/latest/_autosummary/scico.linop.radon_svmbir.html)

**Find the appropriate functionality and use it to compute 120 projections of the Siemens Star.**
"""
# startq

num_angles = 120
angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)
num_channels = ...
A = ...
sino = A @ x_ss
# starta
from scico.linop.radon_svmbir import TomographicProjector

num_angles = 120
angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)
num_channels = N
A = TomographicProjector(x_ss.shape, angles, num_channels)
sino = A @ x_ss
# endqa

"""
Run the next cell to plot the original signal and the corresponding sinogram.
"""
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(15, 5))
plot.imview(x_ss, title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    sino,
    title="Sinogram",
    cbar=None,
    fig=fig,
    ax=ax[1],
)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
fig.show()

"""
You are done with part 1. Please report back in the Webex chat: **done with part 1**.

While you wait for others to finish, you could explore the other SCICO CT interface
(https://scico.readthedocs.io/en/latest/_autosummary/scico.linop.radon_astra.html)
and think about what modifications are needed to construct the ASTRA-based CT operator.

🛑 **PAUSE HERE** 🛑
"""

"""
## Solving a regularized reconstruction: TV regularization

Now that you know how to construct an operator to represent tha CT projection, you can pose the reconstruction
problem as a regularized least squares problem

$$ \min_x \| y - Ax \|_2^2 + \, \lambda \, r(x),$$

where $A$ is the forward model (i.e. the CT projector), $y$ contains the measurements (i.e. the sinogram)
and $x$ contains the object you want to recover (i.e. the reconstruction). Since this is an ill-posed
problem, you need to provide some prior information to be able to find a meaningful solution. This is expressed
via $r$, which represents an appropriate regularization of the solution $x$, weighted by a constant $\lambda > 0$,
which establishes the trade-off between the fidelity to the measurements and the regularization of the solution.

In this case, let's assume that you know that the solution is piece-wise constant. Total-Variation (TV) is
a good model to represent such previous knowledge. TV is the $L_1$ norm of the gradient of the input, which can
be written as:

$$ r(x) = \| D x \|_1,$$

where $D$ computes finite differences. In the previous part of the tutorial you used the corresponding SCICO
operator. If you do not remember, **look through the [list of SCICO operators](https://scico.readthedocs.io/en/latest/_autosummary/scico.linop.html#module-scico.linop) to find the appropriate one
and instantiate it.**
"""


import scico.linop

# Fill in with the linear operator that computes finite differences
# startq
D = scico.linop.FindTheCorrectOperator(input_shape=x_ss.shape)
# starta
D = scico.linop.FiniteDifference(input_shape=x_ss.shape)
# endqa

"""
Run the next cell to see $D$ in action.
"""

Dx = D @ x_ss


fig, ax = plt.subplots()
ax.imshow(Dx[1])
ax.set_title(r"$\nabla_x x_{ss}$")
fig.show()

fig, ax = plt.subplots()
ax.imshow(Dx[0])
ax.set_title(r"$\nabla_y x_{ss}$")
fig.show()

"""
Since the regularization has some values where it is non-derivable, we will use the Alternating Direction
Method of Multipliers `ADMM` optimizer (see https://scico.readthedocs.io/en/latest/_autosummary/scico.optimize.html#scico.optimize.ADMM).

The `ADMM` optimizer solves problems of the form

$$\min_{\mathbf{x}, \mathbf{z}} \; f(\mathbf{x}) + g(\mathbf{z}) \; \text{such that}
   \; \acute{A} \mathbf{x} + \acute{B} \mathbf{z} = \mathbf{c} \;,$$

where $f$ and $g$ are convex (but not necessarily smooth)
functions, $\acute{A}$ and $\acute{B}$ are linear operators,
and $\mathbf{c}$ is a constant vector.

We can cast our problem in the ADMM structure as follows

$$f(x) = \| y - Ax \|_2^2,$$
$$g(z) = \lambda \| z \|_1,$$
$$z = Dx,$$

which implies that $\acute{A} = -D$, $\acute{B} = I$ and $c = 0$, with $I$ the identity operator.
"""

"""
✋ Isotropic TV: the isotropic version of TV exhibits fewer block-like artifacts on edges that are not vertical or
horizontal. Implementing the isotropic TV in scico requires two modifications: (i) use the L21Norm,
instead of the L1Norm implied by the equation (ii) make the results of horizontal and vertical
finite differences to have the same shape (which is required for the L21Norm).

Look into the finite differences documentation
(https://scico.readthedocs.io/en/latest/_autosummary/scico.linop.html#scico.linop.FiniteDifference)
and **figure out how to make the results to have the same shape**.
"""

# startq
D2 = scico.linop.FiniteDifference(...)
# starta
D2 = scico.linop.FiniteDifference(input_shape=(N, N), append=0)
# endqa

"""
**Given the ADMM formulation of the CT reconstruction problem and the
constants included in the following cell, construct a SCICO ADMM optimizer using isotropic TV.**

Note that the ADMM optimizer from SCICO requires the specification of the subproblem solver.
When $f$ takes the form $\|\mathbf{A} \mathbf{x} - \mathbf{y}\|^2$ an `ADMM.LinearSubproblemSolver` can be used.
It makes use of the conjugate gradient method, and is significantly more efficient than the
`ADMM.GenericSubproblemSolver` (see: https://scico.readthedocs.io/en/latest/optimizer.html).

Remember, however, that you are trying to reconstruct the sinogram $y$ that has 45 equally-spaced projections.
Therefore, you need to start by defining the correct tomographic operator.
"""

λ = 2e0  # L1 norm regularization parameter
ρ = 5e0  # ADMM penalty parameter
maxiter = 25  # number of ADMM iterations
cg_tol = 1e-4  # CG relative tolerance
cg_maxiter = 25  # maximum CG iterations per ADMM iteration

""""""


from scico import functional, loss
from scico.optimize.admm import ADMM, LinearSubproblemSolver

# startq
A = TomographicProjector(...)
g = ...
f = ...

x0 = A.T @ y / N


solver = ADMM(
    ...,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 5},
)
# starta
num_angles, N = y.shape
angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)
num_channels = N
A = TomographicProjector((N, N), angles, num_channels)

g = λ * functional.L21Norm()
f = loss.SquaredL2Loss(y=y, A=A)

x0 = A.T @ y / N

solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[D2],
    rho_list=[ρ],
    x0=x0,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 5},
)
# endqa

"""
Run the next cell to see an example of running ADMM to compute $x$ from our previous setup.
"""

x_hat = solver.solve()

"""
Run the cell below to see your results!
"""
from scico import metric

x_reconstruction = snp.clip(x_hat, 0, 1.0)

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(y, title="Sinogram", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    x0,
    title="Initial Reconstruction",
    cbar=None,
    fig=fig,
    ax=ax[1],
)
plot.imview(
    x_reconstruction,
    title="TV Reconstruction",
    fig=fig,
    ax=ax[2],
)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[2].get_images()[0], cax=cax, label="arbitrary units")
fig.show()

"""
Run the cell below to plot some convergence statistics.
"""
hist = solver.itstat_object.history(transpose=True)

fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 5))
plot.plot(
    hist.Objective,
    title="Objective function",
    xlbl="Iteration",
    ylbl="Functional value",
    fig=fig,
    ax=ax[0],
)
plot.plot(
    snp.vstack((hist.Prml_Rsdl, hist.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
    fig=fig,
    ax=ax[1],
)
fig.show()

"""
Run the cell below to load the ground truth object and allow for a quantitative evaluation of the quality of the results.
"""
from tutorial_funcs_part2 import load_gt_ct

x_gt_ct = load_gt_ct()

print("Range of ground truth")
print("Minimum: ", x_gt_ct.min())
print("Maximum: ", x_gt_ct.max())

"""
Run the cell below to see your results and a quantitative comparison against the ground truth.
"""
fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(x_gt_ct, title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    x0,
    title="Initial Reconstruction: \nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(x_gt_ct, x0), metric.mae(x_gt_ct, x0)),
    cbar=None,
    fig=fig,
    ax=ax[1],
)
plot.imview(
    x_reconstruction,
    title="TV Reconstruction\nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(x_gt_ct, x_reconstruction), metric.mae(x_gt_ct, x_reconstruction)),
    fig=fig,
    ax=ax[2],
)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[2].get_images()[0], cax=cax, label="arbitrary units")
fig.show()

"""
You are done with part 2. Please report back in the Webex chat: **done with part 2**.

While you wait for others to finish, you could think about how you would solve this
problem with the PGM optimizer you used in the previous tutorial session.

🛑 **PAUSE HERE** 🛑
"""

"""
## Solving a regularized reconstruction with implicit prior: Plug-and-Play (PnP) Priors Framework

In the previous part of the tutorial we assumed that the solution was piece-wise constant and
applied a Total-Variation (TV) regularization. However, although the ground truth is effectively piece-wise
constant, some artifacts can be seen in the TV-based solution.

A less structured regularization, which may be useful in more general cases, is to assume that a good solution
is one that has less noise. Denoisers, such as BM3D or DnCNN, are operators, or trained models, that reduce the
Gaussian white noise of a signal and can take the role of $g$. In this way, they provide an implicit mechanism
to reduce certain artifacts of the reconstructed image, without strictly requiring the definition of a function
to be optimized. Hence the name plug-and-play.

We can complement this with another regularization enforcing the solution to be nonnegative, since
negative values might be nonphysical. We already implemented this type of regularization in a previous tutorial.

The only modifications that we need to do is to define these alternative regularizers as follows

$$g_1(z) = \mathrm{Denoiser}(z)$$
$$g_2(z) = \iota_{\mathbb{NN}}(z),$$

with $\iota_{\mathbb{NN}}$ representing a nonnegative indicator function.

**Construct a SCICO ADMM optimizer for this new regularized formulation of the problem.**
"""
from scico.functional import BM3D, NonNegativeIndicator
from scico.linop import Identity

ρ = 15  # ADMM penalty parameter
σ = 0.18  # denoiser sigma

# startq
g1 = σ * ρ * ...
g2 = ...

solver_pnp = ADMM(
    ...,
    maxiter=20,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 100}),
    itstat_options={"display": True, "period": 5},
)
# starta
g1 = σ * ρ * BM3D()
g2 = NonNegativeIndicator()

solver_pnp = ADMM(
    f=f,
    g_list=[g1, g2],
    C_list=[Identity((N, N)), Identity((N, N))],
    rho_list=[ρ, ρ],
    x0=x0,
    maxiter=20,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 100}),
    itstat_options={"display": True, "period": 5},
)
# endqa
"""
Run the next cell to see an example of running ADMM to compute $x$ from the PnP setup.
"""
x_bm3d = solver_pnp.solve()
"""
Run the cell below to plot some convergence statistics.
"""
hist_bm3d = solver_pnp.itstat_object.history(transpose=True)

plot.plot(
    snp.vstack((hist_bm3d.Prml_Rsdl, hist_bm3d.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
)

"""
We do not have statistics for the objective function because...
"""

# startq
"""

"""
# starta
"""
because the implicit prior used in the PnP framework is not the product of optimizing a function, it is a
unsupervised improvement of the solution image.
"""
# endqa

"""
Run the cell below to see your results and a quantitative comparison against the ground truth.
"""

x_bm3d_reconstruction = snp.clip(x_bm3d, 0, 1.0)

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(x_gt_ct, title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    x0,
    title="Initial Reconstruction: \nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(x_gt_ct, x0), metric.mae(x_gt_ct, x0)),
    cbar=None,
    fig=fig,
    ax=ax[1],
)
plot.imview(
    x_bm3d_reconstruction,
    title="TV Reconstruction\nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(x_gt_ct, x_bm3d_reconstruction), metric.mae(x_gt_ct, x_bm3d_reconstruction)),
    fig=fig,
    ax=ax[2],
)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[2].get_images()[0], cax=cax, label="arbitrary units")
fig.show()

"""
## Conclusion
This tutorial has shown how to set up and solve a simple CT reconstruction problem in
SCICO using regularized least squares formulations. In doing so, it has demonstrated
a diverse set of classes provided by SCICO such as operators, solvers and denoisers
which make expressing regularized optimization problems more straightforward.
"""

"""
You are done with this tutorial! Please report back in the Webex chat: **done with the PnP tutorial**.

While you wait for others to finish, you could check out the [Denoiser documentation](https://scico.readthedocs.io/en/latest/_autosummary/scico.denoiser.html#module-scico.denoiser)
to understand other types of denoisers available in SCICO for the `PnP` framework.
"""
