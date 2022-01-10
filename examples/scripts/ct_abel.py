# %%
import numpy as np

import matplotlib.pyplot as plt

# %%


def dist_map_2D(img_shape, center=None):

    if center == None:
        center = [img_dim // 2 for img_dim in img_shape]

    coords = [np.arange(0, img_dim) for img_dim in img_shape]
    coord_mesh = np.meshgrid(*coords, sparse=True, indexing="ij")

    dist_map = sum([(coord_mesh[i] - center[i]) ** 2 for i in range(len(coord_mesh))])
    dist_map = np.sqrt(dist_map)

    return dist_map


def create_french_test_phantom(img_shape, radius_list, val_list, center=None):

    dist_map = dist_map_2D(img_shape, center)

    img = np.zeros(img_shape)
    for r, val in zip(radius_list, val_list):
        img[dist_map < r] = val

    return img


x_gt = create_french_test_phantom((256, 254), [100, 50, 25], [1, 0, 0.5])

# %%

import abel

# x_gt = create_french_test_phantom((256, 256), [100, 50, 25], [1, 0, 0.5], center=(150,150))

y = abel.Transform(np.array(x_gt), direction="forward", method="daun").transform
x_inv = abel.Transform(np.array(y), direction="inverse", method="daun").transform

plt.imshow(y)
plt.show()

plt.imshow(x_inv)
plt.show()

plt.imshow(x_gt)
plt.show()

# %%

import scico
from scico.linop.abel import AbelProjector

A = AbelProjector(x_gt.shape)
y = A @ x_gt
y = y + 1 * np.random.normal(size=y.shape)
ATy = A.T @ y

# plt.imshow(y)
# plt.show()
# plt.imshow(ATy)
# plt.show()

scico.linop.valid_adjoint(A, A.H, eps=None, x=x_gt, y=y)

# %%

import scico.numpy as snp
from scico import functional, linop, loss
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Set up ADMM solver object.
"""
λ = 1.71e01  # L1 norm regularization parameter
ρ = 4.83e01  # ADMM penalty parameter
maxiter = 100  # number of ADMM iterations
cg_tol = 1e-4  # CG relative tolerance
cg_maxiter = 25  # maximum CG iterations per ADMM iteration

g = λ * functional.L1Norm()  # regularization functionals gi
C = linop.FiniteDifference(input_shape=x_gt.shape)  # analysis operators Ci

f = loss.SquaredL2Loss(y=y, A=A)

# x0 = snp.clip(y, 0, 1.0)
x_inv = A.inverse(y)
x0 = snp.clip(x_inv, 0, 1.0)

solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[ρ],
    x0=x0,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 5},
)


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
solver.solve()
hist = solver.itstat_object.history(transpose=True)
x_reconstruction = snp.clip(solver.x, 0, 1.0)

# %%
plt.imshow(x_reconstruction)
plt.title("TV regularized inverse abel")
plt.show()

plt.imshow(x0)
plt.title("Starting Point")
plt.show()

plt.imshow(x_gt)
plt.title("Ground Truth")
plt.show()

plt.imshow(y)
plt.title("Measurements")
plt.show()

# %%
