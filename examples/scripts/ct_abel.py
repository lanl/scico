# %%
import numpy as np

import matplotlib.pyplot as plt

# %%

data_dir = "/nh/netscratch/smajee/Data/Dynamic/Hydro/Hydro_data_2D/"
fname_x = data_dir + "density_t_6MeV_all_quad_256_case0_test.npy"

x_gt = np.load(fname_x)[40]
x_gt = x_gt / np.max(x_gt)
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
λ = 2e-0  # L1 norm regularization parameter
ρ = 5e-0  # ADMM penalty parameter
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
