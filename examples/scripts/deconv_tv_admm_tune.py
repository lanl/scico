import numpy as np

import jax

from xdesign import SiemensStar, discrete_phantom

import scico.numpy as snp
import scico.random
import scico.ray as ray
from scico import functional, linop, loss, metric
from scico.admm import ADMM, LinearSubproblemSolver
from scico.ray import tune

"""
Create a ground truth image.
"""
phantom = SiemensStar(32)
x_gt = snp.pad(discrete_phantom(phantom, 120), 8)


"""
Set up the forward operator and create a test signal consisting of a
blurred signal with additive Gaussian noise.
"""
n = 5  # convolution kernel size
σ = 20.0 / 255  # noise level

psf = snp.ones((n, n)) / (n * n)
A = linop.Convolve(h=psf, input_shape=x_gt.shape)

Ax = A(x_gt)  # blurred image
noise, key = scico.random.randn(Ax.shape, seed=0)
y = Ax + σ * noise


ray_x_gt, ray_psf, ray_y = ray.put(np.array(x_gt)), ray.put(np.array(psf)), ray.put(np.array(y))


def eval_params(config):
    λ, ρ = config["lambda"], config["rho"]
    x_gt, psf, y = ray.get([ray_x_gt, ray_psf, ray_y])
    x_gt, psf, y = jax.device_put([x_gt, psf, y])
    A = linop.Convolve(h=psf, input_shape=x_gt.shape)
    f = loss.SquaredL2Loss(y=y, A=A)
    g = λ * functional.L1Norm()
    C = linop.FiniteDifference(input_shape=x_gt.shape)
    solver = ADMM(
        f=f,
        g_list=[g],
        C_list=[C],
        rho_list=[ρ],
        x0=A.adj(y),
        maxiter=5,
        subproblem_solver=LinearSubproblemSolver(),
        verbose=False,
    )
    for step in range(10):
        x_admm = solver.solve()
        tune.report(psnr=float(metric.psnr(x_gt, x_admm)))


config = {"lambda": tune.loguniform(1e-2, 1), "rho": tune.loguniform(1e-1, 1e1)}
resources = {"gpu": 0, "cpu": 1}  # gpus per trial, cpus per trial
analysis = tune.run(
    eval_params,
    metric="psnr",
    mode="max",
    num_samples=100,
    config=config,
    resources_per_trial=resources,
    verbose=True,
)

best_config = analysis.get_best_config(metric="psnr", mode="max")
print("Best config: " + ", ".join([f"{k}: {v:.2e}" for k, v in best_config.items()]))
print(f"Best psnr: {analysis.get_best_trial().last_result['psnr']:.2f} dB")
