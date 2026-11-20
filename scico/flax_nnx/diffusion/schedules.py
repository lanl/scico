# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

r"""Schedule functions for variance in diffusion generative models.

Each function returns the sequence :math:`\beta_1, \ldots, \beta_T` of
per-step variances of the Gaussian noise added by the forward diffusion
process, where :math:`T` is the number of time steps. The reverse process
is parameterized in terms of :math:`\alpha_t = 1 - \beta_t` and its
cumulative product, as in :class:`~scico.flax_nnx.diffusion.trainer_ddpm.FlaxNNXDDPMTrainer`.
The schedules differ in how the variance grows with :math:`t`.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from numpy import pi

import jax.numpy as jnp

from flax.nnx import sigmoid
from scico.numpy import Array


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Array:
    r"""Cosine beta schedule :cite:`nichol-2021-improved`.

    Args:
        timesteps: Number of diffusion time steps :math:`T`, i.e. the
            length of the returned schedule.
        s: Small offset preventing :math:`\beta_t` from being too small
            near :math:`t = 0`.

    Returns:
        Array of `timesteps` per-step variances, clipped to
        :math:`[10^{-4}, 1 - 10^{-4}]`.
    """
    steps = timesteps + 1
    x = jnp.linspace(0, timesteps, steps)
    alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> Array:
    r"""Linear beta schedule, increasing uniformly from start to end.

    Args:
        timesteps: Number of diffusion time steps :math:`T`, i.e. the
            length of the returned schedule.
        beta_start: Variance of the first step of the schedule.
        beta_end: Variance of the final step of the schedule.

    Returns:
        Array of `timesteps` per-step variances.
    """
    return jnp.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> Array:
    r"""Quadratic beta schedule, linear in :math:`\sqrt{\beta}`.

    Args:
        timesteps: Number of diffusion time steps :math:`T`, i.e. the
            length of the returned schedule.
        beta_start: Variance of the first step of the schedule.
        beta_end: Variance of the final step of the schedule.

    Returns:
        Array of `timesteps` per-step variances.
    """
    return jnp.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> Array:
    r"""Sigmoid beta schedule, mapping a sigmoid over :math:`[-6, 6]`.

    Args:
        timesteps: Number of diffusion time steps :math:`T`, i.e. the
            length of the returned schedule.
        beta_start: Variance of the first step of the schedule.
        beta_end: Variance of the final step of the schedule.

    Returns:
        Array of `timesteps` per-step variances.
    """
    betas = jnp.linspace(-6, 6, timesteps)
    return sigmoid(betas) * (beta_end - beta_start) + beta_start
