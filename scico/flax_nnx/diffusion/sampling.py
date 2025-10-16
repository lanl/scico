# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of functions to generate new samples from diffusion models."""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

PyTree = Any


def score_fn(score_model: Callable, x: ArrayLike, t: ArrayLike):
    """Function to evaluate score model.

    Args:
        model: Score model to generate samples from.
        x: Current state to evaluate.
        t: Current time to evaluate.
    """
    return score_model(x, t)


# Define parallel evaluation of model
p_score_fn = jax.pmap(score_fn, static_broadcasted_argnums=(0,))


def Euler_Maruyama_sampler(
    key: ArrayLike,
    score_model: flax.nnx.Module,
    stddev_prior: float,
    xshape: Tuple[int],
    num_steps: int,
    batch_size: int,
    eps: float = 1e-3,
):
    """Euler-Maruyama sampling.

    Generate samples from score-based models with the Euler-Maruyama
    sampler.

    Args:
        key: A JAX random state.
        score_model: A `flax.linen.Module` object that represents the
            architecture of a score-based model.
        stddev_prior: Standard deviation of prior noise.
        xshape: Shape of signal to generate.
        num_steps: The number of sampling steps.
            Equivalent to the number of discretized time steps.
        batch_size: The number of samplers to generate by calling this
            function once.
        eps: The smallest time step for numerical stability.

    Returns:
        Samples.
    """
    score_model.eval()

    time_shape = (jax.local_device_count(), batch_size // jax.local_device_count(), 1)
    sample_shape = (jax.local_device_count(), batch_size // jax.local_device_count(), *xshape)
    x_tot = jnp.zeros(
        (jax.local_device_count(), batch_size // jax.local_device_count(), num_steps, *xshape)
    )
    key, step_key = jax.random.split(key)
    t = 1.0
    marginal_prob_std = jnp.sqrt((stddev_prior ** (2 * t) - 1.0) / 2.0 / jnp.log(stddev_prior))
    x = jax.random.normal(step_key, sample_shape) * marginal_prob_std
    time_steps = jnp.linspace(1.0, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    for ind, time_step in enumerate(time_steps):
        x_tot = x_tot.at[:, :, ind, :].set(x)
        batch_time_step = jnp.ones(time_shape) * time_step
        g = stddev_prior**time_step
        mean_x = x + (g**2) * p_score_fn(score_model, x, batch_time_step) * step_size
        key, step_key = jax.random.split(key)
        x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(step_key, x.shape)
    # Do not include any noise in the last sampling step.
    return mean_x, x_tot
