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
from jax.random import PRNGKey
from jax.typing import ArrayLike

import flax.nnx

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


def sampling_ddpm(
    batch_size: int,
    xshape: Tuple[int],
    maxsteps: int,
    alpha_schedule: ArrayLike,
    alpha_bar_schedule: ArrayLike,
    sigma: ArrayLike,
    model: Callable,
    key: PRNGKey,
    return_path: bool = False,
):
    """Function to sample using Denoising Diffusion Probabilistic Models
    (DDPM) formulation.

    Args:
        batch_size: Size of sample to generate.
        xshape: Shape of signal to generate.
        maxsteps: Maximum steps to use in DDPM. Allows for different range than
            the one used in training.
        alpha_schedule: Function of variance (beta) schedule. Allows for different schedule
            than the one used in training.
        alpha_bar_schedule: Function of alpha schedule. Allows for different schedule
            than the one used in training.
        sigma: Noise schedule, usually function of beta schedule. Allows for different
            schedule than the one used in training.
        model: Trained model to sample from.
        key: Key for jax random generation.
        return_path: Flag to indicate if generation path is to be returned.

    Returns:
        Array with generated samples, generation path (optional) and key.
    """
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (batch_size,) + xshape)
    if return_path:
        xpath = jnp.zeros((maxsteps + 1,) + x.shape)
        xpath = xpath.at[0].set(x)
    scale = (1.0 - alpha_schedule) / jnp.sqrt(1.0 - alpha_bar_schedule)
    t_int = maxsteps
    while t_int > 0:
        t = float(t_int) / maxsteps
        x = (x - model(x, t) * scale[t_int - 1]) / jnp.sqrt(alpha_schedule[t_int - 1])
        if t_int > 1:
            key, subkey = jax.random.split(key)
            z = jax.random.normal(subkey, x.shape)
            x = x + sigma[t_int - 1] * z
        if return_path:
            xpath = xpath.at[t_int].set(x)
        t_int = t_int - 1
    if return_path:
        return x, xpath, key
    return x, key
