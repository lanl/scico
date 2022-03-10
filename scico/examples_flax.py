# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utility functions used by Flax example scripts."""

from xdesign import UnitCircle, SimpleMaterial, discrete_phantom

from typing import Any, Callable, List, Union

import jax
import jax.numpy as jnp

import os

from scico.typing import Array

# Arbitray process count: only applies if GPU is not available.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


class Foam2(UnitCircle):
    """Define functionality to generate phantom
    with structure similar to foam with two
    different attenuation properties."""

    def __init__(
        self,
        size_range: Union[float, List[float]] = [0.05, 0.01],
        gap: float = 0,
        porosity: float = 1,
        attn1: float = 1.0,
        attn2: float = 10.0,
    ):
        """Foam-like structure with two different
        attenuations. Circles for material 1 are
        more sparse than for material 2 by design.

        Args:
            size_range : The radius, or range of radius, of the circles to be added. Default: [0.05, 0.01].
            gap : Minimum distance between circle boundaries. Default: 0.
            porosity : Target porosity. Must be a value between [0, 1]. Default: 1.
            attn1 : Mass attenuation parameter for material 1. Default: 1.
            attn2 : Mass attenuation parameter for material 2. Default: 10.
        """
        super(Foam2, self).__init__(radius=0.5, material=SimpleMaterial(attn1))
        if porosity < 0 or porosity > 1:
            raise ValueError("Porosity must be in the range [0,1).")
        self.sprinkle(
            300, size_range, gap, material=SimpleMaterial(attn2), max_density=porosity / 2.0
        ) + self.sprinkle(300, size_range, gap, material=SimpleMaterial(20), max_density=porosity)


def generate_foam2_images(seed: float, size: int, ndata: int) -> Array:
    """Generation of batch of images with
    :class:`Foam2` structure (foam-like structure
    with two different attenuations).

    Args:
        seed : Seed for data generation.
        size : Size of image to generate.
        ndata : Number of images to generate.

    Returns:
        nd-array of generated data.
    """
    key = jax.random.PRNGKey(seed)  # In XDesign?
    oneimg = lambda _: jnp.atleast_3d(
        discrete_phantom(Foam2(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=size)
    )
    saux = jax.vmap(oneimg)(jnp.arange(ndata))

    return saux


def distributed_data_generation(
    imgenf: Callable, size: int, nimg: int, sharded: bool = True
) -> Array:
    """Data generation distributed among processes.

    Args:
        imagenf : Function for batch-data generation.
        size : Size of image to generate.
        ndata : Number of images to generate.
        sharded : Flag to indicate if data is to
            be returned as the chunks generated by
            each process or consolidated. Default: True.

    Returns:
        nd-array of generated data.
    """
    nproc = jax.device_count()
    seeds = jnp.arange(nproc)
    if nimg % nproc > 0:
        raise ValueError("Number of images to generate must be divisible by the number of devices")

    ndata_per_proc = nimg // nproc

    imgs = jax.pmap(imgenf, static_broadcasted_argnums=(1, 2))(seeds, size, ndata_per_proc)

    if not sharded:
        imgs = imgs.reshape((-1, size, size, 1))

    return imgs
