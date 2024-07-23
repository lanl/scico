# -*- coding: utf-8 -*-
# Copyright (C) 2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Generate training data for Flax example scripts using ray.

Functions for generating xdesign foam phantoms and generation in parallel
using ray.
"""

import os
from typing import Callable, List, Union

import numpy as np

try:
    import ray  # noqa: F401
except ImportError:
    raise RuntimeError("Package ray is required for use of this module.")

try:
    import xdesign  # noqa: F401
except ImportError:
    raise RuntimeError("Package xdesign is required for use of this module.")
from xdesign import Foam, SimpleMaterial, UnitCircle, discrete_phantom


class Foam2(UnitCircle):
    """Foam-like material with two attenuations.

    Define functionality to generate phantom with structure similar
    to foam with two different attenuation properties."""

    def __init__(
        self,
        size_range: Union[float, List[float]] = [0.05, 0.01],
        gap: float = 0,
        porosity: float = 1,
        attn1: float = 1.0,
        attn2: float = 10.0,
    ):
        """Foam-like structure with two different attenuations.
        Circles for material 1 are more sparse than for material 2
        by design.

        Args:
            size_range: The radius, or range of radius, of the
                circles to be added. Default: [0.05, 0.01].
            gap: Minimum distance between circle boundaries.
                Default: 0.
            porosity: Target porosity. Must be a value between
                [0, 1]. Default: 1.
            attn1: Mass attenuation parameter for material 1.
                Default: 1.
            attn2: Mass attenuation parameter for material 2.
                Default: 10.
        """
        super().__init__(radius=0.5, material=SimpleMaterial(attn1))
        if porosity < 0 or porosity > 1:
            raise ValueError("Porosity must be in the range [0,1).")
        self.sprinkle(
            300, size_range, gap, material=SimpleMaterial(attn2), max_density=porosity / 2.0
        ) + self.sprinkle(300, size_range, gap, material=SimpleMaterial(20), max_density=porosity)


def generate_foam1_images(seed: float, size: int, ndata: int) -> np.ndarray:
    """Generate batch of xdesign foam-like structures.

    Generate batch of images with `xdesign` foam-like structure, which
    uses one attenuation.

    Args:
        seed: Seed for data generation.
        size: Size of image to generate.
        ndata: Number of images to generate.

    Returns:
        Array of generated data.
    """
    np.random.seed(seed)
    saux = np.zeros((ndata, size, size, 1), dtype=np.float32)
    for i in range(ndata):
        foam = Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1)
        saux[i, ..., 0] = discrete_phantom(foam, size=size)

    return saux


def generate_foam2_images(seed: float, size: int, ndata: int) -> np.ndarray:
    """Generate batch of foam2 structures.

    Generate batch of images with :class:`Foam2` structure
    (foam-like material with two different attenuations).

    Args:
        seed: Seed for data generation.
        size: Size of image to generate.
        ndata: Number of images to generate.

    Returns:
        Array of generated data.
    """
    np.random.seed(seed)
    saux = np.zeros((ndata, size, size, 1), dtype=np.float32)
    for i in range(ndata):
        foam = Foam2(size_range=[0.075, 0.0025], gap=1e-3, porosity=1)
        saux[i, ..., 0] = discrete_phantom(foam, size=size)
    # normalize
    saux /= np.max(saux, axis=(1, 2), keepdims=True)

    return saux


def distributed_data_generation(
    imgenf: Callable, size: int, nimg: int, seedg: float = 123
) -> np.ndarray:
    """Data generation distributed among processes using ray.

    *Warning:* callable `imgenf` should not make use of any jax functions
    to avoid the risk of errors when running with GPU devices, in which
    case jax is initialized to expect the availability of GPUs, which are
    then not available within the `ray.remote` function due to the absence
    of any declared GPUs as a `num_gpus` parameter of `@ray.remote`.

    Args:
        imagenf: Function for batch-data generation.
        size: Size of image to generate.
        ndata: Number of images to generate.
        seedg: Base seed for data generation.

    Returns:
        Array of generated data.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized via ray.init() before calling this function.")

    # Use half of available CPU resources
    ar = ray.available_resources()
    nproc = max(int(ar.get("CPU", 1)) // 2, 1)
    if nproc > nimg:
        nproc = nimg
    if nproc > 1 and nimg % nproc > 0:
        raise ValueError(
            f"Number of images to generate ({nimg}) must be divisible by "
            f"the number of available devices ({nproc})."
        )

    ndata_per_proc = int(nimg // nproc)

    # Attempt to avoid ray/jax conflicts.
    if "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" in os.environ:
        ray_noset_cuda = os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]
    else:
        ray_noset_cuda = None
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["JAX_PLATFORMS"] = "cpu"

    @ray.remote(num_gpus=0.001)
    def data_gen(seed, size, ndata, imgf):
        import os
        import sys

        os.environ["JAX_PLATFORMS"] = "cpu"
        sys.modules.pop("jax")
        sys.modules.pop("scico")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            del os.environ["CUDA_VISIBLE_DEVICES"]
        return imgf(seed, size, ndata)

    ray_return = ray.get(
        [data_gen.remote(seed + seedg, size, ndata_per_proc, imgenf) for seed in range(nproc)]
    )
    imgs = np.vstack([t for t in ray_return])

    if ray_noset_cuda is not None:
        os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = ray_noset_cuda

    return imgs
