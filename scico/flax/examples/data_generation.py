# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionality to generate training data for Flax example scripts.

Computation is distributed via ray (if available) or jax or to reduce
processing time.
"""

import os
from time import time
from typing import Callable, List, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp

try:
    import xdesign  # noqa: F401
except ImportError:
    have_xdesign = False
else:
    have_xdesign = True

try:
    import ray  # noqa: F401
except ImportError:
    have_ray = False
else:
    have_ray = True

if have_xdesign:
    from xdesign import Foam, SimpleMaterial, UnitCircle, discrete_phantom

from scico.linop import CircularConvolve
from scico.numpy import Array

try:
    import astra  # noqa: F401
except ImportError:
    have_astra = False
else:
    have_astra = True

if have_astra:
    from scico.linop.radon_astra import TomographicProjector


# Arbitrary process count: only applies if GPU is not available.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


if have_xdesign:

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
            super(Foam2, self).__init__(radius=0.5, material=SimpleMaterial(attn1))
            if porosity < 0 or porosity > 1:
                raise ValueError("Porosity must be in the range [0,1).")
            self.sprinkle(
                300, size_range, gap, material=SimpleMaterial(attn2), max_density=porosity / 2.0
            ) + self.sprinkle(
                300, size_range, gap, material=SimpleMaterial(20), max_density=porosity
            )


def generate_foam2_images(seed: float, size: int, ndata: int) -> Array:
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
    if not have_xdesign:
        raise RuntimeError("Package xdesign is required for use of this function.")

    np.random.seed(seed)
    saux = np.zeros((ndata, size, size, 1))
    for i in range(ndata):
        foam = Foam2(size_range=[0.075, 0.0025], gap=1e-3, porosity=1)
        saux[i, ..., 0] = discrete_phantom(foam, size=size)

    # normalize
    saux = saux / np.max(saux, axis=(1, 2), keepdims=True)

    return saux


def generate_foam1_images(seed: float, size: int, ndata: int) -> Array:
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
    if not have_xdesign:
        raise RuntimeError("Package xdesign is required for use of this function.")

    np.random.seed(seed)
    saux = np.zeros((ndata, size, size, 1))
    for i in range(ndata):
        foam = Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1)
        saux[i, ..., 0] = discrete_phantom(foam, size=size)

    return saux


def generate_ct_data(
    nimg: int,
    size: int,
    nproj: int,
    imgfunc: Callable = generate_foam2_images,
    seed: int = 1234,
    verbose: bool = False,
    test_flag: bool = False,
    prefer_ray: bool = True,
) -> Tuple[Array, ...]:
    """Generate batch of computed tomography (CT) data.

    Generate batch of CT data for training of machine learning network
    models.

    Args:
        nimg: Number of images to generate.
        size: Size of reconstruction images.
        nproj: Number of CT views.
        imgfunc: Function for generating input images (e.g. foams).
        seed: Seed for data generation.
        verbose: Flag indicating whether to print status messages.
            Default: ``False``.
        test_flag: Flag to indicate if running in testing mode. Testing
            mode requires a different initialization of ray. Default:
            ``False``.
        prefer_ray: Use ray for distributed processing if available.
            Default: ``True``.

    Returns:
       tuple: A tuple (img, sino, fbp) containing:

           - **img** : (:class:`jax.Array`): Generated foam images.
           - **sino** : (:class:`jax.Array`): Corresponding sinograms.
           - **fbp** : (:class:`jax.Array`) Corresponding filtered back projections.
    """
    if not have_astra:
        raise RuntimeError("Package astra is required for use of this function.")

    # Generate input data.
    if have_ray and prefer_ray:
        start_time = time()
        img = ray_distributed_data_generation(imgfunc, size, nimg, seed, test_flag)
        time_dtgen = time() - start_time
    else:
        start_time = time()
        img = imgfunc(seed, size, nimg)
        time_dtgen = time() - start_time
    # Clip to [0,1] range.
    img = jnp.clip(img, a_min=0, a_max=1)
    # Shard array
    nproc = jax.device_count()
    imgshd = img.reshape((nproc, -1, size, size, 1))

    # Configure a CT projection operator to generate synthetic measurements.
    angles = np.linspace(0, jnp.pi, nproj)  # evenly spaced projection angles
    gt_sh = (size, size)
    detector_spacing = 1
    A = TomographicProjector(gt_sh, detector_spacing, size, angles)  # Radon transform operator

    # Compute sinograms in parallel.
    a_map = lambda v: jnp.atleast_3d(A @ v.squeeze())
    start_time = time()
    sinoshd = jax.pmap(lambda i: jax.lax.map(a_map, imgshd[i]))(jnp.arange(nproc))
    time_sino = time() - start_time
    sino = sinoshd.reshape((-1, nproj, size, 1))
    # Normalize sinogram
    sino = sino / size

    # Compute filter back-project in parallel.
    afbp_map = lambda v: jnp.atleast_3d(A.fbp(v.squeeze()))
    start_time = time()
    fbpshd = jax.pmap(lambda i: jax.lax.map(afbp_map, sinoshd[i]))(jnp.arange(nproc))
    time_fbp = time() - start_time
    # Clip to [0,1] range.
    fbpshd = jnp.clip(fbpshd, a_min=0, a_max=1)
    fbp = fbpshd.reshape((-1, size, size, 1))

    if verbose:  # pragma: no cover
        platform = jax.lib.xla_bridge.get_backend().platform
        print(f"{'Platform':26s}{':':4s}{platform}")
        print(f"{'Device count':26s}{':':4s}{jax.device_count()}")
        print(f"{'Data generation':19s}{'time[s]:':10s}{time_dtgen:>7.2f}")
        print(f"{'Sinogram':19s}{'time[s]:':10s}{time_sino:>7.2f}")
        print(f"{'FBP':19s}{'time[s]:':10s}{time_fbp:>7.2f}")

    return img, sino, fbp


def generate_blur_data(
    nimg: int,
    size: int,
    blur_kernel: Array,
    noise_sigma: float,
    imgfunc: Callable,
    seed: int = 4321,
    verbose: bool = False,
    test_flag: bool = False,
    prefer_ray: bool = True,
) -> Tuple[Array, ...]:
    """Generate batch of blurred data.

    Generate batch of blurred data for training of machine learning
    network models.

    Args:
        nimg: Number of images to generate.
        size: Size of reconstruction images.
        blur_kernel: Kernel for blurring the generated images.
        noise_sigma: Level of additive Gaussian noise to apply.
        imgfunc: Function to generate foams.
        seed: Seed for data generation.
        verbose: Flag indicating whether to print status messages.
            Default: ``False``.
        test_flag: Flag to indicate if running in testing mode. Testing
            mode requires a different initialization of ray.
            Default: ``False``.
        prefer_ray: Use ray for distributed processing if available.
            Default: ``True``.

    Returns:
       tuple: A tuple (img, blurn) containing:

           - **img** : Generated foam images.
           - **blurn** : Corresponding blurred and noisy images.
    """
    if have_ray and prefer_ray:
        start_time = time()
        img = ray_distributed_data_generation(imgfunc, size, nimg, seed, test_flag)
        time_dtgen = time() - start_time
    else:
        start_time = time()
        img = imgfunc(seed, size, nimg)
        time_dtgen = time() - start_time
    # Clip to [0,1] range.
    img = jnp.clip(img, a_min=0, a_max=1)
    # Shard array
    nproc = jax.device_count()
    imgshd = img.reshape((nproc, -1, size, size, 1))

    # Configure blur operator
    ishape = (size, size)
    A = CircularConvolve(h=blur_kernel, input_shape=ishape)

    # Compute blurred images in parallel
    a_map = lambda v: jnp.atleast_3d(A @ v.squeeze())
    start_time = time()
    blurshd = jax.pmap(lambda i: jax.lax.map(a_map, imgshd[i]))(jnp.arange(nproc))
    time_blur = time() - start_time
    blur = blurshd.reshape((-1, size, size, 1))
    # Normalize blurred images
    blur = blur / jnp.max(blur, axis=(1, 2), keepdims=True)
    # Add Gaussian noise
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, blur.shape)
    blurn = blur + noise_sigma * noise
    # Clip to [0,1] range.
    blurn = jnp.clip(blurn, a_min=0, a_max=1)

    if verbose:  # pragma: no cover
        platform = jax.lib.xla_bridge.get_backend().platform
        print(f"{'Platform':26s}{':':4s}{platform}")
        print(f"{'Device count':26s}{':':4s}{jax.device_count()}")
        print(f"{'Data generation':19s}{'time[s]:':10s}{time_dtgen:>7.2f}")
        print(f"{'Blur generation':19s}{'time[s]:':10s}{time_blur:>7.2f}")

    return img, blurn


def distributed_data_generation(
    imgenf: Callable, size: int, nimg: int, sharded: bool = True
) -> Array:
    """Data generation distributed among processes using jax.

    Args:
        imagenf: Function for batch-data generation.
        size: Size of image to generate.
        ndata: Number of images to generate.
        sharded: Flag to indicate if data is to be returned as the
            chunks generated by each process or consolidated.
            Default: ``True``.

    Returns:
        Array of generated data.
    """
    nproc = jax.device_count()
    seeds = jnp.arange(nproc)
    if nproc > 1 and nimg % nproc > 0:
        raise ValueError("Number of images to generate must be divisible by the number of devices")

    ndata_per_proc = int(nimg // nproc)

    imgs = jax.pmap(imgenf, static_broadcasted_argnums=(1, 2))(seeds, size, ndata_per_proc)

    if not sharded:
        imgs = imgs.reshape((-1, size, size, 1))

    return imgs


def ray_distributed_data_generation(
    imgenf: Callable, size: int, nimg: int, seedg: float = 123, test_flag: bool = False
) -> Array:
    """Data generation distributed among processes using ray.

    Args:
        imagenf: Function for batch-data generation.
        size: Size of image to generate.
        ndata: Number of images to generate.
        seedg: Base seed for data generation. Default: 123.
        test_flag: Flag to indicate if running in testing mode. Testing
            mode requires a different initialization of ray. Default:
            ``False``.

    Returns:
        Array of generated data.
    """
    if not have_ray:
        raise RuntimeError("Package ray is required for use of this function.")

    if test_flag:
        ray.init(ignore_reinit_error=True)
    else:
        ray.init()

    @ray.remote
    def data_gen(seed, size, ndata, imgf):
        return imgf(seed, size, ndata)

    ar = ray.available_resources()
    # Usage of half available CPU resources.
    nproc = max(int(ar["CPU"]) // 2, 1)
    if nproc > nimg:
        nproc = nimg
    if nproc > 1 and nimg % nproc > 0:
        raise ValueError(
            f"Number of images to generate ({nimg}) "
            f"must be divisible by the number of available devices ({nproc})"
        )

    ndata_per_proc = int(nimg // nproc)

    ray_return = ray.get(
        [data_gen.remote(seed + seedg, size, ndata_per_proc, imgenf) for seed in range(nproc)]
    )
    imgs = np.vstack([t for t in ray_return])
    ray.shutdown()

    return imgs
