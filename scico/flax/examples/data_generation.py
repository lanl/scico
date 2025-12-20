# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionality to generate training data for Flax example scripts.

Computation is distributed via ray to reduce processing time.
"""

from functools import partial
from time import time
from typing import Callable, List, Tuple, Union

import numpy as np

try:
    import xdesign  # noqa: F401
except ImportError:
    have_xdesign = False

    # pylint: disable=missing-class-docstring
    class UnitCircle:
        pass

    # pylint: enable=missing-class-docstring
else:
    have_xdesign = True
    from xdesign import (  # type: ignore
        Foam,
        SimpleMaterial,
        UnitCircle,
        discrete_phantom,
    )

try:
    import os

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # suppress ray warning
    import ray  # noqa: F401
except ImportError:
    have_ray = False
else:
    have_ray = True

import jax
import jax.numpy as jnp

try:
    from jax.extend.backend import get_backend  # introduced in jax 0.4.33
except ImportError:
    from jax.lib.xla_bridge import get_backend

from scico.linop import CircularConvolve
from scico.linop.xray import XRayTransform2D
from scico.numpy import Array


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
        if porosity < 0 or porosity > 1:
            raise ValueError("Argument 'porosity' must be in the range [0,1).")
        super().__init__(radius=0.5, material=SimpleMaterial(attn1))  # type: ignore
        self.sprinkle(  # type: ignore
            300, size_range, gap, material=SimpleMaterial(attn2), max_density=porosity / 2.0
        ) + self.sprinkle(  # type: ignore
            300, size_range, gap, material=SimpleMaterial(20), max_density=porosity
        )


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
    if not have_xdesign:
        raise RuntimeError("Package xdesign is required for use of this function.")
    np.random.seed(seed)
    saux: np.ndarray = np.zeros((ndata, size, size, 1), dtype=np.float32)
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
    if not have_xdesign:
        raise RuntimeError("Package xdesign is required for use of this function.")
    np.random.seed(seed)
    saux: np.ndarray = np.zeros((ndata, size, size, 1), dtype=np.float32)
    for i in range(ndata):
        foam = Foam2(size_range=[0.075, 0.0025], gap=1e-3, porosity=1)
        saux[i, ..., 0] = discrete_phantom(foam, size=size)
    # normalize
    saux /= np.max(saux, axis=(1, 2), keepdims=True)

    return saux


def vector_f(f_: Callable, v: Array) -> Array:
    """Vectorize application of operator.

    Args:
        f_: Operator to apply.
        v:  Array to evaluate.

    Returns:
       Result of evaluating operator over given arrays.
    """
    lf = lambda x: jnp.atleast_3d(f_(x.squeeze()))
    auto_batch = jax.vmap(lf)
    return auto_batch(v)


def batched_f(f_: Callable, vr: Array) -> Array:
    """Distribute application of operator over a batch of vectors
       among available processes.

    Args:
        f_: Operator to apply.
        vr: Batch of arrays to evaluate.

    Returns:
       Result of evaluating operator over given batch of arrays. This
       evaluation preserves the batch axis.
    """
    nproc = jax.device_count()
    if vr.shape[0] != nproc:
        vrr = vr.reshape((nproc, -1, *vr.shape[:1]))
    else:
        vrr = vr
    res = jax.pmap(partial(vector_f, f_))(vrr)
    return res


def generate_ct_data(
    nimg: int,
    size: int,
    nproj: int,
    imgfunc: Callable = generate_foam2_images,
    seed: int = 1234,
    verbose: bool = False,
) -> Tuple[Array, Array, Array]:
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

    Returns:
       tuple: A tuple (img, sino, fbp) containing:

           - **img** : (:class:`jax.Array`): Generated foam images.
           - **sino** : (:class:`jax.Array`): Corresponding sinograms.
           - **fbp** : (:class:`jax.Array`) Corresponding filtered back projections.
    """
    if not (have_ray and have_xdesign):
        raise RuntimeError("Packages ray and xdesign are required for use of this function.")

    # Generate input data.
    start_time = time()
    img = distributed_data_generation(imgfunc, size, nimg, seed)
    time_dtgen = time() - start_time
    # clip to [0,1] range
    img = jnp.clip(img, 0, 1)

    nproc = jax.device_count()
    if img.shape[0] % nproc > 0:
        # Decrease nimg to be a multiple of nproc if it isn't already
        nimg = (img.shape[0] // nproc) * nproc
        img = img[:nimg]

    # Configure a CT projection operator to generate synthetic measurements.
    angles = np.linspace(0, jnp.pi, nproj)  # evenly spaced projection angles
    gt_shape = (size, size)
    dx = 1.0 / np.sqrt(2)
    det_count = int(size * 1.05 / np.sqrt(2.0))
    A = XRayTransform2D(gt_shape, angles, dx=dx, det_count=det_count)
    # Compute sinograms in parallel.
    start_time = time()
    if nproc > 1:
        # shard array
        imgshd = img.reshape((nproc, -1, size, size, 1))
        sinoshd = batched_f(A, imgshd)
        sino = sinoshd.reshape((-1, nproj, sinoshd.shape[-2], 1))
    else:
        sino = vector_f(A, img)

    time_sino = time() - start_time

    # Compute filtered back-projection in parallel.
    start_time = time()
    if nproc > 1:
        fbpshd = batched_f(A.fbp, sinoshd)
        fbp = fbpshd.reshape((-1, size, size, 1))
    else:
        fbp = vector_f(A.fbp, sino)
    time_fbp = time() - start_time

    # Normalize sinogram.
    sino = sino / size
    # Clip FBP to [0,1] range.
    fbp = np.clip(fbp, 0, 1)

    if verbose:  # pragma: no cover
        platform = get_backend().platform
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
    imgfunc: Callable = generate_foam1_images,
    seed: int = 4321,
    verbose: bool = False,
) -> Tuple[Array, Array]:
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

    Returns:
       tuple: A tuple (img, blurn) containing:

           - **img** : Generated foam images.
           - **blurn** : Corresponding blurred and noisy images.
    """
    if not (have_ray and have_xdesign):
        raise RuntimeError("Packages ray and xdesign are required for use of this function.")
    start_time = time()
    img = distributed_data_generation(imgfunc, size, nimg, seed)
    time_dtgen = time() - start_time

    # Clip to [0,1] range.
    img = jnp.clip(img, 0, 1)

    nproc = jax.device_count()
    if img.shape[0] % nproc > 0:
        # Decrease nimg to be a multiple of nproc if it isn't already
        nimg = (img.shape[0] // nproc) * nproc
        img = img[:nimg]

    # Configure blur operator
    ishape = (size, size)
    A = CircularConvolve(h=blur_kernel, input_shape=ishape)

    # Compute blurred images in parallel
    start_time = time()
    if nproc > 1:
        # Shard array
        imgshd = img.reshape((nproc, -1, size, size, 1))
        blurshd = batched_f(A, imgshd)
        blur = blurshd.reshape((-1, size, size, 1))
    else:
        blur = vector_f(A, img)
    time_blur = time() - start_time
    # Normalize blurred images
    blur = blur / jnp.max(blur, axis=(1, 2), keepdims=True)
    # Add Gaussian noise
    key = jax.random.key(seed)
    noise = jax.random.normal(key, blur.shape)
    blurn = blur + noise_sigma * noise
    # Clip to [0,1] range.
    blurn = jnp.clip(blurn, 0, 1)

    if verbose:  # pragma: no cover
        platform = get_backend().platform
        print(f"{'Platform':26s}{':':4s}{platform}")
        print(f"{'Device count':26s}{':':4s}{jax.device_count()}")
        print(f"{'Data generation':19s}{'time[s]:':10s}{time_dtgen:>7.2f}")
        print(f"{'Blur generation':19s}{'time[s]:':10s}{time_blur:>7.2f}")

    return img, blurn


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
    if not have_ray:
        raise RuntimeError("Package ray is required for use of this function.")
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized via ray.init() before calling this function.")

    # Use half of available CPU resources
    ar = ray.available_resources()
    nproc = max(int(ar.get("CPU", 1)) // 2, 1)

    # Attempt to avoid ray/jax conflicts. This solution is a nasty hack that
    # can severely limit parallel execution (since ray will ensure that only
    # as many actors as available GPUs are created), and is expected to be
    # rather brittle.
    if "GPU" in ar:
        num_gpus = 1
        nproc = min(nproc, int(ar.get("GPU")))
    else:
        num_gpus = 0

    if nproc > nimg:
        nproc = nimg
    if nimg % nproc > 0:
        # Increase nimg to be a multiple of nproc if it isn't already
        nimg = (nimg // nproc + 1) * nproc

    ndata_per_proc = int(nimg // nproc)

    @ray.remote(num_gpus=num_gpus)
    def data_gen(seed, size, ndata, imgf):
        return imgf(seed, size, ndata)

    ray_return = ray.get(
        [data_gen.remote(seed + seedg, size, ndata_per_proc, imgenf) for seed in range(nproc)]
    )
    imgs = np.vstack([t for t in ray_return])

    return imgs
