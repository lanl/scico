# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utility functions used by Flax example scripts."""

import glob
import math
import os
import tarfile
import tempfile
from time import time
from typing import Any, Callable, List, Optional, Tuple, TypedDict, Union

import numpy as np

import jax
import jax.numpy as jnp

import imageio

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

from scico import util
from scico.examples import rgb2gray
from scico.flax.train.input_pipeline import DataSetDict
from scico.linop import CircularConvolve, LinearOperator
from scico.typing import Array, Shape

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


class CTDataSetDict(TypedDict):
    """Definition of the dictionary structure
    constructed in CT data generation."""

    img: Array  # original image
    sino: Array  # sinogram
    fbp: Array  # filtered back projection


if have_xdesign:

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
                size_range: The radius, or range of radius, of the
                    circles to be added. Default: [0.05, 0.01].
                gap: Minimum distance between circle boundaries. Default: 0.
                porosity: Target porosity. Must be a value between [0, 1]. Default: 1.
                attn1: Mass attenuation parameter for material 1. Default: 1.
                attn2: Mass attenuation parameter for material 2. Default: 10.
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
    """Generation of batch of images with
    :class:`Foam2` structure (foam-like structure
    with two different attenuations).

    Args:
        seed: Seed for data generation.
        size: Size of image to generate.
        ndata: Number of images to generate.

    Returns:
        nd-array of generated data.
    """
    if not have_xdesign:
        raise RuntimeError("Package xdesign is required for use of this function.")

    np.random.seed(seed)
    saux = np.zeros((ndata, size, size, 1))
    for i in range(ndata):
        foam = Foam2(size_range=[0.075, 0.0025], gap=1e-3, porosity=1)
        saux[i, ..., 0] = discrete_phantom(foam, size=size)

    # Normalize
    saux = saux / np.max(saux, axis=(1, 2), keepdims=True)

    return saux


def generate_foam_images(seed: float, size: int, ndata: int) -> Array:
    """Generation of xdesign foam-like batch of images.

    Args:
        seed: Seed for data generation.
        size: Size of image to generate.
        ndata: Number of images to generate.

    Returns:
        nd-array of generated data.
    """
    if not have_xdesign:
        raise RuntimeError("Package xdesign is required for use of this function.")

    np.random.seed(seed)
    saux = np.zeros((ndata, size, size, 1))
    for i in range(ndata):
        foam = Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1)
        saux[i, ..., 0] = discrete_phantom(foam, size=size)

    return saux


def distributed_data_generation(
    imgenf: Callable, size: int, nimg: int, sharded: bool = True
) -> Array:
    """Data generation distributed among processes.

    Args:
        imagenf: Function for batch-data generation.
        size: Size of image to generate.
        ndata: Number of images to generate.
        sharded: Flag to indicate if data is to
            be returned as the chunks generated by
            each process or consolidated. Default: ``True``.

    Returns:
        nd-array of generated data.
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
        test_flag: Flag to indicate if running in testing mode. Testing mode requires a different initialization of ray. Default: ``False``.

    Returns:
        nd-array of generated data.
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
            f"Number of images to generate ({nimg}) must be divisible by the number of available devices ({nproc})"
        )

    ndata_per_proc = int(nimg // nproc)

    ray_return = ray.get(
        [data_gen.remote(seed + seedg, size, ndata_per_proc, imgenf) for seed in range(nproc)]
    )
    imgs = np.vstack([t for t in ray_return])
    ray.shutdown()

    return imgs


def ct_data_generation(
    nimg: int,
    size: int,
    nproj: int,
    imgfunc: Callable = generate_foam2_images,
    seed: int = 1234,
    verbose: bool = False,
    test_flag: bool = False,
) -> Tuple[Array, ...]:
    """
    Generate CT data.

    Generate CT data for training of machine
    learning network models.

    Args:
        nimg: Number of images to generate.
        size: Size of reconstruction images.
        nproj: Number of CT views.
        imgfunc: Function for generating input images (e.g. foams).
        seed: Seed for data generation.
        verbose: Flag indicating whether to print status messages. Default: ``False``.
        test_flag: Flag to indicate if running in testing mode. Testing mode requires a different initialization of ray. Default: ``False``.

    Returns:
       tuple: A tuple (img, sino, fbp) containing:

           - **img** : (DeviceArray): Generated foam images.
           - **sino** : (DeviceArray): Corresponding sinograms.
           - **fbp** : (DeviceArray) Corresponding filtered back projections.
    """
    if not have_astra:
        raise RuntimeError("Package astra is required for use of this function.")

    # Generate input data.
    if have_ray:
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

    if verbose:
        platform = jax.lib.xla_bridge.get_backend().platform
        print(f"{'Platform':28s}{':':4s}{platform}")
        print(f"{'Device count':28s}{':':4s}{jax.device_count()}")
        print(f"{'Data generation':21s}{'time[s]:':10s}{time_dtgen:>7.2f}")
        print(f"{'Sinogram generation':21s}{'time[s]:':10s}{time_sino:>7.2f}")
        print(f"{'FBP generation':21s}{'time[s]:':10s}{time_fbp:>7.2f}")

    return img, sino, fbp


def load_ct_data(
    train_nimg: int,
    test_nimg: int,
    size: int,
    nproj: int,
    cache_path: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[CTDataSetDict, ...]:  # pragma: no cover
    """
    Load or generate CT data.

    Load or generate CT data for training of machine learning network models.
    If cached file exists and enough data of the requested size is available, data is
    loaded and returned.

    If either `size` or `nproj` requested does
    not match the data read from the cached
    file, a `RunTimeError` is generated.

    If no cached file is found or not enough data
    is contained in the file a new data set
    is generated and stored in `cache_path`. The
    data is stored in `.npz` format for
    convenient access via :func:`numpy.load`.
    The data is saved in two distinct files:
    `ct_foam2_train.npz` and `ct_foam2_test.npz`
    to keep separated training and testing
    partitions.

    Args:
        train_nimg: Number of images required for training.
        test_nimg: Number of images required for testing.
        size: Size of reconstruction images.
        nproj: Number of CT views.
        cache_path: Directory in which generated data is saved. Default: ``None``.
        verbose: Flag indicating whether to print status messages. Default: ``False``.

    Returns:
       tuple: A tuple (trdt, ttdt) containing:

           - **trdt** : (Dictionary): Collection of images (key `img`),
               sinograms (key `sino`) and filtered back projections (key `fbp`) for training.
           - **ttdt** : (Dictionary): Collection of images (key `img`),
               sinograms (key `sino`) and filtered back projections (key `fbp`) for testing.
    """
    # Set default cache path if not specified.
    if cache_path is None:
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "data")
        subpath = str.split(cache_path, ".cache")
        cache_path_display = "~/.cache" + subpath[-1]
    else:
        cache_path_display = cache_path

    # Create cache directory and generate data if not already present.
    npz_train_file = os.path.join(cache_path, "ct_foam2_train.npz")
    npz_test_file = os.path.join(cache_path, "ct_foam2_test.npz")

    if os.path.isfile(npz_train_file) and os.path.isfile(npz_test_file):
        # Load data.
        trdt_in = np.load(npz_train_file)
        ttdt_in = np.load(npz_test_file)
        # Check image size.
        if (trdt_in["img"].shape[1] != size) or (ttdt_in["img"].shape[1] != size):
            raise RuntimeError(
                f"{'Provided size: '}{size}{' does not match read train size: '}"
                f"{trdt_in['img'].shape[1]}{' or test size: '}{ttdt_in['img'].shape[1]}"
            )
        # Check number of projections.
        if (trdt_in["sino"].shape[1] != nproj) or (ttdt_in["sino"].shape[1] != nproj):
            raise RuntimeError(
                f"{'Provided views: '}{nproj}{' does not match read train views: '}"
                f"{trdt_in['sino'].shape[1]}{' or test views: '}{ttdt_in['sino'].shape[1]}"
            )
        # Check that enough data is available.
        if trdt_in["img"].shape[0] >= train_nimg:
            if ttdt_in["img"].shape[0] >= test_nimg:
                trdt: CTDataSetDict = {
                    "img": trdt_in["img"][:train_nimg],
                    "sino": trdt_in["sino"][:train_nimg],
                    "fbp": trdt_in["fbp"][:train_nimg],
                }
                ttdt: CTDataSetDict = {
                    "img": ttdt_in["img"][:test_nimg],
                    "sino": ttdt_in["sino"][:test_nimg],
                    "fbp": ttdt_in["fbp"][:test_nimg],
                }
                if verbose:
                    print(f"{'Data read from path':28s}{':':4s}{cache_path_display}")
                    print(f"{'Train images':28s}{':':4s}{trdt['img'].shape[0]}")
                    print(f"{'Test images':28s}{':':4s}{ttdt['img'].shape[0]}")
                    print(
                        f"{'Data range images':25s}{'Min:':6s}{trdt['img'].min():>5.2f}"
                        f"{', Max:':6s}{trdt['img'].max():>5.2f}"
                    )
                    print(
                        f"{'Data range sinograms':25s}{'Min:':6s}{trdt['sino'].min():>5.2f}"
                        f"{', Max:':6s}{trdt['sino'].max():>5.2f}"
                    )
                    print(
                        f"{'Data range FBP':25s}{'Min:':6s}{trdt['fbp'].min():>5.2f}"
                        f"{', Max:':6s}{trdt['fbp'].max():>5.2f}"
                    )

                return trdt, ttdt

            elif verbose:
                print(
                    f"{'Not enough data in testing file':34s}{'Requested:':12s}{test_nimg}"
                    f"{' Available:':12s}{ttdt_in['img'].shape[0]}"
                )
        elif verbose:
            print(
                f"{'Not enough data in training file':34s}{'Requested:':12s}{train_nimg}"
                f"{' Available:':12s}{trdt_in['img'].shape[0]}"
            )

    # Generate new data.
    nimg = train_nimg + test_nimg
    img, sino, fbp = ct_data_generation(
        nimg,
        size,
        nproj,
        verbose=verbose,
    )
    # Separate training and testing partitions.
    trdt = {"img": img[:train_nimg], "sino": sino[:train_nimg], "fbp": fbp[:train_nimg]}
    ttdt = {"img": img[train_nimg:], "sino": sino[train_nimg:], "fbp": fbp[train_nimg:]}

    # Store images, sinograms and filtered back-projections.
    os.makedirs(cache_path, exist_ok=True)
    np.savez(
        npz_train_file,
        img=img[:train_nimg],
        sino=sino[:train_nimg],
        fbp=fbp[:train_nimg],
    )
    np.savez(
        npz_test_file,
        img=img[train_nimg:],
        sino=sino[train_nimg:],
        fbp=fbp[train_nimg:],
    )

    if verbose:
        print(f"{'Storing data in path':28s}{':':4s}{cache_path_display}")
        print(f"{'Train images':28s}{':':4s}{train_nimg}")
        print(f"{'Test images':28s}{':':4s}{test_nimg}")
        print(
            f"{'Data range images':25s}{'Min:':6s}{img.min():>5.2f}{', Max:':6s}{img.max():>5.2f}"
        )
        print(
            f"{'Range sinograms':25s}{'Min:':6s}{sino.min():>5.2f}{', Max:':6s}{sino.max():>5.2f}"
        )
        print(f"{'Range FBP':25s}{'Min:':6s}{fbp.min():>5.2f}{', Max:':6s}{fbp.max():>5.2f}")

    return trdt, ttdt


def blur_data_generation(
    nimg: int,
    size: int,
    blur_kernel: Array,
    noise_sigma: float,
    imgfunc: Callable,
    seed: int = 4321,
    verbose: bool = False,
    test_flag: bool = False,
) -> Tuple[Array, ...]:
    """
    Generate blurred data based on xdesign foam structures.

    Generate blurred data for training of machine
    learning network models.

    Args:
        nimg: Number of images to generate.
        size: Size of reconstruction images.
        blur_kernel: Kernel for blurring the generated images.
        noise_sigma: Level of additive Gaussian noise to apply.
        imgfunc: Function to generate foams.
        seed: Seed for data generation.
        verbose: Flag indicating whether to print status messages. Default: ``False``.
        test_flag: Flag to indicate if running in testing mode. Testing mode requires a different initialization of ray. Default: ``False``.

    Returns:
       tuple: A tuple (img, blurn) containing:

           - **img** : (DeviceArray): Generated foam images.
           - **blurn** : (DeviceArray) Corresponding blurred and noisy images.
    """
    if have_ray:
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
        print(f"{'Platform':28s}{':':4s}{platform}")
        print(f"{'Device count':28s}{':':4s}{jax.device_count()}")
        print(f"{'Data generation':21s}{'time[s]:':10s}{time_dtgen:>7.2f}")
        print(f"{'Blur generation':21s}{'time[s]:':10s}{time_blur:>7.2f}")

    return img, blurn


def load_foam_blur_data(
    train_nimg: int,
    test_nimg: int,
    size: int,
    blur_kernel: Array,
    noise_sigma: float,
    cache_path: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[DataSetDict, ...]:  # pragma: no cover
    """
    Load or generate blurred data based on xdesign foam structures.

    Load or generate blurred data for training of machine learning network models.
    If cached file exists and enough data of the requested size is available, data is
    loaded and returned.

    If `size` requested does not match the data read from the cached
    file, a `RunTimeError` is generated. In contrast, there is no checking for the
    specific contamination (i.e. noise level, blur
    kernel, etc.).

    If no cached file is found or not enough data
    is contained in the file a new data set
    is generated and stored in `cache_path`. The
    data is stored in `.npz` format for
    convenient access via :func:`numpy.load`.
    The data is saved in two distinct files:
    `dcnv_foam_train.npz` and `dcnv_foam_test.npz`
    to keep separated training and testing
    partitions.

    Args:
        train_nimg: Number of images required for training.
        test_nimg: Number of images required for testing.
        size: Size of reconstruction images.
        blur_kernel: Kernel for blurring the generated images.
        noise_sigma: Level of additive Gaussian noise to apply.
        cache_path: Directory in which generated data is saved. Default: ``None``.
        verbose: Flag indicating whether to print status messages. Default: ``False``.

    Returns:
       tuple: A tuple (train_ds, test_ds) containing:

           - **train_ds** : (DataSetDict): Dictionary of training data (includes images and labels).
           - **test_ds** : (DataSetDict): Dictionary of testing data (includes images and labels).
    """
    # Set default cache path if not specified.
    if cache_path is None:
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "data")
        subpath = str.split(cache_path, ".cache")
        cache_path_display = "~/.cache" + subpath[-1]
    else:
        cache_path_display = cache_path

    # Create cache directory and generate data if not already present.
    npz_train_file = os.path.join(cache_path, "dcnv_foam_train.npz")
    npz_test_file = os.path.join(cache_path, "dcnv_foam_test.npz")

    if os.path.isfile(npz_train_file) and os.path.isfile(npz_test_file):
        # Load data and convert arrays to float32.
        trdt = np.load(npz_train_file)  # Training
        ttdt = np.load(npz_test_file)  # Testing
        train_in = trdt["image"].astype(np.float32)
        train_out = trdt["label"].astype(np.float32)
        test_in = ttdt["image"].astype(np.float32)
        test_out = ttdt["label"].astype(np.float32)

        # Check image size.
        if (train_in.shape[1] != size) or (test_in.shape[1] != size):
            raise RuntimeError(
                f"{'Provided size: '}{size}{' does not match read train size: '}"
                f"{train_in.shape[1]}{' or test size: '}{test_in.shape[1]}"
            )

        # Check that enough images were restored.
        if trdt["numimg"] >= train_nimg:
            if ttdt["numimg"] >= test_nimg:
                train_ds: DataSetDict = {
                    "image": train_in,
                    "label": train_out,
                }
                test_ds: DataSetDict = {
                    "image": test_in,
                    "label": test_out,
                }
                if verbose:
                    print(f"{'Data read from path':28s}{':':4s}{cache_path_display}")
                    print(f"{'Train images':28s}{':':4s}{train_ds['image'].shape[0]}")
                    print(f"{'Test images':28s}{':':4s}{test_ds['image'].shape[0]}")
                    print(
                        f"{'Data range images':25s}{'Min:':6s}{train_ds['image'].min():>5.2f}"
                        f"{', Max:':6s}{train_ds['image'].max():>5.2f}"
                    )
                    print(
                        f"{'Data range labels':25s}{'Min:':6s}{train_ds['label'].min():>5.2f}"
                        f"{', Max:':6s}{train_ds['label'].max():>5.2f}"
                    )
                    print(
                        "NOTE: If blur kernel or noise parameter are changed, the cache"
                        " must be manually deleted to ensure that the training data "
                        " is regenerated with these new parameters."
                    )

                return train_ds, test_ds

            elif verbose:
                print(
                    f"{'Not enough images for testing in file':34s}{'Requested:':12s}"
                    f"{test_nimg}{'Available:':12s}{ttdt['numimg']}"
                )
        elif verbose:
            print(
                f"{'Not enough images for training in file':34s}{'Requested:':12s}{train_nimg}"
                f"{' Available:':12s}{trdt['numimg']}"
            )

    # Generate new data.
    nimg = train_nimg + test_nimg
    img, blrn = blur_data_generation(
        nimg,
        size,
        blur_kernel,
        noise_sigma,
        imgfunc=generate_foam_images,
        verbose=verbose,
    )
    # Separate training and testing partitions.
    train_ds = {"image": blrn[:train_nimg], "label": img[:train_nimg]}
    test_ds = {"image": blrn[train_nimg:], "label": img[train_nimg:]}

    # Store original and blurred images.
    os.makedirs(cache_path, exist_ok=True)
    np.savez(
        npz_train_file,
        image=train_ds["image"],
        label=train_ds["label"],
        numimg=train_nimg,
    )
    np.savez(
        npz_test_file,
        image=test_ds["image"],
        label=test_ds["label"],
        numimg=test_nimg,
    )

    if verbose:
        print(f"{'Storing data in path':28s}{':':4s}{cache_path_display}")
        print(f"{'Train images':28s}{':':4s}{train_ds['image'].shape[0]}")
        print(f"{'Test images':28s}{':':4s}{test_ds['image'].shape[0]}")
        print(
            f"{'Data range images':25s}{'Min:':6s}{train_ds['image'].min():>5.2f}"
            f"{', Max:':6s}{train_ds['image'].max():>5.2f}"
        )
        print(
            f"{'Data range labels':25s}{'Min:':6s}{train_ds['label'].min():>5.2f}"
            f"{', Max:':6s}{train_ds['label'].max():>5.2f}"
        )

    return train_ds, test_ds


# Image manipulation utils
def rotation90(img: Array) -> Array:
    """Rotates an image, or a batch of images,
    by 90 degrees.

    Rotates an image or a batch of images by 90
    degrees counterclockwise. An image is an
    nd-array with size H x W x C with H and W
    spatial dimensions and C number of channels.
    A batch of images is an nd-array with size
    N x H x W x C with N number of images.

    Args:
        img: The nd-array to be rotated.

    Returns:
       An image, or batch of images, rotated by 90 degrees counterclockwise.
    """
    if img.ndim < 4:
        return np.swapaxes(img, 0, 1)
    else:
        return np.swapaxes(img, 1, 2)


def flip(img: Array) -> Array:
    """Horizontal flip of an image or a batch of
    images.

    Horizontally flips an image or a batch of
    images. An image is an nd-array with size
    H x W x C with H and W spatial dimensions
    and C number of channels. A batch of images
    is an nd-array with size N x H x W x C with
    N number of images.

    Args:
        img: The nd-array to be flipped.

    Returns:
       An image, or batch of images, flipped horizontally.
    """
    if img.ndim < 4:
        return img[:, ::-1, ...]
    else:
        return img[..., ::-1, :]


class CenterCrop:
    """Crop central part of an image to a specified size.

    Crops central part of an image. An image
    is an nd-array with size H x W x C with H
    and W spatial dimensions and C number of
    channels.
    """

    def __init__(self, output_size: Union[Shape, int]):
        """
        Args:
            output_size: Desired output size. If int, square crop is made.
        """
        # assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size: Shape = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image: Array) -> Array:
        """Apply center crop.

        Args:
            image: The nd-array to be cropped.

        Returns:
            The cropped image.
        """

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[top : top + new_h, left : left + new_w]

        return image


class PositionalCrop:
    """Crop an image from a given corner to a specified size.

    Crops an image from a given corner. An image
    is an nd-array with size H x W x C with H
    and W spatial dimensions and C number of
    channels.
    """

    def __init__(self, output_size: Union[Shape, int]):
        """
        Args:
            output_size: Desired output size. If int, square crop is made.
        """
        # assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size: Shape = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image: Array, top: int, left: int) -> Array:
        """Apply positional crop.

        Args:
            image: The nd-array to be cropped.
            top: Vertical top coordinate of corner to start cropping.
            left: Horizontal left coordinate of corner to start cropping.

        Returns:
            The cropped image.
        """

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        image = image[top : top + new_h, left : left + new_w]

        return image


class RandomNoise:
    """Adds Gaussian noise to an image or a
    batch of images.

    Adds Gaussian noise to an image or a batch
    of images. An image is an nd-array with size
    H x W x C with H and W spatial dimensions
    and C number of channels. A batch of images
    is an nd-array with size N x H x W x C with
    N number of images. The Gaussian noise is
    a Gaussian random variable with mean zero and
    given standard deviation. The standard
    deviation can be a fix value corresponding
    to the specified noise level or randomly
    selected on a range between 50% and 100% of
    the specified noise level.
    """

    def __init__(self, noise_level: float, range_flag: bool = False):
        """
        Args:
            noise_level: Standard dev of the Gaussian noise.
            range_flag: If true, the standard dev is randomly selected
                between 50% and 100% of `noise_level` set. Default: ``False``.
        """
        self.range_flag = range_flag
        if range_flag:
            self.noise_level_low = 0.5 * noise_level
        self.noise_level = noise_level

    def __call__(self, image: Array) -> Array:
        """Add Gaussian noise.

        Args:
            image: The nd-array to add noise to.

        Returns:
            The noisy image.
        """

        noise_level = self.noise_level

        if self.range_flag:
            if image.ndim > 3:
                num_img = image.shape[0]
            else:
                num_img = 1
            noise_level_range = np.random.uniform(self.noise_level_low, self.noise_level, num_img)
            noise_level = noise_level_range.reshape(
                (noise_level_range.shape[0],) + (1,) * (image.ndim - 1)
            )

        imgnoised = image + np.random.normal(0.0, noise_level, image.shape)
        imgnoised = np.clip(imgnoised, 0.0, 1.0)

        return imgnoised


def reconfigure_images(
    images: Array,
    output_size: Union[Shape, int],
    gray_flag: bool = False,
    num_img: Optional[int] = None,
    multi_flag: bool = False,
    stride: Optional[Union[Shape, int]] = None,
    dtype: Any = np.float32,
) -> Array:
    """Reconfigure set of images, converting to
    gray scale, or cropping or sampling multiple
    patches from each one, or selecting a subset
    of them, according to specified setup.

    Args:
        images: Array of color images.
        output_size: Desired output size. If int, square crop is made.
        gray_flag: If true, converts to gray scale.
        num_img: If specified, reads that number of images, if not reads all the images in path.
        multi_flag: If true, samples multiple patches of specified size in each image.
        stride: Stride between patch origins (indexed from left-top
            corner). If int, the same stride is used in h and w.
        dtype: type of array. Default: ``np.float32``.

    Returns:
        Reconfigured nd-array.
    """

    # Get number of images to use.
    if num_img is None:
        num_img = images.shape[0]

    # Get channels of ouput image.
    C = 3
    if gray_flag:
        C = 1

    # Define functionality to crop and create signal array.
    if multi_flag:
        tsfm = PositionalCrop(output_size)
        assert stride is not None
        if isinstance(stride, int):
            stride_multi = (stride, stride)
        S = np.zeros((num_img, images.shape[1], images.shape[2], C), dtype=dtype)
    else:
        tsfm_crop = CenterCrop(output_size)
        S = np.zeros((num_img, tsfm_crop.output_size[0], tsfm_crop.output_size[1], C), dtype=dtype)

    # Convert to gray scale and/or crop.
    for i in range(S.shape[0]):
        img = images[i] / 255.0
        if gray_flag:
            imgG = rgb2gray(img)
            # Keep channel singleton.
            img = imgG.reshape(imgG.shape + (1,))
        if not multi_flag:
            # Crop image
            img = tsfm_crop(img)
        S[i] = img

    if multi_flag:
        # Sample multiple patches from image
        h = S.shape[1]
        w = S.shape[2]
        nh = int(math.floor((h - tsfm.output_size[0]) / stride_multi[0])) + 1
        nw = int(math.floor((w - tsfm.output_size[1]) / stride_multi[1])) + 1
        saux = np.zeros(
            (nh * nw * num_img, tsfm.output_size[0], tsfm.output_size[1], S.shape[-1]), dtype=dtype
        )
        count2 = 0
        for i in range(S.shape[0]):
            for top in range(0, h - tsfm.output_size[0], stride_multi[0]):
                for left in range(0, w - tsfm.output_size[1], stride_multi[1]):
                    saux[count2, ...] = tsfm(S[i], top, left)
                    count2 += 1
        S = saux
    return S


class ConfigImageSetDict(TypedDict):
    """Definition of the dictionary structure
    expected for building and image data set
    for training."""

    output_size: Union[int, Shape]
    stride: Optional[Union[Shape, int]]
    multi: bool
    augment: bool
    run_gray: bool
    num_img: int
    test_num_img: int
    data_mode: str
    noise_level: float
    noise_range: bool
    test_split: float
    seed: float


def build_image_dataset(
    imgs_train, imgs_test, config: ConfigImageSetDict, transf: Optional[Callable] = None
) -> Tuple[DataSetDict, ...]:
    """Pre-process images according to the
    specified configuration. Keep training and
    testing partitions. Each dictionary returned
    has images and labels, which are nd-arrays
    of dimensions (N, H, W, C) with
    N: number of images; H, W: spatial dimensions
    and C: number of channels.

    Args:
        imgs_train: 4D array (NHWC) with images for training.
        imgs_test: 4D array (NHWC) with images for testing.
        config: Configuration of image data set to read.
        transf: Operator for blurring or other non-trivial transformations. Default: ``None``.

    Returns:
       tuple: A tuple (train_ds, test_ds) containing:

           - **train_ds** : (DataSetDict): Dictionary of training data (includes images and labels).
           - **test_ds** : (DataSetDict): Dictionary of testing data (includes images and labels).
    """
    # Reconfigure images by converting to gray scale or sampling multiple
    # patches according to specified configuration.
    S_train = reconfigure_images(
        imgs_train,
        config["output_size"],
        gray_flag=config["run_gray"],
        num_img=config["num_img"],
        multi_flag=config["multi"],
        stride=config["stride"],
    )
    S_test = reconfigure_images(
        imgs_test,
        config["output_size"],
        gray_flag=config["run_gray"],
        num_img=config["test_num_img"],
        multi_flag=config["multi"],
        stride=config["stride"],
    )

    # Check for transformation
    tsfm: Optional[Callable] = None
    # Processing: add noise or blur or etc.
    if config["data_mode"] == "dn":  # Denoise problem
        tsfm = RandomNoise(config["noise_level"], config["noise_range"])
    elif config["data_mode"] == "dcnv":  # Deconvolution problem
        assert transf is not None
        tsfm = transf

    if config["augment"]:  # Augment training data set by flip and 90 degrees rotation

        strain1 = rotation90(S_train.copy())
        strain2 = flip(S_train.copy())

        S_train = np.concatenate((S_train, strain1, strain2), axis=0)

    # Processing: apply transformation
    if tsfm is not None:
        if config["data_mode"] == "dn":
            Stsfm_train = tsfm(S_train.copy())
            Stsfm_test = tsfm(S_test.copy())
        elif config["data_mode"] == "dcnv":
            tsfm2 = RandomNoise(config["noise_level"], config["noise_range"])
            Stsfm_train = tsfm2(tsfm(S_train.copy()))
            Stsfm_test = tsfm2(tsfm(S_test.copy()))

    # Shuffle data
    rng = np.random.default_rng(config["seed"])
    perm_tr = rng.permutation(Stsfm_train.shape[0])
    perm_tt = rng.permutation(Stsfm_test.shape[0])
    train_ds: DataSetDict = {"image": Stsfm_train[perm_tr], "label": S_train[perm_tr]}
    test_ds: DataSetDict = {"image": Stsfm_test[perm_tt], "label": S_test[perm_tt]}

    return train_ds, test_ds


def images_read(path: str, ext: str = "jpg") -> Array:  # pragma: no cover
    """Read a collection of color images from a set of files in the specified directory.

    All files with extension `ext` (i.e. matching glob `*.ext`)
    in directory `path` are assumed to be image files and are read.
    Images may have different aspect ratios, therefore, they are
    transposed to keep the aspect ratio of the first image read.

    Args:
        path: Path to directory containing the image files.
        ext: Filename extension.

    Returns:
        Collection of color images as a 4D array.
    """

    slices = []
    shape = None
    for file in sorted(glob.glob(os.path.join(path, "*." + ext))):
        image = imageio.imread(file)
        if shape is None:
            shape = image.shape[:2]
        if shape != image.shape[:2]:
            image = np.transpose(image, (1, 0, 2))
        slices.append(image)
    return np.stack(slices)


def get_bsds_data(path: str, verbose: bool = False):  # pragma: no cover
    """Download BSDS500 data from the Berkeley Segmentation Dataset and Benchmark project.

    Download the BSDS500 dataset, a set of 500 color images of size
    481x321 or 321x481, from the Berkeley Segmentation Dataset and
    Benchmark project.

    The downloaded data is converted to `.npz` format for
    convenient access via :func:`numpy.load`. The converted data
    is saved in a file `bsds500.npz` in the directory specified by
    `path`. Note that train and test folders are merged to get a
    set of 400 images for training while the val folder is reserved
    as a set of 100 images for testing. This is done in multiple
    works such as :cite:`zhang-2017-dncnn`.

    Args:
        path: Directory in which converted data is saved.
        verbose: Flag indicating whether to print status messages.
    """
    # data source URL and filenames
    data_base_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/"
    data_tar_file = "BSR_bsds500.tgz"
    # ensure path directory exists
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} does not exist or is not a directory")
    # create temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    if verbose:
        print(f"Downloading {data_tar_file} from {data_base_url}")
    data = util.url_get(data_base_url + data_tar_file)
    f = open(os.path.join(temp_dir.name, data_tar_file), "wb")
    f.write(data.read())
    f.close()
    if verbose:
        print("Download complete")

    # untar downloaded data into temporary directory
    if verbose:
        print(f"Extracting content from tar file {data_tar_file}")

    with tarfile.open(os.path.join(temp_dir.name, data_tar_file), "r") as tar_ref:
        tar_ref.extractall(temp_dir.name)

    # read untared data files into 4D arrays and save as .npz
    data_path = os.path.join("BSR", "BSDS500", "data", "images")
    train_path = os.path.join(data_path, "train")
    imgs_train = images_read(os.path.join(temp_dir.name, train_path))
    val_path = os.path.join(data_path, "val")
    imgs_val = images_read(os.path.join(temp_dir.name, val_path))
    test_path = os.path.join(data_path, "test")
    imgs_test = images_read(os.path.join(temp_dir.name, test_path))

    # Train and test data merge into train.
    # Leave val data for testing.
    imgs400 = np.vstack([imgs_train, imgs_test])
    if verbose:
        print(f"Read {imgs400.shape[0]} images for training")
        print(f"Read {imgs_val.shape[0]} images for testing")

    npz_file = os.path.join(path, "bsds500.npz")
    if verbose:
        print(f"Saving as {npz_file}")
    np.savez(npz_file, imgstr=imgs400, imgstt=imgs_val)


def check_img_data_requirements(
    train_nimg: int,
    test_nimg: int,
    size: int,
    gray_flag: bool,
    train_in_shp: Shape,
    test_in_shp: Shape,
    train_nimg_avail: int,
    test_nimg_avail: int,
    verbose: bool,
) -> bool:  # pragma: no cover
    """Check data loaded vs. data requirements.

    Args:
        train_nimg: Number of images required for training data.
        test_nimg: Number of images required for testing data.
        size: Size of images requested.
        gray_flag: Flag to indicate if gray scale images or color images are requested.
            When ``True`` gray scale images are used, therefore, one channel is expected.
        train_in_shp: Shape of images/patches loaded as training data.
        test_in_shp: Shape of images/patches loaded as testing data.
        train_nimg_avail: Number of images available in  loaded training image data.
        test_nimg_avail: Number of images available in loaded testing image data.
        verbose: Flag indicating whether to print status messages.

    Returns:
       True if the loaded image data satifies requirements of size, number of samples
       and number of channels and False otherwise.
    """

    # Check image size.
    if (train_in_shp[1] != size) or (test_in_shp[1] != size):
        raise RuntimeError(
            f"{'Provided size: '}{size}{' does not match read train size: '}"
            f"{train_in_shp[1]}{' or test size: '}{test_in_shp[1]}"
        )
    # Check gray scale or color images.
    C_train = train_in_shp[-1]
    C_test = test_in_shp[-1]
    if gray_flag:
        C = 1
    else:
        C = 3
    if (C_train != C) or (C_test != C):
        raise RuntimeError(
            f"{'Provided channels: '}{C}{' do not match read train channels: '}"
            f"{C_train}{' or test channels: '}{C_test}"
        )
    # Check that enough images were sampled.
    if train_nimg_avail >= train_nimg:
        if test_nimg_avail >= test_nimg:
            return True

        elif verbose:
            print(
                f"{'Not enough images sampled in testing file':34s}{'Requested:':12s}"
                f"{test_nimg}{'Sampled:':12s}{test_nimg_avail}"
            )
    elif verbose:
        print(
            f"{'Not enough images sampled in training file':34s}{'Requested:':12s}{train_nimg}"
            f"{' Available:':12s}{train_nimg_avail}"
        )
    return False


def load_image_data(
    train_nimg: int,
    test_nimg: int,
    size: int,
    gray_flag: bool,
    data_mode: str = "dn",
    cache_path: Optional[str] = None,
    verbose: bool = False,
    noise_level: float = 0.1,
    noise_range: bool = False,
    transf: Optional[Callable] = None,
    stride: Optional[int] = None,
    augment: bool = False,
) -> Tuple[DataSetDict, ...]:  # pragma: no cover
    """
    Load and/or pre-process image data.

    Load and/or pre-process image data for
    training of neural network models. The original
    source is the BSDS500 data from the Berkeley
    Segmentation Dataset and Benchmark project.
    Depending on the intended applications, different
    pre-processings can be performed to the source data.

    If a cached file exists, and enough images
    were sampled, data is loaded and returned.

    If either `size` or type of data (gray
    scale or color) requested does not match
    the data read from the cached file, a
    `RunTimeError` is generated. In contrast,
    there is no checking for the specific
    contamination (i.e. noise level, blur
    kernel, etc.).

    If no cached file is found or not enough
    images were sampled and stored in the
    file, a new data set is generated and stored
    in `cache_path`. The data is stored in
    `.npz` format for convenient access via
    :func:`numpy.load`. The data is saved in two
    distinct files: `*_bsds_train.npz` and
    `*_bsds_test.npz` to keep separated training
    and testing partitions. The * stands for
    `dn` if denoising problem or `dcnv` if
    deconvolution problem. Other types of pre-processings
    may be specified via the `transf` operator.

    Args:
        train_nimg: Number of images required for sampling training data.
        test_nimg: Number of images required for sampling testing data.
        size: Size of reconstruction images.
        gray_flag: Flag to indicate if gray scale images or color images.
            When ``True`` gray scale images are used.
        data_mode: Type of image problem. Options are: `dn` for denosing, `dcnv` for deconvolution.
        cache_path: Directory in which processed data is saved. Default: ``None``.
        verbose: Flag indicating whether to print status messages. Default: ``False``.
        noise_level: Standard deviation of the Gaussian noise.
        noise_range: Flag to indicate if a fixed or a random standard deviation must be used.
            Default: ``False`` i.e. fixed standard deviation given by `noise_level`.
        transf: Operator for blurring or other non-trivial transformations.
            Should be able to handle batched (NHWC) data. Default: ``None``.
        stride: Stride between patch origins (indexed from left-top corner).
            Default: 0 (i.e. no stride, only one patch per image).
        augment: Augment training data set by flip and 90 degrees rotation.
            Default: ``False`` (i.e. no augmentation).

    Returns:
       tuple: A tuple (train_ds, test_ds) containing:

           - **train_ds** : (DataSetDict): Dictionary of training data (includes images and labels).
           - **test_ds** : (DataSetDict): Dictionary of testing data (includes images and labels).
    """
    # Set default cache path if not specified.
    if cache_path is None:
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "data")
        subpath = str.split(cache_path, ".cache")
        cache_path_display = "~/.cache" + subpath[-1]
    else:
        cache_path_display = cache_path

    # Create cache directory and generate data if not already present.
    npz_train_file = os.path.join(cache_path, data_mode + "_bsds_train.npz")
    npz_test_file = os.path.join(cache_path, data_mode + "_bsds_test.npz")

    if os.path.isfile(npz_train_file) and os.path.isfile(npz_test_file):
        # Load data and convert arrays to float32.
        trdt = np.load(npz_train_file)  # Training
        ttdt = np.load(npz_test_file)  # Testing
        train_in = trdt["image"].astype(np.float32)
        train_out = trdt["label"].astype(np.float32)
        test_in = ttdt["image"].astype(np.float32)
        test_out = ttdt["label"].astype(np.float32)

        if check_img_data_requirements(
            train_nimg,
            test_nimg,
            size,
            gray_flag,
            train_in.shape,
            test_in.shape,
            trdt["numimg"],
            ttdt["numimg"],
            verbose,
        ):

            train_ds: DataSetDict = {
                "image": train_in,
                "label": train_out,
            }
            test_ds: DataSetDict = {
                "image": test_in,
                "label": test_out,
            }
            if verbose:
                print(f"{'Data read from path':28s}{':':4s}{cache_path_display}")
                print(f"{'Train images':28s}{':':4s}{train_ds['image'].shape[0]}")
                print(f"{'Test images':28s}{':':4s}{test_ds['image'].shape[0]}")
                print(
                    f"{'Data range images':25s}{'Min:':6s}{train_ds['image'].min():>5.2f}"
                    f"{', Max:':6s}{train_ds['image'].max():>5.2f}"
                )
                print(
                    f"{'Data range labels':25s}{'Min:':6s}{train_ds['label'].min():>5.2f}"
                    f"{', Max:':6s}{train_ds['label'].max():>5.2f}"
                )
                print(
                    "NOTE: If blur kernel or noise parameter are changed, the cache"
                    " must be manually deleted to ensure that the training data "
                    " is regenerated with these new parameters."
                )

            return train_ds, test_ds

    # Check if BSDS folder exists if not create and download BSDS data.
    bsds_cache_path = os.path.join(cache_path, "BSDS")
    if not os.path.isdir(bsds_cache_path):
        os.makedirs(bsds_cache_path)
        get_bsds_data(path=bsds_cache_path, verbose=verbose)
    # Load data, convert arrays to float32 and return after pre-processing for specified data_mode.
    npz_file = os.path.join(bsds_cache_path, "bsds500.npz")
    npz = np.load(npz_file)
    imgs_train = npz["imgstr"].astype(np.float32)
    imgs_test = npz["imgstt"].astype(np.float32)

    # Generate new data.
    if stride is None:
        multi = False
    else:
        multi = True

    config: ConfigImageSetDict = {
        "output_size": size,
        "stride": stride,
        "multi": multi,
        "augment": augment,
        "run_gray": gray_flag,
        "num_img": train_nimg,
        "test_num_img": test_nimg,
        "data_mode": data_mode,
        "noise_level": noise_level,
        "noise_range": noise_range,
        "test_split": 0.2,
        "seed": 1234,
    }
    train_ds, test_ds = build_image_dataset(imgs_train, imgs_test, config, transf)
    # Store generated images.
    os.makedirs(cache_path, exist_ok=True)
    np.savez(
        npz_train_file,
        image=train_ds["image"],
        label=train_ds["label"],
        numimg=train_nimg,
    )
    np.savez(
        npz_test_file,
        image=test_ds["image"],
        label=test_ds["label"],
        numimg=test_nimg,
    )

    if verbose:
        print(f"{'Storing data in path':28s}{':':4s}{cache_path_display}")
        print(f"{'Train images':28s}{':':4s}{train_ds['image'].shape[0]}")
        print(f"{'Test images':28s}{':':4s}{test_ds['image'].shape[0]}")
        print(
            f"{'Data range images':25s}{'Min:':6s}{train_ds['image'].min():>5.2f}"
            f"{', Max:':6s}{train_ds['image'].max():>5.2f}"
        )
        print(
            f"{'Data range labels':25s}{'Min:':6s}{train_ds['label'].min():>5.2f}"
            f"{', Max:':6s}{train_ds['label'].max():>5.2f}"
        )

    return train_ds, test_ds


def build_blur_kernel(
    kernel_size: Shape,
    blur_sigma: float,
    dtype: Any = np.float32,
):
    """Construct a blur kernel as specified.

    Args:
        kernel_size: Size of the blur kernel.
        blur_sigma: Standard deviation of the blur kernel.
        dtype: Output data type. Default: ``np.float32``.
    """
    kernel = 1.0
    meshgrids = np.meshgrid(*[np.arange(size, dtype=dtype) for size in kernel_size])
    for size, mgrid in zip(kernel_size, meshgrids):
        mean = (size - 1) / 2
        kernel *= np.exp(-(((mgrid - mean) / blur_sigma) ** 2) / 2)
    # Make sure norm of values in gaussian kernel equals 1.
    knorm = np.sqrt(np.sum(kernel * kernel))
    kernel = kernel / knorm

    return kernel


class PaddedCircularConvolve(LinearOperator):
    """Define padded convolutional operator.

    The operator pads the signal with a reflection of the borders before convolving with the kernel
    provided at initialization. It crops the result of the convolution to maintain the same
    signal size.
    """

    def __init__(
        self,
        output_size: Union[Shape, int],
        channels: int,
        kernel_size: Union[Shape, int],
        blur_sigma: float,
        dtype: Any = np.float32,
    ):
        """
        Args:
            output_size: Size of the image to blur.
            channels: Number of channels in image to blur.
            kernel_size: Size of the blur kernel.
            blur_sigma: Standard deviation of the blur kernel.
            dtype: Output data type. Default: ``np.float32``.
        """
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2

        # Define padding.
        self.padsz = (
            (kernel_size[0] // 2, kernel_size[0] // 2),
            (kernel_size[1] // 2, kernel_size[1] // 2),
            (0, 0),
        )

        shape = (output_size[0], output_size[1], channels)
        with_pad = (
            output_size[0] + self.padsz[0][0] + self.padsz[0][1],
            output_size[1] + self.padsz[1][0] + self.padsz[1][1],
        )
        shape_padded = (with_pad[0], with_pad[1], channels)

        # Define data types.
        input_dtype = dtype
        output_dtype = dtype

        # Construct blur kernel as specified.
        kernel = build_blur_kernel(kernel_size, blur_sigma)

        # Define convolution part.
        self.conv = CircularConvolve(kernel, input_shape=shape_padded, ndims=2, input_dtype=dtype)

        # Initialize Linear Operator.
        super().__init__(
            input_shape=shape,
            output_shape=shape,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            jit=True,
        )

    def _eval(self, x: Array) -> Array:
        """Apply operator.

        Args:
            x: The nd-array with input signal. The input to the constructed operator should
                be HWC with H and W spatial dimensions given by `output_size` and C the given
                `channels`.

        Returns:
            The result of padding, convolving and cropping the signal. The output signal has
                the same HWC dimensions as the input signal.
        """
        xpadd: Array = jnp.pad(x, self.padsz, mode="reflect")
        rconv: Array = self.conv(xpadd)
        return rconv[self.padsz[0][0] : -self.padsz[0][1], self.padsz[1][0] : -self.padsz[1][1], :]
