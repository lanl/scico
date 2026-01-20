# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utility functions used by example scripts."""

import glob
import os
import tempfile
import zipfile
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp

import imageio.v3 as iio

import scico.numpy as snp
from scico import random, util
from scico.typing import Shape
from scipy.io import loadmat
from scipy.ndimage import zoom


def rgb2gray(rgb: snp.Array) -> snp.Array:
    """Convert an RGB image (or images) to grayscale.

    Args:
        rgb: RGB image as Nr x Nc x 3 or Nr x Nc x 3 x K array.

    Returns:
        Grayscale image as Nr x Nc or Nr x Nc x K array.
    """

    shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]]
    if rgb.ndim == 3:
        shape = (1, 1, 3)
    else:
        shape = (1, 1, 3, 1)
    w = snp.array([0.299, 0.587, 0.114], dtype=rgb.dtype).reshape(shape)
    return snp.sum(w * rgb, axis=2)


def volume_read(path: str, ext: str = "tif") -> np.ndarray:
    """Read a 3D volume from a set of files in the specified directory.

    All files with extension `ext` (i.e. matching glob `*.ext`)
    in directory `path` are assumed to be image files and are read.
    The filenames are assumed to be such that their alphanumeric
    ordering corresponds to their order as volume slices.

    Args:
        path: Path to directory containing the image files.
        ext: Filename extension.

    Returns:
        Volume as a 3D array.
    """

    slices = []
    for file in sorted(glob.glob(os.path.join(path, "*." + ext))):
        image = iio.imread(file)
        slices.append(image)
    return np.dstack(slices)


def get_epfl_deconv_data(channel: int, path: str, verbose: bool = False):  # pragma: no cover
    """Download example data from EPFL Biomedical Imaging Group.

    Download deconvolution problem data from EPFL Biomedical Imaging
    Group. The downloaded data is converted to `.npz` format for
    convenient access via :func:`numpy.load`. The converted data is saved
    in a file `epfl_big_deconv_<channel>.npz` in the directory specified
    by `path`.

    Args:
        channel: Channel number between 0 and 2.
        path: Directory in which converted data is saved.
        verbose: Flag indicating whether to print status messages.
    """

    # data source URL and filenames
    data_base_url = "http://bigwww.epfl.ch/deconvolution/bio/"
    data_zip_files = ["CElegans-CY3.zip", "CElegans-DAPI.zip", "CElegans-FITC.zip"]
    psf_zip_files = ["PSF-" + data for data in data_zip_files]

    # ensure path directory exists
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} does not exist or is not a directory.")

    # create temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    # download data and psf files for selected channel into temporary directory
    for zip_file in (data_zip_files[channel], psf_zip_files[channel]):
        if verbose:
            print(f"Downloading {zip_file} from {data_base_url}")
        data = util.url_get(data_base_url + zip_file)
        f = open(os.path.join(temp_dir.name, zip_file), "wb")
        f.write(data.read())
        f.close()
        if verbose:
            print("Download complete")

    # unzip downloaded data into temporary directory
    for zip_file in (data_zip_files[channel], psf_zip_files[channel]):
        if verbose:
            print(f"Extracting content from zip file {zip_file}")
        with zipfile.ZipFile(os.path.join(temp_dir.name, zip_file), "r") as zip_ref:
            zip_ref.extractall(temp_dir.name)

    # read unzipped data files into 3D arrays and save as .npz
    zip_file = data_zip_files[channel]
    y = volume_read(os.path.join(temp_dir.name, zip_file[:-4]))
    zip_file = psf_zip_files[channel]
    psf = volume_read(os.path.join(temp_dir.name, zip_file[:-4]))

    npz_file = os.path.join(path, f"epfl_big_deconv_{channel}.npz")
    if verbose:
        print(f"Saving as {npz_file}")
    np.savez(npz_file, y=y, psf=psf)


def epfl_deconv_data(
    channel: int, verbose: bool = False, cache_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get deconvolution problem data from EPFL Biomedical Imaging Group.

    If the data has previously been downloaded, it will be retrieved from
    a local cache.

    Args:
        channel: Channel number between 0 and 2.
        verbose: Flag indicating whether to print status messages.
        cache_path: Directory in which downloaded data is cached. The
           default is `~/.cache/scico/examples`, where `~` represents
           the user home directory.

    Returns:
       tuple: A tuple (y, psf) containing:

           - **y** : (np.ndarray): Blurred channel data.
           - **psf** : (np.ndarray): Channel psf.
    """

    # set default cache path if not specified
    if cache_path is None:  # pragma: no cover
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples")

    # create cache directory and download data if not already present
    npz_file = os.path.join(cache_path, f"epfl_big_deconv_{channel}.npz")
    if not os.path.isfile(npz_file):  # pragma: no cover
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        get_epfl_deconv_data(channel, path=cache_path, verbose=verbose)

    # load data and return y and psf arrays converted to float32
    npz = np.load(npz_file)
    y = npz["y"].astype(np.float32)
    psf = npz["psf"].astype(np.float32)
    return y, psf


def get_ucb_diffusercam_data(path: str, verbose: bool = False):  # pragma: no cover
    """Download data from UC Berkeley Waller Lab diffusercam project.

    Download deconvolution problem data from UC Berkeley Waller Lab
    diffusercam project.  The downloaded data is converted to `.npz`
    format for convenient access via :func:`numpy.load`.  The
    converted data is saved in a file `ucb_diffcam_data.npz.npz` in
    the directory specified by `path`.

    Args:
        path: Directory in which converted data is saved.
        verbose: Flag indicating whether to print status messages.
    """

    # data source URL and filenames
    data_base_url = "https://github.com/Waller-Lab/DiffuserCam/blob/master/example_data/"
    data_files = ["example_psfs.mat", "example_raw.png"]

    # ensure path directory exists
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} does not exist or is not a directory.")

    # create temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    # download data files into temporary directory
    for data_file in data_files:
        if verbose:
            print(f"Downloading {data_file} from {data_base_url}")
        data = util.url_get(data_base_url + data_file + "?raw=true")
        f = open(os.path.join(temp_dir.name, data_file), "wb")
        f.write(data.read())
        f.close()
        if verbose:
            print("Download complete")

    # load data, normalize it, and save as npz
    y = iio.imread(os.path.join(temp_dir.name, "example_raw.png"))
    y = y.astype(np.float32)
    y -= 100.0
    y /= y.max()
    mat = loadmat(os.path.join(temp_dir.name, "example_psfs.mat"))
    psf = mat["psf"].astype(np.float64)
    psf -= 102.0
    psf /= np.linalg.norm(psf, axis=(0, 1)).min()

    # save as .npz
    npz_file = os.path.join(path, "ucb_diffcam_data.npz")
    if verbose:
        print(f"Saving as {npz_file}")
    np.savez(npz_file, y=y, psf=psf)


def ucb_diffusercam_data(
    verbose: bool = False, cache_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get example data from UC Berkeley Waller Lab diffusercam project.

    If the data has previously been downloaded, it will be retrieved from
    a local cache.

    Args:
        verbose: Flag indicating whether to print status messages.
        cache_path: Directory in which downloaded data is cached. The
           default is `~/.cache/scico/examples`, where `~` represents
           the user home directory.

    Returns:
       tuple: A tuple (y, psf) containing:

           - **y** : (np.ndarray): Measured image
           - **psf** : (np.ndarray): Stack of psfs.
    """

    # set default cache path if not specified
    if cache_path is None:  # pragma: no cover
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples")

    # create cache directory and download data if not already present
    npz_file = os.path.join(cache_path, "ucb_diffcam_data.npz")
    if not os.path.isfile(npz_file):  # pragma: no cover
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        get_ucb_diffusercam_data(path=cache_path, verbose=verbose)

    # load data and return y and psf arrays converted to float32
    npz = np.load(npz_file)
    y = npz["y"].astype(np.float32)
    psf = npz["psf"].astype(np.float64)
    return y, psf


def downsample_volume(vol: snp.Array, rate: int) -> snp.Array:
    """Downsample a 3D array.

    Downsample a 3D array. If the volume dimensions can be divided by
    `rate`, this is achieved via averaging distinct `rate` x `rate` x
    `rate` block in `vol`. Otherwise it is achieved via a call to
    :func:`scipy.ndimage.zoom`.

    Args:
        vol: Input volume.
        rate: Downsampling rate.

    Returns:
        Downsampled volume.
    """

    if rate == 1:
        return vol

    if np.all([n % rate == 0 for n in vol.shape]):
        vol = snp.mean(snp.reshape(vol, (-1, rate, vol.shape[1], vol.shape[2])), axis=1)
        vol = snp.mean(snp.reshape(vol, (vol.shape[0], -1, rate, vol.shape[2])), axis=2)
        vol = snp.mean(snp.reshape(vol, (vol.shape[0], vol.shape[1], -1, rate)), axis=3)
    else:
        vol = zoom(vol, 1.0 / rate)

    return vol


def tile_volume_slices(x: snp.Array, sep_width: int = 10) -> snp.Array:
    """Make an image with tiled slices from an input volume.

    Make an image with tiled `xy`, `xz`, and `yz` slices from an input
    volume.

    Args:
        x: Input volume consisting of a 3D or 4D array. If the input is
           4D, the final axis represents a channel index.
        sep_width: Number of pixels separating the slices in the output
           image.

    Returns:
        Image containing tiled slices.
    """

    if x.ndim == 3:
        fshape: Tuple[int, ...] = (x.shape[0], sep_width)
    else:
        fshape = (x.shape[0], sep_width, 3)
    out = snp.concatenate(
        (
            x[:, :, x.shape[2] // 2],
            snp.full(fshape, snp.nan),
            x[:, x.shape[1] // 2, :],
        ),
        axis=1,
    )

    if x.ndim == 3:
        fshape0: Tuple[int, ...] = (sep_width, out.shape[1])
        fshape1: Tuple[int, ...] = (x.shape[2], x.shape[2] + sep_width)
        trans: Tuple[int, ...] = (1, 0)

    else:
        fshape0 = (sep_width, out.shape[1], 3)
        fshape1 = (x.shape[2], x.shape[2] + sep_width, 3)
        trans = (1, 0, 2)
    out = snp.concatenate(
        (
            out,
            snp.full(fshape0, snp.nan),
            snp.concatenate(
                (
                    x[x.shape[0] // 2, :, :].transpose(trans),
                    snp.full(fshape1, snp.nan),
                ),
                axis=1,
            ),
        ),
        axis=0,
    )

    out = snp.where(snp.isnan(out), snp.nanmax(out), out)

    return out


def gaussian(shape: Shape, sigma: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Construct a multivariate Gaussian distribution function.

    Construct a zero-mean multivariate Gaussian distribution function

    .. math::
        f(\mb{x}) = (2 \pi)^{-N/2} \, \det(\Sigma)^{-1/2} \, \exp \left(
        -\frac{\mb{x}^T \, \Sigma^{-1} \, \mb{x}}{2} \right) \;,

    where :math:`\Sigma` is the covariance matrix of the distribution.

    Args:
        shape: Shape of output array.
        sigma: Covariance matrix.

    Returns:
        Sampled function.

    Raises:
        ValueError: If the array `sigma` cannot be inverted.
    """

    if sigma is None:
        sigma = np.diag(np.array(shape) / 7) ** 2
    N = len(shape)
    try:
        sigmainv = np.linalg.inv(sigma)
        sigmadet = np.linalg.det(sigma)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Invalid covariance matrix {sigma}.") from e
    grd = np.stack(np.mgrid[[slice(-(n - 1) / 2, (n + 1) / 2) for n in shape]], axis=-1)
    sigmax = np.dot(grd, sigmainv)
    xtsigmax = np.sum(grd * np.dot(grd, sigmainv), axis=-1)
    const = ((2.0 * np.pi) ** (-N / 2.0)) * (sigmadet ** (-1.0 / 2.0))
    return const * np.exp(-xtsigmax / 2.0)


def create_cone(shape: Shape, center: Optional[List[float]] = None) -> snp.Array:
    """Compute a map of distances from a center pixel.

    Args:
        shape: Shape of the array for which the distance map is to be
            computed.
        center: Tuple of center coordinates. If ``None``, it is set to
            the center of the array.

    Returns:
        An array containing a map of the distances.
    """

    if center is None:
        center = [(dim - 1) / 2 for dim in shape]

    coords = [snp.arange(0, dim) for dim in shape]
    coord_mesh = snp.meshgrid(*coords, sparse=True, indexing="ij")

    dist_map = sum([(coord_mesh[i] - center[i]) ** 2 for i in range(len(coord_mesh))])
    dist_map = snp.sqrt(dist_map)

    return dist_map


def create_circular_phantom(
    shape: Shape, radius_list: list, val_list: list, center: Optional[list] = None
) -> snp.Array:
    """Construct a circular phantom with given radii and intensities.

    This functions supports both circular (``shape`` is 2D) and spherical
    (``shape`` is 3D) phantoms.

    Args:
        shape: Shape of the phantom to be created.
        radius_list: List of radii of the rings in the phantom.
        val_list: List of intensity values of the rings in the phantom.
        center: Tuple of center coordinates. If ``None``, it is set to
           the center of the array.

    Returns:
        The computed phantom.
    """

    dist_map = create_cone(shape, center)

    img = snp.zeros(shape)
    for r, val in zip(radius_list, val_list):
        # In numpy: img[dist_map < r] = val
        img = img.at[dist_map < r].set(val)

    return img


def create_3d_foam_phantom(
    im_shape: Shape,
    N_sphere: int,
    r_mean: float = 0.1,
    r_std: float = 0.001,
    pad: float = 0.01,
    is_random: bool = False,
) -> snp.Array:
    """Construct a 3D phantom with random radii and centers.

    Args:
        im_shape: Shape of input image.
        N_sphere: Number of spheres added.
        r_mean: Mean radius of sphere (normalized to 1 along each axis).
                Default 0.1.
        r_std: Standard deviation of radius of sphere (normalized to 1
                along each axis). Default 0.001.
        pad: Padding length (normalized to 1 along each axis). Default 0.01.
        is_random: Flag used to control randomness of phantom generation.
                If ``False``, random seed is set to 1 in order to make the
                process deterministic. Default ``False``.

    Returns:
        3D phantom of shape `im_shape`.
    """
    c_lo = 0.0
    c_hi = 1.0

    if not is_random:
        np.random.seed(1)

    coord_list = [snp.linspace(0, 1, N) for N in im_shape]
    x = snp.stack(snp.meshgrid(*coord_list, indexing="ij"), axis=-1)

    centers = np.random.uniform(low=r_mean + pad, high=1 - r_mean - pad, size=(N_sphere, 3))
    radii = r_std * np.random.randn(N_sphere) + r_mean

    im = snp.zeros(im_shape) + c_lo
    for c, r in zip(centers, radii):  # type: ignore
        dist = snp.sum((x - c) ** 2, axis=-1)
        if snp.mean(im[dist < r**2] - c_lo) < 0.01 * c_hi:
            # equivalent to im[dist < r**2] = c_hi in numpy
            im = im.at[dist < r**2].set(c_hi)

    return im


def create_conv_sparse_phantom(Nx: int, Nnz: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a disc dictionary and sparse coefficient maps.

    Construct a disc dictionary and a corresponding set of sparse
    coefficient maps for testing convolutional sparse coding algorithms.

    Args:
        Nx: Size of coefficient maps (3 x Nx x Nx).
        Nnz: Number of non-zero coefficients across all coefficient maps.

    Returns:
        A tuple consisting of a stack of 2D filters and the coefficient
           map array.
    """

    # constant parameters
    M = 3
    Nh = 7
    e = 1

    # create disc filters
    h = np.zeros((M, 2 * Nh + 1, 2 * Nh + 1))
    gr, gc = np.ogrid[-Nh : Nh + 1, -Nh : Nh + 1]
    for m in range(M):
        r = 2 * m + 3
        d = np.sqrt(gr**2 + gc**2)
        v = (np.clip(d, r - e, r + e) - (r - e)) / (2 * e)
        v = 1.0 - v
        h[m] = v

    # create sparse random coefficient maps
    np.random.seed(1234)
    x = np.zeros((M, Nx, Nx))
    idx0 = np.random.randint(0, M, size=(Nnz,))
    idx1 = np.random.randint(0, Nx, size=(2, Nnz))
    val = np.random.uniform(0, 5, size=(Nnz,))
    x[idx0, idx1[0], idx1[1]] = val

    return h, x


def create_tangle_phantom(nx: int, ny: int, nz: int) -> snp.Array:
    """Construct a 3D phantom using the tangle function.

    Args:
        nx: x-size of output.
        ny: y-size of output.
        nz: z-size of output.

    Returns:
        An array with shape (nz, ny, nx).

    """
    xs = 1.0 * np.linspace(-1.0, 1.0, nx)
    ys = 1.0 * np.linspace(-1.0, 1.0, ny)
    zs = 1.0 * np.linspace(-1.0, 1.0, nz)

    # default ordering for meshgrid is `xy`, this makes inputs of length
    # M, N, P will create a mesh of N, M, P. Thus we want ys, zs and xs.
    xx, yy, zz = np.meshgrid(ys, zs, xs, copy=True)
    xx = 3.0 * xx
    yy = 3.0 * yy
    zz = 3.0 * zz
    values = (
        xx * xx * xx * xx
        - 5.0 * xx * xx
        + yy * yy * yy * yy
        - 5.0 * yy * yy
        + zz * zz * zz * zz
        - 5.0 * zz * zz
        + 11.8
    ) * 0.2 + 0.5
    return (values < 2.0).astype(float)


@partial(jax.jit, static_argnums=0)
def create_block_phantom(out_shape: Shape) -> snp.Array:
    """Construct a blocky 3D phantom.

    Args:
        out_shape: desired phantom shape.

    Returns:
        Phantom.

    """
    # make the phantom at a low resolution
    low_res = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )
    positions = jnp.stack(
        jnp.meshgrid(*[jnp.linspace(-0.5, 2.5, s) for s in out_shape], indexing="ij")
    )
    indices = jnp.round(positions).astype(int)
    return low_res[indices[0], indices[1], indices[2]]


def spnoise(
    img: Union[np.ndarray, snp.Array], nfrac: float, nmin: float = 0.0, nmax: float = 1.0
) -> Union[np.ndarray, snp.Array]:
    """Return image with salt & pepper noise imposed on it.

    Args:
        img: Input image.
        nfrac: Desired fraction of pixels corrupted by noise.
        nmin: Lower value for noise (pepper). Default 0.0.
        nmax: Upper value for noise (salt). Default 1.0.

    Returns:
        Noisy image
    """

    if isinstance(img, np.ndarray):
        spm = np.random.uniform(-1.0, 1.0, img.shape)  # type: ignore
        imgn = img.copy()
        imgn[spm < nfrac - 1.0] = nmin
        imgn[spm > 1.0 - nfrac] = nmax
    else:
        spm, key = random.uniform(shape=img.shape, minval=-1.0, maxval=1.0, seed=0)  # type: ignore
        imgn = img
        imgn = imgn.at[spm < nfrac - 1.0].set(nmin)  # type: ignore
        imgn = imgn.at[spm > 1.0 - nfrac].set(nmax)  # type: ignore
    return imgn


def phase_diff(x: snp.Array, y: snp.Array) -> snp.Array:
    """Distance between phase angles.

    Compute the distance between two arrays of phase angles, with
    appropriate phase wrapping to minimize the distance.

    Args:
        x: Input array.
        y: Input array.

    Returns:
        Array of angular distances.
    """

    mod = snp.mod(snp.abs(x - y), 2 * snp.pi)
    return snp.minimum(mod, 2 * snp.pi - mod)
