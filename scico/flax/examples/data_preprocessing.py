# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Image manipulation utils."""

import glob
import math
import os
import tarfile
import tempfile
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

import jax.numpy as jnp

import imageio.v3 as iio

from scico import util
from scico.examples import rgb2gray
from scico.flax.train.typed_dict import DataSetDict
from scico.linop import CircularConvolve, LinearOperator
from scico.numpy import Array
from scico.typing import Shape

from .typed_dict import ConfigImageSetDict


def rotation90(img: Array) -> Array:
    """Rotate an image, or a batch of images, by 90 degrees.

    Rotate an image or a batch of images by 90 degrees counterclockwise.
    An image is an array with size H x W x C with H and W spatial
    dimensions and C number of channels. A batch of images is an
    array with size N x H x W x C with N number of images.

    Args:
        img: The array to be rotated.

    Returns:
       An image, or batch of images, rotated by 90 degrees
       counterclockwise.
    """
    if img.ndim < 4:
        return np.swapaxes(img, 0, 1)
    else:
        return np.swapaxes(img, 1, 2)


def flip(img: Array) -> Array:
    """Horizontal flip of an image or a batch of images.

    Horizontally flip an image or a batch of images. An image is an
    array with size H x W x C with H and W spatial dimensions and C
    number of channels. A batch of images is an array with size
    N x H x W x C with N number of images.

    Args:
        img: The array to be flipped.

    Returns:
       An image, or batch of images, flipped horizontally.
    """
    if img.ndim < 4:
        return img[:, ::-1, ...]
    else:
        return img[..., ::-1, :]


class CenterCrop:
    """Crop central part of an image to a specified size.

    Crop central part of an image. An image is an array with size
    H x W x C with H and W spatial dimensions and C number of channels.
    """

    def __init__(self, output_size: Union[Shape, int]):
        """
        Args:
            output_size: Desired output size. If int, square crop is
                made.
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
            image: The array to be cropped.

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

    Crop an image from a given corner. An image is an array with size
    H x W x C with H and W spatial dimensions and C number of channels.
    """

    def __init__(self, output_size: Union[Shape, int]):
        """
        Args:
            output_size: Desired output size. If int, square crop is
                made.
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
            image: The array to be cropped.
            top: Vertical top coordinate of corner to start cropping.
            left: Horizontal left coordinate of corner to start
                cropping.

        Returns:
            The cropped image.
        """

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        image = image[top : top + new_h, left : left + new_w]

        return image


class RandomNoise:
    """Add Gaussian noise to an image or a batch of images.

    Add Gaussian noise to an image or a batch of images. An image is
    an array with size H x W x C with H and W spatial dimensions
    and C number of channels. A batch of images is an array with
    size N x H x W x C with N number of images. The Gaussian noise is
    a Gaussian random variable with mean zero and given standard
    deviation. The standard deviation can be a fix value corresponding
    to the specified noise level or randomly selected on a range
    between 50% and 100% of the specified noise level.
    """

    def __init__(self, noise_level: float, range_flag: bool = False):
        """
        Args:
            noise_level: Standard dev of the Gaussian noise.
            range_flag: If ``True``, the standard dev is randomly
                selected between 50% and 100% of `noise_level` set.
                Default: ``False``.
        """
        self.range_flag = range_flag
        if range_flag:
            self.noise_level_low = 0.5 * noise_level
        self.noise_level = noise_level

    def __call__(self, image: Array) -> Array:
        """Add Gaussian noise.

        Args:
            image: The array to add noise to.

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


def preprocess_images(
    images: Array,
    output_size: Union[Shape, int],
    gray_flag: bool = False,
    num_img: Optional[int] = None,
    multi_flag: bool = False,
    stride: Optional[Union[Shape, int]] = None,
    dtype: Any = np.float32,
) -> Array:
    """Preprocess (scale, crop, etc.) set of images.

    Preprocess set of images, converting to gray scale, or cropping or
    sampling multiple patches from each one, or selecting a subset of
    them, according to specified setup.

    Args:
        images: Array of color images.
        output_size: Desired output size. If int, square crop is made.
        gray_flag: If ``True``, converts to gray scale.
        num_img: If specified, reads that number of images, if not reads
            all the images in path.
        multi_flag: If ``True``, samples multiple patches of specified
            size in each image.
        stride: Stride between patch origins (indexed from left-top
            corner). If int, the same stride is used in h and w.
        dtype: dtype of array. Default: :attr:`~numpy.float32`.

    Returns:
        Preprocessed array.
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


def build_image_dataset(
    imgs_train, imgs_test, config: ConfigImageSetDict, transf: Optional[Callable] = None
) -> Tuple[DataSetDict, ...]:
    """Preprocess and assemble dataset for training.

    Preprocess images according to the specified configuration and
    assemble a dataset into a structure that can be used for training
    machine learning models. Keep training and testing partitions.
    Each dictionary returned has images and labels, which are arrays
    of dimensions (N, H, W, C) with N: number of images; H,
    W: spatial dimensions and C: number of channels.

    Args:
        imgs_train: 4D array (NHWC) with images for training.
        imgs_test: 4D array (NHWC) with images for testing.
        config: Configuration of image data set to read.
        transf: Operator for blurring or other non-trivial
            transformations. Default: ``None``.

    Returns:
       tuple: A tuple (train_ds, test_ds) containing:

           - **train_ds** : Dictionary of training data (includes images and labels).
           - **test_ds** : Dictionary of testing data (includes images and labels).
    """
    # Preprocess images by converting to gray scale or sampling multiple
    # patches according to specified configuration.
    S_train = preprocess_images(
        imgs_train,
        config["output_size"],
        gray_flag=config["run_gray"],
        num_img=config["num_img"],
        multi_flag=config["multi"],
        stride=config["stride"],
    )
    S_test = preprocess_images(
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
    """Read a collection of color images from a set of files.

    Read a collection of color images from a set of files in the
    specified directory. All files with extension `ext` (i.e.
    matching glob `*.ext`) in directory `path` are assumed to be image
    files and are read. Images may have different aspect ratios,
    therefore, they are transposed to keep the aspect ratio of the first
    image read.

    Args:
        path: Path to directory containing the image files.
        ext: Filename extension.

    Returns:
        Collection of color images as a 4D array.
    """

    slices = []
    shape = None
    for file in sorted(glob.glob(os.path.join(path, "*." + ext))):
        image = iio.imread(file)
        if shape is None:
            shape = image.shape[:2]
        if shape != image.shape[:2]:
            image = np.transpose(image, (1, 0, 2))
        slices.append(image)
    return np.stack(slices)


def get_bsds_data(path: str, verbose: bool = False):  # pragma: no cover
    """Download BSDS500 data from the BSDB project.

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
    data_base_url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/"
    data_tar_file = "BSR_bsds500.tgz"
    # ensure path directory exists
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} does not exist or is not a directory.")
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
        subpath = str.split(npz_file, ".cache")
        npz_file_display = "~/.cache" + subpath[-1]
        print(f"Saving as {npz_file_display}")
    np.savez(npz_file, imgstr=imgs400, imgstt=imgs_val)


def build_blur_kernel(
    kernel_size: Shape,
    blur_sigma: float,
    dtype: Any = np.float32,
):
    """Construct a blur kernel as specified.

    Args:
        kernel_size: Size of the blur kernel.
        blur_sigma: Standard deviation of the blur kernel.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
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

    The operator pads the signal with a reflection of the borders
    before convolving with the kernel provided at initialization. It
    crops the result of the convolution to maintain the same signal
    size.
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
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
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
            x: The array with input signal. The input to the
                constructed operator should be HWC with H and W spatial
                dimensions given by `output_size` and C the given
                `channels`.

        Returns:
            The result of padding, convolving and cropping the signal.
            The output signal has the same HWC dimensions as the input
            signal.
        """
        xpadd: Array = jnp.pad(x, self.padsz, mode="reflect")
        rconv: Array = self.conv(xpadd)
        return rconv[self.padsz[0][0] : -self.padsz[0][1], self.padsz[1][0] : -self.padsz[1][1], :]
