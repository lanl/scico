# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Generation and loading of data used in Flax example scripts."""

import os
from typing import Callable, Optional, Tuple, Union

import numpy as np

from scico.flax.train.typed_dict import DataSetDict
from scico.numpy import Array
from scico.typing import Shape

from .data_generation import generate_blur_data, generate_ct_data
from .data_preprocessing import ConfigImageSetDict, build_image_dataset, get_bsds_data
from .typed_dict import CTDataSetDict


def get_cache_path(cache_path: Optional[str] = None) -> Tuple[str, str]:
    """Get input/output SCICO cache path.

    Args:
        cache_path: Given cache path. If ``None`` SCICO default cache
            path is constructed.

    Returns:
        The cache path and a display string with private user path
        information stripped.
    """
    if cache_path is None:
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "data")
        subpath = str.split(cache_path, ".cache")
        cache_path_display = "~/.cache" + subpath[-1]
    else:
        cache_path_display = cache_path

    return cache_path, cache_path_display


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

    Load or generate CT data for training of machine learning network
    models. If cached file exists and enough data of the requested
    size is available, data is loaded and returned.

    If either `size` or `nproj` requested does not match the data read
    from the cached file, a `RunTimeError` is generated.

    If no cached file is found or not enough data is contained in the
    file a new data set is generated and stored in `cache_path`. The
    data is stored in `.npz` format for convenient access via
    :func:`numpy.load`. The data is saved in two distinct files:
    `ct_foam2_train.npz` and `ct_foam2_test.npz` to keep separated
    training and testing partitions.

    Args:
        train_nimg: Number of images required for training.
        test_nimg: Number of images required for testing.
        size: Size of reconstruction images.
        nproj: Number of CT views.
        cache_path: Directory in which generated data is saved.
            Default: ``None``.
        verbose: Flag indicating whether to print status messages.
            Default: ``False``.

    Returns:
       tuple: A tuple (trdt, ttdt) containing:

           - **trdt** : (Dictionary): Collection of images (key `img`),
               sinograms (key `sino`) and filtered back projections
               (key `fbp`) for training.
           - **ttdt** : (Dictionary): Collection of images (key `img`),
               sinograms (key `sino`) and filtered back projections
               (key `fbp`) for testing.
    """
    # Set default cache path if not specified
    cache_path, cache_path_display = get_cache_path(cache_path)

    # Create cache directory and generate data if not already present.
    npz_train_file = os.path.join(cache_path, "ct_foam2_train.npz")
    npz_test_file = os.path.join(cache_path, "ct_foam2_test.npz")

    if os.path.isfile(npz_train_file) and os.path.isfile(npz_test_file):
        # Load data
        trdt_in = np.load(npz_train_file)
        ttdt_in = np.load(npz_test_file)
        # Check image size
        if trdt_in["img"].shape[1] != size:
            runtime_error_scalar("size", "training", size, trdt_in["img"].shape[1])
        if ttdt_in["img"].shape[1] != size:
            runtime_error_scalar("size", "testing", size, ttdt_in["img"].shape[1])
        # Check number of projections
        if trdt_in["sino"].shape[1] != nproj:
            runtime_error_scalar("views", "training", nproj, trdt_in["sino"].shape[1])
        if ttdt_in["sino"].shape[1] != nproj:
            runtime_error_scalar("views", "testing", nproj, ttdt_in["sino"].shape[1])
        # Check that enough data is available
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
                    print_input_path(cache_path_display)
                    print_data_size("training", trdt["img"].shape[0])
                    print_data_size("testing ", ttdt["img"].shape[0])
                    print_data_range("images  ", trdt["img"])
                    print_data_range("sinogram", trdt["sino"])
                    print_data_range("FBP     ", trdt["fbp"])

                return trdt, ttdt

            elif verbose:
                print_data_warning("testing", test_nimg, ttdt_in["img"].shape[0])
        elif verbose:
            print_data_warning("training", train_nimg, trdt_in["img"].shape[0])

    # Generate new data.
    nimg = train_nimg + test_nimg
    img, sino, fbp = generate_ct_data(
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
        print_output_path(cache_path_display)
        print_data_size("training", train_nimg)
        print_data_size("testing ", test_nimg)
        print_data_range("images  ", img)
        print_data_range("sinogram", sino)
        print_data_range("FBP     ", fbp)

    return trdt, ttdt


def load_blur_data(
    train_nimg: int,
    test_nimg: int,
    size: int,
    blur_kernel: Array,
    noise_sigma: float,
    cache_path: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[DataSetDict, ...]:  # pragma: no cover
    """Load or generate blurred data based on xdesign foam structures.

    Load or generate blurred data for training of machine learning
    network models. If cached file exists and enough data of the
    requested size is available, data is loaded and returned.

    If `size`, `blur_kernel` or `noise_sigma` requested do not match
    the data read from the cached file, a `RunTimeError` is generated.

    If no cached file is found or not enough data is contained in the
    file a new data set is generated and stored in `cache_path`. The
    data is stored in `.npz` format for convenient access via
    :func:`numpy.load`. The data is saved in two distinct files:
    `dcnv_foam1_train.npz` and `dcnv_foam1_test.npz` to keep separated
    training and testing partitions.

    Args:
        train_nimg: Number of images required for training.
        test_nimg: Number of images required for testing.
        size: Size of reconstruction images.
        blur_kernel: Kernel for blurring the generated images.
        noise_sigma: Level of additive Gaussian noise to apply.
        cache_path: Directory in which generated data is saved.
            Default: ``None``.
        verbose: Flag indicating whether to print status messages.
            Default: ``False``.

    Returns:
       tuple: A tuple (train_ds, test_ds) containing:

           - **train_ds** : Dictionary of training data (includes images
                            and labels).
           - **test_ds** : Dictionary of testing data (includes images
                           and labels).
    """
    # Set default cache path if not specified
    cache_path, cache_path_display = get_cache_path(cache_path)

    # Create cache directory and generate data if not already present.
    npz_train_file = os.path.join(cache_path, "dcnv_foam1_train.npz")
    npz_test_file = os.path.join(cache_path, "dcnv_foam1_test.npz")

    if os.path.isfile(npz_train_file) and os.path.isfile(npz_test_file):
        # Load data and convert arrays to float32.
        trdt = np.load(npz_train_file)  # Training
        ttdt = np.load(npz_test_file)  # Testing
        train_in = trdt["image"].astype(np.float32)
        train_out = trdt["label"].astype(np.float32)
        test_in = ttdt["image"].astype(np.float32)
        test_out = ttdt["label"].astype(np.float32)

        # Check image size
        if train_in.shape[1] != size:
            runtime_error_scalar("size", "training", size, train_in.shape[1])
        if test_in.shape[1] != size:
            runtime_error_scalar("size", "testing ", size, test_in.shape[1])

        # Check noise_sigma
        if trdt["noise"] != noise_sigma:
            runtime_error_scalar("noise", "training", noise_sigma, trdt["noise"])
        if ttdt["noise"] != noise_sigma:
            runtime_error_scalar("noise", "testing ", noise_sigma, ttdt["noise"])

        # Check blur kernel
        blur_train = trdt["blur"].astype(np.float32)
        if not np.allclose(blur_kernel, blur_train):
            runtime_error_array("blur", "testing ", np.abs(blur_kernel - blur_train).max())
        blur_test = ttdt["blur"].astype(np.float32)
        if not np.allclose(blur_kernel, blur_test):
            runtime_error_array("blur", "testing ", np.abs(blur_kernel - blur_test).max())

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
                    print_info(
                        "in",
                        cache_path_display,
                        train_ds["image"],
                        train_ds["label"],
                        test_ds["image"].shape[0],
                    )

                return train_ds, test_ds

            elif verbose:
                print_data_warning("testing ", test_nimg, ttdt["numimg"])
        elif verbose:
            print_data_warning("training", train_nimg, trdt["numimg"])

    # Generate new data.
    nimg = train_nimg + test_nimg
    img, blrn = generate_blur_data(
        nimg,
        size,
        blur_kernel,
        noise_sigma,
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
        noise=noise_sigma,
        blur=blur_kernel.astype(np.float32),
    )
    np.savez(
        npz_test_file,
        image=test_ds["image"],
        label=test_ds["label"],
        numimg=test_nimg,
        noise=noise_sigma,
        blur=blur_kernel.astype(np.float32),
    )

    if verbose:
        print_info(
            "out",
            cache_path_display,
            train_ds["image"],
            train_ds["label"],
            test_ds["image"].shape[0],
        )

    return train_ds, test_ds


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
    """Load or load and preprocess image data.

    Load or load and preprocess image data for training of neural
    network models. The original source is the BSDS500 data from the
    Berkeley Segmentation Dataset and Benchmark project. Depending on
    the intended applications, different preprocessings can be performed
    to the source data.

    If a cached file exists, and enough images were sampled, data is
    loaded and returned.

    If either `size` or type of data (gray scale or color) requested
    does not match the data read from the cached file, a
    `RunTimeError` is generated. In contrast, there is no checking for
    the specific contamination (i.e. noise level, blur kernel, etc.).

    If no cached file is found or not enough images were sampled and
    stored in the file, a new data set is generated and stored in
    `cache_path`. The data is stored in `.npz` format for convenient
    access via :func:`numpy.load`. The data is saved in two distinct
    files: `*_bsds_train.npz` and `*_bsds_test.npz` to keep separated
    training and testing partitions. The * stands for `dn` if
    denoising problem or `dcnv` if deconvolution problem. Other types
    of pre-processings may be specified via the `transf` operator.

    Args:
        train_nimg: Number of images required for sampling training data.
        test_nimg: Number of images required for sampling testing data.
        size: Size of reconstruction images.
        gray_flag: Flag to indicate if gray scale images or color
            images. When ``True`` gray scale images are used.
        data_mode: Type of image problem. Options are: `dn` for
            denoising, `dcnv` for deconvolution.
        cache_path: Directory in which processed data is saved.
            Default: ``None``.
        verbose: Flag indicating whether to print status messages.
            Default: ``False``.
        noise_level: Standard deviation of the Gaussian noise.
        noise_range: Flag to indicate if a fixed or a random standard
            deviation must be used. Default: ``False`` i.e. fixed
            standard deviation given by `noise_level`.
        transf: Operator for blurring or other non-trivial
            transformations. Should be able to handle batched (NHWC)
            data. Default: ``None``.
        stride: Stride between patch origins (indexed from left-top
            corner). Default: 0 (i.e. no stride, only one patch per
            image).
        augment: Augment training data set by flip and 90 degrees
            rotation. Default: ``False`` (i.e. no augmentation).

    Returns:
       tuple: A tuple (train_ds, test_ds) containing:

           - **train_ds** : (DataSetDict): Dictionary of training data
                            (includes images and labels).
           - **test_ds** : (DataSetDict): Dictionary of testing data
                           (includes images and labels).
    """
    # Set default cache path if not specified
    cache_path, cache_path_display = get_cache_path(cache_path)

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
                print_info(
                    "in",
                    cache_path_display,
                    train_ds["image"],
                    train_ds["label"],
                    test_ds["image"].shape[0],
                )

                print(
                    "NOTE: If blur kernel or noise parameter are changed, the cache "
                    "must be manually\n      deleted to ensure that the training data"
                    " is regenerated with the new\n      parameters."
                )

            return train_ds, test_ds

    # Check if BSDS folder exists if not create and download BSDS data.
    bsds_cache_path = os.path.join(cache_path, "BSDS")
    if not os.path.isdir(bsds_cache_path):
        os.makedirs(bsds_cache_path)
        get_bsds_data(path=bsds_cache_path, verbose=verbose)
    # Load data, convert arrays to float32 and return
    # after pre-processing for specified data_mode.
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
        print_info(
            "out",
            cache_path_display,
            train_ds["image"],
            train_ds["label"],
            test_ds["image"].shape[0],
        )

    return train_ds, test_ds


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
    """Check data loaded with respect to data requirements.

    Args:
        train_nimg: Number of images required for training data.
        test_nimg: Number of images required for testing data.
        size: Size of images requested.
        gray_flag: Flag to indicate if gray scale images or color images
            are requested. When ``True`` gray scale images are used,
            therefore, one channel is expected.
        train_in_shp: Shape of images/patches loaded as training data.
        test_in_shp: Shape of images/patches loaded as testing data.
        train_nimg_avail: Number of images available in loaded training
            image data.
        test_nimg_avail: Number of images available in loaded testing
            image data.
        verbose: Flag indicating whether to print status messages.

    Returns:
       ``True`` if the loaded image data satifies requirements of size,
       number of samples and number of channels and ``False`` otherwise.
    """
    # Check image size
    if train_in_shp[1] != size:
        runtime_error_scalar("size", "training", size, train_in_shp[1])
    if test_in_shp[1] != size:
        runtime_error_scalar("size", "testing ", size, test_in_shp[1])

    # Check gray scale or color images.
    C_train = train_in_shp[-1]
    C_test = test_in_shp[-1]
    if gray_flag:
        C = 1
    else:
        C = 3
    if C_train != C:
        runtime_error_scalar("channels", "training", C, C_train)
    if C_test != C:
        runtime_error_scalar("channels", "testing ", C, C_test)

    # Check that enough images were sampled.
    if train_nimg_avail >= train_nimg:
        if test_nimg_avail >= test_nimg:
            return True

        elif verbose:
            print_data_warning("testing ", test_nimg, test_nimg_avail)
    elif verbose:
        print_data_warning("training", train_nimg, train_nimg_avail)

    return False


def print_input_path(path_display: str):  # pragma: no cover
    """Display path from where data is being loaded.

    Args:
        path_display: Path for loading data.
    """
    print(f"Data read from path: {path_display}")


def print_output_path(path_display: str):  # pragma: no cover
    """Display path where data is being stored.

    Args:
        path_display: Path for storing data.
    """
    print(f"Storing data in path: {path_display}")


def print_data_range(idstring: str, data: Array):  # pragma: no cover
    """Display min and max values of given data array.

    Args:
        idstring: Data descriptive string.
        data: Array to compute min and max.
    """
    print(f"Data range --{idstring}--  Min: {data.min():>5.2f}  " f"Max: {data.max():>5.2f}")


def print_data_size(idstring: str, size: int):  # pragma: no cover
    """Display integer given.

    Args:
        idstring: Data descriptive string.
        size: Integer representing size of a set.
    """
    print(f"Set --{idstring}-- size: {size}")


def print_info(
    iomode: str, path_display: str, train_in: Array, train_out: Array, test_size: int
):  # pragma: no cover
    """Display information related to data input/output.

    Args:
        iomode: Identification of input (load) or ouput (save)
            operation.
        path_display: Input or output path.
        train_in: Input features in training set.
        train_out: Outputs in training set.
        test_size: Size of testing set.
    """
    if iomode == "in":
        print_input_path(path_display)
    else:
        print_output_path(path_display)
    print_data_size("training", train_in.shape[0])
    print_data_size("testing ", test_size)
    print_data_range(" images ", train_in)
    print_data_range(" labels ", train_out)


def print_data_warning(idstring: str, requested: int, available: int):  # pragma: no cover
    """Display warning related to data size demands not satisfied.

    Args:
        idstring: Data descriptive string.
        requested: Size of data set requested.
        available: Size of data set available.
    """
    print(
        f"Not enough images sampled in {idstring} file. "
        f"Requested: {requested}  Available: {available}"
    )


def runtime_error_scalar(
    type: str, idstring: str, requested: Union[int, float], available: Union[int, float]
):
    """Raise run time error related to unsatisfied scalar parameter request.

    Raise run time error related to scalar parameter request not satisfied
    in available data.

    Args:
        type: Type of parameter in the request.
        idstring: Data descriptive string.
        requested: Parameter value requested.
        available: Parameter value available in data.
    """
    raise RuntimeError(
        f"Requested value of argument '{type}' does not match value "
        f"read from {idstring} file. Requested: {requested}  Available: "
        f"{available}.\nDelete cache and check data source."
    )


def runtime_error_array(type: str, idstring: str, maxdiff: float):
    """Raise run time error related to unsatisfied array parameter request.

    Raise run time error related to array parameter request not satisfied
    in available data.

    Args:
        type: Type of parameter in the request.
        idstring: Data descriptive string.
        maxdiff: Maximum error between requested and available array
           entries.
    """
    raise RuntimeError(
        f"Requested value of argument '{type}' does not match value "
        f"read from {idstring} file. Maximum array difference: "
        f"{maxdiff:>5.3f}.\nDelete cache and check data source."
    )
