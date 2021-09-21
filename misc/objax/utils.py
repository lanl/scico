#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functionality for image manipulation.
"""

import copy
import glob
import math
import os

import numpy as np

import jax
import jax.numpy as jnp

import imageio
from sklearn.model_selection import train_test_split


def rgb2gray(rgb):
    """Utility to convert color to grayscale image"""

    w = np.array([0.299, 0.587, 0.114], dtype=rgb.dtype)[np.newaxis, np.newaxis]
    return np.sum(w * rgb, axis=2)


class CenterCrop:
    """Crop the image to the specified size. The central part of the image
       is preserved.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[top : top + new_h, left : left + new_w]

        return image


class PositionalCrop:
    """Crop the image from a given corner.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, top, left):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        image = image[top : top + new_h, left : left + new_w]

        return image


class ImageCollection:
    """Reads a collection of images and represents it as 4D
    numpy array of dimensions (N, H, W, C).
    N: number of images
    H, W: spatial dimensions
    C: number of channels
    """

    def __init__(self, root_dir: str, ext: str, output_size, gray_flag=False, num_img=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            ext (string): Extension of images to read
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
            gray_flag (boolean): If true, converts to gray scale
            num_img (int, optional): If specified, reads that number of
                images, if not reads all the images in directory
        """
        # Functionality to crop
        tsfm = CenterCrop(output_size)
        # Read images
        count = 0
        S = None
        for filename in sorted(glob.glob(os.path.join(root_dir, "*." + ext))):
            img = np.float32(imageio.imread(filename)) / 255.0
            if img.ndim < 3:  # gray scale image --> add singleton for channels and images
                img = np.reshape(img, (img.shape + (1,) + (1,)))
            else:  # color image
                if gray_flag:  # convert to gray scale and add singleton for channels and images
                    imgG = rgb2gray(img)
                    img = np.reshape(imgG, (imgG.shape + (1,) + (1,)))
                else:  # add singleton for images
                    img = np.reshape(img, (img.shape + (1,)))
            # Crop image
            img = tsfm(img)
            # Add image read to collection
            if S is None:
                S = img
            else:
                S = np.concatenate((S, img), axis=3)
            count += 1
            if num_img is not None and count == num_img:
                break
        self.S = np.transpose(S, (3, 0, 1, 2))

    def __len__(self):
        return self.S.shape[0]


class ImageCollectionMulti:
    """Reads a collection of images and represents it as 4D
    numpy array of dimensions (N, H, W, C):
    N: number of images
    H, W: spatial dimensions
    C: number of channels
    Takes multiple patches per image
    """

    def __init__(self, root_dir, ext, output_size, stride, gray_flag=False, num_img=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            ext (string): Extension of images to read
            output_size (tuple or int): Desired output size. If int, square crop is made.
            stride (tuple or int): separation between patch origins (indexed from left-top corner). If int, the same stride is used.
            gray_flag (boolean): If true, converts to gray scale
            num_img (int, optional): If specified, reads that number of
                images, if not reads all the images in directory
        """
        # Functionality to crop
        tsfm = PositionalCrop(output_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        # Read images
        count = 0
        S = None
        for filename in sorted(glob.glob(os.path.join(root_dir, "*." + ext))):
            img = np.float32(imageio.imread(filename)) / 255.0
            if img.ndim < 3:  # gray scale image --> add singleton for channels and images
                img = np.reshape(img, (img.shape + (1,) + (1,)))
            else:  # color image
                if gray_flag:  # convert to gray scale and add singleton for channels and images
                    imgG = rgb2gray(img)
                    img = np.reshape(imgG, (imgG.shape + (1,) + (1,)))
                else:  # add singleton for images
                    img = np.reshape(img, (img.shape + (1,)))
            # Add image read to collection
            if S is None:
                S = []
                S.append(img)
            else:
                S.append(img)
            count += 1
            if num_img is not None and count == num_img:
                break

        # Sample multiple patches from image
        h = img.shape[0]
        w = img.shape[1]
        nh = int(math.floor((h - tsfm.output_size[0]) / stride[0])) + 1
        nw = int(math.floor((w - tsfm.output_size[1]) / stride[1])) + 1
        saux = np.zeros((nh * nw * count, tsfm.output_size[0], tsfm.output_size[1], S[0].shape[2]))
        count2 = 0
        for i in range(len(S)):
            h = S[i].shape[0]
            w = S[i].shape[1]
            for top in range(0, h - tsfm.output_size[0], stride[0]):
                for left in range(0, w - tsfm.output_size[1], stride[1]):
                    saux[count2, ...] = np.transpose(tsfm(S[i], top, left), (3, 0, 1, 2))
                    count2 += 1
        self.S = saux

    def __len__(self):
        return self.S.shape[0]


class RandomNoise:
    """Adds Gaussian noise to the image.

    Args:
        noise_level (float): Standard dev of the Gaussian noise.
        range_flag (boolean): If true, the standard dev is randomly
            selected between 5% and 100% of noise_level set.
    """

    def __init__(self, noise_level, range_flag=False):
        self.noise_level = noise_level
        self.range_flag = range_flag

    def __call__(self, image):

        noise_level = self.noise_level

        if self.range_flag:
            if image.ndim > 3:
                num_img = image.shape[0]
            else:
                num_img = 1
            noise_level = np.random.uniform(0.5 * self.noise_level, self.noise_level, num_img)
            noise_level = noise_level.reshape((noise_level.shape[0],) + (1,) * (image.ndim - 1))

        imgnoised = image + np.random.normal(0.0, noise_level, image.shape)

        return imgnoised


def build_img_dataset(config, channel_first=False, multi=True):
    r"""Reads images from files, prepares the input data and
        returns training and testing dictionaries. Each
        dictionary has images and labels, which are ndarrays
        of dimensions (N, H, W, C)
        N: number of images
        H, W: spatial dimensions
        C: number of channels
        If channel first is true, the ndarrays have
        dimensions (N, C, H, W).

    Args:
        config : configuration of data set to read.
        channel_first : Flag indicating if channel should come before image spatial dimensions.  Default: False.
        multi : Flag indicating cropping of multiple patches per image read.  Default: True.
    """
    # Load data
    if multi:
        Sobj = ImageCollectionMulti(
            config.path,
            config.ext,
            config.output_size,
            config.stride,
            gray_flag=config.run_gray,
            num_img=config.num_img,
        )
    else:
        Sobj = ImageCollection(
            config.path,
            config.ext,
            config.output_size,
            gray_flag=config.run_gray,
            num_img=config.num_img,
        )
    C = Sobj.S.shape[-1]
    S = Sobj.S
    Sobj_test = None
    if config.test_path is not None:  # Read a separate training set
        if multi:
            Sobj_test = ImageCollectionMulti(
                config.test_path,
                config.test_ext,
                config.output_size,
                config.stride,
                gray_flag=config.run_gray,
                num_img=config.test_numimg,
            )
        else:
            Sobj_test = ImageCollection(
                config.test_path,
                config.test_ext,
                config.output_size,
                gray_flag=config.run_gray,
                num_img=config.test_numimg,
            )
        C_test = Sobj_test.S.shape[-1]
        assert C_test == C

    tsfm = None
    # Processing: add noise, blur
    if config.data_mode == "dn":  # Denoise problem
        noise_tsfm = RandomNoise(config.noise_level, config.noise_range)
        tsfm = noise_tsfm
    # elif data_mode == 'dblur': # Debluring problem

    # Generate Data
    if Sobj_test is not None:  # Separate testing data
        S_train = Sobj.S
        ntrain = Sobj.S.shape[0]
        S_test = Sobj_test.S
        ntest = S_test.shape[0]
    else:
        ntest = np.int(Sobj.S.shape[0] * config.test_split)
        ntrain = Sobj.S.shape[0] - ntest
        # train/test split
        y = range(Sobj.S.shape[0])
        train_id, test_id = train_test_split(y, train_size=ntrain, test_size=ntest)
        S_train = Sobj.S[train_id]
        S_test = Sobj.S[test_id]

    if tsfm is not None:  # Transform input
        Stsfm_train = tsfm(S_train)
        Stsfm_test = tsfm(S_test)
        if channel_first:
            train_ds = {
                "images": np.transpose(Stsfm_train, (0, 3, 1, 2)),
                "labels": np.transpose(S_train, (0, 3, 1, 2)),
            }
            test_ds = {
                "images": np.transpose(Stsfm_test, (0, 3, 1, 2)),
                "labels": np.transpose(S_test, (0, 3, 1, 2)),
            }
        else:
            train_ds = {"images": Stsfm_train, "labels": S_train}
            test_ds = {"images": Stsfm_test, "labels": S_test}
    else:
        if channel_first:
            train_ds = {
                "images": np.transpose(S_train, (0, 3, 1, 2)),
                "labels": np.transpose(S_train, (0, 3, 1, 2)),
            }
            test_ds = {
                "images": np.transpose(S_test, (0, 3, 1, 2)),
                "labels": np.transpose(S_test, (0, 3, 1, 2)),
            }
        else:
            train_ds = {"images": S_train, "labels": S_train}
            test_ds = {"images": S_test, "labels": S_test}

    return train_ds, test_ds


class IterateData:
    """Class to prepare image data for training and
    testing. It uses the generator pattern to obtain
    an iterable object.
    """

    def __init__(self, dt, batch_size, is_training=False, rng=None):
        r"""Initialize a :class:`IterateData` object.

        Args:
            dt : dictionary of data including images and labels.
            batch_size (int) : size of batch for iterating through the data.
            is_training (bool) : Flag indicating use of iterator for training.  Iterator for training is infinite, iterator for testing passes once through the data.  Default: False.
            rng : a PRNGKey used as the random key.  Default: None.
        """
        self.dt = dt
        self.is_training = is_training
        self.n = dt["images"].shape[0]
        self.batch_size = batch_size
        self.rng = rng
        self.steps_per_epoch = self.n // batch_size
        self.reset()

    def reset(self):
        """Re-shuffles data in training"""
        if self.is_training:
            self.perms = jax.random.permutation(self.rng, self.n)
        else:
            self.perms = jnp.arange(self.n)

        self.perms = self.perms[: self.steps_per_epoch * self.batch_size]  # skips incomplete batch
        self.perms = self.perms.reshape((self.steps_per_epoch, self.batch_size))
        self.ns = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Gets next batch.
        During training it reshuffles the batches when
        the data is exhausted.
        During testing, it finishes the iterator when
        data is exhausted.
        """
        if self.ns >= self.steps_per_epoch:
            if self.is_training:
                self.reset()
            else:
                self.ns = 0
                raise StopIteration()
        batch = {k: v[self.perms[self.ns], ...] for k, v in self.dt.items()}
        self.ns += 1
        return batch
