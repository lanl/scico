# -*- coding: utf-8 -*-
# Copyright (C) 2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functions for extracting blocks from images and sets of images."""

import glob
import math
import random
from collections.abc import Sequence
from itertools import pairwise, zip_longest
from operator import itemgetter
from typing import Any, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

import imageio.v3 as iio
from imageio.core.v3_plugin_api import ImageProperties

T = TypeVar("T")


def split_sequence(
    items: Sequence[T],
    fractions: Sequence[float],
    randomize: bool = True,
) -> list[list[T]]:
    """Split a sequence into partitions according to relative fractions.

    Args:
        items: Sequence of elements to partition.
        fractions: Sequence of fractional partition sizes. If they do not
            sum to 1.0, they are rescaled so that they do.
        randomize: If ``True``, shuffle element indices before partitioning.

    Returns:
        A list of sublists corresponding to each entry in `fractions`.
    """
    if np.sum(fractions) != 1.0:
        fractions = (np.asarray(fractions, dtype=float) / np.sum(fractions)).tolist()

    indices = list(range(len(items)))
    if randomize:
        random.shuffle(indices)

    # Convert cumulative fractions into integer boundary indices.
    split_indices = [0] + np.rint(np.cumsum(fractions) * len(indices)).astype(np.uint).tolist()

    if randomize:
        partition_indices = [sorted(indices[start:end]) for start, end in pairwise(split_indices)]
    else:
        partition_indices = [indices[start:end] for start, end in pairwise(split_indices)]

    # Preserve original element ordering even when shuffled by sorting indices per partition.
    partitions = [list(itemgetter(*part_idx)(items)) for part_idx in partition_indices]
    return partitions


def image_nbytes(props: ImageProperties) -> int:
    """Compute the uncompressed byte size of an image from its metadata.

    Args:
        props: Image properties object (from `imageio.v3.improps`) exposing
            `shape` and `dtype`.

    Returns:
        Total size in bytes.
    """
    return int(np.prod(props.shape).item() * np.dtype(props.dtype).itemsize)


def image_channels(props: ImageProperties) -> int:
    """Return the number of channels for an image given its metadata.

    Args:
        props: Image properties object (from `imageio.v3.improps`) exposing
            `shape`.

    Returns:
        Channel count (:code:`shape[-1]` for 3D arrays, otherwise ``1``).
    """
    return int(props.shape[-1]) if len(props.shape) == 3 else 1


def extract_blocks(
    image: npt.NDArray[Any] | tuple[npt.NDArray[Any], ...],
    block_shape: Tuple[int, ...],
    block_step: Optional[Tuple[int, ...]] = None,
) -> npt.NDArray[Any]:
    """Extract sliding blocks from an image array using strided views.

    Args:
        image: Input array (or tuple of arrays stacked along the last axis) from
            which blocks are extracted.
        block_shape: Tuple specifying the block dimensions along the leading axes.
        block_step: Tuple specifying the step size between adjacent blocks along
            each dimension in `block_shape`. Defaults to a step of 1 per axis.

    Returns:
        A strided array view with shape
        :code:`(*num_blocks, *block_shape, *image.shape[len(block_shape):])`, where
        `num_blocks` gives the number of blocks along each axis in `block_shape`.
    """
    # Reference: http://stackoverflow.com/questions/16774148 and
    # sklearn.feature_extraction.image.extract_patches_2d.
    if isinstance(image, tuple):
        image = np.stack(image, axis=-1)

    if block_step is None:
        block_step = (1,) * len(block_shape)

    image_shape = image.shape

    # Number of full blocks that fit along each leading dimension.
    num_blocks = tuple(
        int(np.floor((dim_size - blk_size) / step_size) + 1)
        for dim_size, blk_size, step_size in zip(image_shape, block_shape, block_step)
    )
    # Stride required to jump from one block origin to the next along each axis.
    block_strides = tuple(
        arr_stride * step_size for arr_stride, step_size in zip(image.strides, block_step)
    )

    new_shape = num_blocks + block_shape + image_shape[len(block_shape) :]
    new_strides = block_strides + image.strides

    # `as_strided` creates a zero-copy view; avoid mutating the returned array in place.
    blocks = np.lib.stride_tricks.as_strided(image, new_shape, new_strides)
    return blocks


def num_blocks_in_image(
    image_shape: Tuple[int, ...],
    block_shape: Tuple[int, ...],
    block_step: Tuple[int, ...],
) -> int:
    """Compute the total number of blocks extractable from a given image shape.

    Args:
        image_shape: Spatial or array dimensions of the image.
        block_shape: Target block dimensions.
        block_step: Step sizes between blocks; missing entries default to ``1``.

    Returns:
        Total number of blocks across all dimensions.
    """
    num_blocks = tuple(
        int(np.floor((dim_size - blk_size) / step_size) + 1)
        for dim_size, blk_size, step_size in zip_longest(
            image_shape, block_shape, block_step, fillvalue=1
        )
    )
    return math.prod(num_blocks)


def select_image_files(
    glob_pattern: str,
    num_channels: int,
    dtype: npt.DTypeLike,
) -> list[str]:
    """Find image files matching a glob pattern that have a target channel count and dtype.

    Args:
        glob_pattern: File path pattern passed to `glob.glob`.
        num_channels: Required number of image channels.
        dtype: Required NumPy-compatible data type.

    Returns:
        Sorted list of matching file paths.
    """
    target_dtype = np.dtype(dtype)
    selected_paths: list[str] = []

    for path in sorted(glob.glob(glob_pattern)):
        props = iio.improps(path)
        if image_channels(props) == num_channels and np.dtype(props.dtype) == target_dtype:
            selected_paths.append(path)

    return selected_paths


def num_blocks_per_image(
    paths: Sequence[str],
    block_shape: Tuple[int, ...],
    block_step: Tuple[int, ...],
) -> npt.NDArray[np.uint32]:
    """Compute the number of extractable 2D blocks for each image path.

    Args:
        paths: Sequence of image file paths.
        block_shape: Block height and width.
        block_step: Step size between blocks along height and width.

    Returns:
        1D `uint32` array containing the block count for each file in `paths`.
    """
    counts: npt.NDArray[np.uint32] = np.zeros(len(paths), dtype=np.uint32)

    for idx, path in enumerate(paths):
        props = iio.improps(path)
        # Only inspect the first two spatial axes (height, width).
        counts[idx] = num_blocks_in_image(props.shape[0:2], block_shape, block_step)

    return counts


def sample_image_blocks(
    paths: Sequence[str],
    block_shape: Tuple[int, ...],
    block_step: Tuple[int, ...],
    num_blocks: int,
) -> npt.NDArray[Any]:
    """Randomly sample a fixed number of blocks across a collection of images.

    Args:
        paths: Sequence of image file paths.
        block_shape: Block dimensions passed to `extract_blocks`.
        block_step: Step dimensions passed to `extract_blocks`.
        num_blocks: Total number of blocks to sample across all images.

    Returns:
        Array of sampled blocks with shape
        :code:`(num_blocks, *block_shape, *trailing_dims)`.

    Raises:
        ValueError: If `num_blocks` exceeds the total number of available blocks.
    """
    blocks_per_image = num_blocks_per_image(paths, block_shape, block_step)
    total_blocks = int(blocks_per_image.sum().item())

    if num_blocks > total_blocks:
        raise ValueError(f"Only {total_blocks} available and {num_blocks} requested.")

    # Cumulative block offsets map global block indices to per-image local indices.
    split_indices = [0] + np.cumsum(blocks_per_image).tolist()
    selected_global_indices = np.array(sorted(random.sample(range(total_blocks), num_blocks)))

    selected_local_indices_by_image = [
        selected_global_indices[
            np.logical_and(selected_global_indices >= start, selected_global_indices < end)
        ]
        - start
        for start, end in pairwise(split_indices)
    ]

    sampled_blocks: Optional[npt.NDArray[Any]] = None
    write_offset = 0

    for img_idx, path in enumerate(paths):
        local_blk_indices = selected_local_indices_by_image[img_idx]
        if local_blk_indices.size == 0:
            continue

        image = iio.imread(path)
        image_blocks = extract_blocks(image, block_shape, block_step)

        if sampled_blocks is None:
            # Allocate output buffer once the block shape and dtype are
            # known from the first non-empty image.
            sampled_blocks = np.zeros(
                (num_blocks,) + image_blocks.shape[2:],
                dtype=image_blocks.dtype,
            )

        # Unravel local indices against `image_blocks.shape` (note: first
        # two axes are grid indices).
        grid_coords = np.unravel_index(local_blk_indices, image_blocks.shape)
        selected_slice = image_blocks[grid_coords[0], grid_coords[1]]

        next_offset = write_offset + selected_slice.shape[0]
        sampled_blocks[write_offset:next_offset] = selected_slice
        write_offset = next_offset

    if sampled_blocks is None:
        return np.empty((0,), dtype=np.float64)

    return sampled_blocks
