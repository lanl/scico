# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.


"""
X-ray projector classes.
"""
from functools import partial
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from scico.typing import Shape

from ._linop import LinearOperator


class XRayProject(LinearOperator):
    """X-ray projection operator.

    Wraps an X-ray projector object in a SCICO
    :class:`LinearOperator`.
    """

    def __init__(self, projector):
        r"""
        Args:
            projector: instance of an X-ray projector object to wrap,
                currently the only option is
                :class:`ParallelFixedAxis2dProjector`
        """
        self._eval = projector.project

        super().__init__(
            input_shape=projector.im_shape,
            output_shape=(len(projector.angles), *projector.det_shape),
        )


class ParallelFixedAxis2dProjector:
    """Parallel ray, single axis, 2D X-ray projector."""

    def __init__(
        self,
        im_shape: Shape,
        angles: ArrayLike,
        det_length: Optional[int] = None,
        do_dithering: bool = True,
    ):
        r"""
        Args:
            im_shape: Shape of input array.
            angles: (num_angles,) array of angles in radians.
            det_length: Length of detector, in ``None``, defaults to the
                length of diagonal of `im_shape`.
            do_dither: If ``True`` randomly shift pixel locations to
                reduce projection artifacts caused by aliasing.
        """
        self.im_shape = im_shape
        self.angles = angles

        im_shape = np.array(im_shape)

        x0 = -(im_shape - 1) / 2

        if det_length is None:
            det_length = int(np.ceil(np.linalg.norm(im_shape)))
        self.det_shape = (det_length,)

        y0 = -det_length / 2

        @jax.vmap
        def compute_inds(angle: float) -> ArrayLike:
            """Project pixel positions on to a detector at the given
            angle, determine which detector element they contribute to.
            """
            x = jnp.stack(
                jnp.meshgrid(
                    *(
                        jnp.arange(shape_i) * step_i + start_i
                        for start_i, step_i, shape_i in zip(x0, [1, 1], im_shape)
                    ),
                    indexing="ij",
                ),
                axis=-1,
            )

            # dither
            if do_dithering:
                key = jax.random.PRNGKey(0)
                x = x + jax.random.uniform(key, shape=x.shape, minval=-0.5, maxval=0.5)

            # project
            Px = x[..., 0] * jnp.cos(angle) + x[..., 1] * jnp.sin(angle)

            # quantize
            inds = jnp.floor((Px - y0)).astype(int)

            # map negative inds to y_size, which is out of bounds and will be ignored
            # otherwise they index from the end like x[-1]
            inds = jnp.where(inds < 0, det_length, inds)

            return inds

        inds = compute_inds(angles)  # (len(angles), *im_shape)

        @partial(jax.vmap, in_axes=(None, 0))
        def project_inds(im: ArrayLike, inds: ArrayLike) -> ArrayLike:
            """Compute the projection at a single angle."""
            return jnp.zeros(det_length).at[inds].add(im)

        @jax.jit
        def project(im: ArrayLike) -> ArrayLike:
            """Compute the projection for all angles."""
            return project_inds(im, inds)

        self.project = project
