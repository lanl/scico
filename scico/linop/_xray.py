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

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._linop import LinearOperator


class XRayProject(LinearOperator):
    """options to select type of projection"""

    def __init__(self, projector):
        self._eval = projector.project

        super().__init__(
            input_shape=projector.im_shape,
            output_shape=(len(projector.angles), *projector.det_shape),
        )


class ParallelFixedAxis2dProjector:
    """Parallel ray, single axis, 2D X-ray projector"""

    def __init__(self, im_shape, angles, det_length=None, dither=True):
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
            # fast, but does not allow easy dithering
            # dydx = jnp.stack((jnp.cos(angle), jnp.sin(angle)))
            # Px0 = jnp.dot(x0, dydx)
            # Px = (
            #     Px0
            #     + dydx[0] * jnp.arange(im_shape[0])[:, jnp.newaxis]
            #     + dydx[1] * jnp.arange(im_shape[1])[jnp.newaxis, :]
            # )

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
            if dither:
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

        inds = compute_inds(angles)

        @partial(jax.vmap, in_axes=(None, 0))
        def project_inds(im: ArrayLike, inds: ArrayLike) -> ArrayLike:
            return jnp.zeros(det_length).at[inds].add(im)

        @jax.jit
        def project(im: ArrayLike) -> ArrayLike:
            return project_inds(im, inds)

        self.project = project


# num_angles = 127

# x = jnp.ones((128, 129))


# H = ParallelFixedAxis2dProjector(x.shape, angles)
# y1 = H.project(x)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# ax.imshow(y)
# fig.show()

# f = lambda x: H.project(x)[65, 90]
# grad_f = jax.grad(f)

# fig, ax = plt.subplots()
# ax.imshow(grad_f(x))
# fig.show()


# ## back project


# bad_angle = jnp.array([jnp.pi / 4])
# H = ParallelFixedAxis2dProjector(x.shape, bad_angle)
# y = H.project(x)


# fig, ax = plt.subplots()
# ax.plot(y[0])
# fig.show()
