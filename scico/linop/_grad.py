# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Non-Cartesian gradient linear operators."""


# Needed to annotate a class method that returns the encapsulating class
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.typing import DType, JaxArray, Shape

from ._linop import LinearOperator


class ProjectedGradient(LinearOperator):
    """Gradient projected onto local coordinate system.

    This class represents a linear operator that computes gradients of
    arrays projected onto a local coordinate system that may differ at
    every position in the array. In the 2D illustration below :math:`x`
    and :math:`y` represent the standard coordinate system defined by the
    array axes, :math:`(g_x, g_y)` is the gradient vector within that
    coordinate system, :math:`x'` and :math:`y'` are the local coordinate
    axes, and :math:`(g_x', g_y')` is the gradient vector within the
    local coordinate system.

    .. image:: /figures/projgrad.svg
         :align: center
         :alt: Figure illustrating projection of gradient onto local coordinate system.

    Each of the local coordinate axes (e.g. :math:`x'` and :math:`y'` in
    the illustration above) is represented by a separate array in the
    `coord` tuple of arrays parameter of the class initializer.

    .. note::

       This operator should not be confused with the Projected Gradient
       optimization algorithm (a special case of Proximal Gradient), with
       which it is unrelated.
    """

    def __init__(
        self,
        input_shape: Shape,
        axes: Optional[Tuple[int]] = None,
        coord: Optional[Tuple[Union[JaxArray, BlockArray]]] = None,
        input_dtype: DType = np.float32,
        jit: bool = True,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            axes: Axes over which to compute the gradient. Defaults to
                ``None``, in which case the gradient is computed along
                all axes.
            coord: A tuple of arrays, each of which specifies a local
                coordinate axis direction.  Each member of the tuple
                should either be a `DeviceArray` or a
                :class:`.BlockArray`. If it is the former, it should
                have shape :math:`N \times M_0 \times M_1 \times
                \ldots`, where :math:`N` is the number of axes
                specified by parameter `axes`, and :math:`M_i` is the
                size of the :math:`i^{\mrm{th}}` axis. If it is the
                latter, it should consist of :math:`N` blocks, each of
                which has a shape that is suitable for multiplication
                with an array of shape :math:`M_0 \times M_1 \times
                \ldots`. If `coord` is a singleton tuple, the result
                of applying the operator is a `DeviceArray`; otherwise
                it consists of the gradients for each of the local
                coordinate axes in `coord` stacked into a
                :class:`.BlockArray`. If `coord` is ``None``, which is
                the default, gradients are computed in the standard
                axis-aligned coordinate system, and the return type
                depends on the number of axes on which the gradient is
                calculated, as specified explicitly or implicitly via
                the `axes` parameter.
            input_dtype: `dtype` for input argument. Default is
                ``float32``.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """
        if axes is None:
            # If axes is None, set it to all axes in input shape.
            self.axes = tuple(range(len(input_shape)))
        else:
            # Ensure no invalid axis indices specified.
            if np.any(np.array(axes) >= len(input_shape)):
                raise ValueError(
                    "Invalid axes specified; all elements of `axes` must be less than "
                    f"len(input_shape)={len(input_shape)}"
                )
            self.axes = axes
        if coord is None:
            # If coord is None, output shape is determined by number of axes.
            if len(self.axes) == 1:
                output_shape = input_shape
            else:
                output_shape = (input_shape,) * len(self.axes)
        else:
            # If coord is not None, output shape is determined by number of coord arrays.
            if len(coord) == 1:
                output_shape = input_shape
            else:
                output_shape = (input_shape,) * len(coord)
        self.coord = coord
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
            jit=jit,
        )

    def _eval(self, x: JaxArray) -> Union[JaxArray, BlockArray]:

        grad = snp.gradient(x, axis=self.axes)
        if self.coord is None:
            # If coord attribute is None, just return gradients on specified axes.
            return BlockArray.array(grad)
        else:
            # If coord attribute is not None, return gradients projected onto specified local.
            # coordinate systems
            projgrad = [sum([c[m] * grad[m] for m in range(len(grad))]) for c in self.coord]
            if len(self.coord) == 1:
                return projgrad[0]
            else:
                return BlockArray.array(projgrad)


class PolarGradient(ProjectedGradient):
    """Gradient projected into polar coordinates.

    Compute gradients projected onto angular and/or radial axis
    directions. Local coordinate axes are illustrated in the figure
    below.

    .. plot:: figures/polargrad.py
       :align: center
       :include-source: False

    |

    If only one of `angular` and `radial` is ``True``, the operator output
    is a `DeviceArray`, otherwise it is a :class:`.BlockArray`.
    """

    def __init__(
        self,
        input_shape: Shape,
        axes: Optional[Tuple[int]] = None,
        center: Optional[Union[Tuple[int], JaxArray]] = None,
        angular: bool = True,
        radial: bool = True,
        input_dtype: DType = np.float32,
        jit: bool = True,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            axes: Axes over which to compute the gradient. Defaults to
                ``None``, in which case the axes are taken to be `(0, 1)`.
            center: Center of the polar coordinate system in array
                indexing coordinates. Default is ``None``, which places
                the center at the center of the input array.
            angular: Flag indicating whether to compute gradients in the
                angular (i.e. tangent to circles) direction.
            radial: Flag indicating whether to compute gradients in the
                radial (i.e. directed outwards from the origin) direction.
            input_dtype: `dtype` for input argument. Default is ``float32``.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """

        if len(input_shape) < 2:
            raise ValueError("Invalid input shape; input must have at least two axes")
        if axes is not None and len(axes) != 2:
            raise ValueError("Invalid axes specified; exactly two axes must be specified")
        if not angular and not radial:
            raise ValueError("At least one of angular and radial must be True")

        if axes is None:
            axes = (0, 1)
        axes_shape = [input_shape[ax] for ax in axes]
        if center is None:
            center = (snp.array(axes_shape) - 1) / 2
        end = snp.array(axes_shape) - center
        g0, g1 = np.mgrid[-center[0] : end[0], -center[1] : end[1]]
        theta = snp.arctan2(g0, g1)
        if len(input_shape) > 2:
            # Construct list of input axes that are not included in the gradient axes.
            single = tuple(set(range(len(input_shape))) - set(axes))
            # Insert singleton axes to align theta for multiplication with gradients.
            theta = snp.expand_dims(theta, single)
        coord = []
        if angular:
            coord.append(BlockArray.array([-snp.cos(theta), snp.sin(theta)]))
        if radial:
            coord.append(BlockArray.array([snp.sin(theta), snp.cos(theta)]))
        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            axes=axes,
            coord=coord,
            jit=jit,
        )


class CylindricalGradient(ProjectedGradient):
    """Gradient projected into cylindrical coordinates.

    Compute gradients projected onto cylindrical coordinate axes. The
    local coordinate axes are illustrated in the figure below.

    .. plot:: figures/cylindgrad.py
       :align: center
       :include-source: False

    |

    If only one of `angular`, `radial`, and `axial` is ``True``, the
    operator output is a `DeviceArray`, otherwise it is a
    :class:`.BlockArray`.
    """

    def __init__(
        self,
        input_shape: Shape,
        axes: Optional[Tuple[int]] = None,
        center: Optional[Union[Tuple[int], JaxArray]] = None,
        angular: bool = True,
        radial: bool = True,
        axial: bool = True,
        input_dtype: DType = np.float32,
        jit: bool = True,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            axes: Axes over which to compute the gradient. Defaults to
                ``None``, in which case the axes are taken to be
                `(0, 1, 2)`. If an integer, this operator returns a
                `DeviceArray`. If a tuple or ``None``, the resulting
                arrays are stacked into a :class:`.BlockArray`.
            center: Center of the cylindrical coordinate system in array
                indexing coordinates. Default is ``None``, which places
                the center at the center of the two polar axes of the
                input array and at the zero index of the axial axis.
            angular: Flag indicating whether to compute gradients in the
                angular (i.e. tangent to circles) direction.
            radial: Flag indicating whether to compute gradients in the
                radial (i.e. directed outwards from the origin) direction.
            axial: Flag indicating whether to compute gradients in the
                direction of the axis of the cylinder.
            input_dtype: `dtype` for input argument. Default is
                ``float32``.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """

        if len(input_shape) < 3:
            raise ValueError("Invalid input shape; input must have at least three axes")
        if axes is not None and len(axes) != 3:
            raise ValueError("Invalid axes specified; exactly three axes must be specified")
        if not angular and not radial and not axial:
            raise ValueError("At least one of angular, radial, and axial must be True")

        if axes is None:
            axes = (0, 1, 2)
        axes_shape = [input_shape[ax] for ax in axes]
        if center is None:
            center = (snp.array(axes_shape) - 1) / 2
            center = center.at[-1].set(0)
        end = snp.array(axes_shape) - center
        g0, g1 = np.mgrid[-center[0] : end[0], -center[1] : end[1]]
        g0 = g0[..., np.newaxis]
        g1 = g1[..., np.newaxis]
        theta = snp.arctan2(g0, g1)
        if len(input_shape) > 3:
            # Construct list of input axes that are not included in the gradient axes.
            single = tuple(set(range(len(input_shape))) - set(axes))
            # Insert singleton axes to align theta for multiplication with gradients.
            theta = snp.expand_dims(theta, single)
        coord = []
        if angular:
            coord.append(BlockArray.array([-snp.cos(theta), snp.sin(theta), np.array([0])]))
        if radial:
            coord.append(BlockArray.array([snp.sin(theta), snp.cos(theta), np.array([0])]))
        if axial:
            coord.append(BlockArray.array([np.array([0]), np.array([0]), np.array([1])]))
        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            axes=axes,
            coord=coord,
            jit=jit,
        )


class SphericalGradient(ProjectedGradient):
    """Gradient projected into spherical coordinates.

    Compute gradients projected onto spherical coordinate axes. The local
    coordinate axes are illustrated in the figure below.

    .. plot:: figures/spheregrad.py
       :align: center
       :include-source: False

    |

    If only one of `azimuthal`, `polar`, and `radial` is ``True``, the
    operator output is a `DeviceArray`, otherwise it is a
    :class:`.BlockArray`.
    """

    def __init__(
        self,
        input_shape: Shape,
        axes: Optional[Tuple[int]] = None,
        center: Optional[Union[Tuple[int], JaxArray]] = None,
        azimuthal: bool = True,
        polar: bool = True,
        radial: bool = True,
        input_dtype: DType = np.float32,
        jit: bool = True,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            axes: Axes over which to compute the gradient.  Defaults to
                ``None``, in which case the axes are taken to be
                `(0, 1, 2)`. If an integer, this operator returns a
                `DeviceArray`. If a tuple or ``None``, the resulting
                arrays are stacked into a :class:`.BlockArray`.
            center: Center of the spherical coordinate system in array
                indexing coordinates. Default is ``None``, which places
                the center at the center of the input array.
            azimuthal: Flag indicating whether to compute gradients in
                the azimuthal direction.
            polar: Flag indicating whether to compute gradients in the
                polar direction.
            radial: Flag indicating whether to compute gradients in the
                radial direction.
            input_dtype: `dtype` for input argument. Default is
                ``float32``.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """

        if len(input_shape) < 3:
            raise ValueError("Invalid input shape; input must have at least three axes")
        if axes is not None and len(axes) != 3:
            raise ValueError("Invalid axes specified; exactly three axes must be specified")
        if not azimuthal and not polar and not radial:
            raise ValueError("At least one of azimuthal, polar, and radial must be True")

        if axes is None:
            axes = (0, 1, 2)
        axes_shape = [input_shape[ax] for ax in axes]
        if center is None:
            center = (snp.array(axes_shape) - 1) / 2
        end = snp.array(axes_shape) - center
        g0, g1, g2 = np.mgrid[-center[0] : end[0], -center[1] : end[1], -center[2] : end[2]]
        theta = snp.arctan2(g1, g0)
        phi = snp.arctan2(np.sqrt(g0**2 + g1**2), g2)
        if len(input_shape) > 3:
            # Construct list of input axes that are not included in the gradient axes.
            single = tuple(set(range(len(input_shape))) - set(axes))
            # Insert singleton axes to align theta for multiplication with gradients.
            theta = snp.expand_dims(theta, single)
            phi = snp.expand_dims(phi, single)
        coord = []
        if azimuthal:
            coord.append(BlockArray.array([snp.sin(theta), -snp.cos(theta), np.array([0])]))
        if polar:
            coord.append(
                BlockArray.array(
                    [snp.cos(phi) * snp.cos(theta), snp.cos(phi) * snp.sin(theta), -snp.sin(phi)]
                )
            )
        if radial:
            coord.append(
                BlockArray.array(
                    [snp.sin(phi) * snp.cos(theta), snp.sin(phi) * snp.sin(theta), snp.cos(phi)]
                )
            )
        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            axes=axes,
            coord=coord,
            jit=jit,
        )
