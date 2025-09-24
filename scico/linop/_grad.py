# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Non-Cartesian gradient linear operators."""


# Needed to annotate a class method that returns the encapsulating class
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

import scico.numpy as snp
from scico.numpy import Array, BlockArray
from scico.typing import DType, Shape

from ._linop import LinearOperator


def diffstack(x: Array, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    """Compute the discrete difference along multiple axes.

    Apply :func:`snp.diff` along multiple axes, stacking the results on
    a newly inserted axis at index 0. The `append` parameter of
    :func:`snp.diff` is exploited to give output of the same length as
    the input, which is achieved by zero-padding the output at the end
    of each axis.


    """
    if axis is None:
        axis = tuple(range(x.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    dstack = [
        snp.diff(
            x,
            axis=ax,
            append=x[tuple(slice(-1, None) if i == ax else slice(None) for i in range(x.ndim))],
        )
        for ax in axis
    ]
    return snp.stack(dstack)


class ProjectedGradient(LinearOperator):
    """Gradient projected onto local coordinate system.

    This class represents a linear operator that computes gradients of
    arrays projected onto a local coordinate system that may differ at
    every position in the array, as described in
    :cite:`hossein-2024-total`. In the 2D illustration below :math:`x`
    and :math:`y` represent the standard coordinate system defined by the
    array axes, :math:`(g_x, g_y)` is the gradient vector within that
    coordinate system, :math:`x'` and :math:`y'` are the local coordinate
    axes, and :math:`(g_x', g_y')` is the gradient vector within the
    local coordinate system.

    .. image:: /figures/projgrad.svg
         :align: center
         :alt: Figure illustrating projection of gradient onto local
               coordinate system.

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
        axes: Optional[Tuple[int, ...]] = None,
        coord: Optional[Sequence[Union[Array, BlockArray]]] = None,
        cdiff: bool = False,
        input_dtype: DType = np.float32,
        jit: bool = True,
    ):
        r"""

        The result of applying the operator is always a
        :class:`jax.Array`. If `coord` is a singleton tuple, it has the
        same shape as the input array. Otherwise, the gradients for each
        of the local coordinate axes are stacked on an additional axis at
        index 0.

        If `coord` is ``None``, which is the default, gradients are
        computed in the standard axis-aligned coordinate system, and the
        shape of the returned array depends on the number of axes on
        which the gradient is calculated, as specified explicitly or
        implicitly via the `axes` parameter.

        Args:
            input_shape: Shape of input array.
            axes: Axes over which to compute the gradient. Defaults to
                ``None``, in which case the gradient is computed along
                all axes.
            coord: A tuple of arrays, each of which specifies a local
                coordinate axis direction. Each member of the tuple
                should either be a :class:`jax.Array` or a
                :class:`.BlockArray`. If it is the former, it should have
                shape :math:`N \times M_0 \times M_1 \times \ldots`,
                where :math:`N` is the number of axes specified by
                parameter `axes`, and :math:`M_i` is the size of the
                :math:`i^{\mrm{th}}` axis. If it is the latter, it should
                consist of :math:`N` blocks, each of which has a shape
                that is suitable for multiplication with an array of
                shape :math:`M_0 \times M_1 \times \ldots`.
            cdiff: If ``True``, estimate gradients using the second order
                central different returned by :func:`snp.gradient`,
                otherwise use the first order asymmetric difference
                returned by :func:`snp.diff`.
            input_dtype: `dtype` for input argument. Default is
                :attr:`~numpy.float32`.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """
        if axes is None:
            # If axes is None, set it to all axes in input shape.
            self.axes = tuple(range(len(input_shape)))
        else:
            # Ensure no invalid axis indices specified.
            if snp.any(np.array(axes) >= len(input_shape)):
                raise ValueError(
                    "Invalid axes specified; all elements of argument 'axes' must "
                    f"be less than len(input_shape)={len(input_shape)}."
                )
            self.axes = axes
        output_shape: Shape
        if coord is None:
            # If coord is None, output shape is determined by number of axes.
            if len(self.axes) == 1:
                output_shape = input_shape
            else:
                output_shape = (len(self.axes),) + input_shape
        else:
            # If coord is not None, output shape is determined by number of coord arrays.
            if len(coord) == 1:
                output_shape = input_shape
            else:
                output_shape = (len(coord),) + input_shape
        self.coord = coord
        self.cdiff = cdiff
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
            jit=jit,
        )

    def _eval(self, x: Array) -> Union[Array, BlockArray]:

        if self.cdiff:
            grad = snp.stack(snp.gradient(x, axis=self.axes))
        else:
            grad = diffstack(x, axis=self.axes)
        if self.coord is None:
            # If coord attribute is None, just return gradients on specified axes.
            if len(self.axes) == 1:
                return grad[0]
            else:
                return grad
        else:
            # If coord attribute is not None, return gradients projected onto specified local
            # coordinate systems.
            projgrad = [sum([c[m] * grad[m] for m in range(len(self.axes))]) for c in self.coord]
            if len(self.coord) == 1:
                return projgrad[0]
            else:
                return snp.stack(projgrad)


class PolarGradient(ProjectedGradient):
    """Gradient projected into polar coordinates.

    Compute gradients projected onto angular and/or radial axis
    directions, as described in :cite:`hossein-2024-total`. Local
    coordinate axes are illustrated in the figure below.

    .. plot:: pyfigures/polargrad.py
       :align: center
       :include-source: False
       :show-source-link: False

    |

    If only one of `angular` and `radial` is ``True``, the operator
    output has the same shape as the input, otherwise the gradients for
    the two local coordinate axes are stacked on an additional axis at
    index 0.
    """

    def __init__(
        self,
        input_shape: Shape,
        axes: Optional[Tuple[int, ...]] = None,
        center: Optional[Union[Tuple[int, ...], Array]] = None,
        angular: bool = True,
        radial: bool = True,
        cdiff: bool = False,
        input_dtype: DType = np.float32,
        jit: bool = True,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            axes: Axes over which to compute the gradient. Should be a
                tuple :math:`(i_x, i_y)`, where :math:`i_x` and
                :math:`i_y` are input array axes assigned to :math:`x`
                and :math:`y` coordinates respectively. Defaults to
                ``None``, in which case the axes are taken to be `(0, 1)`.
            center: Center of the polar coordinate system in array
                indexing coordinates. Default is ``None``, which places
                the center at the center of the input array.
            angular: Flag indicating whether to compute gradients in the
                angular (i.e. tangent to circles) direction.
            radial: Flag indicating whether to compute gradients in the
                radial (i.e. directed outwards from the origin) direction.
            cdiff: If ``True``, estimate gradients using the second order
                central different returned by :func:`snp.gradient`,
                otherwise use the first order asymmetric difference
                returned by :func:`snp.diff`.
            input_dtype: `dtype` for input argument. Default is
                :attr:`~numpy.float32`.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """

        if len(input_shape) < 2:
            raise ValueError("Invalid input shape; input must have at least two axes.")
        if axes is not None and len(axes) != 2:
            raise ValueError("Invalid axes specified; exactly two axes must be specified.")
        if not angular and not radial:
            raise ValueError("At least one of angular and radial must be True.")

        real_input_dtype = snp.util.real_dtype(input_dtype)
        if axes is None:
            axes = (0, 1)
        axes_shape = [input_shape[ax] for ax in axes]
        if center is None:
            center = (snp.array(axes_shape, dtype=real_input_dtype) - 1) / 2
        else:
            center = snp.array(center, dtype=real_input_dtype)
        end = snp.array(axes_shape, dtype=real_input_dtype) - center
        g0, g1 = snp.ogrid[-center[0] : end[0], -center[1] : end[1]]
        theta = snp.arctan2(g0, g1)
        # Re-order theta axes in case indices in axes parameter are not in increasing order.
        axis_order = np.argsort(axes)
        theta = snp.transpose(theta, axis_order)
        if len(input_shape) > 2:
            # Construct list of input axes that are not included in the gradient axes.
            single = tuple(set(range(len(input_shape))) - set(axes))
            # Insert singleton axes to align theta for multiplication with gradients.
            theta = snp.expand_dims(theta, single)
        coord = []
        if angular:
            coord.append(snp.blockarray([-snp.cos(theta), snp.sin(theta)]))
        if radial:
            coord.append(snp.blockarray([snp.sin(theta), snp.cos(theta)]))
        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            axes=axes,
            coord=coord,
            cdiff=cdiff,
            jit=jit,
        )


class CylindricalGradient(ProjectedGradient):
    """Gradient projected into cylindrical coordinates.

    Compute gradients projected onto cylindrical coordinate axes, as
    described in :cite:`hossein-2024-total`. The local coordinate axes
    are illustrated in the figure below.

    .. plot:: pyfigures/cylindgrad.py
       :align: center
       :include-source: False
       :show-source-link: False

    |

    If only one of `angular`, `radial`, and `axial` is ``True``, the
    operator output has the same shape as the input, otherwise the
    gradients for the selected local coordinate axes are stacked on an
    additional axis at index 0.
    """

    def __init__(
        self,
        input_shape: Shape,
        axes: Optional[Tuple[int, ...]] = None,
        center: Optional[Union[Tuple[int, ...], Array]] = None,
        angular: bool = True,
        radial: bool = True,
        axial: bool = True,
        cdiff: bool = False,
        input_dtype: DType = np.float32,
        jit: bool = True,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            axes: Axes over which to compute the gradient. Should be a
                tuple :math:`(i_x, i_y, i_z)`, where :math:`i_x`,
                :math:`i_y` and :math:`i_z` are input array axes assigned
                to :math:`x`, :math:`y`, and :math:`z` coordinates
                respectively. Defaults to ``None``, in which case the
                axes are taken to be `(0, 1, 2)`. If an integer, this
                operator returns a :class:`jax.Array`. If a tuple or
                ``None``, the resulting arrays are stacked into a
                :class:`.BlockArray`.
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
            cdiff: If ``True``, estimate gradients using the second order
                central different returned by :func:`snp.gradient`,
                otherwise use the first order asymmetric difference
                returned by :func:`snp.diff`.
            input_dtype: `dtype` for input argument. Default is
                :attr:`~numpy.float32`.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """

        if len(input_shape) < 3:
            raise ValueError("Invalid input shape; input must have at least three axes.")
        if axes is not None and len(axes) != 3:
            raise ValueError("Invalid axes specified; exactly three axes must be specified.")
        if not angular and not radial and not axial:
            raise ValueError("At least one of angular, radial, and axial must be True.")

        real_input_dtype = snp.util.real_dtype(input_dtype)
        if axes is None:
            axes = (0, 1, 2)
        axes_shape = [input_shape[ax] for ax in axes]
        if center is None:
            center = (snp.array(axes_shape, dtype=real_input_dtype) - 1) / 2
            center = center.at[-1].set(0)  # type: ignore
        else:
            center = snp.array(center, dtype=real_input_dtype)
        end = snp.array(axes_shape, dtype=real_input_dtype) - center
        g0, g1 = snp.ogrid[-center[0] : end[0], -center[1] : end[1]]
        g0 = g0[..., np.newaxis]
        g1 = g1[..., np.newaxis]
        theta = snp.arctan2(g0, g1)
        # Re-order theta axes in case indices in axes parameter are not in increasing order.
        axis_order = np.argsort(axes)
        theta = snp.transpose(theta, axis_order)
        if len(input_shape) > 3:
            # Construct list of input axes that are not included in the gradient axes.
            single = tuple(set(range(len(input_shape))) - set(axes))
            # Insert singleton axes to align theta for multiplication with gradients.
            theta = snp.expand_dims(theta, single)
        coord = []
        if angular:
            coord.append(
                snp.blockarray(
                    [-snp.cos(theta), snp.sin(theta), snp.array([0.0], dtype=real_input_dtype)]
                )
            )
        if radial:
            coord.append(
                snp.blockarray(
                    [snp.sin(theta), snp.cos(theta), snp.array([0.0], dtype=real_input_dtype)]
                )
            )
        if axial:
            coord.append(
                snp.blockarray(
                    [
                        snp.array([0.0], dtype=real_input_dtype),
                        snp.array([0.0], dtype=real_input_dtype),
                        snp.array([1.0], dtype=real_input_dtype),
                    ]
                )
            )
        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            axes=axes,
            cdiff=cdiff,
            coord=coord,
            jit=jit,
        )


class SphericalGradient(ProjectedGradient):
    """Gradient projected into spherical coordinates.

    Compute gradients projected onto spherical coordinate axes, based on
    the approach described in :cite:`hossein-2024-total`. The local
    coordinate axes are illustrated in the figure below.

    .. plot:: pyfigures/spheregrad.py
       :align: center
       :include-source: False
       :show-source-link: False

    |

    If only one of `azimuthal`, `polar`, and `radial` is ``True``, the
    operator output has the same shape as the input, otherwise the
    gradients for the selected local coordinate axes are stacked on an
    additional axis at index 0.
    """

    def __init__(
        self,
        input_shape: Shape,
        axes: Optional[Tuple[int, ...]] = None,
        center: Optional[Union[Tuple[int, ...], Array]] = None,
        azimuthal: bool = True,
        polar: bool = True,
        radial: bool = True,
        cdiff: bool = False,
        input_dtype: DType = np.float32,
        jit: bool = True,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            axes: Axes over which to compute the gradient. Should be a
                tuple :math:`(i_x, i_y, i_z)`, where :math:`i_x`,
                :math:`i_y` and :math:`i_z` are input array axes assigned
                to :math:`x`, :math:`y`, and :math:`z` coordinates
                respectively. Defaults to ``None``, in which case the
                axes are taken to be `(0, 1, 2)`. If an integer, this
                operator returns a :class:`jax.Array`. If a tuple or
                ``None``, the resulting arrays are stacked into a
                :class:`.BlockArray`.
            center: Center of the spherical coordinate system in array
                indexing coordinates. Default is ``None``, which places
                the center at the center of the input array.
            azimuthal: Flag indicating whether to compute gradients in
                the azimuthal direction.
            polar: Flag indicating whether to compute gradients in the
                polar direction.
            radial: Flag indicating whether to compute gradients in the
                radial direction.
            cdiff: If ``True``, estimate gradients using the second order
                central different returned by :func:`snp.gradient`,
                otherwise use the first order asymmetric difference
                returned by :func:`snp.diff`.
            input_dtype: `dtype` for input argument. Default is
                :attr:`~numpy.float32`.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """

        if len(input_shape) < 3:
            raise ValueError("Invalid input shape; input must have at least three axes.")
        if axes is not None and len(axes) != 3:
            raise ValueError("Invalid axes specified; exactly three axes must be specified.")
        if not azimuthal and not polar and not radial:
            raise ValueError("At least one of azimuthal, polar, and radial must be True.")

        real_input_dtype = snp.util.real_dtype(input_dtype)
        if axes is None:
            axes = (0, 1, 2)
        axes_shape = [input_shape[ax] for ax in axes]
        if center is None:
            center = (snp.array(axes_shape, dtype=real_input_dtype) - 1) / 2
        else:
            center = snp.array(center, dtype=real_input_dtype)
        end = snp.array(axes_shape, dtype=real_input_dtype) - center
        g0, g1, g2 = snp.ogrid[-center[0] : end[0], -center[1] : end[1], -center[2] : end[2]]
        theta = snp.arctan2(g1, g0)
        phi = snp.arctan2(snp.sqrt(g0**2 + g1**2), g2)
        # Re-order theta and phi axes in case indices in axes parameter are not in
        # increasing order.
        axis_order = np.argsort(axes)
        theta = snp.transpose(theta, axis_order)
        phi = snp.transpose(phi, axis_order)
        if len(input_shape) > 3:
            # Construct list of input axes that are not included in the gradient axes.
            single = tuple(set(range(len(input_shape))) - set(axes))
            # Insert singleton axes to align theta for multiplication with gradients.
            theta = snp.expand_dims(theta, single)
            phi = snp.expand_dims(phi, single)
        coord = []
        if azimuthal:
            coord.append(
                snp.blockarray(
                    [snp.sin(theta), -snp.cos(theta), snp.array([0.0], dtype=real_input_dtype)]
                )
            )
        if polar:
            coord.append(
                snp.blockarray(
                    [snp.cos(phi) * snp.cos(theta), snp.cos(phi) * snp.sin(theta), -snp.sin(phi)]
                )
            )
        if radial:
            coord.append(
                snp.blockarray(
                    [snp.sin(phi) * snp.cos(theta), snp.sin(phi) * snp.sin(theta), snp.cos(phi)]
                )
            )
        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            axes=axes,
            coord=coord,
            cdiff=cdiff,
            jit=jit,
        )
