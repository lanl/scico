# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

r"""Optical propagator classes.

This module provides classes that model the propagation of a
monochromatic waveform between two parallel planes in a homogeneous
medium. The corresponding linear operators are referred to here as
"propagators", which represents a departure from standard terminology,
in which "propagator" refers specifically to the Fourier domain
component of the linear operator, i.e. if the full linear operator
can be written as :math:`F^{-1} D F` where :math:`F` is the Fourier
transform, then :math:`D` is usually referred to as the propagator.


The following notation is used throughout the module:

.. math ::
     \begin{align}
     \Delta x, \Delta y  & \quad \text{Sampling intervals in } x
     \text{ and } y \text{ axes}\\
     z  & \quad \text{Propagation distance} \;\; (z \geq 0) \\
     N_x, N_y  & \quad \text{Number of samples in } x \text{ and } y
     \text{ axes}\\
     k_0 & \quad \text{Illumination wavenumber corresponding to } 2\pi /
     \text{wavelength} \;.
     \end{align}

Variables :math:`\Delta x, \Delta y, z,` and :math:`k_0` represent
physical quantities. Any units may be chosen, but they must be consistent
across all of these variables, e.g. m (metres) for :math:`\Delta x,
\Delta y, z,` and :math:`\mathrm{m}^{-1}` for :math:`k_0`, as well as with
the units for the physical dimensions of the source wavefield.

Subscripts :math:`S` and :math:`D` are used to refer to the source and
destination planes respectively when it is necessary to distinguish
between them. In the absence of subscripts, the variables refer to the
source plane (e.g. both :math:`\Delta x` and :math:`\Delta x_S` refer to
the :math:`x`-axis sampling interval in the source plane, while
:math:`\Delta x_D` refers to it in the destination plane).

Note that :math:`x` corresponds to axis 0 (rows, increasing downwards)
and :math:`y` to axis 1 (columns, increasing to the right).
"""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
from numpy.lib.scimath import sqrt  # complex sqrt

import jax

from typing_extensions import TypeGuard

import scico.numpy as snp
from scico.linop import Diagonal, Identity, LinearOperator
from scico.numpy.util import no_nan_divide
from scico.typing import Shape

from ._dft import DFT


def _isscalar(element: Any) -> TypeGuard[Union[int, float]]:
    """Type guard interface to `snp.isscalar`."""
    return snp.isscalar(element)


def radial_transverse_frequency(
    input_shape: Shape, dx: Union[float, Tuple[float, ...]]
) -> np.ndarray:
    r"""Construct radial Fourier coordinate system.

    Args:
        input_shape: Tuple of length 1 or 2 containing the number of
            samples per dimension, i.e. :math:`(N_x,)` or
            :math:`(N_x, N_y)`
        dx: Sampling interval at source plane. If a float and
            `len(input_shape)==2` the same sampling interval is applied
            to both dimensions. If `dx` is a tuple, it must have same
            length as `input_shape`, and corresponds to either
            :math:`(\Delta x,)` or :math:`(\Delta x, \Delta y)`.

    Returns:
        If `len(input_shape)==1`, returns an ndarray containing
        corresponding Fourier coordinates. If `len(input_shape) == 2`,
        returns an ndarray containing the radial Fourier coordinates
        :math:`\sqrt{k_x^2 + k_y^2}\,`.
    """

    ndim: int = len(input_shape)  # 1 or 2 dimensions
    if ndim not in (1, 2):
        raise ValueError("Invalid input dimensions; must be 1 or 2")

    if _isscalar(dx):
        dx = (dx,) * ndim
    else:
        assert isinstance(dx, tuple)
        if len(dx) != ndim:
            raise ValueError(
                "dx must be a scalar or have len(dx) == len(input_shape); "
                f"got len(dx)={len(dx)}, len(input_shape)={ndim}"
            )
    assert isinstance(dx, tuple)

    if ndim == 1:
        kx = 2 * np.pi * np.fft.fftfreq(input_shape[0], dx[0])
        kp = kx
    elif ndim == 2:
        kx = 2 * np.pi * np.fft.fftfreq(input_shape[0], dx[0])
        ky = 2 * np.pi * np.fft.fftfreq(input_shape[1], dx[1])
        kp = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    return kp


class Propagator(LinearOperator):
    """Base class for angular spectrum and Fresnel propagators."""

    def __init__(
        self,
        input_shape: Shape,
        dx: Union[float, Tuple[float, ...]],
        k0: float,
        z: float,
        pad_factor: int = 1,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array as a tuple of length
               1 or 2, corresponding to :math:`(N_x,)` or
               :math:`(N_x, N_y)`.
            dx: Sampling interval at source plane. If a float and
               `len(input_shape)==2` the same sampling interval is applied
               to both dimensions. If `dx` is a tuple, it must have same
               length as `input_shape`, and corresponds to either
               :math:`(\Delta x,)` or :math:`(\Delta x, \Delta y)`.
            k0: Illumination wavenumber, :math:`k_0`, corresponding to
               :math:`2 \pi` / wavelength.
            z: Propagation distance, :math:`z`.
            pad_factor: The padded input shape is the input shape
               multiplied by this integer factor.
        """

        ndim = len(input_shape)  # 1 or 2 dimensions
        if ndim not in (1, 2):
            raise ValueError("Invalid input dimensions; must be 1 or 2")

        if _isscalar(dx):
            dx = (dx,) * ndim
        else:
            assert isinstance(dx, tuple)
            if len(dx) != ndim:
                raise ValueError(
                    "dx must be a scalar or have len(dx) == len(input_shape); "
                    f"got len(dx)={len(dx)}, len(input_shape)={ndim}"
                )
        assert isinstance(dx, tuple)

        #: Illumination wavenumber; 2𝜋/wavelength
        self.k0: float = k0
        #: Shape of input after padding
        self.padded_shape: Shape = tuple(pad_factor * s for s in input_shape)
        #: Padded source plane side length (dx[i] * padded_shape[i])
        self.L: Tuple[float, ...] = tuple(
            s * d for s, d in zip(self.padded_shape, dx)
        )  # computational plane size
        #: Transverse Fourier coordinates (radial)
        self.kp = radial_transverse_frequency(self.padded_shape, dx)
        #: Source plane sampling interval
        self.dx: Union[float, Tuple[float, ...]] = dx
        #: Propagation distance
        self.z: float = z

        # Fourier operator
        self.F = DFT(input_shape=input_shape, axes_shape=self.padded_shape, jit=False)

        # Diagonal operator; phase shifting
        self.D: LinearOperator = Identity(self.kp.shape)

        super().__init__(
            input_shape=input_shape,
            input_dtype=np.complex64,
            output_shape=input_shape,
            output_dtype=np.complex64,
            adj_fn=None,
            **kwargs,
        )

    def __repr__(self):
        extra_repr = f"""
k0          : {self.k0}
λ           : {2*np.pi/self.k0}
z           : {self.z}
dx          : {self.dx}
L           : {self.L}
        """
        return LinearOperator.__repr__(self) + extra_repr

    def _eval(self, x):
        return self.F.inv(self.D @ self.F @ x)


class AngularSpectrumPropagator(Propagator):
    r"""Angular spectrum propagator.

    Propagates a planar source field with coordinates :math:`(x, y, z_0)`
    to a destination plane at a distance :math:`z` with coordinates
    :math:`(x, y, z_0 + z)`. The action of this linear operator is
    given by (Eq. 3.74, :cite:`goodman-2005-fourier`)

    .. math ::
         (A \mb{u})(x, y, z_0 + z) = \frac{1}{2 \pi} \iint_{-\infty}^{\infty}
         \mb{\hat{u}}(k_x, k_y) e^{j \sqrt{k_0^2 - k_x^2 - k_y^2} \,
         z} e^{j (x k_x + y k_y) } d k_x \ d k_y \;,

    where the :math:`\mb{\hat{u}}` is the Fourier transform of the
    field :math:`\mb{u}(x, y)` in the plane :math:`z=z_0`, given by

    .. math ::
         \mb{\hat{u}}(k_x, k_y) = \iint_{-\infty}^{\infty}
         \mb{u}(x, y) e^{- j (x k_x + y k_y)} d k_x \ d k_y \;,

    where :math:`(k_x, k_y)` are the :math:`x` and :math:`y` components
    respectively of the wave-vector of the plane wave, and :math:`j` is
    the imaginary unit.

    The angular spectrum propagator can be written

    .. math ::
         A\mb{u} = F^{-1} D F \mb{u} \;,

    where :math:`F` is the Fourier transform with respect to
    :math:`(x, y)`, :math:`F^{-1}` is the inverse transform with respect
    to :math:`(k_x, k_y)`, and the propagator term is given by

    .. math ::
         D = \exp \left( j \sqrt{k_0^2 - k_x^2 - k_y^2} \, z \right) \;.

    Aliasing of the wavefield at the destination plane is avoided when
    the propagator term is adequately sampled according to
    :cite:`voelz-2009-digital`

    .. math ::
         (\Delta x)^2 \geq \frac{\pi}{k_0 N_x} \sqrt{ (\Delta x)^2 N_x^2 +
         4 z^2} \quad \text{and} \quad
         (\Delta y)^2 \geq \frac{\pi}{k_0 N_y} \sqrt{ (\Delta y)^2 N_y^2 +
         4 z^2} \;.
    """

    def __init__(
        self,
        input_shape: Shape,
        dx: Union[float, Tuple[float, ...]],
        k0: float,
        z: float,
        pad_factor: int = 1,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array. Can be a tuple of length
               2 or 3.
            dx: Sampling interval, :math:`\Delta x`, at source plane. If
               a float and `len(input_shape)==2` the same sampling
               interval is applied to both dimensions. If `dx` is a tuple,
               must have same length as `input_shape`.
            k0: Illumination wavenumber, :math:`k_0`, corresponding to
               :math:`2 \pi` / wavelength.
            z: Propagation distance, :math:`z`.
            pad_factor: The padded input shape is the input shape
               multiplied by this integer factor.
            jit: If ``True``, call :meth:`~.Operator.jit` on this
               :class:`LinearOperator` to jit the forward, adjoint, and
               gram functions. Same as calling :meth:`~.Operator.jit`
               after the :class:`LinearOperator` is created.
        """

        # Diagonal operator; phase shifting
        super().__init__(
            input_shape=input_shape, dx=dx, k0=k0, z=z, pad_factor=pad_factor, **kwargs
        )

        self.phase = jax.device_put(
            np.exp(1j * z * sqrt(self.k0**2 - self.kp**2)).astype(np.complex64)
        )
        self.D = Diagonal(self.phase)
        self._set_adjoint()

        if jit:
            self.jit()

    def adequate_sampling(self):
        r"""Verify the angular spectrum kernel is not aliased.

        Checks the condition for adequate sampling
        :cite:`voelz-2009-digital`,

         .. math ::
             (\Delta x)^2 \geq \frac{\pi}{k_0 N_x} \sqrt{ (\Delta x)^2 N_x^2 +
             4 z^2} \quad \text{and} \quad
             (\Delta y)^2 \geq \frac{\pi}{k_0 N_y} \sqrt{ (\Delta y)^2 N_y^2 +
             4 z^2} \;.

        Returns:
             ``True`` if the angular spectrum kernel is adequately sampled,
             ``False`` otherwise.
        """
        tmp = []
        for d, N in zip(self.dx, self.padded_shape):
            tmp.append(d**2 > np.pi / (self.k0 * N) * np.sqrt(d**2 * N**2 + 4 * self.z**2))
        return np.all(tmp)

    def pinv(self, y):
        """Apply pseudoinverse of Angular Spectrum propagator."""
        diag_inv = no_nan_divide(1, self.D.diagonal)
        return self.F.inv(diag_inv * self.F(y))


class FresnelPropagator(Propagator):
    r"""Fresnel (small-angle/paraxial) propagator.

    Propagates a planar source field with coordinates :math:`(x, y, z_0)`
    to a destination plane at a distance :math:`z` with coordinates
    :math:`(x, y, z_0 + z)`. The action of this linear operator is given
    by (Eq. 4.20, :cite:`goodman-2005-fourier`)

    .. math ::
        (A \mb{u})(x, y, z + z_0) = e^{j k_0 z} \frac{1}{2 \pi}
        \iint_{-\infty}^{\infty} \mb{\hat{u}}(k_x, k_y)
        e^{-j \frac{z}{2 k_0}\left(k_x^2 + k_y^2\right) }
        e^{j (x k_x + y k_y) } d k_x \ d k_y \;,

    where the :math:`\mb{\hat{u}}` is the Fourier transform of the field
    in the source plane, given by

    .. math ::
        \mb{\hat{u}}(k_x, k_y) = \iint_{-\infty}^{\infty} \mb{u}(x, y)
        e^{- j (x k_x + y k_y)} d k_x \ d k_y \;.

    This linear operator is valid when :math:`k_0^2 << k_x^2 + k_y^2`.
    The Fresnel propagator can be written

    .. math ::
        A\mb{u} = F^{-1} D F \mb{u} \;,

    where :math:`F` is the Fourier transform with respect to
    :math:`(x, y)`, :math:`F^{-1}` is the inverse transform with respect
    to :math:`(k_x, k_y)`, and the propagator term is given by

    .. math ::
        D = \exp \left( -j \frac{z}{2 k_0}\left(k_x^2 + k_y^2 \right)
        \right) \;,

    where :math:`(k_x, k_y)` are the :math:`x` and :math:`y` components
    respectively of the wave-vector of the plane wave, and :math:`j` is
    the imaginary unit.

    The propagator term is adequately sampled when
    :cite:`voelz-2011-computational`

    .. math ::
         (\Delta x)^2 \geq \frac{2 \pi z }{k_0 N_x} \quad \text{and}
         \quad (\Delta y)^2 \geq \frac{2 \pi z }{k_0 N_y} \;.
    """

    def __init__(
        self,
        input_shape: Shape,
        dx: float,
        k0: float,
        z: float,
        pad_factor: int = 1,
        jit: bool = True,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape, dx=dx, k0=k0, z=z, pad_factor=pad_factor, **kwargs
        )

        self.phase = jax.device_put(
            np.exp(1j * z * (self.k0 - self.kp**2 / (2 * self.k0))).astype(np.complex64)
        )
        self.D = Diagonal(self.phase)

        self._set_adjoint()

        if jit:
            self.jit()

    def adequate_sampling(self):
        r"""Verify the Fresnel propagation kernel is not aliased.

        Checks the condition for adequate sampling
        :cite:`voelz-2011-computational`,

        .. math ::
            (\Delta x)^2 \geq \frac{2 \pi z }{k_0 N_x} \quad \text{and}
            \quad (\Delta y)^2 \geq \frac{2 \pi z }{k_0 N_y} \;.


        Returns:
            ``True`` if the Fresnel propagation kernel is adequately sampled,
            ``False`` otherwise.
        """
        tmp = []
        for d, N in zip(self.dx, self.padded_shape):
            tmp.append(d**2 > 2 * np.pi * self.z / (self.k0 * N))
        return np.all(tmp)


class FraunhoferPropagator(LinearOperator):
    r"""Fraunhofer (far-field) propagator.

    Propagates a source field with coordinates :math:`(x_S, y_S)` to
    a destination plane at a distance :math:`z` with coordinates
    :math:`(x_D, y_D)`.

    The action of this linear operator is given by (Eq. 4.25,
    :cite:`goodman-2005-fourier`)

    .. math ::
        (A \mb{u})(x_D, y_D) = \underbrace{\frac{k_0}{2 \pi}
        \frac{e^{j k_0 z}}{j z} \mathrm{exp} \left( j \frac{k_0}{2 z}
        (x_D^2 + y_D^2) \right)}_{\triangleq P(x_D, y_D)}
        \int \mb{u}(x_S, y_S) e^{-j \frac{k_0}{z} (x_D x_S + y_D y_S)
        } dx_S \ dy_S \;.

    This is valid when :math:`N_F << 1`, where :math:`N_F` is the
    Fresnel number (Sec. 1.5, Sec. 4.7.2.1) :cite:`paganin-2006-coherent`.
    Writing the Fourier transform of the field :math:`\mb{u}` as

    .. math ::
        \hat{\mb{u}}(k_x, k_y) = \int e^{-j (k_x x + k_y y)}
        \mb{u}(x, y) dx \ dy \;,

    the action of this linear operator can be written

    .. math ::
        (A \mb{u})(x_D, y_D) = P(x_D, y_D) \ \hat{\mb{u}}
        \left({\frac{k_0}{z} x_D, \frac{k_0}{z} y_D}\right) \;.

    Ignoring multiplicative prefactors, the Fraunhofer propagated
    field is the Fourier transform of the source field, evaluated at
    coordinates :math:`(k_x, k_y) = (\frac{k_0}{z} x_D,
    \frac{k_0}{z} y_D)`.

    In general, the sampling intervals (and thus plane lengths)
    differ between source and destination planes. In particular,
    (Eq. 5.18, :cite:`voelz-2011-computational`)

    .. math ::
        \Delta x_D =  \frac{2 \pi z}{k_0 L_{Sx} } \quad \text{and}
        \quad L_{Dx} =  \frac{2 \pi z}{k_0 \Delta x_S } \;,

    and similarly for the :math:`y` axis.

    The Fraunhofer propagator term :math:`P(x_D, y_D)` is adequately
    sampled when

    .. math ::
         \Delta x_S \geq \sqrt{\frac{2 \pi z}{N_x k_0}} \quad \text{and}
         \quad \Delta y_S \geq \sqrt{\frac{2 \pi z}{N_y k_0}} \;.
    """

    def __init__(
        self,
        input_shape: Shape,
        dx: Union[float, Tuple[float, ...]],
        k0: float,
        z: float,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array as a tuple of length
               1 or 2, corresponding to :math:`(N_x,)` or
               :math:`(N_x, N_y)`.
            dx: Sampling interval at source plane. If a float and
               `len(input_shape)==2` the same sampling interval is applied
               to both dimensions. If `dx` is a tuple, it must have same
               length as `input_shape`, and corresponds to either
               :math:`(\Delta x,)` or :math:`(\Delta x, \Delta y)`.
            k0: Illumination wavenumber, :math:`k_0`, corresponding to
               :math:`2 \pi` / wavelength.
            z: Propagation distance, :math:`z`.
            jit: If ``True``, jit the evaluation, adjoint, and gram
               functions of this :class:`LinearOperator`. Default:
               ``True``.
        """

        ndim = len(input_shape)  # 1 or 2 dimensions
        if ndim not in (1, 2):
            raise ValueError("Invalid input dimensions; must be 1 or 2")

        if _isscalar(dx):
            dx = (dx,) * ndim
        else:
            assert isinstance(dx, tuple)
            if len(dx) != ndim:
                raise ValueError(
                    "dx must be a scalar or have len(dx) == len(input_shape); "
                    f"got len(dx)={len(dx)}, len(input_shape)={ndim}"
                )
        assert isinstance(dx, tuple)

        L: Tuple[float, ...] = tuple(s * d for s, d in zip(input_shape, dx))

        #: Illumination wavenumber
        self.k0: float = k0
        #: Propagation distance
        self.z: float = z
        #: Source plane side length (dx[i] * input_shape[i])
        self.L: Tuple[float, ...] = L
        #: Source plane sampling interval
        self.dx: Tuple[float, ...] = dx

        #: Destination plane sampling interval
        self.dx_D: Tuple[float, ...] = tuple(np.abs(2 * np.pi * z / (k0 * l)) for l in L)
        #: Destination plane side length
        self.L_D: Tuple[float, ...] = tuple(np.abs(2 * np.pi * z / (k0 * d)) for d in dx)
        x_D = tuple(np.r_[-l / 2 : l / 2 : d] for l, d in zip(self.L_D, self.dx_D))  # type: ignore

        # set up radial coordinate system; either x^2 or (x^2 + y^2)
        if ndim == 1:
            self.r2 = x_D[0]
        elif ndim == 2:
            self.r2 = np.sqrt(x_D[0][:, None] ** 2 + x_D[1][None, :] ** 2)

        phase = -1j * snp.exp(1j * k0 * z) * snp.exp(1j * 0.5 * k0 / z * self.r2**2)
        phase *= k0 / (2 * np.pi) * np.abs(1 / z)
        phase *= np.prod(dx)  # from approximating continouous FT with DFT
        phase = phase.astype(np.complex64)

        self.F = DFT(input_shape=input_shape, jit=False)
        self.D = Diagonal(phase)
        super().__init__(
            input_shape=input_shape,
            input_dtype=np.complex64,
            output_shape=input_shape,
            output_dtype=np.complex64,
            **kwargs,
        )

        if jit:
            self.jit()

    def __repr__(self):
        extra_repr = f"""
k0          : {self.k0}
λ           : {2*np.pi/self.k0}
z           : {self.z}
dx          : {self.dx}
L           : {self.L}
dx_D        : {self.dx_D}
L_D         : {self.L_D}
        """
        return LinearOperator.__repr__(self) + extra_repr

    def _eval(self, x):
        x = snp.fft.fftshift(x)
        y = self.D @ self.F @ x
        y = snp.fft.ifftshift(y)
        return y

    def adequate_sampling(self):
        r"""Verify the Fraunhofer propagation kernel is not aliased.

        Checks the condition for adequate sampling
        :cite:`voelz-2011-computational`,

        .. math ::
            \Delta x_S \geq \sqrt{\frac{2 \pi z}{N_x k_0}} \quad \text{and}
            \quad \Delta y_S \geq \sqrt{\frac{2 \pi z}{N_y k_0}} \;.

        Returns:
             ``True`` if the Fraunhofer propagation kernel is adequately
             sampled, ``False`` otherwise.
        """
        tmp = []
        for d, N in zip(self.dx, self.input_shape):
            tmp.append(d**2 > 2 * np.pi * self.z / (self.k0 * N))
        return np.all(tmp)
