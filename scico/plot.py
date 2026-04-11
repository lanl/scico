# -*- coding: utf-8 -*-
# Copyright (C) 2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Plotting/visualization functions.

Optional alternative high-level interface to selected :mod:`matplotlib`
plotting functions.
"""

import warnings

import komplot as kplt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, gca, gcf, savefig, subplot, subplots  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa

__all__ = [
    "plot",
    "surf",
    "contour",
    "imview",
    "close",
    "set_ipython_plot_backend",
    "set_notebook_plot_backend",
    "config_notebook_plotting",
]


def plot(y, x=None, ptyp="plot", xlbl=None, ylbl=None, title=None, lgnd=None, lglc=None, **kwargs):
    """Plot points or lines in 2D.

    Plot points or lines in 2D. If a figure object is specified then the
    plot is drawn in that figure, and `fig.show()` is not called. The
    figure is closed on key entry 'q'.

    Args:
        y (array_like): 1d or 2d array of data to plot. If a 2d array,
            each column is plotted as a separate curve.
        x (array_like, optional (default ``None``)): Values for x-axis of
            the plot.
        ptyp (string, optional (default 'plot')): Plot type specification
            (options are 'plot', 'semilogx', 'semilogy', and 'loglog').
        xlbl (string, optional (default ``None``)): Label for x-axis.
        ylbl (string, optional (default ``None``)): Label for y-axis.
        title (string, optional (default ``None``)): Figure title.
        lgnd (list of strings, optional (default ``None``)): List of
            legend string.
        lglc (string, optional (default ``None``)): Legend location string.
        **kwargs: :class:`matplotlib.lines.Line2D` properties or figure
            properties.

            Keyword arguments specifying :class:`matplotlib.lines.Line2D`
            properties, e.g. `lw=2.0` sets a line width of 2, or
            properties of the figure and axes. If not specified, the
            defaults for line width (`lw`) and marker size (`ms`) are
            1.5 and 6.0 respectively. The valid figure and axes keyword
            arguments are listed below:

            .. |mplfg| replace:: :class:`matplotlib.figure.Figure` object
            .. |mplax| replace:: :class:`matplotlib.axes.Axes` object

            .. rst-class:: kwargs

            =====  ==================== ===================================
            kwarg  Accepts              Description
            =====  ==================== ===================================
            fgsz   tuple (width,height) Specify figure dimensions in inches
            fgnm   integer              Figure number of figure
            fig    |mplfg|              Draw in specified figure instead of
                                        creating one
            ax     |mplax|              Plot in specified axes instead of
                                        current axes of figure
            =====  ==================== ===================================

    Returns:
        - **fig** (:class:`matplotlib.figure.Figure` object):
          Figure object for this figure.
        - **ax** (:class:`matplotlib.axes.Axes` object):
          Axes object for this plot.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    ptyp_options = {
        "plot": (False, False),
        "semilogx": (True, False),
        "semilogy": (False, True),
        "loglog": (True, True),
    }
    xlog, ylog = ptyp_options[ptyp]
    fgsz = kwargs.pop("fgsz", None)
    fgnm = kwargs.pop("fgnm", None)
    fig = kwargs.pop("fig", None)
    ax = kwargs.pop("ax", None)
    if x is None:
        lplt = kplt.plot(
            y,
            xlog=xlog,
            ylog=ylog,
            xlabel=xlbl,
            ylabel=ylbl,
            title=title,
            legend=lgnd,
            legend_loc=lglc,
            figsize=fgsz,
            fignum=fgnm,
            ax=ax,
            **kwargs,
        )
    else:
        lplt = kplt.plot(
            x,
            y,
            xlog=xlog,
            ylog=ylog,
            xlabel=xlbl,
            ylabel=ylbl,
            title=title,
            legend=lgnd,
            legend_loc=lglc,
            figsize=fgsz,
            fignum=fgnm,
            ax=ax,
            **kwargs,
        )
    return lplt.figure, lplt.axes


def surf(
    z,
    x=None,
    y=None,
    elev=None,
    azim=None,
    xlbl=None,
    ylbl=None,
    zlbl=None,
    title=None,
    lblpad=8.0,
    alpha=1.0,
    cntr=None,
    cmap=None,
    fgsz=None,
    fgnm=None,
    fig=None,
    ax=None,
):
    """Plot a 2D surface in 3D.

    Plot a 2D surface in 3D. If a figure object is specified then the
    surface is drawn in that figure, and `fig.show()` is not called.
    The figure is closed on key entry 'q'.

    Args:
        z (array_like): 2d array of data to plot.
        x (array_like, optional (default ``None``)): Values for x-axis of
            the plot.
        y (array_like, optional (default ``None``)): Values for y-axis of
            the plot.
        elev (float): Elevation angle (in degrees) in the z plane.
        azim (float): Azimuth angle  (in degrees) in the x,y plane.
        xlbl (string, optional (default ``None``)): Label for x-axis.
        ylbl (string, optional (default ``None``)): Label for y-axis.
        zlbl (string, optional (default ``None``)): Label for z-axis.
        title (string, optional (default ``None``)): Figure title.
        lblpad (float, optional (default 8.0)): Label padding.
        alpha (float between 0.0 and 1.0, optional (default 1.0)):
            Transparency.
        cntr (int or sequence of ints, optional (default ``None``)): If
            not ``None``, plot contours of the surface on the lower end
            of the z-axis. An int specifies the number of contours to
            plot, and a sequence specifies the specific contour levels to
            plot.
        cmap (:class:`matplotlib.colors.Colormap` object, optional (default ``None``)):
            Color map for surface. If none specifed, defaults to `cm.YlOrRd`.
        fgsz (tuple (width,height), optional (default ``None``)): Specify
            figure dimensions in inches.
        fgnm (integer, optional (default ``None``)): Figure number of figure.
        fig (:class:`matplotlib.figure.Figure` object, optional (default ``None``)):
            Draw in specified figure instead of creating one.
        ax (:class:`matplotlib.axes.Axes` object, optional (default ``None``)):
            Plot in specified axes instead of creating one.

    Returns:
        - **fig** (:class:`matplotlib.figure.Figure` object):
          Figure object for this figure.
        - **ax** (:class:`matplotlib.axes.Axes` object):
          Axes object for this plot.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    splt = kplt.surface(
        z,
        x=x,
        y=y,
        elev=elev,
        azim=azim,
        alpha=alpha,
        cmap=cmap,
        levels=cntr,
        xlabel=xlbl,
        ylabel=ylbl,
        zlabel=zlbl,
        labelpad=lblpad,
        title=title,
        figsize=fgsz,
        fignum=fgnm,
        ax=ax,
    )
    return splt.figure, splt.axes


def contour(
    z,
    x=None,
    y=None,
    v=5,
    xlog=False,
    ylog=False,
    xlbl=None,
    ylbl=None,
    title=None,
    cfmt=None,
    cfntsz=10,
    lfntsz=None,
    alpha=1.0,
    cmap=None,
    vmin=None,
    vmax=None,
    fgsz=None,
    fgnm=None,
    fig=None,
    ax=None,
):
    """Contour plot of a 2D surface.

    Contour plot of a 2D surface. If a figure object is specified then
    the plot is drawn in that figure, and `fig.show()` is not called.
    The figure is closed on key entry 'q'.

    Args:
        z (array_like): 2d array of data to plot.
        x (array_like, optional (default ``None``)): Values for x-axis of
            the plot.
        y (array_like, optional (default ``None``)): Values for y-axis of
            the plot.
        v (int or sequence of floats, optional (default 5)): An int
            specifies the number of contours to plot, and a sequence
            specifies the specific contour levels to plot.
        xlog (boolean, optional (default ``False``)): Set x-axis to log
            scale.
        ylog (boolean, optional (default ``False``)): Set y-axis to log
            scale.
        xlbl (string, optional (default ``None``)): Label for x-axis.
        ylbl (string, optional (default ``None``)): Label for y-axis.
        title (string, optional (default ``None``)): Figure title.
        cfmt (string, optional (default ``None``)): Format string for
            contour labels.
        cfntsz (int or ``None``, optional (default 10)): Contour label
            font size. No contour labels are displayed if set to 0 or
            ``None``.
        lfntsz (int, optional (default ``None``)): Axis label font size.
            The default font size is used if set to ``None``.
        alpha (float, optional (default 1.0)): Underlying image display
            alpha value.
        cmap (:class:`matplotlib.colors.Colormap`, optional (default ``None``)):
            Color map for surface. If none specifed, defaults to `cm.YlOrRd`.
        vmin, vmax (float, optional (default ``None``)): Set upper and
            lower bounds for the color map (see the corresponding
            parameters of :meth:`matplotlib.axes.Axes.imshow`).
        fgsz (tuple (width,height), optional (default ``None``)): Specify
            figure dimensions in inches.
        fgnm (integer, optional (default ``None``)): Figure number of figure.
        fig (:class:`matplotlib.figure.Figure` object, optional (default ``None``)):
            Draw in specified figure instead of creating one.
        ax (:class:`matplotlib.axes.Axes` object, optional (default ``None``)):
            Plot in specified axes instead of current axes of figure.

    Returns:
        - **fig** (:class:`matplotlib.figure.Figure` object):
          Figure object for this figure.
        - **ax** (:class:`matplotlib.axes.Axes` object):
          Axes object for this plot.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    cplt = kplt.contour(
        z,
        x=x,
        y=y,
        xlog=xlog,
        ylog=ylog,
        levels=v,
        clabel_format=cfmt,
        clabel_fontsize=cfntsz,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        xlabel=xlbl,
        ylabel=ylbl,
        xylabel_fontsize=lfntsz,
        title=title,
        figsize=fgsz,
        fignum=fgnm,
        ax=ax,
    )
    return cplt.figure, cplt.axes


def imview(
    img,
    title=None,
    copy=True,
    fltscl=False,
    intrp="nearest",
    norm=None,
    cbar=False,
    cmap=None,
    fgsz=None,
    fgnm=None,
    fig=None,
    ax=None,
):
    """Display an image.

    Display an image. Pixel values are displayed when the pointer is over
    valid image data. If a figure object is specified then the image is
    drawn in that figure, and `fig.show()` is not called. The figure is
    closed on key entry 'q'.

    Args:
        img (array_like, shape (Nr, Nc) or (Nr, Nc, 3) or (Nr, Nc, 4)):
            Image to display.
        title (string, optional (default ``None``)): Figure title.
        copy (boolean, optional (default ``True``)): If ``True``, create
            a copy of input `img` as a reference for displayed pixel
            values, ensuring that displayed values do not change when the
            array changes in the calling scope. Set this flag to
            ``False`` if the overhead of an additional copy of the input
            image is not acceptable.
        fltscl (boolean, optional (default ``False``)): If ``True``,
            rescale and shift floating point arrays to [0,1].
        intrp (string, optional (default 'nearest')): Specify type of
            interpolation used to display image (see `interpolation`
            parameter of :meth:`matplotlib.axes.Axes.imshow`).
        norm (:class:`matplotlib.colors.Normalize` object, optional (default ``None``)):
            Specify the :class:`matplotlib.colors.Normalize` instance used
            to scale pixel values for input to the color map.
        cbar (boolean, optional (default ``False``)): Flag indicating
            whether to display colorbar.
        cmap (:class:`matplotlib.colors.Colormap`, optional (default ``None``)):
            Color map for image. If none specifed, defaults to
            `cm.Greys_r` for monochrome image.
        fgsz (tuple (width,height), optional (default ``None``)): Specify
            figure dimensions in inches.
        fgnm (integer, optional (default ``None``)): Figure number of
            figure.
        fig (:class:`matplotlib.figure.Figure` object, optional (default ``None``)):
            Draw in specified figure instead of creating one.
        ax (:class:`matplotlib.axes.Axes` object, optional (default ``None``)):
            Plot in specified axes instead of current axes of figure.

    Returns:
        - **fig** (:class:`matplotlib.figure.Figure` object):
          Figure object for this figure.
        - **ax** (:class:`matplotlib.axes.Axes` object):
          Axes object for this plot.

    Raises:
        ValueError: If the input array is not of the required shape.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    imv = kplt.imview(
        img,
        interpolation=intrp,
        norm=norm,
        show_cbar=cbar,
        cmap=cmap,
        title=title,
        figsize=fgsz,
        fignum=fgnm,
        ax=ax,
    )
    return imv.figure, imv.axes


def close(fig=None):
    """Close figure(s).

    Close figure(s). If a figure object reference or figure number is
    provided, close the specified figure, otherwise close all figures.

    Args:
        fig (:class:`matplotlib.figure.Figure` object or integer (optional (default ``None``)):
          Figure object or number of figure to close.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    if fig is None:
        plt.close("all")
    else:
        plt.close(fig)


def _in_ipython():
    """Determine whether code is running in an ipython shell.

    Returns:
        bool: ``True`` if running in an ipython shell, ``False``
           otherwise.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        # See https://stackoverflow.com/questions/15411967
        shell = get_ipython().__class__.__name__
        return bool(shell == "TerminalInteractiveShell")
    except NameError:
        return False


def _in_notebook():
    """Determine whether code is running in a Jupyter Notebook shell.

    Returns:
        bool: ``True`` if running in a notebook shell, ``False``
           otherwise.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        # See https://stackoverflow.com/questions/15411967
        shell = get_ipython().__class__.__name__
        return bool(shell == "ZMQInteractiveShell")
    except NameError:
        return False


def set_ipython_plot_backend(backend="qt"):
    """Set matplotlib backend within an ipython shell.

    Set matplotlib backend within an ipython shell. This function has the
    same effect as the line magic `%matplotlib [backend]` but is called
    as a function and includes a check to determine whether the code is
    running in an ipython shell, so that it can safely be used within a
    normal python script since it has no effect when not running in an
    ipython shell.

    Args:
        backend (string, optional (default 'qt')): Name of backend to be
            passed to the `%matplotlib` line magic command.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    if _in_ipython():
        # See https://stackoverflow.com/questions/35595766
        get_ipython().run_line_magic("matplotlib", backend)


def set_notebook_plot_backend(backend="inline"):
    """Set matplotlib backend within a Jupyter Notebook shell.

    Set matplotlib backend within a Jupyter Notebook shell. This function
    has the same effect as the line magic `%matplotlib [backend]` but
    is called as a function and includes a check to determine whether the
    code is running in a notebook shell, so that it can safely be used
    within a normal python script since it has no effect when not running
    in a notebook shell.

    Args:
        backend (string, optional (default 'inline')): Name of backend to
            be passed to the `%matplotlib` line magic command.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    if _in_notebook():
        # See https://stackoverflow.com/questions/35595766
        get_ipython().run_line_magic("matplotlib", backend)


def config_notebook_plotting():
    """Configure plotting functions for inline plotting.

    Configure plotting functions for inline plotting within a Jupyter
    Notebook shell. This function has no effect when not within a
    notebook shell, and may therefore be used within a normal python
    script. If environment variable ``MATPLOTLIB_IPYNB_BACKEND`` is set,
    the matplotlib backend is explicitly set to the specified value.
    """
    warnings.warn(
        "The scico.plot submodule is deprecated; use the corresponding "
        "functions in the komplot package",
        DeprecationWarning,
        stacklevel=2,
    )
    kplt.config_notebook_plotting()
