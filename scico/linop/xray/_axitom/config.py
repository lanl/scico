"""
This file is a modified version of "config.py" from the
[AXITOM](https://github.com/PolymerGuy/AXITOM) package.

Config object and factory.

This module contains the Config class which has all the settings that are
used during the reconstruction of the tomogram.
"""

import numpy as np


class Config:
    """Configuration object for the forward projection."""

    def __init__(
        self,
        n_pixels_u: int,
        n_pixels_v: int,
        pixel_size_u: float,
        pixel_size_v: float,
        source_to_detector_dist: float,
        source_to_object_dist: float,
        **kwargs,
    ):
        """
        Note that invalid arguments are neglected without warning.

        Args:
            n_pixels_u: Number of pixels in the u direction of the sensor.
            n_pixels_v: Number of pixels in the u direction of the sensor.
            pixel_size_u: Pixel size in the u direction [mm].
            pixel_size_v: Pixel size in the v direction [mm].
            source_to_detector_dist: Distance between source and
              detector [mm].
            source_to_object_dist: Distance between source and object
              [mm].
        """

        self.n_pixels_u = n_pixels_u
        self.n_pixels_v = n_pixels_v

        self.pixel_size_u = pixel_size_u
        self.pixel_size_v = pixel_size_v

        self.detector_size_u = self.pixel_size_u * self.n_pixels_u
        self.detector_size_v = self.pixel_size_v * self.n_pixels_v

        self.source_to_detector_dist = source_to_detector_dist
        self.source_to_object_dist = source_to_object_dist

        # All values below are calculated

        self.object_size_x = (
            self.detector_size_u * self.source_to_object_dist / self.source_to_detector_dist
        )
        self.object_size_y = (
            self.detector_size_u * self.source_to_object_dist / self.source_to_detector_dist
        )
        self.object_size_z = (
            self.detector_size_v * self.source_to_object_dist / self.source_to_detector_dist
        )

        self.voxel_size_x = self.object_size_x / self.n_pixels_u
        self.voxel_size_y = self.object_size_y / self.n_pixels_u
        self.voxel_size_z = self.object_size_z / self.n_pixels_v

        self.object_ys = (
            np.arange(self.n_pixels_u, dtype=np.float32) - self.n_pixels_u / 2.0
        ) * self.voxel_size_y
        self.object_xs = (
            np.arange(self.n_pixels_u, dtype=np.float32) - self.n_pixels_u / 2.0
        ) * self.voxel_size_x
        self.object_zs = (
            np.arange(self.n_pixels_v, dtype=np.float32) - self.n_pixels_v / 2.0
        ) * self.voxel_size_z

        self.detector_us = (
            np.arange(self.n_pixels_u, dtype=np.float32) - self.n_pixels_u / 2.0
        ) * self.pixel_size_u
        self.detector_vs = (
            np.arange(self.n_pixels_v, dtype=np.float32) - self.n_pixels_v / 2.0
        ) * self.pixel_size_v

    def __repr__(self):
        str = f"Source-object distance: {self.source_to_object_dist}  "
        str += f"Source-detector distance: {self.source_to_detector_dist}\n"
        str += f"Detector pixels: {self.n_pixels_u}, {self.n_pixels_v}  "
        str += f"Detector size: {self.detector_size_u:.3e}, {self.detector_size_v:.3e}\n"
        str += f"Pixel size: {self.pixel_size_u:.3e}, {self.pixel_size_v:.3e}\n"
        str += (
            f"Voxel size: {self.voxel_size_x:.3e}, {self.voxel_size_y:.3e}, "
            f"{self.voxel_size_z:.3e}"
        )
        return str

    def with_param(self, **kwargs):
        """Get a clone of the object with changed parameters.

        Get a clone of the object with changed parameters and all
        calculations updated.

        Args:
          kwargs: The arguments of the config object that should be
            changed.

        Returns:
          obj: Config object with modified settings.

        """
        params = self.__dict__.copy()

        for arg, value in kwargs.items():
            params[arg] = value
        return Config(**params)
