# -*- coding: utf-8 -*-
"""
This module defines the `RegularRasterGrid` class, which manages 2D raster data
with associated geometric information.

It provides a structured way to handle gridded data, including properties for
spatial dimensions and methods for interoperability with other libraries.
"""

# __author__ = "B.G."

import numpy as np
import matplotlib.pyplot as plt
from scabbard import io
from scabbard import geometry as geo
import dagger as dag
import scabbard as scb
from scipy.ndimage import gaussian_filter
import random
from copy import deepcopy

class RegularRasterGrid(object):
    """
    Manages a regular grid with helper functions.

    This class encapsulates a 2D numpy array (Z) representing raster data
    and a `RegularGeometry` object (geo) describing its spatial properties.
    """

    def __init__(self, value, geometry, dtype=np.float32):
        """
        Initializes a RegularRasterGrid object.

        Args:
            value (numpy.ndarray): The 2D numpy array containing the raster data.
            geometry (scabbard.geometry.RegularGeometry): The geometry object describing the grid.
            dtype (numpy.dtype, optional): The desired data type for the raster values.
                                         Defaults to `np.float32`.

        Raises:
            AttributeError: If the provided geometry is not of type `RegularGeometry`
                            or if the shape of `value` does not match the geometry.
        """
        super().__init__()

        # Validate geometry type
        if not isinstance(geometry, geo.RegularGeometry):
            raise AttributeError(
                "A RegularRasterGrid object must be created with a geometry of type RegularGeometry."
            )
        
        # Validate data shape against geometry
        if value.shape != geometry.shape:
            raise AttributeError(
                "Input array shape does not match the geometry's shape."
            )

        # Assign values and geometry
        self.Z = value
        self.geo = geometry

        # Convert data type if necessary
        if self.Z.dtype != dtype:
            self.Z = self.Z.astype(dtype)

    def duplicate_with_other_data(self, value):
        """
        Creates a new RegularRasterGrid instance with the same geometry but different data.

        This is useful for creating new raster layers that align perfectly with an existing one.

        Args:
            value (numpy.ndarray): The 2D numpy array to replace the current raster data.

        Returns:
            RegularRasterGrid: A new instance with the same geometry and the provided data.
        """
        co = deepcopy(self)
        co.Z = value
        return co

    @property
    def dims(self):
        """
        Returns a TopoToolbox-friendly dimension array (ny, nx).
        """
        return np.array([self.geo.ny, self.geo.nx], dtype=np.uint64)

    @property
    def z(self):
        """
        Alias for the raster data array (self.Z), for TopoToolbox compatibility.
        """
        return self.Z

    @z.setter
    def z(self, val):
        """
        Setter for the raster data array (self.Z).
        """
        self.Z = val

    @property
    def rshp(self):
        """
        Returns the shape tuple (ny, nx) suitable for `numpy.reshape`.
        """
        return np.array([self.geo.ny, self.geo.nx], dtype=np.uint64)

    def grid2ttb(self):
        """
        Converts the RegularRasterGrid object to a TopoToolbox-compatible GridObject.

        Returns:
            topotoolbox.GridObject: The converted TopoToolbox grid object.
        """
        from rasterio.crs import CRS
        import topotoolbox as ttb

        ttbgrid = ttb.GridObject()
        ttbgrid.z = self.Z
        ttbgrid.cellsize = self.geo.dx
        ttbgrid.bounds = self.geo.extent
        ttbgrid.name = "from_scabbard"

        # Set CRS if available
        if self.geo.crs:
            ttbgrid.crs = CRS.from_string(self.geo.crs)

        return ttbgrid

    # --- Operator Overloads for element-wise operations ---
    def __add__(self, other):
        """Element-wise addition. Supports adding another RegularRasterGrid or a scalar."""
        if isinstance(other, RegularRasterGrid):
            return RegularRasterGrid(self.Z + other.Z, self.geo)
        else:
            return RegularRasterGrid(self.Z + other, self.geo)

    def __sub__(self, other):
        """Element-wise subtraction. Supports subtracting another RegularRasterGrid or a scalar."""
        if isinstance(other, RegularRasterGrid):
            return RegularRasterGrid(self.Z - other.Z, self.geo)
        else:
            return RegularRasterGrid(self.Z - other, self.geo)

    def __mul__(self, other):
        """Element-wise multiplication. Supports multiplying by another RegularRasterGrid or a scalar."""
        if isinstance(other, RegularRasterGrid):
            return RegularRasterGrid(self.Z * other.Z, self.geo)
        else:
            return RegularRasterGrid(self.Z * other, self.geo)

    def __truediv__(self, other):
        """Element-wise division. Supports dividing by another RegularRasterGrid or a scalar."""
        if isinstance(other, RegularRasterGrid):
            return RegularRasterGrid(self.Z / other.Z, self.geo)
        else:
            return RegularRasterGrid(self.Z / other, self.geo)

def raster_from_array(Z, dx=1.0, xmin=0.0, ymin=0.0, dtype=np.float32):
    """
    Helper function to create a RegularRasterGrid object from a 2D NumPy array.

    Args:
        Z (numpy.ndarray): The 2D NumPy array containing the raster data.
        dx (float, optional): The spatial step (cell size). Defaults to 1.0.
        xmin (float, optional): The minimum x-coordinate of the grid's origin. Defaults to 0.0.
        ymin (float, optional): The minimum y-coordinate of the grid's origin. Defaults to 0.0.
        dtype (numpy.dtype, optional): The desired data type for the raster values.
                                     Defaults to `np.float32`.

    Returns:
        RegularRasterGrid: A new RegularRasterGrid object.
    """
    geometry = scb.geometry.RegularGeometry(Z.shape[1], Z.shape[0], dx, xmin, ymin)
    return RegularRasterGrid(Z, geometry, dtype=dtype)

def raster_from_ttb(ttbgrid):
    """
    Converts a TopoToolbox `GridObj` to a `RegularRasterGrid` object.

    Args:
        ttbgrid (topotoolbox.GridObject): The TopoToolbox grid object.

    Returns:
        RegularRasterGrid: A new `RegularRasterGrid` object.
    """
    # Extract geometry information from the TopoToolbox grid
    geometry = scb.geometry.RegularGeometry(
        ttbgrid.columns, ttbgrid.rows, ttbgrid.cellsize,
        ttbgrid.bounds[0], ttbgrid.bounds[1] # xmin, ymin
    )
    # Create the RegularRasterGrid using the data and extracted geometry
    return RegularRasterGrid(ttbgrid.z, geometry, dtype=ttbgrid.z.dtype)