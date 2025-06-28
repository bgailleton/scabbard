# -*- coding: utf-8 -*-
"""
This module defines the abstract base class for all geometry objects within scabbard.

It establishes a common interface for accessing geometric properties and performing
coordinate transformations, ensuring consistency across different grid types.
"""

# __author__ = "B.G."

import numpy as np
from abc import ABC, abstractmethod

class BaseGeometry(ABC):
    """
    Base abstract class for geometry objects.

    This class defines the essential properties and methods that any concrete
    geometry implementation must provide, such as dimensions, spatial extent,
    and coordinate conversion functions.
    """

    def __init__(self):
        """
        Initializes the BaseGeometry object.
        """
        self._crs = None  # Coordinate Reference System

    @property
    @abstractmethod
    def N(self):
        """Returns the total number of nodes/cells in the geometry."""
        pass

    @property
    def nxy(self):
        """Alias for the total number of nodes/cells (N)."""
        return self.N

    @property
    @abstractmethod
    def dx(self):
        """Returns the spatial step (cell size) in the x-direction."""
        pass

    @property
    @abstractmethod
    def nx(self):
        """Returns the number of nodes/columns in the x-direction."""
        pass

    @property
    def ncolumns(self):
        """Alias for the number of columns (nx)."""
        return self.nx

    @property
    @abstractmethod
    def ny(self):
        """Returns the number of nodes/rows in the y-direction."""
        pass

    @property
    def nrows(self):
        """Alias for the number of rows (ny)."""
        return self.ny

    @property
    @abstractmethod
    def xmin(self):
        """Returns the minimum x-coordinate of the geometry's extent."""
        pass

    @property
    def Xmin(self):
        """Alias for the minimum x-coordinate (xmin)."""
        return self.xmin

    @property
    @abstractmethod
    def xmax(self):
        """Returns the maximum x-coordinate of the geometry's extent."""
        pass

    @property
    def Xmax(self):
        """Alias for the maximum x-coordinate (xmax)."""
        return self.xmax

    @property
    @abstractmethod
    def ymin(self):
        """Returns the minimum y-coordinate of the geometry's extent."""
        pass

    @property
    def Ymin(self):
        """Alias for the minimum y-coordinate (ymin)."""
        return self.ymin

    @property
    @abstractmethod
    def ymax(self):
        """Returns the maximum y-coordinate of the geometry's extent."""
        pass

    @property
    def Ymax(self):
        """Alias for the maximum y-coordinate (ymax)."""
        return self.ymax

    @property
    @abstractmethod
    def shape(self):
        """Returns the shape of the geometry (e.g., (ny, nx) for a 2D grid)."""
        pass

    @property
    def crs(self):
        """Returns the Coordinate Reference System (CRS) of the geometry."""
        return self._crs

    @abstractmethod
    def row_col_to_flatID(self, row, col):
        """
        Converts row and column indices to a flat (1D) index.

        Args:
            row (int or numpy.ndarray): Row index or array of row indices.
            col (int or numpy.ndarray): Column index or array of column indices.

        Returns:
            int or numpy.ndarray: The corresponding flat index or array of flat indices.
        """
        pass

    @abstractmethod
    def flatID_to_row_col(self, flatID):
        """
        Converts a flat (1D) index to row and column indices.

        Args:
            flatID (int or numpy.ndarray): Flat index or array of flat indices.

        Returns:
            tuple: A tuple (row, col) or (numpy.ndarray, numpy.ndarray) of row and column indices.
        """
        pass

    @abstractmethod
    def row_col_to_X_Y(self, row, col):
        """
        Converts row and column indices to real-world X and Y coordinates.

        Args:
            row (int or numpy.ndarray): Row index or array of row indices.
            col (int or numpy.ndarray): Column index or array of column indices.

        Returns:
            tuple: A tuple (X, Y) or (numpy.ndarray, numpy.ndarray) of X and Y coordinates.
        """
        pass

    @abstractmethod
    def X_Y_to_row_col(self, X, Y):
        """
        Converts real-world X and Y coordinates to row and column indices.

        Args:
            X (float or numpy.ndarray): X coordinate or array of X coordinates.
            Y (float or numpy.ndarray): Y coordinate or array of Y coordinates.

        Returns:
            tuple: A tuple (row, col) or (numpy.ndarray, numpy.ndarray) of row and column indices.
        """
        pass

    @abstractmethod
    def flatID_to_X_Y(self, flatID):
        """
        Converts a flat (1D) index to real-world X and Y coordinates.

        Args:
            flatID (int or numpy.ndarray): Flat index or array of flat indices.

        Returns:
            tuple: A tuple (X, Y) or (numpy.ndarray, numpy.ndarray) of X and Y coordinates.
        """
        pass

    @abstractmethod
    def X_Y_to_flatID(self, X, Y):
        """
        Converts real-world X and Y coordinates to a flat (1D) index.

        Args:
            X (float or numpy.ndarray): X coordinate or array of X coordinates.
            Y (float or numpy.ndarray): Y coordinate or array of Y coordinates.

        Returns:
            int or numpy.ndarray: The corresponding flat index or array of flat indices.
        """
        pass

    @property
    def extent(self):
        """
        Returns the bounding box of the geometry in a format suitable for Matplotlib's imshow.

        Returns:
            list: A list [xmin, xmax, ymax, ymin].
        """
        return [self.xmin, self.xmax, self.ymax, self.ymin]