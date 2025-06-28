# -*- coding: utf-8 -*-
"""
This module defines the `RegularGeometry` class, which represents the geometric properties
of a regular 2D grid.

It extends the `BaseGeometry` abstract class, providing concrete implementations for
properties and methods relevant to regularly spaced grids.
"""

# __author__ = "B.G."

import numpy as np
from .geometry import BaseGeometry

class RegularGeometry(BaseGeometry):
    """
    Geometry class for any object with a regular 2D structure.

    This class describes a grid using its number of rows and columns, spatial step,
    and origin coordinates. It provides methods for converting between different
    coordinate systems (row/column, flat index, real-world X/Y).
    """

    def __init__(self, nx, ny, dx, xmin, ymin):
        """
        Initializes a RegularGeometry object.

        Args:
            nx (int): Number of columns (cells in x-direction).
            ny (int): Number of rows (cells in y-direction).
            dx (float): Spatial step (cell size) in both x and y directions.
            xmin (float): Minimum Easting coordinate of the grid's origin.
            ymin (float): Minimum Northing coordinate of the grid's origin.
        """
        super().__init__()

        self._nx = nx
        self._ny = ny
        self._nxy = nx * ny
        self._dx = dx
        self._lx = dx * nx  # Total length in x-direction
        self._ly = dx * ny  # Total length in y-direction
        self._dxy = 2**0.5 * dx # Diagonal distance
        self._xmin = xmin
        self._ymin = ymin

    @property
    def dx(self):
        """Returns the spatial step (cell size)."""
        return self._dx

    @property
    def dxy(self):
        """Returns the diagonal spatial step."""
        return self._dxy

    @property
    def N(self):
        """Returns the total number of nodes/cells."""
        return self._nxy

    @property
    def nx(self):
        """Returns the number of columns."""
        return self._nx

    @property
    def ny(self):
        """Returns the number of rows."""
        return self._ny

    @property
    def lx(self):
        """Returns the total length of the grid in the x-direction."""
        return self._lx

    @property
    def ly(self):
        """Returns the total length of the grid in the y-direction."""
        return self._ly

    @property
    def shape(self):
        """Returns the shape of the grid as a tuple (ny, nx)."""
        return (self.ny, self.nx)

    @property
    def xmin(self):
        """Returns the minimum x-coordinate."""
        return self._xmin

    @property
    def xmax(self):
        """Returns the maximum x-coordinate."""
        return self._xmin + (self.nx) * self.dx

    @property
    def ymin(self):
        """Returns the minimum y-coordinate."""
        return self._ymin

    @property
    def ymax(self):
        """Returns the maximum y-coordinate."""
        return self._ymin + (self.ny) * self.dx

    def row_col_to_flatID(self, row, col):
        """
        Converts row and column indices to a flat (1D) index.

        Args:
            row (int or numpy.ndarray): Row index or array of row indices.
            col (int or numpy.ndarray): Column index or array of column indices.

        Returns:
            int or numpy.ndarray: The corresponding flat index or array of flat indices.
        """
        return row * self.nx + col

    def flatID_to_row_col(self, flatID):
        """
        Converts a flat (1D) index to row and column indices.

        Args:
            flatID (int or numpy.ndarray): Flat index or array of flat indices.

        Returns:
            tuple: A tuple (row, col) or (numpy.ndarray, numpy.ndarray) of row and column indices.
        """
        return flatID // self.nx, flatID % self.nx

    def row_col_to_X_Y(self, row, col):
        """
        Converts row and column indices to real-world X and Y coordinates.

        Args:
            row (int or numpy.ndarray): Row index or array of row indices.
            col (int or numpy.ndarray): Column index or array of column indices.

        Returns:
            tuple: A tuple (X, Y) or (numpy.ndarray, numpy.ndarray) of X and Y coordinates.
        """
        # Calculate X coordinate (column * cell_size + x_origin + half_cell_size)
        X = (col * self.dx) + self.xmin + self.dx / 2
        
        # Calculate Y coordinate, considering potential Y-axis inversion
        if self.Y_inverted:
            Y = ((self.ny - 1 - row) * self.dx) + self.ymin + self.dx / 2
        else:
            Y = (row * self.dx) + self.ymin + self.dx / 2
        return X, Y

    def X_Y_to_row_col(self, X, Y):
        """
        Converts real-world X and Y coordinates to row and column indices.

        Args:
            X (float or numpy.ndarray): X coordinate or array of X coordinates.
            Y (float or numpy.ndarray): Y coordinate or array of Y coordinates.

        Returns:
            tuple: A tuple (row, col) or (numpy.ndarray, numpy.ndarray) of row and column indices.
        """
        # Calculate column index
        col = int(np.floor((X - self.xmin) / self.dx))

        # Calculate row index, considering potential Y-axis inversion
        if self.Y_inverted:
            row = int(self.ny - 1 - np.floor((Y - self.ymin) / self.dx))
        else:
            row = int(np.floor((Y - self.ymin) / self.dx))
        return row, col

    def flatID_to_X_Y(self, flatID):
        """
        Converts a flat (1D) index to real-world X and Y coordinates.

        Args:
            flatID (int or numpy.ndarray): Flat index or array of flat indices.

        Returns:
            tuple: A tuple (X, Y) or (numpy.ndarray, numpy.ndarray) of X and Y coordinates.
        """
        row, col = self.flatID_to_row_col(flatID)
        return self.row_col_to_X_Y(row, col)

    def X_Y_to_flatID(self, X, Y):
        """
        Converts real-world X and Y coordinates to a flat (1D) index.

        Args:
            X (float or numpy.ndarray): X coordinate or array of X coordinates.
            Y (float or numpy.ndarray): Y coordinate or array of Y coordinates.

        Returns:
            int or numpy.ndarray: The corresponding flat index or array of flat indices.
        """
        row, col = self.X_Y_to_row_col(X, Y)
        return self.row_col_to_flatID(row, col)

    @property
    def xl_centered(self):
        """Returns the x-coordinate of the center of the leftmost column."""
        return self.xmin + self.dx / 2.

    @property
    def xr_centered(self):
        """Returns the x-coordinate of the center of the rightmost column."""
        return self.xmax - self.dx / 2.

    @property
    def yt_centered(self):
        """Returns the y-coordinate of the center of the topmost row."""
        return self.ymin + self.dx / 2.

    @property
    def yb_centered(self):
        """Returns the y-coordinate of the center of the bottommost row."""
        return self.ymax - self.dx / 2.

    @property
    def X(self):
        """Returns an array of x-coordinates for the center of each column."""
        return np.linspace(self.xl_centered, self.xr_centered, self.nx)

    @property
    def Y(self):
        """Returns an array of y-coordinates for the center of each row."""
        return np.linspace(self.yt_centered, self.yb_centered, self.ny)

    @property
    def Y_inverted(self):
        """
        Checks if the Y-axis is inverted (i.e., top-to-bottom increasing Y).

        Returns:
            bool: True if Y-axis is inverted, False otherwise.
        """
        return self.Y[0] > self.Y[-1]

    @property
    def XY(self):
        """
        Returns a meshgrid of X and Y coordinates for all cell centers.

        Returns:
            tuple: A tuple containing two 2D numpy arrays (X_coords, Y_coords).
        """
        return np.meshgrid(self.X, self.Y)

    @property
    def XX(self):
        """Returns the 2D array of X coordinates from the meshgrid."""
        return self.XY[0]

    @property
    def YY(self):
        """Returns the 2D array of Y coordinates from the meshgrid."""
        return self.XY[1]

    @property
    def dxnxny(self):
        """Returns a tuple (dx, nx, ny)."""
        return self.dx, self.nx, self.ny