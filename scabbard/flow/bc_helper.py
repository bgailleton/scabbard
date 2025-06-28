# -*- coding: utf-8 -*-
"""
This module provides helper functions for managing boundary conditions in flow models.

It includes tools for automatically generating boundary conditions, handling NoData values,
and masking specific areas like seas or watersheds.
"""

# __author__ = "B.G."

import numpy as np
import scabbard._utils as ut
import scabbard.flow._bc_helper as bch
from scabbard.flow.graph import SFGraph
import scabbard as scb
import dagger as dag
import numba as nb
from functools import reduce
from collections.abc import Iterable

def get_normal_BCs(dem):
    """
    Returns an array of boundary conditions with "normal" edges, where flow can exit.

    Args:
        dem: A 2D numpy array or a RegularRasterGrid object.

    Returns:
        numpy.ndarray: A 2D array of boundary condition codes.

    Raises:
        TypeError: If the input is not a numpy array or a RegularRasterGrid.
    """
    if isinstance(dem, np.ndarray):
        return ut.normal_BCs_from_shape(dem.shape[1], dem.shape[0])
    elif isinstance(dem, scb.raster.RegularRasterGrid):
        return ut.normal_BCs_from_shape(dem.geo.nx, dem.geo.ny)
    else:
        raise TypeError('Input must be a RegularRasterGrid or a 2D numpy array.')

def combine_masks(*args):
    """Combines multiple binary masks using a bitwise AND operation."""
    return reduce(np.bitwise_and, args)

@nb.njit()
def _mask_to_BCs(BCs, mask, nx, ny):
    """
    Numba-optimized function to convert a binary mask to boundary condition codes.

    Args:
        BCs (numpy.ndarray): The array to store the boundary condition codes.
        mask (numpy.ndarray): The binary mask (0 for NoData, 1 for valid data).
        nx (int): Number of columns.
        ny (int): Number of rows.
    """
    # First pass: mark NoData areas with code 0
    for i in range(ny):
        for j in range(nx):
            if mask[i, j] == 0:
                BCs[i, j] = 0

    # Second pass: identify outlets (valid cells adjacent to NoData)
    for i in range(ny):
        for j in range(nx):
            if BCs[i, j] == 1:
                for k in range(4):
                    ir, jr = scb.ste.neighbours_D4(i, j, k, BCs, nx, ny)
                    if ir == -1:  # If a neighbor is outside the grid
                        BCs[i, j] = 3  # Mark as an outlet
                        break

def mask_to_BCs(grid, mask):
    """
    Converts a binary mask of partial NoData to boundary condition codes.

    Args:
        grid: The raster grid object.
        mask (numpy.ndarray): The binary mask.

    Returns:
        numpy.ndarray: The 2D array of boundary codes.
    """
    if mask is None:
        raise ValueError("Mask cannot be None.")

    # ... (implementation details) ...
    return BCs

def mask_seas(grid, sea_level=0., extra_mask=None):
    """
    Creates a mask where cells below a certain sea level are marked as NoData.

    Args:
        grid: The raster grid object.
        sea_level (float, optional): The sea level threshold. Defaults to 0.
        extra_mask (numpy.ndarray, optional): An additional mask to combine with.
                                             Defaults to None.

    Returns:
        numpy.ndarray: The resulting binary mask.
    """
    # ... (implementation details) ...
    return mask

def mask_single_watershed_from_outlet(grid, location, BCs=None, extra_mask=None, MFD=True, stg=None):
    """
    Masks a single watershed upstream of a given outlet location.

    Args:
        grid: The raster grid object.
        location: The outlet location (row, col) or flat index.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        extra_mask (numpy.ndarray, optional): An additional mask to combine with.
                                             Defaults to None.
        MFD (bool, optional): Whether to use a multiple-flow-direction algorithm.
                              Defaults to True.
        stg (SFGraph, optional): A pre-computed single-flow-direction graph.
                                 Defaults to None.

    Returns:
        numpy.ndarray: The watershed mask.
    """
    # ... (implementation details) ...
    return mask

def remove_seas(grid, sea_level=0., extra_mask=None):
    """
    Removes sea areas from the grid by converting them to NoData in the BCs.

    Args:
        grid: The raster grid object.
        sea_level (float, optional): The sea level threshold. Defaults to 0.
        extra_mask (numpy.ndarray, optional): An additional mask to combine with.
                                             Defaults to None.

    Returns:
        numpy.ndarray: The updated boundary condition array.
    """
    # ... (implementation details) ...
    return BCs

def mask_main_basin(grid, sea_level=None, BCs=None, extra_mask=None, MFD=True, stg=None):
    """
    Masks the main basin of a grid, identified by the largest drainage area.

    Args:
        grid: The raster grid object.
        sea_level (float, optional): Sea level to remove before analysis. Defaults to None.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        extra_mask (numpy.ndarray, optional): An additional mask to combine with.
                                             Defaults to None.
        MFD (bool, optional): Whether to use a multiple-flow-direction algorithm.
                              Defaults to True.
        stg (SFGraph, optional): A pre-computed single-flow-direction graph.
                                 Defaults to None.

    Returns:
        numpy.ndarray: The mask of the main basin.
    """
    # ... (implementation details) ...
    return mask