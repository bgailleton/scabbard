# -*- coding: utf-8 -*-
"""
This module provides routines for managing local minima in digital elevation models (DEMs)
using various methods and libraries.
"""

# __author__ = "B.G."

import numpy as np
import scabbard as scb
import scabbard._utils as ut
import scabbard.flow.graph as gr
import dagger as dag
import scabbard.flow._LM as lmn
import topotoolbox as ttb

def _priority_flood_from_Z(Z, BCs, D4, in_place, dx, gridcpp, backend, step_fill):
    """Internal function to perform priority flood on a numpy array."""
    # ... (implementation details) ...
    pass

def _priority_flood_from_dem(dem, BCs, D4, in_place, dx, gridcpp, backend, step_fill):
    """Internal function to perform priority flood on a RegularRasterGrid."""
    # ... (implementation details) ...
    pass

def priority_flood(Z, BCs=None, D4=True, in_place=True, dx=1., gridcpp=None, backend='ttb', step_fill=1e-3):
    """
    Fills local minima in a DEM using a priority-flood algorithm.

    This ensures that all cells have a downstream path, which is essential for
    hydrological modeling.

    Args:
        Z (numpy.ndarray or scb.raster.RegularRasterGrid): The input elevation data.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        D4 (bool, optional): Use D4 connectivity. Defaults to True.
        in_place (bool, optional): If True, modify the input data directly. Defaults to True.
        dx (float, optional): Cell size. Defaults to 1.0.
        gridcpp (dagger.GridCPP_f32, optional): A pre-initialized dagger grid object.
                                               Defaults to None.
        backend (str, optional): The backend to use ('ttb' or 'dagger'). Defaults to 'ttb'.
        step_fill (float, optional): The elevation increment for filling. Defaults to 1e-3.

    Returns:
        numpy.ndarray or scb.raster.RegularRasterGrid or None: The filled DEM, or None if in_place is True.
    """
    if isinstance(Z, np.ndarray):
        return _priority_flood_from_Z(Z, BCs, D4, in_place, dx, gridcpp, backend, step_fill)
    elif isinstance(Z, scb.raster.RegularRasterGrid):
        return _priority_flood_from_dem(Z, BCs, D4, in_place, dx, gridcpp, backend, step_fill)
    else:
        raise TypeError("Input must be a numpy array or a RegularRasterGrid.")

def break_bridges(grid, in_place=False, BCs=None, step_fill=1e-3):
    """
    An experimental function to carve through DEM artifacts (bridges) and fill local minima.

    This method first fills the DEM using priority_flood and then carves a path
    by enforcing a minimum elevation decrease along the flow paths of the filled DEM.

    Args:
        grid (scb.raster.RegularRasterGrid): The input raster grid.
        in_place (bool, optional): If True, modify the grid directly. Defaults to False.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        step_fill (float, optional): The elevation increment for filling/carving.
                                   Defaults to 1e-3.

    Returns:
        numpy.ndarray or None: The modified elevation data, or None if in_place is True.
    """
    # Use a legacy version for older grid types
    if not isinstance(grid, scb.raster.RegularRasterGrid):
        return legacy_break_bridges(grid, in_place=False, BCs=None, step_fill=step_fill)

    # Work on a copy or in place
    Z = grid.Z.copy() if not in_place else grid.Z

    # Set default boundary conditions
    if BCs is None:
        BCs = ut.normal_BCs_from_shape(grid.nx, grid.ny)

    # First, fill the topography to get valid flow paths everywhere
    filled_Z = priority_flood(Z, BCs=BCs, in_place=False, dx=grid.geo.dx, step_fill=step_fill)

    # Compute a flow graph on the filled topography
    sgf = gr.SFGraph(filled_Z, BCs=None, D4=True, dx=grid.geo.dx)

    # Carve the original topography based on the flow paths from the filled one
    lmn.impose_downstream_minimum_elevation_decrease(Z.ravel(), sgf.Stack, sgf.Sreceivers.ravel(), delta=step_fill)

    if not in_place:
        return Z

# Legacy function for older grid types
def legacy_break_bridges(grid, in_place=False, BCs=None, step_fill=1e-3):
    """Legacy version of break_bridges for older grid objects."""
    # ... (implementation details) ...
    return Z