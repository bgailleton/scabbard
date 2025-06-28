# -*- coding: utf-8 -*-
"""
This module provides functions for calculating flow distance from outlets.
"""

# __author__ = "B.G."

import scabbard as scb
import numpy as np
import numba as nb

def compute_flow_distance_from_outlet(
    grid: scb.raster.RegularRasterGrid,
    method: str = "mean",
    BCs=None,
    Stack=None,
    fill_LM=False,
    step_fill=1e-3,
    D8=True,
):
    """
    Computes the flow distance from each cell to the watershed outlet.

    This function can use different methods to calculate the distance in a
    multiple-flow-direction context (mean, min, or max distance from downstream neighbors).

    Args:
        grid (scb.raster.RegularRasterGrid): The input raster grid.
        method (str, optional): The distance calculation method. Can be "mean", "min", or "max".
                                Defaults to "mean".
        BCs (numpy.ndarray, optional): Boundary conditions. If None, normal BCs are generated.
                                     Defaults to None.
        Stack (numpy.ndarray, optional): A pre-computed flow stack. If None, it will be calculated.
                                       Defaults to None.
        fill_LM (bool, optional): Whether to fill local minima before calculating the stack.
                                Defaults to False.
        step_fill (float, optional): The step used for filling local minima. Defaults to 1e-3.
        D8 (bool, optional): If True, use D8 connectivity. If False, use D4. Defaults to True.

    Returns:
        numpy.ndarray: A 2D array of flow distances.

    Raises:
        RuntimeError: If the input grid is not a RegularRasterGrid or if the method is not supported.
    """
    # Validate input grid type
    if not isinstance(grid, scb.raster.RegularRasterGrid):
        raise RuntimeError("compute_flow_distance_from_outlet requires a RegularRasterGrid as grid")

    # Set default boundary conditions if not provided
    tBCs = scb.flow.get_normal_BCs(grid) if BCs is None else BCs

    # Calculate the processing stack if not provided
    if Stack is None:
        if fill_LM:
            Stack = np.zeros_like(grid.Z.ravel(), dtype=np.uint64)
            # Use a priority flood algorithm to create a hydrologically correct stack
            scb.ttb.graphflood.funcdict['priority_flood_TO'](
                grid.Z.ravel(), Stack, tBCs.ravel(), grid.dims, D8, step_fill
            )
        else:
            # A simple sort by elevation is faster but may not be correct for all DEMs
            Stack = np.argsort(grid.Z.ravel()).astype(np.uint64)

    # Select the appropriate distance calculation function based on the method
    if method.lower() == "mean":
        dist_func = scb.ste.mean_dist_to_outlet
    elif method.lower() == "min":
        dist_func = scb.ste.min_dist_to_outlet
    elif method.lower() == "max":
        dist_func = scb.ste.max_dist_to_outlet
    else:
        raise RuntimeError("Supported methods are: 'mean', 'min', or 'max'.")

    # Calculate and return the flow distance
    return dist_func(
        Stack,
        grid.Z.ravel(),
        tBCs.ravel(),
        D8,
        grid.geo.nx,
        grid.geo.ny,
        grid.geo.dx,
    ).reshape(grid.rshp)