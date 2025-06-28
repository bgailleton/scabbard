# -*- coding: utf-8 -*-
"""
This module provides routines for calculating drainage area and propagating values
through a flow network.
"""

# __author__ = "B.G."

import numba as nb
import numpy as np
import scabbard.steenbok as st
from scabbard.flow import SFGraph
import scabbard as scb

@nb.njit()
def _drainage_area_sfg(Stack, Sreceivers, dx=1., BCs=None):
    """
    Numba-optimized function to calculate drainage area for a single-flow-direction graph.

    Args:
        Stack (numpy.ndarray): The flow stack (upstream to downstream order).
        Sreceivers (numpy.ndarray): The receiver for each node.
        dx (float, optional): The cell size. Defaults to 1.
        BCs (numpy.ndarray, optional): Boundary conditions. Not currently used.

    Returns:
        numpy.ndarray: The drainage area for each node.
    """
    A = np.zeros_like(Sreceivers, dtype=np.float32)

    # Iterate from upstream to downstream
    for i in range(Stack.shape[0]):
        node = Stack[Stack.shape[0] - 1 - i]
        rec = Sreceivers[node]

        if node == rec:  # Skip pits
            continue

        A[node] += dx * dx  # Add own cell area
        A[rec] += A[node]   # Add to receiver

    return A

def drainage_area(input_data):
    """
    Calculates the drainage area for a given input.

    Args:
        input_data: Can be a pre-computed SFGraph or a RegularRasterGrid.

    Returns:
        numpy.ndarray: A 2D array of drainage areas.
    """
    if isinstance(input_data, SFGraph):
        return _drainage_area_sfg(input_data.Stack, input_data.Sreceivers, dx=input_data.dx).reshape(input_data.ny, input_data.nx)
    elif isinstance(input_data, scb.raster.RegularRasterGrid):
        sgraph = scb.flow.SFGraph(input_data, D4=True, dx=input_data.geo.dx, backend='ttb', fill_LM=False)
        return _drainage_area_sfg(sgraph.Stack, sgraph.Sreceivers, dx=sgraph.dx).reshape(sgraph.ny, sgraph.nx)
    else:
        raise TypeError("Input must be an SFGraph or RegularRasterGrid.")

@nb.njit()
def _propagate_sfg(Stack, Sreceivers, values, dx=1., BCs=None):
    """
    Numba-optimized function to propagate values downstream in a single-flow-direction graph.

    Args:
        Stack (numpy.ndarray): The flow stack.
        Sreceivers (numpy.ndarray): The receiver for each node.
        values (numpy.ndarray): The values to propagate.
        dx (float, optional): Cell size. Not currently used. Defaults to 1.
        BCs (numpy.ndarray, optional): Boundary conditions. Not currently used.

    Returns:
        numpy.ndarray: The propagated values.
    """
    A = np.zeros_like(Sreceivers, dtype=np.float32)

    for i in range(Stack.shape[0]):
        node = Stack[Stack.shape[0] - 1 - i]
        rec = Sreceivers[node]

        if node == rec:
            continue

        A[node] += values[node]
        A[rec] += A[node]

    return A

@nb.njit()
def _propagate_mfd_propS(Z, Stack, values, nx, ny, dx=1., BCs=None, D4=True):
    """
    Numba-optimized function to propagate values using a multiple-flow-direction, slope-proportional method.

    Args:
        Z (numpy.ndarray): Elevation data.
        Stack (numpy.ndarray): The flow stack.
        values (numpy.ndarray): The values to propagate.
        nx (int): Number of columns.
        ny (int): Number of rows.
        dx (float, optional): Cell size. Defaults to 1.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        D4 (bool, optional): Use D4 connectivity. Defaults to True.

    Returns:
        numpy.ndarray: The propagated values.
    """
    # ... (implementation details) ...
    return A

def propagate(input_data, input_values, method='sfd', BCs=None, D4=True, fill_LM=False, step_fill=1e-3, out=None):
    """
    Propagates values with the flow, like drainage area calculation.

    Args:
        input_data: Topographic data (RegularRasterGrid or SFGraph).
        input_values (numpy.ndarray): The values to propagate.
        method (str, optional): 'sfd' or 'mfd_S'. Defaults to 'sfd'.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        D4 (bool, optional): Use D4 connectivity. Defaults to True.
        fill_LM (bool, optional): Fill local minima. Defaults to False.
        step_fill (float, optional): Step for filling. Defaults to 1e-3.
        out (dict, optional): A dictionary to store intermediate results. Defaults to None.

    Returns:
        numpy.ndarray: The propagated values.
    """
    # ... (implementation details) ...
    pass