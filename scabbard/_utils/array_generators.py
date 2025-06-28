# -*- coding: utf-8 -*-
"""
Set of functions to generate specific arrays that I commonly use.

This module provides functions for creating numpy arrays with specific
configurations, such as boundary conditions for simulations.
"""

# __author__ = "B.G."

import numpy as np

def normal_BCs_from_shape(nx, ny, out_code=3):
    """
    Returns an array of uint8 boundary codes with "normal" edges.

    This function creates a 2D numpy array representing boundary conditions,
    where the edges are marked with a specific `out_code`.

    Args:
        nx (int): Number of columns.
        ny (int): Number of rows.
        out_code (uint8, optional): Code for the edges. Defaults to 3, which
                                  is a permissive code to let flow out if no
                                  downstream neighbour exists.

    Returns:
        numpy.ndarray: A 2D numpy array of boundary conditions.
    """
    # Create an array of ones with the given shape
    BCs = np.ones((ny, nx), dtype=np.uint8)
    # Set the boundary columns to the out_code
    BCs[:, [-1, 0]] = out_code
    # Set the boundary rows to the out_code
    BCs[[-1, 0], :] = out_code

    return BCs

def periodic_EW_BCs_from_shape(nx, ny, out_code=3):
    """
    Returns an array of uint8 boundary codes with periodic East-West edges.

    This function creates a 2D numpy array representing boundary conditions,
    where the East and West edges are periodic (code 9) and the North and
    South edges are normal.

    Args:
        nx (int): Number of columns.
        ny (int): Number of rows.
        out_code (uint8, optional): Code for the North/South edges. Defaults to 3.

    Returns:
        numpy.ndarray: A 2D numpy array of boundary conditions.
    """
    # Create an array of ones with the given shape
    BCs = np.ones((ny, nx), dtype=np.uint8)
    # Set the East/West boundaries to periodic
    BCs[:, [-1, 0]] = 9
    # Set the North/South boundaries to the out_code
    BCs[[-1, 0], :] = out_code

    return BCs

def periodic_NS_BCs_from_shape(nx, ny, out_code=3):
    """
    Returns an array of uint8 boundary codes with periodic North-South edges.

    This function creates a 2D numpy array representing boundary conditions,
    where the North and South edges are periodic (code 9) and the East and
    West edges are normal.

    Args:
        nx (int): Number of columns.
        ny (int): Number of rows.
        out_code (uint8, optional): Code for the East/West edges. Defaults to 3.

    Returns:
        numpy.ndarray: A 2D numpy array of boundary conditions.
    """
    # Create an array of ones with the given shape
    BCs = np.ones((ny, nx), dtype=np.uint8)
    # Set the North/South boundaries to periodic
    BCs[[-1, 0], :] = 9
    # Set the East/West boundaries to the out_code
    BCs[:, [-1, 0]] = out_code

    return BCs