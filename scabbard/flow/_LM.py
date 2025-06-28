# -*- coding: utf-8 -*-
"""
This module contains Numba-optimized functions for handling local minima in elevation data.
"""

# __author__ = "B.G."

import numpy as np
import numba as nb

@nb.njit()
def impose_downstream_minimum_elevation_decrease(Z, Sstack, Sreceivers, delta=1e-4):
    """
    Enforces a minimum elevation decrease between a node and its downstream receiver.

    This function iterates through a flow stack from downstream to upstream and adjusts
    the elevation of receiver nodes to ensure a consistent downstream slope.
    This is a common technique for filling or breaching digital elevation models (DEMs)
    to ensure proper flow routing.

    Args:
        Z (numpy.ndarray): A 1D array of elevations.
        Sstack (numpy.ndarray): A 1D array of node indices, ordered from upstream to downstream.
        Sreceivers (numpy.ndarray): A 1D array where each element is the receiver of the
                                  node at that index.
        delta (float, optional): The minimum elevation difference to enforce.
                               Defaults to 1e-4.
    """
    # Iterate through the stack from downstream to upstream
    for i in range(Z.shape[0]):
        node = Sstack[Z.shape[0] - 1 - i]
        rec = Sreceivers[node]

        # If the node is not its own receiver (i.e., not a pit)
        if node != rec:
            # If the receiver's elevation is not sufficiently lower than the node's elevation
            if Z[rec] >= Z[node] - delta:
                # Lower the receiver's elevation to enforce the minimum decrease
                Z[rec] = Z[node] - delta