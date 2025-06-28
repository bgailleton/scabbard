# -*- coding: utf-8 -*-
"""
This module provides helper functions for handling boundary conditions in flow models.

It includes Numba-optimized functions for tasks like watershed masking.
"""

# __author__ = "B.G."

import numba as nb
import numpy as np

@nb.njit()
def mask_watershed_SFD(start_node, Stack, Sreceivers):
    """
    Masks all nodes draining to a single point in a single-flow-direction (SFD) graph.

    This function traverses a pre-calculated flow stack to identify all nodes
    that contribute to a given starting node.

    Args:
        start_node (int): The index of the node at the outlet of the watershed to be masked.
        Stack (numpy.ndarray): A 1D array of node indices, ordered from upstream to downstream.
        Sreceivers (numpy.ndarray): A 1D array where each element is the receiver of the
                                  node at that index.

    Returns:
        numpy.ndarray: A binary mask (uint8) of the same shape as `Sreceivers`,
                       where 1 indicates a node is within the watershed of `start_node`
                       and 0 otherwise.
    """
    # Initialize a mask with zeros
    mask = np.zeros_like(Sreceivers, dtype=np.uint8)

    # Mark the starting node as part of the watershed
    mask[start_node] = 1

    # Iterate through the flow stack
    for node in Stack:
        # Skip the starting node itself
        if node == start_node:
            continue

        # If the receiver of the current node is in the mask, the current node is also in the mask
        mask[node] = mask[Sreceivers[node]]

    return mask