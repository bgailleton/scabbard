# -*- coding: utf-8 -*-
"""
This module defines the base class for a river network.
"""

# __author__ = "B.G."

import scabbard as scb
import numpy as np
import numba as nb

class RivNet:
    """
    A simple base class for a river network.

    This class is intended to hold the minimal data structure required to define
    a river network, such as the node IDs from the original grid.
    """

    def __init__(self):
        """
        Initializes the RivNet object.
        """
        # A 1D array to hold the flat indices of the nodes in the river network
        self.nodes = None
