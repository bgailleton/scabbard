# -*- coding: utf-8 -*-
"""
This module defines the `GFUI` class, intended as a Work-In-Progress (WIP)
universal interface for Graphflood, abstracting different backends.
"""

# __author__ = "B.G."

import numpy as np
import scabbard as scb
import scipy.io as sio

class GFUI:
    """
    Graphflood Universal Interface (WIP).

    This class aims to provide a unified interface to different Graphflood
    backends (GPU, Dagger, TopoToolbox), managing the underlying model objects.
    """

    def __init__(self, backend, grid):
        """
        Initializes the GFUI object with a specified backend and grid.

        Args:
            backend (str): The name of the backend to use ('gpu', 'dagger', 'cpu', 'ttb', 'topotoolbox').
            grid: The grid object for the simulation.

        Raises:
            RuntimeError: If the specified backend is not recognized.
        """
        backend = backend.lower()
        # Set the backend type
        if backend in ['gpu', 'dagger', 'cpu', 'ttb', 'topotoolbox']:
            self.backend = backend
        else:
            raise RuntimeError(f'Backend "{backend}" not recognized.')

        #########################
        # Internal backend objects
        # These attributes will hold references to the specific model objects
        # depending on the chosen backend.

        ## Graphflood OG from DAGGER (v1)
        self._cpp_gf = None         # Graphflood C++ object
        self._cpp_graph = None      # Dagger graph object
        self._cpp_connector = None  # Dagger connector object

        ## Induced graph and v2 (future Dagger versions)
        self._cpp_env = None

        ## lib-py-TopoToolBox
        # TopoToolbox typically operates on its own grid objects, no persistent model object here

        ## GPU backend (Riverdale/Taichi)
        self._param = None          # Riverdale parameter object
        self._rd = None             # Riverdale model object

    def _init_riverdale_backend(self, grid):
        """
        Initializes the Riverdale (GPU) backend specific components.

        Args:
            grid: The grid object to be used by the Riverdale backend.
        """
        # This method would contain the logic to set up `self._param` and `self._rd`
        # based on the provided `grid` and other simulation parameters.
        # For now, it's a placeholder.
        pass