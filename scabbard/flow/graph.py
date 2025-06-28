# -*- coding: utf-8 -*-
"""
This module provides different graph wrappers for various flow routing libraries.

It defines a standardized `SFGraph` (Single-Flow Graph) class that can be used
with different backends like `topotoolbox` or a custom C++ backend (`dagger`).
"""

# __author__ = "B.G."

import numpy as np
import dagger as dag
import scabbard as scb
import scabbard._utils as ut
import topotoolbox as ttb

class SFGraph(object):
    """
    A container for a single-flow-direction (SFD) graph.

    This class stores the minimal structure of a flow graph, including receivers,
    donors, and a topological ordering (stack), and provides methods to create
    and update this structure from elevation data.
    """

    def __init__(self, Z, BCs=None, D4=True, dx=1., backend='ttb', fill_LM=False, step_fill=1e-3):
        """
        Initializes the SFGraph object.

        Args:
            Z (numpy.ndarray or scb.raster.RegularRasterGrid): The elevation data.
            BCs (numpy.ndarray, optional): Boundary conditions. If None, normal BCs are created.
                                         Defaults to None.
            D4 (bool, optional): If True, use D4 connectivity. If False, use D8.
                               Defaults to True.
            dx (float, optional): The cell size. Defaults to 1.0.
            backend (str, optional): The backend library to use ('ttb' or 'dagger').
                                   Defaults to 'ttb'.
            fill_LM (bool, optional): If True, fill local minima in the DEM.
                                    Defaults to False.
            step_fill (float, optional): The elevation increment for filling minima.
                                       Defaults to 1e-3.
        """
        # Validate and extract elevation data
        if isinstance(Z, np.ndarray):
            tZ = Z
        elif isinstance(Z, scb.raster.RegularRasterGrid):
            tZ = Z.Z
        else:
            raise ValueError("Elevation data must be a numpy array or RegularRasterGrid.")

        if tZ.ndim != 2:
            raise ValueError("Elevation data must be a 2D array.")

        # --- Set up geometrical and graph properties ---
        self.shape = tZ.shape
        self.dim = np.array(self.shape, dtype=np.uint64)
        self.D4 = D4
        self.dx = dx
        self.backend = backend
        self.gridcpp = dag.GridCPP_f32(self.nx, self.ny, dx, dx, 3)

        # --- Initialize graph arrays ---
        dtype = np.uint64 if self.backend == 'ttb' else np.int32
        self.Sreceivers = np.zeros(self.nxy, dtype=dtype)  # Steepest receiver
        self.Sdx = np.zeros(self.nxy, dtype=np.float32)    # Distance to receiver
        self.Ndonors = np.zeros(self.nxy, dtype=np.uint8 if self.backend == 'ttb' else np.int32) # Number of donors
        self.donors = np.zeros(self.nxy * (4 if self.D4 else 8), dtype=dtype) # Donors
        self.Stack = np.zeros(self.nxy, dtype=dtype) # Topological order

        # Build the initial graph
        self.update(tZ, BCs, fill_LM, step_fill)

    def update(self, Z, BCs=None, fill_LM=False, step_fill=1e-3):
        """
        Updates the graph with a new topography and/or boundary conditions.

        Args:
            Z (numpy.ndarray or scb.raster.RegularRasterGrid): The new elevation data.
            BCs (numpy.ndarray, optional): New boundary conditions. Defaults to None.
            fill_LM (bool, optional): Whether to fill local minima. Defaults to False.
            step_fill (float, optional): The increment for filling. Defaults to 1e-3.
        """
        # Validate and extract elevation data
        if isinstance(Z, np.ndarray):
            tZ = Z
        elif isinstance(Z, scb.raster.RegularRasterGrid):
            tZ = Z.Z
        else:
            raise ValueError("Elevation data must be a numpy array or RegularRasterGrid.")

        if tZ.shape != self.shape:
            raise AttributeError("New topography must have the same shape as the original.")

        # Set default boundary conditions if not provided
        if BCs is None:
            BCs = ut.normal_BCs_from_shape(self.nx, self.ny)

        # --- Build the graph using the selected backend ---
        if self.backend == 'dagger':
            if self.D4:
                dag.compute_SF_stack_D4_full_f32(self.gridcpp, tZ, self.Sreceivers.reshape(self.ny, self.nx), self.Ndonors.reshape(self.ny, self.nx), self.donors.reshape(self.ny, self.nx), self.Stack, BCs)
            else:
                raise NotImplementedError("D8 SFGraph not yet implemented for the dagger backend.")
        elif self.backend == 'ttb':
            self.Ndonors.fill(0)
            if fill_LM:
                # Use priority flood to create a hydrologically correct stack
                scb.ttb.graphflood.funcdict['priority_flood_TO'](tZ.ravel(), self.Stack, BCs.ravel(), self.dim, not self.D4, step_fill)
            
            # Compute the single-flow graph using topotoolbox
            ttb.graphflood.funcdict['sfgraph'](tZ.ravel(), self.Sreceivers, self.Sdx, self.donors, self.Ndonors, self.Stack, BCs.ravel(), self.dim, self.dx * self.dx, not self.D4, False, step_fill)

    @property
    def nx(self):
        """Number of columns in the grid."""
        return self.shape[1]

    @property
    def ny(self):
        """Number of rows in the grid."""
        return self.shape[0]

    @property
    def nxy(self):
        """Total number of cells in the grid."""
        return self.shape[0] * self.shape[1]