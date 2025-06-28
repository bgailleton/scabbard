'''
This module provides Numba-optimized implementations of Stream Power Models (SPL) within the Steenbok library.
It focuses on functional implementations for topographic evolution.

Author: B.G. (last modification: 07/2024 - Acigné)
'''

import numba as nb
import numpy as np
from enum import Enum
import scabbard as scb


@nb.njit()
def _impl_spl_SFD_single(
		Stack,       # Topologically ordered nodes
		Sreceivers,  # Steepest Receiver (flat index)
		Sdx,         # Distance to steepest receiver
		Z,           # Topography (edited in place)
		A,           # Drainage area, or Discharge
		nx,          # Number of columns
		ny,          # Number of rows
		dx,          # Spatial step
		BCs,         # Boundary codes
		K,           # Erodability coefficient
		m,           # Area exponent 
		n,           # Slope exponent
		dt,          # Time Step
	):
	"""
	Numba-optimized implementation of a single-flow direction (SFD) Stream Power Law (SPL) model.

	This function simulates topographic evolution by iteratively adjusting elevation based on
	drainage area and local slope, assuming a single erodability coefficient for the entire domain.

	Args:
		Stack (numpy.ndarray): 1D array of node indices, topologically ordered from downstream to upstream.
		Sreceivers (numpy.ndarray): 1D array where each element is the flat index of the steepest receiver for the corresponding node.
		Sdx (numpy.ndarray): 1D array where each element is the distance to the steepest receiver for the corresponding node.
		Z (numpy.ndarray): 1D array of topographic elevations (modified in-place).
		A (numpy.ndarray): 1D array of drainage area or discharge values for each node.
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.
		dx (float): Spatial step size.
		BCs (numpy.ndarray): 1D array of boundary condition codes.
		K (float): Erodability coefficient.
		m (float): Area exponent in the Stream Power Law.
		n (float): Slope exponent in the Stream Power Law.
		dt (float): Time step for the simulation.

	Returns:
		None: The `Z` array is modified in-place.

	Author: B.G. (last modifications 09/2024)
	"""
	
	# Traverse the nodes in ascending order (downstream to upstream)
	for node in Stack:

		# Ignore outlet nodes, inactive nodes, or nodes that are their own receiver (e.g., pits)
		if node == Sreceivers[node] or scb.ste.can_out_flat(node,BCs) or scb.ste.is_active_flat(node,BCs) == False:
			continue

		# Calculate the local slope (dz/dx) to the steepest receiver
		dzdx = (Z[node] - Z[Sreceivers[node]])/Sdx[node]

		# If the slope is non-positive (uphill or flat), slightly raise the current node's elevation
		# to ensure a positive slope for erosion calculation and recompute dzdx.
		if(dzdx<=0):
			Z[node] = Z[Sreceivers[node]] + 1e-4
			dzdx = 1e-4/dx

		# Calculate the erosion factor based on K, dt, drainage area, and distance to receiver
		factor = K * dt * (A[node])**m / (Sdx[node]**n);

		# Initialize variables for iterative elevation adjustment
		ielevation = Z[node];
		irec_elevation = Z[Sreceivers[node]];
		elevation_k = ielevation;
		elevation_prev = Z[node] + 500; # Set to a value far from current to ensure first iteration
		tolerance = 1e-5;

		# Iteratively adjust the elevation until convergence
		while (abs(elevation_k - elevation_prev) > tolerance) :
			elevation_prev = elevation_k
			slope = max(elevation_k - irec_elevation, 1e-6) # Ensure positive slope for calculation
			diff = (elevation_k - ielevation + factor * slope**n) / (1. + factor * n * slope**(n - 1))
			elevation_k -= diff; # Update elevation
		
		Z[node] = elevation_k # Set the final adjusted elevation

@nb.njit()
def _impl_spl_SFD_variable(
		Stack,       # Topologically ordered nodes
		Sreceivers,  # Steepest Receiver (flat index)
		Sdx,         # Distance to steepest receiver
		Z,           # Topography (edited in place)
		A,           # Drainage area, or Discharge
		nx,          # Number of columns
		ny,          # Number of rows
		dx,          # Spatial step
		BCs,         # Boundary codes
		K,           # Erodability (variable per node)
		m,           # Area exponent 
		n,           # Slope exponent
		dt,          # Time Step
	):
	"""
	Numba-optimized implementation of a single-flow direction (SFD) Stream Power Law (SPL) model
	with spatially variable erodability.

	This function simulates topographic evolution by iteratively adjusting elevation based on
	drainage area and local slope, allowing for a different erodability coefficient (`K`) at each node.

	Args:
		Stack (numpy.ndarray): 1D array of node indices, topologically ordered from downstream to upstream.
		Sreceivers (numpy.ndarray): 1D array where each element is the flat index of the steepest receiver for the corresponding node.
		Sdx (numpy.ndarray): 1D array where each element is the distance to the steepest receiver for the corresponding node.
		Z (numpy.ndarray): 1D array of topographic elevations (modified in-place).
		A (numpy.ndarray): 1D array of drainage area or discharge values for each node.
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.
		dx (float): Spatial step size.
		BCs (numpy.ndarray): 1D array of boundary condition codes.
		K (numpy.ndarray): 1D array of erodability coefficients, one for each node.
		m (float): Area exponent in the Stream Power Law.
		n (float): Slope exponent in the Stream Power Law.
		dt (float): Time step for the simulation.

	Returns:
		None: The `Z` array is modified in-place.

	Author: B.G. (last modifications 09/2024)
	"""
	
	# Traverse the nodes in ascending order (downstream to upstream)
	for node in Stack:

		# Ignore outlet nodes, inactive nodes, or nodes that are their own receiver (e.g., pits)
		if node == Sreceivers[node] or scb.ste.can_out_flat(node,BCs) or scb.ste.is_active_flat(node,BCs) == False:
			continue

		# Calculate the local slope (dz/dx) to the steepest receiver
		dzdx = (Z[node] - Z[Sreceivers[node]])/Sdx[node]

		# If the slope is non-positive (uphill or flat), slightly raise the current node's elevation
		# to ensure a positive slope for erosion calculation and recompute dzdx.
		if(dzdx<=0):
			Z[node] = Z[Sreceivers[node]] + 1e-4
			dzdx = 1e-4/dx

		# Calculate the erosion factor based on K (at current node), dt, drainage area, and distance to receiver
		factor = K[node] * dt * (A[node])**m / (Sdx[node]**n);

		# Initialize variables for iterative elevation adjustment
		ielevation = Z[node];
		irec_elevation = Z[Sreceivers[node]];
		elevation_k = ielevation;
		elevation_prev = Z[node] + 500; # Set to a value far from current to ensure first iteration
		tolerance = 1e-5;

		# Iteratively adjust the elevation until convergence
		while (abs(elevation_k - elevation_prev) > tolerance) :
			elevation_prev = elevation_k
			slope = max(elevation_k - irec_elevation, 1e-6) # Ensure positive slope for calculation
			diff = (elevation_k - ielevation + factor * slope**n) / (1. + factor * n * slope**(n - 1))
			elevation_k -= diff; # Update elevation
		
		Z[node] = elevation_k # Set the final adjusted elevation


def run_SPL_on_topo(
		dem,         # Topography (edited in place)
		BCs = None,  # Boundary codes
		graph = None,# Pre-computed flow graph
		K = 1e-5,    # Erodability
		m = 0.45,    # Area exponent 
		n = 1.11,    # Slope exponent
		dt = 1e3,    # Time Step	
	):
	"""
	Runs a Stream Power Law (SPL) simulation on a given topography.

	This function orchestrates the SPL simulation, handling the creation or update of the
	flow graph, computation of drainage area, and application of the SPL erosion/deposition
	model using either a single or spatially variable erodability coefficient.

	Args:
		dem (scabbard.raster.RegularRasterGrid or similar): The digital elevation model object.
													Its `Z` attribute (or `Z2D` for older versions) will be modified in-place.
		BCs (numpy.ndarray, optional): 2D NumPy array of boundary condition codes. If None,
										normal boundary conditions are assumed. Defaults to None.
		graph (scabbard.flow.SFGraph, optional): An existing flow graph object. If None,
												a new one will be created. Defaults to None.
		K (float or numpy.ndarray, optional): Erodability coefficient. Can be a single float
											for uniform erodability or a 2D NumPy array for spatially variable erodability.
											Defaults to 1e-5.
		m (float, optional): Area exponent in the Stream Power Law. Defaults to 0.45.
		n (float, optional): Slope exponent in the Stream Power Law. Defaults to 1.11.
		dt (float, optional): Time step for the simulation. Defaults to 1e3.

	Returns:
		None: The `dem.Z` (or `dem.Z2D`) array is modified in-place.

	Author: B.G. (last modifications 09/2024)
	"""
	dx,nx,ny = dem.geo.dxnxny # Extract grid dimensions and spatial step from DEM object

	# If boundary conditions are not provided, assume normal boundary conditions
	if BCs is None:
		BCs = scb.ut.normal_BCs_from_shape(nx, ny, out_code = 3)

	# Create or update the flow graph
	if(graph is None):
		# Create a new SFGraph if not provided, with D4 flow routing and local minima filling
		graph = scb.flow.SFGraph(dem, BCs = BCs, D4 = False, dx = 1., backend = 'ttb', fill_LM = True, step_fill = 1e-3)
	else:
		# Update existing graph with new DEM and BCs
		graph.update(dem, BCs = BCs, fill_LM = True, step_fill = 1e-3)

	# Compute drainage area (or discharge) for the current topography
	A = scb.flow.drainage_area(graph)


	# Run the appropriate SPL implementation based on whether K is a scalar or an array
	if(isinstance(K,np.ndarray)):
		# Use variable K implementation if K is a NumPy array
		_impl_spl_SFD_variable(graph.Stack, graph.Sreceivers, graph.Sdx, dem.Z.ravel(), A.ravel(), nx, ny, dx, BCs.ravel(), K.ravel(), m, n, dt)
	else:
		# Use single K implementation if K is a scalar
		_impl_spl_SFD_single(graph.Stack, graph.Sreceivers, graph.Sdx, dem.Z.ravel(), A.ravel(), nx, ny, dx, BCs.ravel(), K, m, n, dt)
	

