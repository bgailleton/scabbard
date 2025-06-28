'''
This module provides a set of internal helper functions for computing gradients and surfaces
from combinations of multiple fields/data within the Riverdale model.
Some functions may appear redundant but are named differently for clarity and user-friendliness.

Author: B.G. (last modification: 29/04/2024)
'''

import taichi as ti
import numpy as np
from scabbard.riverdale.rd_grid import GRID
import scabbard.riverdale.rd_grid as gridfuncs


###############################################################
############# Surfaces ########################################
###############################################################


@ti.func
def Zw(Z: ti.template(), hw: ti.template(), i:ti.i32, j:ti.i32) -> ti.f32:
	"""
	Internal helper function returning the hydraulic surface elevation (elevation of the water surface).

	Args:
		Z (ti.template()): A 2D Taichi field representing the topographic elevation.
		hw (ti.template()): A 2D Taichi field representing the flow depth.
		i (ti.i32): The row index of the grid cell.
		j (ti.i32): The column index of the grid cell.

	Returns:
		ti.f32: The hydraulic surface elevation (Z + max(0, hw)).

	Author: B.G. (last modification 30/04/2024)
	"""

	return Z[i,j] + ti.max(0.,hw[i,j])


@ti.func
def Zw_drape(Z: ti.template(), hw: ti.template(), i:ti.i32, j:ti.i32) -> ti.f32:
	"""
	Internal helper function returning the hydraulic surface elevation for draped conditions.

	This function is similar to `Zw` but does not clamp `hw` to be non-negative,
	which is useful for scenarios where `hw` can represent a negative value (e.g., for pits).

	Args:
		Z (ti.template()): A 2D Taichi field representing the topographic elevation.
		hw (ti.template()): A 2D Taichi field representing the flow depth (can be negative).
		i (ti.i32): The row index of the grid cell.
		j (ti.i32): The column index of the grid cell.

	Returns:
		ti.f32: The hydraulic surface elevation (Z + hw).

	Author: B.G. (last modification 30/04/2024)
	"""

	return Z[i,j] + hw[i,j]



###############################################################
############# Gradients #######################################
###############################################################

@ti.func
def Sw(Z: ti.template(), hw: ti.template(), i:ti.template(), j:ti.template(), ir:ti.template(), jr:ti.template())->ti.f32:
	"""
	Internal helper function returning the hydraulic slope between two cells.

	Args:
		Z (ti.template()): A 2D Taichi field representing the topographic elevation.
		hw (ti.template()): A 2D Taichi field representing the flow depth.
		i (ti.template()): The row index of the current cell.
		j (ti.template()): The column index of the current cell.
		ir (ti.template()): The row index of the receiver cell.
		jr (ti.template()): The column index of the receiver cell.

	Returns:
		ti.f32: The hydraulic slope.

	Author: B.G. (last modification 20/05/2024)
	"""
	return (Zw(Z,hw, i,j) - Zw(Z,hw, ir,jr))/GRID.dx

@ti.func
def Sz(Z: ti.template(), i:ti.template(), j:ti.template(), ir:ti.template(), jr:ti.template())->ti.f32:
	"""
	Internal helper function returning the topographic slope between two cells.

	Args:
		Z (ti.template()): A 2D Taichi field representing the topographic elevation.
		i (ti.template()): The row index of the current cell.
		j (ti.template()): The column index of the current cell.
		ir (ti.template()): The row index of the receiver cell.
		jr (ti.template()): The column index of the receiver cell.

	Returns:
		ti.f32: The topographic slope.

	Author: B.G. (last modification 20/05/2024)
	"""
	return (Z[i,j] - Z[ir,jr])/GRID.dx


@ti.func
def hydraulic_gradient_value(Z:ti.template(), hw:ti.template(),BCs:ti.template(),i:ti.i32, j:ti.i32 ) -> ti.f32:
	"""
	Calculates the local hydraulic gradient value at a given grid cell.

	The gradient is computed as the square root of the sum of the squares of the
	steepest slopes in the x and y directions (gradient = sqrt(max_slope_in_X^2 + max_slope_in_y^2)).
	This function is optimized for cases where only the hydraulic gradient value is needed.

	Args:
		Z (ti.template()): The Taichi field representing topographic elevation.
		hw (ti.template()): The Taichi field representing flow depth.
		BCs (ti.template()): The Taichi field representing boundary conditions.
		i (ti.i32): The row index of the node in question.
		j (ti.i32): The column index of the node in question.

	Returns:
		ti.f32: The value of the steepest hydraulic gradient, or 0 if no downslope neighbors exist.

	Author: B.G. (last modification: 06/2024)
	"""


	# Initialize steepest slopes in x and y directions
	SSx = 0.
	SSy = 0.
	gradSw = 0.

	# Iterate over neighbors (D4 directions)
	for k in range(4):

		# Get neighbor coordinates
		ir,jr = gridfuncs.neighbours(i,j,k, BCs)

		# Skip if not a valid neighbor (e.g., outside boundaries)
		if(ir == -1):
			continue

		# Skip if the neighbor cannot receive flow
		if(gridfuncs.can_receive(ir,jr, BCs) == False):
			continue

		# Calculate local hydraulic slope to the neighbor
		tS = Sw(Z,hw,i,j,ir,jr)

		# If slope is non-positive, neighbor is a donor or at same elevation, so skip
		if(tS <= 0):
			continue

		# Register the steepest slope in the appropriate direction (y for k=0 or k=3, x for k=1 or k=2)
		if(k == 0 or k == 3):
			if(tS > SSy):
				SSy = tS
		else:
			if(tS > SSx):
				SSx = tS

		# Done with processing this particular neighbor


	# Calculate the magnitude of the hydraulic gradient
	# The condition manages the boundary conditions
	gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)

	return gradSw




