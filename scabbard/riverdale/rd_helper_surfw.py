'''
Set of internal functions to help computing gradients/surfaces from the combination of multiple fields/data
Some functions will be redundant but named differently for user friendliness 

B.G. - 29/04/2024
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
	'''
	Internal helping function returning the hydrayulic surface (elevation of the water surface)
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- i,j: the row col indices
	Returns:
		- the hydraulic surface
	Authors:
		- B.G. (last modification 30/04/2024)
	'''

	return Z[i,j] + ti.max(0.,hw[i,j])


@ti.func
def Zw_drape(Z: ti.template(), hw: ti.template(), i:ti.i32, j:ti.i32) -> ti.f32:
	'''
	Internal helping function returning the hydrayulic surface (elevation of the water surface)
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- i,j: the row col indices
	Returns:
		- the hydraulic surface
	Authors:
		- B.G. (last modification 30/04/2024)
	'''

	return Z[i,j] + hw[i,j]



###############################################################
############# Gradients #######################################
###############################################################

@ti.func
def Sw(Z: ti.template(), hw: ti.template(), i:ti.template(), j:ti.template(), ir:ti.template(), jr:ti.template())->ti.f32:
	'''
	Internal helping function returning the hydrayulic slope
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- i,j: the row col indices
		- ir,jr: the row col indices of the receivers node
	Returns:
		- the hydraulic surface
	Authors:
		- B.G. (last modification 20/05/2024)
	'''
	return (Zw(Z,hw, i,j) - Zw(Z,hw, ir,jr))/GRID.dx

@ti.func
def Sz(Z: ti.template(), i:ti.template(), j:ti.template(), ir:ti.template(), jr:ti.template())->ti.f32:
	'''
	Internal helping function returning the topographic slope
	Arguments:
		- Z: a 2D field of topographic elevation
		- i,j: the row col indices
		- ir,jr: the row col indices of the receivers node
	Returns:
		- the hydraulic surface
	Authors:
		- B.G. (last modification 20/05/2024)
	'''
	return (Z[i,j] - Z[ir,jr])/GRID.dx


@ti.func
def hydraulic_gradient_value(Z:ti.template(), hw:ti.template(),BCs:ti.template(),i:ti.i32, j:ti.i32 ) -> ti.f32:
	'''
	Calculates the local hydraulic gradient value.
	gradient = sqrt(max_slope_in_X^2 + max_slope_in_y^2)
	Convenient function, but only optimised for cases where calculating the hydraulic gradient value is the only neighbouring operation.

	Arguments:
		- Z: the field of topographic elevation
		- hw: the field of flow depth
		- BCs: the field of boundary conditions
		- i: the row index of the node in question
		- j: the column index of the node in question
	Returns:
		- the value of the steepest slope, 0 if no downslope neighbours
	Authors:
		- B.G. (last modification: 06/2024)
	'''


	# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
	SSx = 0.
	SSy = 0.
	gradSw = 0.

	# Traversing Neighbours
	for k in range(4):

		# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
		ir,jr = gridfuncs.neighbours(i,j,k, BCs)

		# if not a neighbours, by convention is < 0 and I pass
		if(ir == -1):
			continue

		if(gridfuncs.can_receive(ir,jr, BCs) == False):
			continue

		# Local hydraulic slope
		tS = Sw(Z,hw,i,j,ir,jr)

		# If < 0, neighbour is a donor and I am not interested
		if(tS <= 0):
			continue

		# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
		if(k == 0 or k == 3):
			if(tS > SSy):
				SSy = tS
		else:
			if(tS > SSx):
				SSx = tS

		# Done with processing this particular neighbour


	# Calculating local norm for the gradient
	# The condition manages the boundary conditions
	gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)

	return gradSw




