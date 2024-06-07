'''
Set of internal functions to help computing gradients/surfaces from the combination of multiple fields/data
Some functions will be redundant but named differently for user friendliness 

B.G. - 29/04/2024
'''

import taichi as ti
import numpy as np
from scabbard.riverdale.rd_grid import GRID


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

