'''
Help with managing boundary conditions
Weither it is about automatically generating them or manage more complex cases like no data and all


B.G.
'''

import numpy as np
import scabbard._utils as ut
import scabbard.flow._bc_helper as bch
from scabbard.flow.graph import SFGraph
import dagger as dag
from functools import reduce
from collections.abc import Iterable

def get_normal_BCs(nx,ny):
	'''
	Returns an array of boundary conditions with "normal edges"
	Flow can out at each edge

	parameters:
		- nx: number of columns
		- ny: number of rows
	
	Authors:
	- B.G (alst modification: 07/2024)
	'''
	return ut.normal_BCs_from_shape(nx,ny)

def combine_masks(*args):
	return reduce(np.bitwise_and, args) if all(isinstance(arr, np.ndarray) and arr.dtype in [np.uint8, np.bool_] for arr in args) else None


def mask_to_BCs(grid, mask):
	# Preprocessing the boundary conditions
	gridcpp = dag.GridCPP_f32(grid.nx,grid.ny,grid.dx,grid.dx,3)
	BCs = np.ones_like(grid.Z2D, dtype = np.uint8)
	dag.mask_to_BCs_f32(gridcpp, mask, BCs, False)

	BCs[[0,-1],:] = 3
	BCs[:, [0,-1]] = 3

	return BCs


def mask_seas(grid, sea_level = 0., extra_mask = None):

	mask = np.ones_like(grid.Z2D, dtype = np.uint8)
	mask[grid.Z2D < sea_level] = 0
	
	if (extra_mask is None):
		return mask
	else:
		return (extra_mask & mask)

def mask_single_watershed_from_outlet(grid, location, BCs = None, extra_mask = None, MFD = True, stg = None):

	# Checks if the input is flat index or rows col
	if(isinstance(location, Iterable) and not isinstance(location, (str, bytes))):
		row,col = location
		index = row * grid.nx + col
	else:
		row,col = index // grid.nx, index % grid.nx

	
	if(BCs is None):
		BCs = get_normal_BCs(grid.nx,grid.ny)
	gridcpp = dag.GridCPP_f32(grid.nx,grid.ny,grid.dx,grid.dx,3)

	if(MFD):
		mask = np.zeros_like(grid.Z2D,dtype = np.uint8)
		dag.mask_upstream_MFD_f32(gridcpp, mask, grid.Z2D, BCs, row, col)

	else:
		if(stg is None):
			stg = SFGraph(grid.Z2D, BCs = BCs, D4 = True, dx = 1.)
			
		mask = bch.mask_watershed_SFD(index, stg.Stack, stg.Sreceivers).reshape(grid.rshp)


	return mask if extra_mask is None else combine_masks(mask,extra_mask)


def remove_seas(grid, sea_level = 0., extra_mask = None):

	# Preprocessing the boundary conditions
	gridcpp = dag.GridCPP_f32(grid.nx,grid.ny,grid.dx,grid.dx,3)

	BCs = np.ones_like(grid.Z2D, dtype = np.uint8)
	
	mask = mask_seas(gridcpp, sea_level, extra_mask)
	
	if (extra_mask is None):
		mask = combine_masks(mask, extra_mask)

	return mask_to_BCs(grid,mask)