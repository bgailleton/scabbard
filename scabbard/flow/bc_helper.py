'''
Help with managing boundary conditions
Weither it is about automatically generating them or manage more complex cases like no data and all


B.G.
'''

import numpy as np
import scabbard._utils as ut
import scabbard.flow._bc_helper as bch
from scabbard.flow.graph import SFGraph
import scabbard as scb
import dagger as dag
import numba as nb
from functools import reduce
from collections.abc import Iterable


def get_normal_BCs(dem):
	'''
	Returns an array of boundary conditions with "normal edges"
	Flow can out at each edge

	parameters:
		- can be a 2D numpy array or a raster object
	
	Authors:
	- B.G (alst modification: 07/2024)
	'''
	if(isinstance(dem,np.ndarray)):
		return ut.normal_BCs_from_shape(dem.shape[1], dem.shape[0])
	elif(isinstance(dem,scb.raster.RegularRasterGrid)):
		return ut.normal_BCs_from_shape(dem.geo.nx, dem.geo.ny)
	else:
		raise TypeError('boundary conditions can only be obtained from a RegularRasterGrid or a 2D numpy array')

def combine_masks(*args):
	return reduce(np.bitwise_and, args) if all(isinstance(arr, np.ndarray) and arr.dtype in [np.uint8, np.bool_] for arr in args) else None

@nb.njit()
def _mask_to_BCs(BCs, mask, nx, ny):
	'''
	Internal numba function helping with converting a mask of partial nodata to BCs codes

	Arguments:
		- BCs: the 2D array of BCs
		- mask: the binary mask  of reference
		- nx and ny: the number of cols and rows
	Returns:
		- Nothing, edits in place

	Author:
		- B.G. (last modification: 09/2024 for MTJ project)
	'''

	# First pass to mark nodata
	for i in range(ny):
		for j in range(nx):
			if mask[i,j] == 0:
				BCs[i,j] = 0

	# second to find outlet
	for i in range(ny):
		for j in range(nx):

			if(BCs[i,j] == 1):
				for k in range(4):
					ir,jr = scb.ste.neighbours_D4(i,j,k,BCs,nx,ny)
					if(ir == -1):
						BCs[i,j] = 3
				


def mask_to_BCs(grid, mask):
	'''
	Converts a mask of partial nodata to BCs codes

	Arguments:
		- grid: raster grid
		- mask: the binary mask of reference
	Returns:
		- The 2D array of boundary codes

	Author:
		- B.G. (last modification: 09/2024 for MTJ project)
	'''

	tgrid = grid.Z if isinstance(grid, scb.raster.RegularRasterGrid) else grid.Z2D
	ny,nx = (grid.geo.ny, grid.geo.nx) if isinstance(grid, scb.raster.RegularRasterGrid) else (grid.ny, grid.nx)
	dx = grid.geo.dx if isinstance(grid, scb.raster.RegularRasterGrid) else grid.dx

	# Preprocessing the boundary conditions
	gridcpp = dag.GridCPP_f32(nx,ny,dx,dx,3)
	BCs = np.ones_like(tgrid, dtype = np.uint8)
	# dag.mask_to_BCs_f32(gridcpp, mask, BCs, False)
	_mask_to_BCs(BCs,mask,nx,ny)

	BCs[[0,-1],:] = 3
	BCs[:, [0,-1]] = 3

	return BCs


def mask_seas(grid, sea_level = 0., extra_mask = None):
	'''
	Returns or append a mask with 0s where  
	'''
	
	tgrid = grid.Z if isinstance(grid, scb.raster.RegularRasterGrid) else grid.Z2D

	mask = np.ones_like(tgrid, dtype = np.uint8)
	mask[tgrid < sea_level] = 0
	
	if (extra_mask is None):
		return mask
	else:
		return (extra_mask & mask)

def mask_single_watershed_from_outlet(grid, location, BCs = None, extra_mask = None, MFD = True, stg = None):

	tgrid = grid.Z if isinstance(grid, scb.raster.RegularRasterGrid) else grid.Z2D
	ny,nx = (grid.geo.ny, grid.geo.nx) if isinstance(grid, scb.raster.RegularRasterGrid) else (grid.ny, grid.nx)
	dx = grid.geo.dx if isinstance(grid, scb.raster.RegularRasterGrid) else grid.dx

	# Checks if the input is flat index or rows col
	if(isinstance(location, Iterable) and not isinstance(location, (str, bytes))):
		row,col = location
		index = row * nx + col
	else:
		index = location
		row,col = index // nx, index % nx

	
	if(BCs is None):
		BCs = get_normal_BCs(nx,ny)
	gridcpp = dag.GridCPP_f32(nx,ny,dx,dx,3)

	if(MFD):
		mask = np.zeros_like(grid.Z2D,dtype = np.uint8)
		dag.mask_upstream_MFD_f32(gridcpp, mask, tgrid, BCs, row, col)

	else:
		if(stg is None):
			stg = SFGraph(tgrid, BCs = BCs, D4 = True, dx = 1.)
			
		mask = bch.mask_watershed_SFD(index, stg.Stack, stg.Sreceivers).reshape((ny,nx))


	return mask if extra_mask is None else combine_masks(mask,extra_mask)


def remove_seas(grid, sea_level = 0., extra_mask = None):

	# Preprocessing the boundary conditions
	gridcpp = dag.GridCPP_f32(grid.nx,grid.ny,grid.dx,grid.dx,3)

	BCs = np.ones_like(grid.Z2D, dtype = np.uint8)
	
	mask = mask_seas(gridcpp, sea_level, extra_mask)
	
	if (extra_mask is None):
		mask = combine_masks(mask, extra_mask)

	return mask_to_BCs(grid,mask)