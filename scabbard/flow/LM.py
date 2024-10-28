'''
This script contains set of routines to manage local minima using different methods and/or libraries


B.G
'''

import numpy as np
import scabbard as scb
import scabbard._utils as ut
import scabbard.flow.graph as gr
import dagger as dag
import scabbard.flow._LM as lmn
import topotoolbox as ttb


def _priority_flood_from_Z(Z, BCs, D4, in_place, dx, gridcpp, backend, step_fill):

	if(in_place):
		tZ = Z
	else:
		tZ = Z.copy()

	if(BCs is None):
		BCs = ut.normal_BCs_from_shape(Z.shape[1], Z.shape[0])

	if(backend == 'ttb'):
		dims = np.array( [Z.shape[0], Z.shape[1]],dtype = np.uint64)
		# ttb.graphflood.funcdict['priority_flood'](Z.ravel(), BCs.ravel(), dims, not D4)
		ttb.graphflood.funcdict['priority_flood_TO'](Z.ravel(), np.zeros_like(Z, dtype = np.uint64), BCs.ravel(), dims, not D4, step_fill)

	elif backend == 'dagger':
		if(gridcpp is None):
			gridcpp = dag.GridCPP_f32(Z.shape[1],Z.shape[0], dx, dx,3)
		dag._PriorityFlood_D4_f32(tZ, gridcpp, BCs, 1e-4)

	if(in_place == False):
		return tZ

def _priority_flood_from_dem(dem, BCs, D4, in_place, dx, gridcpp, backend, step_fill):

	if(in_place):
		tZ = dem.Z
	else:
		# print("COPY")
		tZ = dem.Z.copy()

	if(BCs is None):
		BCs = ut.normal_BCs_from_shape(tZ.shape[0], tZ.shape[1])

	if(backend == 'ttb'):
		dims = np.array( [tZ.shape[0], tZ.shape[1]],dtype = np.uint64)
		ttb.graphflood.funcdict['priority_flood'](tZ.ravel(), BCs.ravel(), dem.dims, not D4, step_fill)
		# ttb.graphflood.funcdict['priority_flood_TO'](tZ.ravel(), np.zeros_like(tZ, dtype = np.uint64), BCs.ravel(), dims, not D4, step_fill)

	elif backend == 'dagger':
		if(gridcpp is None):
			gridcpp = dag.GridCPP_f32(tZ.shape[1],tZ.shape[0], dx, dx,3)
		dag._PriorityFlood_D4_f32(tZ, gridcpp, BCs, 1e-4)

	if(in_place == False):
		return dem.duplicate_with_other_data(tZ)
	else:
		dem.Z[:,:] = tZ[:,:]

def priority_flood(Z, BCs = None, D4 = True, in_place = True, dx = 1., gridcpp = None, backend = 'ttb', step_fill = 1e-3):
	'''
	perform priority flood + slope on a 2D DEM
	'''

	if(isinstance(Z,np.ndarray)):
		return _priority_flood_from_Z(Z, BCs, D4, in_place, dx, gridcpp, backend, step_fill)
	elif(isinstance(Z,scb.raster.RegularRasterGrid)):
		print("FILLING")
		return _priority_flood_from_dem(Z, BCs, D4, in_place, dx, gridcpp, backend, step_fill)

	

def legacy_break_bridges(grid, in_place = False, BCs = None, step_fill = 1e-3):
	'''
	Experimental function to break bridges and local minimas in a general way
	
	argument:
		- grid: A Rgrid object (TODO adapt to new grid systems)

	B.G.
	'''

	Z = grid.Z2D.copy()

	if(BCs is None):
		BCs = ut.normal_BCs_from_shape(grid.nx,grid.ny)

	gridcpp = dag.GridCPP_f32(Z.shape[1],Z.shape[0], grid.dx, grid.dx, 3)

	# first filling the topo
	filled_Z = priority_flood(Z, BCs = BCs, in_place = False, gridcpp = gridcpp, dx = grid.dx, step_fill = step_fill)

	# COmputing a first graph
	sgf = gr.SFGraph(filled_Z, BCs = None, D4 = True, dx = grid.dx)

	lmn.impose_downstream_minimum_elevation_decrease(Z.ravel(), sgf.Stack, sgf.Sreceivers.ravel(), delta = step_fill)

	return Z
	


def break_bridges(grid, in_place = False, BCs = None, step_fill = 1e-3):

	if(isinstance(grid,scb.raster.RegularRasterGrid) == False):
		return legacy_break_bridges(grid, in_place = False, BCs = None, step_fill = 1e-3)


	Z = grid.Z.copy() if not in_place else grid.Z

	if(BCs is None):
		BCs = ut.normal_BCs_from_shape(grid.nx,grid.ny)


	# first filling the topo
	filled_Z = priority_flood(Z, BCs = BCs, in_place = False, dx = grid.geo.dx, step_fill = step_fill)

	# COmputing a first graph
	sgf = gr.SFGraph(filled_Z, BCs = None, D4 = True, dx = grid.geo.dx)

	lmn.impose_downstream_minimum_elevation_decrease(Z.ravel(), sgf.Stack, sgf.Sreceivers.ravel(), delta = step_fill)

	if(in_place == False):
		return Z








