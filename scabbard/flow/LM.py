'''
This script contains set of routines to manage local minima using different methods and/or libraries


B.G
'''

import numpy as np
import scabbard._utils as ut
import scabbard.flow.graph as gr
import dagger as dag
import scabbard.flow._LM as lmn


def priority_flood(Z, BCs = None, D4 = True, in_place = True, dx = 1., gridcpp = None):
	'''
	perform priority flood + slope on a 2D DEM
	'''

	if(in_place):
		tZ = Z
	else:
		tZ = Z.copy()


	if(gridcpp is None):
		gridcpp = dag.GridCPP_f32(Z.shape[1],Z.shape[0], dx, dx,3)

	if(BCs is None):
		BCs = ut.normal_BCs_from_shape(Z.shape[1], Z.shape[0])

	dag._PriorityFlood_D4_f32(tZ, gridcpp, BCs, 1e-4)

	if(in_place == False):
		return tZ


def break_bridges(grid, in_place = False, BCs = None):
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
	filled_Z = priority_flood(Z, BCs = BCs, in_place = False, gridcpp = gridcpp, dx = grid.dx)

	# COmputing a first graph
	sgf = gr.SFGraph(filled_Z, BCs = None, D4 = True, dx = grid.dx)

	lmn.impose_downstream_minimum_elevation_decrease(Z.ravel(), sgf.Stack, sgf.Sreceivers.ravel(), delta = 1e-4)

	return Z
	













