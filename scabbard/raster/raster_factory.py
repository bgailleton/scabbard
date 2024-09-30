'''
Contains routines to create raster grids from scratch.

For example white noise, perlin noise or spectral stuff for initial conditions,
or constant slope raster to test river LEMs, ...

B.G. - created 09/2024
'''

import scabbard as scb
import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import numba as nb
import time


def slope2D_S(
	nx = 256, 
	ny = 512, 
	dx = 2.,
	z_base = 0,
	slope = 1e-3, 
	):
	'''
		Generates a slopping grid adn initialise an env with given boundaries to it. 
		It comes with a connector and a graph (todo) alredy pre-prepared for boundaries
	'''

	# Zeros Topo
	Z = np.zeros((ny, nx), dtype = np.float32)

	lx = (nx+1) * dx
	ly = (ny+1) * dx

	grid = scb.raster.raster_from_array(Z, dx=dx, xmin=0.0, ymin=0.0, dtype=np.float32)

	XX,YY = grid.geo.XY
	
	Z = (YY * slope)[::-1]
	grid.Z[:,:] = Z[:,:]

	BCs = np.ones((ny,nx),dtype = np.uint8)
	BCs[:,[0,-1]] = 0
	BCs[0,:] = 7
	BCs[-1,:] = 3

	return grid, BCs