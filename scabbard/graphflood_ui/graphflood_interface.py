'''
High-level interface to graphflood, starting point to run the model to other backend
'''

import numpy as np
import scabbard as scb
import dagger as dag
import taichi as ti

def _std_run_gpu(
	grid,
	P = None, # precipitations, numpy array or scalar
	BCs = None, # Boundary codes
	N_dt = 5000,
	dt = 1e-3

):
	ti.init(ti.gpu)

	param = scb.rvd.param_from_grid(grid)
	if(BCs is not None):
		param.BCs = BCs

	param.dt_hydro = dt

	param.precipitations = P
	# print('Trying to create')
	rd = scb.rvd.create_from_params(param)

	# print('up to here')

	# for i in range(N_dt):
	rd.run_hydro(N_dt)

	return {'hw':rd.hw.to_numpy()}

def std_run(
	inputte, # Sting or grid so far
	P = 1e-5, # precipitations, numpy array or scalar
	BCs = None, # Boundary codes
	N_dt = 5000,
	backend = 'gpu',
	dt = 1e-3
):		
	
	if(isinstance(inputte, scb.raster.RegularRasterGrid) == False):
		grid = scb.io.load_raster(inputte)
	else:
		grid = inputte
	
	if(backend.lower() == 'gpu'):
		return _std_run_gpu(grid, P, BCs, N_dt, dt)

