'''
High-level interface to graphflood, starting point to run the model to other backend
'''

import numpy as np
import scabbard as scb
import dagger as dag
import taichi as ti

def _std_run_gpu_ndt(
	grid,
	P = None, # precipitations, numpy array or scalar
	BCs = None, # Boundary codes
	N_dt = 5000,
	dt = 1e-3,
	manning = 0.033,
	init_hw = None,

):
	'''Internal runner function for the standalone gpu runner (Runs N iteration from "scratch")'''
	try:
		ti.init(ti.gpu)
	except:
		raise RuntimeError('Could not initiate taichi to GPU')

	# Creating the parameter file
	param = scb.rvd.param_from_grid(grid)
	param.manning = manning

	# Feeding in initial hw
	if(init_hw is not None):
		param.initial_hw = init_hw

	# Feeding the BCs (if not default to flow everywhere, and can out from the edges)
	if(BCs is not None):
		param.BCs = BCs

	# Simulation time steps
	param.dt_hydro = dt

	# Field of precipitation inputs
	param.precipitations = P

	# COmpiling param to model object
	rd = scb.rvd.create_from_params(param)

	# Running the N time steps
	rd.run_hydro(N_dt)

	# Return metrics
	return {'h':rd.hw.to_numpy(), 'Qi': rd.QwA.to_numpy(), 'Qo':rd.QwC.to_numpy(), 'model':rd, 'param':param, 'backend_graphflood': 'riverdale'}


def _std_rerun_gpu_ndt(
	model_input,
	N_dt = 1000,
	new_dt = None):

	if(new_dt is not None):
		scb.rvd.change_dt_hydro(model_input['model'], model_input['param'], new_dt)

	rd = model_input['model']
	# Running the N time steps
	rd.run_hydro(N_dt)

	# Return metrics
	return {'h':rd.hw.to_numpy(), 'Qi': rd.QwA.to_numpy(), 'Qo':rd.QwC.to_numpy(), 'model':rd, 'param':param, 'backend_graphflood': 'riverdale'}

def _std_run_dagger_ndt(
	grid,
	P = None, # precipitations, numpy array or scalar
	BCs = None,
	N_dt = 5000,
	dt = 1e-3,
	manning = 0.033,
	init_hw = None,

	**kwargs

	):
	'''
		Internal function managing the DAGGER interface
	'''

	# init the DAGGER engine (connector and graph fix the topology of the grid)
	con = dag.D8N(grid.geo.nx, grid.geo.ny, grid.geo.dx, grid.geo.dx, grid.geo.xmin, grid.geo.ymin)
	graph = dag.graph(con)

	# Eventual custom boundary conditions
	if(BCs is not None):
		con.set_custom_boundaries(BCs.ravel())

	# Initialising the c++ graphflood object
	flood = dag.graphflood(graph, con)

	# feeding the topo
	flood.set_topo(grid.Z.ravel())
	
	# Setting the topology
	setsfd = False 
	if ('SFD' in kwargs.keys()):
		if(kwargs['SFD']):
			setsfd = True
	flood.enable_SFD() if setsfd else flood.enable_MFD() 

	# Legacy options, ignore
	flood.fill_minima()
	flood.disable_courant_dt_hydro()

	# Setting the time step
	flood.set_dt_hydro(dt)

	flood.set_mannings(manning)

	if(isinstance(P, np.ndarray)):
		flood.set_water_input_by_variable_precipitation_rate(P.ravel())
	else:
		flood.set_water_input_by_constant_precipitation_rate(P)

	if(init_hw is not None):
		flood.set_hw(init_hw.ravel())

	for i in range(N_dt):
		flood.run()

	return {'h':flood.get_hw().reshape(grid.rshp), 'Qi': flood.get_Qwin().reshape(grid.rshp), 
		'Qo':flood.compute_tuqQ(3).reshape(grid.rshp), 'model':flood, 'param':None, 'backend_graphflood': 'dagger'}


def _std_run_ttb_ndt(
	grid,
	P = None, # precipitations, numpy array or scalar
	BCs = None,
	N_dt = 5000,
	dt = 1e-3,
	manning = 0.033,
	init_hw = None,

	**kwargs
	):
	
	import topotoolbox as ttb

	ttbgrid = grid.grid2ttb()

	sfd = False
	if('SFD' in kwargs.keys()):
		if(kwargs['SFD'] == True):
			sfd = True

	D8 = True
	if('D4' in kwargs.keys()):
		if(kwargs['D4'] == True):
			D8 = False
	
	res = ttb.run_graphflood(
		ttbgrid,
		initial_hw=init_hw,
		bcs=BCs,
		dt=dt,
		p=P,
		manning=manning,
		sfd=sfd,
		d8=D8,
		n_iterations=N_dt
	)

	return {'h': scb.raster.raster_from_ttb(res),'Qi': None,'Qo':None, 'model':None, 'param':None, 'backend_graphflood': 'ttb'}

def std_run(
	model_input, # Sting or grid so far
	P = 1e-5, # precipitations, numpy array or scalar
	BCs = None, # Boundary codes
	N_dt = 5000,
	backend = 'gpu',
	dt = 1e-3,
	manning = 0.033,
	init_hw = None,
	**kwargs
):

	if(isinstance(model_input, str)):
		grid = scb.io.load_raster(model_input)
	else:
		grid = model_input
	
	if(backend.lower() == 'gpu'):
		return _std_run_gpu_ndt(grid, P, BCs, N_dt, dt, manning, init_hw = init_hw, **kwargs)
	elif(backend.lower() in [ 'dagger' , 'cpu']):
		return _std_run_dagger_ndt(grid, P,	BCs, N_dt, dt, manning, init_hw, **kwargs)
	elif(backend.lower() in ['ttb', 'topotoolbox']):
		return _std_run_ttb_ndt(grid, P, BCs, N_dt, dt, manning, init_hw, **kwargs)



def std_rerun(model_input, N_dt = 1000, new_dt = None):
	'''function to run further dts from an existing model output
	'''

	# Checking if I can run these

	if(isinstance(model_input,dict) == False):
		raise RuntimeError(f'std_rerun takes an existing ouput from std_run to run it further, not a {type(model_input)}.')
	elif('backend_graphflood' not in model_input):
		raise RuntimeError(f'std_rerun takes an existing ouput from std_run to run it further, not a {type(model_input)}.')

	# Else starting again:
