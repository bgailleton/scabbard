'''
Riverdale environment:

This is the main high level class interfacing with the model. 
This is the API users use to get the data out of the model and run time steps.
The other high level class class managing the model inputs and parameters is the RDParams class (see rd_params.py) 
The (internal) structure is a bit convoluted for the sake of optimising calculation time.

'''

import taichi as ti
import numpy as np
from enum import Enum
import dagger as dag
import scabbard.riverdale.rd_params as rdpa
import scabbard.riverdale.rd_grid as rdgd
import scabbard.riverdale.rd_hydrodynamics as rdhy
import scabbard.riverdale.rd_morphodynamics as rdmo


# @ti.data_oriented
class Riverdale:
	'''
	Main class controlling riverdale's model.
	Cannot be directly instantiated, please use the factory function at the end of this script.
	'''

	#
	_already_created = False
	_instance_created = False
	
	def __init__(self):

		if not Riverdale._instance_created:
			raise Exception("Riverdale cannot be instantiated directly. Please use the factory functions.")

		if Riverdale._already_created:
			raise Exception("""
Riverdale cannot be instantiated more than once so far within the same runtime (~= within the same script). 
This is because only one unique taichi lang context can exist at once as far as I know, I am working on solutions 
but in the meantime you can run the model in batch using subprocess.
""")
		
		# Place holders for the different variables
		## This is the parameter sheet
		self.param = None

		## Model-side params
		self.GRID = rdgd.GRID
		self.PARAMHYDRO = rdhy.PARAMHYDRO
		self.PARAMMORPHO = rdmo.PARAMMORPHO


		## Fields for hydro
		self.QwA = None
		self.QwB = None
		self.QwC = None
		self.Z = None
		self.hw = None
		self.P = None
		self.input_rows_Qw = None
		self.input_cols_Qw = None
		self.input_Qw = None
		self.convrat = None

		## Field for morpho
		self.QsA = None
		self.QsB = None
		self.QsC = None
		self.input_rows_Qs = None
		self.input_cols_Qs = None
		self.input_Qs = None



	def run_hydro(self, n_steps):
		'''
		Main runner function for the hydrodynamics part of the model.
		NOte that all the parameters have been compiled prior to running that functions, so not much to control here
		Arguments:
			- nsteps: The number of time step to run
		returns:
			- Nothing, update the model internally
		Authors:
			- B.G (last modification 20/05/2024)
		'''

		# Running loop
		for _ in range(n_steps):
			
			# Initialising step: setting stuff to 0, reset some counters and things like that
			self._run_init_hydro()
			
			# Add external water inputs: rain, discrete entry points, ...
			self._run_hydro_inputs()

			# Actually runs the runoff simulation
			self._run_hydro()

		# That's it reallly, see bellow for the internal functions

	def _run_init_hydro(self):
		rdhy.initiate_step(self.QwB)

	def _run_hydro_inputs(self):
		'''
		Internal function automating the runniong of anything that adds water to the model (precipitations, input points, ...)
		It determines everything automatically from the param sheet (RDparam class)
		
		Returns: 
			- Nothing, runs a subpart of the model
		
		Authors:
			- B.G (last modification: 28/05/2024)
		'''

		# First checking if we need the  precipitations inputs
		if(self.param.need_precipitations):

			# then running the 2D precipitations
			if(self.param._2D_precipitations):
					rdhy.variable_rain(self.QwA, self.QwB, self.P,self.BCs)
			# or the 1D
			else:
				rdhy.constant_rain(self.QwA, self.QwB, self.P,self.BCs)
		# Secondly, applying the discrete inputs if needed too				
		if(self.param.need_input_Qw):
			rdhy.input_discharge_points(self.input_rows_Qw, self.input_cols_Qw, self.input_Qw, self.QwA, self.QwB, self.BCs)

	def _run_hydro(self):
		rdhy.compute_Qw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs)
		rdhy.compute_hw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs)


	def run_morpho(self, n_steps):
		'''
		Main runner function for the morphodynamics part of the model.
		NOte that all the parameters have been compiled prior to running that functions, so not much to control here
		Arguments:
			- nsteps: The number of time step to run
		returns:
			- Nothing, update the model internally
		Authors:
			- B.G (last modification 20/05/2024)
		'''
		for _ in range(n_steps):
			###
			# TODO:: Add and test morpho stuff here
			self._run_init_morpho()

			# NOT READY YET
			# self._run_inputs_Qs()

			self._run_morpho()



	

	def _run_init_morpho(self):
		rdmo.initiate_step(self.QsB)
	def _run_inputs_Qs(self):
		rdmo.input_discharge_sediment_points(self.input_rows_Qs, self.input_cols_Qs, self.input_Qs, self.QsA, self.QsB, self.BCs)
	def _run_morpho(self):
		rdmo.compute_Qs(self.Z, self.hw, self.QsA, self.QsB, self.QsC, self.BCs )
		rdmo.compute_hs(self.Z, self.hw, self.QsA, self.QsB, self.QsC, self.BCs )


	@property
	def convergence_ratio(self):
		if(self.param is None):
			raise ValueError('cannot return convergence ratio if the model is not initialised')
		if(self.convrat is None):
			self.convrat = ti.field(dtype = ti.f32, shape = ())
		rdhy.check_convergence(self.QwA, self.QwC, 0.01, self.convrat, self.BCs)
		return float(self.convrat.to_numpy())

	def get_GridCPP(self):
		'''
		Returns a GridCPP object corresponding to the grid geometry and boundary conditions.
		GridCPP objectss are used to interact with the DAGGER c++ engine which I use for CPU intensive tasks.
		It will probably also be used for communication with TTBlib and fastscapelib

		Returns:
			- a GridCPP object ready to be passed to the C++ engine
		Authors:
			- B.G. (last modification: 30/05/2024)
		'''
		return dag.GridCPP_f32(self.param._nx,self.param._ny,self.param._dx,self.param._dy,0 if self.param.boundaries == rdgd.BoundaryConditions.normal else 3)

	@classmethod
	def _create_instance(cls):
		'''
		Private function creating an empty instance and returning it
		'''
		
		cls._instance_created = True
		instance = cls()
		cls._already_created = True
		cls._instance_created = False

		return instance

# External factory function
def create_from_params(param):

	# Generating the empty instance, which should be the sole one
	instance = Riverdale._create_instance()

	# Referencing it in the Param files thus fixing it 
	param._RD = instance
	instance.param = param

	if(
		param._boundaries != rdgd.BoundaryConditions.normal 
		and param._boundaries != rdgd.BoundaryConditions.customs
	):
		raise Exception("Selected Boundary conditions is not available yet (WIP)")

	# Setting up the grid
	instance.GRID.dx = param._dx
	instance.GRID.dy = param._dy
	instance.GRID.nx = param._nx
	instance.GRID.ny = param._ny
	instance.GRID.nxy = param._nxy
	instance.GRID.boundaries = param._boundaries

	instance.PARAMHYDRO.hydro_slope_bc_mode = int(param._boundary_slope_mode.value)
	instance.PARAMHYDRO.hydro_slope_bc_val = param._boundary_slope_value


	if(param.BCs is None):
		instance.BCs = ti.field(ti.int32, shape = (1,1))
	else:
		instance.BCs = ti.field(ti.uint8, shape = (param._ny, param._nx))
		instance.BCs.from_numpy(param.BCs)

	# Compiling the grid functions
	rdgd.set_grid_CC()

	# Setting up the flow conditions
	instance.PARAMHYDRO.manning = param.manning
	instance.PARAMHYDRO.dt_hydro = param.dt_hydro

	# Compiling the hydrodynamics
	rdhy.set_hydro_CC()

	instance.QwA = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.QwA.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))
	instance.QwB = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.QwB.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))
	instance.QwC = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.QwC.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))

	instance.Z = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.Z.from_numpy(param.initial_Z)
	instance.hw = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx))
	
	if(param.initial_hw is not None):
		instance.hw.from_numpy(param.initial_hw)
	else:
		instance.hw.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))

	if param._2D_precipitations:
		instance.P = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx)) 
		instance.P.from_numpy(param.precipitations.astype(np.float32))
	else:
		instance.P = param.precipitations

	if(param.need_input_Qw):
		n_inputs_QW = param._input_rows_Qw.shape[0]
		instance.input_rows_Qw = ti.field(ti.i32, shape = (n_inputs_QW))
		instance.input_rows_Qw.from_numpy(param._input_rows_Qw)
		instance.input_cols_Qw = ti.field(ti.i32, shape = (n_inputs_QW))
		instance.input_cols_Qw.from_numpy(param._input_cols_Qw)
		instance.input_Qw = ti.field(ti.f32, shape = (n_inputs_QW))
		instance.input_Qw.from_numpy(param._input_Qw)

	if(param.morpho):

		instance.PARAMMORPHO.dt_morpho = param.dt_morpho
		instance.PARAMMORPHO.morphomode = param.morphomode
		instance.PARAMMORPHO.GRAVITY = param.GRAVITY
		instance.PARAMMORPHO.rho_water = param.rho_water
		instance.PARAMMORPHO.rho_sediment = param.rho_sediment
		instance.PARAMMORPHO.k_z = param.k_z
		instance.PARAMMORPHO.k_h = param.k_h
		instance.PARAMMORPHO.k_erosion = param.k_erosion
		instance.PARAMMORPHO.alpha_erosion = param.alpha_erosion
		instance.PARAMMORPHO.D = param.D
		instance.PARAMMORPHO.tau_c = param.tau_c
		instance.PARAMMORPHO.transport_length = param.transport_length

		rdmo.set_morpho_CC()

		instance.QsA = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx))
		instance.QsA.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))
		instance.QsB = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx))
		instance.QsB.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))
		instance.QsC = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx))
		instance.QsC.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))

	
	instance.input_rows_Qs = None
	instance.input_cols_Qs = None
	instance.input_Qs = None

	return instance



































































# end of file
