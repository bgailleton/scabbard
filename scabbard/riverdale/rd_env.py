'''
Riverdale environment

'''

import taichi as ti
import numpy as np
from enum import Enum
import scabbard.riverdale.rd_params as rdpa
import scabbard.riverdale.rd_grid as rdgd
import scabbard.riverdale.rd_hydrodynamics as rdhy


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


		## Fields for hydro
		self.QwA = None
		self.QwB = None
		self.QwC = None
		self.Z = None
		self.hw = None
		self.P = None
		self.input_rows = None
		self.input_cols = None
		self.input_Qw = None


	def run(self, n_steps):
		for _ in range(n_steps):

			rdhy.initiate_step(self.QwB)
			
			if(self.param._2D_precipitations):
				rdhy.variable_rain(self.QwA, self.QwB, self.P)
			else:
				rdhy.constant_rain(self.QwA, self.QwB, self.P)				
			
			rdhy.compute_Qw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs)
			rdhy.compute_hw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs)

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

	if(param._boundaries != rdgd.BoundaryConditions.normal):
		raise Exception("Boundary consitions otehr than normal are currently being implemented")

	# Setting up the grid
	instance.GRID.dx = param._dx
	instance.GRID.dy = param._dy
	instance.GRID.nx = param._nx
	instance.GRID.ny = param._ny
	instance.GRID.nxy = param._nxy
	instance.GRID.boundaries = param._boundaries
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


	# TODO
	instance.input_rows = None
	instance.input_cols = None
	instance.input_Qw = None

	return instance



































































# end of file
