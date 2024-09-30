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
import scabbard.riverdale.rd_flow as rdfl
import scabbard.riverdale.rd_hydrodynamics as rdhy
import scabbard.riverdale.rd_morphodynamics as rdmo
import scabbard.riverdale.rd_LM as rdlm
import scabbard as scb
from scipy.ndimage import gaussian_filter


# @ti.data_oriented
class Riverdale:
	'''
	Main class controlling riverdale's model.
	Cannot be directly instantiated, please use the factory function at the end of this script.
	'''

	#
	# _already_created = False
	# _instance_created = False
	
	def __init__(self):

# 		if not Riverdale._instance_created:
# 			raise Exception("Riverdale cannot be instantiated directly. Please use the factory functions.")

# 		if Riverdale._already_created:
# 			raise Exception("""
# Riverdale cannot be instantiated more than once so far within the same runtime (~= within the same script). 
# This is because only one unique taichi lang context can exist at once as far as I know, I am working on solutions 
# but in the meantime you can run the model in batch using subprocess.
# """)
		
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
		self.constraints = None

		self.fdir = None
		self.fsurf = None

		## Field for morpho
		self.QsA = None
		self.QsB = None
		self.QsC = None
		self.input_rows_Qs = None
		self.input_cols_Qs = None
		self.input_Qs = None

		# temporary fields
		self.temp_fields = {ti.u8:[], ti.i32:[], ti.f32:[], ti.f64:[]}



	def run_hydro(self, n_steps, recompute_flow = False, expe_N_prop = 0, expe_CFL_variable = False, flush_LM = False):
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

		if(recompute_flow):
			if(self.param.use_fdir_D8):
				rdfl.compute_D4_Zw(self.Z, self.hw, self.fdir, self.BCs)
			else:
				fsurf = scb.raster.raster_from_array(self.Z.to_numpy()+self.hw.to_numpy(),self.param._dx)
				tBCs = None if self.param.BCs is None else self.BCs.to_numpy()
				scb.flow.priority_flood(fsurf, in_place = True,BCs = tBCs)
				scb.filters.gaussian_fourier(fsurf, in_place = True, magnitude = 50,BCs = tBCs)				 
				scb.flow.priority_flood(fsurf, in_place = True,BCs = tBCs)
				self.fsurf.from_numpy(fsurf.Z.astype(np.float32))

		# Running loop
		for it in range(n_steps):
			
			# Initialising step: setting stuff to 0, reset some counters and things like that
			self._run_init_hydro()
			
			# Add external water inputs: rain, discrete entry points, ...
			self._run_hydro_inputs()

			# Actually runs the runoff simulation
			# self._run_hydro() if expe_N_prop <= 1 or (it % expe_N_prop == 0) else rdhy._propagate_QwA_only(self.Z, self.hw, self.QwA, self.QwB, self.BCs )
			self._run_hydro(expe_CFL_variable = expe_CFL_variable, flush_LM = flush_LM)

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
			if(self.param.precipitations_are_2D):
					rdhy.variable_rain(self.QwA, self.QwB, self.P,self.BCs)
			# or the 1D
			else:
				rdhy.constant_rain(self.QwA, self.QwB, self.P,self.BCs)
		# Secondly, applying the discrete inputs if needed too				
		if(self.param.need_input_Qw):
			rdhy.input_discharge_points(self.input_rows_Qw, self.input_cols_Qw, self.input_Qw, self.QwA, self.QwB, self.BCs)

	def _run_hydro(self, expe_CFL_variable = False, flush_LM = False):
		'''
		Internal runner for hydro functions

		'''

		if(flush_LM):
			rdhy._flush_QwA_only(self.Z, self.hw, self.QwA, self.QwB, self.BCs, self.fdir)
			return

		if(self.param.use_fdir_D8):
			if(self.param.stationary):
				rdhy._compute_Qw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs, self.fdir) #if rdhy.FlowMode.static_drape != self.param._hydro_compute_mode else rdhy.compute_Qw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.constraints, self.BCs)
			else:
				rdhy._compute_Qw_dynamic(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs) #if rdhy.FlowMode.static_drape != self.param._hydro_compute_mode else rdhy.compute_Qw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.constraints, self.BCs)
		else:
			rdhy._compute_Qw_surfrec(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs, self.fsurf) #if rdhy.FlowMode.static_drape != self.param._hydro_compute_mode else rdhy.compute_Qw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.constraints, self.BCs)
		
		if(expe_CFL_variable):
			rdhy._compute_hw_CFL(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs, 1e-4, 0.001 )
		else:	
			if(self.param.stationary):		
				rdhy.compute_hw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs) if rdhy.FlowMode.static_drape != self.param.hydro_compute_mode else rdhy.compute_hw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.constraints, self.BCs)
			else:
				rdhy.compute_hw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.BCs) if rdhy.FlowMode.static_drape != self.param.hydro_compute_mode else rdhy.compute_hw(self.Z, self.hw, self.QwA, self.QwB, self.QwC, self.constraints, self.BCs)
				
	def raise_analytical_hw(self):
		'''
		This function uses the current fields of hw and Qwin (QwA) to calculate an analytical hw.
		This is useful in multiple occasions, for example speed up convergence of hillslopes if the rest is fine or creating an initial guess from an external Qwin.
		WARNING: this is not a magical analytical solution, most of the time it creates a noisy combination of flow depth with walls. 
		If 2D SWEs had a true easy analytical solution not involving gigantic inverse matrices it would be known

		Arguments:
			- None
		returns:
			- Nothing, update the model internally
		Authors:
			- B.G (last modification 09/2024)
		'''
		temp, = self.query_temporary_fields(1)
		rdhy._raise_analytical_hw(self.Z, self.hw, self.QwA, temp, self.BCs)

	def diffuse_hw(self, n_steps = 100):

		temp, = self.query_temporary_fields(1)
		for i in range(n_steps):
			rdhy._CA_smooth(self.Z, self.hw, temp, self.BCs)
		ti.sync()
		

	def propagate_QwA(self, SFD = True):

		self._run_init_hydro()
		self._run_hydro_inputs()
		input_values = self.QwB.to_numpy()

		if(SFD):
			tZ = (self.Z.to_numpy() + self.hw.to_numpy())
			tBCs = scb.flow.get_normal_BCs(tZ) if self.param.BCs is None else self.BCs.to_numpy()
			stg = scb.flow.SFGraph(tZ, BCs = tBCs, D4 = True, dx = self.param._dx, backend = 'ttb', fill_LM = True, step_fill = 1e-3)
			Qw = scb.flow.propagate(stg, input_values, step_fill = 1e-3)
		else:
			tZ = (self.Z.to_numpy() + self.hw.to_numpy())
			tBCs = scb.flow.get_normal_BCs(tZ) if self.param.BCs is None else self.BCs.to_numpy()
			grid = scb.raster.raster_from_array(tZ, dx = self.param._dx, xmin = 0., ymin = 0., dtype = np.float32)
			Qw = scb.flow.propagate(grid, input_values, method = 'mfd_S', BCs = tBCs, D4 = True, fill_LM = True, step_fill = 1e-3)

		self.QwA.from_numpy(Qw.astype(np.float32))



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
		rdmo.compute_Qs(self.Z, self.hw, self.QsA, self.QsB, self.QsC, self.QwA, self.QwC, self.BCs )
		rdmo.compute_hs(self.Z, self.hw, self.QsA, self.QsB, self.QsC, self.BCs )


	@property
	def convergence_ratio(self):

		if(self.param is None):
			raise ValueError('cannot return convergence ratio if the model is not initialised')

		if(self.convrat is None):
			self.convrat = ti.field(dtype = self.param.dtype_float, shape = ())

		rdhy.check_convergence(self.QwA, self.QwC, 0.01, self.convrat, self.BCs)

		return float(self.convrat.to_numpy())

	def get_GridCPP(self):
		'''
		PENDING DEPRECATION
		Returns a GridCPP object corresponding to the grid geometry and boundary conditions.
		GridCPP objectss are used to interact with the DAGGER c++ engine which I use for CPU intensive tasks.
		It will probably also be used for communication with TTBlib and fastscapelib

		Returns:
			- a GridCPP object ready to be passed to the C++ engine
		Authors:
			- B.G. (last modification: 30/05/2024)
		'''
		return dag.GridCPP_f32(self.param._nx,self.param._ny,self.param._dx,self.param._dy,0 if self.param.boundaries == rdgd.BoundaryConditions.normal else 3)


	def query_temporary_fields(self, N, dtype = 'f32'):
		'''
		Ask riverdale for a number of temporary fields of a given type. 
		Effectively avoids unecessary new fields and memory leaks.
		It only creates a temporary field if it does not exist yet.
			
		Arguments:
			- N: the number of fields to return
			- dtype: the data type: f32, u8 or i32 so far
		Returns:
			- a tuple with all the fields
		Authors:
			- B.G. (last modification: 12/06/2024)
		'''
		
		# Data type conversion to the dict key
		if(dtype in ['f32',self.param.dtype_float, np.float32]):
			dtype = ti.f32
		elif(dtype in ['u8',np.uint8, ti.u8]):
			dtype = ti.u8
		elif(dtype in ['i32',np.int32, ti.i32]):
			dtype = ti.i32
		else:
			raise TypeError(f"dtype {dtype} is not recognised. Should be one of ['f32','u8','i32'] or their taichi/numpy equivalents (e.g. np.float32 or ti.f32)")

		# gathering the results in a list
		output = []
		for i in range(N):
			# DO I need to create the field or does it exist already?
			if(i >= len(self.temp_fields[dtype])):
				self.temp_fields[dtype].append(ti.field(dtype=dtype, shape = (self.GRID.ny,self.GRID.nx)))
			# Filling it with 0s by default	
			self.temp_fields[dtype][i].fill(0)
			#saving a ref in the list
			output.append(self.temp_fields[dtype][i])

		# Done, returning the list conerted into a tuple
		return tuple(output)



	def save(self, fname = 'riverdale_run'):
		'''
		Experiments on saving files
		'''

		import pickle
		# import copy
		param = self.param
		param._RD = None
		tosave = {'param':param, 'hw':self.hw.to_numpy(), 'Z':self.Z.to_numpy(), 'QwA':self.QwA.to_numpy(), 'QwC':self.QwC.to_numpy()}
		with open(fname+'.rvd', 'wb') as f:
			pickle.dump(tosave, f)








# External factory function
def create_from_params(param):
	'''
		This function creates the instance of the model and binds the input param sheet
		In some sort it "compiles" the model. 
		IMPORTANT: every static variable and constants will be fixed at that point
	'''

	# Generating the empty instance, which should be the sole one
	# instance = Riverdale._create_instance()
	instance = Riverdale()

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
	instance.PARAMHYDRO.flowmode = param.hydro_compute_mode
	instance.PARAMHYDRO.clamp_div_hw_val = param._clamp_div_hw_val
	# instance.PARAMHYDRO.flowmode = param.hydro_compute_mode


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

	#WILL NEED TO ADD THE OPTION  LATER
	instance.fdir = ti.field(ti.u8, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.PARAMMORPHO.use_original_dir_for_LM = True
	instance.PARAMHYDRO.use_original_dir_for_LM = True
	instance.PARAMHYDRO.LM_pathforcer = param._LM_npath
	if(param.use_fdir_D8):
		instance.fsurf = ti.field(ti.f32, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.PARAMMORPHO.use_original_dir_for_LM = False
	instance.PARAMHYDRO.use_original_dir_for_LM = False

	

	# Compiling the hydrodynamics
	rdhy.set_hydro_CC()

	instance.QwA = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.QwA.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))
	instance.QwB = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.QwB.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))
	instance.QwC = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.QwC.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))
	instance.constraints = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx,2)) if rdhy.FlowMode.static_drape == instance.param._hydro_compute_mode else None

	instance.Z = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx))
	instance.Z.from_numpy(param.initial_Z)
	instance.hw = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx))
	
	if(param.initial_hw is not None):
		instance.hw.from_numpy(param.initial_hw)
	else:
		instance.hw.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))

	if param.precipitations_are_2D:
		instance.P = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx)) 
		instance.P.from_numpy(param.precipitations.astype(np.float32))
	else:
		instance.P = param.precipitations

	if(param.need_input_Qw):
		n_inputs_QW = param._input_rows_Qw.shape[0]
		instance.input_rows_Qw = ti.field(ti.i32, shape = (n_inputs_QW))
		instance.input_rows_Qw.from_numpy(param._input_rows_Qw)
		instance.input_cols_Qw = ti.field(ti.i32, shape = (n_inputs_QW))
		instance.input_cols_Qw.from_numpy(param._input_cols_Qw)
		instance.input_Qw = ti.field(instance.param.dtype_float, shape = (n_inputs_QW))
		instance.input_Qw.from_numpy(param._input_Qw)

	if(param.precompute_Qw):
		# Precomputing a D8 propagation of QwA within the cpu
		instance.propagate_QwA()


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

		instance.QsA = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx))
		instance.QsA.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))
		instance.QsB = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx))
		instance.QsB.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))
		instance.QsC = ti.field(instance.param.dtype_float, shape = (instance.GRID.ny,instance.GRID.nx))
		instance.QsC.from_numpy(np.zeros((param._ny,param._nx), dtype = np.float32))

	
	instance.input_rows_Qs = None
	instance.input_cols_Qs = None
	instance.input_Qs = None


	#running eventual preprocessors
	if(param.use_fdir_D8):
		# print('debug info: computing fdir')
		# creating the original hydraulic pattern by pre filling the topography with water
		rdlm.priority_flood(instance)
		# Calculating the motherflow direction, used to trasfer Qw out of local minimas
		rdfl.compute_D4_Zw(instance.Z, instance.hw, instance.fdir, instance.BCs)
		# print(np.unique(instance.fdir.to_numpy()))
	else:
		fsurf = scb.raster.raster_from_array(instance.Z.to_numpy()+instance.hw.to_numpy(),instance.param._dx)
		tBCs = None if instance.param.BCs is None else instance.BCs.to_numpy()
		scb.flow.priority_flood(fsurf, in_place = True,BCs = tBCs)
		scb.filters.gaussian_fourier(fsurf, in_place = True, magnitude = 50,BCs = tBCs)				 
		scb.flow.priority_flood(fsurf, in_place = True,BCs = tBCs)
		instance.fsurf.from_numpy(fsurf.Z.astype(np.float32))

	return instance


def load_riverdale(fname):
	import pickle
	with open(fname, 'rb') as f:
		loaded_dict = pickle.load(f)

	rd = create_from_params(loaded_dict['param'])

	rd.hw.from_numpy(loaded_dict['hw'])
	rd.Z.from_numpy(loaded_dict['Z'])
	rd.QwA.from_numpy(loaded_dict['QwA'])
	rd.QwC.from_numpy(loaded_dict['QwC'])
	
	return rd
































































# end of file
