'''
Riverdale environment:

This module defines the `Riverdale` class, which serves as the primary high-level interface for interacting with the Riverdale model.
It provides the API for users to extract data from the model and advance simulation time steps.
The `RDParams` class (defined in `rd_params.py`) manages the model's input parameters and configurations.
The internal structure of the `Riverdale` class is designed for optimizing calculation time, which may result in a somewhat convoluted organization.

Author: B.G.
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
	"""
	Main class controlling the Riverdale model.

	This class cannot be directly instantiated. Instead, use the `create_from_params`
	factory function at the end of this script to create an instance of `Riverdale`
	and bind the input parameter sheet.

	Attributes:
		param (RDParams): The parameter sheet for the model.
		GRID (rdgd.GRID): Grid parameters for the model.
		PARAMHYDRO (rdhy.PARAMHYDRO): Hydrodynamic parameters for the model.
		PARAMMORPHO (rdmo.PARAMMORPHO): Morphodynamic parameters for the model.
		QwA (ti.field): Taichi field for water discharge component A.
		QwB (ti.field): Taichi field for water discharge component B.
		QwC (ti.field): Taichi field for water discharge component C.
		Z (ti.field): Taichi field for topography.
		hw (ti.field): Taichi field for water depth.
		P (ti.field or float): Taichi field or float for precipitation.
		input_rows_Qw (ti.field): Taichi field for row indices of water input points.
		input_cols_Qw (ti.field): Taichi field for column indices of water input points.
		input_Qw (ti.field): Taichi field for water input values.
		convrat (ti.field): Taichi field for convergence ratio.
		constraints (ti.field): Taichi field for hydrodynamic constraints.
		fdir (ti.field): Taichi field for flow direction (D8).
		fsurf (ti.field): Taichi field for surface elevation.
		QsA (ti.field): Taichi field for sediment discharge component A.
		QsB (ti.field): Taichi field for sediment discharge component B.
		QsC (ti.field): Taichi field for sediment discharge component C.
		input_rows_Qs (ti.field): Taichi field for row indices of sediment input points.
		input_cols_Qs (ti.field): Taichi field for column indices of sediment input points.
		input_Qs (ti.field): Taichi field for sediment input values.
		temp_fields (dict): Dictionary to store temporary Taichi fields for memory management.

	Author: B.G.
	"""

	def __init__(self):
		"""
		Initializes the Riverdale model.

		This constructor should not be called directly. Use the `create_from_params`
		factory function to properly initialize a Riverdale instance.
		"""
		
		# Placeholders for the different variables
		self.param = None  # The parameter sheet

		# Model-side parameters
		self.GRID = rdgd.GRID
		self.PARAMHYDRO = rdhy.PARAMHYDRO
		self.PARAMMORPHO = rdmo.PARAMMORPHO

		# Fields for hydrodynamics
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

		# Fields for morphodynamics
		self.QsA = None
		self.QsB = None
		self.QsC = None
		self.input_rows_Qs = None
		self.input_cols_Qs = None
		self.input_Qs = None

		# Temporary fields for memory management
		self.temp_fields = {ti.u8:[], ti.i32:[], ti.f32:[], ti.f64:[]}



	def run_hydro(self, n_steps, recompute_flow = False, expe_N_prop = 0, expe_CFL_variable = False, flush_LM = False):
		"""
		Runs the main hydrodynamic simulation loop for a specified number of steps.

		This function orchestrates the hydrodynamic calculations, including flow recomputation,
		handling external water inputs (precipitation, discrete points), and running the runoff simulation.
		Parameters are typically compiled prior to calling this function, limiting direct control here.

		Args:
			n_steps (int): The number of time steps to run the hydrodynamic simulation.
			recompute_flow (bool, optional): If True, recomputes the flow direction (fdir) or surface (fsurf)
											 based on the current topography and water depth. Defaults to False.
			expe_N_prop (int, optional): Experimental parameter, not currently used in the active code path. Defaults to 0.
			expe_CFL_variable (bool, optional): Experimental parameter to use a variable CFL condition for `hw` computation. Defaults to False.
			flush_LM (bool, optional): If True, flushes the water discharge (QwA) only, typically used for
									   local minima handling. Defaults to False.

		Returns:
			None: Updates the model's internal state (e.g., water depth, discharge fields).

		Author: B.G. (last modification: 20/05/2024)
		"""

		if(recompute_flow):
			if(self.param.use_fdir_D8):
				# Recompute D8 flow direction based on current Z and hw
				rdfl.compute_D4_Zw(self.Z, self.hw, self.fdir, self.BCs)
			else:
				# Recompute surface for flow routing if not using D8
				fsurf = scb.raster.raster_from_array(self.Z.to_numpy()+self.hw.to_numpy(),self.param._dx)
				tBCs = None if self.param.BCs is None else self.BCs.to_numpy()
				# Apply priority flood and Gaussian filter for surface smoothing
				scb.flow.priority_flood(fsurf, in_place = True,BCs = tBCs)
				scb.filters.gaussian_fourier(fsurf, in_place = True, magnitude = 50,BCs = tBCs)				 
				scb.flow.priority_flood(fsurf, in_place = True,BCs = tBCs)
				self.fsurf.from_numpy(fsurf.Z.astype(np.float32))

		# Main simulation loop
		for it in range(n_steps):
			
			# Initialize step: reset counters and temporary variables
			self._run_init_hydro()
			
			# Apply external water inputs (e.g., rain, discrete sources)
			self._run_hydro_inputs()

			# Run the core runoff simulation
			self._run_hydro(expe_CFL_variable = expe_CFL_variable, flush_LM = flush_LM)

		# Internal functions are defined below for further details.

	def _run_init_hydro(self):
		"""
		Initializes the hydrodynamic step.

		This internal function prepares the model for a new hydrodynamic time step,
		typically by resetting or initializing relevant fields.

		Returns:
			None

		Author: B.G.
		"""
		rdhy.initiate_step(self.QwB)

	def _run_hydro_inputs(self):
		"""
		Automates the application of water inputs to the model.

		This internal function handles various water input mechanisms, such as precipitation
		(both 1D and 2D) and discrete input points. It automatically determines which inputs
		to apply based on the settings in the `RDParam` (param) sheet.

		Returns:
			None: Runs a subpart of the model, updating internal water fields.

		Author: B.G. (last modification: 28/05/2024)
		"""

		# Check if precipitation inputs are needed
		if(self.param.need_precipitations):

			# Run 2D precipitations if specified
			if(self.param.precipitations_are_2D):
					rdhy.variable_rain(self.QwA, self.QwB, self.P,self.BCs)
			# Otherwise, run 1D constant precipitation
			else:
				rdhy.constant_rain(self.QwA, self.QwB, self.P,self.BCs)
		# Apply discrete water inputs if needed				
		if(self.param.need_input_Qw):
			rdhy.input_discharge_points(self.input_rows_Qw, self.input_cols_Qw, self.input_Qw, self.QwA, self.QwB, self.BCs)

	def _run_hydro(self, expe_CFL_variable = False, flush_LM = False):
		"""
		Internal runner for the core hydrodynamic functions.

		This function computes water discharge (Qw) and water depth (hw) based on the
		model's current state and parameters. It handles different flow computation modes
		(D8 or surface reconstruction) and can optionally flush local minima or use
		an experimental variable CFL condition.

		Args:
			expe_CFL_variable (bool, optional): If True, uses an experimental variable CFL
												condition for `hw` computation. Defaults to False.
			flush_LM (bool, optional): If True, flushes the water discharge (QwA) only,
									   typically used for local minima handling. Defaults to False.

		Returns:
			None: Updates the internal water discharge and water depth fields.

		Author: B.G.
		"""

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
		"""
		Calculates an analytical water depth (`hw`) based on current `hw` and `QwA` fields.

		This function can be useful for various purposes, such as accelerating the convergence
		of hillslope simulations or generating an initial guess for `hw` from an external `QwA`.
		
		WARNING: This is not a perfect analytical solution and may produce noisy results,
		especially in complex terrains. A truly simple analytical solution for 2D SWEs
		(Shallow Water Equations) involving large inverse matrices is generally not known.

		Returns:
			None: Updates the model's internal `hw` field.

		Author: B.G. (last modification 09/2024)
		"""
		temp, = self.query_temporary_fields(1)
		rdhy._raise_analytical_hw(self.Z, self.hw, self.QwA, temp, self.BCs)

	def diffuse_hw(self, n_steps = 100):
		"""
		Diffuses the water depth (`hw`) field using a cellular automaton (CA) smoothing approach.

		This function applies a smoothing operation to the water depth field over a specified
		number of steps, which can help in stabilizing the simulation or reducing noise.

		Args:
			n_steps (int, optional): The number of smoothing steps to apply. Defaults to 100.

		Returns:
			None: Updates the internal `hw` field.

		Author: B.G.
		"""

		temp, = self.query_temporary_fields(1)
		for i in range(n_steps):
			rdhy._CA_smooth(self.Z, self.hw, temp, self.BCs)
		ti.sync()
		

	def propagate_QwA(self, SFD = True):
		"""
		Propagates water discharge (QwA) based on the current model state.

		This function initializes hydrodynamic conditions, applies water inputs,
		and then propagates the water discharge using either a Single Flow Direction (SFD)
		graph or a Multiple Flow Direction (MFD) approach.

		Args:
			SFD (bool, optional): If True, uses a Single Flow Direction (SFD) graph for propagation.
								If False, uses a Multiple Flow Direction (MFD) approach. Defaults to True.

		Returns:
			None: Updates the internal `QwA` field.

		Author: B.G.
		"""

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

	def save_outputs(self, fname = 'riverdale_outputs'):
		'''
		Export a bunch of GraphFlood outputs
		'''
		import pickle

		param = self.param
		param._RD = None

		tosave = {
			'hw' :self.hw.to_numpy(), 
			'Z'  :self.Z.to_numpy(), 
			'QwA':self.QwA.to_numpy(), 
			'QwC':self.QwC.to_numpy(),
			'Sw':scb.rvd.compute_hydraulic_gradient(self),
			'tau':scb.rvd.compute_shear_stress(self),
			'u':scb.rvd.compute_flow_velocity(self, use_Qwin = False),
			'q':scb.rvd.compute_flow_velocity(self, use_Qwin = False) * self.hw.to_numpy(),
			}


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
		temphw = instance.hw.to_numpy()
		rdlm.priority_flood(instance, Zw = True)
		# Calculating the motherflow direction, used to trasfer Qw out of local minimas
		rdfl.compute_D4_Zw(instance.Z, instance.hw, instance.fdir, instance.BCs)
		instance.hw.from_numpy(temphw)
		del temphw
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
