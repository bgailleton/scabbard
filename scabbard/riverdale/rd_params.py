'''
Parameter class and function for Riverdale
This is the user-side parameter sheet, 
not to be confused with the singleton parameter classes managing the compile time constants located at the top of some files

To some exceptions, this parameter class manages all the inputs to the model (e.g. initial topo, uplift, precipitations, manning, ...)

Authors:
	- B.G.
'''

import numpy as np
from enum import Enum
import scabbard.riverdale.rd_grid as rdgd
import scabbard.riverdale.rd_morphodynamics as rdmo
import scabbard.utils as scaut 
import scabbard as scb
import dagger as dag
import warnings


class RDParams:
	'''
	Parameter class for RiverDale, contains the sheet of parameters required Riverdale's model
	This is the user-side parameter sheet, managing the precipitation, roughness, erodability, critical shear stress ...

	It manages inputs for the model, some of them are to be fixed before initiating the model and others can be edited during the model lifetime.

	Authors:
		- B.G (last modification on the 30/04/1992)
	'''


	############################################################
	# CONSTRUCTOR ##############################################
	############################################################


	def __init__(self):
		'''
			The constructor does not actually take parameters: it initiates all the parameters to their default values
			Note, internal variables starts with _. Do not use them directly if you do not know what you are doing.
			Most of the API is exposed via cleaner properties (setters and getters) without the _ (see all the @property) or direct functions

			Returns:
				- A default RDParams parameter sheet
			Authors:
				- B.G. (last modifications: 29/05/2024)
		'''

		# _RD is a reference to the model object (RiverDale)
		# Once initialised, RD also gets a ref to this param sheet
		# Constant parameters becomes lock and the variable inputs will be automatically transmitted on modification
		self._RD = None

		##############################
		# GRID Parameters
		##############################

		# Grid Geometry, probably no need to expose them directly:
		## spatial step in the x direction
		self._dx = 1.
		## spatial step in the y direction (Warning, some of the codes only considers dx=dy)
		self._dy = 1.
		## Number of columns
		self._nx = 50
		## Number of rows
		self._ny = 50
		## Number of nodes in total
		self._nxy = self._nx * self._ny

		##############################
		# Boundary conditions
		##############################
		
		# Type of boundary conditions for the flow connectivity and no data management
		## This enum is defined in rd_grid.py and determines how the codes reads/deal bcs
		self._boundaries = rdgd.BoundaryConditions.normal
		## Arrays of boundary conditions - used if the boundary conditions are set to customs
		## See rd_grid.py for the meaning (or the future doc at the date of the 29/05/2024)
		self._BCs = None

		# Slope boundary conditions: what slope to apply on outting cells
		self._boundary_slope_mode = rdgd.BoundaryConditionsSlope.fixed_slope
		self._boundary_slope_value = 1e-2


		##############################
		# Initial conditions
		##############################

		# Initial topography, fed to the model at initiation and also kept here for comparisons or things like that
		self.initial_Z = np.random.rand(self._ny, self._nx).astype(dtype = np.float32)
		
		# Initial flow depth - kept to 0 if None
		self.initial_hw = None

		
		##############################
		# HYDRO Parameters
		##############################
		
		# Manning's Roughness/friction coefficient
		self._manning = 0.033
		
		# time step for the hydrodynamics model
		self._dt_hydro = 1e-3

		# Precipitation rates
		## The value (2 or 2D)
		## Once the model is initialised, it cannot be switched from 1D to 2D
		self._precipitations = 50*1e-3/3600
		## Does the model even need to run the precipitation input?
		self._has_precipitations = True
		## Is the precipitation field field in 2D
		self._2D_precipitations = False
		## are precipitations manually cut off
		self._force_no_prec = False

		# Discrete input points
		# Default is no direct inputs
		# The models checks if this is required with the need_input_Qw property
		## Indices of the row inputs
		self._input_rows_Qw = None
		## Indices of the columns inputs
		self._input_cols_Qw = None
		## Input values in m^3/s
		self._input_Qw = None



		##############################
		# MORPHO Parameters
		##############################


		# Is the Morphodynamics module enabled?
		## If False, it does not run a single morpho functions and you can ignore the params
		self._morpho = False

		# Time step for the morphodynamics modelling		
		self._dt_morpho = 1e-3

		# Enum class for the type of fluvial process activated (Ignore, so far just one)
		self._morphomode = rdmo.MorphoMode.fbal

		# Gravitational constant g
		self._GRAVITY = 9.81

		# Water density
		self._rho_water = 1000

		# Sediment density
		self._rho_sediment = 2600

		# Gravitational erosion coeff
		self._k_z = 1.

		# shear-stress entrainement local coeff erosion coeff
		self._k_h = 1.

		# Fluvial erosion coeff (the dimentional E in MPM)
		self._k_erosion = 1e-5

		# Fluvial exponent
		self._alpha_erosion = 1.5

		# Grainsize
		self._D = 4e-3 

		# Critical shear stress
		self._tau_c = 4

		# critical shear strass
		self._transport_length = 4

		
	############################################################
	# GRID Stuffs ##############################################
	############################################################


	def set_initial_conditions(self, nx, ny, dx, dy, Z, hw = None, BCs = None):
		'''
		Set up the initial conditions of the model, from the grid to the Z, BCs or eventually an initial flow depth
		Needs to be done before the model setup

		Arguments:
			- nx: number of columns in the grid
			- ny: number of rows in the grid
			- dx: spatial step in the x direction
			- dy: spatial step in the y direction
			- Z: a numpy array of 2D initial elevation 
			- hw: an optional array of initial flow depth
			- BCs: an optional array of Boundary conditions code in uint8
		returns:
			- Nothing but sets up the inital conditions of the model
		Authors:
			- B.G (last modifications: 30/04/2024)
		TODO:
			- add checks on the dimensions and boundary conditions, also warning is BCs is given but boundaries are set to another mode
		'''

		if(self._RD is not None):
			raise Exception("Cannot modify initial condition and grid parameters after model initiation")

		if (dx != dy):
			raise Exception("Differential spatial step in x and y in WIP, I am even considering dropping it, let me know if needed")

		self._dx = dx
		self._dy = dy
		self._nx = nx
		self._ny = ny
		self._nxy = self._nx * self._ny
		self._boundaries = rdgd.BoundaryConditions.normal
		self.BCs = BCs
		self.initial_Z = Z
		self.initial_hw = hw if(hw is not None) else np.zeros_like(self.initial_Z)


	@property
	def boundaries(self):
		'''
			Boundary enum code of the model, see BoundaryConditions enum class in rd_grid.py
			Authors:
				- B.G (last modification: 29/05/2024)
		'''
		return self._boundaries

	@boundaries.setter
	def boundaries(self, val):
		'''
			Setter for changing the type of boundary conditions in the model.
			It cannot be changed once the model is instantiated.
			Note that if the boundaries are set to `customs`, you can change the boundary codes by modifying BCs.

			Authors:
				- B.G. (last modification: 29/05/2024)	
		'''
		if(val == rdgd.BoundaryConditions.normal):
			self._boundaries = val
		elif(val == rdgd.BoundaryConditions.customs):
			raise ValueError("To set the boundary conditions to custom values in the parameter sheet, you need to directly provide a 2D numpy array of np.uint8 codes to the member param.BCs (where param is the parameter sheet object)")
		else:
			raise NotImplementedError('Boundary condition not implemented yet')

	@property
	def BCs(self):
		'''
		 Actual boundary codes used in the case of customs boundary conditions 
		 See the convention in rd_grid.py
		 Authors:
		 	B.G. (last modification: 29/05/2024)
		'''
		return self._BCs

	@BCs.setter
	def BCs(self, val):
		'''
			Setter for the boundary condition codes.
			Needs to be an uint8 nupy array following the convention in rd_grid.py.
			Note that setting this values to a 2D numpy array sets the boundary condition type to customs
			Authors:
				- B.G (29/05/2024)
		'''
		# if the value is None there is not much to do
		if val is None:
			return

		 # Sanitising input
		if not isinstance(val, np.ndarray) or not len(val.shape) == 2 or not val.shape[0] == self._ny or not val.shape[1] == self._nx or not val.dtype == np.uint8:
			raise ValueError(f"If you want to directly set customs boundary conditions, please provide a 2D array of shape {self._ny},{self._nx} of type np.uint8")

		# Registering it if the model is not instanciated
		if(self._RD is None):
			self._BCs = val
			self._boundaries = rdgd.BoundaryConditions.customs
		# If the model is instanciated I need extra checks and work
		else:
			# Checking if the mdel has been initialised to the right mode
			if(self.boundaries != rdgd.BoundaryConditions.customs):
				raise ValueError('Model is already initialised in another boundary mode than customs. BCs cannot be modified.')
			else:
				# Directly transferring data to the GPU BCs code
				self._RD.BCs.from_numpy(val)


	def set_boundary_slope(self, val, mode = 'slope'):

		if(self._RD is not None):
			raise ValueError("Cannot change the boundary value for slope calculation if the model is already instantiated")

		if(mode == 'slope'):
			self._boundary_slope_mode = rdgd.BoundaryConditionsSlope.fixed_slope
			if(val < 0):
				warnings.warn('Recasting negative slope to 0 for the boundary condition slope calculation')
				vel = 0.

		elif(mode == 'elevation'):
			self._boundary_slope_mode = rdgd.BoundaryConditionsSlope.fixed_elevation
			if(val < 0):
				warnings.warn('Recasting negative slope to 0 for the boundary condition slope calculation')
		
		else:
			raise ValueError(f'mode needs to be "slope" or "elevation", it cannot be {mode}')

		self._boundary_slope_value = val
		




	@property
	def manning(self):
		'''
			Returns the roughness coefficient for the friction equation used to calculate flow velocity
			Authors:
				- B.G. (last modification 30/04/2024)
		'''
		return self._manning

	@manning.setter
	def manning(self, value):
		'''
			Setter function for the roughness coefficient, essentially checking if the input is 1 or 2D
			Authors:
				- B.G. (last modification 30/04/2024)
		'''

		# here I'll lay out the code for 2D mannings
		if(isinstance(value,np.ndarray)):
			raise Exception("2D Roughness coefficient not supported yet")

		# At the moment it needs to be compile-time constant
		if(self._RD is not None):
			raise Exception("Roughness coefficient cannot yet be modified after setting up the model. Working on a robust way to do so but so far it is a compile-time constant for performance reasons and therefore cannot be changed after creating the model instance")
		else:
			self._manning = value

	@property
	def dt_hydro(self):
		'''
			Returns the time step used in the simulation for hydrodynamics
			Authors:
				- B.G. (last modification 30/04/2024)
		'''
		return self._dt_hydro

	@dt_hydro.setter
	def dt_hydro(self, value):
		'''
			Setter function for the time step used in the simulation for hydrodynamics
			nothing particular but is future proofed
			Authors:
				- B.G. (last modification 30/04/2024)
		'''

		self._dt_hydro = value

	@property
	def precipitations(self):
		'''
			Returns the field of precipitation rates (whether it si 1D or 2D)
			Authors:
				- B.G. (last modification 30/04/2024)
		'''
		return self._precipitations

	@precipitations.setter
	def precipitations(self, value):
		'''
			Setter function for the precipitation rates, essentially checking if the input is 1 or 2D
			Authors:
				- B.G. (last modification 23/05/2024)
		'''


		if(self._force_no_prec):
			warnings.warn('Precipitations have been manually disabled using the method `disable_precipitations`, new values will not be taken into account unless the method `enable_precipitations` is run later on.')

		# Checking if the input is 1 or 2D
		if(isinstance(value,np.ndarray)):
			input2D = True
		else:
			input2D = False

		if(not input2D and value == 0):
			self._has_precipitations = False
		else:
			self._has_precipitations = True

		# Cannot change the type of input if the runtime is already initiated
		if(self._2D_precipitations != input2D and self._RD is not None):
			raise Exception("You cannot change precipitations from 1D to 2D (or the other way) once the model is instantiated (working on that).")

		# if I reach here I can set the parameters
		self._precipitations = value

		if(self._RD is not None):
			if(isinstance(value,np.ndarray)):
				value = value.astype(np.float32)
				self._RD.P.from_numpy(value)
			else:
				value = np.float32(value)
				self._RD.P = value

	def set_input_Qw(self, rows, cols, values, disable_precipitations = False):
		'''
		Function to set the discrete input points for the hydrodynamics model.
		Note that it replaces any previous discrete inputs, but it does not replace the precipitations if there are any
		Arguments:
			- rows: indices of the rows where the inputs flow in
			- cols: indices of the columns where the inputs flow in
			- values: input Qw in m^3/s
			- disable_precipitations: switch off any precipitation inputs if true
		'''
		# couple of checks first
		if scaut.is_numpy(values) == False or scaut.is_numpy(rows, dtype = np.integer) == False or scaut.is_numpy(cols, dtype = np.integer) == False:
			print(scaut.is_numpy(values) ,'||', scaut.is_numpy(rows, dtype = np.integer) ,'||', scaut.is_numpy(cols, dtype = np.integer))
			raise ValueError('rows and cols must be 1D numpy arrays of integers and values a numpy array of float')

		tgshape = values.shape

		if(values.shape != tgshape or rows.shape != tgshape or cols.shape != tgshape):
			raise ValueError('Inputs arrays must have the same shape')

		self._input_rows_Qw = rows
		self._input_cols_Qw = cols
		self._input_Qw = values


	@property
	def need_precipitations(self):
		'''
		Property determining if I need to run the pieces of code adding the precipitation or not
		returns True or False
		Authors:
				- B.G. (last modification 23/05/2024)
		'''
		return self._has_precipitations and self._force_no_prec == False

	@need_precipitations.setter
	def need_precipitations(self, val):
		'''
		Explicitely forbidding the manual assignement for this property
		It is computed from the various inputs or more explicit function
		Authors:
			- B.G. (last modification 23/05/2024)
		'''
		print("Cannot manually set need_precipitations")

	def disable_precipitations(self):
		'''
		Explicitely disable precipitations
		Authors:
				- B.G. (last modification 23/05/2024)
		'''
		self._force_no_prec = True

	def enable_precipitations(self):
		'''
		Explicitely re-enable precipitations (only required if it has been explicitely disabled before)
		Authors:
				- B.G. (last modification 23/05/2024)
		'''
		self._force_no_prec = False

	@property
	def need_input_Qw(self):
		'''
		Determines whether the model needs to run the discrete input function or not.
		Authors:
			- B.G (last modification: 28/05/2024)
		'''
		if(self._input_rows_Qw is not None):
			return True
		else:
			return False



	@property
	def dt_morpho(self):
		return self._dt_morpho

	@dt_morpho.setter
	def dt_morpho(self,val):
		self._dt_morpho = val

	@property
	def morpho(self):
		return self._morpho

	@morpho.setter
	def morpho(self,val):
		self._morpho = val

	@property
	def morphomode(self):
		return self._morphomode

	@morphomode.setter
	def morphomode(self,val):
		self._morphomode = val

	@property
	def GRAVITY(self):
		return self._GRAVITY

	@GRAVITY.setter
	def GRAVITY(self,val):
		self._GRAVITY = val

	@property
	def rho_water(self):
		return self._rho_water

	@rho_water.setter
	def rho_water(self,val):
		self._rho_water = val

	@property
	def rho_sediment(self):
		return self._rho_sediment

	@rho_sediment.setter
	def rho_sediment(self,val):
		self._rho_sediment = val

	@property
	def k_z(self):
		return self._k_z

	@k_z.setter
	def k_z(self,val):
		self._k_z = val

	@property
	def k_h(self):
		return self._k_h

	@k_h.setter
	def k_h(self,val):
		self._k_h = val

	@property
	def k_erosion(self):
		return self._k_erosion

	@k_erosion.setter
	def k_erosion(self,val):
		self._k_erosion = val

	@property
	def alpha_erosion(self):
		return self._alpha_erosion

	@alpha_erosion.setter
	def alpha_erosion(self,val):
		self._alpha_erosion = val

	@property
	def D(self):
		return self._D

	@D.setter
	def D(self,val):
		self._D = val

	@property
	def tau_c(self):
		return self._tau_c

	@tau_c.setter
	def tau_c(self,val):
		self._tau_c = val

	@property
	def transport_length(self):
		return self._transport_length

	@transport_length.setter
	def transport_length(self,val):
		self._transport_length = val




def param_from_grid(grid):
	'''
	TODo
	'''
	param = RDParams()
	param.set_initial_conditions(grid.nx, grid.ny, grid.dx, grid.dx, grid.Z2D)

	# TO MOVE IN STANDALONE FUNCTION
	# if(initial_fill):
	# 	if(param._boundaries == rdgd.BoundaryConditions.normal):
	# 		ftopo = param.initial_Z + param.initial_hw
	# 		dag._PriorityFlood_D4_normal_f32(ftopo)
	# 		param.initial_hw += ftopo - (param.initial_Z + param.initial_hw)
	# 	else:
	# 		print("filling with PFD4 not implemented for boundary type")

	return param

def param_from_dem(dem, initial_fill = True):
	return param_from_grid(scb.grid.raster2RGrid(dem), initial_fill = initial_fill)
