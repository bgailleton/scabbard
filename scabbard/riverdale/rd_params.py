'''
Parameter class and function for Riverdale
This is the user-side parameter sheet, not to be confused with the singletons parameter classes managing the compile time constant

Authors:
	- B.G.

'''

import numpy as np
from enum import Enum
import scabbard.riverdale.rd_grid as rdgd
import scabbard.riverdale.rd_morphodynamics as rdmo
import scabbard as scb
import dagger as dag


class RDParams:
	'''
	Parameter class for RiverDale, contains the sheet of parameters required Riverdale's model
	This is the user-side parameter sheet, managing the precipitation, roughness, erodability, critical shear stress ...

	Authors:
		- B.G (last modification on the 30/04/1992)
	'''


	def __init__(self):

		# Initialising to the default parameters
		self._RD = None

		# Initial grid parameters, probably no need to expose them directly:
		self._dx = 1.
		self._dy = 1.
		self._nx = 50
		self._ny = 50
		self._nxy = self._nx * self._ny
		self._boundaries = rdgd.BoundaryConditions.normal
		self.initial_Z = np.random.rand(self._ny, self._nx).astype(dtype = np.float32)
		self.initial_hw = None
		self._BCs = None



		# Parameters for the hydrology
		
		## Roughness coefficient
		self._manning = 0.033
		## time step
		self._dt_hydro = 1e-3
		## Precipitation rates
		### The value (2 or 2D)
		self._precipitations = 50*1e-3/3600
		###	is the field in 2D (Default is none)
		self._2D_precipitations = False


		# Parameters for Morphology
		self._morpho = False
		# Time for the morphodynamics modelling		
		self._dt_morpho = 1e-3
		# What kind of erosion
		self._morphomode = rdmo.MorphoMode.fbal
		# gravitational constant
		self._GRAVITY = 9.81
		# Water density
		self._rho_water = 1000
		# Sediment density
		self._rho_sediment = 2600
		# Gravitational erosion coeff
		self._k_z = 1.
		self._k_h = 1.
		# Fluvial erosion coeff
		self._k_erosion = 1e-5
		# Fluvial exponent
		self._alpha_erosion = 1.5
		# Grainsize
		self._D = 4e-3 
		#critical shear strass
		self._tau_c = 4

		



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
			- boundaries: a BoundaryConditions code
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
		return self._boundaries

	@boundaries.setter
	def boundaries(self, val):
		if(val == rdgd.BoundaryConditions.normal):
			self._boundaries = val
		elif(val == rdgd.BoundaryConditions.customs):
			raise ValueError("To set the boundary conditions to custom values in the parameter sheet, you need to directly provide a 2D numpy array of np.uint8 codes to the member param.BCs (where param is the parameter sheet object)")
		else:
			raise NotImplementedError('Boundary condition not implemented yet')

	@property
	def BCs(self):
		return self._BCs

	@BCs.setter
	def BCs(self, val):
		if(val is None):
			self._boundaries = rdgd.BoundaryConditions.normal
			self._BCs = val
		elif not isinstance(val, np.ndarray) or not len(val.shape) == 2 or not val.shape[0] == self._ny or not val.shape[1] == self._nx or not val.dtype == np.uint8:
			raise ValueError(f"If you want to directly set customs boundary conditions, please provide a 2D array of shape {self._ny},{self._nx} of type np.uint8")
		self._BCs = val
		self._boundaries = rdgd.BoundaryConditions.customs

	@property
	def dt_morpho(self):
		return self._dt_morpho

	@dt_morpho.setter
	def dt_morpho(self,val):
		self._dt_morpho = val

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
				- B.G. (last modification 30/04/2024)
		'''

		# Checking if the input is 1 or 2D
		if(isinstance(value,np.ndarray)):
			input2D = True
		else:
			input2D = False

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
