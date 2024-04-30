'''
Parameter class and function for Riverdale
This is the user-side parameter sheet, not to be confused with the singletons parameter classes managing the compile time constant

Authors:
	- B.G.

'''

import numpy as np
from enum import Enum
import scabbard.riverdale.rd_grid as rdgd
import scabbard as scb

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
		self.BCs = None



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



	def set_initial_conditions(self, nx, ny, dx, dy, Z, hw = None, BCs = None, boundaries = rdgd.BoundaryConditions.normal):
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
		self._boundaries = boundaries
		self.BCs = BCs
		self.initial_Z = Z
		self.initial_hw = hw



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
			else:
				value = np.float32(value)
			self._RD.precipitations.from_numpy(value)



def param_from_grid(grid):
	'''
	TODo
	'''
	param = RDParams()
	param.set_initial_conditions(grid.nx, grid.ny, grid.dx, grid.dx, grid.Z2D, boundaries = rdgd.BoundaryConditions.normal)
	return param

def param_from_dem(dem):
	return param_from_grid(scb.grid.raster2RGrid(dem))
