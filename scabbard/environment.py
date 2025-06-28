import numpy as np
import matplotlib.pyplot as plt
import dagger as dag
import scabbard as scb


class Environment(object):
	"""
	Represents the simulation environment for geomorphic models.

	This class acts as a container for various components of the simulation
	environment, including the grid, data fields, and connections to external
	libraries like Dagger for computations. It provides methods for initializing
	and accessing data within the environment.

	Attributes:
		grid (scabbard.RGrid): The regular grid object defining the spatial domain.
		data (dagger.Hermes): The data container from the Dagger library, holding
							simulation fields like surface elevation, water depth, etc.
		graph (object): Placeholder for a graph object (e.g., flow network).
		connector (dagger.Connector8): The Dagger connector object for managing
							grid connectivity and flow routing.
		graphflood (dagger.GF2): The Dagger GraphFlood object for flood simulation.
		param (dagger.ParamBag): A parameter bag from Dagger for general parameters.

	Author: B.G.
	"""
	def __init__(self):
		
		super(Environment, self).__init__()

		self.grid = None
		self.data = None
		self.graph = None
		self.connector = None
		self.graphflood = None
		self.param = dag.ParamBag()

	def init_connector(self):
		"""
		Initializes the Dagger connector.

		This method sets up the connectivity and prepares the connector for computations.
		TODO: Add parameters to set up boundary conditions and other connector-specific settings.

		Returns:
			None

		Author: B.G.
		"""
		self.connector.init()
		self.connector.compute()



	def init_GF2(self):
		"""
		Initializes the Dagger GraphFlood (GF2) object.

		This method sets up the GraphFlood solver within the environment.
		TODO: Provide more detailed documentation for GF2 initialization.

		Returns:
			None

		Author: B.G.
		"""
		self.graphflood = dag.GF2(self.connector,0,self.data, self.param)
		self.graphflood.init()


	def change_BCs(self, BCs):
		"""
		Changes the boundary conditions of the environment.

		This method updates the boundary conditions in the Dagger data container
		and reinitializes the connector with the new boundary settings.

		Args:
			BCs (numpy.ndarray): A NumPy array representing the new boundary conditions.

		Returns:
			None

		Author: B.G.
		"""
		self.data.set_boundaries(BCs.ravel())
		self.connector = dag.Connector8(self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.data)
		self.connector.set_condou(dag.CONBOU.CUSTOM)
		self.connector.init()

	def d(self, data_name = 'surface'):
		"""
		Shortcut function for `get_data`.

		Args:
			data_name (str, optional): The name of the data to retrieve. Defaults to 'surface'.

		Returns:
			numpy.ndarray: The 2D NumPy array corresponding to the requested data.

		Author: B.G.
		"""
		return self.get_data(data_name);

	def get_data(self, data_name = 'hw'):
		"""
		Returns the 2D NumPy array corresponding to the specified data field.

		Args:
			data_name (str, optional): The name of the data to retrieve. Supported names
								(case-insensitive, spaces ignored) include 'hw', 'Qw', 'Qwin',
								'Qwout', 'Qs', 'Qsin', 'Qsout', 'surface', 'bed_surface'.
								Defaults to 'hw'.

		Returns:
			numpy.ndarray: The 2D NumPy array of the requested data, reshaped to the grid's dimensions.

		Raises:
			ValueError: If `data_name` is not a recognized data identifier.

		Author: B.G.
		"""

		if(data_name.lower().replace(' ','') == 'hw'):
			return self.data.get_hw().reshape(self.grid.rshp)
		elif(data_name.replace(' ','') == 'Qw' or data_name.replace(' ','') == 'Qwin'):
			return self.data.get_Qwin().reshape(self.grid.rshp)
		elif(data_name.replace(' ','') == 'Qwout'):
			return self.data.get_Qwout().reshape(self.grid.rshp)
		elif(data_name.replace(' ','') == 'Qs' or data_name.replace(' ','') == 'Qsin'):
			return self.data.get_Qsin().reshape(self.grid.rshp)
		elif(data_name.replace(' ','') == 'Qsout'):
			return self.data.get_Qsout().reshape(self.grid.rshp)
		elif(data_name.replace(' ','') == 'surface'):
			return self.data.get_surface().reshape(self.grid.rshp)
		elif(data_name.replace(' ','') == 'bed_surface'):
			return self.data.get_surface().reshape(self.grid.rshp) if(self.data.get_hw().shape[0] == 0) else (self.data.get_surface() - self.data.get_hw()).reshape(self.grid.rshp)
		else:
			raise ValueError(data_name + " is not recognised as a data identifier");









def env_from_DEM(fname):
	"""
	Loads a raster file into an `Environment` object with default connector settings.

	Args:
		fname (str): The filename of the raster to load.

	Returns:
		Environment: An initialized `Environment` object containing the loaded DEM.

	Author: B.G.
	"""
	env = Environment()
	env.grid = scb.grid.raster2RGrid(fname, np.float64)
	env.data= dag.Hermes()
	env.data.set_surface(env.grid._Z)
	env.connector = dag.Connector8(env.grid.nx, env.grid.ny, env.grid.dx, env.grid.dy, env.data)
	return env


def env_from_slope(
	nx = 512, 
	ny = 512, 
	dx = 5,
	dy = 5,
	z_base = 0,
	slope = 1e-3, 
	noise_magnitude = 1,
	EW = "periodic",
	S = "out",
	N = "force"

	):
	"""
	Generates a sloping grid and initializes an `Environment` object with specified boundary conditions.

	This function creates a synthetic topographic grid with a constant slope and optional noise.
	It then sets up an `Environment` object with a Dagger connector, configured with the given
	boundary conditions.

	Args:
		nx (int, optional): Number of columns in the grid. Defaults to 512.
		ny (int, optional): Number of rows in the grid. Defaults to 512.
		dx (float, optional): Spatial step in the x-direction. Defaults to 5.
		dy (float, optional): Spatial step in the y-direction. Defaults to 5.
		z_base (float, optional): Base elevation for the topography. Defaults to 0.
		slope (float, optional): The slope of the generated topography. Defaults to 1e-3.
		noise_magnitude (float, optional): Magnitude of random noise added to the topography.
								Defaults to 1.
		EW (str, optional): Boundary condition for East-West boundaries. Can be "periodic",
						"noflow", "out", or "force". Defaults to "periodic".
		S (str, optional): Boundary condition for South boundary. Can be "force", "out", or
						"noflow". Defaults to "out".
		N (str, optional): Boundary condition for North boundary. Can be "force", "out", or
						"noflow". Defaults to "force".

	Returns:
		Environment: An initialized `Environment` object with the generated sloping grid.

	Author: B.G.
	"""

	# Initialize topography with zeros
	Z = np.zeros(ny * nx)

	# Create a RegularGrid object
	grid = scb.RGrid(nx, ny, dx, dy, Z)

	# Get X, Y, Z coordinates from the grid
	X,Y,Z = grid.XYZ

	# Apply slope to the topography
	Z = - slope * Y
	Z += z_base -  Z.min()

	# Update the grid's internal Z array
	grid._Z = Z.ravel()

	# Add random noise if specified
	if(noise_magnitude > 0):
		grid.add_random_noise(0, noise_magnitude)

	# Initialize boundary condition array
	bc = np.ones((ny,nx),dtype = np.uint8)
	# Set East-West boundary conditions
	bc[:,[0,-1]] = 9 if EW == "periodic" else (0 if EW == "noflow" else (4 if EW == "out" else 3)) #9 is periodic, 0 is no flow, 4 is normal out
	# Set North boundary conditions
	bc[0,:] = 8 if N == "force" else (0 if N == "noflow" else (4 if N == "out" else 3))
	# Set South boundary conditions
	bc[-1,:] = 5 if S == "force" else (4 if S == "out" else 3)
	
	# Special handling for no-flow or periodic boundaries at corners
	if(EW == "noflow" or EW == "periodic"):
		bc[[0,-1],0] = 0
		bc[[0,-1],-1] = 0

	# Create and initialize the Environment object
	env = Environment()
	env.grid = grid
	env.data = dag.Hermes()
	env.data.set_surface(grid._Z)
	env.data.set_boundaries(bc.ravel())
	env.connector = dag.Connector8(env.grid.nx, env.grid.ny, env.grid.dx, env.grid.dy, env.data)
	env.connector.set_condou(dag.CONBOU.CUSTOM)

	return env


def env_from_array(
	arr,
	nx = 512, 
	ny = 512, 
	dx = 5,
	dy = 5,
	N = 'out',
	S = 'out',
	E = 'out',
	W = 'out',

	):
	"""
	Generates an `Environment` object from a NumPy array with specified boundary conditions.

	This function takes a 2D NumPy array representing topography and initializes an
	`Environment` object with a Dagger connector, configured with the given boundary conditions.

	Args:
		arr (numpy.ndarray): The 2D NumPy array of topographic elevation data.
		nx (int, optional): Number of columns in the grid. Defaults to 512.
		ny (int, optional): Number of rows in the grid. Defaults to 512.
		dx (float, optional): Spatial step in the x-direction. Defaults to 5.
		dy (float, optional): Spatial step in the y-direction. Defaults to 5.
		N (str, optional): Boundary condition for North boundary. Can be "periodic",
						"noflow", "out", "forcein", "forceout", or "noout". Defaults to 'out'.
		S (str, optional): Boundary condition for South boundary. Defaults to 'out'.
		E (str, optional): Boundary condition for East boundary. Defaults to 'out'.
		W (str, optional): Boundary condition for West boundary. Defaults to 'out'.

	Returns:
		Environment: An initialized `Environment` object with the provided topographic data.

	Author: B.G.
	"""

	def _get_bound(bound):
		"""
		Internal helper function to map string boundary types to Dagger's integer codes.
		"""
		if(bound == "periodic"):
			return 9
		elif(bound == "noflow"):
			return 0
		elif(bound == "out"):
			return 3
		elif(bound == "forcein"):
			return 8
		elif(bound == "forceout"):
			return 5
		elif(bound == "noout"):
			return 6
		else:
			raise ValueError(f"Boundary type needs to be one of ['periodic','noflow','out','forcein','forceout','noout'], but i got {bound}")


	# Flatten the input array
	Z = arr.ravel()

	# Create a RegularGrid object
	grid = scb.RGrid(nx, ny, dx, dy, Z)

	# Initialize boundary condition array
	bc = np.ones((ny,nx),dtype = np.uint8)
	# Set boundary conditions for each side
	bc[:,0] = _get_bound(W)
	bc[:,-1] = _get_bound(E)
	bc[0,:] = _get_bound(N)
	bc[-1,:] = _get_bound(S)
	
	# Special handling for no-flow or periodic boundaries at corners
	if(E == "noflow" or E == "periodic" or W == "noflow" or W == "periodic"):
		bc[[0,-1],0] = 0
		bc[[0,-1],-1] = 0

	# Create and initialize the Environment object
	env = Environment()
	env.grid = grid
	env.data = dag.Hermes()
	env.data.set_surface(grid._Z)
	env.data.set_boundaries(bc.ravel())
	env.connector = dag.Connector8(env.grid.nx, env.grid.ny, env.grid.dx, env.grid.dy, env.data)
	env.connector.set_condou(dag.CONBOU.CUSTOM)

	return env





