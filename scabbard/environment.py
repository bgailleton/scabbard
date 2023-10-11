import numpy as np
import matplotlib.pyplot as plt
import dagger as dag
import scabbard as scb


class Environment(object):
	"""
	docstring for Environment
	"""
	def __init__(self):
		
		super(Environment, self).__init__()

		self.grid = None
		self.data = None
		self.graph = None
		self.connector = None
		self.graphflood = None

	def init_connector(self):
		'''
		TODO:: add aprameters to set up boundary conditions and all
		'''
		self.connector.init()
		self.connector.compute()



	def init_GF2(self):
		'''
		TODO:: WRITE THE TO DO
		'''
		self.graphflood = dag.GF2(self.connector,0,self.data)
		self.graphflood.init() 










def env_from_DEM(fname):
	'''
		Load a raster file into an environment with default connector and all
	'''
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
	'''
		Generates a slopping grid adn initialise an env with given boundaries to it. 
		It comes with a connector and a graph (todo) alredy pre-prepared for boundaries
	'''

	# Zeros Topo
	Z = np.zeros(ny * nx)

	# Generating the grid
	grid = scb.RGrid(nx, ny, dx, dy, Z)

	# Getting the matrix
	X,Y,Z = grid.XYZ

	Z = - slope * Y
	Z += z_base -  Z.min()

	grid._Z = Z.ravel()

	if(noise_magnitude > 0):
		grid.add_random_noise(0, noise_magnitude)

	bc = np.ones((ny,nx),dtype = np.uint8)
	bc[:,[0,-1]] = 9 if EW == "periodic" else (0 if EW == "noflow" else (4 if EW == "out" else 3)) #9 is periodic, 0 is no flow, 4 is normal out
	bc[0,:] = 8 if N == "force" else (0 if N == "noflow" else (4 if N == "out" else 3))
	bc[-1,:] = 5 if S == "force" else (4 if S == "out" else 3)
	
	if(EW == "noflow" or EW == "periodic"):
		bc[[0,-1],0] = 0
		bc[[0,-1],-1] = 0

	env = Environment()
	env.grid = grid
	env.data = dag.Hermes()
	env.data.set_surface(grid._Z)
	env.data.set_boundaries(bc.ravel())
	env.connector = dag.Connector8(env.grid.nx, env.grid.ny, env.grid.dx, env.grid.dy, env.data)
	env.connector.set_condou(dag.CONBOU.CUSTOM)

	return env


