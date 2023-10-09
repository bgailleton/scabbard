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










def load_DEM(fname):
	'''
		Load a raster file into an environment with default connector and all
	'''
	env = Environment()
	env.grid = scb.grid.raster2RGrid(fname, np.float64)
	env.data= dag.Hermes()
	env.data.set_surface(env.grid._Z)
	env.connector = dag.Connector8(env.grid.nx, env.grid.ny, env.grid.dx, env.grid.dy, env.data)
	return env



