'''
grid module to help with generic grid manipulations
'''
import numpy as np
from scabbard import io
from scabbard import geography as geo
import dagger as dag
from scipy.ndimage import gaussian_filter
import random

class RGrid(object):

	"""
	Small helper class for regular grid
	"""
	

	def __init__(self, nx, ny, dx, dy, Z, geography = None):

		super(RGrid, self).__init__()
		
		# Number of col
		self.nx = nx
		# Number of rows
		self.ny = ny
		#nnodes
		self.nxy = self.nx * self.ny
		# Spatial step in X dir
		self.dx = dx
		# Spatial step in Y dir
		self.dy = dy

		self.lx = (nx+1) * dx
		self.ly = (ny+1) * dy

		# Converts 1D flattened to 2D grid
		self.rshp = (ny,nx)

		if(geography is None):
			self.geography = geo.geog(xmin = 0., ymin = 0., xmax = self.lx, ymax = self.ly)
		else:
			self.geography = geography

		self._Z = Z.ravel()

		self.con = None

		

	def extent(self, y_min_top = True):
		return [self.geography.xmin, self.geography.xmax, self.geography.ymin if y_min_top else self.geography.ymax, self.geography.ymax if y_min_top else self.geography.ymin ]


	@property
	def X(self):
		return np.linspace(self.geography.xmin + self.dx/2, self.geography.xmax - self.dx/2, self.nx)

	@property
	def Y(self):
		return np.linspace(self.geography.ymin + self.dy/2, self.geography.ymax - self.dy/2, self.ny)

	@property
	def Z(self):
		return self._Z

	@property
	def Z2D(self):
		return self._Z.reshape(self.ny,self.nx)

	@property
	def XYZ(self):
		xx,yy = np.meshgrid(self.X, self.Y)
		return xx, yy, self.Z2D

	@property
	def hillshade(self):
		if(self.con is None):
			con = dag.D8N(self.nx, self.ny, self.dx, self.dy, self.geography.xmin, self.geography.ymin)
		else:
			con = self.con
		return dag.hillshade(con, self._Z).reshape(self.rshp)

	def get_graphcon(self, process = True):
		con = dag.D8N(self.nx, self.ny, self.dx, self.dy, self.geography.xmin, self.geography.ymin)
		graph = dag.graph(con)
		if(process):
			graph.compute_graph(self._Z, True, False)

		return graph, con

	def min(self):
		return np.nanmin(self._Z)

	def max(self):
		return np.nanmax(self._Z)








def generate_noise_RGrid( 
	nx = 256, ny = 256, dx = 30., dy = 30., # Dimensions
	noise_type = "white", # noise type: white or Perlin
	magnitude = 1,
	frequency = 4., octaves = 8, seed = None, # Perlin noise options and seed
	n_gaussian_smoothing = 0 # seed
	):
	
	
	if(noise_type.lower() == "white"):
		Z = np.random.rand(nx*ny)
	elif(noise_type.lower() == "perlin"):
		con = dag.D8N(nx,ny,dx,dy,0,0)
		Z = dag.generate_perlin_noise_2D(con, frequency, octaves, np.uint32(seed) if seed is not None else np.uint32(random.randrange(0,32000) ) )

	if(n_gaussian_smoothing > 0):
		Z = gaussian_filter(Z,n_gaussian_smoothing)
	Z = Z.reshape(ny, nx)
	Z[[0,-1],:] = 0
	Z[:, [-1,0]] = 0

	return RGrid(nx, ny, dx, dy, Z, geography = None)



def raster2RGrid(fname):

	dem = io.load_raster(fname)
	geog = geo.geog(dem["x_min"],dem["x_max"],dem["y_min"],dem["y_max"],dem["crs"])
	return RGrid(dem["nx"], dem["ny"], dem["dx"], dem["dy"], dem["array"].ravel(),geography=geog)






























































# end of file