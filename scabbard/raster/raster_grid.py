'''
Describes a raster-like regular grid.
Will replace the Rgrid object at some points
WIP
'''


import numpy as np
import matplotlib.pyplot as plt
from scabbard import io
from scabbard import geometry as geo
import dagger as dag
from scipy.ndimage import gaussian_filter
import random

class RegularRasterGrid(object):

	"""
	Manages a regular grid with helper functions
	"""
	

	def __init__(self, value, geometry, dtype = np.float32):

		'''
			
		'''

		super().__init__()

		# Check compiance with geometry:
		if(isinstance(geometry, geo.RegularGeometry) == False):
			raise AttributeError("a RegularRasterGrid object must be created with a geometry of type RegularGeometry")
			# Checking if the geometry correspond to the shape of the array
			if(Z.shape != geometry.shape):
				raise AttributeError("Matrix not the size indicated in geometry when trying to instanciate RegularRasterGrid")

		# All good then I can instanciate the values
		self.Z = value
		self.geo = geometry

		# Conveting type if needed
		if(self.Z.dtype != dtype):
			self.Z = self.Z.astype(dtype)

		


		

	