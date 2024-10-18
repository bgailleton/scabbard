'''
this script defines the geometry features of given elements
They are made to be accessed from the top level (end users)


B.G.
'''

import numpy as np
from .geometry import BaseGeometry

class RegularGeometry(BaseGeometry):
	"""
	geometry class for any object with a regular 2D structure
	that can be described with a number of rows and col

	Authors:
		- B.G. (last modifications: 08/2024)


	"""

	def __init__(self, nx, ny, dx, xmin, ymin):
		'''
		Creates the Regular Geometry object

		Arguments:
			- nx: number of columns
			- ny: number of rows
			- dx: spatial step
			- xmin: minimum Easting coordinate
			- ymin: minimum Northing coordinate
		
		Returns:
			- an instance of RegularGeometry

		Authors:
			- B.G. (last modification: 08/2024)
		'''

		super().__init__()

		self._nx = nx
		self._ny = ny
		self._nxy = nx * ny
		self._dx = dx
		self._lx = dx * (nx+1)
		self._ly = dx * (ny+1)
		self._dxy = 2**0.5 * dx
		self._xmin = xmin
		self._ymin = ymin


	@property
	def dx(self):
		'''
		return the spatial step

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._dx

	@property
	def dxy(self):
		'''
		return the spatial step

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._dxy

	@property
	def N(self):
		'''
		return the number of nodes in the element

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._nxy


	@property
	def nx(self):
		'''
		return the number of nodes in the x directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._nx


	@property
	def ny(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._ny

	@property
	def lx(self):
		'''
		return the length in the x directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._lx


	@property
	def ly(self):
		'''
		return the length in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._ly

	def shape(self):
		'''
		Returns a numpy=like shape
		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return (self.ny,self.nx)


	@property
	def xmin(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._xmin

	@property
	def xmax(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._xmin + (self.nx + 1) * self.dx

	@property
	def ymin(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._ymin

	@property
	def ymax(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self._ymin + (self.ny + 1) * self.dx



	def row_col_to_flatID(self, row, col):
		'''
		Take row col (single or array) and returns the flat index for regular datas

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return row * self.nx + col

	def flatID_to_row_col(self, flatID):
		'''
		Take row col (single or array) and returns the flat index for regular datas

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return flatID // self.nx, flatID % self.nx


	def row_col_to_X_Y(self, row, col):
		'''
		Converts row col (for regular grids) to X Y coordinates (real world)

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return (col * self.dx) + self.xmin + self.dx/2, ( (self.ny - 1 - row) * self.dx) + self.ymin + self.dx/2 



	def X_Y_to_row_col(self, X, Y):
		'''
		Converts  X Y coordinates (real world) to row col (for regular grids)

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self.ny - 1 - np.floor(Y - self.ymin), np.floor(X - self.xmin)

	def flatID_to_X_Y(self, flatID):
		'''
		Converts  flat ID to XY coordinates

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		row,col = self.flatID_to_row_col(flatID)
		return self.row_col_to_X_Y(row,col)
		

	def X_Y_to_flatID(self, X, Y):
		'''
		Converts  X Y coordinates (real world) to flat ID (for regular grids)

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		row,col = self.X_Y_to_row_col(X,Y)
		return self.row_col_to_flatID(row,col)

	@property
	def xl_centered(self):
		return self.xmin+self.dx/2.

	@property
	def xr_centered(self):
		return self.xmax-self.dx/2.

	@property
	def yt_centered(self):
		return self.ymin+self.dx/2.

	@property
	def yb_centered(self):
		return self.ymax-self.dx/2.

	@property
	def X(self):
		'''
		array of X indices at the center of nodes
		'''
		return np.linspace(self.xl_centered,self.xr_centered, self.nx)

	@property
	def Y(self):
		'''
		array of Y indices at the center of nodes
		'''
		return np.linspace(self.yt_centered, self.yb_centered, self.ny)

	@property
	def XY(self):
		'''
		return a mesh grid of X and Y coorditates
		i.e. 2 2d arrays of Xs and Ys for each nodes
		'''
		return np.meshgrid(self.X, self.Y)

	@property
	def dxnxny(self):
		return self.dx,self.nx,self.ny	 

