'''
this script defines the geometry features of given elements
They are made to be accessed from the top level (end users)


B.G.
'''

import numpy as np
from abc import ABC, abstractmethod


class BaseGeometry(ABC):
	"""
	Base abstract class for the geometry objects
	Defines all the functions that have to be defined for each geometry type, 
	even if only to throw an error if not possible (e.g. row and cols for a irregular grid)

	Authors:
		- B.G. (last modifications: 08/2024)


	"""

	def __init__(self):
		self._crs = None

	@property
	@abstractmethod
	def N(self):
		'''
		return the number of nodes in the element

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass

	@property
	def nxy(self):
		'''
		Alias for the number of nodes

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self.N

	@property
	@abstractmethod
	def dx(self):
		'''
		return the spatial step

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass

	@property
	@abstractmethod
	def nx(self):
		'''
		return the number of nodes in the x directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass

	@property
	def ncolumns(self):
		return self.nx


	@property
	@abstractmethod
	def ny(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass

	@property
	def nrows(self):
		'''
		Alias for ny

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		return self.ny


	@property
	@abstractmethod
	def xmin(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass

	@property
	def Xmin(self):
		return self.xmin

	@property
	@abstractmethod
	def xmax(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass

	@property
	def Xmax(self):
		return self.xmax

	@property
	@abstractmethod
	def ymin(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass

	@property
	def Ymin(self):
		return self.ymin

	@property
	@abstractmethod
	def ymax(self):
		'''
		return the number of nodes in the y directions

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass

	@property
	def Ymax(self):
		return self.ymax


	@property
	@abstractmethod
	def shape(self):
		pass

	@property
	def crs(self):
		return crs

	@abstractmethod
	def row_col_to_flatID(self, row, col):
		'''
		Take row col (single or array) and returns the flat index for regular datas

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass

	@abstractmethod
	def flatID_to_row_col(self, flatID):
		'''
		Take row col (single or array) and returns the flat index for regular datas

		Authors:
		- B.G. (last modifications: 08/2024)
		'''
		pass


	@abstractmethod
	def row_col_to_X_Y(self, row, col):
		'''
		Converts row col (for regular grids) to X Y coordinates (real world)

		Authors:
		- B.G. (last modifications: 08/2024)
		'''

		pass



	@abstractmethod
	def X_Y_to_row_col(self, X, Y):
		'''
		Converts  X Y coordinates (real world) to row col (for regular grids)

		Authors:
		- B.G. (last modifications: 08/2024)
		'''

		pass

	@abstractmethod
	def flatID_to_X_Y(self, flatID):
		'''
		Converts  flat ID to XY coordinates

		Authors:
		- B.G. (last modifications: 08/2024)
		'''

		pass

	@abstractmethod
	def X_Y_to_flatID(self, X, Y):
		'''
		Converts  X Y coordinates (real world) to flat ID (for regular grids)

		Authors:
		- B.G. (last modifications: 08/2024)
		'''

		pass


	@property
	def extent(self):
		'''
		Matplotlib friendly extent argument for imshow or more generally any bounding box

		Authors:
		- B.G. (last modifications: 08/2024)

		'''
		return [self.xmin,self.xmax, self.ymax, self.ymin]


