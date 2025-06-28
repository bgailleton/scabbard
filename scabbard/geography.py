'''
This module manages geographical information and conversions within the scabbard framework.
It is currently under development and will evolve significantly.
'''


class geog:
	"""
	Simple class wrapping geographical information for a grid or dataset.

	This class stores the spatial extent (minimum and maximum x and y coordinates)
	and the Coordinate Reference System (CRS) of a geographical area.
	It is designed to be a straightforward container for these properties and is
	expected to evolve with further development.

	Attributes:
		xmin (float): Minimum x-coordinate of the geographical extent.
		ymin (float): Minimum y-coordinate of the geographical extent.
		xmax (float): Maximum x-coordinate of the geographical extent.
		ymax (float): Maximum y-coordinate of the geographical extent.
		crs (object): The Coordinate Reference System object (e.g., from `pyproj` or `rasterio`).
					Defaults to None.

	Author: B.G.
	"""

	def __init__(self, 
		xmin = 0., 
		ymin = 0.,
		xmax = 0.,
		ymax = 0.,
		crs = None

	):
		
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.crs = crs


