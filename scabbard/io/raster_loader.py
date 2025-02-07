"""
This class deals with loading raster informations
Authors: B.G.
"""
import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.transform import from_origin
import scabbard as scb

def load_raster(fname, dtype = np.float32):
	"""
	Load a raster file into a RegularRasterGrid object
	Made to read DEMs and DEM-like data (i.e. single band geolocalised raster data), does not handle multi-band.
	I can make a multi-band option if needed
	Uses rasterio and gdal behind the scene, see https://gdal.org/drivers/raster/index.html for accepted formats

	Arguments:
		- fname (str or any sort of path): full file name + path if not local
		- dtype (numpy type): forces a type to the raster, by default float 32 bits (single recision) 
	
	Returns:
		- The raster object with data loaded
	
	Authors:
		- B.G. (last modification: 08/2024)

	"""

	# Loading the raster with rasterio
	this_raster = rio.open(fname)

	# Getting the resolution
	gt = this_raster.res
	
	# Creating the underlying geomoetry object
	geom = scb.geometry.RegularGeometry(this_raster.width, this_raster.height, gt[0], this_raster.bounds[0], this_raster.bounds[1])
	
	# Getting the actual data
	Z = this_raster.read(1).astype(dtype)
	
	# NO no data handling so far
	# Z[Z == this_raster.nodatavals] = np.nan

	# Checks if the DEM has a crs or not
	try:
		geom._crs = this_raster.crs['init']
	except (TypeError, KeyError) as e:
		geom._crs = u'epsg:32601'

	return scb.raster.RegularRasterGrid(Z, geom, dtype = dtype)


def save_raster(grid, fname, crs='EPSG:32601', dtype=np.float32, driver = 'GTiff'):
	"""
	Save a raster to a file using rasterio.

	Arguments:
		- raster_data (numpy array): The raster data to save.
		- fname (str or any sort of path): Full file name + path if not local.
		- crs (str): Coordinate Reference System (default is 'EPSG:32601').
		- dtype (numpy type): Data type of the raster (default is float32).

	Returns:
		- None

	Authors:
		- Your Name (last modification: 02/2025)
	"""

	# Assuming raster_data is a 2D numpy array
	height, width = grid.Z.shape
	
	if(grid.Z.dtype != dtype):
		grid.Z = grid.Z.astype(dtype)


	# Define the transform (affine transformation)
	# This is an example; you need to adjust it based on your actual data
	transform = rio.transform.from_origin(grid.geo.xmin, grid.geo.ymax, grid.geo.dx, grid.geo.dx)

	# Define the metadata
	meta = {
		'driver': driver,
		'height': height,
		'width': width,
		'count': 1,
		'dtype': dtype,
		'crs': crs,
		'transform': transform
	}

	print(meta)

	# Write the raster data to a file
	with rio.open(fname, 'w', **meta) as dst:
		dst.write(grid.Z.astype(dtype), 1)


def save_ascii_grid(grid, fname, dtype = np.float32):
	"""
	Save a raster to an ASCII grid file.

	Arguments:
		- raster_data (numpy array): The raster data to save.
		- fname (str or any sort of path): Full file name + path if not local.
		- xllcorner (float): X-coordinate of the lower-left corner of the grid.
		- yllcorner (float): Y-coordinate of the lower-left corner of the grid.
		- cellsize (float): Size of each cell in the grid.
		- nodata_value (float): Value to represent no data in the grid (default is -9999).

	Returns:
		- None

	Authors:
		- Your Name (last modification: 02/2025)
	"""

	# Assuming raster_data is a 2D numpy array
	nrows, ncols = grid.Z.shape

	if(grid.Z.dtype != dtype):
		grid.Z = grid.Z.astype(dtype)
	
	# Open the file for writing
	with open(fname, 'w') as f:
		# Write the header
		f.write(f"ncols         {ncols}\n")
		f.write(f"nrows         {nrows}\n")
		f.write(f"xllcorner     {grid.geo.xmin}\n")
		f.write(f"yllcorner     {grid.geo.ymin}\n")
		f.write(f"cellsize      {grid.geo.dx}\n")
		f.write(f"NODATA_value  {-9999}\n")

		grid.Z[np.isfinite(grid.Z) == False] = -9999

		# Write the raster data
		for row in grid.Z:
			f.write(' '.join(map(str, row)) + '\n')
