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