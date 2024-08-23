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
	
	"""

	# Loading the raster with rasterio
	this_raster = rio.open(fname)

	geom = scb.geometry.RegularGeometry(this_raster.width, this_raster.height, gt[0], this_raster.bounds[0], this_raster.bounds[1])
	Z = this_raster.read(1).astype(dtype)
	Z[Z == this_raster.nodatavals] = np.nan

	try:
		geom._crs = this_raster.crs['init']
	except (TypeError, KeyError) as e:
		geom._crs = u'epsg:32601'

	return scb.raster.RegularRasterGrid(Z, geom, dtype = dtype)