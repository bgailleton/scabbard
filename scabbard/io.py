"""
This class deals with loading raster informations
Authors: B.G.
"""
import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds
import dagger as dag


def load_raster(fname):
	"""
	Load a raster array with different options. It uses rasterio that itself uses gdal.
	Arguments:
		fname (str): raster to load (path+file_name+format)
	Returns:
		A python dictionnary containing the following "key" -> val:
			"res" -> Resolution of the DEM
			"ncols" -> number of columns
			"nrows" -> number of rows
			"x_min" -> well x minimum
			"y_min" -> and y minimum
			"x_max" -> and x maximum
			"y_max" -> and x maximum
			"extent" -> extent combined in order to match matplotlib
			"array" -> numpy 2D array containing the data
			"crs" -> The crs string (geolocalisation)
			"nodata" -> list of nodata values
	Authors:
		B.G.
	Date:
		23/02/2019
	"""

	# Loading the raster with rasterio
	this_raster = rio.open(fname)

	# Initialising a dictionary containing the raster info for output
	out = {}
	# I am padding a no_data contour
	gt = this_raster.res
	out['dx'] = gt[0]
	out['dy'] = gt[1]
	out["nx"] = this_raster.width
	out["ny"] = this_raster.height
	out["x_min"] = this_raster.bounds[0]
	out["y_min"] = this_raster.bounds[1]
	out["x_max"] = this_raster.bounds[2]
	out["y_max"] = this_raster.bounds[3]
	corr = out['dx'] + out['dy']
	out["extent"] = [out["x_min"],out["x_max"]-corr,out["y_min"],out["y_max"]-corr]
	out["array"] = this_raster.read(1).astype(np.float64)
	try:
		out['crs'] = this_raster.crs['init']
	except (TypeError, KeyError) as e:
		out['crs'] = u'epsg:32601'
	out['nodata'] = this_raster.nodatavals

	
	
	# pixelSizeY =-gt[4]
	# (left=358485.0, bottom=4028985.0, right=590415.0, top=4265115.0)

	return out




def raster2graphcon(file_name):
	"""
	Ingest a raster with rasterio adn load a connector and graph from it
	"""
		
	# Loading DEM data with rasterio
	dem = load_raster(file_name)

	connector = dag.D8N(dem["nx"], dem["ny"], dem["dx"], dem["dy"], dem["x_min"], dem["y_min"])
	graph = dag.graph(connector)
	
	
	
	return connector, graph, dem

def raster2con(file_name):
	"""
	Ingest a raster with rasterio adn load a connector and graph from it
	"""
		
	# Loading DEM data with rasterio
	dem = load_raster(file_name)

	connector = dag.D8N(dem["nx"], dem["ny"], dem["dx"], dem["dy"], dem["x_min"], dem["y_min"])
	
	return connector, dem