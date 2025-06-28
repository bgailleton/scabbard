"""
This module provides functionalities for loading and saving raster data (e.g., DEMs) using `rasterio`.
It acts as an interface between common geospatial file formats and NumPy arrays for use within scabbard.
"""
import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds
import dagger as dag
from rasterio.transform import from_origin


def legacy_load_raster(fname):
	"""
	Loads a raster file into a Python dictionary.

	This function uses `rasterio` (which in turn uses GDAL) to read geospatial raster data
	and extracts relevant information into a dictionary format.

	Args:
		fname (str): The path to the raster file (e.g., a GeoTIFF).

	Returns:
		dict: A dictionary containing the following keys:
			- "dx" (float): Resolution in the x-direction.
			- "dy" (float): Resolution in the y-direction.
			- "nx" (int): Number of columns.
			- "ny" (int): Number of rows.
			- "x_min" (float): Minimum x-coordinate of the raster extent.
			- "y_min" (float): Minimum y-coordinate of the raster extent.
			- "x_max" (float): Maximum x-coordinate of the raster extent.
			- "y_max" (float): Maximum y-coordinate of the raster extent.
			- "extent" (list): [xmin, xmax, ymin, ymax] formatted for Matplotlib.
			- "array" (numpy.ndarray): 2D NumPy array containing the data.
			- "crs" (str): The Coordinate Reference System (CRS) string.
			- "nodata" (list): List of nodata values.

	Author: B.G.
	Date: 23/02/2019
	"""

	# Loading the raster with rasterio
	this_raster = rio.open(fname)

	# Initializing a dictionary to store raster information
	out = {}
	# Get resolution
	gt = this_raster.res
	out['dx'] = gt[0]
	out['dy'] = gt[1]
	# Get dimensions
	out["nx"] = this_raster.width
	out["ny"] = this_raster.height
	# Get bounds
	out["x_min"] = this_raster.bounds[0]
	out["y_min"] = this_raster.bounds[1]
	out["x_max"] = this_raster.bounds[2]
	out["y_max"] = this_raster.bounds[3]
	# Calculate correction for extent (often needed for plotting)
	corr = out['dx'] + out['dy']
	out["extent"] = [out["x_min"],out["x_max"]-corr,out["y_min"],out["y_max"]-corr]
	# Read raster data into a NumPy array
	out["array"] = this_raster.read(1).astype(np.float64)
	# Get CRS information
	try:
		out['crs'] = this_raster.crs['init']
	except (TypeError, KeyError) as e:
		out['crs'] = u'epsg:32601' # Default CRS if not found
	# Get nodata values
	out['nodata'] = this_raster.nodatavals

	return out




def raster2graphcon(file_name):
	"""
	Loads a raster file and initializes Dagger connector and graph objects from it.

	Args:
		file_name (str): The path to the raster file.

	Returns:
		tuple: A tuple containing:
			- connector (dagger.D8N): The Dagger D8N connector object.
			- graph (dagger.graph): The Dagger graph object.
			- dem (dict): The dictionary containing raster information (as returned by `load_raster`).

	Author: B.G.
	"""
		
	# Loading DEM data with rasterio
	dem = legacy_load_raster(file_name)

	# Initialize Dagger connector and graph
	connector = dag.D8N(dem["nx"], dem["ny"], dem["dx"], dem["dy"], dem["x_min"], dem["y_min"])
	graph = dag.graph(connector)
	
	return connector, graph, dem

def raster2con(file_name):
	"""
	Loads a raster file and initializes a Dagger connector object from it.

	Args:
		file_name (str): The path to the raster file.

	Returns:
		tuple: A tuple containing:
			- connector (dagger.D8N): The Dagger D8N connector object.
			- dem (dict): The dictionary containing raster information (as returned by `load_raster`).

	Author: B.G.
	"""
		
	# Loading DEM data with rasterio
	dem = legacy_load_raster(file_name)

	# Initialize Dagger connector
	connector = dag.D8N(dem["nx"], dem["ny"], dem["dx"], dem["dy"], dem["x_min"], dem["y_min"])
	
	return connector, dem


# import rasterio
def save_raster(file_name, array, x_min, x_max, y_min, y_max, dx, dy, crs = None):
	"""
	Saves a NumPy array as a GeoTIFF raster file.

	Args:
		file_name (str): The path and filename for the output GeoTIFF.
		array (numpy.ndarray): The 2D NumPy array to save.
		x_min (float): Minimum x-coordinate of the raster extent.
		x_max (float): Maximum x-coordinate of the raster extent.
		y_min (float): Minimum y-coordinate of the raster extent.
		y_max (float): Maximum y-coordinate of the raster extent.
		dx (float): Pixel size in the x-direction.
		dy (float): Pixel size in the y-direction.
		crs (str, optional): The Coordinate Reference System (CRS) string (e.g., "EPSG:4326").
						Defaults to "EPSG:35653" if None.

	Returns:
		None: A GeoTIFF file is created at the specified `file_name`.

	Author: B.G.
	"""
	# Define file mode
	file_mode = "w+"

	# Specify dimensions of the array
	height, width = array.shape
	count = 1  # Number of bands in the array
	dtype = array.dtype

	# Specify transformation parameters
	x_res = dx  # Pixel size in the x-direction
	y_res = dy  # Pixel size in the y-direction
	# Create a transform matrix from origin and resolutions
	transform = from_origin(x_min, y_max, dx, dy)

	# Set default CRS if not provided
	if(crs is None):
		crs = "EPSG:35653"

	# Create output GeoTIFF file
	with rio.open(file_name, file_mode, driver='GTiff', height=height, width=width, dtype=dtype, count=count, transform=transform, nodata = -9999, crs = crs) as dst:
	    dst.write(array, 1)  # Write the array to the GeoTIFF file as the first band (band index is 1-based)
