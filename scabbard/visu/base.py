'''
These scripts provide 
'''


import matplotlib.pyplot as plt
import numpy as np
import scabbard as scb



def _legacy_hillshaded_basemap(dem, sea_level = None, fig = None, ax = None, **kwargs):
	'''
	Ongoing deprecation: switching to the new grid system
	Return a fig,ax with a hillshaded relief to the right extent. Ready to be used as base map to plot something on the top

	Arguments:
		- dem: the Rgrid/DEM object with the topography
		- sea_level: if not None, remove everything < sea_level
		- **kwargs: any arguments that would go with matplotlib.pyplot.subplots

	Returns:
		- fig,ax objects

	Authors:
		- B.G.

	'''

	# Creating the figure
	if(fig is None):
		fig,ax = plt.subplots(1, 1, **kwargs)

	# array to plot
	tp = scb.rvd.std_hillshading(dem.Z2D, direction = 40., inclinaison = 55., exaggeration = 1.2, use_gpu = False, D4 = True, dx = dem.dx) + scb.rvd.std_hillshading(dem.Z2D, direction = 85., inclinaison = 55., exaggeration = 1.2, use_gpu = False, D4 = True, dx = dem.dx)
	
	tp /= 2
	
	if not (sea_level is None):
		tp[dem.Z2D<sea_level] = np.nan

	ax.imshow(
		tp,
		cmap = 'gray',
		extent = dem.extent()
		)

	ax.set_xlabel("Easting (m)")
	ax.set_ylabel("Northing (m)")
	
	return fig, ax



def hillshaded_basemap(dem, sea_level = None, fig = None, ax = None, use_gpu = False, mask = None, **kwargs):
	'''
	Return a fig,ax with a hillshaded relief to the right extent. Ready to be used as base map to plot something on the top

	Arguments:
		- dem: raster object with the topography, will be redirected to the right function if former Rgrid format
		- sea_level: if not None, remove everything < sea_level
		- **kwargs: any arguments that would go with matplotlib.pyplot.subplots

	Returns:
		- fig,ax objects

	Authors:
		- B.G.

	'''

	# Legacy compatibility layer
	if(isinstance(dem, scb.RGrid)):
		return _legacy_hillshaded_basemap(dem, sea_level = sea_level,fig = fig, ax = ax, **kwargs)

	# Creating the figure
	if(fig is None):
		fig,ax = plt.subplots(1, 1, **kwargs)

	# array to plot
	tp = scb.rvd.std_hillshading(dem.Z, 
		direction = 40., inclinaison = 55., exaggeration = 1.2, 
		use_gpu = use_gpu, D4 = True, dx = dem.geo.dx) + scb.rvd.std_hillshading(dem.Z, 
		direction = 85., inclinaison = 55., exaggeration = 1.2, use_gpu = use_gpu,
		 D4 = True, dx = dem.geo.dx)
	
	tp /= 2
	
	if not (sea_level is None):
		tp[dem.Z<sea_level] = np.nan

	if not (mask is None):
		tp[mask] = np.nan

	ax.imshow(
		tp,
		cmap = 'gray',
		extent = dem.geo.extent
		)

	ax.set_xlabel("Easting (m)")
	ax.set_ylabel("Northing (m)")


	return fig, ax


def hs_drape(dem, arr2D, cmap = 'cividis', label = 'Metrics', alpha = 0.6, 
	cut_off_min = None, cut_off_max = None, sea_level = None, vmin = None,
	vmax = None, res = None, fig = None, ax = None, mask = None, kwargs_imshow = None, **kwargs):
	'''
	Quick visualisation of a DEM as a hillshade + an imshow-like data on the top of it with the same extent

	Arguments:
		- dem: the topographic grid
		- arr2D: the array to be draped, to the extent of the dem
		- cmap: the colormap to use
		- label: label of the colorbar
		- alpha: the transparency of the arr2D
		- cut_off_min/max: any data on arr2D below or above this value will be ignored
		- sea_level: any data on topo and arr2D where topo < that value will be ignored
		- vmin/vmax: the extents of the colors for the cmap of arr2D
		- mask: a binary mask to set additional nodata sections manually
		- **kwargs: anything that goes into fig, ax = plt.subplots(...)
	Returns:
		- fig,ax  with everything plotted on it

	Author:
		- B.G. (last modification: 08/2024)
	'''

	if(fig is None):
		fig, ax = hillshaded_basemap(dem, sea_level = sea_level, mask = mask, **kwargs)
	else:
		hillshaded_basemap(dem, sea_level = sea_level, fig = fig, ax = ax, mask = mask, **kwargs)

	if(isinstance(dem, scb.RGrid)):
		legacy = True
	else:
		legacy = False

	tp = arr2D.copy()
	if not (cut_off_min is None):
		tp[arr2D<cut_off_min] = np.nan
	if not (cut_off_max is None):
		tp[arr2D>cut_off_max] = np.nan

	if not (sea_level is None ):
		tp[(dem.Z2D <= sea_level) if legacy else (dem.Z <= sea_level) ] = np.nan

	if not (mask is None ):
		tp[mask] = np.nan
	

	if(kwargs_imshow is None):
		im = ax.imshow(tp, extent = dem.extent() if legacy else dem.geo.extent, cmap = cmap, alpha = alpha, vmin = vmin, vmax = vmax)
	else:
		im = ax.imshow(tp, extent = dem.extent() if legacy else dem.geo.extent, **kwargs_imshow)

	if(isinstance(res,dict)):
		res['fig'] = fig
		res['ax'] = ax
		res['im'] = im


	return fig,ax
