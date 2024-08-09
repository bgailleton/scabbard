import numpy as np
import matplotlib.pyplot as plt
import scabbard.visu.base as base


def nice_terrain(dem, cmap = 'terrain', label = 'Metrics', alpha = 0.6, 
	cut_off_min = None, cut_off_max = None, sea_level = None, vmin = None,
	vmax = None, **kwargs):
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
		- **kwargs: anything that goes into fig, ax = plt.subplots(...)
	Returns:
		- fig,ax  with everything plotted on it

	Author:
		- B.G. (last modification: 08/2024)
	'''
	res = {}

	fig,ax = base.hs_drape(dem, dem.Z2D, cmap = 'cividis', label = 'Metrics', alpha = 0.6, 
	cut_off_min = None, cut_off_max = None, sea_level = None, vmin = None,
	vmax = None, outputs = res,  **kwargs)

	return fig,ax

