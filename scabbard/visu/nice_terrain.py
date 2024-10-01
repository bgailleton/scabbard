import numpy as np
import matplotlib.pyplot as plt
import scabbard.visu.base as base
import scabbard as scb
import click
import os

@click.command()
@click.option('-s', '--sea_level', 'sea_lvl',  type = float, default = 0.)
@click.argument('fname', type = str)
def cli_nice_terrain(fname, sea_lvl):
	
	if(os.path.exists(fname) == False):
		raise ValueError(f'FileDoesNotExists: {fname}')
	
	dem = scb.io.load_raster(fname)
	
	fig,ax = base.hs_drape(dem, dem.Z, cmap = 'terrain', label = 'Elevation', alpha = 0.4, 
	cut_off_min = None, cut_off_max = None, sea_level = sea_lvl, vmin = None,
	vmax = None, res = None, fig = None, ax = None)
	
	plt.show()

def nice_terrain(dem, cmap = 'terrain', alpha = 0.55, 
	sea_level = None, vmin = None,
	vmax = None, res = None, mask = None, **kwargs):
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

	fig,ax = base.hs_drape(dem, (dem.Z if isinstance(dem,scb.raster.RegularRasterGrid) else dem.Z2D), cmap = cmap, label = 'Elevation', alpha = alpha, 
		sea_level = sea_level, vmin = vmin, vmax = vmax, mask = mask, **kwargs)

	return fig,ax

