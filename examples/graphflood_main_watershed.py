'''
In this example, I 
'''
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import scabbard as scb


# Loading the DEM
grid = scb.io.load_raster('dem.tif')

# Isolating the main drainage basin
## this function returns a mask of 0,1 showing the main watershed
mask = scb.flow.mask_main_basin(grid, MFD = False)
## You could also, alternatively, mask seas in case you wanna remove all the data below a certain values
# mask = mask_seas(grid, sea_level = 0.)

## Encode the mask into boundary conditions
BCs = scb.flow.mask_to_BCs(grid,mask)

# Change this variable to change the backend:

## GPU backend with taichi 
### (if you have a MAC this remains untested but theoretically works)
### (If you are on Linux/Windows WITHOUT Nvidia GPU, you need to install the vulkan SDK/drivers -> https://vulkan.lunarg.com/)
backend = 'gpu'
## CPU backend with DAGGER (OG graphflood - both options work)
# backend = 'cpu'
# backend = 'dagger'

## CPU libTopoToolbox backend
# backend = 'ttb'

# running Graphflood dor 1000 iterations
results = scb.graphflood.std_run(
	grid, # Sting or grid so far
	P = 1e-4, # precipitations, numpy array or scalar
	BCs = BCs, # Boundary codes
	N_dt = 5000,
	backend = backend,
	dt = 1e-2,
	init_hw = None)

fig,ax = scb.visu.hillshaded_basemap(grid)


hw = results['h'].copy()
hw[hw<0.01] = np.nan
im = ax.imshow(hw, cmap = 'Blues', vmax = 3., extent = grid.geo.extent, alpha = 0.75)
plt.colorbar(im, label = 'Flow Depth (m)')
plt.show()



