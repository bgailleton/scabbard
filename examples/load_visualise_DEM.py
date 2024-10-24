import scabbard as scb
import matplotlib.pyplot as plt

dem = scb.io.load_raster('dem.tif')
fig,ax = scb.visu.nice_terrain(dem)
plt.show()