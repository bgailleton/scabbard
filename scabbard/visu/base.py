'''
These scripts provide 
'''


import matplotlib.pyplot as plt
import numpy as np
import scabbard as scb




def hillshaded_basemap(dem, **kwargs):
	'''
		
	'''

	# Creating the figure
	fig,ax = plt.subplots(**kwargs)

	ax.imshow(
		scb.rvd.std_hillshading(dem.Z2D, direction = 40., inclinaison = 55., exaggeration = 1.2, use_gpu = False, D4 = True, dx = dem.dx),
		cmap = 'gray',
		extent = dem.extent()
		)

	ax.set_xlabel("Easting (m)")
	ax.set_ylabel("Northing (m)")

	return fig, ax