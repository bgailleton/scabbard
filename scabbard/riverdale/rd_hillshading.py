'''
Sets of function to compute hydrodynamics with RiverDale

B.G. - 29/04/2024
'''

import taichi as ti
import numpy as np
from enum import Enum
import scabbard.utils as scaut 
from scabbard.riverdale.rd_grid import GRID
import scabbard.riverdale.rd_grid as gridfuncs


@ti.kernel
def _hillshading(Z:ti.template(), hillshade:ti.template(), BCs:ti.template(), light_dir: ti.f32, light_elev: ti.f32, Z_exaggeration:ti.f32, inverted:bool):
	'''
	🛠🛠🛠 core kernel for producing a nice hillshade

	Arguments:
		- Z: the 2D field of topographic surface
		- hillshade: the 2D field of hillshade to be computed in place
		- BCs: 2D field of boundary conditions
		- light_dir: 0 - 360,  direction of the light
		- light_elev: 0 - 90 inclinaison ("elevation") of the light
		- Z_exaggeration: higher values = higher contrast. Neutral is 1.
		- inverted: True -> "natural" hillshade, False -> "crayon style"

	Returns:
		- Nothing, edits hillshade in place

	Authors:
		- B.G. (last modifications: 06/2024) 
	'''

	# Yolo yolo beng beng
	for i, j in Z:

		# defaulting to no shade
		shaded = 0.

		# Ignoring nodes inactive (e.g. no data and all)
		if gridfuncs.is_active(i,j,BCs):

			# Get elevation differences between D4 neighbors
			dzdx = (Z[i + 1, j] - Z[i - 1, j]) * Z_exaggeration / 2.0
			dzdy = (Z[i, j + 1] - Z[i, j - 1]) * Z_exaggeration / 2.0
			
			# Calculate slope and aspect
			slope = ti.atan2(ti.sqrt(dzdx**2 + dzdy**2),1)
			aspect = ti.atan2(dzdy, dzdx)
			
			# Calculate illumination angle
			zenith = ti.math.radians(90 - light_elev)
			azimuth = ti.math.radians(light_dir)
			
			# Calculate the hillshade value
			shaded = ti.cos(zenith) * ti.cos(slope) + ti.sin(zenith) * ti.sin(slope) * ti.cos(azimuth - aspect)
			# shaded = ti.max(0, shaded)  # Ensure non-negative
			shaded = ti.abs(shaded)  # Ensure non-negative

		# Saving the value aaaaaand
		hillshade[i, j] = shaded if(inverted) else (1 - shaded)

	# Done


@ti.kernel
def _std_hillshading(Z:ti.template(), hillshade:ti.template(), light_dir: ti.f32, light_elev: ti.f32, Z_exaggeration:ti.f32, inverted:bool):
	'''
	🛠🛠🛠 core kernel for producing a nice hillshade

	Arguments:
		- Z: the 2D field of topographic surface
		- hillshade: the 2D field of hillshade to be computed in place
		- BCs: 2D field of boundary conditions
		- light_dir: 0 - 360,  direction of the light
		- light_elev: 0 - 90 inclinaison ("elevation") of the light
		- Z_exaggeration: higher values = higher contrast. Neutral is 1.
		- inverted: True -> "natural" hillshade, False -> "crayon style"

	Returns:
		- Nothing, edits hillshade in place

	Authors:
		- B.G. (last modifications: 06/2024) 
	'''

	# Yolo yolo beng beng
	for i, j in Z:

		# defaulting to no shade
		shaded = 0.

		# Ignoring nodes inactive (e.g. no data and all)
		if i > 0 and i < Z.shape[0] - 1 and j>0 and j < Z.shape[1] - 1:

			# Get elevation differences between D4 neighbors
			dzdx = (Z[i + 1, j] - Z[i - 1, j]) * Z_exaggeration / 2.0
			dzdy = (Z[i, j + 1] - Z[i, j - 1]) * Z_exaggeration / 2.0
			
			# Calculate slope and aspect
			slope = ti.atan2(ti.sqrt(dzdx**2 + dzdy**2),1)
			aspect = ti.atan2(dzdy, dzdx)
			
			# Calculate illumination angle
			zenith = ti.math.radians(90 - light_elev)
			azimuth = ti.math.radians(light_dir)
			
			# Calculate the hillshade value
			shaded = ti.cos(zenith) * ti.cos(slope) + ti.sin(zenith) * ti.sin(slope) * ti.cos(azimuth - aspect)
			shaded = ti.max(0, shaded)  # Ensure non-negative

		# Saving the value aaaaaand
		hillshade[i, j] = shaded if(inverted) else (1 - shaded)

def hillshading(rd, direction = 315., inclinaison = 45., exaggeration = 4.):
	'''
		🔭🔭🔭 Analysis function
		Hillshades the topography (field Z) of a riverdale's instance.

		Arguments:
			- rd: the riverdale's object
			- direction: 0 - 360, direction of the lighting
			- inclinaison: 0 - 90, inclinaison of the lighting (morning/evening light to noon)
			- exaggeration: higher values means higher contrast

		Returns:
			- A 2D numpy array of hillshade
		Authors:
			- B.G. (last modifications: 06/2024)
	'''

	# Prefetching the field
	hillshade, = rd.query_temporary_fields(1,dtype = ti.f32)

	# Running the GPU kernel
	_hillshading(rd.Z,hillshade,rd.BCs, direction, inclinaison, exaggeration, True)

	# Done, copying back to cpu and returning
	return hillshade.to_numpy()



def std_hillshading(Z2D, direction = 315., inclinaison = 45., exaggeration = 4.):
	'''
		🔭🔭🔭 Analysis function
		Hillshades the topography (field Z) of a riverdale's instance.

		Arguments:
			- rd: the riverdale's object
			- direction: 0 - 360, direction of the lighting
			- inclinaison: 0 - 90, inclinaison of the lighting (morning/evening light to noon)
			- exaggeration: higher values means higher contrast

		Returns:
			- A 2D numpy array of hillshade
		Authors:
			- B.G. (last modifications: 06/2024)
	'''
	try:
		ti.init(ti.gpu)
	except:
		ti.init(ti.cpu)

	# Prefetching the field
	hillshade = ti.field(dtype = ti.f32, shape = Z2D.shape)
	tZ = ti.field(dtype = ti.f32, shape = Z2D.shape)
	tZ.from_numpy(Z2D)

	# Running the GPU kernel
	_std_hillshading(tZ,hillshade, direction, inclinaison, exaggeration, True)

	# Done, copying back to cpu and returning
	ret =  hillshade.to_numpy()
	ti.init()
	return ret
