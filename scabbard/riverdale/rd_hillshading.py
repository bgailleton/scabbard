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
import scabbard.steenbok as ste
import math
import numba as nb


@ti.kernel
def _hillshading(Z:ti.template(), hillshade:ti.template(), BCs:ti.template(), light_dir: ti.f32, light_elev: ti.f32, Z_exaggeration:ti.f32, inverted:bool, dx:ti.f32):
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
			dzdx = (Z[i + 1, j] - Z[i - 1, j]) * Z_exaggeration / (2.0 * dx)
			dzdy = (Z[i, j + 1] - Z[i, j - 1]) * Z_exaggeration / (2.0 * dx)
			
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
def _std_hillshading(Z:ti.template(), hillshade:ti.template(), light_dir: ti.f32, light_elev: ti.f32, Z_exaggeration:ti.f32, inverted:bool, dx:ti.f32):
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
			dzdx = (Z[i + 1, j] - Z[i - 1, j]) * Z_exaggeration / (2.0 * dx)
			dzdy = (Z[i, j + 1] - Z[i, j - 1]) * Z_exaggeration / (2.0 * dx)
			
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

@nb.njit()
def _std_hillshading_cpu_D4(Z, light_dir, light_elev, Z_exaggeration, inverted:bool, dx:float):
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

	hillshade = np.zeros_like(Z)
	BCs = np.ones(Z.shape).astype(np.uint8)
	ny,nx = BCs.shape
	

	# Yolo yolo beng beng
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):

			# defaulting to no shade
			shaded = 0.

			# Ignoring nodes inactive (e.g. no data and all)
			if i > 0 and i < Z.shape[0] - 1 and j>0 and j < Z.shape[1] - 1:

				bottom = ste.neighbours_D4(i, j, 3, BCs, nx, ny)
				top = ste.neighbours_D4(i, j, 0, BCs, nx, ny)
				left = ste.neighbours_D4(i, j, 1, BCs, nx, ny)
				right = ste.neighbours_D4(i, j, 2, BCs, nx, ny)

				# Get elevation differences between D4 neighbors
				dzdx = (Z[bottom[0],bottom[1]] - Z[top[0],top[1]]) * Z_exaggeration / (2*dx)
				dzdy = (Z[right[0],right[1]] - Z[left[0],left[1]]) * Z_exaggeration / (2*dx)
				
				# Calculate slope and aspect
				slope = math.atan2(math.sqrt(dzdx**2 + dzdy**2),1)
				aspect = math.atan2(dzdy, dzdx)
				
				# Calculate illuminamathon angle
				zenith = math.radians(90 - light_elev)
				azimuth = math.radians(light_dir)
				
				# Calculate the hillshade value
				shaded = math.cos(zenith) * math.cos(slope) + math.sin(zenith) * math.sin(slope) * math.cos(azimuth - aspect)
				shaded = max(0, shaded)  # Ensure non-negative

			# Saving the value aaaaaand
			hillshade[i, j] = shaded if(inverted) else (1 - shaded)

	return hillshade


@nb.njit()
def _std_hillshading_cpu_D8(Z, light_dir, light_elev, Z_exaggeration, inverted:bool, dx:float):
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

	hillshade = np.zeros_like(Z)
	BCs = np.ones(Z.shape).astype(np.uint8)
	BCs[0,  :] = 3
	BCs[-1, :] = 3
	BCs[:,  0] = 3
	BCs[:, -1] = 3
	ny,nx = BCs.shape
	
	hillshade = hillshade.ravel()
	BCs = BCs.ravel()

	dxy = dx*np.sqrt(2.)

	# Yolo yolo beng beng
	for i in range(Z.ravel().shape[0]):

		# defaulting to no shade
		shaded = 0.

		# Ignoring nodes inactive (e.g. no data and all)
	

		bottomleft = ste.neighbours_D8_flat(i, 5, BCs, nx, ny)
		bottom = ste.neighbours_D8_flat(i, 6, BCs, nx, ny)
		bottomright = ste.neighbours_D8_flat(i, 7, BCs, nx, ny)
		topleft = ste.neighbours_D8_flat(i, 0, BCs, nx, ny)
		top = ste.neighbours_D8_flat(i, 1, BCs, nx, ny)
		topright = ste.neighbours_D8_flat(i, 2, BCs, nx, ny)
		left = ste.neighbours_D8_flat(i, 3, BCs, nx, ny)
		right = ste.neighbours_D8_flat(i, 4, BCs, nx, ny)

		if bottomleft == -1 or bottom == -1 or bottomright == -1 or topleft == -1 or top == -1 or topright == -1 or left == -1 or right == -1:
			continue

		# Get elevation differences between D4 neighbors
		dzdx = (Z.ravel()[bottom] - Z.ravel()[top]) * Z_exaggeration / (2*dx)
		dzdy = (Z.ravel()[right] - Z.ravel()[left]) * Z_exaggeration / (2*dx)
		dzdxy1 = (Z.ravel()[bottomright] - Z.ravel()[topleft]) * Z_exaggeration / (2*dxy)
		dzdxy2 = (Z.ravel()[bottomleft] - Z.ravel()[topright]) * Z_exaggeration / (2*dxy)
		# print(dzdx)
		
		# Calculate slope and aspect
		slope = np.atan2(np.sqrt(dzdx**2 + dzdy**2),1)
		aspect = np.atan2(dzdy, dzdx)

		# Calculate slope and aspect (diag)
		slope_diag = np.atan2(np.sqrt(dzdxy1**2 + dzdxy2**2),1)
		aspect_diag = np.atan2(dzdy, dzdx)
		
		# Calculate illuminanpon angle
		zenith = np.radians(90 - light_elev)
		azimuth = np.radians(light_dir)

		
		# Calculate the hillshade value
		shaded = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(azimuth - aspect)
		shaded_diag = np.cos(zenith) * np.cos(slope_diag) + np.sin(zenith) * np.sin(slope_diag) * np.cos(azimuth - aspect_diag)
		shaded = max(0, (shaded + shaded_diag)/2.)  # Ensure non-negative

		# Saving the value aaaaaand
		hillshade[i] = shaded if(inverted) else (1 - shaded)

	return hillshade.reshape(Z.shape)

def _compute_hillshade(Z, light_dir, light_elev, Z_exaggeration, inverted, dx):
	"""
	Optimized hillshading function using numpy.
	"""
	# Precompute light direction components
	zenith_rad = np.radians(90 - light_elev)
	azimuth_rad = np.radians(light_dir)

	cos_zenith = np.cos(zenith_rad)
	sin_zenith = np.sin(zenith_rad)

	# Compute gradients
	dzdx, dzdy = np.gradient(Z * Z_exaggeration, dx)

	# Compute slope and aspect
	slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
	aspect = np.arctan2(dzdy, dzdx)

	# Compute hillshade
	shaded = cos_zenith * np.cos(slope) + sin_zenith * np.sin(slope) * np.cos(azimuth_rad - aspect)
	shaded = np.clip(shaded, 0, 1)  # Ensure non-negative

	return 1 - shaded if inverted else shaded

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



def std_hillshading(Z2D, direction = 315., inclinaison = 45., exaggeration = 4., use_gpu = False, D4 = False, dx = 1.):
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
	
	if(use_gpu):
		try:
			ti.init(ti.gpu)
		except:
			ti.init(ti.cpu)

		# Prefetching the field
		hillshade = ti.field(dtype = ti.f32, shape = Z2D.shape)
		tZ = ti.field(dtype = ti.f32, shape = Z2D.shape)
		tZ.from_numpy(Z2D)

		# Running the GPU kernel
		_std_hillshading(tZ,hillshade, direction, inclinaison, exaggeration, True, dx)

		# Done, copying back to cpu and returning
		ret =  hillshade.to_numpy()
		ti.init()
		return ret

	else:

		return _compute_hillshade(Z2D, direction, inclinaison, exaggeration, False, dx) if D4 else _compute_hillshade(Z2D, direction, inclinaison, exaggeration, False, dx)

