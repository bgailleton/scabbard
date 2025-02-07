'''
Contains routines to create raster grids from scratch.

For example white noise, perlin noise or spectral stuff for initial conditions,
or constant slope raster to test river LEMs, ...

B.G. - created 09/2024
'''

import scabbard as scb
import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import numba as nb
import time


def slope2D_S(
	nx = 256, 
	ny = 512, 
	dx = 2.,
	z_base = 0,
	slope = 1e-3, 
	):
	'''
		Generates a slopping grid adn initialise an env with given boundaries to it. 
		It comes with a connector and a graph (todo) alredy pre-prepared for boundaries

		Arguments:
			- nx, ny: number of columns and rows
			- dx: regular spacing
			- z_base: elevation at the Southern boundary
			- slope: the gradient of the surface

		Return:
			- A tuple of the RegularRasterGrid and its boundary conditions array

		Author:
			- B.G. (last modifications: 01/2025)
	'''

	# Zeros Topo
	Z = np.zeros((ny, nx), dtype = np.float32)

	# Length
	lx = (nx+1) * dx
	ly = (ny+1) * dx

	# creating the flat grid
	grid = scb.raster.raster_from_array(Z, dx=dx, xmin=0.0, ymin=0.0, dtype=np.float32)

	# getting XY matrices
	XX,YY = grid.geo.XY
	
	# Imposing the slope	
	Z = (YY * slope)[::-1]
	grid.Z[:,:] = Z[:,:]

	# Boundary conditions
	BCs = np.ones((ny,nx),dtype = np.uint8)
	BCs[:,[0,-1]] = 0
	BCs[0,:] = 7
	BCs[-1,:] = 3

	return grid, BCs

def white_noise(nx, ny, dx, magnitude = 1., BCs = None):
	'''
		Generates a white noise array

		Arguments:
			- nx, ny: number of columns and rows
			- dx: regular spacing
			- magnitude (Default: 1): magnitude of the noise (0 to magnitude)
			- BCs (optional): pre-created boundary conditions

		Return:
			- A tuple of the RegularRasterGrid and its boundary conditions array

		Author:
			- B.G. (last modifications: 01/2025)
	'''

	Z = np.random.rand(ny,nx) * magnitude
	grid = scb.raster.raster_from_array(Z, dx=dx, xmin=0.0, ymin=0.0, dtype=np.float32)

	if(BCs is None):
		BCs = scb.ut.normal_BCs_from_shape(nx,ny)
	elif isinstance(BCs, str):
		if(BCs == '4edges'):
			BCs = scb.ut.normal_BCs_from_shape(nx,ny)
		elif(BCs == 'periodicNS'):
			BCs = scb.ut.periodic_NS_BCs_from_shape(nx,ny)
		elif(BCs == 'periodicEW'):
			BCs = scb.ut.periodic_EW_BCs_from_shape(nx,ny)
		else:
			raise ValueError('scabbard.raster.raster_factory.white_noise::BCs is string but not 4edges, periodicNS or periodicEW')
	elif isinstance(BCs,scb.raster.RegularRasterGrid):
		grid.Z[BCs == 0] = 0.
	else:
		raise ValueError('scabbard.raster.raster_factory.white_noise::BCs is not recognised')

	return grid, BCs



def red_noise(ny, nx, dx = 1., beta=2, variance=1, periodic=False, BCs = None):
	"""
	Generates a self-affine surface characterized by:
	P = f^-beta
	where P is spectral power and f is spatial frequency.

	Parameters:
		ny (int): Number of rows in the output matrix.
		nx (int, optional): Number of columns in the output matrix. Defaults to ny.
		beta (float, optional): Power law exponent. Default is 2.
		variance (float, optional): Variance of the surface. Default is 1.
		periodic (bool, optional): Whether the surface should be periodic. Default is False.

	Returns:
		numpy.ndarray: A 2D array representing the self-affine surface.
	"""

	if nx is None:
		nx = ny

	if periodic:
		if ny % 2 != 0 or nx % 2 != 0:
			raise ValueError("scabbard.raster.raster_factory.::ny and nx must be even for periodic output.")

		# Create a grid of coordinates
		x, y = np.meshgrid(np.arange(nx), np.arange(ny))

		# Generate random Fourier components
		F = (np.random.rand(ny, nx) - 0.5) + 1j * (np.random.rand(ny, nx) - 0.5)

		# Identify the DC component
		xc, yc = nx // 2, ny // 2

		# Create frequency matrix
		freq = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
		freq[yc, xc] = 1  # Avoid division by zero at the DC component

		# Apply the power-law reduction
		F *= freq ** -beta

		# Set DC component to zero
		F[yc, xc] = 0

		# Inverse FFT to generate the surface
		M = np.real(np.fft.ifft2(np.fft.ifftshift(F)))

		# Scale to desired variance
		M = M * np.sqrt(variance) / np.std(M)

	else:
		# Non-periodic case
		n = int(2 ** np.ceil(np.log2(max(ny, nx))))  # Next power of 2
		x, y = np.meshgrid(np.arange(n), np.arange(n))

		# Generate random Fourier components
		F = (np.random.rand(n, n) - 0.5) + 1j * (np.random.rand(n, n) - 0.5)

		# Identify the DC component
		nc = n // 2

		# Create frequency matrix
		freq = np.sqrt((x - nc) ** 2 + (y - nc) ** 2)
		freq[nc, nc] = 1  # Avoid division by zero

		# Apply the power-law reduction
		F *= freq ** -beta

		# Set DC component to zero
		F[nc, nc] = 0

		# Inverse FFT to generate the surface
		M = np.real(np.fft.ifft2(np.fft.ifftshift(F)))

		# Clip to requested size
		M = M[:ny, :nx]

		# Scale to desired variance
		M = M * np.sqrt(variance) / np.std(M)


	grid = scb.raster.raster_from_array(M, dx=dx, xmin=0.0, ymin=0.0, dtype=np.float32)

	if(BCs is None):
		BCs = scb.ut.normal_BCs_from_shape(nx,ny)
	elif isinstance(BCs, str):
		if(BCs == '4edges'):
			BCs = scb.ut.normal_BCs_from_shape(nx,ny)
		elif(BCs == 'periodicNS'):
			BCs = scb.ut.periodic_NS_BCs_from_shape(nx,ny)
		elif(BCs == 'periodicEW'):
			BCs = scb.ut.periodic_EW_BCs_from_shape(nx,ny)
		else:
			raise ValueError('scabbard.raster.raster_factory.white_noise::BCs is string but not 4edges, periodicNS or periodicEW')
	elif isinstance(BCs,scb.raster.RegularRasterGrid):
		grid.Z[BCs == 0] = 0.
	else:
		raise ValueError('scabbard.raster.raster_factory.white_noise::BCs is not recognised')

	return grid, BCs