import numpy as np
import scabbard as scb
from scipy.ndimage import distance_transform_edt


def gaussian_fourier(
	grid: scb.raster.RegularRasterGrid, 
	in_place = False, 
	BCs = None, magnitude = 5):
	
	# masking
	mask = np.ones_like(grid.Z).astype(np.uint8) if (BCs is None) else (np.where(BCs == 0, 0, 1).astype(np.uint8))
	
	# Value to filter
	topography = grid.Z.copy()

	if(BCs is None) == False:
		# Compute the distance transform
		distance, indices = distance_transform_edt((mask == 0), return_indices=True)

		# Extract the nearest valid values using the indices
		topography = topography[tuple(indices)]
		

	# Perform the 2D Fourier Transform
	fourier_transform = np.fft.fft2(topography)
	fourier_transform_shifted = np.fft.fftshift(fourier_transform)

	# Define a low-pass filter: create a mask with ones in the low frequencies and zeros in the high frequencies
	rows, cols = topography.shape
	crow, ccol = rows // 2, cols // 2  # Center of the frequency domain

	# Create a Gaussian filter
	sigma = magnitude  # Adjust this value to control the level of smoothing (higher = more smoothing)
	y, x = np.ogrid[:rows, :cols]
	distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
	gaussian_filter = np.exp(-(distance**2) / (2 * sigma**2))


	# Apply the Gaussian filter to the Fourier-transformed data
	filtered_fourier = fourier_transform_shifted * gaussian_filter

	# Perform the inverse Fourier Transform to reconstruct the smoothed topography
	inverse_shifted = np.fft.ifftshift(filtered_fourier)
	smoothed_topography = np.fft.ifft2(inverse_shifted).real.astype(grid.Z.dtype)  # Take the real part since the result might be complex

	smoothed_topography[mask == 0] = grid.Z[mask == 0]

	if in_place:
		grid.Z[:,:] = smoothed_topography[:,:]
	else:
		return grid.duplicate_with_other_data(smoothed_topography)