# -*- coding: utf-8 -*-
"""
This module provides spectral filtering capabilities for raster data.

It includes functions for applying filters, such as a Gaussian filter, in the
frequency domain using the Fourier transform.
"""

# __author__ = "B.G."

import numpy as np
import scabbard as scb
from scipy.ndimage import distance_transform_edt

def gaussian_fourier(grid: scb.raster.RegularRasterGrid, in_place=False, BCs=None, magnitude=5):
    """
    Applies a Gaussian low-pass filter to a raster grid in the frequency domain.

    This function smooths the topography by performing a Fourier transform, applying
    a Gaussian filter to attenuate high frequencies, and then performing an
    inverse Fourier transform.

    Args:
        grid (scb.raster.RegularRasterGrid): The input raster grid to filter.
        in_place (bool, optional): If True, modifies the input grid directly.
                                 If False, returns a new filtered grid.
                                 Defaults to False.
        BCs (numpy.ndarray, optional): Boundary conditions. If provided, areas with a
                                     BC of 0 are masked and filled with the nearest
                                     valid values before filtering. Defaults to None.
        magnitude (int, optional): The standard deviation (sigma) of the Gaussian filter.
                                 A larger value results in more smoothing.
                                 Defaults to 5.

    Returns:
        scb.raster.RegularRasterGrid or None: If `in_place` is False, returns a new
                                              `RegularRasterGrid` with the smoothed data.
                                              Otherwise, returns None.
    """
    # Create a mask based on boundary conditions, or a full mask if BCs are not provided
    mask = np.ones_like(grid.Z, dtype=np.uint8) if BCs is None else np.where(BCs == 0, 0, 1).astype(np.uint8)

    # Copy the topography data to avoid modifying the original
    topography = grid.Z.copy()

    # If there are masked areas, fill them with the values of the nearest unmasked cells
    if BCs is not None:
        # Compute the distance to the nearest zero in the mask
        distance, indices = distance_transform_edt((mask == 0), return_indices=True)
        # Use the indices to fill the masked areas
        topography = topography[tuple(indices)]

    # Perform the 2D Fast Fourier Transform (FFT)
    fourier_transform = np.fft.fft2(topography)
    # Shift the zero-frequency component to the center of the spectrum
    fourier_transform_shifted = np.fft.fftshift(fourier_transform)

    # Get the dimensions of the topography
    rows, cols = topography.shape
    crow, ccol = rows // 2, cols // 2  # Center of the frequency domain

    # Create a Gaussian filter
    sigma = magnitude  # Standard deviation of the Gaussian
    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
    gaussian_filter = np.exp(-(distance_from_center**2) / (2 * sigma**2))

    # Apply the Gaussian filter to the shifted Fourier transform
    filtered_fourier = fourier_transform_shifted * gaussian_filter

    # Inverse shift the filtered spectrum
    inverse_shifted = np.fft.ifftshift(filtered_fourier)
    # Perform the inverse FFT to get the smoothed topography
    smoothed_topography = np.fft.ifft2(inverse_shifted).real.astype(grid.Z.dtype)

    # Restore the original values in the masked areas
    smoothed_topography[mask == 0] = grid.Z[mask == 0]

    # Apply the changes to the grid or return a new one
    if in_place:
        grid.Z[:, :] = smoothed_topography[:, :]
    else:
        return grid.duplicate_with_other_data(smoothed_topography)
