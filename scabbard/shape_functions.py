'''
This module provides a collection of functions for generating various mathematical shapes and kernels.
These functions can be used for creating synthetic topographic surfaces, distributing values,
or for other numerical operations.
'''

import numpy as np


def sinusoidal_wave(X, a, b, L):
    """
    Generates a sinusoidal wave.

    Args:
        X (numpy.ndarray): Input array of x-coordinates.
        a (float): Minimum value of the wave.
        b (float): Maximum value of the wave.
        L (float): Wavelength of the sinusoidal wave.

    Returns:
        numpy.ndarray: An array of y-values representing the sinusoidal wave.

    Author: B.G.
    """
    # Compute Y values for the sinusoidal wave
    Y = (b - a) / 2 * np.sin(2 * np.pi * X / L) + (a + b) / 2
    return Y

def generate_u_shaped_sloped_surface(nx, ny, dx, dy, slope=0.1, Umag = 0.8):
    """
    Generates a 2D NumPy array describing a U-shaped surface that is concave (facing upwards)
    and has a slope, making the northern part the highest point.
    
    Args:
        nx (int): Number of cells in the east-west (x) direction.
        ny (int): Number of cells in the north-south (y) direction.
        dx (float): Cell size in the east-west (x) direction.
        dy (float): Cell size in the north-south (y) direction.
        slope (float, optional): Slope of the surface in the north-south direction, making the north higher.
                                 Defaults to 0.1.
        Umag (float, optional): Magnitude of the U-shaped concavity. Defaults to 0.8.

    Returns:
        numpy.ndarray: A 2D NumPy array representing the surface elevation.

    Author: B.G.
    """
    
    # Generate meshgrid for the positions
    x, y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)
    
    # Center the x values around 0 for the U shape calculation
    x_centered = x - (nx * dx) / 2
    
    # Calculate the U shape using a negative quadratic function (for concave shape)
    u_shape = -x_centered**2
    
    # Normalize U shape to have values between 0 and 1
    u_shape_normalized = (u_shape - np.min(u_shape)) / (np.max(u_shape) - np.min(u_shape)) * Umag
    
    # Modify the slope effect to make the northern part the highest
    slope_effect = (ny * dy - y) * slope
    
    # Combine the U shape and the slope to generate the surface
    surface = u_shape_normalized*-1 + slope_effect
    
    # Normalize the surface to have a minimum of 0 (optional)
    surface -= np.min(surface)
    
    return surface

def gaussian_spread_on_1D(X = np.linspace(0, 100-1, 100), M = 50, x_c = 50, sigma = 10):
	"""
	Generates a 1D Gaussian distribution (kernel) and scales its sum to a specified magnitude.

	This function can be used to create a 1D kernel for spreading values, for example,
	in kernel density estimation or for creating smooth distributions.

	Args:
		X (numpy.ndarray, optional): The 1D array of x-coordinates over which to generate the Gaussian.
							Defaults to a linear space from 0 to 99 for 100 points.
		M (float, optional): The desired sum of the Gaussian values. Defaults to 50.
		x_c (float, optional): The center (mean) of the Gaussian distribution. Defaults to 50.
		sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 10.

	Returns:
		numpy.ndarray: A 1D NumPy array representing the scaled Gaussian distribution.

	Author: B.G.
	"""

	# Calculate Gaussian values using the probability density function formula
	gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - x_c) / sigma) ** 2)

	# Scale the Gaussian so its sum equals M
	gaussian *= M / np.sum(gaussian)

	# Verify the sum of the Gaussian is approximately M (for debugging/information)
	print("Sum of Gaussian values:", np.sum(gaussian))

	return gaussian
