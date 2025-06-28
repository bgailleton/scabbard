# -*- coding: utf-8 -*-
"""
This module provides functions for generating various types of raster grids from scratch.

It includes functions for creating constant slope surfaces, white noise, and red noise
(self-affine surfaces), useful for initial conditions in simulations or testing.

Author: B.G.
"""

import scabbard as scb
import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import numba as nb
import time

def slope2D_S(
    nx=256,
    ny=512,
    dx=2.,
    z_base=0,
    slope=1e-3,
):
    """
    Generates a 2D sloping grid and initializes an environment with given boundaries.

    Args:
        nx (int, optional): Number of columns. Defaults to 256.
        ny (int, optional): Number of rows. Defaults to 512.
        dx (float, optional): Regular spatial spacing. Defaults to 2.0.
        z_base (float, optional): Elevation at the Southern boundary. Defaults to 0.
        slope (float, optional): The gradient of the surface. Defaults to 1e-3.

    Returns:
        tuple: A tuple containing:
            - scb.raster.RegularRasterGrid: The generated sloping grid.
            - numpy.ndarray: The boundary conditions array.

    Author: B.G.
    """
    # Initialize a zero array for elevation
    Z = np.zeros((ny, nx), dtype=np.float32)

    # Create a RegularRasterGrid object
    grid = scb.raster.raster_from_array(Z, dx=dx, xmin=0.0, ymin=0.0, dtype=np.float32)

    # Get X and Y coordinate matrices
    XX, YY = grid.geo.XY

    # Impose the slope based on Y coordinates (assuming Y increases upwards)
    # [::-1] is used to ensure the slope is applied correctly if Y is inverted
    Z = (YY * slope)[::-1]
    grid.Z[:, :] = Z[:, :]

    # Define boundary conditions
    BCs = np.ones((ny, nx), dtype=np.uint8)
    BCs[:, [0, -1]] = 0  # Left and right edges are NoData
    BCs[0, :] = 7        # Top edge (North) is inflow
    BCs[-1, :] = 3       # Bottom edge (South) is outflow

    return grid, BCs

def white_noise(nx, ny, dx, magnitude=1., BCs=None):
    """
    Generates a white noise elevation grid.

    Args:
        nx (int): Number of columns.
        ny (int): Number of rows.
        dx (float): Regular spatial spacing.
        magnitude (float, optional): The amplitude of the noise. Defaults to 1.0.
        BCs (numpy.ndarray or str, optional): Boundary conditions. Can be a numpy array,
                                            or a string ('4edges', 'periodicNS', 'periodicEW').
                                            If None, normal BCs are generated. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - scb.raster.RegularRasterGrid: The generated white noise grid.
            - numpy.ndarray: The boundary conditions array.

    Raises:
        ValueError: If the BCs string is not recognized or if BCs type is invalid.

    Author: B.G.
    """
    # Generate random numbers scaled by magnitude
    Z = np.random.rand(ny, nx) * magnitude
    grid = scb.raster.raster_from_array(Z, dx=dx, xmin=0.0, ymin=0.0, dtype=np.float32)

    # Handle boundary conditions
    if BCs is None:
        BCs = scb.ut.normal_BCs_from_shape(nx, ny)
    elif isinstance(BCs, str):
        if BCs == '4edges':
            BCs = scb.ut.normal_BCs_from_shape(nx, ny)
        elif BCs == 'periodicNS':
            BCs = scb.ut.periodic_NS_BCs_from_shape(nx, ny)
        elif BCs == 'periodicEW':
            BCs = scb.ut.periodic_EW_BCs_from_shape(nx, ny)
        else:
            raise ValueError(f"BCs string '{BCs}' not recognized. Use '4edges', 'periodicNS', or 'periodicEW'.")
    elif isinstance(BCs, scb.raster.RegularRasterGrid):
        # If BCs is a grid, assume it's a mask and set Z to 0 where BCs is 0
        grid.Z[BCs.Z == 0] = 0.
        BCs = BCs.Z # Use the data from the BCs grid
    else:
        raise ValueError("BCs type not recognized. Must be None, str, numpy.ndarray, or RegularRasterGrid.")

    return grid, BCs

def red_noise(ny, nx=None, dx=1., beta=2, variance=1, periodic=False, BCs=None):
    """
    Generates a self-affine (red noise) surface.

    This function creates a 2D array with a power-law spectral density,
    P = f^-beta, where P is spectral power and f is spatial frequency.

    Args:
        ny (int): Number of rows in the output matrix.
        nx (int, optional): Number of columns in the output matrix. Defaults to `ny`.
        dx (float, optional): Spatial step. Defaults to 1.0.
        beta (float, optional): Power law exponent. Defaults to 2.
        variance (float, optional): Variance of the surface. Defaults to 1.
        periodic (bool, optional): Whether the surface should be periodic. Defaults to False.
        BCs (numpy.ndarray or str, optional): Boundary conditions. Can be a numpy array,
                                            or a string ('4edges', 'periodicNS', 'periodicEW').
                                            If None, normal BCs are generated. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - scb.raster.RegularRasterGrid: A 2D array representing the self-affine surface.
            - numpy.ndarray: The boundary conditions array.

    Raises:
        ValueError: If `ny` or `nx` are odd for periodic output, or if BCs type is invalid.

    Author: B.G.
    """
    if nx is None:
        nx = ny

    if periodic:
        if ny % 2 != 0 or nx % 2 != 0:
            raise ValueError("For periodic output, ny and nx must be even.")

        # Create a grid of coordinates for Fourier components
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))

        # Generate random complex Fourier components
        F = (np.random.rand(ny, nx) - 0.5) + 1j * (np.random.rand(ny, nx) - 0.5)

        # Identify the DC component (zero frequency)
        xc, yc = nx // 2, ny // 2

        # Create frequency matrix, avoiding division by zero at DC component
        freq = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        freq[yc, xc] = 1

        # Apply the power-law reduction to the Fourier components
        F *= freq ** -beta

        # Set DC component to zero (mean of the surface)
        F[yc, xc] = 0

        # Inverse FFT to generate the surface
        M = np.real(np.fft.ifft2(np.fft.ifftshift(F)))

        # Scale to desired variance
        M = M * np.sqrt(variance) / np.std(M)

    else:
        # Non-periodic case: pad to next power of 2 for FFT efficiency
        n = int(2 ** np.ceil(np.log2(max(ny, nx))))
        x, y = np.meshgrid(np.arange(n), np.arange(n))

        F = (np.random.rand(n, n) - 0.5) + 1j * (np.random.rand(n, n) - 0.5)

        nc = n // 2
        freq = np.sqrt((x - nc) ** 2 + (y - nc) ** 2)
        freq[nc, nc] = 1

        F *= freq ** -beta
        F[nc, nc] = 0

        M = np.real(np.fft.ifft2(np.fft.ifftshift(F)))

        # Clip to the requested size
        M = M[:ny, :nx]

        # Scale to desired variance
        M = M * np.sqrt(variance) / np.std(M)

    grid = scb.raster.raster_from_array(M, dx=dx, xmin=0.0, ymin=0.0, dtype=np.float32)

    # Handle boundary conditions (similar to white_noise)
    if BCs is None:
        BCs = scb.ut.normal_BCs_from_shape(nx, ny)
    elif isinstance(BCs, str):
        if BCs == '4edges':
            BCs = scb.ut.normal_BCs_from_shape(nx, ny)
        elif BCs == 'periodicNS':
            BCs = scb.ut.periodic_NS_BCs_from_shape(nx, ny)
        elif BCs == 'periodicEW':
            BCs = scb.ut.periodic_EW_BCs_from_shape(nx, ny)
        else:
            raise ValueError(f"BCs string '{BCs}' not recognized. Use '4edges', 'periodicNS', or 'periodicEW'.")
    elif isinstance(BCs, scb.raster.RegularRasterGrid):
        grid.Z[BCs.Z == 0] = 0.
        BCs = BCs.Z
    else:
        raise ValueError("BCs type not recognized. Must be None, str, numpy.ndarray, or RegularRasterGrid.")

    return grid, BCs
