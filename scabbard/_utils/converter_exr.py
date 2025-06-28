# -*- coding: utf-8 -*-
"""
This module provides tools for converting elevation data to EXR image files.

It includes functions to calculate normal maps from elevation data and a command-line
interface (CLI) to perform the conversion.
"""

# __author__ = "B.G."

import scabbard as scb
import numpy as np
import rasterio
import click
from scipy.ndimage import sobel
import matplotlib.pyplot as plt

def calculate_normal_map(elevation_data):
    """
    Calculates the normal map from elevation data.

    A normal map is a 3D vector field where each vector is perpendicular to the
    surface at that point. It's commonly used in 3D graphics for lighting effects.

    Args:
        elevation_data (numpy.ndarray): A 2D array representing elevation.

    Returns:
        numpy.ndarray: A 3D array representing the normal map, with values
                       scaled to the [0, 1] range for use as a texture.
    """
    # Compute gradients in x and y directions using the Sobel operator
    gradient_x = sobel(elevation_data, axis=1)
    gradient_y = sobel(elevation_data, axis=0)

    # The normal vector's x and y components are the negative gradients
    neg_gradient_x = -gradient_x
    neg_gradient_y = -gradient_y

    # Create a placeholder for the normal map (x, y, z components)
    normals = np.zeros((*elevation_data.shape, 3), dtype=np.float32)

    # Calculate the length of the normal vector at each point
    length = np.sqrt(neg_gradient_x**2 + neg_gradient_y**2 + 1)

    # Normalize the normal vectors
    normals[..., 0] = neg_gradient_x / length  # X component
    normals[..., 1] = neg_gradient_y / length  # Y component
    normals[..., 2] = 1 / length             # Z component (always up)

    # Map the normal vectors from the [-1, 1] range to [0, 1] for image representation
    normal_map = (normals + 1) / 2
    normal_map = normal_map.astype(np.float32)

    return normal_map

def write_exr(array, nx: int, ny: int, filename: str, norm: bool = True):
    """
    Writes a numpy array to an EXR file.

    This function requires the OpenEXR library. It can handle both single-channel
    (grayscale) and 3-channel (RGB) arrays.

    Args:
        array (numpy.ndarray): The array to write. Can be 2D or 3D (with 3 channels).
        nx (int): The width of the image.
        ny (int): The height of the image.
        filename (str): The name of the output file.
        norm (bool, optional): If True and the input is a 2D array, normalize it
                               to the [0, 1] range. Defaults to True.

    Raises:
        ValueError: If the OpenEXR library is not installed or if the input array
                    has an unsupported number of dimensions.
    """
    try:
        import OpenEXR
        import Imath
    except ImportError:
        raise ValueError("To write data to EXR files, you need the openexr package: pip install openexr")

    # Ensure the filename has the .exr extension
    if '.exr' not in filename:
        filename += '.exr'

    # Create an EXR header and file object
    header = OpenEXR.Header(nx, ny)
    exr = OpenEXR.OutputFile(filename, header)

    # Write pixels based on array dimensions
    if array.ndim == 2:
        # Handle single-channel (grayscale) data
        if norm:
            array -= array.min()
            array /= array.max()
        # Write the same data to all three (R, G, B) channels
        exr.writePixels({'R': array.tobytes(), 'G': array.tobytes(), 'B': array.tobytes()})
    elif array.ndim == 3 and array.shape[2] == 3:
        # Handle three-channel (RGB) data
        exr.writePixels({'R': array[:, :, 0].tobytes(), 'G': array[:, :, 1].tobytes(), 'B': array[:, :, 2].tobytes()})
    else:
        raise ValueError('EXR writer can only save single-channel (2D) or RGB (3D) data.')

    exr.close()

@click.command()
@click.option("--normal", "-n", is_flag=True, show_default=True, default=False, help="Generate a normal map as well.")
@click.argument('fname', type=str)
def cli_convert_to_EXR(fname, normal):
    """
    Command-line tool to convert a DEM to an EXR texture.

    This tool can also generate a normal map from the DEM.

    Args:
        fname (str): Path to the input DEM file.
        normal (bool): If True, generate a normal map.
    """
    # Get the file prefix for output naming
    prefix = '.'.join(fname.split('.')[:-1])

    # Load the DEM raster data
    dem = scb.io.load_raster(fname)

    # Write the elevation data to an EXR file
    print(f"Writing elevation to {prefix + '.exr'}")
    write_exr(dem.Z, dem.geo.nx, dem.geo.ny, prefix + '.exr')

    # If requested, calculate and write the normal map
    if normal:
        print("Calculating normal map...")
        nmap = calculate_normal_map(dem.Z)
        print(f"Writing normal map to {prefix + '_normal.exr'}")
        write_exr(nmap, dem.geo.nx, dem.geo.ny, prefix + '_normal.exr')

    print("Done.")