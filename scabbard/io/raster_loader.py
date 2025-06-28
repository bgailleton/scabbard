# -*- coding: utf-8 -*-
"""
This module provides functions for loading and saving raster data.

It supports reading various raster formats using `rasterio` and saving to GeoTIFF
and ASCII Grid formats.
"""

# __author__ = "B.G."

import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.transform import from_origin
import scabbard as scb

def load_raster(fname, dtype=np.float32):
    """
    Loads a raster file into a RegularRasterGrid object.

    This function is designed to read single-band geolocated raster data (e.g., DEMs).
    It uses `rasterio` and GDAL, supporting a wide range of formats.

    Args:
        fname (str): Full path to the raster file.
        dtype (numpy.dtype, optional): Forces a specific data type for the raster data.
                                     Defaults to `np.float32`.

    Returns:
        scb.raster.RegularRasterGrid: The raster object with data loaded.
    """
    # Open the raster file with rasterio
    with rio.open(fname) as this_raster:
        # Get the resolution (cell size)
        gt = this_raster.res

        # Create the underlying geometry object for the grid
        geom = scb.geometry.RegularGeometry(
            this_raster.width, this_raster.height, gt[0],
            this_raster.bounds.left, this_raster.bounds.bottom
        )

        # Read the first band of the raster data and cast to the specified dtype
        Z = this_raster.read(1).astype(dtype)

        # Attempt to get the Coordinate Reference System (CRS)
        try:
            geom._crs = this_raster.crs.to_string()
        except (TypeError, AttributeError):
            # Fallback if CRS is not found or invalid
            geom._crs = 'EPSG:32601'  # Default to a common UTM CRS

    # Create and return a RegularRasterGrid object
    return scb.raster.RegularRasterGrid(Z, geom, dtype=dtype)

def save_raster(grid, fname, crs='EPSG:32601', dtype=np.float32, driver='GTiff'):
    """
    Saves a RegularRasterGrid object to a raster file using rasterio.

    Args:
        grid (scb.raster.RegularRasterGrid): The raster grid object to save.
        fname (str): Full path for the output file.
        crs (str, optional): Coordinate Reference System string. Defaults to 'EPSG:32601'.
        dtype (numpy.dtype, optional): Data type for the output raster. Defaults to `np.float32`.
        driver (str, optional): GDAL driver name (e.g., 'GTiff', 'AAIGrid'). Defaults to 'GTiff'.
    """
    height, width = grid.Z.shape

    # Ensure the data type matches the desired output dtype
    if grid.Z.dtype != dtype:
        data_to_save = grid.Z.astype(dtype)
    else:
        data_to_save = grid.Z

    # Define the affine transform for the raster
    # from_origin(west, north, xsize, ysize) - north is ymax
    transform = rio.transform.from_origin(grid.geo.xmin, grid.geo.ymax, grid.geo.dx, grid.geo.dx)

    # Define the metadata for the output raster
    meta = {
        'driver': driver,
        'height': height,
        'width': width,
        'count': 1,  # Number of bands
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
        'nodata': -9999.0 # Default nodata value
    }

    print("Saving raster with metadata:", meta)

    # Write the raster data to a file
    with rio.open(fname, 'w', **meta) as dst:
        dst.write(data_to_save, 1) # Write the first band

def save_ascii_grid(grid, fname, dtype=np.float32):
    """
    Saves a RegularRasterGrid object to an ASCII Grid file.

    Args:
        grid (scb.raster.RegularRasterGrid): The raster grid object to save.
        fname (str): Full path for the output file.
        dtype (numpy.dtype, optional): Data type for the output raster. Defaults to `np.float32`.
    """
    nrows, ncols = grid.Z.shape

    # Ensure the data type matches the desired output dtype
    if grid.Z.dtype != dtype:
        data_to_save = grid.Z.astype(dtype)
    else:
        data_to_save = grid.Z

    # Replace NaN values with the NODATA_value for ASCII grid format
    nodata_value = -9999.0
    data_to_save[np.isnan(data_to_save)] = nodata_value

    # Open the file for writing
    with open(fname, 'w') as f:
        # Write the ASCII Grid header
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {grid.geo.xmin}\n")
        f.write(f"yllcorner     {grid.geo.ymin}\n")
        f.write(f"cellsize      {grid.geo.dx}\n")
        f.write(f"NODATA_value  {nodata_value}\n")

        # Write the raster data row by row
        for row in data_to_save:
            f.write(' '.join(map(str, row)) + '\n')