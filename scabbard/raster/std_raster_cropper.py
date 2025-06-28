# -*- coding: utf-8 -*-
"""
This module provides a command-line tool for interactively cropping raster datasets.

It allows users to select a region of interest on a displayed raster using a graphical
interface and then saves the cropped portion to a new GeoTIFF file.

Author: B.G.
"""

# __author__ = "B.G."

import rasterio
from rasterio import windows
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import click
import os

@click.command()
@click.argument("dem_path", type=str, help="The path to the input DEM raster file.")
@click.argument("output", type=str, help="The path for the output cropped raster file.")
def std_crop_raster(dem_path, output):
    """
    Interactively crops a raster dataset.

    This tool displays the input raster and allows the user to draw a rectangle
    to define the cropping area. The selected area is then saved as a new GeoTIFF.

    Args:
        dem_path (str): The path to the input DEM raster file.
        output (str): The path for the output cropped raster file.

    Raises:
        ValueError: If the input DEM file does not exist.

    Author: B.G.
    """
    if not os.path.exists(dem_path):
        raise ValueError(f"Input file does not exist: {dem_path}")

    # Open the DEM raster dataset
    dataset = rasterio.open(dem_path)

    # Read the first band of the raster data
    data = dataset.read(1)

    # Get the extent of the raster for plotting [xmin, xmax, ymin, ymax]
    extent = [
        dataset.bounds.left,
        dataset.bounds.right,
        dataset.bounds.bottom,
        dataset.bounds.top,
    ]

    # Plot the raster for interactive selection
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(data, cmap="terrain", extent=extent)
    ax.set_title("Draw a rectangle to crop the DEM")
    fig.colorbar(cax, ax=ax, label="Elevation (m)")

    # Define the function to be called when a rectangle is selected
    def onselect(eclick, erelease):
        """
        Callback function executed when a rectangle is selected on the plot.

        This function extracts the coordinates of the selected rectangle, crops the
        raster data, and saves the cropped portion to a new GeoTIFF file.

        Args:
            eclick (matplotlib.backend_bases.MouseEvent): The event object for the mouse press.
            erelease (matplotlib.backend_bases.MouseEvent): The event object for the mouse release.
        """
        # Get the coordinates of the selected rectangle
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Ensure coordinates are sorted for window creation
        minx, maxx = sorted([x1, x2])
        miny, maxy = sorted([y1, y2])

        print(f"Selected rectangle from ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})")

        # Create a rasterio window object from the selected bounds
        window = windows.from_bounds(
            minx, miny, maxx, maxy, transform=dataset.transform
        )

        # Read the data within the defined window
        data_crop = dataset.read(1, window=window)

        # Get the new transform object for the cropped data
        transform_crop = windows.transform(window, dataset.transform)

        # Update metadata for the cropped raster output
        out_meta = dataset.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": data_crop.shape[0],
                "width": data_crop.shape[1],
                "transform": transform_crop,
            }
        )

        # Save the cropped raster to the specified output path
        with rasterio.open(output, "w", **out_meta) as dest:
            dest.write(data_crop, 1)

        print(f"Cropped raster saved as {output}")
        plt.close(fig) # Close the plot after saving

    # Create the RectangleSelector widget for interactive cropping
    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,  # Use blitting for faster redraws
        button=[1],    # Only respond to left mouse button clicks
        minspanx=5,    # Minimum horizontal span in pixels
        minspany=5,    # Minimum vertical span in pixels
        spancoords="data", # Coordinates are in data units
        interactive=True, # Allow interactive resizing
    )

    plt.show()