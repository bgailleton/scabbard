import rasterio
from rasterio import windows
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import click
import os


@click.command()
@click.argument("dem_path", type=str)
@click.argument("output", type=str)
def std_crop_raster(dem_path, output):

    if os.path.exists(dem_path) == False:
        raise ValueError("File does not Exists")

    # Open the DEM raster
    dataset = rasterio.open(dem_path)

    # Read the raster data (assuming it's a single band raster)
    data = dataset.read(1)

    # Get the extent of the raster [xmin, xmax, ymin, ymax]
    extent = [
        dataset.bounds.left,
        dataset.bounds.right,
        dataset.bounds.bottom,
        dataset.bounds.top,
    ]

    # Plot the raster
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(data, cmap="terrain", extent=extent)
    ax.set_title("DEM")
    fig.colorbar(cax, ax=ax, label="Elevation (m)")

    # Function to handle the rectangle selection and crop the raster
    def onselect(eclick, erelease):
        # Get the coordinates of the rectangle
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Create a bounding box with the selected coordinates
        minx, maxx = sorted([x1, x2])
        miny, maxy = sorted([y1, y2])

        print(f"Rectangle selected from ({minx}, {miny}) to ({maxx}, {maxy})")

        # Define the window to read
        window = windows.from_bounds(
            minx, miny, maxx, maxy, transform=dataset.transform
        )

        # Read the data within the window
        data_crop = dataset.read(1, window=window)

        # Get the transform object for the cropped data
        transform_crop = windows.transform(window, dataset.transform)

        # Update metadata for the cropped raster
        out_meta = dataset.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": data_crop.shape[0],
                "width": data_crop.shape[1],
                "transform": transform_crop,
            }
        )

        # Save the cropped raster
        cropped_path = output  # You can change the output path here
        with rasterio.open(cropped_path, "w", **out_meta) as dest:
            dest.write(data_crop, 1)

        print(f"Cropped raster saved as {cropped_path}")

    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5,
        minspany=5,
        spancoords="data",
        interactive=True,
    )

    plt.show()
