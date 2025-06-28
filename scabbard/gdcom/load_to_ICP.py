# -*- coding: utf-8 -*-
"""
This module manages the transfer of data from scabbard-python to Inter-Process
Communication (IPC) using shared memory.

Warning: This feature stores data directly in RAM and is prone to memory leaks if not
managed carefully. It is currently designed for Linux-based systems.
"""

# __author__ = "B.G."

import scabbard as scb
import numpy as np
import click
import json
import mmap
import struct

@click.command()
@click.argument('fname', type=str)
@click.argument('prefix', type=str)
def simple_load(fname, prefix):
    """
    Loads a single grid from a raster file into a shared memory map.

    This function creates two files in the shared memory directory (`/dev/shm`):
    - A JSON file with metadata (`{prefix}_meta`).
    - A binary file with the raw grid data (`{prefix}_array`).

    Args:
        fname (str): The path to the raster file to load.
        prefix (str): The prefix for the shared memory file names.
    """
    # Load the raster data using scabbard's I/O functions
    grid = scb.io.load_raster(fname)

    # --- Prepare Metadata ---
    metadata = {
        "nx": grid.geo.nx,
        "ny": grid.geo.ny,
        "dx": grid.geo.dx,
        "xmin": grid.geo.xmin,
        "ymin": grid.geo.ymin,
        "dtype": str(grid.Z.dtype),
        "fname": prefix + "_array",
        "zmin": float(np.nanmin(grid.Z)),
        "zmax": float(np.nanmax(grid.Z)),
    }

    print("Metadata:", metadata)

    # --- Write Metadata to Shared Memory ---
    try:
        with open(f"/dev/shm/{prefix}_meta", 'w+') as f:
            json.dump(metadata, f)
    except FileNotFoundError:
        print("Error: /dev/shm not found. This feature is only supported on Linux.")
        return

    # --- Write Array Data to Shared Memory ---
    try:
        # Create and truncate the file to the required size
        with open(f"/dev/shm/{prefix}_array", "wb+") as f:
            f.truncate(grid.Z.nbytes)

        # Memory-map the file and write the data
        with open(f"/dev/shm/{prefix}_array", "r+b") as f:
            with mmap.mmap(f.fileno(), grid.Z.nbytes) as shared_mem:
                shared_mem.write(grid.Z.tobytes())

    except FileNotFoundError:
        # This error is caught by the metadata check, but included for robustness
        return

    print(f"EXPERIMENTAL: {fname} has been loaded to shared memory with prefix '{prefix}'.")
    print("Warning: Be mindful of potential memory leaks!")