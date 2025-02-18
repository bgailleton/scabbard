import scabbard as scb
import numpy as np
import click
'''
Manage the transfer of data from scabbard-python to IPC (Inter-Processes Communations)
Warning: stores data on the RAM directly, very prone to memory leak if badly manage
So far Only work on Linux (and Mac?). Will port it on Windows as soon as API gets stable.

Authors: B.G.
'''
# Json for writing dictionary-like data
import json
# Memory maps for writing in the memory
import mmap
# Struct helps formatting 
import struct


@click.command()
@click.argument('fname', type = str)
@click.argument('prefix', type = str)
def simple_load(fname, prefix):
	'''
	Load a single grid to the Memory map from a raster file
	Metadata as a json -> {prefix}_meta
	grid as binary data -> {prefix}_array

	Arguments:
		- fname (str) : the file to load
		- prefix (str): the prefix for file

	Returns:
		- Nothing, save a json with meta data and the grid array
	'''
	grid = scb.io.load_raster(fname)

	# Example data to save
	metadata = {
		"nx": grid.geo.nx,  # or "read_write"
		"ny": grid.geo.ny,  # or "read_write"
		"dx": grid.geo.dx,
		"xmin": grid.geo.xmin,
		"ymin": grid.geo.ymin,
		"dtype": str(grid.Z.dtype),
		"fname": prefix + "_array",
		"zmin": float(np.nanmin(grid.Z)),
		"zmax": float(np.nanmax(grid.Z)),
	}

	print(metadata)
	# Save the JSON file to /dev/shm/
	with open("/dev/shm/"+prefix+"_meta", 'w+') as f:
		json.dump(metadata, f)

	# Open the file and resize it to the desired size
	with open("/dev/shm/"+prefix+"_array", "wb+") as f:
	    f.truncate(grid.Z.size * grid.Z.itemsize)

	# Actually saving the data
	with open("/dev/shm/"+prefix+"_array", "r+b") as f:
		shared_mem  = mmap.mmap(f.fileno(), grid.Z.size * grid.Z.itemsize )

		# Pack data and write to shared memory
		packed_data = grid.Z.tobytes()                # NumPy array

		# Write to shared memory
		shared_mem.seek(0)
		shared_mem.write(packed_data)

	print(f'EXPERIMENTAL FEATURE::{fname} has been loaded on the IPC RAM with prefix {prefix}')
	print("Beware of memory leaks!!")


