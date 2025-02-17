import scabbard as scb
import numpy as np
import click
import json
import mmap
import struct


@click.command()
@click.argument('fname', type = str)
@click.argument('prefix', type = str)
def simple_load(fname, prefix):

	grid = scb.io.load_raster(fname)

	# Example data to save
	metadata = {
		"nx": grid.geo.nx,  # or "read_write"
		"ny": grid.geo.ny,  # or "read_write"
		"dx": grid.geo.dx,
		"xmin": grid.geo.xmin,
		"ymin": grid.geo.ymin,
		"dtype": str(grid.Z.dtype),
		"fname": prefix + "_array"
	}

	# Save the JSON file to /dev/shm/
	with open("/dev/shm/"+prefix+"_meta", 'w+') as f:
		json.dump(metadata, f)

	# Open the file and resize it to the desired size
	with open("/dev/shm/"+prefix+"_array", "wb+") as f:
	    f.truncate(grid.Z.size * grid.Z.itemsize)

	

	with open("/dev/shm/"+prefix+"_array", "r+b") as f:
		shared_mem  = mmap.mmap(f.fileno(), grid.Z.size * grid.Z.itemsize )

		# Pack data and write to shared memory
		packed_data = grid.Z.tobytes()                # NumPy array

		# Write to shared memory
		shared_mem.seek(0)
		shared_mem.write(packed_data)

	print(f'EXPERIMENTAL FEATURE::{fname} has been loaded on the IPC RAM with prefix {prefix}')
	print("Beware of memory leaks!!")


