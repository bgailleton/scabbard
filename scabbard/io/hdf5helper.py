import h5py
import numpy as np

def save_array_in_group(file_path, array, group_name, dataset_name, overwrite=True):
	"""Save an array within a specified group in an HDF5 file, with optional overwrite."""
	with h5py.File(file_path, 'a') as f:
		group = f.require_group(group_name)  # Create the group if it doesn't exist
		
		if dataset_name in group:
			if overwrite:
				del group[dataset_name]  # Delete existing dataset if overwrite is enabled
			else:
				print(f"Dataset '{dataset_name}' already exists in '{group_name}'. Skipping.")
				return
		
		group.create_dataset(dataset_name, data=array, compression="gzip")

def load_array_from_group(file_path, group_name, dataset_name):
	"""Load an array from a specified group (subfolder) in an HDF5 file."""
	with h5py.File(file_path, 'r') as f:
		return f[f"{group_name}/{dataset_name}"][()]

def load_all_arrays_in_group(file_path, group_name):
	"""Load all arrays within a specific group into a dictionary."""
	arrays = {}
	with h5py.File(file_path, 'r') as f:
		group = f[group_name]
		for name in group:
			arrays[name] = group[name][()]
	return arrays


def inspect_hdf5(file_path):
	"""Inspect and print details of an HDF5 file."""
	with h5py.File(file_path, 'r') as f:
		def explore(name, obj):
			if isinstance(obj, h5py.Dataset):
				print(f"Dataset: {name}")
				print(f" - Shape: {obj.shape}")
				print(f" - Data Type: {obj.dtype}")
				if obj.compression:
					print(f" - Compression: {obj.compression}")
					print(f" - Compression Options: {obj.compression_opts}")
				else:
					print(" - Compression: None")
			elif isinstance(obj, h5py.Group):
				print(f"Group: {name}")

		# Walk through each item in the HDF5 file
		f.visititems(explore)


def list_datasets_in_group(file_path, group_name):
	"""Return a list of all dataset names in a specified group."""
	with h5py.File(file_path, 'r') as f:
		if group_name in f:
			group = f[group_name]
			return [name for name in group if isinstance(group[name], h5py.Dataset)]
		else:
			print(f"Group '{group_name}' does not exist in the file.")
			return []