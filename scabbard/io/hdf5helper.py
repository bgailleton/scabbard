# -*- coding: utf-8 -*-
"""
This module provides helper functions for interacting with HDF5 files.

It simplifies saving, loading, and inspecting numpy arrays within HDF5 groups.
"""

# __author__ = "B.G."

import h5py
import numpy as np

def save_array_in_group(file_path, array, group_name, dataset_name, overwrite=True):
    """
    Saves a NumPy array into a specified group (like a folder) within an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.
        array (numpy.ndarray): The NumPy array to save.
        group_name (str): The name of the group to save the array in.
        dataset_name (str): The name of the dataset (the array itself) within the group.
        overwrite (bool, optional): If True, overwrites the dataset if it already exists.
                                  If False and the dataset exists, it prints a message and skips.
                                  Defaults to True.
    """
    with h5py.File(file_path, 'a') as f:
        # Create the group if it doesn't exist, or get a reference to it
        group = f.require_group(group_name)
        
        if dataset_name in group:
            if overwrite:
                del group[dataset_name]  # Delete existing dataset
            else:
                print(f"Dataset '{dataset_name}' already exists in '{group_name}'. Skipping.")
                return
        
        # Create the dataset with gzip compression for efficiency
        group.create_dataset(dataset_name, data=array, compression="gzip")

def load_array_from_group(file_path, group_name, dataset_name):
    """
    Loads a NumPy array from a specified dataset within a group in an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.
        group_name (str): The name of the group containing the dataset.
        dataset_name (str): The name of the dataset to load.

    Returns:
        numpy.ndarray: The loaded NumPy array.
    """
    with h5py.File(file_path, 'r') as f:
        # Access the dataset using its full path within the HDF5 file
        return f[f"{group_name}/{dataset_name}"][()]

def load_all_arrays_in_group(file_path, group_name):
    """
    Loads all NumPy arrays within a specific group into a dictionary.

    Args:
        file_path (str): The path to the HDF5 file.
        group_name (str): The name of the group to load arrays from.

    Returns:
        dict: A dictionary where keys are dataset names and values are the loaded NumPy arrays.
    """
    arrays = {}
    with h5py.File(file_path, 'r') as f:
        if group_name in f:
            group = f[group_name]
            for name in group:
                # Check if the item is a dataset before loading
                if isinstance(group[name], h5py.Dataset):
                    arrays[name] = group[name][()]
        else:
            print(f"Group '{group_name}' does not exist in the file.")
    return arrays

def inspect_hdf5(file_path):
    """
    Inspects and prints details of an HDF5 file, including groups and datasets.

    Args:
        file_path (str): The path to the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        def explore(name, obj):
            """Helper function to recursively explore HDF5 objects."""
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

        # Traverse the HDF5 file and apply the explore function to each item
        f.visititems(explore)

def list_datasets_in_group(file_path, group_name):
    """
    Returns a list of all dataset names within a specified group in an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.
        group_name (str): The name of the group to list datasets from.

    Returns:
        list: A list of dataset names (strings) within the specified group.
              Returns an empty list if the group does not exist.
    """
    with h5py.File(file_path, 'r') as f:
        if group_name in f:
            group = f[group_name]
            # Filter for actual datasets (not nested groups)
            return [name for name in group if isinstance(group[name], h5py.Dataset)]
        else:
            print(f"Group '{group_name}' does not exist in the file.")
            return []
