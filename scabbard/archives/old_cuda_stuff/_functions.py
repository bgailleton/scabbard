# -*- coding: utf-8 -*-
"""
This module contains functions for building and debugging CUDA kernels.

It is part of an archived collection of CUDA-related code and is likely outdated.
"""

# __author__ = "B.G."

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import scabbard as scb
import numpy as np
import math as m
import os

# Define the path to the steenbok directory
PATH2STEENBOK = os.path.join(os.path.dirname(__file__))

def concat_kernel_code(topology):
    """
    Concatenates CUDA kernel code from multiple files into a single string.

    This function reads a predefined list of .cu files, concatenates their
    contents, and replaces a macro for the number of neighbors based on the
    specified topology (D4 or D8).

    Args:
        topology (str): The grid topology, either "D8" or "D4".

    Returns:
        str: The combined CUDA kernel code.
    """
    # Initialize an empty string to hold the kernel code
    kernel_code = ""
    # List of CUDA files to be concatenated in a specific order
    files = [
        "includer.cu",
        "macros_holder.cu",
        "constants_holder.cu",
        "bc_helper.cu",
        "neighbourer.cu",
        "grid_tools.cu",
        "array_utils.cu",
        "graphflood_hydro.cu",
    ]

    # Read and append the content of each file
    for file in files:
        with open(os.path.join(PATH2STEENBOK, file), 'r') as f:
            kernel_code += f.read() + '\n\n'

    # Replace the neighbor macro based on the topology
    kernel_code = kernel_code.replace("MACRO2SETNNEIGHBOURS", "8" if topology == "D8" else "4")

    return kernel_code

def debug_kernel(topology):
    """
    Builds and writes a debug version of the CUDA kernel to a file.

    This function concatenates the kernel code, writes it to a file named
    "DEBUGKERNEL.cu", and then compiles it using PyCUDA's SourceModule.

    Args:
        topology (str): The grid topology, either "D8" or "D4".
    """
    # Concatenate the kernel code
    kernel_code = concat_kernel_code(topology)

    # Write the combined code to a debug file
    with open("DEBUGKERNEL.cu", 'w') as f:
        f.write(kernel_code)

    # Compile the kernel
    mod = SourceModule(kernel_code)

def build_kernel(topology):
    """
    Builds the CUDA kernel and extracts all global functions.

    This function concatenates the kernel code, compiles it, and then
    dynamically finds and returns all the `__global__` functions defined
    within the kernel code.

    Args:
        topology (str): The grid topology, either "D8" or "D4".

    Returns:
        tuple: A tuple containing:
            - mod (pycuda.compiler.SourceModule): The compiled CUDA module.
            - functions (dict): A dictionary of the extracted global functions.
    """
    # Concatenate the kernel code
    kernel_code = concat_kernel_code(topology)

    # Compile the kernel
    mod = SourceModule(kernel_code)
    functions = {}

    # Iterate through the lines of the kernel code to find global functions
    for line in kernel_code.splitlines():
        tline = line.split(' ')

        # Check if the line defines a global function
        globi = -1
        for index, item in enumerate(tline):
            if "__global__" in item:
                globi = index

        if globi == -1:
            continue

        # Extract the function name
        tfunc = tline[globi + 2].split('(')[0]
        print('Fetching', tfunc)
        try:
            # Get the function from the compiled module
            functions[tfunc] = mod.get_function(tfunc)
        except pycuda.driver.LogicError:
            print("failed, no", tfunc, "found")
        print('OK')

    return mod, functions