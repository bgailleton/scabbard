# -*- coding: utf-8 -*-
"""
This module provides utility functions for interacting with CUDA kernels and managing data
between the CPU and GPU.

It is part of an archived collection of CUDA-related code.
"""

# __author__ = "B.G."

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import scabbard.steenbok.dtype_helper as typh
import inspect

def get_current_function_name():
    """Returns the name of the function that calls this function."""
    return inspect.currentframe().f_back.f_code.co_name

def set_constant(mod, val, ref, dtype):
    """
    Sets a global constant in a compiled CUDA module.

    Args:
        mod (pycuda.compiler.SourceModule): The compiled CUDA module.
        val: The value to set the constant to.
        ref (str): The name of the constant in the CUDA code.
        dtype (str): The data type of the constant (e.g., 'f32', 'i32').

    Returns:
        tuple: A tuple containing the value on the CPU and the GPU memory object.
    """
    # Convert the string data type to a numpy dtype
    ttype = typh.str2type(dtype)
    val_cpu = ttype(val)

    # Get the GPU memory address for the constant
    val_gpu = mod.get_global(ref)[0]

    # Copy the value from CPU to GPU
    drv.memcpy_htod(val_gpu, val_cpu)
    return val_cpu, val_gpu

def set_array(mod, val, ref, dtype):
    """
    Allocates and transfers a numpy array to the GPU.

    Args:
        mod (pycuda.compiler.SourceModule): The compiled CUDA module.
        val (numpy.ndarray): The array to transfer.
        ref (str): The name of the array in the CUDA code (for reference, not used here).
        dtype (str): The data type of the array.

    Returns:
        tuple: A tuple containing the array on the CPU and its GPU memory allocation.
    """
    # Convert the string data type to a numpy dtype
    ttype = typh.str2type(dtype)
    val_cpu = val.astype(ttype)

    # Allocate memory on the GPU and copy the array to it
    val_gpu = drv.mem_alloc(val_cpu.nbytes)
    drv.memcpy_htod(val_gpu, val_cpu)
    return val_cpu, val_gpu

def get_array(mod, arrcpu, arrgpu):
    """
    Transfers an array from the GPU to the CPU.

    Args:
        mod (pycuda.compiler.SourceModule): The compiled CUDA module.
        arrcpu (numpy.ndarray): The destination array on the CPU.
        arrgpu (pycuda.driver.DeviceAllocation): The source array on the GPU.
    """
    drv.memcpy_dtoh(arrcpu, arrgpu)

class arrayHybrid:
    """
    A helper class for managing arrays that exist on both the CPU and GPU.

    This class simplifies the process of creating, transferring, and deleting
    arrays that are used in CUDA computations.
    """
    def __init__(self, mod, val, ref, ttype):
        """
        Initializes a new arrayHybrid object.

        Args:
            mod (pycuda.compiler.SourceModule): The compiled CUDA module.
            val (numpy.ndarray): The initial array value.
            ref (str): A reference name for the array.
            ttype (str): The data type of the array (e.g., 'f32').
        """
        self.dtype = ttype
        self._dtype = typh.str2type(ttype)
        self.mod = mod
        self.ref = ref
        self._cpu, self._gpu = set_array(self.mod, val, self.ref, self.dtype)
        self.nn = self._cpu.shape[0]

    def cpu2gpu(self):
        """Transfers the array from CPU to GPU."""
        drv.memcpy_htod(self._gpu, self._cpu)

    def delete(self, cpu=False, gpu=True):
        """
        Deletes the array from memory.

        Args:
            cpu (bool, optional): Whether to delete the CPU copy. Defaults to False.
            gpu (bool, optional): Whether to delete the GPU copy. Defaults to True.
        """
        if gpu:
            self._gpu.free()
        if cpu:
            self._cpu = None

    def set(self, val, gpu=True, cpu=True):
        """
        Sets a new value for the array.

        Args:
            val (numpy.ndarray): The new array value.
            gpu (bool, optional): Whether to update the GPU copy. Defaults to True.
            cpu (bool, optional): Whether to update the CPU copy. Defaults to True.
        """
        if gpu and cpu:
            self._cpu, self._gpu = set_array(self.mod, val, self.ref, self.dtype)
        else:
            if cpu:
                self._cpu = val
            if gpu:
                _, self._gpu = set_array(self.mod, val, self.ref, self.dtype)

    def get(self, cpu=False, gpu=True):
        """
        Gets the array from either the CPU or GPU.

        Args:
            cpu (bool, optional): If True, returns the CPU copy. Defaults to False.
            gpu (bool, optional): If True, transfers from GPU and returns the CPU copy.
                                Defaults to True.

        Returns:
            numpy.ndarray: The requested array.
        """
        if cpu == gpu:
            raise ValueError("Please specify either cpu or gpu, not both or neither.")
        
        if cpu:
            return self._cpu

        get_array(self.mod, self._cpu, self._gpu)
        return self._cpu

def aH_zeros_like(mod, arr, ttype, ref="temp"):
    """
    Creates a new arrayHybrid object with the same shape as another array, filled with zeros.

    Args:
        mod (pycuda.compiler.SourceModule): The compiled CUDA module.
        arr (numpy.ndarray): The array to copy the shape from.
        ttype (str): The data type of the new array.
        ref (str, optional): A reference name for the new array. Defaults to "temp".

    Returns:
        arrayHybrid: The new arrayHybrid object.
    """
    val = np.zeros_like(arr, dtype=typh.str2type(ttype))
    return arrayHybrid(mod, val, ref, ttype)

def aH_zeros(mod, sizzla, ttype, ref="temp"):
    """
    Creates a new arrayHybrid object of a given size, filled with zeros.

    Args:
        mod (pycuda.compiler.SourceModule): The compiled CUDA module.
        sizzla (tuple): The shape of the new array.
        ttype (str): The data type of the new array.
        ref (str, optional): A reference name for the new array. Defaults to "temp".

    Returns:
        arrayHybrid: The new arrayHybrid object.
    """
    val = np.zeros(sizzla, dtype=typh.str2type(ttype))
    return arrayHybrid(mod, val, ref, ttype)