# -*- coding: utf-8 -*-
"""
This module provides a helper function for converting string representations of data types
to their corresponding numpy data types.

It is part of an archived collection of CUDA-related code.
"""

# __author__ = "B.G."

import numpy as np

def str2type(tinput):
    """
    Converts a string identifier to a numpy data type.

    This function maps simple string codes (e.g., 'u8', 'i32', 'f32') to their
    respective numpy data type objects.

    Args:
        tinput (str): The string representation of the data type.

    Returns:
        numpy.dtype: The corresponding numpy data type.

    Raises:
        TypeError: If the input string does not match any of the recognized
                   data type codes.
    """
    tinput = tinput.lower()

    if tinput == 'u8':
        return np.uint8
    if tinput == 'i32':
        return np.int32
    if tinput == 'f32':
        return np.float32
    
    raise TypeError(f'Type "{tinput}" not recognized. Use 'u8', 'i32', or 'f32'.')
