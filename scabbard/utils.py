'''
This module provides a collection of general utility functions for scabbard.
It includes decorators, type checking functions, and debugging tools.

Author: B.G.
'''
import numpy as np



def singleton(cls):
	"""
	Decorator ensuring a class becomes a singleton (can only be instantiated once).

	To use:
	@singleton
	class MyClass:
		...

	Then `MyClass` can only be instantiated once within the application's lifetime.

	Args:
		cls (type): The class to be decorated as a singleton.

	Returns:
		callable: A function that returns the single instance of the class.

	Authors:
		- B.G. (last modification: 28/05/2024)
		- Inspired by discussions with ChatGPT 4.
	"""
	instances = {}
	def get_instance(*args, **kwargs):
		if cls not in instances:
			instances[cls] = cls(*args, **kwargs)
		return instances[cls]
	return get_instance


def is_numpy(val, shape = None, dtype = None):
	"""
	Checks if an input is a NumPy array and optionally validates its shape and data type.

	Args:
		val (any): The variable to test.
		shape (tuple, optional): The expected shape of the array. Ignored if None.
							Defaults to None.
		dtype (numpy.dtype, optional): The expected data type of the array. Note that `np.issubdtype`
							is used, so `val.dtype` can be a subtype of the target type (e.g., `np.float32` is a
							subtype of `np.floating`). Ignored if None. Defaults to None.

	Returns:
		bool: True if the input variable is a NumPy array satisfying the required conditions,
			  False otherwise.

	Author: B.G. (last modification: 28/05/2024)
	"""
	
	# First, check if the value is a NumPy array	
	if(isinstance(val, np.ndarray) == False):
		return False

	# Optionally, check its shape
	if(shape is not None and shape != val.shape):
		return False

	# Optionally, check its data type (allowing for subtypes)
	if(dtype is not None and np.issubdtype(val.dtype, dtype) == False):
		return False

	# If all checks pass, the array is valid
	return True


def print_neighbourhood2D(arr, row, col, precision = 6):
	"""
	Prints the 3x3 neighborhood of a specified cell in a 2D array.

	This function is useful for debugging and inspecting local data values around a point.

	Args:
		arr (numpy.ndarray): The 2D NumPy array to inspect.
		row (int): The row index of the central cell.
		col (int): The column index of the central cell.
		precision (int, optional): The number of decimal places to format the printed values.
							Defaults to 6.

	Returns:
		None: Prints the neighborhood to the console.

	Author: B.G.
	"""
	format_spec = f".{precision}f"
	gog = 0
	for i in [-1,0,1]:
		for j in [-1,0,1]:
			print('|', end = '')
			gog += 1
			if(gog == 3):
				gog = 0
				print(f"{ arr[ row + i , col + j ]:{format_spec}}", end = ' \n')
			else:
				print(f"{ arr[ row + i , col + j ]:{format_spec}}", end = ' | ')

	