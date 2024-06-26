'''
General sets of utilities to check type, characterise functions, classes, ...

B.G.
'''
import numpy as np



def singleton(cls):
	'''
	Decorator ensuring a classes becomes a singleton (= can only be instantiated once)
	
	To use: 
	@singleton
	class myclass:
		...
	
	Then myclass canonly be instantiated once

	Authors:
		- B.G. (last modification: 28/05/2024)
		- Let's be honest, chatGPT 4 helped me on that one
	'''
	instances = {}
	def get_instance(*args, **kwargs):
		if cls not in instances:
			instances[cls] = cls(*args, **kwargs)
		return instances[cls]
	return get_instance


def is_numpy(val, shape = None, dtype = None):
	'''
	Determines if an input is a numpy array or not, eventually fitting some conditions about type and shape
	TODO: add dimension check
	Arguments:
		- val: the variable to test
		- shape: tuple the array.shape should return. Ignored if None.
		- dtype: the type array.dtype should belong to. Note that array.dtype can be a subtype of the target type (e.g. np.float32 is a subtype of np.floatings). Ignored if None.
	Returns:
		- A boolean telling if the input variable is a numpy array satisfying the required conditions
	Authors:
		- B.G. (last modification: 28/05/2024)
	'''
	
	# First checking if the value is a numpy array	
	if(isinstance(val, np.ndarray) == False):
		return False

	# Eventually checking its shape
	if(shape is not None and shape != val.shape):
		return False

	# Eventually checking its dtype
	if(dtype is not None and np.issubdtype(val.dtype, dtype) == False):
		return False

	# If the code reaches there then all good, the array is valid
	return True


def print_neighbourhood2D(arr, row, col, precision = 6):
	'''
	'''
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

	