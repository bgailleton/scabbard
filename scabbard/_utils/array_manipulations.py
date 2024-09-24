'''
Contains sets of utils function to manipulate arrays
'''

from scipy.ndimage import zoom, label
import numpy as np
import numba as nb


def resample_to_shape(array, new_shape, order = 3):
	"""
	Resample a 2D array to a new shape.
	
	Parameters:
		- array (2D numpy array): The original array to be resampled.
		- new_shape (tuple): The desired shape (rows, columns) for the resampled array.
	
	Returns:
		- 2D numpy array: The resampled array with the specified shape.
	
	Authors: 
		- B.G.
		- a LLM
	"""
	# Calculate the zoom factors for each dimension
	zoom_factors = (new_shape[0] / array.shape[0], new_shape[1] / array.shape[1])
	
	# Use scipy.ndimage.zoom to resample the array
	resampled_array = zoom(array, zoom_factors, order=order)
	
	return resampled_array


def remove_unconnected_components(mask, th_components = 10, D8 = True):
	'''
	Edit a mask in place: removes (switch to 0) components that are isolated/in patches of less than a threshold

	Arguments:
		- mask: 2D numpy array of 0s and 1s
		- the minimum number of connected component
		- D8: if True, takes diagonals into account, else only cardinals

	Returns:
		Nothing, edits mask in place 
	'''

	# Nested function to apply the removal (much faster than numpy)
	@nb.njit
	def _remover_for_remove_unconnected_components(mask, labels, N, th_components):

		Ns = np.zeros(N, dtype = np.uint32)

		for i in range(labels.shape[0]):
			for j in range(labels.shape[1]):
				Ns[labels[i,j]] += 1

		for i in range(labels.shape[0]):
			for j in range(labels.shape[1]):
				if(Ns[labels[i,j]] <= th_components):
					mask[i,j] = 0


	# defines the neighbouring
	structure = [[0,1,0],[1,1,1],[0,1,0]] if D8 == False else [[1,1,1],[1,1,1],[1,1,1]]

	# connected components
	labeled_array, num_features = label(mask, structure = structure)
	
	# correct the mask
	_remover_for_remove_unconnected_components(mask, labeled_array, num_features, th_components)

