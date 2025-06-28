# -*- coding: utf-8 -*-
"""
This module contains utility functions for manipulating numpy arrays.

It includes functions for resampling arrays to a new shape and removing
isolated components from a binary mask.
"""

# __author__ = "B.G."

from scipy.ndimage import zoom, label
import numpy as np
import numba as nb

def resample_to_shape(array, new_shape, order=3):
    """
    Resamples a 2D array to a new shape using interpolation.

    Args:
        array (numpy.ndarray): The 2D array to be resampled.
        new_shape (tuple): The desired shape (rows, columns) for the resampled array.
        order (int, optional): The order of the spline interpolation. Defaults to 3.

    Returns:
        numpy.ndarray: The resampled array.
    """
    # Calculate the zoom factors for each dimension
    zoom_factors = (new_shape[0] / array.shape[0], new_shape[1] / array.shape[1])

    # Use scipy.ndimage.zoom to resample the array
    resampled_array = zoom(array, zoom_factors, order=order)

    return resampled_array

def remove_unconnected_components(mask, th_components=10, D8=True):
    """
    Removes small, unconnected components from a binary mask in place.

    This function identifies connected components in a binary mask and removes
    those smaller than a given threshold.

    Args:
        mask (numpy.ndarray): A 2D numpy array of 0s and 1s.
        th_components (int, optional): The minimum number of connected pixels
                                     for a component to be retained. Defaults to 10.
        D8 (bool, optional): If True, considers diagonal pixels as connected (D8 connectivity).
                             If False, only considers cardinal directions (D4 connectivity).
                             Defaults to True.
    """

    @nb.njit
    def _remover_for_remove_unconnected_components(mask, labels, N, th_components):
        """
        Numba-optimized function to remove small components based on their labels.

        Args:
            mask (numpy.ndarray): The binary mask to be modified.
            labels (numpy.ndarray): The labeled array of connected components.
            N (int): The number of unique labels.
            th_components (int): The size threshold for removing components.
        """
        # Count the size of each component
        Ns = np.zeros(N, dtype=np.uint32)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                Ns[labels[i, j]] += 1

        # Remove components smaller than the threshold
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if Ns[labels[i, j]] <= th_components:
                    mask[i, j] = 0

    # Define the connectivity structure for labeling
    structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]] if not D8 else [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    # Label the connected components in the mask
    labeled_array, num_features = label(mask, structure=structure)

    # Remove the small components using the Numba-optimized function
    _remover_for_remove_unconnected_components(mask, labeled_array, num_features, th_components)