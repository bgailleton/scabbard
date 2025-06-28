'''
This module provides Numba-optimized functions for flow routing and distance calculations
within the Steenbok (Numba engine) mirror of Riverdale's conventions.

Author: B.G. (last modification: 07/2024 - Acigné)
'''

import numba as nb
import numpy as np
from enum import Enum
import scabbard as scb


@nb.njit()
def mean_dist_to_outlet(Stack, Z, BCs, D8, nx, ny, dx):
	"""
	Internal computation of the mean flow distance to outlets.

	This function calculates the average flow distance from each node to an outlet
	by traversing the topologically ordered nodes from downstream to upstream.

	Args:
		Stack (numpy.ndarray): A 1D NumPy array of topologically ordered node indices.
		Z (numpy.ndarray): A 1D NumPy array of topographic elevations (flattened grid).
		BCs (numpy.ndarray): A 1D NumPy array of boundary condition codes (flattened grid).
		D8 (bool): If True, uses D8 (eight-direction) flow routing; otherwise, uses D4 (four-direction).
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.
		dx (float): Spatial step size.

	Returns:
		numpy.ndarray: A 1D NumPy array of mean flow distances from outlets.

	Author: B.G. (last modifications 09/2024)
	"""

	# Initialize the distance array to -1 (indicating "not computed yet")
	dist = np.zeros_like(Z) - 1

	# Traverse the nodes in ascending order (downstream to upstream)
	for node in Stack:

		# Check if the current node is an outlet (starting point for distance calculation)
		if(scb.ste.can_out_flat(node,BCs)):
			dist[node] = 0

		# Skip if the node is invalid (e.g., inactive or cannot give flow)
		if(scb.ste.is_active_flat(node,BCs) == False) or (scb.ste.can_give_flat(node,BCs) == False):
			continue

		# Initialize prospective value and counter for averaging
		val = 0.
		N = 0
		# Iterate over all neighbors (4 for D4, 8 for D8)
		for k in range(8 if D8 else 4):
			# Get the neighbor's flat index
			rec = scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny) if D8 else scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny)
			# Check if the neighbor is valid (will be -1 if no data or cannot give)
			if(rec == -1):
				continue

			# Check for edge case where internal unprocessed local minima might point to itself
			if(rec == node):
				continue
			# Check if the neighbor is actually a receiver (downslope)
			if(Z[node] <= Z[rec]):
				continue

			# If valid, increment counter and add distance to the prospective value
			N +=1
			val += dist[rec] + (scb.ste.dx_from_k_D8(dx,k) if D8 else dx)
		
		# After checking all neighbors, apply the mean if there are valid receivers
		if(N>0):
			dist[node] = val/N

	return dist

@nb.njit()
def min_dist_to_outlet(Stack, Z, BCs, D8, nx, ny, dx):
	"""
	Internal computation of the minimum flow distance to outlets.

	This function calculates the shortest flow distance from each node to an outlet
	by traversing the topologically ordered nodes from downstream to upstream.

	Args:
		Stack (numpy.ndarray): A 1D NumPy array of topologically ordered node indices.
		Z (numpy.ndarray): A 1D NumPy array of topographic elevations (flattened grid).
		BCs (numpy.ndarray): A 1D NumPy array of boundary condition codes (flattened grid).
		D8 (bool): If True, uses D8 (eight-direction) flow routing; otherwise, uses D4 (four-direction).
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.
		dx (float): Spatial step size.

	Returns:
		numpy.ndarray: A 1D NumPy array of minimum flow distances from outlets.

	Author: B.G. (last modifications 09/2024)
	"""

	# Initialize the distance array to -1 (indicating "not computed yet")
	dist = np.zeros_like(Z) - 1

	# Traverse the nodes in ascending order (downstream to upstream)
	for node in Stack:

		# Check if the current node is an outlet (starting point for distance calculation)
		if(scb.ste.can_out_flat(node,BCs)):
			dist[node] = 0

		# Skip if the node is invalid (e.g., inactive or cannot give flow)
		if(scb.ste.is_active_flat(node,BCs) == False) or (scb.ste.can_give_flat(node,BCs) == False):
			continue

		# Initialize prospective value to a very large number for minimum comparison
		val = 1e32
		# Iterate over all neighbors (4 for D4, 8 for D8)
		for k in range(8 if D8 else 4):
			# Get the neighbor's flat index
			rec = scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny) if D8 else scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny)
			# Check if the neighbor is valid (will be -1 if no data or cannot give)
			if(rec == -1):
				continue

			# Check for edge case where internal unprocessed local minima might point to itself
			if(rec == node):
				continue
			# Check if the neighbor is actually a receiver (downslope)
			if(Z[node] <= Z[rec]):
				continue

			# Update the minimum distance to that receiver and the other
			val = min(dist[rec] + (scb.ste.dx_from_k_D8(dx,k) if D8 else dx), val)

		dist[node] = val

	return dist


@nb.njit()
def max_dist_to_outlet(Stack, Z, BCs, D8, nx, ny, dx):
	"""
	Internal computation of the maximum flow distance to outlets.

	This function calculates the longest flow distance from each node to an outlet
	by traversing the topologically ordered nodes from downstream to upstream.

	Args:
		Stack (numpy.ndarray): A 1D NumPy array of topologically ordered node indices.
		Z (numpy.ndarray): A 1D NumPy array of topographic elevations (flattened grid).
		BCs (numpy.ndarray): A 1D NumPy array of boundary condition codes (flattened grid).
		D8 (bool): If True, uses D8 (eight-direction) flow routing; otherwise, uses D4 (four-direction).
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.
		dx (float): Spatial step size.

	Returns:
		numpy.ndarray: A 1D NumPy array of maximum flow distances from outlets.

	Author: B.G. (last modifications 09/2024)
	"""

	# Initialize the distance array to -1 (indicating "not computed yet")
	dist = np.zeros_like(Z) - 1

	# Traverse the nodes in ascending order (downstream to upstream)
	for node in Stack:

		# Check if the current node is an outlet (starting point for distance calculation)
		if(scb.ste.can_out_flat(node,BCs)):
			dist[node] = 0

		# Skip if the node is invalid (e.g., inactive or cannot give flow)
		if(scb.ste.is_active_flat(node,BCs) == False) or (scb.ste.can_give_flat(node,BCs) == False):
			continue

		# Initialize prospective value to 0 for maximum comparison
		val = 0.
		# Iterate over all neighbors (4 for D4, 8 for D8)
		for k in range(8 if D8 else 4):
			# Get the neighbor's flat index
			rec = scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny) if D8 else scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny)
			# Check if valid (will be -1 if no data or cannot give)
			if(rec == -1):
				continue

			# Check for edge case where internal unprocessed local minima might point to itself
			if(rec == node):
				continue
			# Check if the neighbor is actually a receiver (downslope)
			if(Z[node] <= Z[rec]):
				continue

			# Update the maximum distance to that receiver and the other
			val = max(dist[rec] + (scb.ste.dx_from_k_D8(dx,k) if D8 else dx), val)

		dist[node] = val

	return dist