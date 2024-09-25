'''
Riverdale's mirror for the numba engine (steenbock) convention for the nth neighbouring:

B.G - 07/2024 - Acigné

'''

import numba as nb
import numpy as np
from enum import Enum
import scabbard as scb


@nb.njit()
def mean_dist_to_outlet(Stack, Z, BCs, D8, nx, ny, dx):
	'''
	Internal computation of the mean flow distance to outlets

	Arguments:
		- Stack: topologically ordered nodes
		- Z: array of topography
		- BCs: array of boundary conditions codes
		- D8: bool, D8 if True else D4 flow routing
		- nx,ny,dx: dimensions and spatial steps
	Returns:
		- an array of flow distance from outlet

	Authors:
		- B.G. (last modifications 09/2024)
	'''

	# Init the distance to -1 as "not computed yet"
	dist = np.zeros_like(Z) - 1

	# Traversing the nodes in ascending order, downstream to upstream
	for node in Stack:

		# Checking if node is outlet - i.e. starting point
		if(scb.ste.can_out_flat(node,BCs)):
			dist[node] = 0

		# Checking if the node is valid to receive flow distance (e.g. can it give, is it no data)
		if(scb.ste.is_active_flat(node,BCs) == False) or (scb.ste.can_give_flat(node,BCs) == False):
			continue

		# prospective value
		val = 0.
		# Tracking number of values to average
		N = 0
		# For all neighbours (4 v. 8)
		for k in range(8 if D8 else 4):
			# get the neighbour
			rec = scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny) if D8 else scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny)
			# Check if valid (will be -1 if no data or cannot give)
			if(rec == -1):
				continue

			# Check edge case where internal unprocessed LM
			if(rec == node):
				continue
			# Check if actually is a receiver
			if(Z[node] <= Z[rec]):
				continue

			# Node is valid, increment number
			N +=1
			# incrementing the value
			val += dist[rec] + (scb.ste.dx_from_k_D8(dx,k) if D8 else dx)
		
		# end of loop and applying the mean
		if(N>0):
			dist[node] = val/N

	return dist

@nb.njit()
def min_dist_to_outlet(Stack, Z, BCs, D8, nx, ny, dx):
	'''
	Internal computation of the min flow distance to outlets

	Arguments:
		- Stack: topologically ordered nodes
		- Z: array of topography
		- BCs: array of boundary conditions codes
		- D8: bool, D8 if True else D4 flow routing
		- nx,ny,dx: dimensions and spatial steps
	Returns:
		- an array of flow distance from outlet

	Authors:
		- B.G. (last modifications 09/2024)
	'''

	# Init the distance to -1 as "not computed yet"
	dist = np.zeros_like(Z) - 1

	# Traversing the nodes in ascending order, downstream to upstream
	for node in Stack:

		# Checking if node is outlet - i.e. starting point
		if(scb.ste.can_out_flat(node,BCs)):
			dist[node] = 0

		# Checking if the node is valid to receive flow distance (e.g. can it give, is it no data)
		if(scb.ste.is_active_flat(node,BCs) == False) or (scb.ste.can_give_flat(node,BCs) == False):
			continue

		# prospective value
		val = 1e32
		# For all neighbours (4 v. 8)
		for k in range(8 if D8 else 4):
			# get the neighbour
			rec = scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny) if D8 else scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny)
			# Check if valid (will be -1 if no data or cannot give)
			if(rec == -1):
				continue

			# Check edge case where internal unprocessed LM
			if(rec == node):
				continue
			# Check if actually is a receiver
			if(Z[node] <= Z[rec]):
				continue

			# distance is the min of dist to that receiver and the other
			val = min(dist[rec] + (scb.ste.dx_from_k_D8(dx,k) if D8 else dx), val)

		dist[node] = val

	return dist


@nb.njit()
def max_dist_to_outlet(Stack, Z, BCs, D8, nx, ny, dx):
	'''
	Internal computation of the max flow distance to outlets

	Arguments:
		- Stack: topologically ordered nodes
		- Z: array of topography
		- BCs: array of boundary conditions codes
		- D8: bool, D8 if True else D4 flow routing
		- nx,ny,dx: dimensions and spatial steps
	Returns:
		- an array of flow distance from outlet

	Authors:
		- B.G. (last modifications 09/2024)
	'''

	# Init the distance to -1 as "not computed yet"
	dist = np.zeros_like(Z) - 1

	# Traversing the nodes in ascending order, downstream to upstream
	for node in Stack:

		# Checking if node is outlet - i.e. starting point
		if(scb.ste.can_out_flat(node,BCs)):
			dist[node] = 0

		# Checking if the node is valid to receive flow distance (e.g. can it give, is it no data)
		if(scb.ste.is_active_flat(node,BCs) == False) or (scb.ste.can_give_flat(node,BCs) == False):
			continue

		# prospective value
		val = 0.
		# For all neighbours (4 v. 8)
		for k in range(8 if D8 else 4):
			# get the neighbour
			rec = scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny) if D8 else scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny)
			# Check if valid (will be -1 if no data or cannot give)
			if(rec == -1):
				continue

			# Check edge case where internal unprocessed LM
			if(rec == node):
				continue
			# Check if actually is a receiver
			if(Z[node] <= Z[rec]):
				continue

			# distance is the max of dist to that receiver and the other
			val = max(dist[rec] + (scb.ste.dx_from_k_D8(dx,k) if D8 else dx), val)

		dist[node] = val

	return dist