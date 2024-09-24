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

	dist = np.zeros_like(Z) - 1

	for node in Stack:

		if(scb.ste.can_out_flat(node,BCs)):
			dist[node] = 0

		if(scb.ste.is_active_flat(node,BCs) == False) or (scb.ste.can_give_flat(node,BCs) == False):
			continue

		val = 0.
		N = 0
		for k in range(8 if D8 else 4):
			rec = scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny) if D8 else scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny)
			if(rec == -1):
				continue
			if(rec == node):
				continue

			if(Z[node] <= Z[rec]):
				continue
			N +=1
			val += dist[rec] + (scb.ste.dx_from_k_D8(dx,k) if D8 else dx)
		if(N>0):
			dist[node] = val/N

	return dist

@nb.njit()
def min_dist_to_outlet(Stack, Z, BCs, D8, nx, ny, dx):

	dist = np.zeros_like(Z) - 1

	for node in Stack:

		if(scb.ste.can_out_flat(node,BCs)):
			dist[node] = 0

		if(scb.ste.is_active_flat(node,BCs) == False) or (scb.ste.can_give_flat(node,BCs) == False):
			continue

		val = 1e32
		for k in range(8 if D8 else 4):
			rec = scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny) if D8 else scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny)
			if(rec == -1):
				continue
			if(rec == node):
				continue

			if(Z[node] <= Z[rec]):
				continue
			val = min(dist[rec] + (scb.ste.dx_from_k_D8(dx,k) if D8 else dx), val)

		dist[node] = val

	return dist


@nb.njit()
def max_dist_to_outlet(Stack, Z, BCs, D8, nx, ny, dx):

	dist = np.zeros_like(Z) - 1

	for node in Stack:

		if(scb.ste.can_out_flat(node,BCs)):
			dist[node] = 0

		if(scb.ste.is_active_flat(node,BCs) == False) or (scb.ste.can_give_flat(node,BCs) == False):
			continue

		val = 0.
		for k in range(8 if D8 else 4):
			rec = scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny) if D8 else scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny)
			if(rec == -1):
				continue
			if(rec == node):
				continue

			if(Z[node] <= Z[rec]):
				continue
			val = max(dist[rec] + (scb.ste.dx_from_k_D8(dx,k) if D8 else dx), val)

		dist[node] = val

	return dist