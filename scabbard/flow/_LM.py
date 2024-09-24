'''
Numba scripts dedicated to local minimas
'''


import numpy as np
import numba as nb

@nb.njit()
def impose_downstream_minimum_elevation_decrease(Z, Sstack, Sreceivers, delta = 1e-4):

	for i in range(Z.shape[0]):
		node = Sstack[Z.shape[0] - 1 - i]
		rec = Sreceivers[node]

		if(node != rec):
			if(Z[rec] >= Z[node] - delta):
				Z[rec] = Z[node] - delta
