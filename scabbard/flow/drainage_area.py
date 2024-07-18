'''
Routines to calculate drainage area and other similar stuff
'''

import numba as nb
import numpy as np
import scabbard.steenbok as st
from scabbard.flow import SFGraph


@nb.njit()
def _drainage_area_sfg(Stack,Sreceivers, dx = 1., BCs = None):

	A = np.zeros_like(Sreceivers, dtype = np.float32)

	for i in range(Stack.shape[0]):
		node = Stack[Stack.shape[0] - 1 - i]
		rec = Sreceivers[node]

		if(node == rec):
			continue

		A[node] += dx * dx
		A[rec] += A[node]

	return A



def drainage_area(input_data):

	if(isinstance(input_data, SFGraph) == False):
		raise AttributeError('drainage area WIP, so far requires SFGraph object to calculate')

	return _drainage_area_sfg(input_data.Stack.ravel(), input_data.Sreceivers.ravel(), dx = input_data.dx).reshape(input_data.ny,input_data.nx)


























































# end of file