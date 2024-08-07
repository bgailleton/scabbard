import numba as nb
import numpy as np


@nb.njit()
def mask_watershed_SFD(strat_node, Stack ,Sreceivers):
	'''
	mask all the nodes draining to a single one following an already calculated single flow graph
	'''

	mask = np.zeros_like(Sreceivers, dtype = np.uint8)

	mask[strat_node] = 1

	for node in Stack:
		if(node == strat_node):
			continue
			
		mask[node] = mask[Sreceivers[node]]

	# As simple as that
	return mask


