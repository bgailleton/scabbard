'''
Set of functions to generate specific arrays that I commonly use

B.G. (last modifications: 07/2024)
'''

import numpy as np

def normal_BCs_from_shape(nx,ny, out_code = 3):
	'''
	Returns an array on uint8 boundary codes with "normal" edges

	Arguments:
		- nx (int): number of columns
		- ny (int): number of rows
		- out_code (uint8): code for the edges, 3 is "can_out", a permissive code to let flow out if no downstream neighbour

	returns:
		- a 2D numpy array of boundary conditions
	
	Authors:
		- B.G. (last modification 07/2024)
	'''

	BCs = np.ones((ny,nx), dtype = np.uint8)
	BCs[:,[-1,0]] = out_code
	BCs[[-1,0], :] = out_code

	return BCs
