'''
Riverdale's mirror for the numba engine (steenbock) convention for the nth neighbouring:

B.G - 07/2024 - Acigné

'''

import numba as nb
import numpy as np
from enum import Enum
import scabbard.utils as scaut 


#################################################################################################
############################## Customs Boundaries ###############################################
#################################################################################################

'''
Reminder, I am using the DAGGER convention
// Cannot flow at all = nodata
NO_FLOW = 0,

// Internal Node (can flow in every directions)
FLOW = 1,

// Internal Node (can flow in every directions) BUT neighbours a special flow
// condition and may need specific care
FLOW_BUT = 2,

// flow can out there but can also flow to downstream neighbours
CAN_OUT = 3,

// flow can only out from this cell
OUT = 4,

// Not only flow HAS to out there: neighbouring flows will be drained there no
// matter what
FORCE_OUT = 5,

// Flows through the cell is possible, but the cell CANNOT out fluxes from
// this boundary (reserved to model edges, internal boundaries wont give to
// nodata anyway)
CANNOT_OUT = 6,

// Flow can only flow to potential receivers
IN = 7,

// Forced INFLOW: flow will flow to all neighbours (except other FORCE_IN)
FORCE_IN = 8,

// periodic border
PERIODIC_BORDER = 9
'''

@nb.njit()
def _check_top_row_customs_D4(i:int, j:int, k:int, BCs, valid:bool):
	'''
	Internal function to check if neighbouring is possible for nodes at the top row
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first row
	if(i == 0):
		if(k == 0):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_top_row_customs_D4_flat(i:int, k:int, BCs, valid:bool, nx:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the top row
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first row
	if(i < nx):
		if(k == 0):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_top_row_customs_D8(i:int, j:int, k:int, BCs, valid:bool):
	'''
	Internal function to check if neighbouring is possible for nodes at the top row
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first row
	if(i == 0):
		if(k == 0 or k ==1 or k==2):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_top_row_customs_D8_flat(i:int, k:int, BCs, valid:bool, nx:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the top row
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first row
	if(i < nx):
		if(k == 0 or k ==1 or k==2):
			valid = False
	# Done
	return valid


@nb.njit()
def _check_leftest_col_customs_D4(i:int, j:int, k:int, BCs, valid:bool):
	'''
	Internal function to check if neighbouring is possible for nodes at the leftest column
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first col
	if(j == 0):
		if(k==1):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_leftest_col_customs_D4_flat(i:int, k:int, BCs, valid:bool, nx:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the leftest column
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first col
	if( (i%nx) == 0):
		if(k==1):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_leftest_col_customs_D8(i:int, j:int, k:int, BCs, valid:bool):
	'''
	Internal function to check if neighbouring is possible for nodes at the leftest column
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first col
	if(j == 0):
		if(k==0 or k == 3 or k == 5):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_leftest_col_customs_D8_flat(i:int, k:int, BCs, valid:bool, nx:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the leftest column
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first col
	if((i % nx) == 0):
		if(k==0 or k == 3 or k == 5):
			valid = False
	# Done
	return valid


@nb.njit()
def _check_rightest_col_customs_D4(i:int, j:int, k:int, BCs, valid:bool, nx:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the rightest column
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first col
	if(j == nx-1):
		if(k==2):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_rightest_col_customs_D4_flat(i:int, k:int, BCs, valid:bool, nx:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the rightest column
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first col
	if(i%nx == nx-1):
		if(k==2):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_rightest_col_customs_D8(i:int, j:int, k:int, BCs, valid:bool, nx:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the rightest column
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first col
	if(j == nx-1):
		if(k==2 or k == 4 or k == 7):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_rightest_col_customs_D8_flat(i:int, k:int, BCs, valid:bool, nx:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the rightest column
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first col
	if(i%nx == nx-1):
		if(k==2 or k == 4 or k == 7):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_bottom_row_customs_D4(i:int, j:int, k:int, BCs, valid:bool, ny:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the bottom row
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first row
	if(i == ny-1):
		# Checking all the different cases: firs, last cols and the middle
		if(k == 3):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_bottom_row_customs_D4_flat(i:int, k:int, BCs, valid:bool, nx:int, ny:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the bottom row
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first row
	if(i >= nx*ny - nx):
		# Checking all the different cases: firs, last cols and the middle
		if(k == 3):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_bottom_row_customs_D8(i:int, j:int, k:int, BCs, valid:bool, ny:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the bottom row
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first row
	if(i == ny-1):
		# Checking all the different cases: firs, last cols and the middle
		if(k == 5 or k == 6 or k == 7):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_bottom_row_customs_D8_flat(i:int, k:int, BCs, valid:bool, nx:int, ny:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the bottom row
	Caution: this is optimised for neighbouring checks and ignores the top and bottom rows
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''
	# Only checking if it actually is on the first row
	if(i >= nx*ny - nx):
		# Checking all the different cases: firs, last cols and the middle
		if(k == 5 or k == 6 or k == 7):
			valid = False
	# Done
	return valid

@nb.njit()
def _cast_neighbour_customs_D4(i:int, j:int, k:int, valid:bool, BCs):
	'''
	Internal function that cast the neighbours to the right values in the case of normal boundary conditions
	Caution: this is optimised for neighbouring checks and should not be used on its own
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
		- valid: a boolean from previous checks 
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''

	# Preformat the output
	ir,jr = -1,-1

	# if the neighbouring operation is still valid after that:
	if(valid):
		if(k == 0):
			ir,jr = i-1, j
		if(k == 1):
			ir,jr = i, j-1
		if(k == 2):
			ir,jr = i, j+1
		if(k == 3):
			ir,jr = i+1, j

	if(BCs[i,j] == 0 or ir == -1):
		ir,jr = -1,-1
	elif(BCs[ir,jr] == 0):
		ir,jr = -1,-1
		

	return ir, jr

@nb.njit()
def _cast_neighbour_customs_D4_flat(i:int, k:int, valid:bool, BCs, nx:int):
	'''
	Internal function that cast the neighbours to the right values in the case of normal boundary conditions
	Caution: this is optimised for neighbouring checks and should not be used on its own
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
		- valid: a boolean from previous checks 
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''

	# Preformat the output
	ir = -1

	# if the neighbouring operation is still valid after that:
	if(valid):
		if(k == 0):
			ir = np.int64(i-nx)
		if(k == 1):
			ir = np.int64(i-1)
		if(k == 2):
			ir = np.int64(i+1)
		if(k == 3):
			ir = np.int64(i+nx)

	if(BCs[i] == 0 or ir == -1):
		ir = np.int64(-1)
	elif(BCs[ir] == 0):
		ir = np.int64(-1)

	return ir


@nb.njit()
def _cast_neighbour_customs_D8(i:int, j:int, k:int, valid:bool, BCs):
	'''
	Internal function that cast the neighbours to the right values in the case of normal boundary conditions
	Caution: this is optimised for neighbouring checks and should not be used on its own
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
		- valid: a boolean from previous checks 
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''

	# Preformat the output
	ir,jr = -1,-1

	# if the neighbouring operation is still valid after that:
	if(valid):
		if(k == 0):
			ir,jr = i-1, j-1
		elif(k == 1):
			ir,jr = i-1, j
		elif(k == 2):
			ir,jr = i-1, j+1
		elif(k == 3):
			ir,jr = i, j-1
		elif(k == 4):
			ir,jr = i, j+1
		elif(k == 5):
			ir,jr = i+1, j-1
		elif(k == 6):
			ir,jr = i+1, j
		elif(k == 7):
			ir,jr = i+1, j+1

	if(BCs[i,j] == 0 or ir == -1):
		ir,jr = -1,-1
	elif(BCs[ir,jr] == 0):
		ir,jr = -1,-1
		
	return ir, jr


@nb.njit()
def _cast_neighbour_customs_D8_flat(i:int, k:int, valid:bool, BCs, nx:int):
	'''
	Internal function that cast the neighbours to the right values in the case of normal boundary conditions
	Caution: this is optimised for neighbouring checks and should not be used on its own
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
		- valid: a boolean from previous checks 
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 02/05/2024)
	'''

	# Preformat the output
	ir:nb.int64 = -1
	# if the neighbouring operation is still valid after that:
	if(valid):
		if(k == 0):
			ir = np.int64(i-nx-1)
		elif(k == 1):
			ir = np.int64(i-nx)
		elif(k == 2):
			ir = np.int64(i-nx+1)
		elif(k == 3):
			ir = np.int64(i-1)
		elif(k == 4):
			ir = np.int64(i+1)
		elif(k == 5):
			ir = np.int64(i+nx-1)
		elif(k == 6):
			ir = np.int64(i+nx)
		elif(k == 7):
			ir = np.int64(i+nx+1)

	if(BCs[i] == 0 or ir == -1):
		ir = np.int64(-1)
	elif(BCs[ir] == 0):
		ir = np.int64(-1)
		
	return ir

@nb.njit()
def neighbours_D4(i:int, j:int, k:int, BCs, nx:int, ny:int):
	'''
	GPU function returning the neighbours of a given pixel
	Arguments:\
		- i,j are the row and col indices
		- k is the nth neighbour (4 in D4) following riverdale's convention (see top of this module)
		- BCs: boundary conditions code. Note that for this function it does not do anything and won't be used but it complies to the standard
	Returns:
		- (-1,-1) if hte neighbour is not valid (e.g. normal boundaries at the left border has no left neighbour k=1)
		- the indices of the row/col of the neighbours
	Authors:
		- B.G. (last modification 02/05/2024)
	TODO:
		- adding the Periodic boundary management in the checks
	'''

	# I first assume this mneighbour is valid
	valid = True

	# Breaking down the checks
	valid = _check_top_row_customs_D4(i,j,k,BCs,valid)
	valid = _check_leftest_col_customs_D4(i,j,k,BCs,valid)
	valid = _check_rightest_col_customs_D4(i,j,k,BCs,valid,nx)
	valid = _check_bottom_row_customs_D4(i,j,k,BCs,valid,ny)

	# getting the actual neighbours
	return _cast_neighbour_customs_D4(i,j,k,valid,BCs)

@nb.njit()
def neighbours_D4_flat(i:int, k:int, BCs, nx:int, ny:int):
	'''
	GPU function returning the neighbours of a given pixel
	Arguments:\
		- i,j are the row and col indices
		- k is the nth neighbour (4 in D4) following riverdale's convention (see top of this module)
		- BCs: boundary conditions code. Note that for this function it does not do anything and won't be used but it complies to the standard
	Returns:
		- (-1,-1) if hte neighbour is not valid (e.g. normal boundaries at the left border has no left neighbour k=1)
		- the indices of the row/col of the neighbours
	Authors:
		- B.G. (last modification 02/05/2024)
	TODO:
		- adding the Periodic boundary management in the checks
	'''

	# I first assume this mneighbour is valid
	valid = True

	# Breaking down the checks
	valid = _check_top_row_customs_D4_flat(i,k,BCs,valid,nx)
	valid = _check_leftest_col_customs_D4_flat(i,k,BCs,valid,nx)
	valid = _check_rightest_col_customs_D4_flat(i,k,BCs,valid,nx)
	valid = _check_bottom_row_customs_D4_flat(i,k,BCs,valid,nx,ny)

	# getting the actual neighbours
	return _cast_neighbour_customs_D4_flat(i,k,valid,BCs,nx)


@nb.njit()
def neighbours_D8(i:int, j:int, k:int, BCs, nx:int, ny:int):
	'''
	GPU function returning the neighbours of a given pixel
	Arguments:\
		- i,j are the row and col indices
		- k is the nth neighbour (4 in D8) following riverdale's convention (see top of this module)
		- BCs: boundary conditions code. Note that for this function it does not do anything and won't be used but it complies to the standard
	Returns:
		- (-1,-1) if hte neighbour is not valid (e.g. normal boundaries at the left border has no left neighbour k=1)
		- the indices of the row/col of the neighbours
	Authors:
		- B.G. (last modification 02/05/2024)
	TODO:
		- adding the Periodic boundary management in the checks
	'''

	# I first assume this mneighbour is valid
	valid = True

	# Breaking down the checks
	valid = _check_top_row_customs_D8(i,j,k,BCs,valid)
	valid = _check_leftest_col_customs_D8(i,j,k,BCs,valid)
	valid = _check_rightest_col_customs_D8(i,j,k,BCs,valid,nx)
	valid = _check_bottom_row_customs_D8(i,j,k,BCs,valid,ny)

	# getting the actual neighbours
	return _cast_neighbour_customs_D8(i,j,k,valid,BCs)

@nb.njit()
def neighbours_D8_flat(i:int, k:int, BCs, nx:int, ny:int):
	'''
	GPU function returning the neighbours of a given pixel
	Arguments:\
		- i,j are the row and col indices
		- k is the nth neighbour (4 in D8) following riverdale's convention (see top of this module)
		- BCs: boundary conditions code. Note that for this function it does not do anything and won't be used but it complies to the standard
	Returns:
		- (-1,-1) if hte neighbour is not valid (e.g. normal boundaries at the left border has no left neighbour k=1)
		- the indices of the row/col of the neighbours
	Authors:
		- B.G. (last modification 02/05/2024)
	TODO:
		- adding the Periodic boundary management in the checks
	'''

	# I first assume this mneighbour is valid
	valid = True

	# Breaking down the checks
	valid = _check_top_row_customs_D8_flat(i,k,BCs,valid,nx)
	valid = _check_leftest_col_customs_D8_flat(i,k,BCs,valid,nx)
	valid = _check_rightest_col_customs_D8_flat(i,k,BCs,valid,nx)
	valid = _check_bottom_row_customs_D8_flat(i,k,BCs,valid,nx,ny)

	# getting the actual neighbours
	return _cast_neighbour_customs_D8_flat(i,k,valid,BCs,nx)






@nb.njit()
def can_receive(i:int, j:int, BCs):
	'''
		Standard complying function for the normal boundaries
		Arguments:
			- i: the row index
			- j: the column index
			- BCs: a dummy field to keep the standard consistent
		Returns:
			- True, all the nodes can receive in the normal boundary conditions
	'''
	valid = True
	if(BCs[i,j] == 6 or BCs[i,j] == 7 or BCs[i,j] == 8 or BCs[i,j] == 0):
		valid = False
	return valid

@nb.njit()
def can_give(i:int, j:int, BCs):
	'''
		Standard complying function for the normal boundaries
		Arguments:
			- i: the row index
			- j: the column index
			- BCs: a dummy field to keep the standard consistent
		Returns:
			- True, all the nodes can receive in the normal boundary conditions
		Authors:
		- B.G. (last modification 02/05/2024)
	'''
	valid = False
	if(BCs[i,j] == 1 or BCs[i,j] == 6 or BCs[i,j] == 7 or BCs[i,j] == 8 or BCs[i,j] == 9):
		valid = True
	return valid


@nb.njit()
def can_out(i:int, j:int, BCs):
	'''
		Standard complying function for the normal boundaries
		Arguments:
			- i: the row index
			- j: the column index
			- BCs: a dummy field to keep the standard consistent
		Returns:
			- True, all the nodes can receive in the normal boundary conditions
		Authors:
		- B.G. (last modification 02/05/2024)
	'''
	valid = False
	if(BCs[i,j] == 3 or BCs[i,j] == 4 or BCs[i,j] == 5):
		valid = True
	return valid


@nb.njit()
def can_receive_flat(i:int, BCs):
	'''
		Standard complying function for the normal boundaries
		Arguments:
			- i: the row index
			- j: the column index
			- BCs: a dummy field to keep the standard consistent
		Returns:
			- True, all the nodes can receive in the normal boundary conditions
	'''
	valid = True
	if(BCs[i] == 6 or BCs[i] == 7 or BCs[i] == 8 or BCs[i] == 0):
		valid = False
	return valid

@nb.njit()
def can_give_flat(i:int, BCs):
	'''
		Standard complying function for the normal boundaries
		Arguments:
			- i: the row index
			- j: the column index
			- BCs: a dummy field to keep the standard consistent
		Returns:
			- True, all the nodes can receive in the normal boundary conditions
		Authors:
		- B.G. (last modification 02/05/2024)
	'''
	valid = False
	if(BCs[i] == 1 or BCs[i] == 6 or BCs[i] == 7 or BCs[i] == 8 or BCs[i] == 9):
		valid = True
	return valid


@nb.njit()
def can_out_flat(i:int, BCs):
	'''
		Standard complying function for the normal boundaries
		Arguments:
			- i: the row index
			- j: the column index
			- BCs: a dummy field to keep the standard consistent
		Returns:
			- True, all the nodes can receive in the normal boundary conditions
		Authors:
		- B.G. (last modification 02/05/2024)
	'''
	valid = False
	if(BCs[i] == 3 or BCs[i] == 4 or BCs[i] == 5):
		valid = True
	return valid


@nb.njit()
def is_active_flat(i:int, BCs):
	'''
		Quick utility function determining if a node is active or not for normal boundaries
		Arguments:
			- i: the row index
			- j: the column index
			- BCs: a dummy field to keep the standard consistent
		Returns:
			- True if the node is active
			- False if inactive (i.e in this case outs)
		Authors:
		- B.G. (last modification 02/05/2024)
	'''
	valid = True
	if(BCs[i] == 0): #) or _can_out_customs(i,j,BCs) or _can_receive_customs(i,j,BCs) == False):
		valid = False
	return valid



########################################################################
########################################################################
############### GENERIC  FUNCTIONS #####################################
########################################################################
########################################################################


@nb.njit()
def oppk_D4(k):
	'''
	Returns the opposite neighbour code, e.g. if supplied with 1 (left neighbour), returns 2 (right neighbour)
	Useful to check if a k neighbour points toward a cell
	
	Arguments:
		k: the neihgbour code

	returns:
		The opposite neighbour code

	Authors:
		- B.G. (last modifications: 06/2024)
	'''
	return 3 if k == 0 else (2 if k == 1 else (1 if k == 2 else (0 if k == 3 else 5)))

@nb.njit()
def oppk_D8(k):
	'''
	Returns the opposite neighbour code, e.g. if supplied with 1 (left neighbour), returns 2 (right neighbour)
	Useful to check if a k neighbour points toward a cell
	
	Arguments:
		k: the neihgbour code

	returns:
		The opposite neighbour code

	Authors:
		- B.G. (last modifications: 06/2024)
	'''
	return 7 if k == 0 else (6 if k == 1 else (5 if k == 2 else (4 if k == 3 else (3 if k == 4 else (2 if k == 5 else (1 if k==6 else (0))))) ) )


@nb.njit()
def dx_from_k_D4(dx, k):
	'''
	Gets the distance to the neighbours
	'''
	return dx

@nb.njit()
def dx_from_k_D8(dx, k):
	'''
	Gets the distance to the neighbours
	'''
	return dx if (k == 1 or k == 3 or k == 4 or k == 6) else 1.41421356237*dx

@nb.njit()
def dy_from_k_D4(dx, k):
	'''
	Gets the distance to the neighbours
	'''
	return dx

@nb.njit()
def dy_from_k_D8(dx, k):
	'''
	Gets the distance to the neighbours
	'''
	return dx if (k == 1 or k == 3 or k == 4 or k == 6) == False else 1.41421356237*dx
