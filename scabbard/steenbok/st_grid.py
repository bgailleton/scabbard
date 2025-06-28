'''
This module provides Numba-optimized functions for grid navigation and boundary condition checks
within the Steenbok (Numba engine) mirror of Riverdale's conventions.

Author: B.G. (last modification: 07/2024 - Acigné)
'''

import numba as nb
import numpy as np
from enum import Enum
import scabbard.utils as scaut 


#################################################################################################
############################## Customs Boundaries ###############################################
#################################################################################################

'''
Reminder, I am using the DAGGER convention for boundary conditions:

// Cannot flow at all = nodata
NO_FLOW = 0,

// Internal Node (can flow in every directions)
FLOW = 1,

// Internal Node (can flow in every directions) BUT neighbors a special flow
// condition and may need specific care
FLOW_BUT = 2,

// flow can out there but can also flow to downstream neighbors
CAN_OUT = 3,

// flow can only out from this cell
OUT = 4,

// Not only flow HAS to out there: neighboring flows will be drained there no
// matter what
FORCE_OUT = 5,

// Flows through the cell is possible, but the cell CANNOT out fluxes from
// this boundary (reserved to model edges, internal boundaries wont give to
// nodata anyway)
CANNOT_OUT = 6,

// Flow can only flow to potential receivers
IN = 7,

// Forced INFLOW: flow will flow to all neighbors (except other FORCE_IN)
FORCE_IN = 8,

// periodic border
PERIODIC_BORDER = 9
'''

@nb.njit()
def _check_top_row_customs_D4(i:int, j:int, k:int, BCs, valid:bool):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the top row
	under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-3, see module header for convention).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		valid (bool): Current validity status of the neighbor.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the first row
	if(i == 0):
		if(k == 0):
			valid = False
	return valid

@nb.njit()
def _check_top_row_customs_D4_flat(i:int, k:int, BCs, valid:bool, nx:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the top row
	(flattened index) under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-3, see module header for convention).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the first row (flat index < nx)
	if(i < nx):
		if(k == 0):
			valid = False
	return valid

@nb.njit()
def _check_top_row_customs_D8(i:int, j:int, k:int, BCs, valid:bool):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the top row
	under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-7, see module header for convention).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		valid (bool): Current validity status of the neighbor.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the first row
	if(i == 0):
		if(k == 0 or k ==1 or k==2):
			valid = False
	return valid

@nb.njit()
def _check_top_row_customs_D8_flat(i:int, k:int, BCs, valid:bool, nx:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the top row
	(flattened index) under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-7, see module header for convention).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the first row (flat index < nx)
	if(i < nx):
		if(k == 0 or k ==1 or k==2):
			valid = False
	return valid


@nb.njit()
def _check_leftest_col_customs_D4(i:int, j:int, k:int, BCs, valid:bool):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the leftmost column
	under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-3, see module header for convention).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		valid (bool): Current validity status of the neighbor.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the first column
	if(j == 0):
		if(k==1):
			valid = False
	return valid

@nb.njit()
def _check_leftest_col_customs_D4_flat(i:int, k:int, BCs, valid:bool, nx:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the leftmost column
	(flattened index) under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-3, see module header for convention).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the first column (flat index % nx == 0)
	if( (i%nx) == 0):
		if(k==1):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_leftest_col_customs_D8(i:int, j:int, k:int, BCs, valid:bool):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the leftmost column
	under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-7, see module header for convention).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		valid (bool): Current validity status of the neighbor.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the first column
	if(j == 0):
		if(k==0 or k == 3 or k == 5):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_leftest_col_customs_D8_flat(i:int, k:int, BCs, valid:bool, nx:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the leftmost column
	(flattened index) under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-7, see module header for convention).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the first column (flat index % nx == 0)
	if((i % nx) == 0):
		if(k==0 or k == 3 or k == 5):
			valid = False
	# Done
	return valid


@nb.njit()
def _check_rightest_col_customs_D4(i:int, j:int, k:int, BCs, valid:bool, nx:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the rightmost column
	under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-3, see module header for convention).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the last column
	if(j == nx-1):
		if(k==2):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_rightest_col_customs_D4_flat(i:int, k:int, BCs, valid:bool, nx:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the rightmost column
	(flattened index) under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-3, see module header for convention).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the last column (flat index % nx == nx-1)
	if(i%nx == nx-1):
		if(k==2):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_rightest_col_customs_D8(i:int, j:int, k:int, BCs, valid:bool, nx:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the rightmost column
	under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-7, see module header for convention).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the last column
	if(j == nx-1):
		if(k==2 or k == 4 or k == 7):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_rightest_col_customs_D8_flat(i:int, k:int, BCs, valid:bool, nx:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the rightmost column
	(flattened index) under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-7, see module header for convention).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the last column (flat index % nx == nx-1)
	if(i%nx == nx-1):
		if(k==2 or k == 4 or k == 7):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_bottom_row_customs_D4(i:int, j:int, k:int, BCs, valid:bool, ny:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the bottom row
	under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-3, see module header for convention).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		valid (bool): Current validity status of the neighbor.
		ny (int): Number of rows in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the last row
	if(i == ny-1):
		# Checking all the different cases: first, last cols and the middle
		if(k == 3):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_bottom_row_customs_D4_flat(i:int, k:int, BCs, valid:bool, nx:int, ny:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the bottom row
	(flattened index) under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-3, see module header for convention).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the last row (flat index >= nx*ny - nx)
	if(i >= nx*ny - nx):
		# Checking all the different cases: first, last cols and the middle
		if(k == 3):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_bottom_row_customs_D8(i:int, j:int, k:int, BCs, valid:bool, ny:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the bottom row
	under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-7, see module header for convention).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		valid (bool): Current validity status of the neighbor.
		ny (int): Number of rows in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the last row
	if(i == ny-1):
		# Checking all the different cases: first, last cols and the middle
		if(k == 5 or k == 6 or k == 7):
			valid = False
	# Done
	return valid

@nb.njit()
def _check_bottom_row_customs_D8_flat(i:int, k:int, BCs, valid:bool, nx:int, ny:int):
	"""
	Internal Numba function to check if neighboring is possible for nodes in the bottom row
	(flattened index) under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-7, see module header for convention).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		valid (bool): Current validity status of the neighbor.
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	# Only checking if it actually is on the last row (flat index >= nx*ny - nx)
	if(i >= nx*ny - nx):
		# Checking all the different cases: first, last cols and the middle
		if(k == 5 or k == 6 or k == 7):
			valid = False
	# Done
	return valid

@nb.njit()
def _cast_neighbour_customs_D4(i:int, j:int, k:int, valid:bool, BCs):
	"""
	Internal Numba function that casts neighbor coordinates to the correct values
	for custom boundary conditions with D4 connectivity.

	This function is optimized for neighboring checks and should not be used on its own.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-3, see module header for convention).
		valid (bool): Validity status of the neighbor from previous checks.
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.

	Returns:
		tuple[int, int]: A tuple (ir, jr) representing the row and column indices of the neighbor.
					 If the neighbor is not valid, returns (-1, -1).

	Author: B.G. (last modification 02/05/2024)
	"""

	# Preformat the output to invalid coordinates
	ir,jr = -1,-1

	# If the neighboring operation is still valid after previous checks
	if(valid):
		if(k == 0):
			ir,jr = i-1, j
		if(k == 1):
			ir,jr = i, j-1
		if(k == 2):
			ir,jr = i, j+1
		if(k == 3):
			ir,jr = i+1, j

	# Further checks based on boundary conditions (BCs)
	if(BCs[i,j] == 0 or ir == -1):
		ir,jr = -1,-1
	elif(BCs[ir,jr] == 0):
		ir,jr = -1,-1
		

	return ir, jr

@nb.njit()
def _cast_neighbour_customs_D4_flat(i:int, k:int, valid:bool, BCs, nx:int):
	"""
	Internal Numba function that casts flattened neighbor index to the correct value
	for custom boundary conditions with D4 connectivity.

	This function is optimized for neighboring checks and should not be used on its own.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-3, see module header for convention).
		valid (bool): Validity status of the neighbor from previous checks.
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		nx (int): Number of columns in the grid.

	Returns:
		int: The flat index of the neighbor. If the neighbor is not valid, returns -1.

	Author: B.G. (last modification 02/05/2024)
	"""

	# Preformat the output to invalid index
	ir = -1

	# If the neighboring operation is still valid after previous checks
	if(valid):
		if(k == 0):
			ir = np.int64(i-nx)
		if(k == 1):
			ir = np.int64(i-1)
		if(k == 2):
			ir = np.int64(i+1)
		if(k == 3):
			ir = np.int64(i+nx)

	# Further checks based on boundary conditions (BCs)
	if(BCs[i] == 0 or ir == -1):
		ir = np.int64(-1)
	elif(BCs[ir] == 0):
		ir = np.int64(-1)

	return ir


@nb.njit()
def _cast_neighbour_customs_D8(i:int, j:int, k:int, valid:bool, BCs):
	"""
	Internal Numba function that casts neighbor coordinates to the correct values
	for custom boundary conditions with D8 connectivity.

	This function is optimized for neighboring checks and should not be used on its own.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-7, see module header for convention).
		valid (bool): Validity status of the neighbor from previous checks.
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.

	Returns:
		tuple[int, int]: A tuple (ir, jr) representing the row and column indices of the neighbor.
					 If the neighbor is not valid, returns (-1, -1).

	Author: B.G. (last modification 02/05/2024)
	"""

	# Preformat the output to invalid coordinates
	ir,jr = -1,-1

	# If the neighboring operation is still valid after previous checks
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

	# Further checks based on boundary conditions (BCs)
	if(BCs[i,j] == 0 or ir == -1):
		ir,jr = -1,-1
	elif(BCs[ir,jr] == 0):
		ir,jr = -1,-1
		
	return ir, jr


@nb.njit()
def _cast_neighbour_customs_D8_flat(i:int, k:int, valid:bool, BCs, nx:int):
	"""
	Internal Numba function that casts flattened neighbor index to the correct value
	for custom boundary conditions with D8 connectivity.

	This function is optimized for neighboring checks and should not be used on its own.

	Args:
		i (int): Flat index of the current node.
		k (int): Neighbor number (0-7, see module header for convention).
		valid (bool): Validity status of the neighbor from previous checks.
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		nx (int): Number of columns in the grid.

	Returns:
		int: The flat index of the neighbor. If the neighbor is not valid, returns -1.

	Author: B.G. (last modification 02/05/2024)
	"""

	# Preformat the output to invalid index
	ir:nb.int64 = -1
	# If the neighboring operation is still valid after previous checks
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

	# Further checks based on boundary conditions (BCs)
	if(BCs[i] == 0 or ir == -1):
		ir = np.int64(-1)
	elif(BCs[ir] == 0):
		ir = np.int64(-1)
		
	return ir

@nb.njit()
def neighbours_D4(i:int, j:int, k:int, BCs, nx:int, ny:int):
	"""
	Numba function returning the neighbors of a given pixel (2D coordinates)
	under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Row index of the current pixel.
		j (int): Column index of the current pixel.
		k (int): The nth neighbor (0-3 for D4) following Riverdale's convention (see module header).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.

	Returns:
		tuple[int, int]: A tuple (ir, jr) representing the row and column indices of the neighbor.
					 Returns (-1, -1) if the neighbor is not valid.

	Author: B.G. (last modification 02/05/2024)
	TODO:
		- Add periodic boundary management in the checks.
	"""

	# Assume the neighbor is valid initially
	valid = True

	# Perform boundary checks sequentially
	valid = _check_top_row_customs_D4(i,j,k,BCs,valid)
	valid = _check_leftest_col_customs_D4(i,j,k,BCs,valid)
	valid = _check_rightest_col_customs_D4(i,j,k,BCs,valid,nx)
	valid = _check_bottom_row_customs_D4(i,j,k,BCs,valid,ny)

	# Get the actual neighbor coordinates based on validity and custom BCs
	return _cast_neighbour_customs_D4(i,j,k,valid,BCs)

@nb.njit()
def neighbours_D4_flat(i:int, k:int, BCs, nx:int, ny:int):
	"""
	Numba function returning the flattened index of the neighbor of a given pixel
	under custom boundary conditions for D4 connectivity.

	Args:
		i (int): Flat index of the current pixel.
		k (int): The nth neighbor (0-3 for D4) following Riverdale's convention (see module header).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.

	Returns:
		int: The flat index of the neighbor. Returns -1 if the neighbor is not valid.

	Author: B.G. (last modification 02/05/2024)
	TODO:
		- Add periodic boundary management in the checks.
	"""

	# Assume the neighbor is valid initially
	valid = True

	# Perform boundary checks sequentially
	valid = _check_top_row_customs_D4_flat(i,k,BCs,valid,nx)
	valid = _check_leftest_col_customs_D4_flat(i,k,BCs,valid,nx)
	valid = _check_rightest_col_customs_D4_flat(i,k,BCs,valid,nx)
	valid = _check_bottom_row_customs_D4_flat(i,k,BCs,valid,nx,ny)

	# Get the actual neighbor flattened index based on validity and custom BCs
	return _cast_neighbour_customs_D4_flat(i,k,valid,BCs,nx)


@nb.njit()
def neighbours_D8(i:int, j:int, k:int, BCs, nx:int, ny:int):
	"""
	Numba function returning the neighbors of a given pixel (2D coordinates)
	under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Row index of the current pixel.
		j (int): Column index of the current pixel.
		k (int): The nth neighbor (0-7 for D8) following Riverdale's convention (see module header).
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.

	Returns:
		tuple[int, int]: A tuple (ir, jr) representing the row and column indices of the neighbor.
					 Returns (-1, -1) if the neighbor is not valid.

	Author: B.G. (last modification 02/05/2024)
	TODO:
		- Add periodic boundary management in the checks.
	"""

	# Assume the neighbor is valid initially
	valid = True

	# Perform boundary checks sequentially
	valid = _check_top_row_customs_D8(i,j,k,BCs,valid)
	valid = _check_leftest_col_customs_D8(i,j,k,BCs,valid)
	valid = _check_rightest_col_customs_D8(i,j,k,BCs,valid,nx)
	valid = _check_bottom_row_customs_D8(i,j,k,BCs,valid,ny)

	# Get the actual neighbor coordinates based on validity and custom BCs
	return _cast_neighbour_customs_D8(i,j,k,valid,BCs)

@nb.njit()
def neighbours_D8_flat(i:int, k:int, BCs, nx:int, ny:int):
	"""
	Numba function returning the flattened index of the neighbor of a given pixel
	under custom boundary conditions for D8 connectivity.

	Args:
		i (int): Flat index of the current pixel.
		k (int): The nth neighbor (0-7 for D8) following Riverdale's convention (see module header).
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).
		nx (int): Number of columns in the grid.
		ny (int): Number of rows in the grid.

	Returns:
		int: The flat index of the neighbor. Returns -1 if the neighbor is not valid.

	Author: B.G. (last modification 02/05/2024)
	TODO:
		- Add periodic boundary management in the checks.
	"""

	# Assume the neighbor is valid initially
	valid = True

	# Perform boundary checks sequentially
	valid = _check_top_row_customs_D8_flat(i,k,BCs,valid,nx)
	valid = _check_leftest_col_customs_D8_flat(i,k,BCs,valid,nx)
	valid = _check_rightest_col_customs_D8_flat(i,k,BCs,valid,nx)
	valid = _check_bottom_row_customs_D8_flat(i,k,BCs,valid,nx,ny)

	# Get the actual neighbor flattened index based on validity and custom BCs
	return _cast_neighbour_customs_D8_flat(i,k,valid,BCs,nx)






@nb.njit()
def can_receive(i:int, j:int, BCs):
	"""
	Numba function to check if a node (2D coordinates) can receive flow
	under custom boundary conditions.

	Args:
		i (int): Row index.
		j (int): Column index.
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.

	Returns:
		bool: True if the node can receive flow, False otherwise.

	Author: B.G.
	"""
	valid = True
	# A node cannot receive if it's NO_FLOW, IN, FORCE_IN, or CANNOT_OUT
	if(BCs[i,j] == 6 or BCs[i,j] == 7 or BCs[i,j] == 8 or BCs[i,j] == 0):
		valid = False
	return valid

@nb.njit()
def can_give(i:int, j:int, BCs):
	"""
	Numba function to check if a node (2D coordinates) can give flow
	under custom boundary conditions.

	Args:
		i (int): Row index.
		j (int): Column index.
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.

	Returns:
		bool: True if the node can give flow, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	valid = False
	# A node can give flow if it's FLOW, CANNOT_OUT, IN, FORCE_IN, or PERIODIC_BORDER
	if(BCs[i,j] == 1 or BCs[i,j] == 6 or BCs[i,j] == 7 or BCs[i,j] == 8 or BCs[i,j] == 9):
		valid = True
	return valid


@nb.njit()
def can_out(i:int, j:int, BCs):
	"""
	Numba function to check if flow can exit from a node (2D coordinates)
	under custom boundary conditions.

	Args:
		i (int): Row index.
		j (int): Column index.
		BCs (numpy.ndarray): 2D NumPy array of boundary condition codes.

	Returns:
		bool: True if flow can exit from the node, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	valid = False
	# Flow can exit if the node is CAN_OUT, OUT, or FORCE_OUT
	if(BCs[i,j] == 3 or BCs[i,j] == 4 or BCs[i,j] == 5):
		valid = True
	return valid


@nb.njit()
def can_receive_flat(i:int, BCs):
	"""
	Numba function to check if a node (flattened index) can receive flow
	under custom boundary conditions.

	Args:
		i (int): Flat index of the node.
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).

	Returns:
		bool: True if the node can receive flow, False otherwise.

	Author: B.G.
	"""
	valid = True
	# A node cannot receive if it's NO_FLOW, IN, FORCE_IN, or CANNOT_OUT
	if(BCs[i] == 6 or BCs[i] == 7 or BCs[i] == 8 or BCs[i] == 0):
		valid = False
	return valid

@nb.njit()
def can_give_flat(i:int, BCs):
	"""
	Numba function to check if a node (flattened index) can give flow
	under custom boundary conditions.

	Args:
		i (int): Flat index of the node.
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).

	Returns:
		bool: True if the node can give flow, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	valid = False
	# A node can give flow if it's FLOW, CANNOT_OUT, IN, FORCE_IN, or PERIODIC_BORDER
	if(BCs[i] == 1 or BCs[i] == 6 or BCs[i] == 7 or BCs[i] == 8 or BCs[i] == 9):
		valid = True
	return valid


@nb.njit()
def can_out_flat(i:int, BCs):
	"""
	Numba function to check if flow can exit from a node (flattened index)
	under custom boundary conditions.

	Args:
		i (int): Flat index of the node.
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).

	Returns:
		bool: True if flow can exit from the node, False otherwise.

	Author: B.G. (last modification 02/05/2024)
	"""
	valid = False
	# Flow can exit if the node is CAN_OUT, OUT, or FORCE_OUT
	if(BCs[i] == 3 or BCs[i] == 4 or BCs[i] == 5):
		valid = True
	return valid


@nb.njit()
def is_active_flat(i:int, BCs):
	"""
	Numba function determining if a node (flattened index) is active
	under custom boundary conditions.

	Args:
		i (int): Flat index of the node.
		BCs (numpy.ndarray): 1D NumPy array of boundary condition codes (flattened).

	Returns:
		bool: True if the node is active, False if inactive (NO_FLOW).

	Author: B.G. (last modification 02/05/2024)
	"""
	valid = True
	if(BCs[i] == 0): # If current cell is NO_FLOW
		valid = False
	return valid



########################################################################
########################################################################
############### GENERIC  FUNCTIONS #####################################
########################################################################
########################################################################


@nb.njit()
def oppk_D4(k):
	"""
	Returns the opposite neighbor code for D4 connectivity.

	For example, if given 1 (left neighbor), it returns 2 (right neighbor).
	Useful for checking if a neighbor `k` points towards a cell.

	Args:
		k (int): The neighbor code (0-3 for D4, 5 for no flow).

	Returns:
		int: The opposite neighbor code.

	Author: B.G. (last modifications: 06/2024)
	"""
	return 3 if k == 0 else (2 if k == 1 else (1 if k == 2 else (0 if k == 3 else 5)))

@nb.njit()
def oppk_D8(k):
	"""
	Returns the opposite neighbor code for D8 connectivity.

	For example, if given 1 (top neighbor), it returns 6 (bottom neighbor).
	Useful for checking if a neighbor `k` points towards a cell.

	Args:
		k (int): The neighbor code (0-7 for D8).

	Returns:
		int: The opposite neighbor code.

	Author: B.G. (last modifications: 06/2024)
	"""
	return 7 if k == 0 else (6 if k == 1 else (5 if k == 2 else (4 if k == 3 else (3 if k == 4 else (2 if k == 5 else (1 if k==6 else (0))))) ) )


@nb.njit()
def dx_from_k_D4(dx, k):
	"""
	Returns the spatial distance in the x-direction to a D4 neighbor.

	Args:
		dx (float): The grid cell size in the x-direction.
		k (int): The neighbor code (0-3 for D4).

	Returns:
		float: The distance `dx`.

	Author: B.G.
	"""
	return dx

@nb.njit()
def dx_from_k_D8(dx, k):
	"""
	Returns the spatial distance to a D8 neighbor.

	Args:
		dx (float): The grid cell size.
		k (int): The neighbor code (0-7 for D8).

	Returns:
		float: The distance `dx` for cardinal neighbors, or `sqrt(2)*dx` for diagonal neighbors.

	Author: B.G.
	"""
	return dx if (k == 1 or k == 3 or k == 4 or k == 6) else 1.41421356237*dx

@nb.njit()
def dy_from_k_D4(dx, k):
	"""
	Returns the spatial distance in the y-direction to a D4 neighbor.

	Args:
		dx (float): The grid cell size in the y-direction.
		k (int): The neighbor code (0-3 for D4).

	Returns:
		float: The distance `dx`.

	Author: B.G.
	"""
	return dx

@nb.njit()
def dy_from_k_D8(dx, k):
	"""
	Returns the spatial distance in the y-direction to a D8 neighbor.

	Args:
		dx (float): The grid cell size.
		k (int): The neighbor code (0-7 for D8).

	Returns:
		float: The distance `dx` for cardinal neighbors, or `sqrt(2)*dx` for diagonal neighbors.

	Author: B.G.
	"""
	return dx if (k == 1 or k == 3 or k == 4 or k == 6) == False else 1.41421356237*dx
