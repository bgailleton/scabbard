'''
This files contains the functions and compile-time constant to naviagate and test the grid from the GPU point of views. That includes:
- Neighbouring operations
- Management of the boundary conditions
- Grid dimensions and other geometrical characteristics


Riverdale's convention for the nth neighbouring:

D4:

|   | 0 |   |
| 1 |i,j| 2 |
|   | 3 |   |

In other words:
- k=0: top neighbour
- k=1: left neighbour
- k=2: right neighbour
- k=3: bottom neighbour

B.G - 26/04/1992 - Acigné

'''

import taichi as ti
import numpy as np
from enum import Enum
import scabbard.utils as scaut 


class BoundaryConditions(Enum):
	'''
	Enumeration of the different boundary condition types possible
	'''	
	normal = 0
	periodicEW = 1
	periodicNS = 2
	customs = 3


@scaut.singleton
class GridParams:
	'''
		Holds all the constants describing the grid dimensions and boundary conditions
		Needs to be set before compiling the taichi kernels and funcs and cannot be changed dynamically
		This part of the code might evolve later, the reason behind this structure is to force these element to be
		compile-time constants as it boosts performances (we are talking 10 - 20% speed up compared to having them as parameters of functions)
	'''
	def __init__(self):
		# Grid parameters
		self.dx = 1.
		self.dy = 1.
		self.nx = 1.
		self.ny = 1.
		self.nxy = 1.
		self.boundaries = BoundaryConditions.normal

# Class Holder
GRID = GridParams()



@ti.func
def _check_top_row_normal(i:int, j:int, k:int):
	'''
	Internal function to check if neighbouring is possible for nodes at the top row
	Arguments:
		- i: Row index
		- j: column index
		- k: neighbour number (See top of this module for explanations)
	Returns:
		- a boolean: True = neighbour is valid, False: not a neighbour
	Authors:
		- B.G. (last modification 30/04/2024)
	'''
	# I assume it's good
	valid = True
	# Only checking if it actually is on the first row
	if(i == 0):
		# Checking all the different cases: firs, last cols and the middle
		if((j == 0 and k <= 1) or (j == GRID.nx-1 and (k == 0 or k == 2)) or (k==0)):
			valid = False
	# Done
	return valid

@ti.func
def _check_leftest_col_normal(i:int, j:int, k:int):
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
		- B.G. (last modification 30/04/2024)
	'''
	# I assume it's good
	valid = True
	# Only checking if it actually is on the first col
	if(j == 0):
		if(k==1):
			valid = False
	# Done
	return valid

@ti.func
def _check_rightest_col_normal(i:int, j:int, k:int):
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
		- B.G. (last modification 30/04/2024)
	'''
	# I assume it's good
	valid = True
	# Only checking if it actually is on the first col
	if(j == GRID.nx-1):
		if(k==2):
			valid = False
	# Done
	return valid

@ti.func
def _check_bottom_row_normal(i:int, j:int, k:int):
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
		- B.G. (last modification 30/04/2024)
	'''
	# I assume it's good
	valid = True
	# Only checking if it actually is on the first row
	if(i == GRID.ny-1):
		# Checking all the different cases: firs, last cols and the middle
		if((j == 0 and (k == 1 or k == 3)) or (j == GRID.nx-1 and (k == 3 or k == 2)) or (k==3)):
			valid = False
	# Done
	return valid

@ti.func
def _cast_neighbour_normal(i:int, j:int, k:int, valid:bool):
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
		- B.G. (last modification 30/04/2024)
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

	return ir, jr

@ti.func
def _neighbours_normal(i:int, j:int, k:int, BCs:ti.template()):
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
		- B.G. (last modification on the 29th of April)
	'''

	# I first assume this mneighbour is valid
	valid = True

	# Breaking down the checks
	valid = _check_top_row_normal(i,j,k)
	valid = _check_leftest_col_normal(i,j,k)
	valid = _check_rightest_col_normal(i,j,k)
	valid = _check_bottom_row_normal(i,j,k)

	# getting the actual neighbours
	return _cast_neighbour_normal(i,j,k,valid)

@ti.func
def _is_active_normal(i:int, j:int):
	'''
		Quick utility function determining if a node is active or not for normal boundaries
	'''
	valid = True
	if(i==0 or j == 0 or i == GRID.ny - 1 or j == GRID.nx - 1):
		valid = False
	return valid






########################################################################
########################################################################
############### API ####################################################
########################################################################
########################################################################

# Exporting the generic API
neighbours = None
is_active = None

def set_grid_CC():
	'''
		Python side to be called before compilation and set the generic functions matching the grid criterion and boundary conditions
	'''
	# fectch neighbours placeholder
	global neighbours
	global is_active
	
	# Feed it
	if(GRID.boundaries == BoundaryConditions.normal):
		neighbours = _neighbours_normal
		is_active = _is_active_normal
	else:
		raise NotImplementedError('BC Not implemented yet')
