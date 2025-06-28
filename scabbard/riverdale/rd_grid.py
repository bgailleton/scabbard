'''
This module defines functions and compile-time constants for navigating and testing the grid from a GPU perspective.
It includes functionalities for:
- Neighboring operations
- Management of boundary conditions
- Grid dimensions and other geometrical characteristics.

Riverdale's convention for D4 (4-directional) neighboring:

|   | 0 |   |
| 1 |i,j| 2 |
|   | 3 |   |

In other words:
- k=0: top neighbor
- k=1: left neighbor
- k=2: right neighbor
- k=3: bottom neighbor

Author: B.G. (last modification: 26/04/1992 - Acigné)
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

class BoundaryConditionsSlope(Enum):
	'''
	Enumeration of the different boundary condition types possible
	'''	
	fixed_elevation = 0
	fixed_slope = 1



@scaut.singleton
class GridParams:
	"""
	Holds all the constants describing the grid dimensions and boundary conditions.

	These parameters need to be set before compiling the Taichi kernels and functions
	 and cannot be changed dynamically. This structure ensures that these elements
	 become compile-time constants, which significantly boosts performance (10-20% speedup)
	 compared to passing them as function parameters.

	Attributes:
		dx (float): Grid cell size in the x-direction.
		dy (float): Grid cell size in the y-direction.
		nx (int): Number of grid cells in the x-direction.
		ny (int): Number of grid cells in the y-direction.
		nxy (int): Total number of grid cells (nx * ny).
		boundaries (BoundaryConditions): Type of boundary conditions.

	Author: B.G.
	"""
	def __init__(self):
		# Grid parameters
		self.dx = 1.
		self.dy = 1.
		self.nx = 1.
		self.ny = 1.
		self.nxy = 1.
		self.boundaries = BoundaryConditions.normal
		# self.force_permissive_boundaries = False # this parameter forces all the "out" boundary codes to potentially redirect_flow if a neighbour is 





# Class Holder
GRID = GridParams()

def _manual_set_GridParams_custom(dx,dy,nx,ny):
	"""
	Manually sets custom grid parameters.

	Args:
		dx (float): Grid cell size in the x-direction.
		dy (float): Grid cell size in the y-direction.
		nx (int): Number of grid cells in the x-direction.
		ny (int): Number of grid cells in the y-direction.

	Author: B.G.
	"""
	GRID.dx = dx
	GRID.dy = dy
	GRID.nx = nx
	GRID.ny = ny
	GRID.nxy = nx*ny
	GRID.boundaries = BoundaryConditions.customs



#################################################################################################
############################## Normal Boundaries ################################################
#################################################################################################

# These boundaries represents the classic DEM boundaries where all the nodes are valid but flow can escape from the boundaries



@ti.func
def _check_top_row_normal(i:int, j:int, k:int, valid:bool):
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
	
	# Only checking if it actually is on the first row
	if(i == 0):
		# Checking all the different cases: firs, last cols and the middle
		if((j == 0 and k <= 1) or (j == GRID.nx-1 and (k == 0 or k == 2)) or (k==0)):
			valid = False
	# Done
	return valid

	@ti.func
def _check_leftest_col_normal(i:int, j:int, k:int, valid:bool):
	"""
	Internal Taichi function to check if neighboring is possible for nodes in the leftmost column
	under normal boundary conditions.

	This function is optimized for neighboring checks and ignores top and bottom rows.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-3, see module header for convention).
		valid (bool): Current validity status of the neighbor.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 30/04/2024)
	"""
	
	# Only check if the current node is in the first column
	if(j == 0):
		if(k==1):
			valid = False
	return valid

@ti.func
def _check_rightest_col_normal(i:int, j:int, k:int, valid:bool):
	"""
	Internal Taichi function to check if neighboring is possible for nodes in the rightmost column
	under normal boundary conditions.

	This function is optimized for neighboring checks and ignores top and bottom rows.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-3, see module header for convention).
		valid (bool): Current validity status of the neighbor.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 30/04/2024)
	"""
	
	# Only check if the current node is in the last column
	if(j == GRID.nx-1):
		if(k==2):
			valid = False
	return valid

@ti.func
def _check_bottom_row_normal(i:int, j:int, k:int, valid:bool):
	"""
	Internal Taichi function to check if neighboring is possible for nodes in the bottom row
	under normal boundary conditions.

	This function is optimized for neighboring checks and ignores top and bottom rows.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-3, see module header for convention).
		valid (bool): Current validity status of the neighbor.

	Returns:
		bool: True if the neighbor is valid, False otherwise.

	Author: B.G. (last modification 30/04/2024)
	"""
	
	# Only check if the current node is in the last row
	if(i == GRID.ny-1):
		# Check specific cases for corners and general bottom row
		if((j == 0 and (k == 1 or k == 3)) or (j == GRID.nx-1 and (k == 3 or k == 2)) or (k==3)):
			valid = False
	return valid

@ti.func
def _cast_neighbour_normal(i:int, j:int, k:int, valid:bool):
	"""
	Internal Taichi function that casts neighbor coordinates to the correct values
	for normal boundary conditions.

	This function is optimized for neighboring checks and should not be used on its own.

	Args:
		i (int): Row index.
		j (int): Column index.
		k (int): Neighbor number (0-3, see module header for convention).
		valid (bool): Validity status of the neighbor from previous checks.

	Returns:
		tuple[int, int]: A tuple (ir, jr) representing the row and column indices of the neighbor.
					 If the neighbor is not valid, returns (-1, -1).

	Author: B.G. (last modification 30/04/2024)
	"""

	# Initialize output to invalid coordinates
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

	return ir, jr

@ti.func
def _neighbours_normal(i:int, j:int, k:int, BCs:ti.template()):
	"""
	GPU function returning the neighbors of a given pixel under normal boundary conditions.

	Args:
		i (int): Row index of the current pixel.
		j (int): Column index of the current pixel.
		k (int): The nth neighbor (0-3 for D4) following Riverdale's convention (see module header).
		BCs (ti.template()): Boundary conditions field. (Not directly used in this function but kept for API consistency).

	Returns:
		tuple[int, int]: A tuple (ir, jr) representing the row and column indices of the neighbor.
					 Returns (-1, -1) if the neighbor is not valid (e.g., attempting to get a left neighbor
					 at the left boundary).

	Author: B.G. (last modification on the 29th of April)
	"""

	# Assume the neighbor is valid initially
	valid = True

	# Perform boundary checks sequentially
	valid = _check_top_row_normal(i,j,k,valid)
	valid = _check_leftest_col_normal(i,j,k,valid)
	valid = _check_rightest_col_normal(i,j,k,valid)
	valid = _check_bottom_row_normal(i,j,k,valid)

	# Get the actual neighbor coordinates based on validity
	return _cast_neighbour_normal(i,j,k,valid)

@ti.func
def _is_active_normal(i:int, j:int, BCs:ti.template()):
	"""
	Quick utility function determining if a node is active for normal boundaries.

	In normal boundary conditions, all nodes are considered active.

	Args:
		i (int): The row index.
		j (int): The column index.
		BCs (ti.template()): A dummy field to maintain API consistency.

	Returns:
		bool: Always True, as all nodes are active in normal boundary conditions.

	Author: B.G.
	"""
	valid = True
	return valid

@ti.func
def _can_receive_normal(i:int, j:int, BCs:ti.template()):
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
	return valid

@ti.func
def _can_give_normal(i:int, j:int, BCs:ti.template()):
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
	if(i==0 or j == 0 or i == GRID.ny - 1 or j == GRID.nx - 1):
		valid = False
	return valid


@ti.func
def _can_out_normal(i:int, j:int, BCs:ti.template()):
	'''
		Standard complying function for the normal boundaries
		Arguments:
			- i: the row index
			- j: the column index
			- BCs: a dummy field to keep the standard consistent
		Returns:
			- True, all the nodes can receive in the normal boundary conditions
	'''
	valid = False
	if(i==0 or j == 0 or i == GRID.ny - 1 or j == GRID.nx - 1):
		valid = True
	return valid



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


@ti.func
def _check_top_row_customs(i:int, j:int, k:int, BCs:ti.template(), valid:bool):
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
	# I assume it's good
	# Only checking if it actually is on the first row
	if(i == 0):
		# Checking all the different cases: firs, last cols and the middle
		if(k==0):
			valid = False
	# Done
	return valid

@ti.func
def _check_leftest_col_customs(i:int, j:int, k:int, BCs:ti.template(), valid:bool):
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
	# I assume it's good
	# Only checking if it actually is on the first col
	if(j == 0):
		if(k==1):
			valid = False
	# Done
	return valid

@ti.func
def _check_rightest_col_customs(i:int, j:int, k:int, BCs:ti.template(), valid:bool):
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
	# I assume it's good
	# Only checking if it actually is on the first col
	if(j == GRID.nx-1):
		if(k==2):
			valid = False
	# Done
	return valid

@ti.func
def _check_bottom_row_customs(i:int, j:int, k:int, BCs:ti.template(), valid:bool):
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
	# I assume it's good
	# Only checking if it actually is on the first row
	if(i == GRID.ny-1):
		# Checking all the different cases: firs, last cols and the middle
		if(k==3):
			valid = False
	# Done
	return valid

@ti.func
def _cast_neighbour_customs(i:int, j:int, k:int, valid:bool, BCs:ti.template()):
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

	if(ir == -1 or jr == -1):
		ir,jr = -1,-1
	elif(BCs[i,j] == 0):
		ir,jr = -1,-1
	elif(BCs[ir,jr] == 0):
		ir,jr = -1,-1
		

	return ir, jr

@ti.func
def _neighbours_customs(i:int, j:int, k:int, BCs:ti.template()):
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
	valid = _check_top_row_customs(i,j,k,BCs,valid)
	valid = _check_leftest_col_customs(i,j,k,BCs,valid)
	valid = _check_rightest_col_customs(i,j,k,BCs,valid)
	valid = _check_bottom_row_customs(i,j,k,BCs,valid)

	# getting the actual neighbours
	return _cast_neighbour_customs(i,j,k,valid,BCs)

@ti.func
def _can_receive_customs(i:int, j:int, BCs:ti.template()):
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

@ti.func
def _can_give_customs(i:int, j:int, BCs:ti.template()):
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


@ti.func
def _can_out_customs(i:int, j:int, BCs:ti.template()):
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


@ti.func
def _is_active_customs(i:int, j:int, BCs:ti.template()):
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
	if(BCs[i,j] == 0): #) or _can_out_customs(i,j,BCs) or _can_receive_customs(i,j,BCs) == False):
		valid = False
	return valid



########################################################################
########################################################################
############### GENERIC  FUNCTIONS #####################################
########################################################################
########################################################################


@ti.func
def oppk(k:ti.i32):
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

@ti.func
def parak(i:ti.i32, j:ti.i32, k:ti.i32, baseref:ti.u1):
	'''
	Returns the parallel neighbours to a direction given by k
	
	Arguments:
		i: row index
		j: col index
		k: the neihgbour code
		baseref: based in receiver pov in False else base node 
	returns:
		The opposite neighbour code

	Authors:
		- B.G. (last modifications: 06/2024)
	'''
	ret0,ret1,ret2,ret3 = -1,-1,-1,-1
	if(k == 0):
		ret0,ret1,ret2,ret3 = (i,j-1,i,j+1) if baseref else (i-1,j-1,i-1,j+1)
	elif(k == 3):
		ret0,ret1,ret2,ret3 = (i,j-1,i,j+1) if baseref else (i+1,j-1,i+1,j+1)
	elif(k == 1):
		ret0,ret1,ret2,ret3 = (i-1,j,i+1,j) if baseref else (i-1,j-1,i+1,j-1)
	elif(k == 2):
		ret0,ret1,ret2,ret3 = (i-1,j,i+1,j) if baseref else (i-1,j+1,i+1,j+1)
	return ret0,ret1,ret2,ret3


########################################################################
########################################################################
############### EXPOSED API ############################################
########################################################################
########################################################################

# Exporting the generic API
neighbours = None
is_active = None
can_receive = None
can_give = None
can_out = None

def set_grid_CC():
	'''
		Python side to be called before compilation and set the generic functions matching the grid criterion and boundary conditions
	'''
	# fectch neighbours placeholder
	global neighbours
	global is_active
	global can_receive
	global can_give
	global can_out
	
	# Feed it
	if(GRID.boundaries == BoundaryConditions.normal):
		neighbours = _neighbours_normal
		is_active = _is_active_normal
		can_receive = _can_receive_normal
		can_give = _can_give_normal
		can_out = _can_out_normal
	elif(GRID.boundaries == BoundaryConditions.customs):
		# print('DEBUG::BC::CUSTOM')
		neighbours = _neighbours_customs
		is_active = _is_active_customs
		can_receive = _can_receive_customs
		can_give = _can_give_customs
		can_out = _can_out_customs
	else:
		raise NotImplementedError('BC Not implemented yet')
