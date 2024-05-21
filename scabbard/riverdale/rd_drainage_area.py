'''
Sets of function to compute drainage area metrics with riverdale
EXPERIMENTAL, no warranty it evens do what it is supposed to do yet

B.G. - 29/04/2024
'''

import taichi as ti
import numpy as np
from enum import Enum
import scabbard.utils as scaut 
from scabbard.riverdale.rd_grid import GRID
import scabbard.riverdale.rd_grid as gridfuncs


@ti.kernel
def compute_D4_nolm(Z:ti.template(), D4dir:ti.template(), BCs:ti.template()):
	'''
	Experimental tests on drainage area calculations
	Do not use at the moment
	B.G.
	'''

	# Traversing each nodes
	for i,j in Z:

		D4dir[i,j] = ti.uint8(5)

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SS = 0.
		lowest_higher_Z = 0.
		checked = True

	
		# Traversing Neighbours
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			# Local hydraulic slope
			tS = Z[i,j] - Z[ir,jr]
			tS /= GRID.dx

			# If < 0, neighbour is a donor and I am not interested
			if(tS <= 0):
				if(Z[ir,jr] < lowest_higher_Z or lowest_higher_Z == 0.):
					lowest_higher_Z = Z[ir,jr]
				continue

			# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
			if(tS > SS):
				D4dir[i,j] = ti.uint8(k)
				SS = tS


			# Local minima management (cheap but works)
			## If I have no downward slope, I increase the elevation by a bit
			# if(SS == 0.):
			# 	if(checked):
			# 		checker[None] += 1
			# 		checked = False
			# 	Z[i,j] = max(lowest_higher_Z,Z[i,j]) + 1e-4

@ti.kernel
def compute_D4(Z:ti.template(), D4dir:ti.template(), BCs:ti.template(), checker:ti.template() ):
	'''
	Experimental tests on drainage area calculations
	Do not use at the moment
	B.G.
	'''

	# Traversing each nodes
	for i,j in Z:

		D4dir[i,j] = ti.uint8(5)

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SS = 0.
		lowest_higher_Z = 0.
		checked = True

		# While I do not have external slope
		while(SS == 0.):
			
			# Traversing Neighbours
			for k in range(4):
				# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)

				# if not a neighbours, by convention is < 0 and I pass
				if(ir == -1):
					continue

				# Local hydraulic slope
				tS = Z[i,j] - Z[ir,jr]
				tS /= GRID.dx

				# If < 0, neighbour is a donor and I am not interested
				if(tS <= 0):
					if(Z[ir,jr] < lowest_higher_Z or lowest_higher_Z == 0.):
						lowest_higher_Z = Z[ir,jr]
					continue

				# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
				if(tS > SS):
					D4dir[i,j] = ti.uint8(k)
					SS = tS


			# Local minima management (cheap but works)
			## If I have no downward slope, I increase the elevation by a bit
			if(SS == 0.):
				if(checked):
					checker[None] += 1
					checked = False
				Z[i,j] = max(lowest_higher_Z,Z[i,j]) + 1e-4


	# if(globcheck > 0):
	# 	compute_D4(Z,D4dir,BCs)


@ti.kernel
def step_DA_D4(Z:ti.template(), DA:ti.template(), temp:ti.template(), D4dir:ti.template(), BCs:ti.template() ):
	'''
	Compute and transfer QwA (in from t-1) into a temporary QwB (in for t).
	Also computes QwC (out at t) 
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- QwA: a 2D field of discharge A (in)
		- QwB: a 2D field of discharge B (in t+1)
		- QwC: a 2D field of discharge C (out)
		- BCs: a 2D field of boundary conditions
	Returns:
		- Nothing, Caluclates disccharges in place
	Authors:
		- B.G. (last modification 03/05/2024)
	'''

	for i,j in Z:
		temp[i,j] = GRID.dx * GRID.dy
	
	for i,j in Z:
		if(gridfuncs.is_active(i,j,BCs)):
			if(D4dir[i,j] == 5):
				continue
			ir,jr = gridfuncs.neighbours(i,j,D4dir[i,j],BCs)
			if(ir>-1):
				ti.atomic_add(temp[ir,jr], DA[i,j])
	
	for i,j in Z:
		DA[i,j] = temp[i,j]
	

		