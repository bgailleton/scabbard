'''
Sets of function to compute hydrodynamics with RiverDale

B.G. - 29/04/2024
'''

import taichi as ti
import numpy as np
from enum import Enum
import scabbard.utils as scaut 
from scabbard.riverdale.rd_grid import GRID
import scabbard.riverdale.rd_grid as gridfuncs


class FlowMode(Enum):
	'''
	Enumeration of the different boundary condition types possible
	'''	
	static_incremental = 0


@scaut.singleton
class HydroParams:
	'''
		Internal singleton class holding all the compile time constant parameters for the hydro simulations 
		Not for users
	'''
	def __init__(self):		
		self.dt_hydro = 1e-3
		self.manning = 0.033
		self.flowmode = FlowMode.static_incremental


PARAMHYDRO = HydroParams()

@ti.kernel
def initiate_step(QwB: ti.template()):
	'''
	Runs the initial operations when running a step
	Mostly a placeholder so far but keeps things consistents
	Arguments:
		- QwB: a 2D field of discharge B (in_temp)
	Returns:
		- Nothing, edits in place
	Authors:
		- B.G. (last modification 30/04/2024)
	'''
	# Reinitialise QwB to 0.
	for i,j in QwB:
		QwB[i,j] = 0.


@ti.kernel
def constant_rain(QwA: ti.template(), QwB: ti.template(), P: ti.f32):
	'''
	Adds a constant precipitation rates to every cells
	Arguments:
		- QwA: a 2D field of discharge A (in)
		- QwB: a 2D field of discharge B (in_temp)
		- P: a constant floating point of precipitation rates in m/s
	Returns:
		- Nothing, edits in place
	Authors:
		- B.G. (last modification 30/04/2024)
	'''
	for i,j in QwA:

		if(gridfuncs.is_active(i,j) == False):
			continue
		
		QwA[i,j] += P * GRID.dx * GRID.dy
		QwB[i,j] = P * GRID.dx * GRID.dy

@ti.kernel
def variable_rain(QwA: ti.template(), QwB: ti.template(), P: ti.template()):
	'''
	Adds a spatially variable precipitation rates to every cells
	Arguments:
		- QwA: a 2D field of discharge A (in)
		- QwB: a 2D field of discharge B (in_temp)
		- P: a 2D field of precipitation rates in m/s
	Returns:
		- Nothing, edits in place
	Authors:
		- B.G. (last modification 30/04/2024)
	'''
	for i,j in Z:

		if(gridfuncs.is_active(i,j) == False):
			continue

		QwA[i,j] += P[i,j] * GRID.dx * GRID.dy
		QwB[i,j] += P[i,j] * GRID.dx * GRID.dy

@ti.kernel
def input_discharge_points(input_rows: ti.template(), input_cols:ti.template(), input_values:ti.template(), QwA: ti.template(), QwB: ti.template()):
	'''
	Adds discharge in m^3/s into specific input points
	Arguments:
		- input_rows: a 1D field of input point row coordinates (integrer 32 bits)
		- input_cols: a 1D field of input point column coordinates (integrer 32 bits)
		- input_values: a 1D field of input discharges in m^3/s (floating point)
		- QwA: a 2D field of discharge A (in)
		- QwB: a 2D field of discharge B (in_temp)
	Returns:
		- Nothing, edits in place
	Authors:
		- B.G. (last modification 30/04/2024)
	'''

	for i in input_rows:
		QwA[input_rows[i],input_cols[i]] += input_values[i]
		QwB[input_rows[i],input_cols[i]] += input_values[i]



@ti.func
def Zw(Z: ti.template(), hw: ti.template(), i, j) -> ti.f32:
	'''
	Internal helping function returning the hydrayulic surface (elevation of the water surface)
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- i,j: the row col indices
	Returns:
		- the hydraulic surface
	Authors:
		- B.G. (last modification 30/04/2024)
	'''

	return Z[i,j] + hw[i,j]

@ti.func
def Sw(Z: ti.template(), hw: ti.template(), i, j, ir, jr)->ti.f32:
	'''
	Internal helping function returning the hydrayulic slope
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- i,j: the row col indices
		- i,j: the row col indices of the receivers node
	Returns:
		- the hydraulic surface
	Authors:
		- B.G. (last modification 30/04/2024)
	'''
	return (Zw(Z,hw, i,j) - Zw(Z,hw, ir,jr))/GRID.dx


@ti.kernel
def _compute_Qw(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template() ):
	'''
	Compute and transfer QwA (in from t-1) into a temporary QwB (in for t).
	Also computes QwC (out at t) 
	Arguments:
		- QwA: a 2D field of discharge A (in)
		- QwB: a 2D field of discharge B (out)
		- P: a 2D field of precipitation rates in m/s
	Returns:
		- Nothing, edits in place
	Authors:
		- B.G. (last modification 30/04/2024)
	'''

	for i,j in Z:

		if(i == GRID.ny-1 or i ==0 or j==0 or j == GRID.nx-1 or Z[i,j] < 0.):
			continue

		Sws = ti.math.vec4(0.,0.,0.,0.)
		sumSw = 0.
		SSx = 0.
		SSy = 0.
		lockcheck = 0
		while(sumSw == 0.):
			lockcheck += 1
			for k in range(4):
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)
				if(ir == -1):
					continue

				tS = Sw(Z,hw,i,j,ir,jr)
				if(tS <= 0):
					continue

				if(k == 0 or k == 3):
					if(tS > SSy):
						SSy = tS
				else:
					if(tS > SSx):
						SSx = tS

				Sws[k] = tS
				sumSw += tS
			if(sumSw == 0.):
				hw[i,j] += 1e-4
			if(lockcheck > 10000):
				break

		gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
		if(gradSw == 0):
			continue
		QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(hw[i,j], 5./3) * sumSw/ti.math.sqrt(gradSw)
		# print(QwC[i,j])
		for k in range(4):
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)
			if(ir == -1):
				continue
			ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])





@ti.kernel
def _compute_hw(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template() ):
	
	for i,j in Z:
	
		QwA[i,j] = QwB[i,j]
	
		if(i == GRID.ny-1 or i == 0 or j==0 or j == GRID.nx-1 or Z[i,j] < 0.):
			continue
		tdt = PARAMHYDRO.dt_hydro * 1.
		hw[i,j] = ti.math.max(0.,hw[i,j] + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy) ) 


# Exporting the generic API
compute_hw = None
compute_Qw = None


def set_hydro_CC():
	'''
		Python side to be called before compilation and set the generic functions matching the grid criterion and boundary conditions
	'''
	# # fectch neighbours placeholder
	global compute_hw
	global compute_Qw

	
	# Feed it
	if(PARAMHYDRO.flowmode == FlowMode.static_incremental):
		compute_hw = _compute_hw
		compute_Qw = _compute_Qw
	else:
		raise NotImplementedError('FLOWMODEW Not implemented yet')
