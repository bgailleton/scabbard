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
import scabbard.riverdale.rd_helper_surfw as hsw



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
		# Constant dt for hydrodynamics
		self.dt_hydro = 1e-3
		# Constant manning coefficient
		self.manning = 0.033
		self.flowmode = FlowMode.static_incremental
		self.hydro_slope_bc_mode = 0
		self.hydro_slope_bc_val = 0



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
def constant_rain(QwA: ti.template(), QwB: ti.template(), P: ti.f32, BCs:ti.template()):
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

		if(gridfuncs.is_active(i,j, BCs) == False):
			continue
		
		QwA[i,j] += P * GRID.dx * GRID.dy
		QwB[i,j] = P * GRID.dx * GRID.dy

@ti.kernel
def variable_rain(QwA: ti.template(), QwB: ti.template(), P: ti.template(), BCs:ti.template()):
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

		if(gridfuncs.is_active(i,j, BCs) == False):
			continue

		QwA[i,j] += P[i,j] * GRID.dx * GRID.dy
		QwB[i,j] += P[i,j] * GRID.dx * GRID.dy

@ti.kernel
def input_discharge_points(input_rows: ti.template(), input_cols:ti.template(), input_values:ti.template(), QwA: ti.template(), QwB: ti.template(), BCs:ti.template()):
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



@ti.kernel
def _compute_Qw(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template() ):
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


	# Traversing each nodes
	for i,j in Z:

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.is_active(i,j,BCs) == False):
			continue

		# I'll store the hydraulic slope in this vector
		Sws = ti.math.vec4(0.,0.,0.,0.)

		# I'll need the sum of the hydraulic slopes in the positive directions
		sumSw = 0.

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SSx = 0.
		SSy = 0.

		# Safety check: gets incremented at each while iteration and manually breaks the loop if > 10k (avoid getting stuck in an infinite hole)
		lockcheck = 0


		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):
			# While I do not have external slope
			while(sumSw == 0.):
				
				# First incrementing the safety check
				lockcheck += 1

				# Traversing Neighbours
				for k in range(4):

					# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
					ir,jr = gridfuncs.neighbours(i,j,k, BCs)

					# if not a neighbours, by convention is < 0 and I pass
					if(ir == -1):
						continue

					if(gridfuncs.can_receive(ir,jr, BCs) == False):
						continue

					# Local hydraulic slope
					tS = hsw.Sw(Z,hw,i,j,ir,jr)

					# If < 0, neighbour is a donor and I am not interested
					if(tS <= 0):
						continue

					# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
					if(k == 0 or k == 3):
						if(tS > SSy):
							SSy = tS
					else:
						if(tS > SSx):
							SSx = tS

					# Registering local slope
					Sws[k] = tS
					# Summing it to global
					sumSw += tS

					# Done with processing this particular neighbour

				# Local minima management (cheap but works)
				## If I have no downward slope, I increase the elevation by a bit
				if(sumSw == 0.):
					hw[i,j] += 1e-4

				## And if I added like a metre and it did not slolve it, I stop for that node
				if(lockcheck > 100):
					break

			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			
			# Not sure I still need that
			if(gradSw == 0):
				continue

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), 5./3) * sumSw/ti.math.sqrt(gradSw)

			# Transferring flow to neighbours
			for k in range(4):

				# local neighbours
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)
				
				# checking if neighbours
				if(ir == -1):
					continue
				
				# Transferring prop to the hydraulic slope
				ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])

		# Boundary case
		else:
			tSw = ti.max(hsw.Zw(Z,hw,i,j) -  PARAMHYDRO.hydro_slope_bc_val, 1e-6)/GRID.dx if PARAMHYDRO.hydro_slope_bc_mode == 0 else PARAMHYDRO.hydro_slope_bc_val
			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), 5./3) * ti.math.sqrt(tSw)

@ti.kernel
def _compute_Qw_drape(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), constrains:ti.template(), BCs:ti.template() ):
	'''
	EXPERIMENTAL: testing some dynamic draping
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


	# Traversing each nodes
	for i,j in Z:

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.is_active(i,j,BCs) == False):
			continue

		# I'll store the hydraulic slope in this vector
		Sws = ti.math.vec4(0.,0.,0.,0.)

		# I'll need the sum of the hydraulic slopes in the positive directions
		sumSw = 0.

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SSx = 0.
		SSy = 0.

		# Safety check: gets incremented at each while iteration and manually breaks the loop if > 10k (avoid getting stuck in an infinite hole)
		lockcheck = 0


		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):
			# While I do not have external slope
			while(sumSw == 0.):
				
				# First incrementing the safety check
				lockcheck += 1

				# Traversing Neighbours
				for k in range(4):

					# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
					ir,jr = gridfuncs.neighbours(i,j,k, BCs)

					# if not a neighbours, by convention is < 0 and I pass
					if(ir == -1):
						continue

					if(gridfuncs.can_receive(ir,jr, BCs) == False):
						continue

					# Local hydraulic slope
					tS = hsw.Sw(Z,hw,i,j,ir,jr)

					# If < 0, neighbour is a donor and I am not interested
					if(tS <= 0):
						continue

					# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
					if(k == 0 or k == 3):
						if(tS > SSy):
							SSy = tS
					else:
						if(tS > SSx):
							SSx = tS

					# Registering local slope
					Sws[k] = tS
					# Summing it to global
					sumSw += tS

					# Done with processing this particular neighbour

				# Local minima management (cheap but works)
				## If I have no downward slope, I increase the elevation by a bit
				if(sumSw == 0.):
					hw[i,j] += 1e-4

				## And if I added like a metre and it did not slolve it, I stop for that node
				if(lockcheck > 100):
					break

			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			
			# Not sure I still need that
			if(gradSw == 0):
				continue

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), 5./3) * sumSw/ti.math.sqrt(gradSw)

			# Transferring flow to neighbours
			for k in range(4):

				# local neighbours
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)
				
				# checking if neighbours
				if(ir == -1):
					continue
				
				# Transferring prop to the hydraulic slope
				ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])

		# Boundary case
		else:
			tSw = ti.max(hsw.Zw(Z,hw,i,j) -  PARAMHYDRO.hydro_slope_bc_val, 1e-6)/GRID.dx if PARAMHYDRO.hydro_slope_bc_mode == 0 else PARAMHYDRO.hydro_slope_bc_val
			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), 5./3) * ti.math.sqrt(tSw)


@ti.kernel
def _compute_Qw_noLM(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template() ):
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


	# Traversing each nodes
	for i,j in Z:

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.is_active(i,j,BCs) == False):
			continue

		# I'll store the hydraulic slope in this vector
		Sws = ti.math.vec4(0.,0.,0.,0.)

		# I'll need the sum of the hydraulic slopes in the positive directions
		sumSw = 0.

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SSx = 0.
		SSy = 0.

		# Safety check: gets incremented at each while iteration and manually breaks the loop if > 10k (avoid getting stuck in an infinite hole)
		lockcheck = 0


		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):

			
			# First incrementing the safety check
			lockcheck += 1

			# Traversing Neighbours
			for k in range(4):

				# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)

				# if not a neighbours, by convention is < 0 and I pass
				if(ir == -1):
					continue

				if(gridfuncs.can_receive(ir,jr, BCs) == False):
					continue

				# Local hydraulic slope
				tS = hsw.Sw(Z,hw,i,j,ir,jr)

				# If < 0, neighbour is a donor and I am not interested
				if(tS <= 0):
					continue

				# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
				if(k == 0 or k == 3):
					if(tS > SSy):
						SSy = tS
				else:
					if(tS > SSx):
						SSx = tS

				# Registering local slope
				Sws[k] = tS
				# Summing it to global
				sumSw += tS


			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			
			# Not sure I still need that
			if(gradSw == 0):
				continue

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), 5./3) * sumSw/ti.math.sqrt(gradSw)

			# Transferring flow to neighbours
			for k in range(4):

				# local neighbours
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)
				
				# checking if neighbours
				if(ir == -1):
					continue
				
				# Transferring prop to the hydraulic slope
				ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])

		# Boundary case
		else:
			tSw = ti.max(hsw.Zw(Z,hw,i,j) -  PARAMHYDRO.hydro_slope_bc_val, 1e-6)/GRID.dx if PARAMHYDRO.hydro_slope_bc_mode == 0 else PARAMHYDRO.hydro_slope_bc_val
			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), 5./3) * ti.math.sqrt(tSw)





@ti.kernel
def _compute_hw(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template() ):
	'''
	Compute flow depth by div.Q and update discharge to t+1.
	Also computes QwC (out at t) 
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- QwA: a 2D field of discharge A (in)
		- QwB: a 2D field of discharge B (in t+1)
		- QwC: a 2D field of discharge C (out)
		- BCs: a 2D field of boundary conditions
	Returns:
		- Nothing, update flow depth in place
	Authors:
		- B.G. (last modification 03/05/2024)
	'''	

	# Traversing nodes
	for i,j in Z:
		
		# Updating local discharge to new time step
		QwA[i,j] = QwB[i,j]
		
		# ONGOING TEST DO NOT DELETE
		# # Only where nodes are active (i.e. flow cannot leave and can traverse)
		# if(gridfuncs.can_out(i,j,BCs)):
		# 	continue


		# Updating flow depth (cannot be < 0)
		hw[i,j] = hw[i,j] + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy)


@ti.kernel
def _compute_hw_drape(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), D4dir:ti.template(), constrains:ti.template(), BCs:ti.template() ):
	'''
	EXPERIMENTAL
	Compute flow depth by div.Q and update discharge to t+1.
	Also computes QwC (out at t) 
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- QwA: a 2D field of discharge A (in)
		- QwB: a 2D field of discharge B (in t+1)
		- QwC: a 2D field of discharge C (out)
		- constrains: a 3D field of minimum [i,j,0] and maximum [i,j,1] Zw possible for every nodes
		- BCs: a 2D field of boundary conditions
	Returns:
		- Nothing, update flow depth in place
	Authors:
		- B.G. (last modification 03/05/2024)
	'''	

	for i,j in Z:

		constrains[i,j,0] = -1e6
		constrains[i,j,1] = 1e6
		D4dir[i,j] = 5


	for i,j in Z:

		tZw = hsw.Zw_drape(Z,hw,i,j) + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy)
		if gridfuncs.is_active(i,j,BCs) == False:
			continue

		kmin = 5
		Zdkmin = tZw

		# Traversing Neighbours
		for k in range(4):

			# Getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			if(ir == -1):
				continue

			# tSwr = hsw.Zw_drape(Z,hw,ir,jr) + (QwA[ir,jr] - QwC[ir,jr]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy)
			tZwr = hsw.Zw_drape(Z,hw,ir,jr)

			if(tZwr < Zdkmin):
				Zdkmin = tZwr
				kmin = k

		if kmin==5:
			continue

		ir,jr = gridfuncs.neighbours(i,j,kmin, BCs)


		D4dir[i,j] = kmin

		constrains[i,j,0] = 0.49 * (hsw.Zw_drape(Z,hw,ir,jr) + tZw)


		constrains[ir,jr,1] = ti.atomic_min(constrains[ir,jr,1] ,0.49 * (hsw.Zw_drape(Z,hw,ir,jr) + tZw))

		# constrains[i,j,0] -= Z[i,j]
		# constrains[i,j,1] -= Z[i,j]



	# Traversing nodes
	for i,j in Z:
		
		# Updating local discharge to new time step
		QwA[i,j] = QwB[i,j]
		
		# Updating flow depth (cannot be < 0)
		# hw[i,j]  =  ti.math.clamp(
		# 					 hw[i,j] + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy) ,
		# 					 # TODO try to keep track of the receivers and to force no inversion
		# 				constrains[i,j,0], # min
		# 				# constrains[i,j,1]  # max
		# 				1e6  # max
		# 			)
		hw[i,j] = hw[i,j] + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy)

		# if(constrains[i,j,0] == 1e6):
		# 	constrains[i,j,0] = Z[i,j]
		# if(constrains[i,j,1] == 1e6):
		# 	constrains[i,j,1] = Z[i,j]
		
		constrains[i,j,0] -= Z[i,j]
		constrains[i,j,1] -= Z[i,j]


		
		if(gridfuncs.can_out(i,j,BCs) == False):
			if(hw[i,j] < constrains[i,j,0]):
				hw[i,j] = constrains[i,j,0]
			elif(hw[i,j] > constrains[i,j,1]):
				hw[i,j] = constrains[i,j,1]





@ti.kernel
def check_convergence(QwA : ti.template(), QwC : ti.template(), tolerance:ti.f32, converged:ti.template(), BCs:ti.template()):
	'''
	Warning, slow function-ish (to call every 100s of iterations is OK) that check the proportion of nodes that have reached convergence.
	Only valid for steady flow assumption.
	Computes the ratio between Qw_out and Qw_in and save the proportion of nodes that have reached convergence within a tolerance factor.

	Arguments:
		- QwA: Discharge input to every cell (calculated by the compute_Qw function)
		- QwC: Discharge output to every cell (calculated by the compute_Qw function)
		- tolerance: determines if a node has converged if its tolerance > |1 - ratio|
		- converged: the convergence rate = number of nodes converged / total number of (active) nodes
		- BCs: the field of boundary conditions
		

	'''
	# Final count of converged
	count = 0
	# Total number of active nodes
	tot = 0
	# main loop
	for i,j in QwA:
		# Ignoring points without QwA
		if(QwA[i,j] > 0):
			# Is active: incrementing
			tot += 1
			# Ration Qwout / Qwin
			rat = QwC[i,j]/QwA[i,j]
			if(rat >= 1-tolerance and rat < (1 + tolerance)):
				count += 1

	# The final scalar result
	converged[None] = count/tot



########################################################################
########################################################################
############### EXPOSED API ############################################
########################################################################
########################################################################

# Note that the input functions are already exposed

compute_hw = None
compute_Qw = None


def set_hydro_CC():
	'''
	Expose the right API function of the comnpile time parameters in PARAMHYDRO.
	Need to be called after setting hte different parameters in the singleton PARAMHYDRO
	Returns:
		- Nothing, update global function refs
	Authors:
		- B.G. (last modification 03/05/2024)
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
