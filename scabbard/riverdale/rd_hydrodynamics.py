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
import scabbard.riverdale.rd_utils as rut



class FlowMode(Enum):
	'''
	Enumeration of the different boundary condition types possible
	'''	
	static_incremental = 0
	static_drape = 1
	static_link = 2


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
		self.exponent_flow = np.float32(5./3.)
		self.flowmode = FlowMode.static_incremental
		self.hydro_slope_bc_mode = 0
		self.hydro_slope_bc_val = 0

		# Constant for minimal step in the draping function
		self.mini_drape_step = np.float32(0.0005)

		self.use_heffmax = False

		self.use_original_dir_for_LM = True
		self.LM_pathforcer = 1
		self.clamp_div_hw = True
		self.clamp_div_hw_val = 1e-3

		self.checkboard_checker = False

		self.CA_redis = 0.1

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

		if(gridfuncs.is_active(i,j, BCs) == False or gridfuncs.can_give(i,j,BCs) == False):
			continue
		
		# QwA[i,j] += P * GRID.dx * GRID.dy
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
	for i,j in QwA:

		if(gridfuncs.is_active(i,j, BCs) == False or gridfuncs.can_give(i,j,BCs) == False):
			continue

		# QwA[i,j] += P[i,j] * GRID.dx * GRID.dy
		QwB[i,j] = P[i,j] * GRID.dx * GRID.dy

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
		# QwA[input_rows[i],input_cols[i]] += input_values[i]
		QwB[input_rows[i],input_cols[i]] += input_values[i]



@ti.kernel
def _compute_Qw(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template(), flowdir:ti.template() ):
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
		- fdir: a 2D field of flow direction to follow in the event of a local minima
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

		if(gridfuncs.can_give(i,j,BCs) == False and gridfuncs.can_out(i,j,BCs) == False):
			continue

		if(QwA[i,j] == 0 and hw[i,j] == 0):
			continue

		# I'll store the hydraulic slope in this vector
		Sws = ti.math.vec4(0.,0.,0.,0.)

		# I'll need the sum of the hydraulic slopes in the positive directions
		sumSw = 0.

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SSx = 0.
		SSy = 0.

		# allmore = True
		# allless = True

		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):
				
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

				# if(QwA[ir,jr] < QwA[i,j]):
				# 	allmore = False
				# elif(QwA[ir,jr] > QwA[i,j]):
				# 	allless = False

				# Done with processing this particular neighbour

			# Local minima management (cheap but works)
			## If I have no downward slope, I increase the elevation by a bit
			if(sumSw == 0.): # or ((allmore or allless) and PARAMHYDRO.checkboard_checker)):
				# I am in a local minima
				# this option ensures the drainage of LM through the original rail
				# Flow dir == 5 is no flow
				if(flowdir[i,j] != 5):
					# That section ensures that a stochastic number of receivers are traversed to flush the local minima
					# And avoid ping-pong or localisation based biases
					ii,jj = i,j
					ir,jr = i,j
					first = 0
					# Receivers are poped out at least once, and then has a probability of 0.5 to continue
					while(flowdir[ir,jr] != 5):
						if((first>PARAMHYDRO.LM_pathforcer) and ti.random() < 0.5):
							break
						ii,jj = ir,jr
						first += 1
						ir,jr = gridfuncs.neighbours(ii, jj, flowdir[ii,jj], BCs)
						# ti.atomic_add(QwB[ir,jr], QwA[i,j])
					ti.atomic_add(QwB[ir,jr], QwA[i,j])
				else:
					ti.atomic_add(QwB[i,j], QwA[i,j])

				QwC[i,j] = 0.
				continue

			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			thw = ti.math.max(0., hw[i,j])

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(thw, PARAMHYDRO.exponent_flow) * sumSw/ti.math.sqrt(gradSw)

				
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
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), PARAMHYDRO.exponent_flow) * ti.math.sqrt(tSw)

@ti.kernel
def _compute_Qw_dynamic(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template()):
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
		- fdir: a 2D field of flow direction to follow in the event of a local minima
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

		if(gridfuncs.can_give(i,j,BCs) == False and gridfuncs.can_out(i,j,BCs) == False):
			continue

		# I'll store the hydraulic slope in this vector
		Sws = ti.math.vec4(0.,0.,0.,0.)

		# I'll need the sum of the hydraulic slopes in the positive directions
		sumSw = 0.

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SSx = 0.
		SSy = 0.

		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):
				
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

			# No receivers, no output discharge
			if(sumSw == 0.):
				QwC[i,j] = 0.
				QwB[i,j] = QwA[i,j]
				continue

			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			thw = ti.math.max(0., hw[i,j])

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(thw, PARAMHYDRO.exponent_flow) * sumSw/ti.math.sqrt(gradSw)

				
			# Transferring flow to neighbours
			for k in range(4):

				# local neighbours
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)
				
				# checking if neighbours
				if(ir == -1):
					continue
				
				# Transferring prop to the hydraulic slope
				ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwC[i,j])

		# Boundary case
		else:
			tSw = ti.max(hsw.Zw(Z,hw,i,j) -  PARAMHYDRO.hydro_slope_bc_val, 1e-6)/GRID.dx if PARAMHYDRO.hydro_slope_bc_mode == 0 else PARAMHYDRO.hydro_slope_bc_val
			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), PARAMHYDRO.exponent_flow) * ti.math.sqrt(tSw)





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
	# print(PARAMHYDRO.dt_hydro,"OIOIOIOI")
	# Traversing nodes
	for i,j in Z:
		
		# Updating local discharge to new time step
		QwA[i,j] = QwB[i,j]
		
		# ONGOING TEST DO NOT DELETE
		# # Only where nodes are active (i.e. flow cannot leave and can traverse)
		# if(gridfuncs.can_out(i,j,BCs)):
		# 	continue


		# Updating flow depth (cannot be < 0)
		dhw = (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy)

		if(PARAMHYDRO.clamp_div_hw):
			if(dhw>0):
				dhw = ti.math.min(PARAMHYDRO.clamp_div_hw_val, dhw)
			else:
				dhw = ti.math.max(-PARAMHYDRO.clamp_div_hw_val, dhw)

		hw[i,j] = ti.max(0.,hw[i,j] + dhw)

@ti.kernel
def _compute_hw_dynamic(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template() ):
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
	# print(PARAMHYDRO.dt_hydro,"OIOIOIOI")
	# Traversing nodes
	for i,j in Z:
		
		
		
		# ONGOING TEST DO NOT DELETE
		# # Only where nodes are active (i.e. flow cannot leave and can traverse)
		# if(gridfuncs.can_out(i,j,BCs)):
		# 	continue


		# Updating flow depth (cannot be < 0)
		dhw = (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy)

		if(PARAMHYDRO.clamp_div_hw):
			if(dhw>0):
				dhw = ti.math.min(PARAMHYDRO.clamp_div_hw_val, dhw)
			else:
				dhw = ti.math.max(-PARAMHYDRO.clamp_div_hw_val, dhw)

		hw[i,j] = ti.max(0.,hw[i,j] + dhw)

		# Updating local discharge to new time step
		QwA[i,j] = QwB[i,j]


@ti.kernel
def _raise_analytical_hw(Z:ti.template(), hw:ti.template(), QwA:ti.template(), temp:ti.template(), BCs:ti.template()):
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

		temp[i,j] = hw[i,j]

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SSx = 0.
		SSy = 0.

		# Safety check: gets incremented at each while iteration and manually breaks the loop if > 10k (avoid getting stuck in an infinite hole)
		lockcheck = 0

		if(gridfuncs.can_give(i,j,BCs) == False and gridfuncs.can_out(i,j,BCs) == False):
			continue

		thw = 0.

		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):

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

				thw = ti.math.max(thw, ti.max(hsw.Zw(Z,hw,i,j),hsw.Zw(Z,hw,ir,jr)) - ti.max(Z[i,j],Z[ir,jr]))

				# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
				if(k == 0 or k == 3):
					if(tS > SSy):
						SSy = tS
				else:
					if(tS > SSx):
						SSx = tS


			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy) if SSx > 0 or SSy > 0 else 1e-5

			temp[i,j] = (QwA[i,j] * PARAMHYDRO.manning/(GRID.dx * (gradSw)**(0.5) ))**(3./5.)

	for i,j in Z:
		hw[i,j] = temp[i,j]


@ti.kernel
def _CA_smooth(Z:ti.template(), hw:ti.template(), hw_new:ti.template(), BCs:ti.template()):

	for i,j in Z:
		hw_new[i,j] = hw[i,j]
	
	for i,j in Z:
	
		if(gridfuncs.can_out(i, j, BCs) or gridfuncs.is_active(i,j,BCs) == False):
			continue

		totS = 0.
		dh = PARAMHYDRO.CA_redis * hw[i,j]

		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)
			if(ir == -1):
				continue

			tS = (Z[i,j] - Z[ir,jr] + hw[i,j] - hw[ir,jr])/GRID.dx
			if tS > 0.:
				totS += tS

		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)
			if(ir == -1):
				continue
			tS = (Z[i,j] - Z[ir,jr] + hw[i,j] - hw[ir,jr])/GRID.dx
			if tS > 0.:
				hw_new[ir,jr] += tS * dh / totS

		hw_new[i,j] -= dh

	for i,j in Z:
		hw[i,j] = hw_new[i,j]


@ti.kernel
def _compute_Qw_surfrec(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template(), surfrec:ti.template() ):
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

		if(gridfuncs.can_give(i,j,BCs) == False and gridfuncs.can_out(i,j,BCs) == False):
			continue

		thw = 0.

		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):
			# While I do not have external slope
		
			# Traversing Neighbours
			for k in range(4):

				# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)

				# if not a neighbours, by convention is < 0 and I pass
				if(ir == -1):
					continue

				if(gridfuncs.can_receive(ir,jr, BCs) == False):
					continue

				if(surfrec[i,j] <= surfrec[ir,jr]):
					continue

				# Local hydraulic slope
				tS = hsw.Sw(Z,hw,i,j,ir,jr)

				if(tS <= 0.):
					tS = 1e-3

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

		
			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			
			# Not sure I still need that
			if(gradSw == 0):
				continue

			thw = ti.max(hw[i,j], 0.)

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(thw, PARAMHYDRO.exponent_flow) * sumSw/ti.math.sqrt(gradSw)


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
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), PARAMHYDRO.exponent_flow) * ti.math.sqrt(tSw)
			
			

@ti.kernel
def _compute_hw_CFL(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template(), alpha : ti.f32, threshold:ti.f32 ):
	'''
	Status: STILL EXPERIMENTAL
	Variant of _compute_hw that does not consider nodes nearly equillibrated AND
	
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

		if(QwA[i,j] <= 0 or abs(1 - (QwC[i,j]/QwA[i,j])) < threshold ):
			continue


		# Updating flow depth (cannot be < 0)
		tdt = PARAMHYDRO.dt_hydro
		if(QwC[i,j] > 0):	
			tdt = ti.math.max(PARAMHYDRO.dt_hydro, alpha *GRID.dx/(QwA[i,j]/(ti.math.max(1e-4, hw[i,j]))) )
			
		hw[i,j] = hw[i,j] + (QwA[i,j] - QwC[i,j]) * tdt/(GRID.dx*GRID.dy) 

@ti.kernel
def _compute_hw_th(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template(), tdt:ti.f32, threshold:ti.f32 ):
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

		if(QwA[i,j] <= 0 or abs(1 - (QwC[i,j]/QwA[i,j])) < threshold ):
			continue


		# Updating flow depth (cannot be < 0)
		# tdt = ti.math.max(PARAMHYDRO.dt_hydro, alpha *GRID.dx/(QwA[i,j]/(ti.math.max(1e-4, hw[i,j]))) )		
		hw[i,j] = hw[i,j] + (QwA[i,j] - QwC[i,j]) * tdt/(GRID.dx*GRID.dy) 


@ti.kernel
def _flush_QwA_only(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), BCs:ti.template(), flowdir:ti.template() ):
	'''
	Compute one iteration of QwA propagation

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
		QwB[i,j] = 0.


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

		if(gridfuncs.can_give(i,j,BCs) == False and gridfuncs.can_out(i,j,BCs) == False):
			continue


		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):


			if(flowdir[i,j] != 5):
				ir,jr = gridfuncs.neighbours(i,j,flowdir[i,j], BCs)
				# Transferring prop to the hydraulic slope
				ti.atomic_add(QwB[ir,jr], QwA[i,j])

	for i,j in Z:
		QwA[i,j] = QwB[i,j]



@ti.kernel
def _propagate_QwA_only(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), BCs:ti.template() ):
	'''
	Compute one iteration of QwA propagation

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
		QwB[i,j] = 0.


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

		if(gridfuncs.can_give(i,j,BCs) == False and gridfuncs.can_out(i,j,BCs) == False):
			continue


		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):


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

			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			
			# Not sure I still need that
			if(gradSw == 0):
				continue

			# Transferring flow to neighbours
			for k in range(4):

				# local neighbours
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)
				
				# checking if neighbours
				if(ir == -1):
					continue
				
				# Transferring prop to the hydraulic slope
				ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])

	for i,j in Z:
		QwA[i,j] = QwB[i,j]


@ti.kernel
def _compute_QwA_from_Zw(Z:ti.template(), hw:ti.template(), QwA:ti.template(), BCs:ti.template() ):
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

		if(gridfuncs.can_give(i,j,BCs) == False and gridfuncs.can_out(i,j,BCs) == False):
			continue

		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):
							


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
					QwA[i,j] = 0.


			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			
			# Not sure I still need that
			if(gradSw == 0):
				QwA[i,j] = 0.
				continue

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwA[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), PARAMHYDRO.exponent_flow) * sumSw/ti.math.sqrt(gradSw)

		# Boundary case
		else:
			tSw = ti.max(hsw.Zw(Z,hw,i,j) -  PARAMHYDRO.hydro_slope_bc_val, 1e-6)/GRID.dx if PARAMHYDRO.hydro_slope_bc_mode == 0 else PARAMHYDRO.hydro_slope_bc_val
			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwA[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), PARAMHYDRO.exponent_flow) * ti.math.sqrt(tSw)

@ti.kernel
def _compute_Qw_drape(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template() ):
	'''
	EXPERIMENTAL: testing some dynamic draping
	Variant of Qw that applies a draping algorithm to avoid the creation of local minimas
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

				# Done with processing this particular neighbour

			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			
			# Not sure I still need that
			if(gradSw == 0):
				continue

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), PARAMHYDRO.exponent_flow) * sumSw/ti.math.sqrt(gradSw)

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
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), PARAMHYDRO.exponent_flow) * ti.math.sqrt(tSw)


@ti.kernel
def _compute_hw_drape(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), constrains:ti.template(), BCs:ti.template() ):
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

	#Reinitialising the constrains to their default values
	for i,j in Z:

		constrains[i,j,0] = -1e6
		constrains[i,j,1] = 1e6

	# Constant of draping, manages
	rat = ti.f32(0.45)

	# First determining the draping constrains
	for i,j in Z:

		# Local elevation
		tZw = hsw.Zw_drape(Z,hw,i,j)

		# Ignoring nodes outletting (they still get a constrain[i,j,1] given by donors)
		if gridfuncs.can_out(i,j,BCs):
			continue

		# keeping track of the mini elev
		Zdkmin = tZw
		kmin = 5

		# Traversing Neighbours
		for k in range(4):

			# Getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# not a neighbour
			if(ir == -1):
				continue

			# Note that Zw_drape() returns the absolute hw + Z, even if hw is < 0
			tZwr = hsw.Zw_drape(Z,hw,ir,jr)

			# Registering new steepest neighbour if needed
			if(tZwr < Zdkmin):
				Zdkmin = tZwr
				kmin = k

		# Local minima ? I skip
		if kmin == 5:
			continue

		# Row col of the receiver
		ir,jr = gridfuncs.neighbours(i,j,kmin, BCs)

		# Calculating constrains
		if(gridfuncs.can_receive(ir,jr,BCs)):

			# Checking if I have enough space between my nodes
			if(tZw - Zdkmin >= PARAMHYDRO.mini_drape_step):

				# if I do, I calculate the min/max function of a slightly unbalanced mean between our node and its receiver
				constrains[i,j,0] = ti.math.max(rat * (hsw.Zw_drape(Z,hw,ir,jr)) + (1 - rat) * (tZw), hsw.Zw_drape(Z,hw,ir,jr) + PARAMHYDRO.mini_drape_step)

				# Propagating to tneighbour's max possible elevation
				ti.atomic_min(constrains[ir,jr,1] , ti.math.min( (1-rat) * hsw.Zw_drape(Z,hw,ir,jr) + rat * tZw, tZw - PARAMHYDRO.mini_drape_step) )

			else:
				# If we do not have enough space
				# i "lock" the node for now. It will be eventually unlocked by reorganisation of neighbours
				constrains[i,j,0] = tZw

				# Propagating to tneighbour's max possible elevation
				ti.atomic_min(constrains[ir,jr,1] , hsw.Zw_drape(Z,hw,ir,jr))


	# Draped increment 
	for i,j in Z:
		

		# Updating local discharge to new time step
		QwA[i,j] = QwB[i,j]
				
		# Applying the increment whatsoever YOLO lol
		hw[i,j] = hw[i,j] + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy) 
		
		# Converting constrains into flow depth clamper
		constrains[i,j,0] -= Z[i,j]
		constrains[i,j,1] -= Z[i,j]


		# Clamp
		if(hw[i,j] < constrains[i,j,0]):
			hw[i,j] = constrains[i,j,0]
		elif(hw[i,j] > constrains[i,j,1]):
			hw[i,j] = constrains[i,j,1]



@ti.kernel
def _compute_hw_drape_th(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), D4dir:ti.template(), constrains:ti.template(), BCs:ti.template(), threshold:ti.f32 ):
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

	rat = 0.45

	for i,j in Z:

		tZw = hsw.Zw_drape(Z,hw,i,j) # + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy)
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

			# if(tZwr == tZw):
			# 	print('HAPPENS:',i,j,'vs',ir,jr, tZwr,tZw)

			if(tZwr < Zdkmin):
				Zdkmin = tZwr
				kmin = k

		if kmin==5:
			continue

		ir,jr = gridfuncs.neighbours(i,j,kmin, BCs)


		D4dir[i,j] = kmin

		# rat = 0.45 + ti.random() * 0.04
		# rat = 1.

		constrains[i,j,0] = (rat * (hsw.Zw_drape(Z,hw,ir,jr)) + (1 - rat) * (tZw))
		# constrains[i,j,0] = hsw.Zw_drape(Z,hw,ir,jr)


		ti.atomic_min(constrains[ir,jr,1] , ((1-rat) * (hsw.Zw_drape(Z,hw,ir,jr)) + rat * (tZw)))

		# ti.atomic_min(constrains[ir,jr,1] , tZw)

		# constrains[i,j,0] -= Z[i,j]
		# constrains[i,j,1] -= Z[i,j]

	# for i,j in Z:
	# 	if(constrains[i,j,0] == constrains[i,j,1]):
	# 		ir,jr = gridfuncs.neighbours(i,j,D4dir[i,j], BCs)
	# 		print('dsafksjkfgksdjflk::', hsw.Zw_drape(Z,hw,i,j), hsw.Zw_drape(Z,hw,ir,jr), rat * hsw.Zw_drape(Z,hw,ir,jr) + (1 - rat) *  hsw.Zw_drape(Z,hw,i,j))


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
		# hw[i,j] = ti.math.max(0.,hw[i,j] + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy) )

		if(QwA[i,j] <= 0 or abs(1 - (QwC[i,j]/QwA[i,j])) < threshold ):
			continue

		hw[i,j] = hw[i,j] + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy) 

		# if(constrains[i,j,0] == 1e6):
		# 	constrains[i,j,0] = Z[i,j]
		# if(constrains[i,j,1] == 1e6):
		# 	constrains[i,j,1] = Z[i,j]

		# if(constrains[i,j,0] == constrains[i,j,1]):
		# 	print("HAPPENS")
		
		constrains[i,j,0] -= Z[i,j]
		constrains[i,j,1] -= Z[i,j]


		
		if(gridfuncs.can_out(i,j,BCs) == False):
			if(hw[i,j] < constrains[i,j,0]):
				hw[i,j] = constrains[i,j,0]
			elif(hw[i,j] > constrains[i,j,1]):
				hw[i,j] = constrains[i,j,1]
	

	# # DEBUG DEBUG DEBUG
	# # Traversing nodes
	# Nhap = 0
	# NNhap = 0
	# for i,j in Z:
	# 	if(D4dir[i,j] == 5 or gridfuncs.can_out(i,j,BCs)):
	# 		continue

	# 	ir,jr = gridfuncs.neighbours(i,j,D4dir[i,j], BCs)

	# 	if(hsw.Zw_drape(Z,hw,i,j) == hsw.Zw_drape(Z,hw,ir,jr) and (ir != i or jr != j)):
	# 		print('gabul', hsw.Zw_drape(Z,hw,i,j) ,'vs',hsw.Zw_drape(Z,hw,ir,jr), 'constrains node were', constrains[i,j,0] + Z[i,j], constrains[i,j,1] + Z[i,j], 'and rec', constrains[ir,jr,0] + Z[ir,jr], constrains[ir,jr,1] + Z[ir,jr]  )

	# # 	if( constrains[i,j,0] + Z[i,j] ==  constrains[ir,jr,0] + Z[ir,jr] or constrains[i,j,0] + Z[i,j] ==  constrains[ir,jr,1] + Z[ir,jr]  or constrains[i,j,1] + Z[i,j] ==  constrains[ir,jr,0] + Z[ir,jr] or constrains[i,j,1] + Z[i,j] ==  constrains[ir,jr,1] + Z[ir,jr]):
	# # 		Nhap +=1
	# # 	else:
	# # 		NNhap +=1
	# # print('error:', Nhap/(NNhap + Nhap))


@ti.kernel
def _compute_hw_drape_CFL(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), D4dir:ti.template(), constrains:ti.template(), BCs:ti.template(), alpha:ti.f32 ):
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

	rat = 0.45

	for i,j in Z:

		tZw = hsw.Zw_drape(Z,hw,i,j) # + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy)
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

			# if(tZwr == tZw):
			# 	print('HAPPENS:',i,j,'vs',ir,jr, tZwr,tZw)

			if(tZwr < Zdkmin):
				Zdkmin = tZwr
				kmin = k

		if kmin==5:
			continue

		ir,jr = gridfuncs.neighbours(i,j,kmin, BCs)


		D4dir[i,j] = kmin

		# rat = 0.45 + ti.random() * 0.04
		# rat = 1.

		constrains[i,j,0] = (rat * (hsw.Zw_drape(Z,hw,ir,jr)) + (1 - rat) * (tZw))
		# constrains[i,j,0] = hsw.Zw_drape(Z,hw,ir,jr)


		ti.atomic_min(constrains[ir,jr,1] , ((1-rat) * (hsw.Zw_drape(Z,hw,ir,jr)) + rat * (tZw)))

		# ti.atomic_min(constrains[ir,jr,1] , tZw)

		# constrains[i,j,0] -= Z[i,j]
		# constrains[i,j,1] -= Z[i,j]

	# for i,j in Z:
	# 	if(constrains[i,j,0] == constrains[i,j,1]):
	# 		ir,jr = gridfuncs.neighbours(i,j,D4dir[i,j], BCs)
	# 		print('dsafksjkfgksdjflk::', hsw.Zw_drape(Z,hw,i,j), hsw.Zw_drape(Z,hw,ir,jr), rat * hsw.Zw_drape(Z,hw,ir,jr) + (1 - rat) *  hsw.Zw_drape(Z,hw,i,j))


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
		# hw[i,j] = ti.math.max(0.,hw[i,j] + (QwA[i,j] - QwC[i,j]) * PARAMHYDRO.dt_hydro/(GRID.dx*GRID.dy) )
		tdt = ti.math.max(PARAMHYDRO.dt_hydro, alpha *GRID.dx/(QwA[i,j]/(ti.math.max(1e-4, hw[i,j]))) )		
		hw[i,j] = hw[i,j] + (QwA[i,j] - QwC[i,j]) * tdt/(GRID.dx*GRID.dy) 

		# if(constrains[i,j,0] == 1e6):
		# 	constrains[i,j,0] = Z[i,j]
		# if(constrains[i,j,1] == 1e6):
		# 	constrains[i,j,1] = Z[i,j]

		# if(constrains[i,j,0] == constrains[i,j,1]):
		# 	print("HAPPENS")
		
		constrains[i,j,0] -= Z[i,j]
		constrains[i,j,1] -= Z[i,j]


		
		if(gridfuncs.can_out(i,j,BCs) == False):
			if(hw[i,j] < constrains[i,j,0]):
				hw[i,j] = constrains[i,j,0]
			elif(hw[i,j] > constrains[i,j,1]):
				hw[i,j] = constrains[i,j,1]
	

	# # DEBUG DEBUG DEBUG
	# # Traversing nodes
	# Nhap = 0
	# NNhap = 0
	# for i,j in Z:
	# 	if(D4dir[i,j] == 5 or gridfuncs.can_out(i,j,BCs)):
	# 		continue

	# 	ir,jr = gridfuncs.neighbours(i,j,D4dir[i,j], BCs)

	# 	if(hsw.Zw_drape(Z,hw,i,j) == hsw.Zw_drape(Z,hw,ir,jr) and (ir != i or jr != j)):
	# 		print('gabul', hsw.Zw_drape(Z,hw,i,j) ,'vs',hsw.Zw_drape(Z,hw,ir,jr), 'constrains node were', constrains[i,j,0] + Z[i,j], constrains[i,j,1] + Z[i,j], 'and rec', constrains[ir,jr,0] + Z[ir,jr], constrains[ir,jr,1] + Z[ir,jr]  )

	# # 	if( constrains[i,j,0] + Z[i,j] ==  constrains[ir,jr,0] + Z[ir,jr] or constrains[i,j,0] + Z[i,j] ==  constrains[ir,jr,1] + Z[ir,jr]  or constrains[i,j,1] + Z[i,j] ==  constrains[ir,jr,0] + Z[ir,jr] or constrains[i,j,1] + Z[i,j] ==  constrains[ir,jr,1] + Z[ir,jr]):
	# # 		Nhap +=1
	# # 	else:
	# # 		NNhap +=1
	# # print('error:', Nhap/(NNhap + Nhap))



@ti.kernel
def _compute_link_based(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template() ):
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
		QwB[i,j] = 0.
		QwC[i,j] = 0.
		# QwD[i,j] = 0.


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

		if(gridfuncs.can_give(i,j,BCs) == False and gridfuncs.can_out(i,j,BCs) == False):
			continue


		# None boundary case
		if(gridfuncs.can_out(i,j,BCs) == False):


			while(True):
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
				
				if(sumSw > 0):
					break

				hw[i,j] += 1e-4

				# Done with processing this particular neighbour

			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			
			# Not sure I still need that
			if(gradSw == 0):
				continue

			# Transferring flow to neighbours
			for k in range(4):

				# local neighbours
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)
				
				# checking if neighbours
				if(ir == -1):
					continue

				if (Sws[k] <= 0 ):
					continue
				
				hw_eff = ti.max(hsw.Zw(Z,hw,i,j),hsw.Zw(Z,hw,ir,jr)) - ti.max(Z[i,j],Z[ir,jr])
				# hw_eff = ti.max(0,hw[i,j])
				tQw = GRID.dx * ti.math.pow(hw_eff,PARAMHYDRO.exponent_flow)/PARAMHYDRO.manning * Sws[k]/ti.math.sqrt(sumSw)
				QwC[i,j] += tQw

				# Transferring prop to the hydraulic slope
				ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])

		# Boundary case
		else:
			tSw = ti.max(hsw.Zw(Z,hw,i,j) -  PARAMHYDRO.hydro_slope_bc_val, 1e-6)/GRID.dx if PARAMHYDRO.hydro_slope_bc_mode == 0 else PARAMHYDRO.hydro_slope_bc_val
			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), PARAMHYDRO.exponent_flow) * ti.math.sqrt(tSw)



@ti.kernel
def _archive_compute_Qw(Z:ti.template(), hw:ti.template(), QwA:ti.template(), QwB:ti.template(), QwC:ti.template(), BCs:ti.template(), flowdir:ti.template() ):
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

		if(gridfuncs.can_give(i,j,BCs) == False and gridfuncs.can_out(i,j,BCs) == False):
			continue

		thw = 0.

		# Tracking if I am in a local minima 
		LM = False

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

					thw = ti.math.max(thw, ti.max(hsw.Zw(Z,hw,i,j),hsw.Zw(Z,hw,ir,jr)) - ti.max(Z[i,j],Z[ir,jr]))

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
					# I am in a local minima
					LM = True
					hw[i,j] += 1e-4

				## And if I added like a metre and it did not slolve it, I stop for that node
				if(lockcheck > 10000 or (LM and PARAMHYDRO.use_original_dir_for_LM)):
					break

			# Calculating local norm for the gradient
			# The condition manages the boundary conditions
			gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
			
			# Not sure I still need that
			if(gradSw == 0):
				continue

			if(PARAMHYDRO.use_heffmax == False):
				thw = ti.math.max(0., hw[i,j])

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(thw, PARAMHYDRO.exponent_flow) * sumSw/ti.math.sqrt(gradSw)

			# If I am in a local minima
			# either I propel the drainage area along a predefined railway to ensure its escape
			# or I just ignore it
			if(LM):
				# this option ensures the drainage of LM through the original rail
				if(PARAMHYDRO.use_original_dir_for_LM):
					# Flow dir == 5 is no flow
					if(flowdir[i,j] != 5):
						# That section ensures that a stochastic number of receivers are traversed to flush the local minima
						# And avoid ping-pong or localisation based biases
						ii,jj = i,j
						ir,jr = i,j
						first = 0
						# Receivers are poped out at least once, and then has a probability of 0.5 to continue
						while(flowdir[ir,jr] != 5 and (first<=PARAMHYDRO.LM_pathforcer)):
							ii,jj = ir,jr
							first += 1
							ir,jr = gridfuncs.neighbours(ii, jj, flowdir[ii,jj], BCs)
							# ti.atomic_add(QwB[ir,jr], QwA[i,j])
						ti.atomic_add(QwB[ir,jr], QwA[i,j])
				else:
					# In that case, I keep everything
					ti.atomic_add(QwB[i,j], QwA[i,j])

				continue

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
			QwC[i,j] = GRID.dx/PARAMHYDRO.manning * ti.math.pow(ti.max(0.,hw[i,j]), PARAMHYDRO.exponent_flow) * ti.math.sqrt(tSw)




###########################################################
# Other sets of functions to help wiith hydrodynamcis
###########################################################

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
	elif(PARAMHYDRO.flowmode == FlowMode.static_drape):
		compute_hw = _compute_hw_drape
		compute_Qw = _compute_Qw_drape
	elif(PARAMHYDRO.flowmode == FlowMode.static_link):
		compute_hw = _compute_hw
		compute_Qw = _compute_link_based
	else:
		raise NotImplementedError('FLOWMODEW Not implemented yet')
