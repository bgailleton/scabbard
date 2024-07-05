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
import scabbard.riverdale.rd_hydrodynamics as hydrofunc



class MorphoMode(Enum):
	'''
	Enumeration of the different boundary condition types possible
	'''	
	fbal = 0,
	MPM_like = 1,
	fbal_links = 2


@scaut.singleton
class MorphoParams:
	'''
		Internal singleton class holding all the compile time constant parameters for the hydro simulations 
		Not for users
	'''
	def __init__(self):

		# Time for the morphodynamics modelling		
		self.dt_morpho = 1e-3
		# What kind of erosion
		self.morphomode = MorphoMode.fbal
		# gravitational constant
		self.GRAVITY = 9.81
		# Water density
		self.rho_water = 1000
		# Sediment density
		self.rho_sediment = 2600
		# Gravitational erosion coeff
		self.k_z = 1.
		self.k_h = 1.
		# Fluvial erosion coeff
		self.k_erosion = 1e-5
		# Fluvial exponent
		self.alpha_erosion = 1.5
		# Grainsize
		self.D = 4e-3 
		# Critical shear strass
		self.tau_c = 4
		# transport length
		self.transport_length = 4
		# water viscosity
		self.viscosity = 15.e-6

		self.update_morpho_at_input_points = False

		self.threshold_Qw_ratio = 0.


PARAMMORPHO = MorphoParams()
PARAMHYDRO = hydrofunc.PARAMHYDRO



@ti.func
def ZPsi(Z:ti.template(), hw: ti.template(), k_h:ti.template(), k_z:ti.template(), i:ti.i32, j:ti.i32) -> ti.f32:
	'''
	Internal helping function returning the Partitioning surface for sediments (Psi)
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- k_h: the internal coefficient to the shear stress component in the MPM-like entrainment formulation
		- k_z: the internal coefficient to the gravitational component in the MPM-like entrainment formulation
		- i,j: the row col indices
	Returns:
		- the hydraulic surface
	Authors:
		- B.G. (last modification 30/04/2024)
	'''
	rho_ratio = (PARAMMORPHO.rho_sediment - PARAMMORPHO.rho_water)/PARAMMORPHO.rho_water
	A = k_z * Z[i,j] + 2 * k_h * hw[i,j]/(PARAMMORPHO.D * rho_ratio)* hsw.Zw(Z,hw,i,j)
	B = k_z + 2 * k_h * hw[i,j]/(PARAMMORPHO.D * rho_ratio)
	return A/B


@ti.func
def SPsi(Z: ti.template(), hw: ti.template(), k_h:ti.template(), k_z:ti.template(), i:ti.template(), j:ti.template(), ir:ti.template(), jr:ti.template())->ti.f32:
	'''
	Internal helping function returning the hydrayulic slope
	Arguments:
		- Z: a 2D field of topographic elevation
		- hw: a 2D field of flow depth
		- i,j: the row col indices
		- ir,jr: the row col indices of the receivers node
	Returns:
		- the hydraulic surface
	Authors:
		- B.G. (last modification 20/05/2024)
	'''
	return (ZPsi(Z, hw, k_h, k_z, i,j) - ZPsi(Z, hw, k_h, k_z, ir,jr))/GRID.dx


@ti.kernel
def initiate_step(QsB: ti.template()):
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
	for i,j in QsB:
		QsB[i,j] = 0.


@ti.kernel
def input_discharge_sediment_points(input_rows: ti.template(), input_cols:ti.template(), input_values:ti.template(), QsA: ti.template(), QsB: ti.template(), BCs:ti.template()):
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
		QsA[input_rows[i],input_cols[i]] += input_values[i]
		QsB[input_rows[i],input_cols[i]] += input_values[i]

@ti.kernel
def _compute_Qs_fbal(Z:ti.template(), hw:ti.template(), QsA:ti.template(), QsB:ti.template(), QsC:ti.template(), QwA:ti.template(), QwC:ti.template(), BCs:ti.template() ):
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

	DENSITY_R = (PARAMMORPHO.rho_sediment - PARAMMORPHO.rho_water) * PARAMMORPHO.GRAVITY
	K_EROSION = (PARAMMORPHO.rho_sediment - PARAMMORPHO.rho_water)/PARAMMORPHO.rho_water * PARAMMORPHO.GRAVITY/PARAMMORPHO.viscosity * ti.math.pow(PARAMMORPHO.D,2)


	# Traversing each nodes
	for i,j in Z:

		# Case of normal, internal node
		if(gridfuncs.can_out(i,j,BCs) == False):

			# I'll store the hydraulic slopes in this vector
			SPsis = ti.math.vec4(0.,0.,0.,0.)

			# I'll need the sum of the hydraulic slopes in the positive directions
			sumSpsi = 0.

			# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
			## Hydraulic slope
			SSwx = 0.
			SSwy = 0.

			## Topographic slope
			SSZx = 0.
			SSZy = 0.

			## Psi slope
			SSPsix = 0.
			SSPsiy = 0.


			thw = 0.
			
			# Traversing Neighbours
			for k in range(4):
				# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)

				# if not a neighbours, by convention is < 0 and I pass
				if(ir == -1): 
					continue

				if(gridfuncs.can_receive(ir,jr,BCs) == False):
					continue

				# Local topographic slope
				tS = ti.max(hsw.Sw(Z,hw,i,j,ir,jr),0.)

				# Local hydraulic slope
				tSz = ti.max(hsw.Sz(Z,i,j,ir,jr), 0.)

				# Local Partitionning slope
				tSPsi = ti.max(SPsi(Z, hw, PARAMMORPHO.k_h, PARAMMORPHO.k_z, i, j, ir, jr), 0.)

				# # If < 0, neighbour is a donor and I am not interested
				# if(tS <= 0):
				# 	continue

				thw = ti.math.max(thw, ti.max(hsw.Zw(Z,hw,i,j),hsw.Zw(Z,hw,ir,jr)) - ti.max(Z[i,j],Z[ir,jr]))

				# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
				if(k == 0 or k == 3):
					if(tS > SSwy):
						SSwy = tS
					if(tSz > SSZy):
						SSZy = tSz
					if(tSPsi > SSPsiy):
						SSPsiy = tSPsi
				else:
					if(tS > SSwx):
						SSwx = tS
					if(tSz > SSZx):
						SSZx = tSz
					if(tSPsi > SSPsix):
						SSPsix = tSPsi

				# Registering local partitioning slope
				SPsis[k] = tSPsi
				# Summing it to global
				sumSpsi += tSPsi

			# Done with processing this particular neighbour

			# Local minima management (cheap but works)
			## If I have no downward slope, I increase the elevation by a bit
			if(sumSpsi == 0.):
				QsC[i,j] = -1
				continue


			# Calculating local norms for the gradients
			gradSw = ti.math.sqrt(SSwx*SSwx + SSwy*SSwy)
			gradSPsi = ti.math.sqrt(SSPsix*SSPsix + SSPsiy*SSPsiy)
			gradSz = ti.math.sqrt(SSZx*SSZx + SSZy*SSZy)

			# Not sure I still need that
			if(gradSw <= 0 or gradSPsi == 0):
				continue

			if(PARAMHYDRO.use_heffmax == False):
				thw = ti.math.max(0., hw[i,j])

			local_erosion_rate = ti.max(PARAMMORPHO.k_erosion * K_EROSION * 
				ti.pow( PARAMMORPHO.k_z * gradSz + 
					2 * PARAMMORPHO.k_h * PARAMMORPHO.rho_water/(PARAMMORPHO.rho_sediment - PARAMMORPHO.rho_water) 
					/ PARAMMORPHO.D * thw * gradSw - PARAMMORPHO.tau_c,
				PARAMMORPHO.alpha_erosion),
			0.)
			# print(gradSPsi)
			# if(local_erosion_rate > 0):
			# 	print(local_erosion_rate)

			CA = GRID.dx*GRID.dx
			beta = gradSPsi/(PARAMMORPHO.transport_length * sumSpsi)

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QsC[i,j] = (QsA[i,j] + local_erosion_rate * CA)/(1 + CA * beta)

			# If the node cannot give and can only receive, I pass this node
			if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.can_receive(i,j,BCs) == False):
				QsC[i,j] = -1
			else:
				# Transferring flow to neighbours
				for k in range(4):

					# local neighbours
					ir,jr = gridfuncs.neighbours(i,j,k, BCs)
					
					# checking if neighbours
					if(ir == -1):
						continue
					
					# Transferring prop to the hydraulic slope
					ti.atomic_add(QsB[ir,jr], SPsis[k]/sumSpsi * QsC[i,j])
		else:

			gradSz = ti.max(hsw.Zw(Z,hw,i,j) -  PARAMHYDRO.hydro_slope_bc_val, 1e-6)/GRID.dx if PARAMHYDRO.hydro_slope_bc_mode == 0 else PARAMHYDRO.hydro_slope_bc_val

			local_erosion_rate = ti.max(PARAMMORPHO.k_erosion * K_EROSION * ti.pow( gradSz - PARAMMORPHO.tau_c, PARAMMORPHO.alpha_erosion),0.)

			CA = GRID.dx*GRID.dx

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			# QsC[i,j] = (QsA[i,j] + local_erosion_rate * CA)/(1 + CA * PARAMMORPHO.transport_length / GRID.dx)
			QsC[i,j] = QsA[i,j]

	for i,j in Z:
		# Updating local discharge to new time step
		# QsA[i,j] = QsB[i,j]

		adamantic = (PARAMMORPHO.threshold_Qw_ratio != 0)
		if(adamantic):
			if(QwA[i,j] > 0):
				if(abs(1 - QwC[i,j]/QwA[i,j]) < PARAMMORPHO.threshold_Qw_ratio ):
					adamantic = False

		if(QsC[i,j] < 0 or adamantic):
			QsC[i,j] = QsA[i,j]

@ti.kernel
def _compute_Qs_fablink(Z:ti.template(), hw:ti.template(), QsA:ti.template(), QsB:ti.template(), QsC:ti.template(), QwA:ti.template(), QwC:ti.template(), BCs:ti.template() ):
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

	DENSITY_R = (PARAMMORPHO.rho_sediment - PARAMMORPHO.rho_water) * PARAMMORPHO.GRAVITY
	K_EROSION = (PARAMMORPHO.rho_sediment - PARAMMORPHO.rho_water)/PARAMMORPHO.rho_water * PARAMMORPHO.GRAVITY/PARAMMORPHO.viscosity * ti.math.pow(PARAMMORPHO.D,2)


	# Traversing each nodes
	for i,j in Z:

		# Case of normal, internal node
		if(gridfuncs.can_out(i,j,BCs) == False):

			# I'll store the hydraulic slopes in this vector
			SPsis = ti.math.vec4(0.,0.,0.,0.)
			heffs = ti.math.vec4(0.,0.,0.,0.)

			# I'll need the sum of the hydraulic slopes in the positive directions
			sumSpsi = 0.

			# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
			## Hydraulic slope
			SSwx = 0.
			SSwy = 0.

			## Topographic slope
			SSZx = 0.
			SSZy = 0.

			## Psi slope
			SSPsix = 0.
			SSPsiy = 0.

			
			# Traversing Neighbours
			for k in range(4):
				# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)

				# if not a neighbours, by convention is < 0 and I pass
				if(ir == -1): 
					continue

				if(gridfuncs.can_receive(ir,jr,BCs) == False):
					continue

				# Local topographic slope
				tS = ti.max(hsw.Sw(Z,hw,i,j,ir,jr),0.)

				# Local hydraulic slope
				tSz = ti.max(hsw.Sz(Z,i,j,ir,jr), 0.)

				# Local Partitionning slope
				tSPsi = ti.max(SPsi(Z, hw, PARAMMORPHO.k_h, PARAMMORPHO.k_z, i, j, ir, jr), 0.)

				heffs[k] = ti.max(hsw.Zw(Z,hw,i,j),hsw.Zw(Z,hw,ir,jr)) - ti.max(Z[i,j],Z[ir,jr])

				# # If < 0, neighbour is a donor and I am not interested
				# if(tS <= 0):
				# 	continue

				# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
				if(k == 0 or k == 3):
					if(tS > SSwy):
						SSwy = tS
					if(tSz > SSZy):
						SSZy = tSz
					if(tSPsi > SSPsiy):
						SSPsiy = tSPsi
				else:
					if(tS > SSwx):
						SSwx = tS
					if(tSz > SSZx):
						SSZx = tSz
					if(tSPsi > SSPsix):
						SSPsix = tSPsi

				# Registering local partitioning slope
				SPsis[k] = tSPsi
				# Summing it to global
				sumSpsi += tSPsi

			# Done with processing this particular neighbour

			# Local minima management (cheap but works)
			## If I have no downward slope, I increase the elevation by a bit
			if(sumSpsi == 0.):
				QsC[i,j] = -1
				continue


			# Calculating local norms for the gradients
			gradSw = ti.math.sqrt(SSwx*SSwx + SSwy*SSwy)
			gradSPsi = ti.math.sqrt(SSPsix*SSPsix + SSPsiy*SSPsiy)
			gradSz = ti.math.sqrt(SSZx*SSZx + SSZy*SSZy)

			# Not sure I still need that
			if(gradSw <= 0 or gradSPsi == 0):
				continue

			local_erosion_rate = 0.
			for k in range(4):
				local_erosion_rate += heffs[k] * SPsis[k]/sumSpsi


			local_erosion_rate = ti.max(PARAMMORPHO.k_erosion * K_EROSION * 
				ti.pow( PARAMMORPHO.k_z * gradSz + 
					2. * PARAMMORPHO.k_h * PARAMMORPHO.rho_water/(PARAMMORPHO.rho_sediment - PARAMMORPHO.rho_water) 
					/ PARAMMORPHO.D * local_erosion_rate * gradSw - PARAMMORPHO.tau_c,
				PARAMMORPHO.alpha_erosion),
			0.)
			# print(gradSPsi)
			# if(local_erosion_rate > 0):
			# 	print(local_erosion_rate)

			CA = GRID.dx*GRID.dx
			beta = gradSPsi/(PARAMMORPHO.transport_length * sumSpsi)

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			QsC[i,j] = (QsA[i,j] + local_erosion_rate * CA)/(1 + CA * beta)

			# If the node cannot give and can only receive, I pass this node
			if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.can_receive(i,j,BCs) == False):
				QsC[i,j] = -1
			else:
				# Transferring flow to neighbours
				for k in range(4):

					# local neighbours
					ir,jr = gridfuncs.neighbours(i,j,k, BCs)
					
					# checking if neighbours
					if(ir == -1):
						continue
					
					# Transferring prop to the hydraulic slope
					ti.atomic_add(QsB[ir,jr], SPsis[k]/sumSpsi * QsC[i,j])
		else:

			gradSz = ti.max(hsw.Zw(Z,hw,i,j) -  PARAMHYDRO.hydro_slope_bc_val, 1e-6)/GRID.dx if PARAMHYDRO.hydro_slope_bc_mode == 0 else PARAMHYDRO.hydro_slope_bc_val

			local_erosion_rate = ti.max(PARAMMORPHO.k_erosion * K_EROSION * ti.pow( gradSz - PARAMMORPHO.tau_c, PARAMMORPHO.alpha_erosion),0.)

			CA = GRID.dx*GRID.dx

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			# QsC[i,j] = (QsA[i,j] + local_erosion_rate * CA)/(1 + CA * PARAMMORPHO.transport_length / GRID.dx)
			QsC[i,j] = QsA[i,j]

	for i,j in Z:
		# Updating local discharge to new time step
		# QsA[i,j] = QsB[i,j]

		adamantic = (PARAMMORPHO.threshold_Qw_ratio != 0)
		if(adamantic):
			if(QwA[i,j] > 0):
				if(abs(1 - QwC[i,j]/QwA[i,j]) < PARAMMORPHO.threshold_Qw_ratio ):
					adamantic = False

		if(QsC[i,j] < 0 or adamantic):
			QsC[i,j] = QsA[i,j]


@ti.kernel
def _compute_Qs_MPM_like(Z:ti.template(), hw:ti.template(), QsA:ti.template(), QsB:ti.template(), QsC:ti.template(), BCs:ti.template() ):
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

		# Case of normal, internal node
		if(gridfuncs.can_out(i,j,BCs) == False):

			# I'll store the hydraulic slopes in this vector
			Sws = ti.math.vec4(0.,0.,0.,0.)

			# I'll need the sum of the hydraulic slopes in the positive directions
			sumSlopes = 0.

			# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
			## Hydraulic slope
			SSwx = 0.
			SSwy = 0.
			
			# Traversing Neighbours
			for k in range(4):
				# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
				ir,jr = gridfuncs.neighbours(i,j,k, BCs)

				# if not a neighbours, by convention is < 0 and I pass
				if(ir == -1):
					continue

				# Local topographic slope
				tSw = ti.max(hsw.Sw(Z,hw,i,j,ir,jr),0.)

				# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
				if(k == 0 or k == 3):
					if(tSw > SSwy):
						SSwy = tSw
				else:
					if(tSw > SSwx):
						SSwx = tSw

				# Registering local partitioning slope
				Sws[k] = tSw
				# Summing it to global
				sumSlopes += tSw

			# Done with processing this particular neighbour

			# Local minima management (cheap but works)
			## If I have no downward slope, I increase the elevation by a bit
			if(sumSlopes == 0.):
				QsC[i,j] = -1
				continue


			# Calculating local norms for the gradients
			gradSw = ti.math.sqrt(SSwx*SSwx + SSwy*SSwy)

			# Not sure I still need that
			if(gradSw <= 0):
				continue

			local_erosion_rate = ti.max(PARAMMORPHO.k_erosion * ti.math.pow(PARAMMORPHO.rho_water * PARAMMORPHO.GRAVITY * max(hw[i,j],0.) * gradSw - PARAMMORPHO.tau_c, PARAMMORPHO.alpha_erosion), 0.)

			CA = GRID.dx*GRID.dx

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			tK = (1. / PARAMMORPHO.transport_length);
			edotpsy = (local_erosion_rate) / tK;
			C1 = QsA[i,j] / GRID.dx - edotpsy;
			QsC[i,j] = GRID.dx * (edotpsy + C1 * ti.math.exp(-GRID.dx * tK));

			# If the node cannot give and can only receive, I pass this node
			if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.can_receive(i,j,BCs) == False):
				QsC[i,j] = -1
			else:
				# Transferring flow to neighbours
				for k in range(4):

					# local neighbours
					ir,jr = gridfuncs.neighbours(i,j,k, BCs)
					
					# checking if neighbours
					if(ir == -1):
						continue
					
					# Transferring prop to the hydraulic slope
					ti.atomic_add(QsB[ir,jr], Sws[k]/sumSlopes * QsC[i,j])
		else:

			gradSw = ti.max(hsw.Zw(Z,hw,i,j) -  PARAMHYDRO.hydro_slope_bc_val, 1e-6)/GRID.dx if PARAMHYDRO.hydro_slope_bc_mode == 0 else PARAMHYDRO.hydro_slope_bc_val

			local_erosion_rate = ti.max(PARAMMORPHO.k_erosion * ti.math.pow(PARAMMORPHO.rho_water * PARAMMORPHO.GRAVITY * max(hw[i,j],0.) * gradSw - PARAMMORPHO.tau_c, PARAMMORPHO.alpha_erosion), 0.)

			CA = GRID.dx*GRID.dx

			# Calculating local discharge: manning's equations for velocity and u*h*W to get Q
			# QsC[i,j] = (QsA[i,j] + local_erosion_rate * CA)/(1 + CA * PARAMMORPHO.transport_length / GRID.dx)
			QsC[i,j] = QsA[i,j]

	for i,j in Z:
		# Updating local discharge to new time step
		# QsA[i,j] = QsB[i,j]

		if(QsC[i,j] < 0):
			QsC[i,j] = QsA[i,j]
			









@ti.kernel
def _compute_hs(Z:ti.template(), hw:ti.template(), QsA:ti.template(), QsB:ti.template(), QsC:ti.template(), BCs:ti.template() ):
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

		if(gridfuncs.can_receive(i,j,BCs) == False and PARAMMORPHO.update_morpho_at_input_points == False):
			continue
		
		# Only where nodes are active (i.e. flow cannot leave and can traverse)
		# if(gridfuncs.is_active(i,j,BCs) == False):
		# 	continue

		dz = (QsA[i,j] - QsC[i,j]) * PARAMMORPHO.dt_morpho/(GRID.dx*GRID.dy)

		# Updating flow depth (cannot be < 0)
		Z[i,j] += dz
		hw[i,j] = hw[i,j] - dz
		QsA[i,j] = QsB[i,j]




########################################################################
########################################################################
############### EXPOSED API ############################################
########################################################################
########################################################################

# Note that the input functions are already exposed

compute_hs = None
compute_Qs = None


def set_morpho_CC():
	'''
	Expose the right API function of the comnpile time parameters in PARAMHYDRO.
	Need to be called after setting hte different parameters in the singleton PARAMHYDRO
	Returns:
		- Nothing, update global function refs
	Authors:
		- B.G. (last modification 03/05/2024)
	'''	


	# # fectch neighbours placeholder
	global compute_hs
	global compute_Qs

	
	# Feed it
	if(PARAMMORPHO.morphomode == MorphoMode.fbal):
		compute_hs = _compute_hs
		compute_Qs = _compute_Qs_fbal
	elif(PARAMMORPHO.morphomode == MorphoMode.MPM_like):
		compute_hs = _compute_hs
		compute_Qs = _compute_Qs_MPM_like
	elif(PARAMMORPHO.morphomode == MorphoMode.fbal_links):
		compute_hs = _compute_hs
		compute_Qs = _compute_Qs_fablink
	else:
		raise NotImplementedError('PARAMMORPHO Not implemented yet')
