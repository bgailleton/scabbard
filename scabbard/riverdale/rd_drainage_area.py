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
import scabbard.riverdale.rd_LM as lmfuncs
import scabbard.riverdale.rd_helper_surfw as hsw
import scabbard.riverdale.rd_utils as rut



@ti.kernel
def compute_Sreceivers_Zw(receivers:ti.template(), Z:ti.template(), hw:ti.template(), BCs:ti.template()):
	'''
	Compute the single flow receivers for each and every nodes
	Arguments:
		- receivers: 2D field holding the flat index of the steepest receivers
		- Z: topographic elevation
		- hw: flow depth
		- BCs: boundary condition codes
	Return:
		- Nothing by fill receivers in place
	Authors:
		- B.G. (last modifications: 06/2024)
	'''
	# Traversing each nodes
	for i,j in Z:

		# Assigning the receiver to itself by default
		receivers[i,j] = i * GRID.nx + j

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SS = 0.
		lowest_higher_Z = 0.
	
		# Traversing Neighbours
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			# Local hydraulic slope
			tS = hsw.Zw(Z,hw,i,j) - hsw.Zw(Z,hw,ir,jr)
			tS /= GRID.dx

			# If < 0, neighbour is a donor and I am not interested
			if(tS <= 0):
				continue

			# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
			if(tS > SS):
				receivers[i,j] = ir * GRID.nx + jr
				SS = tS

@ti.kernel
def compute_Sreceivers_Zw_rand(receivers:ti.template(), Z:ti.template(), hw:ti.template(), BCs:ti.template()):
	'''
	Compute the single flow receivers for each and every nodes
	This variant adds a stochastic component to the selection of the single flow receiver
	Arguments:
		- receivers: 2D field holding the flat index of the steepest receivers
		- Z: topographic elevation
		- hw: flow depth
		- BCs: boundary condition codes
	Return:
		- Nothing by fill receivers in place
	Authors:
		- B.G. (last modifications: 06/2024)
	'''
	# Traversing each nodes
	for i,j in Z:

		# Assigning the receiver to itself by default
		receivers[i,j] = i * GRID.nx + j

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SS = 0.
		lowest_higher_Z = 0.
	
		# Traversing Neighbours
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			# Local hydraulic slope
			tS = hsw.Zw(Z,hw,i,j) - hsw.Zw(Z,hw,ir,jr)
			tS /= GRID.dx
			# Randomiser yolo
			tS *= ti.random()

			# If < 0, neighbour is a donor and I am not interested
			if(tS <= 0):
				continue

			# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
			if(tS > SS):
				receivers[i,j] = ir * GRID.nx + jr
				SS = tS


@ti.kernel
def increment_DAD4(receivers:ti.template(), ptrrec:ti.template(), DA:ti.template()):
	'''
	Function incrementing the drainage area calculation by one step
	Based on a pointerto receiver method NOT OPTIMISED YET!
	Arguments:
		- receivers: 2D field of single flow receivers for each nodes
		- ptrrec: 2D field of pointer to the current receiver to atomic_add
		- DA: 2D field of drainage area
	Return:
		- nothing but edit receivers in place
	Authors:
		- B.G (last modifictations: 06/2024)
	'''

	# Main loop
	for i,j in receivers:
		
		# generating the flat index
		idx = i * GRID.nx + j
		
		# If outlet continue
		if(receivers[i,j] == idx):
			continue
		
		# row col indinces of the current pointer
		ip = ptrrec[i,j] // GRID.nx
		jp = ptrrec[i,j] %  GRID.nx

		# Flat index of the receiver to the current pointer
		newrec = receivers[ip,jp]

		# row col of the receivers of the pointer
		ni = newrec // GRID.nx
		nj = newrec %  GRID.nx

		# If is an outlet or local minima I stop here
		if(receivers[ni,nj] == newrec):
			continue

		# If it is not, I increment drainage area and ...
		ti.atomic_add(DA[ni,nj], GRID.dx * GRID.dy)
		# ... updating the pointer 
		ptrrec[i,j] = newrec

@ti.kernel
def increment_QWD4(receivers:ti.template(), ptrrec:ti.template(), QW:ti.template(), Pf:ti.template()):
	'''
	Function incrementing the drainage area calculation by one step
	Based on a pointerto receiver method NOT OPTIMISED YET!
	Arguments:
		- receivers: 2D field of single flow receivers for each nodes
		- ptrrec: 2D field of pointer to the current receiver to atomic_add
		- QW: 2D field of drainage area weighted by precipitations
		- Pf: 2D field of Precipitation inputs
	Return:
		- nothing but edit receivers in place
	Authors:
		- B.G (last modifictations: 06/2024)
	'''

	# Main loop
	for i,j in receivers:
		
		# generating the flat index
		idx = i * GRID.nx + j
		
		# If outlet continue
		if(receivers[i,j] == idx):
			continue
		
		# row col indinces of the current pointer
		ip = ptrrec[i,j] // GRID.nx
		jp = ptrrec[i,j] %  GRID.nx

		# Flat index of the receiver to the current pointer
		newrec = receivers[ip,jp]

		# row col of the receivers of the pointer
		ni = newrec // GRID.nx
		nj = newrec %  GRID.nx

		# If is an outlet or local minima I stop here
		if(receivers[ni,nj] == newrec):
			continue

		# If it is not, I increment drainage area and ...
		ti.atomic_add(QW[ni,nj], Pf[i,j])

		# print('adding', Pf[i,j])
		# ... updating the pointer 
		ptrrec[i,j] = newrec



def compute_drainage_area_D4(rd, fill = True, N = 'auto', random_rec = False, Precipitations = None):
	'''
	Compute drainage area in D4 direction on GPU. 
	Warning, this is optimised for flexibility to automate the computation fo DA for numerical analysis, not an optimised building block within a model (that's WIP)
	workflow: (optionally) fills the topo, then precalculates single flow receivers and finally transfer flow in a cascading bucekts fashion
	
	Arguments:
		- rd: the initialised RiverDale object
		- fill: if True, runs priority flood to get rid of local minimas
		- N: N iterations for the increment of DA (WIP to get that auto)
		- random_rec: if True, adds a stochastic component in the selection of receivers in the first step of computations
		- Precipitations: if 2D numpy array, accumulates P * drainage area
	Returns:
		- Numpy array of drainage area
	Authors:
		- B.G. (last modifications: 06/2024)
	'''	

	# Optional filling operation
	if(fill):
		lmfuncs.priority_flood(rd)

	# checking if precipitations are needed
	prec = False if Precipitations is None else True
	
	# Fetching fields for receivers and ptr to receivers	
	rec, ptrrec = rd.query_temporary_fields(2,dtype = ti.i32)
	# Fetching field for DA
	if(not prec):
		DA, = rd.query_temporary_fields(1, dtype = ti.f32) 
		# Initialising drainage area to local values
		DA.fill(GRID.dx * GRID.dy) 

	else:
		DA,Pfield = rd.query_temporary_fields(2, dtype = ti.f32) 
		Pfield.from_numpy(Precipitations * GRID.dx * GRID.dy)
		rut.A_equals_B(DA,Pfield)

	# Precomputing the receivers - with or without the stochastic component 
	compute_Sreceivers_Zw(rec, rd.Z, rd.hw, rd.BCs) if random_rec == False else compute_Sreceivers_Zw_rand(rec, rd.Z, rd.hw, rd.BCs)
	
	# initialising the ptr to receivers to the first receivers of each node
	rut.A_equals_B(ptrrec,rec)
	
	# If mode is 'auto', I set N to very high values
	if isinstance(N, str):
		N = 1000000
		auton = True
	else:
		auton = False
	
	#running N iterations (=/- debug logs)
	for i in range(N):
		
		if(i%1000 == 0 and i>0 and auton):
			cop = DA.to_numpy()
			increment_DAD4(rec,ptrrec,DA) if prec == False else increment_QWD4(rec, ptrrec, DA, Pfield)
			if(np.sum(np.abs(cop - DA.to_numpy())) == 0.):
				break
		else:
			increment_DAD4(rec,ptrrec,DA) if prec == False else increment_QWD4(rec, ptrrec, DA, Pfield)


	# Done, returning the results
	return DA.to_numpy()















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
	

		