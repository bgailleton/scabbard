'''
Sets of function to compute local minimas
EXPERIMENTAL, no warranty it evens do what it is supposed to do yet

B.G. - 29/04/2024
'''

import taichi as ti
import numpy as np
from enum import Enum
import scabbard.utils as scaut 
from scabbard.riverdale.rd_grid import GRID
import scabbard.riverdale.rd_grid as gridfuncs
import scabbard.riverdale.rd_helper_surfw as srf
import dagger as dag
from scipy.ndimage import gaussian_filter



# (is cpu)
def priority_flood(rd, Zw = True, step = 1e-3):
	'''
	Applies priority flood's Algorithm in D4 topology to the riverdale's elevation.
	Note that it takes into account the boundary conditions.

	Arguments:
		- rd: An initialised RiverDale's instance
		- Zw: If True, fills the water surface, if False, the topography
	Returns:
		- Nothing, edits hte model in place and retransfer the data to GPU
	Authors:
		- B.G. (last modification: 30/05/2024)
	'''

	tZw = rd.Z.to_numpy() + rd.hw.to_numpy() if Zw else rd.Z.to_numpy()
	gcpp = rd.get_GridCPP()
	dag._PriorityFlood_D4_f32(tZw,gcpp,rd.BCs.to_numpy(),step) if rd.param.dtype_float == ti.f32 else dag._PriorityFlood_D4_f64(tZw,gcpp,rd.BCs.to_numpy(),step)
	if(Zw):
		rd.hw.from_numpy(tZw - rd.Z.to_numpy())
	else:
		rd.Z.from_numpy(tZw)

def smooth_hw(rd, strength = 3, recompute_QwA = True):
	'''
	Applies priority flood's Algorithm in D4 topology to the riverdale's elevation.
	Note that it takes into account the boundary conditions.

	Arguments:
		- rd: An initialised RiverDale's instance
		- Zw: If True, fills the water surface, if False, the topography
	Returns:
		- Nothing, edits hte model in place and retransfer the data to GPU
	Authors:
		- B.G. (last modification: 30/05/2024)
	'''
	import scabbard.riverdale.rd_hydrodynamics as rdhy

	tZw = rd.Z.to_numpy() + rd.hw.to_numpy()
	mask = rd.hw.to_numpy() <= 0
	tZw = gaussian_filter(tZw, strength)
	thw = tZw - rd.Z.to_numpy()
	thw[mask] = 0.
	rd.hw.from_numpy(thw)
	if (recompute_QwA):
		rdhy._compute_QwA_from_Zw(rd.Z, rd.hw, rd.QwA, rd.BCs )



@ti.kernel
def N_conv(hw:ti.template(), QwA:ti.template(), QwC:ti.template(), BCs:ti.template(), threshold_QwA:ti.f32, threshold_hw:ti.f32, threshold_conv:ti.f32) -> (ti.i32, ti.i32):

	N:ti.i32 = 0
	NC:ti.i32 = 0

	for i,j in hw:

		if gridfuncs.is_active(i,j,BCs) == False:
			continue

		if QwA[i,j] < threshold_QwA or threshold_hw > hw[i,j]:
			continue

		N += 1

		if(abs(QwC[i,j]/QwA[i,j] - 1) < threshold_conv):
			NC += 1

	return NC,N

def compute_convergence(rd, threshold_QwA = 1e-4, threshold_hw = 1e-2, threshold_conv = 0.05, min_N = 100):
	NC,N = N_conv(rd.hw, rd.QwA, rd.QwC, rd.BCs, threshold_QwA, threshold_hw, threshold_conv)
	print(NC,N)
	if N < min_N:
		return 0
	else:
		return NC/N


@ti.kernel
def constrain_drape(Z:ti.template(), hw:ti.template(), constrains:ti.template(), BCs:ti.template()):
	'''
	TODO to deprecate probably
	'''

	# So, let's try to constrain the min/max height to add without creating local minimas

	for i,j in Z:

		

		tZw = srf.Zw_drape(Z,hw,i,j)

		constrains[i,j,0] = tZw
		constrains[i,j,1] = tZw

		if gridfuncs.is_active(i,j,BCs) == False:
			continue

		# Traversing Neighbours
		for k in range(4):

			# Getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			if(ir == -1):
				continue

			constrains[i,j,0] = ti.math.min(constrains[i,j,0], srf.Zw_drape(Z,hw,ir,jr))
			constrains[i,j,1] = ti.math.max(constrains[i,j,1], srf.Zw_drape(Z,hw,ir,jr))

		constrains[i,j,0] -= Z[i,j]
		constrains[i,j,1] -= Z[i,j]









@ti.kernel
def label_pit(Z:ti.template(), LM:ti.template(), BCs:ti.template()):

	for i,j in Z:

		if(gridfuncs.is_active(i,j,BCs) == False):
			LM[i,j] = 0
			continue

		isLM = True
		# Traversing Neighbours
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue
			if(Z[i,j] > Z[ir,jr]):
				isLM = False
				break
		lab = i * GRID.nx + j
		if(isLM):
			LM[i,j] = lab 


@ti.kernel
def count_pits(Z:ti.template(), BCs:ti.template())  -> ti.i32:

	count = 0

	for i,j in Z:

		if(gridfuncs.is_active(i,j,BCs) == False):
			continue

		isLM = True
		# Traversing Neighbours
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue
			if(Z[i,j] > Z[ir,jr]):
				isLM = False
				break

		if(isLM):
			count += 1
	return count

@ti.func
def Zw(Z,hw,i,j):
	return Z[i,j] + hw[i,j]

@ti.kernel
def count_pits_Zw(Z:ti.template(), hw:ti.template(), BCs:ti.template())  -> ti.i32:

	count = 0

	for i,j in Z:

		if(gridfuncs.is_active(i,j,BCs) == False or gridfuncs.can_out(i,j,BCs)):
			continue

		isLM = True
		# Traversing Neighbours
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			# if(srf.Zw_drape(Z,hw,i,j) == srf.Zw_drape(Z,hw,ir,jr)):
			# 	print('happens')

			if( srf.Zw_drape(Z,hw,i,j) > srf.Zw_drape(Z,hw,ir,jr) ) :
				isLM = False
				break

		if(isLM):
			count += 1

	return count


@ti.kernel
def label_pits_Zw(Z:ti.template(), hw:ti.template(), LM:ti.template(), BCs:ti.template())  -> ti.i32:

	count = 0

	for i,j in Z:
		LM[i,j] = 0

		if(gridfuncs.is_active(i,j,BCs) == False or gridfuncs.can_out(i,j,BCs)):
			continue

		isLM = True
		# Traversing Neighbours
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			# if(srf.Zw_drape(Z,hw,i,j) == srf.Zw_drape(Z,hw,ir,jr)):
			# 	print('happens')

			if( srf.Zw_drape(Z,hw,i,j) > srf.Zw_drape(Z,hw,ir,jr) ) :
				isLM = False
				break

		if(isLM):
			LM[i,j] = 1
			count += 1

	return count




#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################



# Bellow are experimental stuff


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################







@ti.kernel
def label_depression(LM:ti.template(), D4dir:ti.template(), BCs:ti.template(), count:ti.template()):

	count[None] = 0

	for i,j in LM:
		
		if(LM[i,j] == 0):
			continue

		ti = i
		tj = j

		lab = LM[i,j]

		while(True):
			tti = ti
			ttj = tj

			for k in range(4):
				# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
				ir,jr = gridfuncs.neighbours(ti, tj, k, BCs)

				# if not a neighbours, by convention is < 0 and I pass
				if(ir == -1 or LM[ir,jr]):
					continue
				
				ik = 3 - k
				if(D4dir[ir,jr] != ik):
					continue

				LM[ir,jr] = lab
				ti,tj = ir,jr
				count[None] += 1
				# break

			if(ti == tti and tj == ttj):
				break
@ti.kernel
def compute_depressions_min_pass(Z:ti.template(), LM:ti.template(), Z_LM_min:ti.template(), BCs:ti.template()):
	'''
	'''

	for i,j in Z:
		if(LM[i,j] == 0):
			continue

		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue
			if(LM[ir,jr] == LM[i,j]):
				continue

			ti.atomic_min(Z_LM_min[LM[i,j]], max(Z[i,j],Z[ir,jr]) + 1e-3);

@ti.kernel
def raise_depressions(Z:ti.template(), LM:ti.template(), Z_LM_min:ti.template(), BCs:ti.template()):
	'''
	'''

	for i,j in Z:
		if(LM[i,j] == 0):
			continue

		Z[i,j] = max(Z_LM_min[LM[i,j]], Z[i,j])

@ti.kernel
def detect_flats(Z:ti.template(), D4dir:ti.template(), isflat:ti.template(), BCs:ti.template()):

	for i,j in Z:
		
		allflat = True
		
		if(gridfuncs.is_active(i,j,BCs) == False):
			continue

		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			if(Z[ir,jr] != Z[i,j]):
				allflat = False
				break

		isflat[i,j] = allflat


	for i,j in Z:
		
		if(gridfuncs.is_active(i,j,BCs) == False or isflat[i,j] == False):
			continue

		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			if(Z[ir,jr] == Z[i,j]):
				isflat[ir,jr] = True

	for i,j in Z:
		if(isflat[i,j]):
			D4dir[i,j] = 5

@ti.kernel
def label_flats(Z:ti.template(), LM:ti.template(), isflat:ti.template(), BCs:ti.template(), count:ti.template()):

	label = 0
	for i,j in Z:

		if(is_flat[i,j] == False):
			continue


		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue











@ti.kernel
def resolve_flats(Z:ti.template(), D4dir:ti.template(), isflat:ti.template(), BCs:ti.template(), count:ti.template()):
	'''
		Does not work yet
	'''

	count[None] = 0
	for i,j in Z:
		if(isflat[i,j] == False):
			continue

		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)
			if(ir == -1):
				continue

			if(Z[ir,jr] < Z[i,j] and D4dir[i,j] == 5):
				D4dir[i,j] = k
				count[None] += 1

			# elif(D4dir[ir,jr] == 5 and D4dir[i,j] != 5):
			# 	D4dir[ir,jr] = 3 - k
			# 	count[None] += 1
			
			elif(D4dir[ir,jr] != 5 and D4dir[i,j] == 5 and isflat[ir,jr] and D4dir[ir,jr] != 3 - k):
				D4dir[i,j] = k
				count[None] += 1





@ti.kernel
def label_oedges(edges:ti.template(), BCs:ti.template()):

	for i,j in edges:
		if(gridfuncs.can_out(i,j,BCs)):
			edges[i,j] = 1
		else:
			edges[i,j] = 0


@ti.kernel
def label_edges(edges:ti.template(), D4dir:ti.template(), BCs:ti.template(), count:ti.template()):

	count[None] = 0

	for i,j in edges:
		
		if(edges[i,j] == 0):
			continue

		ti = i
		tj = j

		while(True):
			tti = ti
			ttj = tj

			for k in range(4):
				# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
				ir,jr = gridfuncs.neighbours(ti, tj, k, BCs)

				# if not a neighbours, by convention is < 0 and I pass
				if(ir == -1 or edges[ir,jr]):
					continue
				
				ik = 3 - k
				if(D4dir[ir,jr] != ik):
					continue

				edges[ir,jr] = 1
				ti,tj = ir,jr
				count[None] += 1
				# break

			if(ti == tti and tj == ttj):
				break

@ti.kernel
def reconstruct(edges:ti.template(), D4dir:ti.template(), Z:ti.template(), BCs:ti.template(), count:ti.template() ) :
	'''
	DOES NOT WORK
	'''

	count[None] = 0

	for i,j in edges:
		
		if(edges[i,j] > 0):
			continue

		tZ = Z[i,j]
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i, j, k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1 or edges[ir,jr] == 0):
				continue

			if(Z[ir,jr] > Z[i,j] and Z[ir,jr] <= tZ):
				tZ = Z[ir,jr] 
		
		if(tZ != Z[i,j]):
			count[None] += 1
			edges[i,j] = 1
			Z[i,j] = tZ + 1e-3
			
			# break


	

@ti.kernel
def archive_1_label_edges(edges:ti.template(), D4dir:ti.template(), BCs:ti.template(), count:ti.template()):
	'''
	this experimentation on label edges propagates from neighbours to edges
	'''
	count[None] = 0
	# inei = ti.math.ivec4(3,2,1,0)

	for i,j in edges:
		
		if(edges[i,j] == 0):
			continue

		ti = i
		tj = j

		# while(True):

		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(ti, tj, k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1 or edges[ir,jr]):
				continue
			
			ik = 3 - k
			if(D4dir[ir,jr] != ik):
				continue

			edges[ir,jr] = 1
			count[None] += 1

	




@ti.kernel
def forward_scan( marker : ti.template(), mask:ti.template(), BCs:ti.template(), count : ti.template() ):
	

	for i,j in marker:

		if(gridfuncs.is_active(i,j,BCs) == False):
			continue
		# Compute the maximum of the marker at the current pixel and all
		# of its previously visisted neighbors
		max_height = marker[i,j]
		# Traversing Neighbours
		for k in range(2):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			if max_height > marker[ir,jr]:
				max_height = max_height  
			else:
				max_height = marker[ir,jr]

		# Set the marker at the current pixel to the minimum of the
		# maximum height of the neighborhood and the mask at the current
		# pixel.
		z = max_height  

		if max_height >= mask[i,j]:
			# print(i,j)
			z = mask[i,j]

		if (z != marker[i,j]):
			print(i,j)

			# Increment count only if we change the current pixel
			count[None] += 1; # Should be atomic
			marker[i,j] = z;



@ti.kernel
def backward_scan( marker : ti.template(), mask:ti.template(), BCs:ti.template(), count : ti.template() ):
	# Note that the loop decreases. p must have a signed type for this
	# to work correctly.
	for i,j in marker:

		if(gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Compute the maximum of the marker at the current pixel and all
		# of its previously visisted neighbors
		max_height = marker[i,j]
		# Traversing Neighbours
		for tk in range(2):
			k = tk + 1
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue
			max_height = max_height if max_height > marker[ir,jr] else marker[ir,jr]
	

	# Set the marker at the current pixel to the minimum of the
	# maximum height of the neighborhood and the mask at the current
	# pixel.

		z = max_height if max_height < mask[i,j] else mask[i,j]
		if (z != marker[i,j]):
			# Increment count only if we change the current pixel
			count[None] += 1;
			marker[i,j] = z;




@ti.kernel
def prefillsink(Z:ti.template(), Z_fill:ti.template(), BCs:ti.template()):
	'''
	Mimics the fillsinks function of libtopotoolbox
	Does not work yet
	'''

	for i,j in Z:
		Z[i,j] *= -1 
		if(gridfuncs.is_active(i,j,BCs)):
			Z_fill[i,j] = Z[i,j]

@ti.kernel
def postfillsink(Z:ti.template(), Z_fill:ti.template(), BCs:ti.template()):
	'''
	Mimics the fillsinks function of libtopotoolbox
	Does not work yet
	'''

	for i,j in Z:
		Z[i,j] *= -1 
		Z_fill[i,j] *= -1


def fillsinks(Z,Z_fill,BCs):

	count = ti.field(dtype = ti.f32, shape = ())

	prefillsink(Z,Z_fill,BCs)

	count[None] = 1

	while(count[None] != 0):
		print("run", count[None])
		count[None] = 0
		forward_scan( Z_fill, Z, BCs, count)
		backward_scan( Z_fill, Z, BCs, count)
		print("run", count[None])

	postfillsink(Z,Z_fill,BCs)




# Early tests

@ti.func
def Z_draped(i:ti.int32, j:ti.int32, Z:ti.template(), hw:ti.template()):
	return Z[i,j] + hw[i,j]


@ti.kernel
def pre_drape(Z:ti.template(), hw:ti.template(), recConstrain: ti.template(), donorConstrain:ti.template(), BCs:ti.template(), count:ti.template()):
	'''
		stuff
	'''
	#
	count[None] = 0

	for i,j in Z:
		recConstrain[i,j] = 1e9
		donorConstrain[i,j] = -1e9

	for i,j in Z:

		if(gridfuncs.is_active(i,j,BCs) == False):
			continue

		checker = True

		for k in range(4):

			# Getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# If not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			# computing the minimum height of the donors in order not to invert 
			# Note that I am trying to get the minimum elevation of the donor
			if (Z_draped(ir,jr,Z,hw) > Z_draped(i,j,Z,hw) and (Z_draped(ir,jr,Z,hw) < donorConstrain[i,j] or donorConstrain[i,j] == -1e9)):
				donorConstrain[i,j] = Z_draped(ir,jr,Z,hw)
				checker = False

			# Computing the maximum drop in elevation, in this case this is the lowest receiver
			elif (Z_draped(ir,jr,Z,hw) < Z_draped(i,j,Z,hw) and (Z_draped(ir,jr,Z,hw) < recConstrain[i,j] or recConstrain[i,j] == 1e9)):
				recConstrain[i,j] = Z_draped(ir,jr,Z,hw)
				checker = False
		
		if(checker):
			count[None] += 1


	# Done





@ti.kernel
def drape_iteration(Z:ti.template(), hw:ti.template(), recConstrain: ti.template(), donorConstrain:ti.template(), BCs:ti.template(), count:ti.template()):
	'''
	stuff 2
	'''
	count[None] = 0
	for i,j in Z:

		if(gridfuncs.is_active(i,j,BCs) == False):
			continue

		temp1 = Z_draped(i,j,Z,hw)
		temp1 = ti.min(ti.max(temp1, recConstrain[i,j]), donorConstrain[i,j])
		# ti.math.clamp(temp1, recConstrain[i,j], donorConstrain[i,j])
		temp2 = Z_draped(i,j,Z,hw)
		if(temp1 != temp2):
			count[None] += 1












































































# end of file 