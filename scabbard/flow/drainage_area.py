'''
Routines to calculate drainage area and other similar stuff
'''

import numba as nb
import numpy as np
import scabbard.steenbok as st
from scabbard.flow import SFGraph
import scabbard as scb


@nb.njit()
def _drainage_area_sfg(Stack,Sreceivers, dx = 1., BCs = None):

	A = np.zeros_like(Sreceivers, dtype = np.float32)

	for i in range(Stack.shape[0]):
		node = Stack[Stack.shape[0] - 1 - i]
		rec = Sreceivers[node]

		if(node == rec):
			continue

		A[node] += dx * dx
		A[rec] += A[node]

	return A



def drainage_area(input_data):

	if(isinstance(input_data, SFGraph) == False):
		raise RuntimeError('drainage area WIP, so far requires SFGraph object to calculate')

	return _drainage_area_sfg(input_data.Stack, input_data.Sreceivers, dx = input_data.dx).reshape(input_data.ny,input_data.nx)




@nb.njit()
def _propagate_sfg(Stack,Sreceivers, values, dx = 1., BCs = None):

	A = np.zeros_like(Sreceivers, dtype = np.float32)

	for i in range(Stack.shape[0]):
		node = Stack[Stack.shape[0] - 1 - i]
		rec = Sreceivers[node]

		if(node == rec):
			continue

		A[node] += values[node]
		A[rec] += A[node]

	return A


@nb.njit()
def _propagate_mfd_propS(Z, Stack, values, nx, ny, dx = 1., BCs = None, D4 = True):

	A = np.zeros_like(Z, dtype = np.float32)

	for i in range(Stack.shape[0]):
		node = Stack[Stack.shape[0] - 1 - i]

		
		if(scb.ste.can_out_flat(node,BCs) or scb.ste.is_active_flat(node,BCs) == False):
			continue
		
		A[node] += values[node]

		tot_S = 0.
		nrec = 0
		for k in range(4 if D4 else 8):
			nnode = scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny) if D4 else scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny)
			if(nnode == -1):
				continue

			if(Z[node] <= Z[nnode]):
				continue


			tot_S += (Z[node] - Z[nnode])/(scb.ste.dx_from_k_D4(dx,k) if D4 else scb.ste.dx_from_k_D8(dx,k))
			nrec += 1

		if(tot_S == 0.):
			continue

		for k in range(4 if D4 else 8):
			nnode = scb.ste.neighbours_D4_flat(node,k,BCs,nx,ny) if D4 else scb.ste.neighbours_D8_flat(node,k,BCs,nx,ny)

			if(nnode == -1):
				continue

			if(Z[node] <= Z[nnode]):
				continue

			A[nnode] += A[node] * (Z[node] - Z[nnode])/(scb.ste.dx_from_k_D4(dx,k) if D4 else scb.ste.dx_from_k_D8(dx,k))/tot_S

	return A



def propagate(input_data, input_values, method = 'sfd', BCs = None, D4 = True, fill_LM = False, step_fill = 1e-3):
	'''
	Propagates values with the flow, following a drainage-area-like path

	Arguments:
		- input_data: the topographic data, can be a RegularRasterGrid or for single flow path propagation a SFGraph
		- input_values: a 2D array of input data shape containing the data (e.g. precipitations, sources, ...)
		- method: str, 'sfd' for single flow direction, 'mfd_S' for multiple flow partitionned prop. to the slope, other to come


	'''
		

	if(method.lower() == 'sfd'):
		
		if(isinstance(input_data, scb.raster.RegularRasterGrid)):
			tBCs = scb.flow.get_normal_BCs(input_data) if BCs is None else BCs
			stg = scb.flow.SFGraph(tZ, BCs = tBCs, D4 = True, dx = input_data.geo.dx, backend = 'ttb', fill_LM = fill_LM, step_fill=step_fill)
		elif(isinstance(input_data, SFGraph)):
			stg = input_data
		else:
			raise RuntimeError('drainage area using SFD method requires a SFGraph object or a RegularRasterGrid as input_data')

		return _propagate_sfg(input_data.Stack, input_data.Sreceivers, input_values.ravel(), dx = input_data.dx).reshape(input_data.ny,input_data.nx)

	elif (method.lower() == 'mfd_s'):
		if(isinstance(input_data, scb.raster.RegularRasterGrid)):
			tBCs = scb.flow.get_normal_BCs(input_data) if BCs is None else BCs
		else:
			raise RuntimeError('drainage area using MFD methods requires a RegularRasterGrid as input_data')

		if(fill_LM):
			Stack = np.zeros_like(input_data.Z.ravel(), dtype = np.uint64)
			scb.ttb.graphflood.funcdict['priority_flood_TO']( input_data.Z.ravel(), Stack, BCs.ravel(), input_data.dims, not D4, step_fill)

		else:
			Stack = np.argsort(input_data.Z.ravel()).astype(np.uint64)

		return _propagate_mfd_propS(input_data.Z.ravel(), Stack.ravel(), input_values.ravel(), input_data.geo.nx, input_data.geo.ny, dx = input_data.geo.dx, BCs = tBCs.ravel(), D4 = D4).reshape(input_data.geo.ny,input_data.geo.nx)

	else:
		raise RuntimeError('Supported methods so far: sfd or mfd_S')























































# end of file
