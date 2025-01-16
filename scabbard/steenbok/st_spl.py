'''
Steenbock's implementation of stream power models
Multiple styles, I'll start with functional

B.G - 07/2024 - Acigné

'''

import numba as nb
import numpy as np
from enum import Enum
import scabbard as scb


@nb.njit()
def _impl_spl_SFD_single(
		Stack, # TO order
		Sreceivers, # Steepest Receiver
		Sdx,   # distance to Srec
		Z,     # topography (edited in place)
		A,     # drainage area, or Discharge
		nx,    # Ncols
		ny,	   # Nrows
		dx,    # Spatial step
		BCs,   # Bouncary codes
		K,     # Erodability
		m,     # Area exponent 
		n,     # Slope exponent
		dt,    # Time Step
	):
	
	# Traversing landscapes from downstream to upstream
	for node in Stack:

		# Ignoring outnodes
		if node == Sreceivers[node] or scb.ste.can_out_flat(node,BCs) or scb.ste.is_active_flat(node,BCs) == False:
			continue

		dzdx = (Z[node] - Z[Sreceivers[node]])/Sdx[node]

		if(dzdx<=0):
			Z[node] = Z[Sreceivers[node]] + 1e-4
			dzdx = 1e-4/dx

		factor = K * dt * (A[node])**m / (Sdx[node]**n);

		ielevation = Z[node];
		irec_elevation = Z[Sreceivers[node]];
		elevation_k = ielevation;
		elevation_prev = Z[node] + 500;
		tolerance = 1e-5;

		while (abs(elevation_k - elevation_prev) > tolerance) :
			elevation_prev = elevation_k
			slope = max(elevation_k - irec_elevation, 1e-6)
			diff = (elevation_k - ielevation + factor * slope**n) / (1. + factor * n * slope**(n - 1))
			elevation_k -= diff;
		
		Z[node] = elevation_k

@nb.njit()
def _impl_spl_SFD_variable(
		Stack, # TO order
		Sreceivers, # Steepest Receiver
		Sdx,   # distance to Srec
		Z,     # topography (edited in place)
		A,     # drainage area, or Discharge
		nx,    # Ncols
		ny,	   # Nrows
		dx,    # Spatial step
		BCs,   # Bouncary codes
		K,     # Erodability
		m,     # Area exponent 
		n,     # Slope exponent
		dt,    # Time Step
	):
	
	# Traversing landscapes from downstream to upstream
	for node in Stack:

		# Ignoring ounodes
		if node == Sreceivers[node] or scb.ste.can_out_flat(node,BCs) or scb.ste.is_active_flat(node,BCs) == False:
			continue

		dzdx = (Z[node] - Z[Sreceivers[node]])/Sdx[node]

		if(dzdx<=0):
			Z[node] = Z[Sreceivers[node]] + 1e-4
			dzdx = 1e-4/dx

		factor = K[node] * dt * (A[node])**m / (Sdx[node]**n);

		ielevation = Z[node];
		irec_elevation = Z[Sreceivers[node]];
		elevation_k = ielevation;
		elevation_prev = Z[node] + 500;
		tolerance = 1e-5;

		while (abs(elevation_k - elevation_prev) > tolerance) :
			elevation_prev = elevation_k
			slope = max(elevation_k - irec_elevation, 1e-6)
			diff = (elevation_k - ielevation + factor * slope**n) / (1. + factor * n * slope**(n - 1))
			elevation_k -= diff;
		
		Z[node] = elevation_k

def run_SPL_on_topo(
		dem,     # topography (edited in place)
		BCs = None,   # Bouncary codes
		graph = None,
		K = 1e-5,     # Erodability
		m = 0.45,     # Area exponent 
		n = 1.11,     # Slope exponent
		dt = 1e3,    # Time Step	
	):
	dx,nx,ny = dem.geo.dxnxny

	if BCs is None:
		BCs = scb.ut.normal_BCs_from_shape(nx, ny, out_code = 3)

	if(graph is None):
		graph = scb.flow.SFGraph(dem, BCs = BCs, D4 = False, dx = 1., backend = 'ttb', fill_LM = True, step_fill = 1e-3)
	else:
		graph.update(dem, BCs = BCs, fill_LM = True, step_fill = 1e-3)

	A = scb.flow.drainage_area(graph)


	if(isinstance(K,np.ndarray)):
		_impl_spl_SFD_variable(graph.Stack, graph.Sreceivers, graph.Sdx, dem.Z.ravel(), A.ravel(), nx, ny, dx, BCs.ravel(), K.ravel(), m, n, dt)
	else:
		_impl_spl_SFD_single(graph.Stack, graph.Sreceivers, graph.Sdx, dem.Z.ravel(), A.ravel(), nx, ny, dx, BCs.ravel(), K, m, n, dt)
	

