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
import scabbard.riverdale.rd_drainage_area as rda
import scabbard.riverdale.rd_helper_surfw as hsw
import scabbard.riverdale.rd_LM as rlm



@ti.kernel
def _compute_hydraulic_gradient(Sw:ti.template(),Z:ti.template(), hw:ti.template(), BCs:ti.template()):
	'''
	Taichi kernel to compute the hydraulic gradient (as topographic analysis)
	
	Arguments:
		- Sw: field of hydrualic gradient to be edited in place
		- Z: field of topographic surface
		- hw: field of flow depth
		- BCs: field of boundary conditions
	
	Returns:
		- nothing, but calculates Sw

	Authors:
		- B.G. (06/2024)
	'''

	for i,j in Z:
		if(gridfuncs.is_active(i,j,BCs) and gridfuncs.can_receive(i,j,BCs)):
			Sw[i,j] = hsw.hydraulic_gradient_value(Z, hw, BCs, i, j) 
		else:
			Sw[i,j] = 0.

def compute_hydraulic_gradient(rd, fill_with_PF = False):
	'''
	Computes hydraulic gradient as the norm of Z + hw:
	
	This function is optimised for analysis, not for internal calculation

	Arguments:
		- rd: the riverdale object, initialised
		- fill_with_PF: fill the local minimas with water using an adapted priority flood (Barnes, 2014) if activated
	Returns:
		- a numpy array of hydraulic gradient
	Authors:
		- B.G (last modifications: 06/2024)

	'''

	if(fill_with_PF):
		rlm.priority_flood(rd)

	output, = rd.query_temporary_fields(1,dtype = ti.f32)
	_compute_hydraulic_gradient(output, rd.Z,rd.hw,rd.BCs)
	return output.to_numpy()

@ti.kernel
def _compute_shear_stress(shear_stress:ti.template(),Z:ti.template(), hw:ti.template(), BCs:ti.template(), rho:ti.f32, g:ti.f32):
	'''
	Taichi kernel to compute the shear stress (as topographic analysis)
	
	Arguments:
		- shear_stress: field of shear stress to be edited in place
		- Z: field of topographic surface
		- hw: field of flow depth
		- BCs: field of boundary conditions
		- rho: density of water kg/m^3 (should be 1000 in "normal" cases)
		- g: gravitational acceleration in m/s^2 (9.8 in "nornal" cases)
	
	Returns:
		- nothing, but calculates the shear stress

	Authors:
		- B.G. (06/2024)
	'''



	for i,j in Z:
		if(gridfuncs.is_active(i,j,BCs) and gridfuncs.can_receive(i,j,BCs)):
			shear_stress[i,j] = hsw.hydraulic_gradient_value(Z, hw, BCs, i, j) * ti.math.max(hw[i,j],0.) * rho * g
		else:
			shear_stress[i,j] = 0.

def compute_shear_stress(rd, fill_with_PF = False):
	'''
	Computes the shear stress for every nodes with water for a given riverdale environment:
	τ = ρ g h_w S_w
	with ρ the water density, g the gravitational acceleration, h_w the flow depth and S_w the hydraulic slope

	THis function is optimised for analysis, not for internal calculation

	Arguments:
		- rd: the riverdale object, initialised
		- fill_with_PF: fill the local minimas with water using an adapted priority flood (Barnes, 2014) if activated
	Returns:
		- a numpy array of shear stress
	Authors:
		- B.G (last modifications: 06/2024)

	'''

	if(fill_with_PF):
		rlm.priority_flood(rd)

	output, = rd.query_temporary_fields(1,dtype = ti.f32)
	_compute_shear_stress(output, rd.Z,rd.hw,rd.BCs, 1000., 9.81)
	return output.to_numpy()

def compute_flow_velocity(rd, use_Qwin = True):
	'''
	Computes the flow velocity from a RiverDale object
	'''
	u = rd.QwA.to_numpy() if use_Qwin else rd.QwC.to_numpy()
	hw = rd.hw.to_numpy()
	mask0 = hw<=1e-6
	u[mask0] = 0
	u[~mask0] /= hw[~mask0]*GRID.dx

	return u


def compute_discharge_per_unit_width(rd, use_Qwin = True):
	'''
	Computes the discharge per unit width from a RiverDale object
	'''
	qw = rd.QwA.to_numpy() if use_Qwin else rd.QwC.to_numpy()
	qw /= GRID.dx
	return qw


def compute_effective_drainage_area(rd, use_Qwin = True, custom_r = None):
	'''
	Computes the effective drainage area a_r (s) as of described in e.g. Bernard et al. (2023) 

	Arguments:
		- rd: the initialised riverdale's object
		- use_Qwin: use Qwin if True, else Qwout
		- custom_r: effective drainage area is equal to Discharge per unit width divided by average precipitation rates, if this parameters is not None it replaces the one calculated from rd.param
	'''

	if(rd.param.need_input_Qw and custom_r is None):
		raise NotImplementedError("It seems that your Riverdale's Object has input points for Discharge. I cannot calculate the effective drainage area without a custom_r parameter representing the average precipitation rate.")

	if(rd.param.precipitations_are_2D):
		raise NotImplementedError("WIP, you have 2D precipitations input and I need ot implement a function to compute average upstream precipiation rate for each node.")

	if(custom_r is None):
		custom_r = rd.param.precipitations

	qw = compute_discharge_per_unit_width(rd, use_Qwin)

	return qw/custom_r


def compute_effective_width(rd, use_Qwin = True):
	'''
	EXPERIMENTAL

	Will probably evolve quite a lot in the coming weeks so I'll comment at that moment
	'''

	if(rd.param.precipitations_are_2D):
		prec = rd.param.precipitations
	else:
		prec = np.full(rd.param.reshp, rd.param.precipitations)

	QWD4 = compute_drainage_area_D4(rd, fill = True, N = 'auto', random_rec = True, Precipitations = prec)
	qw   = compute_discharge_per_unit_width(rd, use_Qwin)

	mask0 = qw <= 0

	Weff = np.zeros_like(qw)

	Weff[~mask0] = QWD4[~mask0]/qw[~mask0]

	return Weff




	





















# End of file