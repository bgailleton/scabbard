import numpy as np
from enum import Enum

class InputMode(Enum):
	uniform_P = 0
	varying_P = 1
	input_point = 2

class HydroMode(Enum):
	static = 0
	dynamic = 1
	gp_static = 3
	gp_static_v2 = 4
	gp_static_v3 = 5
	gp_linear_test = 6
	gp_linear_test_v2 = 7
	gp_static_v4 = 8
	gp_static_v5 = 9
class MorphoMode(Enum):
	MPM = 0
	eros_MPM = 1
	gp_morpho_v1 = 2
	gp_morphydro_v1 = 3
	gp_morphydro_dyn_v1 = 4

class ParamGf(object):
	"""
		ParamGf gathers all the parameters specific to the graphflood model
	"""
	def __init__(self, mode = InputMode.uniform_P):
		
		super(ParamGf, self).__init__()
		self.mode = mode
		self.manning = 0.033
		self.dt_hydro = 1e-3
		self.Prate = 50 * 1e-3/3600 # 50 mm.h-1

		self.stabilisator_gphydro = 0.7

		self.iBlock = None
		self.iGrid = None

		self.input_nodes = None
		self.input_Qw = None
		self.input_Qs = None

		
		self.morpho = False
		self.morpho_mode = MorphoMode.eros_MPM
		self.rho_water = 1000
		self.rho_sediment = 2650
		self.gravity = 9.81
		self.tau_c = 4
		self.theta_c = 0.047
		self.E_MPM = 1.
		self.dt_morpho = 1e-3



		self.k_erosion = 1.
		self.kz = 1.
		self.kh = 1.
		self.l_transp = 10.
		self.k_lat = 0.5

		self.bs_k = 1e-6
		self.bs_hw = 0.05
		self.bs_exp = 1.
		self.bs_exp_hw = 1.

		self.hydro_mode = HydroMode.static

		self.boundary_slope = 1e-2


	def calculate_MPM_from_D(self, D):
		R = self.rho_sediment/self.rho_water - 1
		self.tau_c = (self.rho_sediment - self.rho_water) * self.gravity  * D * self.theta_c
		self.E_MPM = 8/(self.rho_water**0.5 * (self.rho_sediment - self.rho_water) * self.gravity)
		self.k_erosion = self.E_MPM/self.l_transp

		print("tau_c is", self.tau_c,"E:", self.E_MPM, "K", self.k_erosion)


	def set_input_points(self, nodes, Qw, Qs = None):
		self.input_nodes = nodes
		self.input_Qw = Qw
		self.input_Qs = Qs if (Qs is not None) else np.zeros_like(Qw)
		self.mode = InputMode.input_point
