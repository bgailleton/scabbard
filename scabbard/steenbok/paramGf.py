import numpy as np
from enum import Enum

class InputMode(Enum):
	uniform_P = 0
	varying_P = 1
	input_point = 2

class HydroMode(Enum):
	static = 0
	dynamic = 1

class MorphoMode(Enum):
	MPM = 0
	eros_MPM = 1

class ParamGf(object):
	"""
		Docstring for ParamGf
	"""
	def __init__(self, mode = InputMode.uniform_P):
		
		super(ParamGf, self).__init__()
		self.mode = mode
		self.manning = 0.033
		self.dt_hydro = 1e-3
		self.Prate = 50 * 1e-3/3600 # 50 mm.h-1

		self.iBlock = None
		self.iGrid = None

		self.input_nodes = None
		self.input_Qw = None
		self.input_Qs = None

		
		self.morpho = False
		self.morphomode = MorphoMode.eros_MPM
		self.rho_water = 1000
		self.rho_sediment = 2650
		self.gravity = 9.81
		self.tau_c = 4
		self.theta_c = 0.047
		self.E_MPM = 1.
		self.dt_morpho = 1e-3


		self.k_erosion = 1.
		self.l_transp = 10.
		self.k_lat = 0.5

		self.hydro_mode = HydroMode.static


	def calculate_MPM_from_D(self, D):
		R = self.rho_sediment/self.rho_water - 1
		self.tau_c = self.rho_water * self.gravity * R * D * self.theta_c
		self.E_MPM = 8/(self.rho_water**0.5 * (self.rho_sediment - self.rho_water) * self.gravity)
		self.k_erosion = self.E_MPM/self.l_transp

		print("tau_c is", self.tau_c,"E:", self.E_MPM, "K", self.k_erosion)


	def set_input_points(self, nodes, Qw, Qs = None):
		self.input_nodes = nodes
		self.input_Qw = Qw
		self.input_Qs = Qs if (Qs is not None) else np.zeros_like(Qw)
		self.mode = InputMode.input_point
