# try:
import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import scabbard.steenbok._functions as funcu
import scabbard.steenbok.dtype_helper as typh
import scabbard.steenbok.kernel_utils as kut
from scabbard.steenbok.paramGf import *
# except:
# 	print('pycuda is not importable, please check its installation in order to use steenbok submodules')

import scabbard as scb
import numpy as np
import math as m
import os


class cuenv:

	def __init__(self, env, topology = "D4"):
		'''

		'''
		
		self.env = env
		
		self._constants = {}
		
		self._arrays = {}
		
		self.topology = topology
		
		self.mod, self.functions = funcu.build_kernel(self.topology)

		self.gBlock = None
		self.gGrid = None

		self.grid_setup = False


		self.param_graphflood = None
		

	def setup_grid(self):
		'''

		'''
		nodata = -2147483648
		nx,ny = self.env.grid.nx, self.env.grid.ny
		dx,dy = self.env.grid.dx, self.env.grid.dy
		dxy = m.sqrt(dx**2 + dy**2)


		# grid dimensions
		kut.set_constant(self.mod, self.env.grid.nx, "NX", 'i32')
		kut.set_constant(self.mod, self.env.grid.ny, "NY", 'i32')
		kut.set_constant(self.mod, nodata, "NODATA", 'i32')
		kut.set_constant(self.mod, self.env.grid.nxy, "NXY", 'i32')

		# grid spacing
		kut.set_constant(self.mod, self.env.grid.dy, "DY", 'f32')
		kut.set_constant(self.mod, self.env.grid.dx, "DX", 'f32')
		kut.set_constant(self.mod, self.env.grid.dx * self.env.grid.dy, "CELLAREA", 'f32')


		# neighbourer and other spatialisers
		# Initialize and copy the constant arrays
		neighbourers = []
		oneighbourersA = []
		oneighbourersB = []
		if(self.topology == "D8"):
			neighbourers.append(np.array([ -nx-1 , -nx , -nx+1 , -1 , 1 , nx-1 , nx , nx+1 ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][3],neighbourers[-1][0],neighbourers[-1][1],neighbourers[-1][5],neighbourers[-1][2],neighbourers[-1][6],neighbourers[-1][7],neighbourers[-1][4]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][1],neighbourers[-1][2],neighbourers[-1][4],neighbourers[-1][0],neighbourers[-1][7],neighbourers[-1][3],neighbourers[-1][5],neighbourers[-1][6]], dtype=np.int32))
			neighbourers.append(np.array([nodata, nodata , nodata , nodata , 1 , nodata , nx , nx+1 ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][3],neighbourers[-1][0],neighbourers[-1][1],neighbourers[-1][5],neighbourers[-1][2],neighbourers[-1][6],neighbourers[-1][7],neighbourers[-1][4]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][1],neighbourers[-1][2],neighbourers[-1][4],neighbourers[-1][0],neighbourers[-1][7],neighbourers[-1][3],neighbourers[-1][5],neighbourers[-1][6]], dtype=np.int32))
			neighbourers.append(np.array([ nodata , nodata , nodata , -1 , 1 , nx-1 , nx , nx+1 ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][3],neighbourers[-1][0],neighbourers[-1][1],neighbourers[-1][5],neighbourers[-1][2],neighbourers[-1][6],neighbourers[-1][7],neighbourers[-1][4]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][1],neighbourers[-1][2],neighbourers[-1][4],neighbourers[-1][0],neighbourers[-1][7],neighbourers[-1][3],neighbourers[-1][5],neighbourers[-1][6]], dtype=np.int32))
			neighbourers.append(np.array([ nodata , nodata , nodata , -1 , nodata , nx-1 , nx , nodata ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][3],neighbourers[-1][0],neighbourers[-1][1],neighbourers[-1][5],neighbourers[-1][2],neighbourers[-1][6],neighbourers[-1][7],neighbourers[-1][4]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][1],neighbourers[-1][2],neighbourers[-1][4],neighbourers[-1][0],neighbourers[-1][7],neighbourers[-1][3],neighbourers[-1][5],neighbourers[-1][6]], dtype=np.int32))
			neighbourers.append(np.array([ nodata , -nx , -nx+1 , nodata , 1 , nodata , nx , nx+1 ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][3],neighbourers[-1][0],neighbourers[-1][1],neighbourers[-1][5],neighbourers[-1][2],neighbourers[-1][6],neighbourers[-1][7],neighbourers[-1][4]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][1],neighbourers[-1][2],neighbourers[-1][4],neighbourers[-1][0],neighbourers[-1][7],neighbourers[-1][3],neighbourers[-1][5],neighbourers[-1][6]], dtype=np.int32))
			neighbourers.append(np.array([ -nx-1 , -nx , nodata , -1 , nodata , nx-1 , nx , nodata ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][3],neighbourers[-1][0],neighbourers[-1][1],neighbourers[-1][5],neighbourers[-1][2],neighbourers[-1][6],neighbourers[-1][7],neighbourers[-1][4]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][1],neighbourers[-1][2],neighbourers[-1][4],neighbourers[-1][0],neighbourers[-1][7],neighbourers[-1][3],neighbourers[-1][5],neighbourers[-1][6]], dtype=np.int32))
			neighbourers.append(np.array([ nodata , -nx , -nx+1 , nodata , 1 , nodata , nodata , nodata ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][3],neighbourers[-1][0],neighbourers[-1][1],neighbourers[-1][5],neighbourers[-1][2],neighbourers[-1][6],neighbourers[-1][7],neighbourers[-1][4]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][1],neighbourers[-1][2],neighbourers[-1][4],neighbourers[-1][0],neighbourers[-1][7],neighbourers[-1][3],neighbourers[-1][5],neighbourers[-1][6]], dtype=np.int32))
			neighbourers.append(np.array([ -nx-1 , -nx , -nx+1 , -1 , 1 , nodata , nodata , nodata ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][3],neighbourers[-1][0],neighbourers[-1][1],neighbourers[-1][5],neighbourers[-1][2],neighbourers[-1][6],neighbourers[-1][7],neighbourers[-1][4]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][1],neighbourers[-1][2],neighbourers[-1][4],neighbourers[-1][0],neighbourers[-1][7],neighbourers[-1][3],neighbourers[-1][5],neighbourers[-1][6]], dtype=np.int32))
			neighbourers.append(np.array([ -nx-1 , -nx , nodata , -1 , nodata , nodata , nodata , nodata ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][3],neighbourers[-1][0],neighbourers[-1][1],neighbourers[-1][5],neighbourers[-1][2],neighbourers[-1][6],neighbourers[-1][7],neighbourers[-1][4]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][1],neighbourers[-1][2],neighbourers[-1][4],neighbourers[-1][0],neighbourers[-1][7],neighbourers[-1][3],neighbourers[-1][5],neighbourers[-1][6]], dtype=np.int32))
		elif(self.topology == "D4"):
			neighbourers.append(np.array([  -nx , -1 , 1 , nx  ], dtype=np.int32))
			# neighbourers.append(np.array([  nodata , nodata , 1 , nodata  ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][1], neighbourers[-1][0], neighbourers[-1][0],neighbourers[-1][2]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][2], neighbourers[-1][3], neighbourers[-1][3],neighbourers[-1][1]], dtype=np.int32))
			neighbourers.append(np.array([ nodata  , nodata , 1  , nx ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][1], neighbourers[-1][0], neighbourers[-1][0],neighbourers[-1][2]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][2], neighbourers[-1][3], neighbourers[-1][3],neighbourers[-1][1]], dtype=np.int32))
			neighbourers.append(np.array([   nodata  , -1 , 1 ,  nx  ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][1], neighbourers[-1][0], neighbourers[-1][0],neighbourers[-1][2]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][2], neighbourers[-1][3], neighbourers[-1][3],neighbourers[-1][1]], dtype=np.int32))
			neighbourers.append(np.array([  nodata ,  -1 , nodata ,  nx  ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][1], neighbourers[-1][0], neighbourers[-1][0],neighbourers[-1][2]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][2], neighbourers[-1][3], neighbourers[-1][3],neighbourers[-1][1]], dtype=np.int32))
			neighbourers.append(np.array([ -nx , nodata , 1 , nx ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][1], neighbourers[-1][0], neighbourers[-1][0],neighbourers[-1][2]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][2], neighbourers[-1][3], neighbourers[-1][3],neighbourers[-1][1]], dtype=np.int32))
			neighbourers.append(np.array([  -nx ,  -1 , nodata , nx  ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][1], neighbourers[-1][0], neighbourers[-1][0],neighbourers[-1][2]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][2], neighbourers[-1][3], neighbourers[-1][3],neighbourers[-1][1]], dtype=np.int32))
			neighbourers.append(np.array([   -nx ,  nodata , 1 ,  nodata  ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][1], neighbourers[-1][0], neighbourers[-1][0],neighbourers[-1][2]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][2], neighbourers[-1][3], neighbourers[-1][3],neighbourers[-1][1]], dtype=np.int32))
			neighbourers.append(np.array([  -nx ,  -1 , 1 ,  nodata  ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][1], neighbourers[-1][0], neighbourers[-1][0],neighbourers[-1][2]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][2], neighbourers[-1][3], neighbourers[-1][3],neighbourers[-1][1]], dtype=np.int32))
			neighbourers.append(np.array([  -nx ,  -1 , nodata , nodata  ], dtype=np.int32))
			oneighbourersA.append(np.array([neighbourers[-1][1], neighbourers[-1][0], neighbourers[-1][0],neighbourers[-1][2]], dtype=np.int32))
			oneighbourersB.append(np.array([neighbourers[-1][2], neighbourers[-1][3], neighbourers[-1][3],neighbourers[-1][1]], dtype=np.int32))

		kut.set_constant(self.mod, neighbourers, "NEIGHBOURERS", 'i32')
		kut.set_constant(self.mod, oneighbourersA, "ONEIGHBOURERSA", 'i32')
		kut.set_constant(self.mod, oneighbourersB, "ONEIGHBOURERSB", 'i32')
		dXs = np.array([dxy, dy, dxy, dx, dx, dxy, dy, dxy], dtype=np.float32) if self.topology == "D8" else np.array([ dy, dx, dx, dy], dtype=np.float32)
		dYs = np.array([dxy, dx, dxy, dy, dy, dxy, dx, dxy], dtype=np.float32) if self.topology == "D8" else np.array([ dx, dy, dy, dx], dtype=np.float32)

		kut.set_constant(self.mod, dXs, "DXS", 'f32')
		kut.set_constant(self.mod, dYs, "DYS", 'f32')


		self._arrays['Z'] = kut.arrayHybrid(self.mod, self.env.grid._Z, "Z", 'f32')
		self._arrays['BC'] = kut.arrayHybrid(self.mod, self.env.data.get_boundaries(), "BC", 'u8')

		# grid block for the landscape
		block_size_x = 32
		block_size_y = 32
		self.gBlock = (int(block_size_x), int(block_size_y), 1)
		grid_size_x = (nx + block_size_x - 1) // block_size_x
		grid_size_y = (ny + block_size_y - 1) // block_size_y
		self.gGrid = (int(grid_size_x), int(grid_size_y))

		self.grid_setup = True


	def setup_graphflood(self, paramGf = ParamGf()):
		'''

		'''

		# Setting up the morpho part of the code

		self.param_graphflood = paramGf

		self._arrays['hw'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "hw")

		self._arrays['QwA'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "QwA")
		self._arrays['QwB'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "QwB")
		self._arrays['QwC'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "QwC")
		# self._arrays['QwD'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "QwD")
		# self._arrays['QwE'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "QwE")

		# self._arrays['Nrecs'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'i32', ref = "Nrecs")
		self.nancheckers = kut.aH_zeros(self.mod, 1, 'i32', ref = "nancheckers")


		if(self.param_graphflood.mode == InputMode.input_point):
			self._arrays['input_Qw'] = kut.arrayHybrid(self.mod, self.param_graphflood.input_Qw, 'input_Qw', 'f32')
			self._arrays['input_Qs'] = kut.arrayHybrid(self.mod, self.param_graphflood.input_Qs, 'input_Qs', 'f32')
			self._arrays['input_nodes'] = kut.arrayHybrid(self.mod, self.param_graphflood.input_nodes, 'input_nodes', 'i32')

		kut.set_constant(self.mod, self.param_graphflood.manning, "MANNING", 'f32')
		kut.set_constant(self.mod, self.param_graphflood.dt_hydro, "DT_HYDRO", 'f32')
		kut.set_constant(self.mod, self.param_graphflood.boundary_slope, "BOUND_SLOPE", 'f32')
		kut.set_constant(self.mod, self.param_graphflood.stabilisator_gphydro, "STABIL_GPHYDRO", 'f32')


		if(self.param_graphflood.mode == InputMode.input_point):
			block_input = 256  # This is an arbitrary value; tune it based on your GPU architecture
			grid_input = (self.param_graphflood.input_nodes.shape[0] + block_input - 1) // block_input
			self.param_graphflood.iBlock = (block_input,1,1)
			self.param_graphflood.iGrid = (grid_input,1)


		if(self.param_graphflood.morpho):
			kut.set_constant(self.mod, self.param_graphflood.rho_water , "RHO_WATER", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.rho_sediment , "RHO_SEDIMENT", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.gravity , "GRAVITY", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.E_MPM , "E_MPM", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.tau_c , "TAU_C", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.k_erosion , "K_EROS", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.l_transp , "L_EROS", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.k_lat , "KL_EROS", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.bs_hw , "BS_MINHW", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.bs_k , "BS_K", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.bs_exp , "BS_EXP", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.bs_exp_hw , "BS_HW_EXP", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.kz , "KZ", 'f32')
			kut.set_constant(self.mod, self.param_graphflood.kh , "KH", 'f32')
			
			
			



			kut.set_constant(self.mod, self.param_graphflood.dt_morpho , "DT_MORPHO", 'f32')

			self._arrays['QsA'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "QsA")
			self._arrays['QsB'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "QsB")
			self._arrays['QsC'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "QsC")
			self._arrays['QsD'] = kut.aH_zeros(self.mod, self.env.grid.nxy, 'f32', ref = "QsD")

			
	def run_graphflood_fillup(self, n_iterations = 1000, verbose = False):

		for _ in range(n_iterations):
			self.functions["grid2val"](self._arrays['QwB']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
			# Managing the local inputs
			if(self.param_graphflood.mode == InputMode.input_point):
				## from specific input points
				self.functions["add_Qw_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qw']._gpu , self._arrays['QwA']._gpu , self._arrays['QwB']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
			else:
				## from precipitation rates
				self.functions["add_Qw_global"](self._arrays['QwA']._gpu, np.float32(self.param_graphflood.Prate), self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			
			self.functions["compute_Qws"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu,self._arrays['QwA']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)


	def run_graphflood(self, n_iterations = 100, verbose = False, nmorpho = 10):
		for i in range(n_iterations):
			if(i % 1000 == 0):
				print(i,end='     \r') if verbose else 0

			if(self.param_graphflood.morpho_mode == MorphoMode.gp_morphydro_v1):
				self.functions["grid2val"](self._arrays['QwB']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
				# Managing the local inputs
				if(self.param_graphflood.mode == InputMode.input_point):
					## from specific input points
					self.functions["add_Qw_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qw']._gpu , self._arrays['QwA']._gpu , self._arrays['QwB']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
				else:
					## from precipitation rates
					self.functions["add_Qw_global"](self._arrays['QwA']._gpu, np.float32(self.param_graphflood.Prate), self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

				if(i%nmorpho == 0 and self.param_graphflood.morpho):

					self.functions["checkGrid4nan"](self._arrays['Z']._gpu, self.nancheckers._gpu ,block = self.gBlock, grid = self.gGrid)

					if(self.nancheckers.get()[0] > 0):
						raise ValueError("NAN FOUND IN Z");

					self.functions["grid2val"](self._arrays['QsC']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
					self.functions["grid2val"](self._arrays['QsD']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
					self.functions["compute_QwsQss"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['QwC']._gpu, self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, self._arrays['QsC']._gpu, self._arrays['QsD']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
					self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
					self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu,self._arrays['QwA']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
					self.functions["increment_hs"](self._arrays['hw']._gpu, self._arrays['Z']._gpu,self._arrays['QsA']._gpu, self._arrays['QsC']._gpu, self._arrays['QsD']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
					self.functions["swapQwin"](self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, block = self.gBlock, grid = self.gGrid)
					if(self.param_graphflood.bs_k > 0):
						self.functions["copygrid"](self._arrays['QsD']._gpu, self._arrays['Z']._gpu, block = self.gBlock, grid = self.gGrid)
						self.functions["diffuse_bed"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsD']._gpu, self._arrays['QsA']._gpu, self._arrays['BC']._gpu,block = self.gBlock, grid = self.gGrid)
						self.functions["swapQsin"](self._arrays['Z']._gpu, self._arrays['QsD']._gpu, block = self.gBlock, grid = self.gGrid)
				
				else:
					self.functions["compute_Qws"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
					self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
					self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu,self._arrays['QwA']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			
			elif(self.param_graphflood.morpho_mode == MorphoMode.gp_morphydro_dyn_v1):
				self.functions["grid2val"](self._arrays['QwB']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
				self.functions["grid2val"](self._arrays['QwD']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
				# Managing the local inputs
				if(self.param_graphflood.mode == InputMode.input_point):
					## from specific input points
					self.functions["add_Qw_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qw']._gpu , self._arrays['QwA']._gpu , self._arrays['QwB']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
				else:
					## from precipitation rates
					self.functions["add_Qw_global"](self._arrays['QwA']._gpu, np.float32(self.param_graphflood.Prate), self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

				self.functions["grid2val"](self._arrays['QsC']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
				self.functions["grid2val"](self._arrays['QwC']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
				self.functions["compute_QwsQss_dyn"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['QwC']._gpu, self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, self._arrays['QsC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu,self._arrays['QwA']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["increment_hs_noQD"](self._arrays['hw']._gpu, self._arrays['Z']._gpu,self._arrays['QsA']._gpu, self._arrays['QsC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["copygrid"](self._arrays['QsD']._gpu, self._arrays['Z']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["diffuse_bed"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsD']._gpu, self._arrays['QsA']._gpu, self._arrays['BC']._gpu,block = self.gBlock, grid = self.gGrid)
				self.functions["swapQwin"](self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["swapQsin"](self._arrays['Z']._gpu, self._arrays['QsD']._gpu, block = self.gBlock, grid = self.gGrid)
			else:
				self.__run_hydro()
				if(i%nmorpho == 0 and self.param_graphflood.morpho):
					self.__run_morpho()

			

	def __run_hydro(self):
		'''
		Internal function managing the hydrodynamics part of graphflood
		'''

		if(self.param_graphflood.hydro_mode == HydroMode.gp_linear_test_v2):

			self.functions["grid2val"](self._arrays['QwA']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
			self.functions["add_Qw_local_linear_test_v2"](self._arrays['input_nodes']._gpu , self._arrays['input_Qw']._gpu , self._arrays['hw']._gpu, self._arrays['QwA']._gpu, np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
			self.functions["linear_test_v2"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["swapQwin"](self._arrays['hw']._gpu, self._arrays['QwA']._gpu, block = self.gBlock, grid = self.gGrid)
		
		elif(self.param_graphflood.hydro_mode ==  HydroMode.gp_static_v4):
			self.functions["grid2val"](self._arrays['QwB']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
			self.functions["compute_static_Qwin_v4"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['Nrecs']._gpu, self._arrays['QwC']._gpu, self._arrays['QwD']._gpu, self._arrays['QwE']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
			# Managing the local inputs
			if(self.param_graphflood.mode == InputMode.input_point):
				## from specific input points
				self.functions["add_Qw_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qw']._gpu , self._arrays['QwA']._gpu , self._arrays['QwB']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
			else:
				## from precipitation rates
				self.functions["add_Qw_global"](self._arrays['QwA']._gpu, np.float32(self.param_graphflood.Prate), self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			# self.functions["add_Qw_local_linear_test_v2"](self._arrays['input_nodes']._gpu , self._arrays['input_Qw']._gpu , self._arrays['hw']._gpu, self._arrays['QwA']._gpu, np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
			self.functions["compute_hw_v4"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['Nrecs']._gpu, self._arrays['QwC']._gpu, self._arrays['QwD']._gpu, self._arrays['QwE']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
								

		else:	
		
			# Whatever happens, init QWB to 0
			self.functions["grid2val"](self._arrays['QwB']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)

			# THe biggest diff between static and transient is that Qwin has no memory of the previous time step in transient
			if(self.param_graphflood.hydro_mode == HydroMode.dynamic) :
				self.functions["grid2val"](self._arrays['QwA']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)


			if(self.param_graphflood.hydro_mode == HydroMode.gp_static_v2):
				self.functions["grid2val"](self._arrays['QwA']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
				self.functions["grid2val"](self._arrays['QwC']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
				self.functions["grid2val"](self._arrays['QwD']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)

			if(self.param_graphflood.hydro_mode == HydroMode.gp_linear_test):
				# self.functions["grid2val"](self._arrays['QwA']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
				self.functions["grid2val_i32"](self._arrays['Nrecs']._gpu, np.int32(0.),block = self.gBlock, grid = self.gGrid)
				self.functions["grid2val"](self._arrays['QwC']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)

				# print('0ed?')
			
			# Managing the local inputs
			if(self.param_graphflood.mode == InputMode.input_point):
				## from specific input points
				self.functions["add_Qw_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qw']._gpu , self._arrays['QwA']._gpu , self._arrays['QwB']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
			else:
				## from precipitation rates
				self.functions["add_Qw_global"](self._arrays['QwA']._gpu, np.float32(self.param_graphflood.Prate), self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

			# RUn graphflood gpu test 1 static
			if(self.param_graphflood.hydro_mode == HydroMode.static):
				self.functions["compute_Qwin"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["compute_Qwout"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			# RUn graphflood gpu test 1 dynamic
			elif(self.param_graphflood.hydro_mode == HydroMode.dynamic):
				self.functions["compute_Qw_dyn"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

			elif(self.param_graphflood.hydro_mode == HydroMode.gp_static_v5):
				self.functions["compute_Qws"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu,self._arrays['QwA']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				
			elif(self.param_graphflood.hydro_mode == HydroMode.gp_static):
				self.functions["compute_static_Qwin"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu,self._arrays['QwC']._gpu,self._arrays['QwD']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["compute_static_h"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu ,self._arrays['QwC']._gpu,self._arrays['QwD']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			elif(self.param_graphflood.hydro_mode == HydroMode.gp_static_v2):
				self.functions["compute_static_Qwin_v2"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["compute_static_h"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			
			elif(self.param_graphflood.hydro_mode == HydroMode.gp_linear_test):
				self.functions["compute_static_Qwin_linear_test"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['Nrecs']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["add_Qw_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qw']._gpu , self._arrays['QwA']._gpu , self._arrays['QwB']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
				self.functions["compute_static_h_linear_test"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['Nrecs']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				
			elif(self.param_graphflood.hydro_mode == HydroMode.gp_static_v3):
				self.functions["compute_static_Qwin_v3"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
				self.functions["increment_hw_v3"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
				

			if(self.param_graphflood.hydro_mode == HydroMode.static or self.param_graphflood.hydro_mode == HydroMode.dynamic):
				self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu,self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

	
	def __run_morpho(self):

		if(self.param_graphflood.morpho_mode == MorphoMode.gp_morpho_v1):

			if(self.param_graphflood.mode == InputMode.input_point):
				self.functions["add_Qs_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qs']._gpu , self._arrays['QsA']._gpu , self._arrays['QsB']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
			
			self.functions["compute_Qsin_v1"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, self._arrays['QsC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["increment_morpho"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsA']._gpu, self._arrays['QsC']._gpu, self._arrays['BC']._gpu,block = self.gBlock, grid = self.gGrid)
			self.functions["swapQsin"](self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["copygrid"](self._arrays['QsD']._gpu, self._arrays['Z']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["diffuse_bed"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsD']._gpu, self._arrays['QsA']._gpu, self._arrays['BC']._gpu,block = self.gBlock, grid = self.gGrid)
			self.functions["swapQsin"](self._arrays['Z']._gpu, self._arrays['QsD']._gpu, block = self.gBlock, grid = self.gGrid)

		
		# OLD TESTS
		elif(self.param_graphflood.morpho_mode == MorphoMode.MPM):
			# self.functions["grid2val"](self._arrays['QsA']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
			self.functions["grid2val"](self._arrays['QsB']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
			self.functions["grid2val"](self._arrays['QsD']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
			# self.functions["grid2val"](self._arrays['QsC']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
		
			if(self.param_graphflood.mode == InputMode.input_point):
				self.functions["add_Qs_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qs']._gpu , self._arrays['QsA']._gpu , self._arrays['QsB']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
			# else:
			# 	self.functions["add_Qs_global"](self._arrays['QsA']._gpu, np.float32(self.param_graphflood.Prate), self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

			self.functions["compute_MPM_SS"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, self._arrays['BC']._gpu,block = self.gBlock, grid = self.gGrid)
			
			self.functions["increment_hs"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, self._arrays['BC']._gpu,block = self.gBlock, grid = self.gGrid)


		elif(self.param_graphflood.morpho_mode == MorphoMode.eros_MPM):
			# self.functions["swapQsin"](self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["grid2val"](self._arrays['QsC']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
			# self.functions["grid2val"](self._arrays['QsB']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)

			if(self.param_graphflood.mode == InputMode.input_point):
				self.functions["add_Qs_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qs']._gpu , self._arrays['QsC']._gpu , self._arrays['QsC']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
			# else:
			# 	self.functions["add_Qs_global"](self._arrays['QsA']._gpu, np.float32(self.param_graphflood.Prate), self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

			self.functions["compute_EROS_SS"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, self._arrays['QsC']._gpu, self._arrays['QsD']._gpu, self._arrays['BC']._gpu,block = self.gBlock, grid = self.gGrid)
			# self.functions["bedslip"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsA']._gpu, self._arrays['QsD']._gpu, self._arrays['BC']._gpu,block = self.gBlock, grid = self.gGrid)
			
			self.functions["increment_hs"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QsA']._gpu, self._arrays['QsB']._gpu, self._arrays['QsD']._gpu, self._arrays['BC']._gpu,block = self.gBlock, grid = self.gGrid)
			self.functions["swapQsin"](self._arrays['QsA']._gpu, self._arrays['QsC']._gpu, block = self.gBlock, grid = self.gGrid)
	


	def HHRG42fZwy(self, N = 1000):
		start = drv.Event()
		end = drv.Event()
		start.record()
		for i in range(N):
			self.functions["compute_Qwin"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
		end.record()
		end.synchronize()
		print("compute_Qwin:",start.time_till(end))	
		
		start.record()
		for i in range(N):
			self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)

		end.record()
		end.synchronize()
		print("swapQwin:",start.time_till(end))	

		start.record()
		for i in range(N):
			self.functions["compute_Qwout"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

		end.record()
		end.synchronize()
		print("compute_Qwout:",start.time_till(end))	

		start.record()
		for i in range(N):
			self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

		end.record()
		end.synchronize()
		print("increment_hw:",start.time_till(end))	

		
	def DEBUGTIME_gphv3(self, N = 1000):
		start = drv.Event()
		end = drv.Event()
		start.record()
		for i in range(N):
			self.functions["compute_static_Qwin_v3"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			# self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

		end.record()
		end.synchronize()
		print("compute_static_Qwin_v3:",start.time_till(end))	

		start.record()
		for i in range(N):
			self.functions["increment_hw_v3"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwC']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			# self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

		end.record()
		end.synchronize()
		print("increment_hw_v3:",start.time_till(end))	




	def graphflood_fillup(self, n_iterations = 100, verbose = False):
		for i in range(n_iterations):
			if(i % 1000 == 0):
				print(i,end='     \r') if verbose else 0
			self.functions["grid2val"](self._arrays['QwB']._gpu, np.float32(0.),block = self.gBlock, grid = self.gGrid)
			
			if(self.param_graphflood.mode == InputMode.input_point):
				self.functions["add_Qw_local"](self._arrays['input_nodes']._gpu , self._arrays['input_Qw']._gpu , self._arrays['QwA']._gpu , self._arrays['QwB']._gpu , np.int32(self.param_graphflood.input_nodes.shape[0]), block = self.param_graphflood.iBlock, grid = self.param_graphflood.iGrid)
			else:
				self.functions["add_Qw_global"](self._arrays['QwA']._gpu, np.float32(self.param_graphflood.Prate), self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)

			self.functions["compute_Qwin"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["swapQwin"](self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, block = self.gBlock, grid = self.gGrid)
			self.functions["compute_Qwout"](self._arrays['hw']._gpu, self._arrays['Z']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)


			self.functions["increment_hw"](self._arrays['hw']._gpu, self._arrays['Z']._gpu,self._arrays['QwA']._gpu, self._arrays['QwB']._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)


	def testSS(self):

		if(self.grid_setup == False):
			print('need to set up the grid before running', kut.get_current_function_name())
			return
			
		tSS = kut.aH_zeros_like(self.mod, self.env.grid._Z, 'f32')
		self.functions["calculate_SS"](self._arrays['Z']._gpu, tSS._gpu, self._arrays['BC']._gpu, block = self.gBlock, grid = self.gGrid)
		ret = tSS.get()
		tSS.delete()
		return ret

























# end of file
		
		
		
		
		
		
		
		