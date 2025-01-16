'''
Different graph wrappers for different libraries.


B.G. (last modification: 07/2024)
'''


import numpy as np
import dagger as dag
import scabbard as scb
import scabbard._utils as ut
import topotoolbox as ttb



class SFGraph(object):
	"""
	Simple single flow direction graph container.
	Only contains the minimal structure of the graph:
		- Single receivers and donors indices (row major flat index)
		- Stack (topological ordering)
		- Conversion helpers

	B.G. (last modification: )
	"""

	def __init__(self, Z, BCs = None, D4 = True, dx = 1., backend = 'ttb', fill_LM = False, step_fill = 1e-3):
		'''
		'''
		if(isinstance(Z, np.ndarray)):
			tZ = Z
		elif(isinstance(Z,scb.raster.RegularRasterGrid)):
			tZ = Z.Z
		else:
			raise ValueError("scabbard.flow.graph.SFGraph: does not recognize elevation data type")

		if(len(tZ.shape) != 2):
			raise ValueError('Need a 2D array as input for topography to infer the shape')

		# Geometrical infos
		## Overall shape
		self.shape = tZ.shape
		self.dim = np.array(self.shape, dtype = np.uint64)
		# Geometrical infos
		self.D4 = D4

		self.dx = dx


		self.backend = backend

		self.gridcpp = dag.GridCPP_f32(self.nx,self.ny, dx, dx,3) 

		# Getting the graph structure ready
		## Steepest receiver flat index
		self.Sreceivers = np.zeros((self.nxy), dtype = (np.uint64 if self.backend == 'ttb' else np.int32))
		## distance to Steepest receiver flat index
		self.Sdx = np.zeros((self.nxy),dtype = np.float32)
		## Number of Steepest donor per node
		self.Ndonors = np.zeros((self.nxy), dtype = np.uint8 if self.backend == 'ttb' else np.int32)
		## Steepest donor per node ([i,j,:Ndonors[i,j]])
		self.donors = np.zeros((self.nxy * (4 if self.D4 else 8) ), dtype = (np.uint64 if self.backend == 'ttb' else np.int32))
		## Topological ordering
		self.Stack = np.zeros(self.nxy, dtype = (np.uint64 if self.backend == 'ttb' else np.int32))

		# Initialising the graph
		self.update(tZ, BCs, fill_LM,step_fill)

	

	def update(self, Z, BCs = None, fill_LM = False, step_fill = 1e-3):
		'''
		Updates the graph to a new topography and optional boundary conditions
		'''

		if(isinstance(Z, np.ndarray)):
			tZ = Z
		elif(isinstance(Z,scb.raster.RegularRasterGrid)):
			tZ = Z.Z
		else:
			raise ValueError("scabbard.flow.graph.SFGraph.update: does not recognize elevation data type")

		if(tZ.shape != self.shape):
			raise AttributeError('topography needs to be same shape as the graph')

		if(BCs is None):
			BCs = ut.normal_BCs_from_shape(self.nx,self.ny)

		if(self.backend == 'dagger'):
			if self.D4:
				dag.compute_SF_stack_D4_full_f32(self.gridcpp, tZ, self.Sreceivers.reshape(self.ny,self.nx), self.Ndonors.reshape(self.ny,self.nx), self.donors.reshape(self.ny,self.nx), self.Stack, BCs)
			else:
				raise ValueError('D8 SFGraph not implemented yet')
		elif self.backend == 'ttb':
			self.Ndonors.fill(0)
			#Theer is a bug in my implementation in ttb, I need to sort it but the vanilla fill LM does not fill the flats
			if(fill_LM):
				scb.ttb.graphflood.funcdict['priority_flood_TO'](tZ.ravel(), self.Stack, BCs.ravel(), self.dim, not self.D4, step_fill)
				
			ttb.graphflood.funcdict['sfgraph'](tZ.ravel(), self.Sreceivers, self.Sdx, self.donors, self.Ndonors, self.Stack, BCs.ravel(), self.dim, self.dx * self.dx, not self.D4, False, step_fill) # False cause I fill lm above


	@property
	def nx(self):
		return self.shape[1]
	@property
	def ny(self):
		return self.shape[0]
	@property
	def nxy(self):
		return self.shape[0]*self.shape[1]
	

