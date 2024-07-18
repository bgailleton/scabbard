'''
Different graph wrappers for different libraries.


B.G. (last modification: 07/2024)
'''


import numpy as np
import dagger as dag
import scabbard._utils as ut



class SFGraph(object):
	"""
	Simple single flow direction graph container.
	Only contains the minimal structure of the graph:
		- Single receivers and donors indices (row major flat index)
		- Stack (topological ordering)
		- Conversion helpers

	B.G. (last modification: )
	"""

	def __init__(self, Z, BCs = None, D4 = True, dx = 1.):
		'''
		'''
		
		if(len(Z.shape) != 2):
			raise ValueError('Need a 2D array as input for topography to infer the shape')

		# Geometrical infos
		## Overall shape
		self.shape = Z.shape
		# Geometrical infos
		self.D4 = D4

		self.dx = dx

		self.gridcpp = dag.GridCPP_f32(self.nx,self.ny, dx, dx,3) 

		# Getting the graph structure ready
		## Steepest receiver flat index
		self.Sreceivers = np.zeros((self.ny,self.nx),dtype = np.int32)
		## Number of Steepest donor per node
		self.Ndonors = np.zeros((self.ny,self.nx),dtype = np.int32)
		## Steepest donor per node ([i,j,:Ndonors[i,j]])
		self.donors = np.zeros((self.ny,self.nx,4 if self.D4 else 8),dtype = np.int32)
		## Topological ordering
		self.Stack = np.zeros(self.nxy,dtype = np.int32)

		# Initialising the graph
		self.update(Z,BCs)

	

	def update(self,Z,BCs = None):
		'''
		Updates the graph to a new topography and optional boundary conditions
		'''

		if(Z.shape != self.shape):
			raise AttributeError('topography needs to be same shape as the graph')

		if(BCs is None):
			BCs = ut.normal_BCs_from_shape(self.nx,self.ny)

		if self.D4:
			dag.compute_SF_stack_D4_full_f32(self.gridcpp, Z, self.Sreceivers, self.Ndonors, self.donors, self.Stack, BCs)
		else:
			raise ValueError('D8 SFGraph not implemented yet')


	@property
	def nx(self):
		return self.shape[1]
	@property
	def ny(self):
		return self.shape[0]
	@property
	def nxy(self):
		return self.shape[0]*self.shape[1]
	

