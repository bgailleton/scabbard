import scabbard as scb
import numpy as np
import numba as nb



class RivNet:
	'''
	Rivet is a simple river network base class holding the minimal data structure to define a river network
	'''

	def __init__(self):

		# Holds the node flat IDs from the original 
		self.nodes = None