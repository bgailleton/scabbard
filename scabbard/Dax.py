import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmcrameri as cm
import string
import random


class Dax(object):
	
	"""
		docstring for Dax
	"""

	def __init__(self, ax, key = None, zorder = 1):
		
		super(Dax, self).__init__()
		
		self.ax = ax

		self.zorder = zorder
		
		# // Generating random key if none is given	
		if(key is not None):
			self.key = key
		else:
			self.key = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

	def __hash__(self):
		return hash(self.key)



class RGridDax(Dax):
	"""docstring for RGridDax"""
	def __init__(self,Rgrid, ax, key = None, cmap = cm.cm.davos, hillshade = True, alpha_hillshade = 0.45, clim = None):
		
		super(RGridDax, self).__init__(ax, key)
		self.grid = Rgrid
		self.base = self.ax.imshow(self.grid.Z2D, extent = self.grid.extent(), cmap = cmap, zorder =  self.zorder)
		self.hillshade_on = hillshade

		self.clim = clim if clim is not None else [self.grid.min(), self.grid.max()]

		if(hillshade):
			self.hillshade = self.ax.imshow(self.grid.hillshade, extent = self.grid.extent(), cmap = cm.cm.grayC, alpha = alpha_hillshade, zorder = self.zorder)

		self.ax.set_xlabel("X (m)")
		self.ax.set_ylabel("Y (m)")





		





















































# dskjhksadfhla