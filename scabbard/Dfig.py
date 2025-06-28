'''
This module provides the `Dfig` class, a first attempt at creating a map automator
for managing and updating Matplotlib figures and axes within the scabbard framework.

Author: B.G.
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scabbard import Dax


class Dfig(object):

	"""
	Manages a Matplotlib Figure and its associated Dax (Dynamic Axes) objects.

	This class provides a way to organize and update multiple plots within a single
	figure, allowing for dynamic visualization of simulation results.

	Attributes:
		fig (matplotlib.figure.Figure): The parent Matplotlib Figure object.
		axes (dict): A dictionary where keys are unique identifiers (from Dax objects)
				 and values are `Dax` instances.

	Author: B.G.
	"""

	def __init__(self, fig = None, axes = None):

		super(Dfig, self).__init__()

		self.fig = None
		self.axes = {}

		if(fig == None):
			self.init_default_fig()
		else:

			self.fig = fig

			if( isinstance(axes,list)):
				for ax in axes:
					self.axes[ax.key] = ax
			else:
				self.axes[axes.key] = axes
				
	def init_default_fig(self):
		"""
		Naive init of default figure
		"""
		fig, ax = plt.subplots()
		self.fig = fig
		tdax = Dax(ax)
		self.axes = {}
		self.axes[tdax.key] =  tdax

	def update(self):
		for ax in self.axes:
			ax.update()
		self.fig.canvas.draw()
			

# class Map(object):

# 	"""
# 		docstring for Map
# 	"""
	
# 	def __init__(self, fig, ax):
# 		super(Map, self).__init__()
# 		self.fig = fig
# 		self.ax
		



















































# end of file