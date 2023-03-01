'''
This is a first attempt to make a map automator.
Probably will just test different way here before making a better one
B.G
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scabbard import Dax

# My cat just walked on the keyboard, let's keep that
# p0----------ppppppppppppppppppppppppp+699999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999990


class Dfig(object):

	"""docstring for Dfig"""

	def __init__(self, fig = None, axes = None):

		super(Dfig, self).__init__()

		self.fig = None
		self.axes = []

		if(fig == None):
			self.init_default_fig()
		else:
			self.fig = fig
			if( isinstance(axes,list)):
				self.axes = axes
			else:
				self.axes.append(axes)

				
	def init_default_fig(self):
		"""
		Naive init of default figure
		"""
		fig, ax = plt.subplots()
		self.fig = fig
		tdax = Dax(ax)
		if(axes is None):
			self.axes = []
		self.axes.append(tdax)
			

# class Map(object):

# 	"""
# 		docstring for Map
# 	"""
	
# 	def __init__(self, fig, ax):
# 		super(Map, self).__init__()
# 		self.fig = fig
# 		self.ax
		



















































# end of file