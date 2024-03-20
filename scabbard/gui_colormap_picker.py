'''
Contains generic widgets customisation for matplotlib qt backend
For example specific types of sliders (float, min max)
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# from PySide6 import QtWidgets, QtCore
from matplotlib.backends.qt_compat import QtWidgets, QtCore
import matplotlib
import cmcrameri as cm

# Ensure using the Qt6Agg backend with PySide6
matplotlib.use('QtAgg')

def str2cmap(tstr):
	'''
	Converts the string received from 
	'''
	if(tstr in plt.colormaps()):
		return tstr
	else:
		raise ValueError(f"{tstr} not a matplotlib colormap, if it is a cmcrameri cmap I am in the process of adding them");


class ColorMapWidget(QtWidgets.QComboBox):
	'''
		Colormap picker, a simple widget containing a list of colormap and sends a signals wiht the right colormap when it has changed
	'''

	colormapChanged = QtCore.Signal((str))

	def __init__(self):
		super(ColorMapWidget, self).__init__()
		self.addItems(['gist_earth', 'magma', 'viridis', 'cividis', 'Blues', 'Reds', 'RdBu'])
		self.currentIndexChanged.connect( lambda : self.colormapChanged.emit(self.currentText()) )




# end of file
