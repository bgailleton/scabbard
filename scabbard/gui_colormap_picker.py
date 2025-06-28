'''
This module contains GUI widgets for selecting colormaps, designed for use with Matplotlib's Qt backend.
It provides a `ColorMapWidget` for easy selection of colormaps.
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

# Ensure using the QtAgg backend with PySide6
matplotlib.use('QtAgg')

def str2cmap(tstr):
	"""
	Converts a string to a valid Matplotlib colormap name.

	Args:
		tstr (str): The string representing the colormap name.

	Returns:
		str: The validated colormap name.

	Raises:
		ValueError: If the provided string is not a recognized Matplotlib colormap.

	Author: B.G.
	"""
	if(tstr in plt.colormaps()):
		return tstr
	else:
		raise ValueError(f"{tstr} not a matplotlib colormap, if it is a cmcrameri cmap I am in the process of adding them");


class ColorMapWidget(QtWidgets.QComboBox):
	"""
	A Qt ComboBox widget for selecting Matplotlib colormaps.

	This widget displays a list of common colormaps and emits a signal whenever
	a new colormap is selected by the user.

	Signals:
		colormapChanged (str): Emitted when the selected colormap changes, carrying the new colormap name as a string.

	Author: B.G.
	"""

	# Signal emitted when a new colormap is chosen
	colormapChanged = QtCore.Signal((str))

	def __init__(self):
		"""
		Initializes the ColorMapWidget.

		Sets up the QComboBox with a predefined list of colormaps and connects
		its `currentIndexChanged` signal to emit the `colormapChanged` signal.
		"""


		super(ColorMapWidget, self).__init__()
		# List of colormaps to display (can be extended)
		self.addItems(['gist_earth', 'magma', 'viridis', 'cividis', 'Blues', 'Reds', 'RdBu'])
		# Connect the built-in signal to the custom signal
		self.currentIndexChanged.connect( lambda : self.colormapChanged.emit(self.currentText()) )




# end of file
