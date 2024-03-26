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

from scabbard.steenbok.gui_colormap_picker import str2cmap, ColorMapWidget

# Ensure using the Qt6Agg backend with PySide6
matplotlib.use('QtAgg')






class MapWidget(QtWidgets.QWidget):

	
	def __init__(self, env):
		super(MapWidget, self).__init__()

		# this widget is made to be connected to an environment, for updates
		self.env = env

		# grid layout is the easiest layout to organise everything by row and col
		grid_layout = QtWidgets.QGridLayout(self)

		# Create two matplotlib figures and set their canvases
		self.figure, self.ax = plt.subplots()
		self.canvas = FigureCanvas(self.figure)


		# Add canvases and toolbars to the layout
		grid_layout.addWidget(self.canvas, 1, 0)  # Row 0, Column 0
		grid_layout.addWidget(NavigationToolbar(self.canvas, self), 0, 0)  # Row 1, Column 0

		# Initial plots
		self.imshow1 = self.ax.imshow(self.env.grid.Z2D, cmap='gist_earth')

		# Controls layout for the first plot
		controls_layout1 = QtWidgets.QHBoxLayout()
		grid_layout.addLayout(controls_layout1, 2, 0)  # Row 2, Column 0

		


		# Colormap selection for the first plot
		self.colormapComboBox1 = QtWidgets.QComboBox()
		self.colormapComboBox1.addItems(['gist_earth', 'magma', 'viridis', 'cividis', 'Blues', 'Reds', 'RdBu_r'])
		self.colormapComboBox1.currentIndexChanged.connect(lambda: self.update_plot('1'))
		controls_layout1.addWidget(self.colormapComboBox1)

		#RangeSelection
		self.rangeSlider1 = phelp.RangeSlider(self.data.min(), self.data.max(), ID = '1')
		# grid_layout.addWidget(self.rangeSlider1,3,0)
		self.rangeSlider1.rangeChanged.connect(self._update_crange)
		controls_layout1.addWidget(self.rangeSlider1)



		# Controls layout for the second plot
		controls_layout2 = QtWidgets.QHBoxLayout()
		grid_layout.addLayout(controls_layout2, 2, 1)  # Row 2, Column 1

		# Colormap selection for the second plot
		# self.colormapComboBox2 = QtWidgets.QComboBox()
		# self.colormapComboBox2.addItems(['magma', 'viridis', 'cividis', 'Blues', 'Reds', 'RdBu_r'])
		# self.colormapComboBox2.currentIndexChanged.connect(lambda: self.update_plot('2'))
		# controls_layout2.addWidget(self.colormapComboBox2)

		self.button = QtWidgets.QPushButton("Cross section",self)
		self.button.clicked.connect(self.switch_coordinate_picking)
		controls_layout2.addWidget(self.button)

		# Draw the canvases
		self.canvas.draw()
		self.canvas2.draw()
