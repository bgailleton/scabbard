'''
This module contains Qt widgets specifically designed for displaying and interacting with maps,
particularly topographic maps from the scabbard environment.
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# from PySide6 import QtWidgets, QtCore
from matplotlib.backends.qt_compat import QtWidgets, QtCore
import matplotlib

import scabbard as scb

# from scabbard.steenbok.gui_colormap_picker import str2cmap, ColorMapWidget

# Ensure using the QtAgg backend with PySide6
matplotlib.use('QtAgg')






class MapWidget(QtWidgets.QWidget):
	"""
	A Qt widget for plotting and interacting with topographic maps from a scabbard environment.

	This widget displays a topographic map, optionally with hillshading and overlaid data
	(e.g., water depth). It includes a Matplotlib figure embedded within a Qt widget,
	along with a navigation toolbar.

	Attributes:
		env (scabbard.Environment): The scabbard environment object providing the map data.
		grid_layout (QtWidgets.QGridLayout): The main layout for organizing map components.
		figure (matplotlib.figure.Figure): The Matplotlib Figure object.
		ax1 (matplotlib.axes.Axes): The Matplotlib Axes object for plotting the map.
		canvas (matplotlib.backends.backend_qtagg.FigureCanvas): The canvas for the Matplotlib figure.
		imTopo (matplotlib.image.AxesImage): The image object for the base topography.
		imHS (matplotlib.image.AxesImage): The image object for the hillshade layer.
		drapePlot (matplotlib.image.AxesImage, optional): The image object for overlaid data (e.g., water depth).
		controls_layout (QtWidgets.QHBoxLayout): Layout for additional map controls.

	Author: B.G.
	"""

	
	def __init__(self, env):
		"""
		Initializes the MapWidget.

		Args:
			env (scabbard.Environment): The scabbard environment object to visualize.
		"""
		super(MapWidget, self).__init__()

		# This widget is made to be connected to an environment, for updates
		self.env = env

		# Grid layout is the easiest layout to organize everything by row and column
		self.grid_layout = QtWidgets.QGridLayout(self)

		# Create a Matplotlib figure and a single axes for plotting
		self.figure, self.ax1 = plt.subplots()
		self.canvas = FigureCanvas(self.figure)


		# Add canvases and toolbars to the layout
		self.grid_layout.addWidget(NavigationToolbar(self.canvas, self), 0, 0)  # Row 0, Column 0 for toolbar
		self.grid_layout.addWidget(self.canvas, 1, 0)  # Row 1, Column 0 for canvas

		# Initial plots for topography and hillshade
		self.imTopo = self.ax1.imshow(self.env.grid.Z2D, cmap='gist_earth')
		self.imHS = self.ax1.imshow(self.env.grid.hillshade, cmap='gray', alpha = 0.6)

		self.drapePlot = None # Placeholder for overlaid data plot

		self.ax1.set_xlabel("X (m)")
		self.ax1.set_ylabel("Y (m)")

		# Controls layout for additional widgets (e.g., sliders, buttons)
		self.controls_layout = QtWidgets.QHBoxLayout()
		self.grid_layout.addLayout(self.controls_layout, 2, 0)  # Row 2, Column 0 for controls



		# Draw the canvases to display initial plots
		self.canvas.draw()




def map_widget_from_fname(fname):
	"""
	Creates a `MapWidget` instance by loading a topographic map from a file.

	Args:
		fname (str): The filename of the topographic data to load.

	Returns:
		MapWidget: A `MapWidget` instance initialized with the loaded data.

	Author: B.G.
	"""
	
	env = scb.env_from_DEM(fname)

	return MapWidget(env)

def map_widget_from_fname_for_graphflood(fname):
	"""
	Creates a `MapWidget` instance from a file, specifically configured for GraphFlood visualization.

	This function loads a topographic map from a file and sets up the `MapWidget`
	with hillshade fully opaque and an overlaid transparent blue layer for water depth,
	which is typical for GraphFlood outputs.

	Args:
		fname (str): The filename of the topographic data to load.

	Returns:
		MapWidget: A `MapWidget` instance configured for GraphFlood visualization.

	Author: B.G.
	"""
	
	env = scb.env_from_DEM(fname)
	mapw = MapWidget(env)
	mapw.imHS.set_alpha(1.) # Set hillshade to fully opaque
	mapw.drapePlot = mapw.ax1.imshow(np.zeros_like(env.grid.Z2D), cmap='Blues', alpha = 0.6, vmin = 0, vmax = 0.8) # Add transparent blue layer for water
	plt.colorbar(mapw.drapePlot, label = 'Flow depth (m)') # Add colorbar for water depth

	return mapw