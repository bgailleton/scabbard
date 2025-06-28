'''
This module defines the main window class for the scabbard GUI, built using PyQt.
It provides a structured layout for adding and managing various widgets and sub-layouts.
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# from PySide6 import QtWidgets, QtCore
from matplotlib.backends.qt_compat import QtWidgets, QtCore
import matplotlib
# Ensure using the QtAgg backend with PySide6
matplotlib.use('QtAgg')



class MainWindow(QtWidgets.QMainWindow):

	"""
	The main window class for the scabbard graphical user interface.

	This class sets up the main application window, including its title, geometry,
	and a central widget with a grid layout. It provides methods for adding other
	widgets and layouts to organize the GUI elements.

	Attributes:
		centralWidget (QtWidgets.QWidget): The central widget of the main window.
		grid_layout (QtWidgets.QGridLayout): The main grid layout for organizing widgets.
		widgets (dict): A dictionary to store references to added widgets.
		layouts (dict): A dictionary to store references to added layouts.

	Author: B.G.
	"""

	
	def __init__(self, title = "Main Window"):
		"""
		Initializes the MainWindow.

		Args:
			title (str, optional): The title of the main window. Defaults to "Main Window".
		"""
		super(MainWindow, self).__init__()

		self.setWindowTitle(title)
		
		# Get the primary screen's geometry to size and position the window
		screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
		width, height = screen_geometry.width() * 0.9, screen_geometry.height() * 0.9

		# Calculate the position to center the window, then move it slightly down
		x = screen_geometry.width() * 0.05
		y = (screen_geometry.height() - height) / 2 + screen_geometry.height() * 0.02

		# Set the window geometry
		self.setGeometry(int(x), int(y), int(width), int(height))
		# Grid layout is used to organize everything by row and column
		self.centralWidget = QtWidgets.QWidget()
		self.grid_layout = QtWidgets.QGridLayout(self.centralWidget)
		# self.centralWidget.setLayout(self.grid_layout)
		self.setCentralWidget(self.centralWidget)
		self.widgets = {}
		self.layouts = {}
		self.layouts['main_grid'] = self.grid_layout


	def add_widget(self, ref, widget, row = None, col = None, rowmax = None, colmax = None, replace_ref = True, parentLayout = None):
		"""
		Adds a QWidget to the main window's layout.

		Args:
			ref (str): A unique reference string for the widget.
			widget (QtWidgets.QWidget): The QWidget instance to add.
			row (int, optional): The row index for the widget in the grid layout. Defaults to None.
			col (int, optional): The column index for the widget in the grid layout. Defaults to None.
			rowmax (int, optional): The number of rows the widget should span. Defaults to None.
			colmax (int, optional): The number of columns the widget should span. Defaults to None.
			replace_ref (bool, optional): If True, replaces an existing widget with the same `ref`.
								Defaults to True.
			parentLayout (str, optional): The reference of the parent layout to add the widget to.
								If None, adds to the main grid layout. Defaults to None.

		Returns:
			None

		Author: B.G.
		"""

		if ref in self.widgets.keys() and replace_ref:
			self.widgets[ref].deleteLater()
		
		self.widgets[ref] = widget

		adada = self.layouts[parentLayout] if parentLayout else self.grid_layout


		if(rowmax):
			adada.addWidget(widget, row, col,rowmax, colmax)
		elif(row):
			adada.addWidget(widget, row, col)
		else:
			adada.addWidget(widget)


		adada.addStretch(1)

	def add_layout(self, ref, layout, row, col,rowmax = None, colmax = None, replace_ref = True, parentLayout = None, style = None):
		"""
		Adds a QLayout to the main window's layout.

		Args:
			ref (str): A unique reference string for the layout.
			layout (QtWidgets.QLayout): The QLayout instance to add.
			row (int): The row index for the layout in the grid layout.
			col (int): The column index for the layout in the grid layout.
			rowmax (int, optional): The number of rows the layout should span. Defaults to None.
			colmax (int, optional): The number of columns the layout should span. Defaults to None.
			replace_ref (bool, optional): If True, replaces an existing layout with the same `ref`.
								Defaults to True.
			parentLayout (str, optional): The reference of the parent layout to add this layout to.
								If None, adds to the main grid layout. Defaults to None.
			style (str, optional): CSS-like style string to apply to the widget containing the layout.
							Defaults to None.

		Returns:
			None

		Author: B.G.
		"""

		if ref in self.layouts.keys() and replace_ref:
			self.layouts[ref].deleteLater()
		
		self.layouts[ref] = layout

		adada = self.layouts[parentLayout] if parentLayout else self.grid_layout

		twidget = QtWidgets.QWidget() # Create a QWidget to hold the layout
		twidget.setLayout(layout)
		if(style):
			twidget.setStyleSheet(style)


		if(rowmax):
			adada.addWidget(twidget, row, col,rowmax, colmax)
		else:
			adada.addWidget(twidget, row, col)