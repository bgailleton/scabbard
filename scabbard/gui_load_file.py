'''
This module provides a Qt widget for loading files, specifically designed for use with
Matplotlib's Qt backend.
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






class FileLoader(QtWidgets.QWidget):

	theChosenFile = QtCore.Signal((str))
	"""
	A custom Qt widget for selecting and loading files.

	This widget provides a button that, when clicked, opens a file dialog
	allowing the user to select a file. The path of the chosen file is then
	emitted via the `theChosenFile` signal.

	Signals:
		theChosenFile (str): Emitted when a file is successfully selected, carrying the file path as a string.

	Author: B.G.
	"""
	
	def __init__(self):
		"""
		Initializes the FileLoader widget.

		Sets up the layout and the "Open File" button, connecting its click event
		to the `openFileDialog` method.
		"""
		super(FileLoader, self).__init__()
		self.layout = QtWidgets.QVBoxLayout(self)

		# Button to open the file dialog
		self.openButton = QtWidgets.QPushButton("Open File")
		self.openButton.clicked.connect(self.openFileDialog)
		self.layout.addWidget(self.openButton)
		self.setMaximumSize(100,100)

	def openFileDialog(self):
		"""
		Opens a file dialog and emits the selected file's path.

		This method is triggered when the "Open File" button is clicked.
		It uses `QtWidgets.QFileDialog.getOpenFileName` to allow the user to select a file.
		If a file is chosen, its path is emitted through the `theChosenFile` signal.
		"""
		# Open a file dialog and print the selected file path
		filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
		self.theChosenFile.emit(filename)
		if filename:
			print(f"Selected file: {filename}")