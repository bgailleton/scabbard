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
# Ensure using the Qt6Agg backend with PySide6
matplotlib.use('QtAgg')





# self.setWindowTitle("Scabbard numpy 2D array explorer")
#  # Get the primary screen's geometry
# screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
# width, height = screen_geometry.width() * 0.8, screen_geometry.height() * 0.8

# # Calculate the position to center the window, then move it slightly down
# x = screen_geometry.width() * 0.1
# y = (screen_geometry.height() - height) / 2 + screen_geometry.height() * 0.05

# # Set the window geometry
# self.setGeometry(int(x), int(y), int(width), int(height))