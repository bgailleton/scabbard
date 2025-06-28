'''
This module serves as a template for creating new GUI widgets and applications using PyQt and Matplotlib.
It includes basic imports and setup for a Matplotlib Qt backend.
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





