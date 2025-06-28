'''
This module contains helper functions for the NiceGUI-based scabbard application.
These functions abstract away some of the logic from the main GUI file (`nice_haguid.py`)
allowing the main file to focus on UI layout and event handling.
'''
import scabbard as scb
from nicegui import ui
import os
import numpy as np
from scabbard.riverdale.rd_params import param_from_dem
from scabbard.riverdale.rd_env import create_from_params, load_riverdale
import taichi as ti
import scabbard as scb
import matplotlib.pyplot as plt
import numpy as np
import time
import scabbard.riverdale.rd_grid as gridfuncs
import scabbard.riverdale.rd_hydrodynamics as hyd
import scabbard.riverdale.rd_hydrometrics as rdta
import scabbard.riverdale.rd_LM as lm
import scabbard.riverdale.rd_drainage_area as rda
from scabbard.riverdale.rd_hillshading import hillshading
import plotly.graph_objects as go
import cmcrameri.cm as cmc
import scabbard.nice_utils as nut
import sys


def _update_clim(stuff):
	"""
	Updates the color limits (zmin, zmax) of the main Plotly heatmap trace.

	This function is typically called when the user adjusts the colormap range sliders.
	It retrieves the current min/max values from the `stuff['model']['range']` dictionary
	and applies them to the Plotly figure's `datamap` trace.

	Args:
		stuff (dict): The global dictionary containing UI and model data.

	Returns:
		None

	Author: B.G.
	"""
	stuff['main_figure'].update_traces(
		zmin=stuff['model']['range']['min'],
		zmax=stuff['model']['range']['max'],
		selector=dict(name='datamap')
	)

	stuff['ui']['plot'].update()