'''
Sets of functions to help with the GUI made with niceGUI
'''

import plotly.graph_objects as go
import cmcrameri.cm as cmc
from nicegui import ui
import numpy as np
from PIL import Image

def cmap2plotly(cmap = cmc.glasgow):
	'''
	Convert a matplotlib-type colormap into a plotly-compatible format
	
	Arguments:
		- cmap: the Matplotlib-like colormap (can be a cmcrameri one)
	
	Returns:
		- the plotly-compatible colormap
	
	Authors:
		- B.G. (last modifications: 06/2024)
	'''

	# Dummy structure for the RGB
	cmap_colors = cmap(np.linspace(0, 1, 256))

	# Colorscale is the plotly colormap
	colorscale = [[i / 255, f'rgb({r*255},{g*255},{b*255})'] for i, (r, g, b, _) in enumerate(cmap_colors)]

	return colorscale
	


def notify_WIP():
	ui.notify("Work In Progress...")

def quit_app():
	ui.run_javascript('window.close()')