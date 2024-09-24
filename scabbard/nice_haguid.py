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
import scabbard.nice_helper as nhe
import sys



# initialising global stuff
# Dark mode on by default
dark = ui.dark_mode()
dark.enable()

# Initialising taichi also by default
ti.init(ti.gpu)

# Global variables
# callbacks rely on global variable retriavals 
# this dictionary just gathers all of them
stuff = {
	'ui':{},
	"fig" : None,
	"rd" : None,
	"colormap" : None,
	"im" : None,
	"color_range" : None,
	"value" : None,
	'model' : {"range": {"min": 0, "max": 100}},
}

# path of execution
CURRENT_PATH = os.getcwd()

def update_clim():
	'''
	Reinit the colorscale to the min/max of the plot
	'''
	global stuff
	nhe._update_clim(stuff)
	

def update_plot_value(stuff, val, cmin = None, cmax = None):
	'''
		Updates the plotted value: in the stuff, on the plots and the min-max of the colorscale
		Note: it does not call the plot.update function
	
	'''

	# Updating the actual values to plots, note the [::-1] to increase Y to the north
	stuff['value'] = val
	# Determining the min and max values
	tmin,tmax = np.nanmin(stuff['value']), np.nanmax(stuff['value'])
	if(cmin is not None):
		tmin = max(cmin, tmin)
	if(cmax is not None):
		tmax = min(cmax, tmax)
		
	# keeping in stuff
	stuff['model']['range']['min'] = tmin
	stuff['model']['range']['max'] = tmax

	# Updating the main figure
	stuff['main_figure'].update_traces(
		z=stuff['value'],
		zmin=stuff['model']['range']['min'],
		zmax=stuff['model']['range']['max'],
		selector=dict(name='datamap')
	)

	with stuff['ui']['r1c1']:
		ui.label('Colormap range')
		color_range = ui.range(min=stuff['model']['range']['min'], max=stuff['model']['range']['max'], 
			step = (stuff['model']['range']['max'] - stuff['model']['range']['min'])/250, 
			value = {'min': stuff['model']['range']['min'], 'max': stuff['model']['range']['max']}) \
		.props('label-always snap label-color="secondary" right-label-text-color="black"', ).bind_value(stuff['model'],'range').on('change', update_clim, throttle = 1)
		# # I'll need to make a factory for that 
	# ui.label('Colormap range')
	# color_range = ui.range(min=stuff['value'].min(), max=stuff['value'].max(), step = (stuff['value'].max() - stuff['value'].min())/250, value = {'min': 1000, 'max': 1400}) \
	# .props('label-always snap label-color="secondary" right-label-text-color="black"', ).bind_value(stuff['model'],'range').on('change', update_clim, throttle = 5)


def update_colorscale(stuff, cmap, title = "Colorbar"):

	with stuff['ui']['r1c1']:
		# Updating the main figure
		stuff['main_figure'].update_traces(
			colorscale=nut.cmap2plotly(cmap),colorbar=dict(title=title),
			selector=dict(name='datamap')
		)


def _update_tool_flowdepth():
	'''
	Sub-function updating the right column part of the tool to display the option for flow depth
	
	Arguments: None
	Returns: None
	Authors: None
	'''
	# Gathering the global stuff (lol)
	global stuff
	# Updating the value
	update_plot_value(stuff, stuff['rd'].hw.to_numpy()[::-1], cmin = 0)
	#Updating the colormap:
	update_colorscale(stuff, plt.get_cmap("Blues"), title = 'Flow Depth (m)')
	#redrawing hte plot
	stuff['ui']['plot'].update()

def _update_tool_topography():
	'''
	Sub-function updating the right column part of the tool to display the option for flow depth
	
	Arguments: None
	Returns: None
	Authors: None
	'''
	# Gathering the global stuff (lol)
	global stuff
	# Updating the value
	update_plot_value(stuff, stuff['rd'].Z.to_numpy()[::-1])
	#Updating the colormap:
	update_colorscale(stuff, plt.get_cmap("gist_earth"), title = 'Elevation (m)')
	#redrawing hte plot
	stuff['ui']['plot'].update()

def _update_tool_flow_velocity():
	'''
	Sub-function updating the right column part of the tool to display the option for flow depth
	
	Arguments: None
	Returns: None
	Authors: None
	'''
	# Gathering the global stuff (lol)
	global stuff
	# Updating the value
	update_plot_value(stuff, rdta.compute_flow_velocity(stuff['rd'])[::-1])
	#Updating the colormap:
	update_colorscale(stuff, plt.get_cmap("gist_earth"), title = 'u (m/s)')
	#redrawing hte plot
	stuff['ui']['plot'].update()

def _update_tool_shear_stress():
	'''
	Sub-function updating the right column part of the tool to display the option for flow depth
	
	Arguments: None
	Returns: None
	Authors: None
	'''
	# Gathering the global stuff (lol)
	global stuff
	# Updating the value
	update_plot_value(stuff, rdta.compute_shear_stress(stuff['rd'])[::-1])
	#Updating the colormap:
	update_colorscale(stuff, plt.get_cmap("magma"), title = 'Shear stress')
	#redrawing the plot
	stuff['ui']['plot'].update()

def _update_tool_a_eff():
	'''
	Sub-function updating the right column part of the tool to display the option for flow depth
	
	Arguments: None
	Returns: None
	Authors: None
	'''
	# Gathering the global stuff (lol)
	global stuff
	# Updating the value
	update_plot_value(stuff, rdta.compute_effective_drainage_area(stuff['rd'])[::-1])
	#Updating the colormap:
	update_colorscale(stuff, plt.get_cmap("cividis"), title = 'Effective drainage area')
	#redrawing the plot
	stuff['ui']['plot'].update()





update_tool_func = {
	'Topography': _update_tool_topography,
	'Flow depth': _update_tool_flowdepth,
	'Shear Stress': _update_tool_shear_stress,
	'Flow Velocity': _update_tool_flow_velocity,
	'Effective Area': _update_tool_a_eff,
}
def update_tool(event):
	'''
		This function is called when when the drop list is updated to a new too. It totally rewrite r1c1 with the new tool options
	'''
	global stuff
	# First, let's remove everything on that section
	stuff['ui']['r1c1'].clear()
	# Get the selected tool
	## Note: the try catch just depends on how the function is actually called
	try:
		tool = event['value']
	except:
		tool = event.value

	# Re-add the tool selector
	with stuff['ui']['r1c1']:
		stuff['ui']['tool_selector'] = ui.select([i for i in update_tool_func.keys()], value = tool)
		stuff['ui']['tool_selector'].on_value_change(update_tool)

	if update_tool_func[tool] is not None:
		update_tool_func[tool]()


async def pick_file() -> None:
	'''
	Load a file.
	Files supported so far:
		- RVD: the buit in riverdale's file format
		- TODO: load DEM from scratch and compute flow depth
	'''

	# just gathering the global dictionary
	global stuff
	# Getting the file name
	result = await scb.local_file_picker(CURRENT_PATH, multiple=False)
	ui.notify(f'Loading {result[0]}')
	# Actual loading operations
	stuff['rd'] = load_riverdale(result[0])
	stuff['value'] = stuff['rd'].Z.to_numpy()
	stuff['ui']['load_button'].delete()
	# Done with the loading
	ui.notify(f'Loaded !')

	# Setting up the Main GUI

	# Place holders:
	# Main row:
	stuff['ui']['r1'] =  ui.row()
	# Columns of main row
	with stuff['ui']['r1']:
		stuff['ui']['r1c0'] =  ui.column()
		stuff['ui']['r1c1'] = ui.column()

	stuff['main_figure'] = go.Figure()

	colorscale = nut.cmap2plotly(cmap = cmc.batlowK)

	# Add first image
	stuff['main_figure'].add_trace(go.Heatmap(
		z=stuff['rd'].Z.to_numpy()[::-1],
		colorscale=colorscale,
		zmin=1300,
		zmax=1320,
		zsmooth = 'best',
		# x0=0,
		# dx=500/A.shape[1],
		# y0=200,
		# dy=(800-200)/A.shape[0],
		colorbar=dict(title='elevation (m)'),
		name = 'datamap'
	))

	# Add second image
	stuff['main_figure'].add_trace(go.Heatmap(
		z=hillshading(stuff['rd'])[::-1],
		colorscale='gray',
		opacity=0.45,
		zsmooth = 'best',
		showscale=False,
		name = 'hillshade'
	))

	# Update layout
	stuff['main_figure'].update_layout(
		width=800,
		height=800,
		xaxis=dict(title='X'),
		yaxis=dict(title='Y'),
		margin=dict(l=20, r=20, t=20, b=20),

	)




	with stuff['ui']['r1c0']:
		stuff['ui']['plot'] = ui.plotly(stuff['main_figure']).classes('w-full h-40')

	# with stuff['ui']['r1c1']:
	update_tool({'value':'Topography'})



# First let's define the main layout of the app

stuff['ui']['r0'] = ui.row()

with stuff['ui']['r0']:
	ui.markdown('# GraphFlood - UI')
	ui.button('Quit', on_click=nut.quit_app)
	ui.button('Documentation', on_click=nut.notify_WIP)

stuff['ui']['load_button'] = ui.button('Load RVD file', on_click=pick_file, icon='folder')




# @ui.page('/')
# def index():
# with ui.row():
# 	ui.markdown('# Graphflood - Riverdale')
# 	ui.button('Quit', on_click=quit_app)
# load_button = ui.button('Choose file', on_click=pick_file, icon='folder')




if __name__ in {"__main__", "__mp_main__"}:
	ui.run()
