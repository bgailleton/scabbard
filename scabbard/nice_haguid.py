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



# Initializing global settings
# Dark mode is enabled by default for the UI
dark = ui.dark_mode()
dark.enable()

# Initializing Taichi with GPU backend by default (falls back to CPU if GPU is not available)
ti.init(ti.gpu)

# Global dictionary to store UI elements, model data, and other global variables.
# Callbacks and UI updates rely on retrieving and modifying values within this dictionary.
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

# Get the current working directory for file operations
CURRENT_PATH = os.getcwd()

def update_clim():
	"""
	Updates the color scale (colorbar range) of the main plot.

	This function is typically called when the user adjusts the color range sliders.
	It retrieves the current min/max values from the UI model and applies them to the plot.

	Returns:
		None

	Author: B.G.
	"""
	global stuff
	nhe._update_clim(stuff)
	

def update_plot_value(stuff, val, cmin = None, cmax = None):
	"""
	Updates the data being plotted on the main figure and adjusts the colormap range sliders.

	This function sets the new data (`val`) for the plot, calculates its min/max values,
	and updates the `z` property of the main Plotly heatmap trace. It also dynamically
	updates the range of the colormap sliders in the UI.

	Args:
		stuff (dict): The global dictionary containing UI and model data.
		val (numpy.ndarray): The new 2D NumPy array of data to be plotted.
		cmin (float, optional): A hardcoded minimum value for the colormap. If provided,
			it overrides the data's natural minimum. Defaults to None.
		cmax (float, optional): A hardcoded maximum value for the colormap. If provided,
			it overrides the data's natural maximum. Defaults to None.

	Returns:
		None

	Author: B.G.
	"""

	# Update the actual values to plot, note the [::-1] to orient Y to the north
	stuff['value'] = val
	# Determine the min and max values for the colormap
	tmin,tmax = np.nanmin(stuff['value']), np.nanmax(stuff['value'])
	if(cmin is not None):
		tmin = max(cmin, tmin)
	if(cmax is not None):
		tmax = min(cmax, tmax)
		
	# Store the determined min/max in the global model dictionary
	stuff['model']['range']['min'] = tmin
	stuff['model']['range']['max'] = tmax

	# Update the main Plotly figure's heatmap trace with new data and color range
	stuff['main_figure'].update_traces(
		z=stuff['value'],
		zmin=stuff['model']['range']['min'],
		zmax=stuff['model']['range']['max'],
		selector=dict(name='datamap')
	)

	# Dynamically update the colormap range sliders in the UI
	with stuff['ui']['r1c1']:
		ui.label('Colormap range')
		color_range = ui.range(min=stuff['model']['range']['min'], max=stuff['model']['range']['max'], 
			step = (stuff['model']['range']['max'] - stuff['model']['range']['min'])/250, 
			value = {'min': stuff['model']['range']['min'], 'max': stuff['model']['range']['max']}) \
		.props('label-always snap label-color="secondary" right-label-text-color="black"', ).bind_value(stuff['model'],'range').on('change', update_clim, throttle = 1)


def update_colorscale(stuff, cmap, title = "Colorbar"):
	"""
	Updates the colormap and colorbar title of the main plot.

	Args:
		stuff (dict): The global dictionary containing UI and model data.
		cmap (matplotlib.colors.Colormap or str): The new colormap to apply (can be a Matplotlib colormap object or its name).
		title (str, optional): The title for the colorbar. Defaults to "Colorbar".

	Returns:
		None

	Author: B.G.
	"""

	with stuff['ui']['r1c1']:
		# Update the main Plotly figure's heatmap trace with the new colorscale and colorbar title
		stuff['main_figure'].update_traces(
			colorscale=nut.cmap2plotly(cmap),colorbar=dict(title=title),
			selector=dict(name='datamap')
		)


def _update_tool_flowdepth():
	"""
	Updates the UI to display flow depth data and sets the appropriate colormap.

	This function is a callback for when the "Flow depth" tool is selected.
	It retrieves the water depth data from the Riverdale model, updates the plot,
	and sets the colormap to "Blues" with a specific title.

	Returns:
		None

	Author: B.G.
	"""
	global stuff
	# Update the plotted value with water depth data (inverted for correct orientation)
	update_plot_value(stuff, stuff['rd'].hw.to_numpy()[::-1], cmin = 0)
	# Update the colormap and colorbar title
	update_colorscale(stuff, plt.get_cmap("Blues"), title = 'Flow Depth (m)')
	# Redraw the plot
	stuff['ui']['plot'].update()

def _update_tool_topography():
	"""
	Updates the UI to display topography data and sets the appropriate colormap.

	This function is a callback for when the "Topography" tool is selected.
	It retrieves the topographic data from the Riverdale model, updates the plot,
	and sets the colormap to "gist_earth" with a specific title.

	Returns:
		None

	Author: B.G.
	"""
	global stuff
	# Update the plotted value with topography data (inverted for correct orientation)
	update_plot_value(stuff, stuff['rd'].Z.to_numpy()[::-1])
	# Update the colormap and colorbar title
	update_colorscale(stuff, plt.get_cmap("gist_earth"), title = 'Elevation (m)')
	# Redraw the plot
	stuff['ui']['plot'].update()

def _update_tool_flow_velocity():
	"""
	Updates the UI to display flow velocity data and sets the appropriate colormap.

	This function is a callback for when the "Flow Velocity" tool is selected.
	It computes the flow velocity from the Riverdale model, updates the plot,
	and sets the colormap to "gist_earth" with a specific title.

	Returns:
		None

	Author: B.G.
	"""
	global stuff
	# Update the plotted value with flow velocity data (inverted for correct orientation)
	update_plot_value(stuff, rdta.compute_flow_velocity(stuff['rd'])[::-1])
	# Update the colormap and colorbar title
	update_colorscale(stuff, plt.get_cmap("gist_earth"), title = 'u (m/s)')
	# Redraw the plot
	stuff['ui']['plot'].update()

def _update_tool_shear_stress():
	"""
	Updates the UI to display shear stress data and sets the appropriate colormap.

	This function is a callback for when the "Shear Stress" tool is selected.
	It computes the shear stress from the Riverdale model, updates the plot,
	and sets the colormap to "magma" with a specific title.

	Returns:
		None

	Author: B.G.
	"""
	global stuff
	# Update the plotted value with shear stress data (inverted for correct orientation)
	update_plot_value(stuff, rdta.compute_shear_stress(stuff['rd'])[::-1])
	# Update the colormap and colorbar title
	update_colorscale(stuff, plt.get_cmap("magma"), title = 'Shear stress')
	# Redraw the plot
	stuff['ui']['plot'].update()

def _update_tool_a_eff():
	"""
	Updates the UI to display effective drainage area data and sets the appropriate colormap.

	This function is a callback for when the "Effective Area" tool is selected.
	It computes the effective drainage area from the Riverdale model, updates the plot,
	and sets the colormap to "cividis" with a specific title.

	Returns:
		None

	Author: B.G.
	"""
	global stuff
	# Update the plotted value with effective drainage area data (inverted for correct orientation)
	update_plot_value(stuff, rdta.compute_effective_drainage_area(stuff['rd'])[::-1])
	# Update the colormap and colorbar title
	update_colorscale(stuff, plt.get_cmap("cividis"), title = 'Effective drainage area')
	# Redraw the plot
	stuff['ui']['plot'].update()





# Dictionary mapping tool names to their respective update functions
update_tool_func = {
	'Topography': _update_tool_topography,
	'Flow depth': _update_tool_flowdepth,
	'Shear Stress': _update_tool_shear_stress,
	'Flow Velocity': _update_tool_flow_velocity,
	'Effective Area': _update_tool_a_eff,
}
def update_tool(event):
	"""
	Callback function triggered when a new tool is selected from the dropdown list.

	This function clears the current tool options section in the UI and then dynamically
	adds the relevant widgets and updates the plot based on the selected tool.

	Args:
		event (dict or nicegui.events.ValueChangeEventArguments): The event object containing the selected tool's value.

	Returns:
		None

	Author: B.G.
	"""
	global stuff
	# Clear the current tool options section
	stuff['ui']['r1c1'].clear()
	# Get the selected tool name
	try:
		tool = event['value']
	except:
		tool = event.value

	# Re-add the tool selector dropdown to the cleared section
	with stuff['ui']['r1c1']:
		stuff['ui']['tool_selector'] = ui.select([i for i in update_tool_func.keys()], value = tool)
		stuff['ui']['tool_selector'].on_value_change(update_tool)

	# Call the appropriate update function for the selected tool
	if update_tool_func[tool] is not None:
		update_tool_func[tool]()


async def pick_file() -> None:
	"""
	Opens a local file picker dialog to load a Riverdale (RVD) file.

	This asynchronous function allows the user to select an RVD file from their local
	filesystem. Once a file is selected, it loads the Riverdale model, initializes
	the main plotting components, and sets up the initial topography visualization.

	Returns:
		None

	Author: B.G.
	"""

	global stuff
	# Open the file picker dialog
	result = await scb.local_file_picker(CURRENT_PATH, multiple=False)
	ui.notify(f'Loading {result[0]}')
	# Load the Riverdale model from the selected file
	stuff['rd'] = load_riverdale(result[0])
	stuff['value'] = stuff['rd'].Z.to_numpy() # Get initial topography data
	stuff['ui']['load_button'].delete() # Remove the load button after file is loaded
	ui.notify(f'Loaded !')

	# Setting up the Main GUI layout after file loading
	stuff['ui']['r1'] =  ui.row() # Main row for plot and controls
	with stuff['ui']['r1']:
		stuff['ui']['r1c0'] =  ui.column() # Column for the main plot
		stuff['ui']['r1c1'] = ui.column() # Column for tool options and sliders

	# Initialize Plotly figure
	stuff['main_figure'] = go.Figure()

	# Convert Matplotlib colormap to Plotly format
	colorscale = nut.cmap2plotly(cmap = cmc.batlowK)

	# Add topography heatmap trace
	stuff['main_figure'].add_trace(go.Heatmap(
		z=stuff['rd'].Z.to_numpy()[::-1], # Topography data (inverted Y-axis)
		colorscale=colorscale,
		zmin=1300,
		zmax=1320,
		zsmooth = 'best',
		colorbar=dict(title='elevation (m)'),
		name = 'datamap'
	))

	# Add hillshade heatmap trace
	stuff['main_figure'].add_trace(go.Heatmap(
		z=hillshading(stuff['rd'])[::-1], # Hillshade data (inverted Y-axis)
		colorscale='gray',
		opacity=0.45,
		zsmooth = 'best',
		showscale=False,
		name = 'hillshade'
	))

	# Update layout of the main figure
	stuff['main_figure'].update_layout(
		width=800,
		height=800,
		xaxis=dict(title='X'),
		yaxis=dict(title='Y'),
		margin=dict(l=20, r=20, t=20, b=20),

	)




	# Add the Plotly figure to the UI
	with stuff['ui']['r1c0']:
		stuff['ui']['plot'] = ui.plotly(stuff['main_figure']).classes('w-full h-40')

	# Initialize the tool options section with Topography selected
	update_tool({'value':'Topography'})



# First, define the main layout of the app

stuff['ui']['r0'] = ui.row() # Top row for title and global buttons

with stuff['ui']['r0']:
	ui.markdown('# GraphFlood - UI') # Application title
	ui.button('Quit', on_click=nut.quit_app) # Quit button
	ui.button('Documentation', on_click=nut.notify_WIP) # Documentation button (WIP)

# Initial load button for RVD files
stuff['ui']['load_button'] = ui.button('Load RVD file', on_click=pick_file, icon='folder')




if __name__ in {"__main__", "__mp_main__"}:
	ui.run()
