# Plotting wizard
import click
import scabbard as scb
import matplotlib.pyplot as plt
import dagger as dag
import numpy as np


# ANSI escape sequences for colors
RESET = "\x1b[0m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"


@click.command()
@click.argument('fname', type = str)
def simplemapwizard(fname):
	"""
	Launches a simple map visualization wizard for a given DEM file.

	This command-line tool loads a digital elevation model (DEM) from the specified file,
	creates a basic map visualization using `scabbard.Dplot.basemap`, and displays it.
	The plot remains open until the user presses Enter.

	Args:
		fname (str): The path to the DEM file.

	Returns:
		None: Displays a Matplotlib figure.

	Author: B.G.
	"""
	plt.ioff() # Turn off interactive plotting mode
	dem = scb.raster2RGrid(fname) # Load the DEM into a RegularGrid object
	atlas = scb.Dplot.basemap(dem) # Create a basic map visualization
	plt.show() # Display the plot
	input("press Enter to continue")










	


@click.command()
@click.option('-c', '--courant', 'courant',  type = float, default = 1e-3, help="Courant number for time step calculation.")
@click.option('-d', '--dt', 'dt',  type = float, default = None, help="Fixed time step for the simulation. If set, overrides Courant condition.")
@click.option('-P', '--precipitations', 'precipitations',  type = float, default = 30, help="Precipitation rate in mm/hr.")
@click.option('-m', '--manning', 'manning',  type = float, default = 0.033, help="Manning's roughness coefficient.")
@click.option('-S', '--SFD', 'SFD', type = bool, default=False, is_flag = True, help="Use Single Flow Direction (SFD) instead of Multiple Flow Direction (MFD).")
@click.option('-U', '--update_step', 'nupdate', type = int, default=10, help="Number of simulation steps between plot updates.")
@click.option('-exp', '--experimental', 'experimental', type = bool, default=False, is_flag = True, help="Enable experimental features.")
@click.argument('fname', type = str, help="Path to the DEM file.")
def graphflood_basic(fname,courant,dt,precipitations,manning,SFD,nupdate, experimental):
	"""
	Runs a basic GraphFlood simulation with specified parameters.

	This command-line tool initializes and runs a hydrological simulation using the
	GraphFlood model. It allows configuration of precipitation, time step (fixed or Courant-based),
	Manning's roughness, flow direction method (SFD/MFD), and visualization update frequency.

	Args:
		fname (str): Path to the DEM file.
		courant (float): Courant number for time step calculation.
		dt (float): Fixed time step for the simulation. If set, overrides Courant condition.
		precipitations (float): Precipitation rate in mm/hr.
		mannings (float): Manning's roughness coefficient.
		SFD (bool): If True, uses Single Flow Direction (SFD) instead of Multiple Flow Direction (MFD).
		nupdate (int): Number of simulation steps between plot updates.
		experimental (bool): If True, enables experimental features.

	Returns:
		None: Runs the simulation and displays plots.

	Author: B.G.
	"""
	print("EXPERIMENTAL IS ", experimental)

	P = precipitations /1e3/3600 # Convert precipitation from mm/hr to m/s
	if(dt is not None and courant == 1e-3):
		print(RED + "WARNING, dt set to constant value AND courant, is that normal?\n\n" + RESET)

	print("+=+=+=+=+=+=+=+=+=+=+=+=+")
	print("+=+=+=GRAPHFLOOD+=+=+=+=+")
	print("+=+=+=+=+=+=+=+=+=+=+=+=+\n\n")

	print("Precipitation rates = ", P, " m/s (",precipitations," mm/h)")


	mod = scb.ModelHelper() # Initialize a ModelHelper object
	mod.init_dem_model(fname, sea_level = 0., P = precipitations) # Initialize model from DEM

	# Get grid dimensions
	ny = mod.grid.ny
	nx = mod.grid.nx


	mod.courant = False if(dt is not None) else True # Set Courant flag based on dt input
	mod.stationary = True # Set stationary mode

	# Set Manning's friction coefficient
	mod.mannings = manning

	# Set flow direction method
	mod.SFD = SFD;


	mod.dt = dt if dt is not None else 1e-3 # Set time step
	mod.min_courant_dt = 1e-6 # Minimum Courant time step
	mod.courant_number = courant # Courant number

	ph = scb.PlotHelper(mod) # Initialize PlotHelper for visualization
	ph.init_hw_plot(use_extent = False) # Initialize water depth plot

	update_fig = nupdate # Figure update frequency
	i = 0
	j = 0
	while True:
		i+=1
		mod.run() if(experimental == False) else mod.gf.flood.run_hydro_only() # Run simulation step
		if(i % update_fig > 0):
			continue # Skip plot update if not at update step
		hw = mod.gf.flood.get_hw().reshape(mod.gf.grid.rshp) # Get water depth data
		print("Running step", i)
		ph.update()



@click.command()
@click.argument('fname', type = str, help="Path to the DEM file.")
@click.option('-p', '--precipitations', 'precipitations', type = float, default=30, help="Precipitation rate in mm/hr.")
@click.option('-d', '--dt', 'dt', type = float, default=1e-3, help="Time step for the simulation.")
@click.option('-v', '--visu', 'visu', type = bool, default=False, is_flag = True, help="Enable visualization during simulation.")
@click.argument('output', type = str, default='flow_depth.tif', help="Output filename for the final flow depth raster.")
def GPUgraphflood(fname, precipitations, dt, visu, output):
	"""
	Runs a GPU-accelerated GraphFlood simulation.

	This command-line tool initializes and runs a hydrological simulation using the
	Riverdale model, leveraging Taichi for GPU acceleration. It includes options for
	setting precipitation, time step, and visualizing convergence.

	Args:
		fname (str): Path to the DEM file.
		precipitations (float): Precipitation rate in mm/hr.
		dt (float): Time step for the simulation.
		visu (bool): If True, enables visualization during simulation.
		output (str): Output filename for the final flow depth raster.

	Returns:
		None: Runs the simulation, optionally displays plots, and saves the final water depth.

	Author: B.G.
	"""

	from scabbard.riverdale.rd_params import param_from_dem
	from scabbard.riverdale.rd_env import create_from_params
	import scabbard.riverdale.rd_morphodynamics as rdmo
	import scabbard.riverdale.rd_LM as rdlm
	import taichi as ti
	import time 

	ti.init(ti.vulkan) # Initialize Taichi with Vulkan backend

	param = param_from_dem(fname) # Create Riverdale parameters from DEM
	param.dt_hydro = dt # Set hydrological time step
	param.P = precipitations * 1e-3/3600 # Convert and set precipitation rate
	param.set_boundary_slope(0,mode = 'elevation') # Set boundary slope mode
	rd = create_from_params(param) # Create Riverdale model instance

	rdlm.priority_flood(rd,Zw = False) # Apply priority flood to topography

	if(visu):
		fig,ax = plt.subplots()
		imhw = ax.imshow(np.zeros( (param._ny, param._nx) ), cmap = "magma", vmin = 0.8, vmax = 1.2)
		plt.colorbar(imhw, label = "Convergence (1 is perfect)")
		fig.show()
		fig.canvas.draw_idle()
		fig.canvas.start_event_loop(0.01)

	NN = 1000 # Number of steps per iteration for printing progress
	for i in range(round(1e6)): # Run simulation for a large number of iterations
		st = time.time()
		# rdlm.priority_flood(rd)
		rd.run_hydro(NN) # Run hydrological simulation steps
		if(visu):
			data = rd.QwC.to_numpy()/rd.QwA.to_numpy() # Calculate Qw ratio for convergence visualization
			data[~np.isfinite(data)] = 1 # Handle non-finite values
			imhw.set_data(data) # Update visualization data

			fig.canvas.draw_idle()
			fig.canvas.start_event_loop(0.01)

		conv = rd.convergence_ratio # Get convergence ratio
		took = time.time() - st # Calculate time taken for iteration

		print('It:',i*NN+1,':',conv, " took ", took, "seconds") # Print progress
		if(conv > 0.90):
			break # Break if convergence is reached

	np.savez_compressed(f'results_{precipitations}.npz', {'hw':rd.hw.to_numpy()}) # Save final water depth

	fig,ax = plt.subplots()
	cb=ax.imshow(rd.hw.to_numpy(), cmap = 'Blues')
	plt.colorbar(cb, label = 'Flow depth (m)')
	plt.show()







@click.command()
@click.argument('fname', type = str, help="Path to the DEM file.")
def _debug_1(fname):
	"""
	Debug function for visualizing and dynamically updating a DEM.

	This function loads a DEM, displays it as a base map, and then continuously
	adds random noise to the topography, updating the plot in real-time.

	Args:
		fname (str): Path to the DEM file.

	Returns:
		None: Displays a Matplotlib figure with dynamic updates.

	Author: B.G.
	"""
	plt.ioff() # Turn off interactive plotting mode
	dem = scb.raster2RGrid(fname) # Load the DEM into a RegularGrid object
	atlas = scb.Dplot.basemap(dem) # Create a basic map visualization
	atlas.fig.show() # Display the figure
	plt.pause(0.01) # Pause briefly for initial rendering
	while(True):
		plt.pause(1) # Pause for 1 second
		dem.add_random_noise(-10,10) # Add random noise to the DEM
		atlas.update() # Update the plot
	input("press Enter to continue") # Pause execution until user input



@click.command()
def haguid():
	"""
	Launches the haGUId (Hydrological and Geomorphological User Interface).

	This command-line tool serves as the entry point for the scabbard GUI application.

	Returns:
		None: Launches the GUI application.

	Author: B.G.
	"""
	from scabbard.haguid import launch_haGUId

	launch_haGUId()

@click.command()
def run_nice_haguid():
	"""
	Runs the NiceGUI-based haGUId application.

	This command-line tool executes the `nice_haguid.py` script as a main module,
	launching the web-based graphical user interface.

	Returns:
		None: Launches the NiceGUI application.

	Author: B.G.
	"""
	import runpy
	import os
	import sys

	# Determine the path to the script directory
	script_dir = os.path.dirname(os.path.abspath(__file__))
	# Run the nice_haguid.py script as a main module
	runpy.run_path(os.path.join(script_dir, "nice_haguid.py"), run_name="__main__")

@click.command()
@click.argument('fname', type = str, help="Path to the 2D NumPy array file (.npy).")
def visu2Dnpy(fname):
	"""
	Launches a visualization tool for 2D NumPy arrays.

	This command-line tool loads a 2D NumPy array from the specified file
	and displays it in an interactive Matplotlib widget. It allows for colormap
	selection, range adjustment, and cross-section plotting.

	Args:
		fname (str): Path to the 2D NumPy array file (.npy).

	Returns:
		None: Displays a PyQt-based Matplotlib visualization window.

	Author: B.G.
	"""

	import sys
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_qtagg import FigureCanvas
	from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

	# from PySide6 import QtWidgets, QtCore
	from matplotlib.backends.qt_compat import QtWidgets, QtCore
	import matplotlib
	from matplotlib.lines import Line2D

	import scabbard.phineas_helper as phelp

	# Ensure using the QtAgg backend with PySide6
	matplotlib.use('QtAgg')

	class MatplotlibWidget(QtWidgets.QWidget):
		"""
		A PyQt widget that embeds Matplotlib figures for 2D NumPy array visualization.

		This widget displays a 2D NumPy array as an image and provides interactive
		features like colormap selection, color range adjustment, and cross-section plotting.

		Attributes:
			data (numpy.ndarray): The 2D NumPy array to visualize.
			cid (int): Connection ID for mouse motion event.
			cid2 (int): Connection ID for mouse button press event.
			figure1 (matplotlib.figure.Figure): Figure for the 2D array plot.
			ax1 (matplotlib.axes.Axes): Axes for the 2D array plot.
			canvas1 (matplotlib.backends.backend_qtagg.FigureCanvas): Canvas for figure1.
			figure2 (matplotlib.figure.Figure): Figure for the cross-section plot.
			ax2 (matplotlib.axes.Axes): Axes for the cross-section plot.
			canvas2 (matplotlib.backends.backend_qtagg.FigureCanvas): Canvas for figure2.
			imshow1 (matplotlib.image.AxesImage): Image object for the 2D array.
			colorbar1 (matplotlib.colorbar.Colorbar): Colorbar for imshow1.
			plot1 (matplotlib.lines.Line2D): Line indicating the cross-section location on imshow1.
			plot2 (list): List containing the Line2D object for the cross-section plot.
			colormapComboBox1 (QtWidgets.QComboBox): Dropdown for colormap selection.
			rangeSlider1 (phelp.RangeSlider): Custom slider for color range adjustment.
			button (QtWidgets.QPushButton): Button to toggle cross-section picking mode.

		Author: B.G.
		"""
	
		def __init__(self, data):
			"""
			Initializes the MatplotlibWidget.

			Args:
				data (numpy.ndarray): The 2D NumPy array to be visualized.
			"""
			super(MatplotlibWidget, self).__init__()

			self.data = data

			self.cid = None # Connection ID for mouse motion event
			self.cid2 = None # Connection ID for mouse button press event

			self.initUI() # Initialize the user interface
			self.setWindowTitle("Scabbard numpy 2D array explorer")
			 # Get the primary screen's geometry to size and position the window
			screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
			width, height = screen_geometry.width() * 0.8, screen_geometry.height() * 0.8

			# Calculate the position to center the window, then move it slightly down
			x = screen_geometry.width() * 0.1
			y = (screen_geometry.height() - height) / 2 + screen_geometry.height() * 0.05

			# Set the window geometry
			self.setGeometry(int(x), int(y), int(width), int(height))

		def initUI(self):
			"""
			Initializes the user interface components of the widget.

			Sets up the grid layout, creates Matplotlib figures and canvases,
			adds toolbars, initializes plots, and connects UI elements to their
			respective callback functions.
			"""
			# Main layout
			grid_layout = QtWidgets.QGridLayout(self)

			# Create two matplotlib figures and set their canvases
			self.figure1, self.ax1 = plt.subplots()
			self.canvas1 = FigureCanvas(self.figure1)
			self.figure2, self.ax2 = plt.subplots()
			self.canvas2 = FigureCanvas(self.figure2)

			# Add canvases and toolbars to the layout
			grid_layout.addWidget(self.canvas1, 1, 0)  # Row 1, Column 0 for 2D plot
			grid_layout.addWidget(NavigationToolbar(self.canvas1, self), 0, 0)  # Row 0, Column 0 for toolbar1
			grid_layout.addWidget(self.canvas2, 1, 1)  # Row 1, Column 1 for cross-section plot
			grid_layout.addWidget(NavigationToolbar(self.canvas2, self), 0, 1)  # Row 0, Column 1 for toolbar2

			# Initial plots
			self.imshow1 = self.ax1.imshow(self.data, cmap='magma')
			self.colorbar1 = self.figure1.colorbar(self.imshow1, ax=self.ax1)
			self.plot1 = Line2D([0,self.data.shape[1]], [0, 0], color='r', linestyle='--') # Cross-section line
			self.ax1.add_line(self.plot1)
			self.plot2 = self.ax2.plot(self.data[0,:], color = 'k', lw = 4) # Initial cross-section plot

			# Controls layout for the first plot
			controls_layout1 = QtWidgets.QHBoxLayout()
			grid_layout.addLayout(controls_layout1, 2, 0)  # Row 2, Column 0

			# Colormap selection for the first plot
			self.colormapComboBox1 = QtWidgets.QComboBox()
			self.colormapComboBox1.addItems(['magma', 'viridis', 'cividis', 'Blues', 'Reds', 'RdBu_r'])
			self.colormapComboBox1.currentIndexChanged.connect(lambda: self.update_plot('1'))
			controls_layout1.addWidget(self.colormapComboBox1)

			# Range selection slider for the first plot
			self.rangeSlider1 = phelp.RangeSlider(self.data.min(), self.data.max(), ID = '1')
			self.rangeSlider1.rangeChanged.connect(self._update_crange)
			controls_layout1.addWidget(self.rangeSlider1)

			# Controls layout for the second plot (cross-section)
			controls_layout2 = QtWidgets.QHBoxLayout()
			grid_layout.addLayout(controls_layout2, 2, 1)  # Row 2, Column 1

			# Button to toggle cross-section picking mode
			self.button = QtWidgets.QPushButton("Cross section",self)
			self.button.clicked.connect(self.switch_coordinate_picking)
			controls_layout2.addWidget(self.button)

			# Draw the canvases to display initial plots
			self.canvas1.draw()
			self.canvas2.draw()

		def update_plot(self, plt_ID):
			"""
			Updates the colormap of the specified plot.

			Args:
				plt_ID (str): Identifier for the plot to update ('1' for the 2D array plot).
			"""
			cmap1 = self.colormapComboBox1.currentText()

			if plt_ID == '1':
				self.imshow1.set_cmap(cmap1)


			# Redraw the canvases
			self.canvas1.draw()

		def _update_crange(self, vmin,vmax, plt_ID):
			"""
			Updates the color range (clim) of the specified plot.

			Args:
				vmin (float): New minimum value for the colormap.
				vmax (float): New maximum value for the colormap.
				plt_ID (str): Identifier for the plot to update ('1' for the 2D array plot).
			"""
			if plt_ID == '1':
				self.imshow1.set_clim(vmin,vmax)


			# Redraw the canvases
			self.canvas1.draw()

		def switch_coordinate_picking(self):
			"""
			Toggles the interactive cross-section picking mode.

			When activated, moving the mouse over the 2D plot updates the cross-section.
			Clicking the mouse deactivates the picking mode.
			"""
			# Disconnect existing connections if any
			if self.cid:
				self.canvas1.mpl_disconnect(self.cid)
				self.cid = None
			else:
				# Connect the mouse motion event to the on_motion function
				self.cid = self.canvas1.mpl_connect('motion_notify_event', self.on_motion)

		def plot_CS_from_coord(self, tx, ty):
			"""
			Plots a cross-section based on the given coordinates.

			Args:
				tx (float): X-coordinate of the mouse event.
				ty (float): Y-coordinate of the mouse event.
			"""
			row = round(ty) # Get the row index from the y-coordinate
			self.plot2[0].set_ydata(self.data[row,:]) # Update cross-section data
			self.ax2.relim() # Recalculate limits
			self.ax2.autoscale_view(True, True, True) # Autoscale view
			self.plot1.set_ydata([row,row]) # Update cross-section line position
			self.canvas1.draw() # Redraw 2D plot
			self.canvas2.draw() # Redraw cross-section plot


		def on_motion(self, event):
			"""
			Event handler for mouse motion.

			If the mouse is within the plot area, it connects a click event handler
			and updates the cross-section plot.

			Args:
				event (matplotlib.backend_bases.MouseEvent): The mouse event object.
			"""
			# Call the function with the coordinates if valid
			if event.xdata is not None and event.ydata is not None:
				self.cid2 = self.canvas1.mpl_connect('button_press_event', self.on_click) # Connect click event
				self.plot_CS_from_coord(event.xdata, event.ydata) # Update cross-section

		def on_click(self, event):
			"""
			Event handler for mouse click.

			Deactivates the cross-section picking mode and disconnects the click event handler.

			Args:
				event (matplotlib.backend_bases.MouseEvent): The mouse event object.
			"""
			# Call the function with the coordinates if valid
			if event.xdata is not None and event.ydata is not None:
				self.switch_coordinate_picking() # Toggle picking mode off
				if self.cid2:
					self.canvas1.mpl_disconnect(self.cid2) # Disconnect click event
					self.cid2 = None



	app = QtWidgets.QApplication(sys.argv) # Create the Qt application
	data = np.load(fname) # Load the 2D NumPy array
	if(len(data.shape) != 2):
		raise ValueError("Input NumPy array needs to be 2D.")
	main = MatplotlibWidget(data) # Create the main visualization widget
	main.show() # Show the widget
	sys.exit(app.exec_()) # Start the Qt application event loop
