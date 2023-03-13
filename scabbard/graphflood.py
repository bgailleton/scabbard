'''
High level grpahflood object
'''
import dagger as dag
import scabbard as scb
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri as cm




class GraphFlood(object):

	"""
		docstring for GraphFlood
	"""

	def __init__(self, 
		grid,
		verbose = False,
		convergence_tracking = True,
		**kwargs
		):

		super(GraphFlood, self).__init__()
		
		# Grid object
		self.grid = grid

		# Checking and or feeding the graph
		if (self.grid.con is None and self.grid.graph is None):
			self.grid.compute_graphcon()
		elif (self.grid.con is not None and self.grid.graph is None):
			self.grid.graph = dag.graph(self.grid.con)
			self.grid.graph.compute_graph(self._Z, False, True)
		
		# Initialising the c++ graphflood object
		self.flood = dag.graphflood(self.grid.graph, self.grid.con)

		# feeding the topo
		self.flood.set_topo(self.grid._Z)

		# run all the configuration options
		self.config(**kwargs)

		self.active_figs = []
		self._callbacks = []

		self.verbose = verbose


		# time
		self.cumulative_time_hydro = 0.
		self.nit_hydro = 0
		self.cumulative_time_morpho = 0.

		# Convergence criteriae
		self.init_convergence_tracking(convergence_tracking)

	def config(self, **kwargs):
		'''
			Configures some of the general model options and inputs.
			Wrapper on the c++ object, not all options are in to keep things clear.
			Options:
				- dt or hydro_dt: hydrologic time ste
				- SFD (bool): if True set the mode to single flow, else multiple flow
				- minima (str): configure the method used for local minima resolution
					+ "reroute": reroute flow without filling the minimas
					+ "ignore": stop the flow at local minimas
					+ "fill" (default AND reccomended): fill the local minima with water

		'''

		if("SFD" in kwargs.keys()):
			if(kwargs["SFD"] == True):
				print("set flow conditions to single flow direction") if self.verbose else 0
				self.flood.enable_SFD()
			else:
				print("set flow conditions to mutliple flow direction") if self.verbose else 0
				self.flood.enable_MFD()

		if("minima" in kwargs.keys()):
			if(kwargs["minima"].lower() == "reroute"):
				self.flood.reroute_minima()
			elif(kwargs["minima"].lower() == "ignore"):
				self.flood.ignore_minima()
			else:
				self.flood.fill_minima()

		if("dt" in kwargs.keys()):
			self.flood.set_dt_hydro(kwargs["dt"])

		if("hydro_dt" in kwargs.keys()):
			self.flood.set_dt_hydro(kwargs["hydro_dt"])

		if("n_trackers" in kwargs.keys()):
			self.n_trackers = kwargs['n_trackers']

	def init_convergence_tracking(self, yes = True):
		self.convergence_trackers = yes
		self.n_trackers = 50
		self.n_pits = []
		self.dhw_monitoring = []
		self.Qwratio = []

		if(yes):
			self.flood.enable_Qwout_recording()
			self.flood.enable_dhw_recording()

	def set_precipitations(self,values = 1e-4):

		if(isinstance(values, np.ndarray)):
			if(np.prod(values.shape) == self.grid.nxy ):
				self.flood.set_water_input_by_variable_precipitation_rate(values.ravel())
			else:
				raise RuntimeError("array of precipitations needs to be of grid size or a single scalar")

		else:
			self.flood.set_water_input_by_constant_precipitation_rate(values)


	def set_input_discharge(self, nodes, values):
		'''
			Sets the input discharge at given locations (nodes need to be in flattened node indices)
		'''

		if(isinstance(nodes,list)):
			nodes = np.array(nodes, dtype=np.int32)
		if(isinstance(values,list)):
			values = np.array(values, dtype=np.float64)
		if(np.prod(nodes.shape) != np.prod(values.shape)):
			raise RuntimeError(f"nodes and values need to be the same size. Currently {np.prod(nodes)} vs {np.prod(values)}")

		# YOLO	
		self.flood.set_water_input_by_entry_points(values, nodes)



	def run_hydro(self, n_steps = 1, fig_update_step = 1, force_morpho = False):
		'''
		'''
		# Only hydro on this version
		if(force_morpho):
			self.flood.enable_morpho()
		else:
			self.flood.disable_morpho()
		

		# Running loop
		for i in range(n_steps):

			self.cumulative_time_hydro += self.flood.get_dt_hydro()
			self.nit_hydro += 1

			# Running hte actual model
			self.flood.run()

			# Balance checker for debugging purposes
			balance = abs(self.flood.get_tot_Qw_input() - self.flood.get_tot_Qwin_output())
			if(balance > 1):
				print("WARNING::Qw-in imbalance -> " + str(self.flood.get_tot_Qw_input()) + " got in and " + str(self.flood.get_tot_Qwin_output()) + " got out. Unbalance is: " + str(balance))


			# Monitoring the water convergence now:
			if(self.convergence_trackers):
				if(len(self.n_pits) == self.n_trackers):
					self.n_pits.pop(0)
					self.dhw_monitoring.pop(0)
					self.Qwratio.pop(0)

				self.n_pits.append(self.grid.graph.get_n_pits())
				arr = self.flood.get_dhw_recording()
				self.dhw_monitoring.append([np.percentile(arr,10), np.percentile(arr,25), np.percentile(arr,50), np.percentile(arr,75), np.percentile(arr,90)])
				arr = self.flood.get_Qwout_recording()
				mask = arr == 0
				arr = self.flood.get_Qwin()/arr
				arr[mask] = 1
				self.Qwratio.append([np.percentile(arr,10), np.percentile(arr,25), np.percentile(arr,50), np.percentile(arr,75), np.percentile(arr,90)])

				if(self.verbose):
					print("\n###############################")
					print("###############################")
					print("Monitoring results:")
					print("N pits:",self.n_pits[-1])
					print("delta hw:",self.dhw_monitoring[-1])
					print("Qwratio:",self.Qwratio[-1])
					print("###############################")
					print("###############################\n")

			

			# updating the figures
			if(i%fig_update_step == 0):
				self.update_figs()

	def update_figs(self):
		for tax in self._callbacks:
			tax.update()
		for tf in self.active_figs:
			tf.canvas.draw_idle()
			tf.canvas.start_event_loop(0.001)


	def get_xmonitor_HYDRO(self):
		tx1 = max(self.nit_hydro - self.n_trackers, 0)
		tx2 = self.nit_hydro
		# print("DEBUG_XMONO::",tx1,tx2)
		return np.arange(tx1 + 1, tx2 + 1)

	def get_monitor_pits(self):
		return [self.get_xmonitor_HYDRO(), self.n_pits]

	def debugyolo(self):
		return np.array(self.grid.graph.get_debug_mask())

	def pop_Qw_fig(self, jupyter = False, clim = None):

		if(jupyter == False):
			plt.ioff()

		fig,ax = plt.subplots()
		dax = scb.RGridDax(self.grid, ax, alpha_hillshade=1)
		arr = self.flood.get_Qwin()
		if(np.prod(arr.shape) == 0):
			arr = self.grid.zeros()
		else:
			arr.reshape(self.grid.rshp)

		Qwax = dax.drape_on(arr, cmap = "Blues", clim = clim, delta_zorder = 1, alpha = 0.7, callback = self.flood.get_Qwin)
		# Qwax = dax.drape_on(arr, cmap = "Blues", clim = None, delta_zorder = 1, alpha = 0.9, callback = self.debugyolo)

		self.active_figs.append(fig)
		self._callbacks.append(dax)
		self._callbacks.append(Qwax)
		fig.show()
		fig.canvas.draw_idle()
		fig.canvas.start_event_loop(0.001)

	def pop_hw_fig(self, jupyter = False, clim = None):

		if(jupyter == False):
			plt.ioff()

		fig,ax = plt.subplots()
		dax = scb.RGridDax(self.grid, ax, alpha_hillshade=1)
		arr = self.flood.get_Qwin()
		if(np.prod(arr.shape) == 0):
			arr = self.grid.zeros()
		else:
			arr.reshape(self.grid.rshp)

		Qwax = dax.drape_on(arr, cmap = "Blues", clim = clim, delta_zorder = 1, alpha = 0.7, callback = self.flood.get_hw)

		plt.colorbar(Qwax.im, label = "Water depth (m)")
		self.active_figs.append(fig)
		self._callbacks.append(dax)
		self._callbacks.append(Qwax)
		fig.show()
		fig.canvas.draw_idle()
		fig.canvas.start_event_loop(0.001)


	def pop_topo_fig(self, jupyter = False, clim = None):

		if(jupyter == False):
			plt.ioff()

		fig,ax = plt.subplots()
		dax = scb.RGridDax(self.grid, ax, alpha_hillshade=1)
		arr = self.flood.get_Qwin()
		if(np.prod(arr.shape) == 0):
			arr = self.grid.zeros()
		else:
			arr.reshape(self.grid.rshp)

		topo = dax.drape_on(arr, cmap = "gist_earth", clim = clim, delta_zorder = 1, alpha = 0.5, callback = self.flood.get_surface_topo)

		plt.colorbar(topo.im, label = "Surface (hw + Z) (m)")
		self.active_figs.append(fig)
		self._callbacks.append(dax)
		self._callbacks.append(topo)
		fig.show()
		fig.canvas.draw_idle()
		fig.canvas.start_event_loop(0.001)



	def pop_monitor_fig(self, jupyter = False):

		if(jupyter == False):
			plt.ioff()

		# # gridspec inside gridspec
		# fig = plt.figure()
		# gs0 = gridspec.GridSpec(1, 2, figure=fig)

		fig,ax1 = plt.subplots()

		ax2 = ax1.twinx()

		ax1.set_xlabel('N iterations')
		ax1.set_ylabel('N internal pits')

		l1 = ax1.plot([0],[0], lw = 2, color = 'r')

		dax1 = scb.callbax_sline(ax1, l1, self.get_monitor_pits, axylim = None)

		

		self.active_figs.append(fig)
		self._callbacks.append(dax1)
		# self._callbacks.append(topo)
		fig.show()
		fig.canvas.draw_idle()
		fig.canvas.start_event_loop(0.001)

























































#End of file