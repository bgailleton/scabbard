import scabbard as scb
import numpy as np
import dagger as dag
import matplotlib.pyplot as plt
from cmcrameri import cm

class ModelHelper:

	def __init__(self):
		self.grid = None
		self.gf = None

		self.dx = 1
		self.dy = 1
		self.nx = 40
		self.ny = 200
		self.S0 = 1e-2
		self.P = 15


		self.mannings = 0.033
		self.dt = 1e-3
		self.courant = True
		self.courant_number = 0.1
		self.SFD = False
		self.min_courant_dt = 1e-4
		self.stationary = True




	def init_slope_model(self, nx, ny, dx, dy, S0, P):
		self.nx = nx
		self.ny = ny
		self.dx = dx
		self.dy = dy
		self.S0 = S0
		self.P = P
		nodes = np.arange(1,self.nx-1, dtype=np.int32) + self.nx
		hwins = self.P * np.ones_like(nodes)
		self.grid = scb.slope_RGrid(self.nx,self.ny,self.dx,self.dy, slope = self.S0, noise_magnitude=0., EW = "noflow", S = "can_out", N = "noflow")
		self.grid.graph.set_LMR_method(dag.LMR.priority_flood)
		# self.grid.graph.set_LMR_method(dag.LMR.priority_full_MFD)
		self.grid.Z2D[:,[0,-1]] += 10000 # Locking the borders
		self.grid.Z2D[0,:] += 5 # Locking the borders
		# self.grid.Z2D[1:-1,:] -= 0.1
		## graphflood python object
		self.gf = scb.GraphFlood(self.grid,verbose = False)
		self.gf.set_input_discharge(nodes, hwins, asprec=True)
		# IMPORTANT: setting boundary slope to bedrock slope
		self.gf.flood.set_fixed_slope_at_boundaries(self.S0)
		self.gf.flood.set_partition_method(dag.MFD_PARTITIONNING.PROPOSLOPE)
		self.apply_model_params()

	def init_dem_model(self, file_name, sea_level = 0., P = 1e-5):
		

		self.P = P

		self.grid = scb.raster2RGrid(file_name)
		
		self.grid.compute_graphcon()
		
		self.grid.graph.set_LMR_method(dag.LMR.priority_flood)

		self.nx = self.grid.nx
		self.ny = self.grid.ny
		self.dx = self.grid.dx
		self.dy = self.grid.dy
		self.S0 = 0.

		BCs = np.zeros((self.ny,self.nx), dtype = np.uint8) + 1
		BCs[[0,-1],:] = 3
		BCs[:, [0,-1]] = 3
		BCs[BCs <= 0] = 0
		self.grid.con.set_custom_boundaries(BCs.ravel())
		dag.set_BC_to_remove_seas(self.grid.con, self.grid._Z, sea_level)


		## graphflood python object
		self.gf = scb.GraphFlood(self.grid,verbose = False)
		# IMPORTANT: setting boundary slope to bedrock slope
		self.gf.flood.set_fixed_slope_at_boundaries(self.S0)
		self.gf.flood.set_partition_method(dag.MFD_PARTITIONNING.PROPOSLOPE)
		self.gf.set_precipitations(P)
		self.apply_model_params()

	def apply_model_params(self):
		self.gf.flood.set_mannings(self.mannings)	
		self.gf.flood.set_dt_hydro(self.dt)
		if(self.courant):
			self.gf.flood.enable_courant_dt_hydro()
			self.gf.flood.set_courant_numer(self.courant_number)
			self.gf.flood.set_min_courant_dt_hydro(self.min_courant_dt)
		if(self.SFD):
			self.gf.flood.enable_SFD()
		else:
			self.gf.flood.enable_MFD()

		if(self.stationary):
			self.gf.flood.enable_hydrostationary()
		else:
			self.gf.flood.disable_hydrostationary()

	@property
	def mannings(self):
		return self._mannings

	@mannings.setter
	def mannings(self,val):
		self._mannings = val
		if(self.gf is not None):
			self.gf.flood.set_mannings(self.mannings)

	@property
	def dt(self):
		return self._dt
	@dt.setter
	def dt(self,val):
		self._dt = val
		if(self.gf is not None):
			self.gf.flood.set_dt_hydro(self.dt)

	@property
	def courant(self):
		return self._courant

	@courant.setter
	def courant(self,val):
		self._courant = val
		if(self.gf is not None and val == True):
			self.gf.flood.enable_courant_dt_hydro()
			self.gf.flood.set_courant_numer(self.courant_number)
			self.gf.flood.set_min_courant_dt_hydro(self.min_courant_dt)

	@property
	def courant_number(self):
		return self._courant_number
		
	@courant_number.setter
	def courant_number(self,val):
		self._courant_number = val
		if(self.gf is not None):
			self.gf.flood.set_courant_numer(self.courant_number)

	@property
	def min_courant_dt(self):
		return self._min_courant_dt
		
	@min_courant_dt.setter
	def min_courant_dt(self,val):
		self._min_courant_dt = val
		if(self.gf is not None):
			self.gf.flood.set_min_courant_dt_hydro(self.min_courant_dt)

	@property
	def SFD(self):
		return self._SFD
	@SFD.setter
	def SFD(self,val):
		self._SFD = val
		if(self.gf is not None):
			if(self.SFD):
				self.gf.flood.enable_SFD()
			else:
				self.gf.flood.enable_MFD()

	@property
	def P(self):
		return self._P
	@P.setter
	def P(self,val):
		self._P = val
		if(self.gf is not None):
			self.gf.set_precipitations(P)

	@property
	def hw(self):

		ret = self.gf.flood.get_hw()
		if(ret.shape[0] > 0):
			ret = ret.reshape(self.rshp)
			return ret
		return np.zeros(self.rshp)

			

	@property
	def stationary(self):
		return self._stationary

	@stationary.setter
	def stationary(self,val):
		
		self._stationary = val

		if(self.gf is not None):
			if(self.stationary):
				self.gf.flood.enable_hydrostationary()
			else:
				self.gf.flood.disable_hydrostationary()

	@property
	def rshp(self):
		return (self.ny,self.nx)


	def run(self):
		self.gf.flood.run()




class PlotHelper(object):
	"""docstring for PlotHelper"""
	def __init__(self, mod):
		super(PlotHelper, self).__init__()
		self.mod = mod
		self.figs = []
		self.figaxel = {}
		self.hw_plot = False 	



	def init_hw_plot(self, alpha_hw = 0.75, cutoff = 0.1, cmap = "Blues", vmin = 0.1, vmax = 1.5, kwargs_subplots = {}, use_extent = True):

		self.hw_plot = True

		fig,ax = plt.subplots(*kwargs_subplots)
		textent = self.mod.grid.extent() if use_extent else None

		imhs = ax.imshow(self.mod.grid.hillshade, extent = textent, vmin =0, vmax = 1, cmap = cm.grayC_r)
		imhw = ax.imshow(self.mod.hw, extent = textent, vmin =vmin, vmax = vmax, cmap = "Blues", alpha = alpha_hw)
		self.figs.append(fig)
		self.figaxel["fig_hw"] = fig
		self.figaxel["ax_hw"] = ax
		self.figaxel["hs_hw"] = imhs
		self.figaxel["hw_hw"] = imhw
		fig.show()

	def update(self):

		if(self.hw_plot):
			self.figaxel["hs_hw"].set_data(self.mod.grid.hillshade)
			self.figaxel["hw_hw"].set_data(self.mod.hw)

		for fig in self.figs:
			fig.canvas.draw_idle()
			fig.canvas.start_event_loop(0.001)


