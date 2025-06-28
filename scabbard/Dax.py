import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmcrameri as cm
import string
import random


class Dax(object):
	
	"""
	Base class for dynamic axes objects in Matplotlib.

	This class provides a common interface for objects that manage and update
	visualizations on a Matplotlib Axes object. It includes functionality for
	assigning a unique key and a z-order for rendering.

	Attributes:
		fig (matplotlib.figure.Figure): The parent Matplotlib Figure object.
		ax (matplotlib.axes.Axes): The Matplotlib Axes object this Dax manages.
		zorder (int): The z-order for rendering, controlling drawing order.
		key (str): A unique identifier for this Dax instance.

	Author: B.G.
	"""

	def __init__(self, ax, key = None, zorder = 1, fig = None):
		
		super(Dax, self).__init__()

		self.fig = fig
		
		self.ax = ax

		self.zorder = zorder
		
		# Generate a random key if none is provided	
		if(key is not None):
			self.key = key
		else:
			self.key = ''.join(random.choices(string.ascii_letters + string.digits, k=8))


	def __hash__(self):
		"""
		Returns the hash of the Dax object, based on its unique key.
		"""
		return hash(self.key)

	def update(self):
		"""
		Abstract method to update the visualization managed by this Dax.

		This method should be overridden by subclasses to implement specific
		update logic for their respective visualizations.
		"""
		pass


class callbax_image(object):
	"""
	Helper class for managing and updating Matplotlib imshow images.

	This class is designed to work with `Dax` objects to dynamically update
	an image displayed on an Axes, optionally adjusting its color limits.

	Attributes:
		im (matplotlib.image.AxesImage): The Matplotlib AxesImage object to update.
		callback (callable): A function that returns the new data for the image.
		rshp (tuple): The desired reshape tuple for the incoming data.
		clim (tuple, optional): A tuple (vmin, vmax) for fixed color limits. If None,
						color limits are automatically adjusted based on the new data.
		callback_params (tuple, optional): Arguments to pass to the callback function.

	Author: B.G.
	"""
	
	def __init__(self, im, callback, rshp, clim = None, callback_params = None):
		self.im = im
		self.callback = callback
		self.callback_params = callback_params
		self.rshp = rshp
		self.clim = clim

	def update(self):
		"""
		Updates the image data and optionally its color limits.

		Calls the callback function to get new data, reshapes it, and updates
		the `AxesImage` object. If `clim` is not set, it automatically adjusts
		the color limits based on the new data's min and max values.
		"""
		if self.callback_params is None :
			arr = self.callback().reshape(self.rshp)
		else:
			arr = self.callback(*self.callback_params).reshape(self.rshp)
		self.im.set_data(arr)
		if(self.clim is None):
			self.im.set_clim(np.nanmin(arr), np.nanmax(arr))

class callbax_sline(object):
	"""
	Helper class for managing and updating Matplotlib Line2D objects (streamlines).

	This class is designed to work with `Dax` objects to dynamically update
	a line plot displayed on an Axes, optionally adjusting its x and y limits.

	Attributes:
		ax (matplotlib.axes.Axes): The Matplotlib Axes object the line is on.
		sline (matplotlib.lines.Line2D): The Matplotlib Line2D object to update.
		callback (callable): A function that returns the new (x, y) data for the line.
		axylim (tuple, optional): A tuple (xmin, xmax, ymin, ymax) for fixed axis limits.
							If None, axis limits are automatically adjusted.
		axylim_ignore (bool, optional): If True, prevents automatic adjustment of axis limits.
							Defaults to False.

	Author: B.G.
	"""
	
	def __init__(self, ax, sline, callback, axylim = None, axylim_ignore = False):
		self.ax = ax
		self.sline = sline[0]
		self.callback = callback
		self.axylim = axylim
		self.axylim_ignore = axylim_ignore

	def update(self):
		"""
		Updates the line data and optionally its axis limits.

		Calls the callback function to get new (x, y) data and updates the `Line2D` object.
		If `axylim` is not set and `axylim_ignore` is False, it automatically adjusts
		the axis limits based on the new data's min and max values with a small padding.
		"""
		
		arr = self.callback()
		self.sline.set_xdata(arr[0])
		self.sline.set_ydata(arr[1])

		if(self.axylim is None and self.axylim_ignore == False):
			perc = 0.05 * (np.nanmax(arr[1]) - np.nanmin(arr[1]))
			self.ax.set_ylim(np.nanmin(arr[1]) - perc, np.nanmax(arr[1]) + perc)
			perc = 0.05 * (np.nanmax(arr[0]) - np.nanmin(arr[0]))
			self.ax.set_xlim(np.nanmin(arr[0]) - perc, np.nanmax(arr[0]) + perc)




class RGridDax(Dax):
	"""
	A Dax subclass for visualizing RegularGrid data with Matplotlib.

	This class extends `Dax` to specifically handle the display of `RegularGrid`
	objects, including their topography and an optional hillshade layer.

	Attributes:
		grid (scabbard.raster.RegularGrid): The RegularGrid object to visualize.
		base_ax (matplotlib.image.AxesImage): The Matplotlib AxesImage for the base topography.
		hillshade_on (bool): True if hillshade is enabled, False otherwise.
		clim (tuple, optional): A tuple (vmin, vmax) for fixed color limits of the topography.
							If None, color limits are automatically determined.
		hillshade_ax (matplotlib.image.AxesImage, optional): The Matplotlib AxesImage for the hillshade layer.

	Author: B.G.
	"""
	def __init__(self,Rgrid, ax, key = None, cmap = "gist_earth", hillshade = True, alpha_hillshade = 0.45, clim = None, callback_topo = None):
		
		super(RGridDax, self).__init__(ax, key)
		self.grid = Rgrid
		self.base_ax = self.ax.imshow(self.grid.Z2D, extent = self.grid.extent(), cmap = cmap, zorder =  self.zorder)
		self.hillshade_on = hillshade

		self.clim = clim if clim is not None else [self.grid.min(), self.grid.max()]

		if(hillshade):
			self.hillshade_ax = self.ax.imshow(self.grid.hillshade, extent = self.grid.extent(), cmap = "gray", alpha = alpha_hillshade, zorder = self.zorder)

		self.ax.set_xlabel("X (m)")
		self.ax.set_ylabel("Y (m)")

	def update(self):
		"""
		Updates the displayed topography and hillshade (if enabled).

		This method refreshes the image data for both the base topography and
		the hillshade layer based on the current state of the `grid` object.
		"""
		self.base_ax.set_data(self.grid.Z2D)
		if(self.hillshade_on):
			self.hillshade_ax.set_data(self.grid.hillshade)


	def drape_on(self, array, cmap = "gray", clim = None, delta_zorder = 1, alpha = 0.5, callback = None, callback_params = None):
		"""
		Overlays an array as an image on top of the existing topography visualization.

		This method allows for dynamic overlay of data (e.g., water depth, flow accumulation)
		on the topographic map. The overlay can be updated via a callback function.

		Args:
			array (numpy.ndarray): The 2D NumPy array to overlay.
			cmap (str, optional): Colormap for the overlaid image. Defaults to "gray".
			clim (tuple, optional): A tuple (vmin, vmax) for fixed color limits of the overlay.
							If None, color limits are automatically determined.
			delta_zorder (int, optional): Z-order offset for the overlay relative to the base topography.
							Defaults to 1.
			alpha (float, optional): Transparency of the overlaid image (0.0 to 1.0). Defaults to 0.5.
			callback (callable, optional): A function that returns the new data for the overlay.
							If None, the overlay remains static after initial drawing.
			callback_params (tuple, optional): Arguments to pass to the callback function.

		Returns:
			callbax_image: An instance of `callbax_image` managing the overlaid image,
						allowing for dynamic updates.

		Author: B.G.
		"""

		if clim is not None:
			vmin = clim[0]
			vmax = clim[1]
		else:
			vmin = np.nanmin(array)
			vmax = np.nanmax(array)


		ttaaxx = self.ax.imshow(array.reshape(self.grid.rshp), extent = self.grid.extent(), cmap = cmap, alpha = alpha, zorder = self.zorder + delta_zorder, vmin = vmin, vmax = vmax)

		if(callback == None):
			def nonecback():
				pass
			callback = nonecback


		return callbax_image(ttaaxx, callback, self.grid.rshp, clim = clim, callback_params = callback_params)


























































# dskjhksadfhla