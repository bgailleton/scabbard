import scabbard as scb
import dagger as dag
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib
import random
from scipy.ndimage import gaussian_filter
from scabbard import ModelHelper, PlotHelper
%matplotlib widget

# Name of the DEM to load
fnamedem = "dem.tif"

# Graphflood object
mod = ModelHelper()

# precipitation rates in m/s
# P = 1e-4
# P = 2.7778e-6 #  => 10 mm/h
# P = 2.7778e-6/2 #  => 5 mm/h
# P = 5.5556e-6 #  => 20 mm/h
# P = 1.3889e-5 # => 30
# P = 1.1111e-5 # => 40 mm/h
# P = 1.3889e-5 # => 50 mm/h
# P = 2.7778e-5 # => 100 mm/h

P = 8.333e-6 # Precipitation rate in m/s (this is super high just for fun)
P = 50 * 1e-3/3600
mod.init_dem_model(fnamedem, sea_level = 0., P = P) 

# if courant is False, this will be used as time step
mod.dt = 5e-3

# Stationary = run to equillibrium
# Stationary = False propagate a trasient flood wave (the model has not been developped for that though)
mod.stationary = True 

# manning friction coeff, 0.033 is an OK value for open flow in channels:
mod.mannings = 0.033

# Single flow solver or Multiple flow solver?
# SFD: faster, less accurate
# MFD: slower, more accurate and prettier
mod.SFD = False

# Run for 500 iterations
for i in range(500):
	mod.run()

fig,ax = plt.subplots()
