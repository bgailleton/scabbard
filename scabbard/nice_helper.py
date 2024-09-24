'''
Helper functions to offset the actual code from the nice_haguid and let it focus on layout
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
	stuff['main_figure'].update_traces(
		zmin=stuff['model']['range']['min'],
		zmax=stuff['model']['range']['max'],
		selector=dict(name='datamap')
	)

	stuff['ui']['plot'].update()