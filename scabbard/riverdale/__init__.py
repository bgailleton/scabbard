# -*- coding: utf-8 -*-
"""
This package provides the Riverdale hydrological model, implemented using Taichi.

It includes modules for defining the simulation environment, managing parameters,
performing hillshading, handling local minima, and calculating hydrometrics.
"""

# __author__ = "B.G."

from .rd_env import *
from .rd_params import *
from .rd_hillshading import hillshading, std_hillshading
from .rd_LM import priority_flood, smooth_hw, N_conv, compute_convergence
from .rd_hydrometrics import *
from . import rd_helper_surfw as helper