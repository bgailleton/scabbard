# -*- coding: utf-8 -*-
"""
This package defines classes and functions for handling raster data.

It includes a core `RegularRasterGrid` class, a factory for creating raster objects,
and utilities for cropping rasters.
"""

# __author__ = "B.G."

from .raster_grid import *
from .raster_factory import *
from .std_raster_cropper import std_crop_raster