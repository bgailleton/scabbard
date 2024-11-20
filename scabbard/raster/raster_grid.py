"""
Describes a raster-like regular grid.
Will replace the Rgrid object at some points
WIP
"""

import numpy as np
import matplotlib.pyplot as plt
from scabbard import io
from scabbard import geometry as geo
import dagger as dag
import scabbard as scb
from scipy.ndimage import gaussian_filter
import random
from copy import deepcopy


class RegularRasterGrid(object):
    """
    Manages a regular grid with helper functions
    """

    def __init__(self, value, geometry, dtype=np.float32):
        """ """

        super().__init__()

        # Check compiance with geometry:
        if isinstance(geometry, geo.RegularGeometry) == False:
            raise AttributeError(
                "a RegularRasterGrid object must be created with a geometry of type RegularGeometry"
            )
            # Checking if the geometry correspond to the shape of the array
            if Z.shape != geometry.shape:
                raise AttributeError(
                    "Matrix not the size indicated in geometry when trying to instanciate RegularRasterGrid"
                )

        # All good then I can instanciate the values
        self.Z = value
        self.geo = geometry

        # Converting type if needed
        if self.Z.dtype != dtype:
            self.Z = self.Z.astype(dtype)

    def duplicate_with_other_data(self, value):
        """
        Returns a copy of this raster with different data
        Useful to create another array with the same geometry and all

        Arguments:
                - value: the 2D numpy array replacing the value

        Returns:
                - A copy of the current raster with the new value field
        Authors:
                - B.G (last modifications 09/2024)
        """

        co = deepcopy(self)
        co.Z = value

        return co

    @property
    def dims(self):
        """
        topotoolbox-friendly dim parameter
        """
        return np.array([self.geo.ny, self.geo.nx], dtype=np.uint64)

    @property
    def rshp(self):
        """
        Arg to feed to np.reshape for 1D->2D
        """
        return np.array([self.geo.ny, self.geo.nx], dtype=np.uint64)

    def grid2ttb(self):
        '''
        Returns a libtopotoolbox-compatible grid
        '''
        import topotoolbox as ttb
        ttbgrid = ttb.GridObject()
        ttbgrid.z = self.Z
        ttbgrid.rows = self.geo.ny
        ttbgrid.columns = self.geo.nx
        ttbgrid.shape = self.Z.shape
        ttbgrid.cellsize = self.geo.dx
        ttbgrid.bounds = self.geo.extent
        ttbgrid.name = "from_scabbard"

        return ttbgrid

    # Common operator overload
    def __add__(self, other):
        if isinstance(other, RegularRasterGrid):
            return RegularRasterGrid(self.Z + other.Z, self.geo)
        else:
            return RegularRasterGrid(self.Z + other, self.geo)

    def __sub__(self, other):
        if isinstance(other, RegularRasterGrid):
            return RegularRasterGrid(self.Z - other.Z, self.geo)
        else:
            return RegularRasterGrid(self.Z - other, self.geo)

    def __mul__(self, other):
        if isinstance(other, RegularRasterGrid):
            return RegularRasterGrid(self.Z * other.Z, self.geo)
        else:
            return RegularRasterGrid(self.Z * other, self.geo)

    def __truediv__(self, other):
        if isinstance(other, RegularRasterGrid):
            return RegularRasterGrid(self.Z / other.Z, self.geo)
        else:
            return RegularRasterGrid(self.Z / other, self.geo)




def raster_from_array(Z, dx=1.0, xmin=0.0, ymin=0.0, dtype=np.float32):
    """
    Helper function to get a RegularRasterGrid from a 2D array

    Arguments:
            - Z: the 2D value
            - dx: the spatial step
            - xmin: left coordinate bound
            - ymin: bottom coordinate bound
            - dtype: numpy data type of Z

    Returns:
            - a RegularRasterGrid object

    Authors:
            - B.G. (last modification 09/2024)
    """
    geometry = scb.geometry.RegularGeometry(Z.shape[1], Z.shape[0], dx, xmin, ymin)
    return RegularRasterGrid(Z, geometry, dtype=dtype)

def raster_from_ttb(ttbgrid):
    '''
    Converts a libtopotoolbox gridObjg to scabbard grid object
    
    Arguments:
        - ttbgrid: the GRidObj from topotoolbox

    Returns:
        - A regularGridObject

    Authors:
        - B.G. (last modification: 11/2024)

    '''

    geometry = scb.geometry.RegularGeometry(ttbgrid.columns, ttbgrid.rows, ttbgrid.cellsize, ttbgrid.bounds[0], ttbgrid.bounds[1])
    return RegularRasterGrid(ttbgrid.z, geometry, dtype=ttbgrid.z.dtype)


