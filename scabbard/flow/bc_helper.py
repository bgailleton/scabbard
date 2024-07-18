'''
Help with managing boundary conditions
Weither it is about automatically generating them or manage more complex cases like no data and all


B.G.
'''

import numpy as np
import scabbard._utils as ut
import dagger as dag


def get_normal_BCs(nx,ny):
	return ut.normal_BCs_from_shape(nx,ny)


