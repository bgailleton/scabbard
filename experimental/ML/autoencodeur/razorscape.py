import numpy as np
import matplotlib.pyplot as plt
import scabbard as scb
import dagger as dag
import click as cli
import numba as nb
from perlin_noise import perlin_noise_2d


def checkdim(pop,minimum_size):
	nx,ny = pop.get_active_nx(),pop.get_active_ny()
	if (nx >= ny) and nx <= minimum_size:
		return False
	elif (nx < ny) and ny <= minimum_size:
		return False

	return True

def generate_landscape(initopo:np.ndarray, dx:float=100., Urate: (float | np.ndarray) = 1e-3, base_K:float = 1e-4, Kmod: (float | np.ndarray) = 1., precipitations: (float | np.ndarray) = 1., uplift: (float | np.ndarray) = 1., m:float=0.45, n:float=1.11, boundary_type:str='periodic_EW' , minimum_size:int = 64):

	# OG shape
	ny,nx = initopo.shape

	# First, check if nx and ny are multiples of 2
	assert (nx%2 == 0) and (ny%2 == 0)

	# perlin_noise_2d(nx, ny, scale=0.1, octaves=1, persistence=0.5, lacunarity=2.0, seed=0)
	rshp = (ny,nx)
	dy = dx
	
	# Creating the connector
	connector = dag.D8N(nx, ny, dx, dx, 0., 0.)
	connector.set_default_boundaries(boundary_type)

	# Generating the first graph
	graph = dag.graph(connector)

	# getting the popscape engine set up
	pop = dag.popscape(graph,connector)

	# Setting parameters
	pop.set_topo(initopo.ravel())
	pop.set_m(m)
	pop.set_n(n)
	pop.set_Kbase(base_K)
	pop.set_Kmod(Kmod) if isinstance(Kmod,np.ndarray) == False else pop.set_Kmod_variable(Kmod)
	pop.set_precip(precipitations) if isinstance(precipitations,np.ndarray) == False else pop.set_precip_variable(precipitations)
	pop.set_UE(Urate) if isinstance(Urate,np.ndarray) == False else pop.set_UE_variable(Urate)

	Nred = 0
	while checkdim(pop, minimum_size):
		Nred += 1
		pop.interpolation()
		pop.StSt(2)
		

	for i in range(Nred-1):
		pop.StSt(20)
		pop.restriction(1e-2)
	
	pop.smooth(1)
	pop.StSt(10)
	pop.restriction(1e-3)
	pop.smooth(1)
	pop.StSt(5)

	ret = pop.get_topo().reshape(ny,nx)

	return ret