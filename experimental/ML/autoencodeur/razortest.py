import numpy as np
import matplotlib.pyplot as plt
import scabbard as scb
import dagger as dag
import click as cli
import numba as nb
from perlin_noise import perlin_noise_2d
from ldnoise import generate_landscape_dataset_pattern
from razorscape import generate_landscape
import random

save = True
visu = False

def rng_complexity():
	return random.choice(['simple','medium','complex'])

def rng_BCs():
	return random.choice(['periodic_EW','periodic_NS','4edges'])

nx,ny = 2048, 2048
dx = 100.
initopo = perlin_noise_2d(nx, ny, scale=0.01, octaves=5, persistence=0.5, lacunarity=2.0, seed=0)

fig,ax = plt.subplots()

im = ax.imshow(initopo, cmap = 'terrain')

if(visu):
	fig.show()

for it in range(1000):
	# initopo = perlin_noise_2d(nx, ny, scale=0.01, octaves=5, persistence=0.5, lacunarity=2.0, seed=0)
	initopo = generate_landscape_dataset_pattern(nx, ny, pattern_type='topography', complexity=rng_complexity(), seed=None)
	# precipitations = perlin_noise_2d(nx, ny, scale=0.001, octaves=5, persistence=0.5, lacunarity=2.0, seed=51)
	precipitations = generate_landscape_dataset_pattern(nx, ny, pattern_type='precipitation', complexity=rng_complexity(), seed=None)
	range_P = random.uniform(0.05, 0.8)
	precipitations *= range_P
	precipitations += (1 - range_P/2)

	# Kmod = perlin_noise_2d(nx, ny, scale=0.05, octaves=5, persistence=0.5, lacunarity=2.0, seed=42)
	Kmod = generate_landscape_dataset_pattern(nx, ny, pattern_type='erodability', complexity=rng_complexity(), seed=None)
	range_K = random.uniform(0.05, 1.9)
	Kmod *= range_K
	Kmod += (1 - range_K/2)

	# UE = perlin_noise_2d(nx, ny, scale=0.001, octaves=5, persistence=0.5, lacunarity=2.0, seed=25)
	UE = generate_landscape_dataset_pattern(nx, ny, pattern_type='tectonic', complexity=rng_complexity(), seed=None)
	range_UE = random.uniform(0.5, 1.)
	UE *= range_UE
	UE += (1 - range_K/2)

	UE *= 1e-3


	tm = random.uniform(0.3,0.8)
	tn = tm/random.uniform(0.1,0.5)

	print('genscape')
	tbc = rng_BCs()
	topo = generate_landscape(initopo, dx=dx, Urate = UE, base_K = 1e-4, Kmod = Kmod, precipitations = precipitations, uplift = UE, m = tm, n=tn, boundary_type= tbc, minimum_size = 32)
	print('done')

	if(visu):
		ax.set_title(tbc)
		im.set_data(topo)
		im.set_clim(topo.min(), topo.max())
		fig.canvas.draw_idle()
		fig.canvas.start_event_loop(0.1)

	if(save):
		topo = (topo - topo.min())/(topo.max() - topo.min())
		np.save(f'./dataset/{str(it)}.npy', topo)