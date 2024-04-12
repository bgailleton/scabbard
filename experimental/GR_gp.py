import scabbard as scb
import scabbard.steenbok as ste
import numpy as np
import matplotlib.pyplot as plt


env = scb.env_from_DEM('/home/bgailleton/Desktop/data/green_river_1.tif')
nx,ny = env.grid.nx, env.grid.ny
dx,dy =  env.grid.dx, env.grid.dy
noise = 1e-3
S0 = 1e-2


env.init_connector()
env.init_GF2()

paramgf = ste.ParamGf()
paramgf.dt_hydro = 5e-3
paramgf.stabilisator_gphydro = 0.8
Qwtot = 80
paramgf.hydro_mode = ste.HydroMode.gp_static
paramgf.hydro_mode = ste.HydroMode.gp_static_v2
# paramgf.hydro_mode = ste.HydroMode.static
# paramgf.hydro_mode = ste.HydroMode.dynamic

paramgf.dt_morpho = 10
paramgf.morpho = False
paramgf.l_transp = 100
paramgf.k_lat = 0.
paramgf.boundary_slope = S0*0.9
paramgf.calculate_MPM_from_D(2e-3)


mid = round(nx/2)
halfw = 30
nodes = np.arange(mid - halfw, mid + halfw)
# QwI = scb.gaussian_spread_on_1D(X = nodes, M = Qwtot, x_c = mid, sigma = 10)
# print('QI =', np.sum(QwI))
# paramgf.set_input_points(nodes, QwI, QwI * 0.01)

cuenv = ste.cuenv(env,'D4')
cuenv.setup_grid()
cuenv.setup_graphflood(paramgf)
# cuenv.graphflood_fillup(n_iterations = 5000)

# cuenv.HHRG42fZwy()
# cuenv.DEBUGTIME_gphv3()
# quit()


plt.ioff()

fig,ax = plt.subplots(figsize = (5,12))
Z = cuenv._arrays['Z'].get().reshape(env.grid.rshp)
hw = cuenv._arrays['hw'].get().reshape(env.grid.rshp)

imZ = ax.imshow(Z, cmap = 'gist_earth')
thw = hw.copy()
thw[thw < 0.05] = np.nan
imhw = ax.imshow(thw, cmap = 'Blues', vmax = 0.4)

plt.colorbar(imhw)
plt.show(block = False)
fig.canvas.start_event_loop(0.001)



# quit()
i = 0
while True:
	i+=1
	if(i%10 == 0):
		print(i)
	Zb = np.copy(Z)

	cuenv.run_graphflood(n_iterations = 1, verbose = False, nmorpho = 5)
	# cuenv.functions["uplift"](cuenv._arrays['Z']._gpu, cuenv._arrays['BC']._gpu, np.float32(5e-2), block = cuenv.gBlock, grid = cuenv.gGrid)


	Z = cuenv._arrays['Z'].get().reshape(env.grid.rshp)
	hw = cuenv._arrays['hw'].get().reshape(env.grid.rshp)
	thw = hw.copy()
	thw[thw < 0.05] = np.nan
	# thw[thw < 50] = np.nan

	imZ.set_data(Z)
	imhw.set_data(thw)

	
	np.save('lastZ.npy',Z)
	np.save('lasthw.npy',hw)
	tqwa = cuenv._arrays['QwA'].get().reshape(env.grid.rshp)
	print(np.sum(tqwa[-1,:]))

	# print(np.max(Z - Zb))
	# imhw.set_data(tqwa)


	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.001)




print("Done")























































# end of the file