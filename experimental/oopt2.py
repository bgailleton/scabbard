import scabbard as scb
import scabbard.steenbok as ste
import numpy as np
import matplotlib.pyplot as plt


env = scb.env_from_DEM('/home/bgailleton/Desktop/data/green_river_1.tif')
env.init_connector()
env.init_GF2()

paramgf = ste.ParamGf()
paramgf.dt_hydro = 1e-3
paramgf.dt_morpho = 1e-2
paramgf.morpho = True
paramgf.Prate = 1e-4
paramgf.calculate_MPM_from_D(1e-3)

cuenv = ste.cuenv(env,'D4')
cuenv.setup_grid()
cuenv.setup_graphflood(paramgf)
cuenv.run_graphflood(n_iterations = 10, verbose = True)


plt.ioff()

fig,ax = plt.subplots()
Z = cuenv._arrays['Z'].get().reshape(env.grid.rshp)
hw = cuenv._arrays['hw'].get().reshape(env.grid.rshp)

imZ = ax.imshow(Z, cmap = 'gist_earth')
thw = hw.copy()
thw[thw < 0.05] = np.nan
imhw = ax.imshow(thw, cmap = 'Blues', vmax = 0.3)

plt.colorbar(imhw)
plt.show(block = False)
fig.canvas.start_event_loop(0.001)



# quit()

while True:
	# print("N")
	Zb = np.copy(Z)

	cuenv.run_graphflood(n_iterations = 1000, verbose = False)

	Z = cuenv._arrays['Z'].get().reshape(env.grid.rshp)
	hw = cuenv._arrays['hw'].get().reshape(env.grid.rshp)
	thw = hw.copy()
	thw[thw < 0.05] = np.nan

	imZ.set_data(Z)
	imhw.set_data(thw)

	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.001)
	print(np.max(Z - Zb))




print("Done")























































# end of the file