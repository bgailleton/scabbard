import scabbard as scb
import scabbard.steenbok as ste
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec
from opensimplex import noise2array

nx,ny = 200, 512
dx,dy = 4., 4.
noise = 0e-1
wnoise = 0e-2
S0 = 1e-2
# seed = 42
# np.random.seed(seed)

x,y = np.linspace(0,12, nx), np.linspace(0, 12,  ny)
pnoise = noise2array(x,y)
# print(pnoise)
# plt.imshow(pnoise)
# plt.show()
# quit()
# xx,yy = np.meshgrid(x,y)


surf = scb.generate_u_shaped_sloped_surface(nx, ny, dx, dy, slope=S0, Umag = 1) + np.random.rand(ny,nx)*noise/5 + pnoise * noise
env = scb.env_from_array(surf ,nx = nx, ny = ny, dx = dx, dy = dy, E="noflow", N="forcein", W="noflow", S='out')


env = scb.env_from_slope(noise_magnitude=wnoise, dx = dx, dy = dy, slope=S0, nx = nx,ny = ny,EW = 'noflow')
env.grid.Z2D[:] += pnoise * noise
# env.grid.Z2D[:, 70:130] -= 2

# plt.imshow(pnoise)
# plt.show()
# quit()


env.init_connector()
env.init_GF2()

paramgf = ste.ParamGf()
paramgf.dt_hydro = 1e-2
paramgf.stabilisator_gphydro = 0.
Qwtot = 50
propsed = 0.
paramgf.hydro_mode = ste.HydroMode.gp_static
# paramgf.hydro_mode = ste.HydroMode.gp_static_v3
# paramgf.hydro_mode = ste.HydroMode.gp_static
paramgf.hydro_mode = ste.HydroMode.gp_static_v4
# paramgf.hydro_mode = ste.HydroMode.gp_linear_test
# paramgf.hydro_mode = ste.HydroMode.gp_linear_test_v2
paramgf.hydro_mode = ste.HydroMode.static
paramgf.hydro_mode = ste.HydroMode.gp_static_v5
# paramgf.hydro_mode = ste.HydroMode.dynamic

paramgf.dt_morpho = 1
paramgf.morpho = True
paramgf.morpho_mode = ste.MorphoMode.eros_MPM
paramgf.morpho_mode = ste.MorphoMode.gp_morpho_v1
paramgf.morpho_mode = ste.MorphoMode.gp_morphydro_v1
# paramgf.morpho_mode = ste.MorphoMode.gp_morphydro_dyn_v1
paramgf.l_transp = 1e2
paramgf.k_lat = 0.
paramgf.bs_k = 0.
paramgf.calculate_MPM_from_D(4e-3)
# paramgf.k_erosion *= 0.1

paramgf.tau_c = 4
paramgf.k_erosion = 1e-4

paramgf.kz = paramgf.tau_c/(0.04 * 1650 * 9.8)
paramgf.kh = paramgf.tau_c/(0.08 * 1000 * 0.01 * 9.8)
# paramgf.kh = 0.7/(paramgf.tau_c * 1650 * 9.8)
# paramgf.kh *= 10
# paramgf.kh = paramgf.kz * 100

print( paramgf.kz, paramgf.kh)
# print((env.grid.Z2D[0,0] - env.grid.Z2D[1,0])/4)
# quit()


paramgf.boundary_slope = S0
# paramgf.k_erosion = 1e-12

mid = round(nx/2)
halfw = 30
# nodes = np.arange(1, nx-1)
nodes = np.arange(mid - halfw, mid + halfw)
QwI = np.zeros_like(nodes) + Qwtot/nodes.shape[0]
# QwI = scb.gaussian_spread_on_1D(X = nodes, M = Qwtot, x_c = mid, sigma = 10)

# plt.plot(QwI)
# plt.show()
# quit()
# QwI = np.array([Qwtot])
# nodes = np.array([100])

print('QI =', np.sum(QwI))
paramgf.set_input_points(nodes, QwI, QwI * propsed)

cuenv = ste.cuenv(env,'D4')
cuenv.setup_grid()
cuenv.setup_graphflood(paramgf)
# cuenv.graphflood_fillup(n_iterations = 5000)
cuenv.run_graphflood_fillup(n_iterations = 5000)


plt.ioff()

# fig = plt.subplots(1,2, figsize = (12,12))
fig = plt.figure(figsize = (12,12))
gs = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)


ax = fig.add_subplot(gs[:,0])
Z = cuenv._arrays['Z'].get().reshape(env.grid.rshp)
hw = cuenv._arrays['hw'].get().reshape(env.grid.rshp)

imZ = ax.imshow(Z, cmap = 'gist_earth', extent = env.grid.extent())
thw = hw.copy()
thw[thw < 0.05] = np.nan
imhw = ax.imshow(thw, cmap = 'Blues', vmax = 0.5, extent = env.grid.extent())

plt.colorbar(imhw)
plt.show(block = False)
fig.canvas.start_event_loop(0.001)


row = 50
ax2 = fig.add_subplot(gs[:2,1])
bd_pl = ax2.plot(Z[row,:], lw = 1, color = "k")
Zw_pl = ax2.plot(Z[row,:] + hw[row,:], lw = 1, color = "b")

col = 100
ax3 = fig.add_subplot(gs[2:,1])
bd_pl3 = ax3.plot(Z[:,col], lw = 1, color = "k")
Zw_pl3 = ax3.plot(Z[:,col] + hw[:,col], lw = 1, color = "b")


# ax.set_xlim(75,125)
# ax.set_ylim(200,-1)
# print('fasdasd')


# quit()
i = 0
while True:
	# time.sleep(0.4)
	i+=1
	if(i%10 == 0):
		print(i)
	Zb = np.copy(Z)

	cuenv.run_graphflood(n_iterations = 1000, verbose = False, nmorpho = 5)
	# cuenv.functions["uplift"](cuenv._arrays['Z']._gpu, cuenv._arrays['BC']._gpu, np.float32(5e-2), block = cuenv.gBlock, grid = cuenv.gGrid)
	# cuenv.functions["uplift_nolastrow"](cuenv._arrays['Z']._gpu, cuenv._arrays['BC']._gpu, np.float32(5e-1), block = cuenv.gBlock, grid = cuenv.gGrid)


	Z = cuenv._arrays['Z'].get().reshape(env.grid.rshp)
	hw = cuenv._arrays['hw'].get().reshape(env.grid.rshp)
	thw = hw.copy()
	thw[thw < 0.05] = np.nan
	# thw[thw < 50] = np.nan

	imZ.set_data(Z)
	imhw.set_data(thw)

	bd_pl[0].set_ydata(Z[row,:])
	Zw_pl[0].set_ydata(Z[row,:] + hw[row,:])

	ax2.set_ylim(np.nanmin(Z[row,1:-1]),np.nanmax(Z[row,1:-1] + hw[row,1:-1]))


	bd_pl3[0].set_ydata(Z[:,col])
	Zw_pl3[0].set_ydata(Z[:,col] + hw[:,col])

	ax3.set_ylim(np.nanmin(Z[1:-1,col]),np.nanmax(Z[1:-1,col] + hw[1:-1,col]))

	
	np.save('lastZ.npy',Z)
	np.save('lasthw.npy',hw)
	np.save('lastZw.npy',hw+Z)
	np.save('lastdZ.npy',Zb - Z)
	if(paramgf.morpho):
		np.save('lastQsA.npy',cuenv._arrays['QsA'].get().reshape(env.grid.rshp))
		np.save('lastQsB.npy',cuenv._arrays['QsB'].get().reshape(env.grid.rshp))
		np.save('lastQsC.npy',cuenv._arrays['QsC'].get().reshape(env.grid.rshp))
	# tqwa = cuenv._arrays['QsA'].get().reshape(env.grid.rshp)
	# tqwa = cuenv._arrays['QwC'].get().reshape(env.grid.rshp)
	# print(np.sum(tqwa[-1,:]))
	# print(np.sum(tqwa))

	# print(np.max(Z - Zb))
	# tqwa = np.log10(tqwa)
	# imhw.set_data(tqwa)
	# imhw.set_clim(np.nanmin(tqwa), np.nanmax(tqwa))
	name = str(i)
	while len(name) < 4:
		name = '0' + name
	plt.savefig(f'outtest/{name}.png')

	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.001)
	# input()



print("Done")























































# end of the file