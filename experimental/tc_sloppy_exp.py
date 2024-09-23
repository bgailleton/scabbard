import taichi as ti
import scabbard as scb
import numpy as np
import matplotlib.pyplot as plt

def calculate_MPM_from_D(D, l_transp, rho_water = 1000, gravity = 9.8, rho_sediment=2600, theta_c = 0.047):
	R = rho_sediment/rho_water - 1
	tau_c = (rho_sediment - rho_water) * gravity  * D * theta_c
	E_MPM = 8/(rho_water**0.5 * (rho_sediment - rho_water) * gravity)
	k_erosion = E_MPM/l_transp
	return k_erosion, tau_c


def f2img(tfield):
	out = tfield.to_numpy()
	return np.rot90((out - np.nanmin(out))/(np.nanmax(out) - np.nanmin(out)), -1)

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



dt = 1e-2
Qwtot = 50
manning = 0.033


l_transp = 1
gravity = 9.8
rho_water = 1000.
rho_sediment = 2560.
visc = 15.e-6 
D = 4e-3

K, tau_c = calculate_MPM_from_D(D, l_transp, rho_water = rho_water, gravity =gravity, rho_sediment=rho_sediment, theta_c = 0.047)


dt_morpho = 10

# rho_ratio = (rho_sediment - rho_water)/rho_water
# K = rho_ratio * gravity/(D * visc)
# K = 1e-4
# K*=1e-9
betalpha = 2
# tau_c = 4
mu_c = 0.7

# K = max(K,1e-6)
kz = K*1e-2


# paramgf.kz = paramgf.tau_c/(0.04 * 1650 * 9.8)
# paramgf.kh = paramgf.tau_c/(0.08 * 1000 * 0.01 * 9.8)

ti.init(arch=ti.gpu)  # Initialize Taichi to use the CPU


Z = ti.field(dtype=ti.f32, shape=(ny,nx))
Z.from_numpy(env.grid.Z2D)
hw = ti.field(dtype=ti.f32, shape=(ny,nx))
QwA = ti.field(dtype=ti.f32, shape=(ny,nx))
QwB = ti.field(dtype=ti.f32, shape=(ny,nx))
QwC = ti.field(dtype=ti.f32, shape=(ny,nx))

Ni = 25
Nid = np.arange(round(nx/2)-Ni, round(nx/2) + Ni, dtype = np.int32)
input_nodes = ti.field(dtype=ti.i32, shape=(Nid.shape[0]))
input_nodes.from_numpy(Nid)

tQw = np.full_like(Nid,Qwtot/Nid.shape[0],dtype = np.float32)
input_Qw = ti.field(dtype=ti.f32, shape=(Nid.shape[0]))
tQw = scb.gaussian_spread_on_1D(X = Nid, M = Qwtot, x_c = 100., sigma = 10)
input_Qw.from_numpy(tQw)

QsA = ti.field(dtype=ti.f32, shape=(ny,nx))
QsB = ti.field(dtype=ti.f32, shape=(ny,nx))
QsC = ti.field(dtype=ti.f32, shape=(ny,nx))

monitorer = ti.field(dtype=ti.f32, shape=(ny,nx))


# Initialize the field with some values (example)
@ti.kernel
def init_field():
	for i, j in Z:
		hw[i,j] = 0
		# Z[i,j] += (nx - 1 - i) * dx * S0
		QsA[i,j] = 0.
		QsB[i,j] = 0.

@ti.kernel
def stepinit():
	for i,j in Z:
		QwB[i,j] = 0
		QsB[i,j] = 0
		QsC[i,j] = 0
	for i in input_Qw:
		QwA[0, input_nodes[i]] = input_Qw[i]
		QwB[0, input_nodes[i]] = input_Qw[i]


# init_field()
# print((Z.to_numpy()[0,0] - Z.to_numpy()[1,0])/dx)
# # plt.imshow(Z.to_numpy())
# # plt.show()
# quit()


@ti.func
def Zw(i,j) -> ti.f32:
	return Z[i,j] + hw[i,j]

@ti.func
def Sw(i,j, ir,jr)->ti.f32:
	return (Zw(i,j) - Zw(ir,jr))/dx

@ti.func
def Psi(i,j) -> ti.f32:
	return ( gravity * (rho_sediment - rho_water) *Z[i,j] + Zw(i,j) * hw[i,j] * rho_water * gravity )/(gravity * (rho_sediment - rho_water) +  hw[i,j] * rho_water * gravity)

@ti.func
def SPsi(i,j, ir,jr)->ti.f32:
	return (Psi(i,j) - Psi(ir,jr))/dx

# @ti.func
# def Serr(i,j, ir,jr)->ti.f32:
# 	return (k_h * hw[i,j] * rho_water * Sw(i,j,ir,jr) - k_z * (rho_sediment - rho_water) * Sz(i,j,ir,jr))

@ti.func
def Sz(i,j, ir,jr)->ti.f32:
	return (Z[i,j] - Z[ir,jr])/dx

@ti.func
def Qw(i,j, tSw:ti.f32)->ti.f32:
	return dx/manning * ti.math.pow(hw[i,j],(5./3.)) * ti.math.pow(tSw,0.5)

@ti.func
def neighbour(i,j,k:int):
	ir,jr = -1,-1
	valid = True
	if(i == 0):
		if(j == 0 and k <= 1):
			valid = False
		elif(j == nx-1 and (k == 0 or k == 2)):
			valid = False
		elif(k==0):
			valid = False
	elif(j == 0 and k == 1):
		valid = False
	elif(j == nx-1 and k == 2):
		valid = False
	elif(i == ny-1):
		if(j == 0 and (k == 1 or k == 3)):
			valid = False
		elif(j == nx-1 and (k == 3 or k == 2)):
			valid = False
		elif(k==3):
			valid = False

	if(valid):
		if(k == 0):
			ir,jr = i-1, j
		if(k == 1):
			ir,jr = i, j-1
		if(k == 2):
			ir,jr = i, j+1
		if(k == 3):
			ir,jr = i+1, j
	return ir, jr



@ti.kernel
def compute_Qw():
	for i,j in Z:

		if(i == ny-1):
			continue

		Sws = ti.math.vec4(0.,0.,0.,0.)
		sumSw = 0.
		SSx = 0.
		SSy = 0.
		lockcheck = 0
		while(sumSw == 0.):
			lockcheck += 1
			for k in range(4):
				ir,jr = neighbour(i,j,k)
				if(ir == -1):
					continue

				tS = Sw(i,j,ir,jr)
				if(tS <= 0):
					continue

				if(k == 0 or k == 3):
					if(tS > SSy):
						SSy = tS
				else:
					if(tS > SSx):
						SSx = tS

				Sws[k] = tS
				sumSw += tS
			if(sumSw == 0.):
				hw[i,j] += 1e-4
			if(lockcheck > 10000):
				break

		gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
		QwC[i,j] = dx/manning * ti.math.pow(hw[i,j], 5./3) *sumSw/ti.math.sqrt(gradSw)
		# print(QwC[i,j])
		for k in range(4):
			ir,jr = neighbour(i,j,k)
			if(ir == -1):
				continue
			ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])

@ti.kernel
def compute_QwQs():
	for i,j in Z:

		if(i == ny-1):
			continue

		Sws = ti.math.vec4(0.,0.,0.,0.)
		SPsis = ti.math.vec4(0.,0.,0.,0.)
		SzS = ti.math.vec4(0.,0.,0.,0.)
		sumSw = 0.
		sumSPsi = 0.
		sumSZ = 0.
		
		SSx = 0.
		SSy = 0.

		SSgraderx = 0.
		SSgradery = 0.

		SSPsix = 0.
		SSPsiy = 0.
		# print("A")

		lockcheck = 0
		while(sumSw == 0.):
			lockcheck += 1
			sumSw = 0.
			sumSPsi = 0.
			for k in range(4):
				ir,jr = neighbour(i,j,k)
				if(ir == -1 or ir == 0):
					continue

				tS = Sw(i,j,ir,jr)
				tSPsi = SPsi(i,j,ir,jr)
				tSerr = Sz(i,j,ir,jr)

				if(k == 0 or k == 3):
					if(tS > SSy):
						SSy = tS
					if(tSerr > SSgradery):
						SSgradery = tSerr
					if(tSPsi > SSPsiy):
						SSPsiy = tSPsi
				else:
					if(tS > SSx):
						SSx = tS
					if(tSerr > SSgraderx):
						SSgraderx = tSerr
					if(tSPsi > SSPsix):
						SSPsix = tSPsi

				Sws[k] = max(0.,tS)
				SPsis[k] = max(0.,tSPsi)

				sumSw += Sws[k]
				sumSPsi += SPsis[k]

				SzS[k] = max(0., tSerr)
				sumSZ += SzS[k]

			if(sumSw == 0.):
				hw[i,j] += 1e-4
			if(lockcheck > 10000):
				break
			# if(sumSPsi == 0):
			# 	hw[i,j] += 1e-4
			# 	Z[i,j] += 1e-4
		# print("B")

		gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
		# graderr = ti.math.sqrt(SSgradery*SSgradery + SSgraderx*SSgraderx)
		
		graviterm = gravity * (rho_sediment - rho_water) * ti.math.sqrt(SSgradery*SSgradery + SSgraderx*SSgraderx)
		waterm = hw[i,j] * gradSw * rho_water * gravity 

		# graderr = graviterm + waterm

		gradSPsi = ti.math.sqrt(SSPsiy*SSPsiy + SSPsix*SSPsix)

		monitorer[i,j] = waterm - tau_c

		QwC[i,j] = dx/manning * ti.math.pow(hw[i,j], 5./3) * sumSw/ti.math.sqrt(gradSw)

		edot = K * ti.math.pow( ti.math.max(waterm - tau_c, 0.),1.5) + kz * ti.math.max(graviterm - mu_c,0.);

		# depcof = ti.math.pow(dx,2) * (gradSPsi/(sumSPsi * l_transp)) if sumSPsi > 0. else 0.;
		# depcof = ti.math.pow(dx,2) * (gradSw/(sumSw * l_transp)) if sumSPsi > 0. else 0.;
		depcof = ti.math.pow(dx,2) * (gradSw/(sumSZ * l_transp)) if sumSZ > 0. else 0.;

		# depcof = 0.

		QsC[i,j] = (QsA[i,j] + ti.math.pow(dx,2) * (edot))/(1 + depcof) if sumSZ > 0. else 0.;
	
		if(i >= ny-2 or i == 0):
			QsC[i,j] = QsA[i,j]



		# print(QwC[i,j])
		for k in range(4):
			ir,jr = neighbour(i,j,k)
			if(ir == -1):
				continue
			ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])
			if(sumSPsi > 0):
				# ti.atomic_add(QsB[ir,jr], SPsis[k]/sumSPsi * QsC[i,j])
				# ti.atomic_add(QsB[ir,jr], SPsis[k]/sumSPsi * QsC[i,j])
				ti.atomic_add(QsB[ir,jr], SzS[k]/sumSZ * QsC[i,j])


@ti.kernel
def compute_hw():
	for i,j in Z:
		QwA[i,j] = QwB[i,j]
		if(i == ny-1):
			continue
		hw[i,j] = ti.math.max(0.,hw[i,j] + (QwA[i,j] - QwC[i,j]) * dt/(dx*dy) ) 


@ti.kernel
def compute_hwhs():
	for i,j in Z:
		QwA[i,j] = QwB[i,j]
		
		if(i == ny-1):
			continue

		hw[i,j] = ti.math.max(0.,hw[i,j] + (QwA[i,j] - QwC[i,j]) * dt/(dx*dy) )
		
		if(i >= ny-2):
			QsA[i,j] = QsB[i,j]
			continue

		Z[i,j] = Z[i,j] + (QsA[i,j] - QsC[i,j]) * dt_morpho/(dx*dy)
		if(ti.math.isnan(Z[i,j])):
			print('NAN')
		QsA[i,j] = QsB[i,j]




# GUI setup
gui = ti.GUI("Field Visualization", res=(nx, ny))
init_field()
# run()
# quit()


print("YOLO")

# fig = plt.subplots(1,2, figsize = (12,12))
fig = plt.figure(figsize = (12,12))
gs = plt.GridSpec(ncols=3, nrows=4, figure=fig)
ax = fig.add_subplot(gs[:,0])
im = ax.imshow(hw.to_numpy(), cmap = "cividis", vmin = 0., vmax = 0.5)
ax2 = fig.add_subplot(gs[:,1:])

pl_cs = ax2.plot(env.grid.X, Z.to_numpy()[50,:], lw = 2, color = 'k')

fig.show()

Zori = Z.to_numpy()

it = 0
morpho = False
while True:
	it+=1
	
	for iii in range(1000):
		stepinit()
		compute_Qw()
		compute_hw()

		if(iii % 5 == 0 and True):
			stepinit()
			compute_QwQs()
			compute_hwhs()

	thw = hw.to_numpy()
	# thw = Z.to_numpy()
	# thw = Zori - Z.to_numpy()
	print(np.nanmax(monitorer.to_numpy()))
	im.set_data(thw)
	im.set_clim(np.nanmin(thw), np.nanmax(thw))

	tZ = Z.to_numpy()[50,:]
	pl_cs[0].set_ydata(tZ)
	ax2.set_ylim(np.nanmin(tZ)-0.1 * (np.nanmax(tZ) - np.nanmin(tZ)), np.nanmax(tZ)+0.1 * (np.nanmax(tZ) - np.nanmin(tZ)))

	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.001)
	np.save("lastZ.npy", Z.to_numpy())
	# plt.pause(0.001)


quit()


# Optionally, print or visualize your results
while gui.running:
	print("time:",clock)
	for _ in range(10000):
		clock += dt
		run_cpu(clock)

	gui.set_image(f2img(hw))
	# For visualizing the gradient magnitude, you could use: gui.set_image(gradient.to_numpy())
	# For a custom visualization (like direction), you'll need additional processing

	gui.show()  # Uptime the GUI window

# import matplotlib.pyplot as plt

# plt.imshow(Z.to_numpy())
# # plt.imshow(A_a.to_numpy())
# plt.show()