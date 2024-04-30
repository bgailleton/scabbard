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
dx,dy = 1, 1
noise = 0e-1
wnoise = 1e-1
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


CFsL = False
CFsL_hydro = 5e-5
dt = 2e-4
Qwtot = 20
manning = 0.033

min_val = 0.5  # New minimum value
max_val = 2.5    # New maximum value
omega = 0.05

# Calculate new amplitude and baseline
A = (max_val - min_val) / 2
B = (max_val + min_val) / 2


morpho = True
NHYDRO = 100
l_transp = 10
gravity = 9.8
rho_water = 1000.
rho_sediment = 2560.
visc = 15.e-6 
D = 4e-3
csin = 0.01
rho_ratio = (rho_sediment - rho_water)/rho_water

# K, tau_c = calculate_MPM_from_D(D, l_transp, rho_water = rho_water, gravity =gravity, rho_sediment=rho_sediment, theta_c = 0.047)
Le = 0.5*D*rho_ratio
# print(np.pi * Le/S0)


dt_morpho = 5e-5
CFsL_morpho = 5e-6

K = rho_ratio * gravity * D**2/(visc)
# K = 1e-4
# K*=1e-9
betalpha = 2
tau_c = 0.7
# mu_c = 0.7

# print(K)
# quit()

kz = 1.


# paramgf.kz = paramgf.tau_c/(0.04 * 1650 * 9.8)
# paramgf.kh = paramgf.tau_c/(0.08 * 1000 * 0.01 * 9.8)

ti.init(arch=ti.gpu)  # Initialize Taichi to use the CPU


Z = ti.field(dtype=ti.f32, shape=(ny,nx))
Z.from_numpy(env.grid.Z2D)
hw = ti.field(dtype=ti.f32, shape=(ny,nx))
Psi_surf = ti.field(dtype=ti.f32, shape=(ny,nx))
fac = ti.field(dtype = ti.f32, shape = ())
maxhwsw = ti.field(dtype = ti.f32, shape = ())



QwA = ti.field(dtype=ti.f32, shape=(ny,nx))
QwB = ti.field(dtype=ti.f32, shape=(ny,nx))
QwC = ti.field(dtype=ti.f32, shape=(ny,nx))

Ni = 25
Nid = np.arange(round(nx/2)-Ni, round(nx/2) + Ni, dtype = np.int32)
input_nodes = ti.field(dtype=ti.i32, shape=(Nid.shape[0]))
input_nodes.from_numpy(Nid)

tQw = np.full_like(Nid,Qwtot/Nid.shape[0],dtype = np.float32)
input_Qw = ti.field(dtype=ti.f32, shape=(Nid.shape[0]))
# tQw = scb.gaussian_spread_on_1D(X = Nid, M = Qwtot, x_c = 100., sigma = 10)
input_Qw.from_numpy(tQw)

tQs = tQw * csin
input_Qs = ti.field(dtype=ti.f32, shape=(Nid.shape[0]))
# tQs = scb.gaussian_spread_on_1D(X = Nid, M = Qstot, x_c = 100., sigma = 10)
input_Qs.from_numpy(tQs)

QsA = ti.field(dtype=ti.f32, shape=(ny,nx))
QsB = ti.field(dtype=ti.f32, shape=(ny,nx))
QsC = ti.field(dtype=ti.f32, shape=(ny,nx))

monitorer = ti.field(dtype=ti.f32, shape=(ny,nx))


@ti.kernel
def stochfac():
	fac[None] += 0.1*ti.random()-0.2
	fac[None] = max(0.5,fac[None])
	fac[None] = min(2,fac[None])
	# print(fac)

@ti.kernel
def set_fac(nfac:ti.f32):
	fac[None] = nfac

# Initialize the field with some values (example)
@ti.kernel
def init_field():
	fac[None] = 1
	for i, j in Z:
		hw[i,j] = 0
		# Z[i,j] += (nx - 1 - i) * dx * S0
		QsA[i,j] = 0.
		QsB[i,j] = 0.

@ti.kernel
def stepinit():
	maxhwsw[None] = 0
	for i,j in Z:
		QwB[i,j] = 0
		QsB[i,j] = 0
		QsC[i,j] = 0
	for i in input_Qw:
		QwA[0, input_nodes[i]] = input_Qw[i] * fac[None]
		QwB[0, input_nodes[i]] = input_Qw[i] * fac[None]
		QsA[0, input_nodes[i]] = input_Qs[i] * fac[None]
		QsB[0, input_nodes[i]] = input_Qs[i] * fac[None]
		# print('fac',fac[None])




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
	return ( betalpha * hw[i,j]/(rho_ratio * D) * Zw(i,j) + kz * Z[i,j])/(kz + betalpha * hw[i,j]/(rho_ratio * D))

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

					if(CFsL):
						ti.atomic_max(maxhwsw[None], hw[i,j] * Sws[k])
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
		sumSw = 0.
		sumSPsi = 0.
		
		SSx = 0.
		SSy = 0.

		SSgradZx = 0.
		SSgradZy = 0.

		SSPsix = 0.
		SSPsiy = 0.
		# print("A")
		hflow = 0;

		Psi_surf[i,j] = Psi(i,j)

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
					if(tSerr > SSgradZy):
						SSgradZy = tSerr
					if(tSPsi > SSPsiy):
						SSPsiy = tSPsi
				else:
					if(tS > SSx):
						SSx = tS
					if(tSerr > SSgradZx):
						SSgradZx = tSerr
					if(tSPsi > SSPsix):
						SSPsix = tSPsi

				Sws[k] = max(0.,tS)
				SPsis[k] = max(0.,tSPsi)

				sumSw += Sws[k]
				sumSPsi += SPsis[k]

				if(CFsL):
					ti.atomic_max(maxhwsw[None], hw[i,j] * Sws[k])

				if(tS>0):
					hflow = ti.max(hflow, hw[i,j] - ti.max(0., Z[ir,jr] - Z[i,j]) )

			if(sumSw == 0.):
				hw[i,j] += 1e-4
			if(lockcheck > 10000):
				break
			# if(sumSPsi == 0):
			# 	hw[i,j] += 1e-4
			# 	Z[i,j] += 1e-4
		# print("B")

		gradSw = ti.math.sqrt(SSx*SSx + SSy*SSy)
		# graderr = ti.math.sqrt(SSgradZy*SSgradZy + SSgradZ*SSgradZ)
		
		graviterm = kz * ti.math.sqrt(SSgradZy*SSgradZy + SSgradZx*SSgradZx)
		
		# trying here to reduce instabilities by using hflow
		# waterm = betalpha * hflow * gradSw/(rho_ratio * D) 
		# if(waterm == 0):
		# 	waterm = betalpha * hw[i,j] * gradSw/(rho_ratio * D) 
		waterm = betalpha * hw[i,j] * gradSw/(rho_ratio * D) 

		graderr = graviterm + waterm

		gradSPsi = ti.math.sqrt(SSPsiy*SSPsiy + SSPsix*SSPsix)

		monitorer[i,j] = max(graviterm,0) / ( max(graviterm,0) + max(waterm,0) )
		# monitorer[i,j] = max(waterm,0)

		QwC[i,j] = dx/manning * ti.math.pow(hw[i,j], 5./3) * sumSw/ti.math.sqrt(gradSw)

		edot = K * ti.math.pow( ti.math.max(graderr - tau_c, 0.),1.5);

		depcof = ti.math.pow(dx,2) * (gradSPsi/(sumSPsi * l_transp)) if sumSPsi > 0. else 0.;
		# depcof = ti.math.pow(dx,2) * (gradSw/(sumSw * l_transp)) if sumSPsi > 0. else 0.;

		# deactivate
		# depcof = 0.

		QsC[i,j] = (QsA[i,j] + ti.math.pow(dx,2) * (edot))/(1 + depcof) if(sumSPsi > 0.) else 0.;
	
		if(i >= ny-10 or i == 0):
			QsC[i,j] = QsA[i,j]



		# print(QwC[i,j])
		for k in range(4):
			ir,jr = neighbour(i,j,k)
			if(ir == -1):
				continue
			ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])
			if(sumSPsi > 0):
				ti.atomic_add(QsB[ir,jr], SPsis[k]/sumSPsi * QsC[i,j])
				# ti.atomic_add(QsB[ir,jr], Sws[k]/sumSw * QsC[i,j])


@ti.kernel
def compute_hw():
	for i,j in Z:
		QwA[i,j] = QwB[i,j]
		if(i == ny-1):
			continue
		tdt = dt if(CFsL == False or maxhwsw[None] <= 0) else  CFsL_hydro / maxhwsw[None]
		hw[i,j] = ti.math.max(0.,hw[i,j] + (QwA[i,j] - QwC[i,j]) * tdt/(dx*dy) ) 



@ti.kernel
def compute_hwhs():
	for i,j in Z:
		QwA[i,j] = QwB[i,j]
		
		if(i == ny-1):
			continue

		tdt = dt if(CFsL == False or maxhwsw[None] <= 0) else CFsL_hydro * maxhwsw[None]
		hw[i,j] = ti.math.max(0.,hw[i,j] + (QwA[i,j] - QwC[i,j]) * tdt/(dx*dy) )
		
		if(i >= ny-10):
			QsA[i,j] = QsB[i,j]
			continue

		tdt_morpho = dt_morpho if(CFsL == False or maxhwsw[None] <= 0) else  CFsL_morpho / maxhwsw[None]
		# print(tdt_morpho)
		incr = (QsA[i,j] - QsC[i,j]) * tdt_morpho/(dx*dy)
		Z[i,j] = Z[i,j] + incr
		hw[i,j] = ti.math.max(hw[i,j]+incr,0.)

		if(ti.math.isnan(Z[i,j])):
			print('NAN')
		QsA[i,j] = QsB[i,j]
	# print(CFsL_morpho / maxhwsw[None])




# GUI setup
gui = ti.GUI("Field Visualization", res=(nx, ny))
init_field()
# run()
# quit()


print("YOLO")

# fig = plt.subplots(1,2, figsize = (12,12))
fig = plt.figure(figsize = (18,12))
gs = plt.GridSpec(ncols=3, nrows=4, figure=fig)
ax = fig.add_subplot(gs[:,0])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
im = ax.imshow(hw.to_numpy(), cmap = "Blues", vmin = 0., vmax = 1.8, extent = env.grid.extent())
plt.colorbar(im,label = 'Flow Depth (m)')
# im = ax.imshow(hw.to_numpy(), cmap = "RdBu_r", vmin = 0., vmax = 2.)
ax2 = fig.add_subplot(gs[0:3,1:])
ax2.set_aspect(1)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Z (m)')
pl_cs_z = ax2.plot(env.grid.X, Z.to_numpy()[50,:], lw = 2, color = 'k')
pl_cs_zw = ax2.plot(env.grid.X, (Z.to_numpy() + hw.to_numpy())[50,:], lw = 2, color = 'b')

ax3 = fig.add_subplot(gs[3,1])
ax3.set_xlabel('')
ax3.set_ylabel('$Q_w$ input ($m^3$)')
pltclim = ax3.axhline(fac.to_numpy() * Qwtot, color = 'blue', lw = 4)
ax3.set_ylim(0.5 * Qwtot, 3*Qwtot)
# ax3.set_aspect(1)
# pl_cs_psi = ax3.plot(env.grid.X, (Psi_surf.to_numpy())[50,:], lw = 2, color = 'orange')
pl_cs_psi2 = ax2.plot(env.grid.X, (Psi_surf.to_numpy())[50,:], lw = 2, color = 'orange' , alpha = 0.)

ax2.set_xlim(70,130)
# ax3.set_xlim(70,130)

fig.show()

Zori = Z.to_numpy()

it = 0
# morpho = False
while True:
	it+=1

	# if(it%10 == 0 and it > 20):
		# stochfac()
	# if(it > 20):
	set_fac(np.float32(A * np.sin(omega * (it)) + B))

	
	for iii in range(10000):
		stepinit()
		compute_Qw()
		compute_hw()

		if(iii % NHYDRO == 0 and morpho):
			stepinit()
			compute_QwQs()
			compute_hwhs()

	thw = hw.to_numpy()
	# thw = monitorer.to_numpy()
	# thw = Zori - Z.to_numpy()
	print(np.nanmedian(hw.to_numpy()))
	im.set_data(thw)
	# im.set_clim(np.nanmin(thw), np.nanmax(thw))

	tZ = Z.to_numpy()[50,:]
	tZw = (Z.to_numpy() + hw.to_numpy())[50,:]
	tpsi = Psi_surf.to_numpy()[50,:]
	pl_cs_z[0].set_ydata(tZ)
	pl_cs_zw[0].set_ydata(tZw)
	# pl_cs_psi[0].set_ydata(tpsi)
	pl_cs_psi2[0].set_ydata(tpsi)

	pltclim.set_ydata(fac.to_numpy() * Qwtot)

	ax2.set_ylim(np.nanmin(tZ)-0.3 * (np.nanmax(tZ) - np.nanmin(tZ)), np.nanmax(tZ)+0.3 * (np.nanmax(tZ) - np.nanmin(tZ)))
	# ax3.set_ylim(np.nanmin(tpsi)-0.3 * (np.nanmax(tpsi) - np.nanmin(tpsi)), np.nanmax(tpsi)+0.3 * (np.nanmax(tpsi) - np.nanmin(tpsi)))


	ax3.set_title(f'$t^*=${it}')

	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.001)

	name = str(it)
	while len(name) < 4:
		name = '0' + name
	plt.savefig(f'outtest/{name}.png')


	# np.save("lastZ.npy", Z.to_numpy())
	# plt.pause(0.001)

