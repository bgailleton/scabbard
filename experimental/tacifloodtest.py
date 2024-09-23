import taichi as ti
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


# Define a 2D field (array) with float32 elements
ny,nx = 512, 200
x,y = np.linspace(0,12, nx), np.linspace(0, 12,  ny)
pnoise = noise2array(x,y).astype(np.float32)
pnoise -= pnoise.min()
pnoise /= pnoise.max()

dt = 1e-5
dx,dy = 2.,2.
S0 = 2e-2
Qwtot = 50
Hprec = Qwtot/(dx*dy)*dt
manning = 0.033
NPREC = 1
clock = 0.
D = 4e-3
l_transp = 10
dilmorph = 1e3
gravity = 9.8
rho_water = 1000
klat = 50

k_erosion, tau_c = calculate_MPM_from_D(D,l_transp)



ti.init(arch=ti.cpu, default_cpu_block_dim = NPREC)  # Initialize Taichi to use the CPU


Z = ti.field(dtype=ti.f32, shape=(ny,nx))
Z.from_numpy(pnoise)
hw = ti.field(dtype=ti.f32, shape=(ny,nx))
globtime = ti.field(dtype=ti.f32, shape=(ny,nx))
position = ti.types.struct(r = ti.i32,c = ti.i32)
precipitons = ti.Struct.field({
	"pos": position,
	"time": float,
	"qs":float,
}, shape=(NPREC,))




@ti.func
def spawn() -> position:
	j = ti.math.round(ti.random() * 50) + 75
	return position(0,j)

# Initialize the field with some values (example)
@ti.kernel
def init_field():
	for i, j in Z:
		hw[i,j] = 0
		globtime[i,j] = 0
		Z[i,j] += (nx - 1 - i) * dx * S0
	for i in range(NPREC):
		precipitons.pos[i] = spawn()
		precipitons.time[i] = 0.
		precipitons.qs[i] = 0.
	# Z[0,50] += 50

# init_field()

# plt.imshow(Z.to_numpy())
# plt.show()
# quit()


@ti.func
def Zw(pos:position) -> ti.f32:
	return Z[pos.r,pos.c] + hw[pos.r,pos.c]

@ti.func
def Sw(pos:position, posr:position)->ti.f32:
	return (Zw(pos) - Zw(posr))/dx

@ti.func
def Sh(pos:position, posr:position)->ti.f32:
	return (Z[pos.r,pos.c] - Z[posr.r,posr.c])/dx

@ti.func
def qw(pos:position, tSw:ti.f32)->ti.f32:
	return 1./manning * ti.math.pow(hw[pos.r,pos.c],(5./3.)) * ti.math.pow(tSw,0.5)

@ti.func
def neighbour(pos:position,k:int) -> position:
	ret = position(-1,-1)
	valid = True
	if(pos.r == 0):
		if(pos.c == 0 and k <= 1):
			valid = False
		elif(pos.c == nx-1 and (k == 0 or k == 2)):
			valid = False
		elif(k==0):
			valid = False
	elif(pos.c == 0 and k == 1):
		valid = False
	elif(pos.c == nx-1 and k == 2):
		valid = False
	elif(pos.r == ny-1):
		if(pos.c == 0 and (k == 1 or k == 3)):
			valid = False
		elif(pos.c == nx-1 and (k == 3 or k == 2)):
			valid = False
		elif(k==3):
			valid = False
	if(valid):
		if(k == 0):
			ret = position(pos.r-1, pos.c)
		if(k == 1):
			ret = position(pos.r, pos.c-1)
		if(k == 2):
			ret = position(pos.r, pos.c+1)
		if(k == 3):
			ret = position(pos.r+1, pos.c)
	return ret

@ti.func
def get_SS(pos:position):
	SS = 0.
	kss = -1
	for k in range(4):
		posr = neighbour(pos,k)
		if(posr.c>=0):
			tSS = Sw(pos,posr)
			# print(tSS)
			if(tSS>SS):
				SS = tSS
				kss = k
	return SS, kss


@ti.func
def get_stochasrec(pos:position)->position:
	
	SS = 0.
	fposr = position(-1,-1)

	for k in range(4):
		posr = neighbour(pos,k)
		if(posr.c>=0):
			tSS = Sw(pos,posr) * ti.random()
			if(tSS > SS):
				# print(tSS)
				SS = tSS
				fposr = posr
	# print('1',fposr, pos)
	return fposr

@ti.func
def get_latneigh(pos:position, k:ti.i32):
	pos1 = position(-1,-1)
	pos2 = position(-1,-1)
	
	if(k == 0):
		pos1 = neighbour(pos,1)
		pos2 = neighbour(pos,2)
	elif(k == 1):
		pos1 = neighbour(pos,0)
		pos2 = neighbour(pos,3)
	elif(k == 2):
		pos1 = neighbour(pos,3)
		pos2 = neighbour(pos,0)
	elif(k == 3):
		pos1 = neighbour(pos,2)
		pos2 = neighbour(pos,1)
	
	return pos1, pos2

@ti.kernel
def run_cpu(new_time:float, morpho:bool):
	for i in precipitons:
		precipitons.pos[i] = spawn()
		precipitons.time[i] = new_time
		# UU=0
		while(True):

			# print("stuff:", precipitons.pos[i].r)

			if(precipitons.pos[i].r == ny-1):
				break
			
			tdt = precipitons.time[i] - globtime[precipitons.pos[i].r,precipitons.pos[i].c]
			# tdt = max(tdt,dt*NPREC)
			tSw,k = get_SS(precipitons.pos[i])
			# print("A")
			while tSw == 0.:
				hw[precipitons.pos[i].r,precipitons.pos[i].c] += Hprec
				tSw,k = get_SS(precipitons.pos[i])
			# print("B")
			if(tdt>0):
				hw[precipitons.pos[i].r, precipitons.pos[i].c] -= qw(precipitons.pos[i],tSw)/dx * tdt*NPREC - Hprec
				hw[precipitons.pos[i].r, precipitons.pos[i].c] = max(0.,hw[precipitons.pos[i].r,precipitons.pos[i].c])
				globtime[precipitons.pos[i].r,precipitons.pos[i].c] = precipitons.time[i]

				if(morpho):
					tau = hw[precipitons.pos[i].r, precipitons.pos[i].c] * tSw * gravity * rho_water
					edot = max(tau - tau_c,0.)
					edot = ti.math.pow(edot,1.5) * k_erosion
					

					lat1,lat2 = get_latneigh(precipitons.pos[i],k)
					elat = 0.
					if(lat1.r >=0 ):
						tS = Sh(precipitons.pos[i],lat1)
						if(tS<0):
							telat= min(klat*abs(tS)*edot, abs(tS)*dx/(tdt*dilmorph))
							Z[lat1.r,lat1.c] -= telat * tdt * dilmorph
							elat += telat
					if(lat2.r >=0 ):
						tS = Sh(precipitons.pos[i],lat2)
						if(tS<0):
							telat= min(klat*abs(tS)*edot, abs(tS)*dx/(tdt*dilmorph))
							Z[lat2.r,lat2.c] -= telat * tdt * dilmorph
							elat += telat

					qsout = ((edot) * l_transp + (precipitons.qs[i]/dx - (edot * l_transp)) * ti.math.exp(-dx/l_transp))
					Z[precipitons.pos[i].r, precipitons.pos[i].c] += (precipitons.qs[i] - qsout)/dx * tdt * dilmorph

					precipitons.qs[i] = qsout + elat * dx
					# precipitons.qs[i] = qsout 

			else:
				hw[precipitons.pos[i].r, precipitons.pos[i].c] += Hprec

		

			newpos = get_stochasrec(precipitons.pos[i])
			# precipitons.pos[i] = newpos
			# print("C")
			# print('2.5',newpos.r, precipitons.pos[i].r)
			while newpos.c == -1:
				# print(hw[precipitons.pos[i].r,precipitons.pos[i].c],precipitons.pos[i].r,precipitons.pos[i].c)
				hw[precipitons.pos[i].r,precipitons.pos[i].c] += Hprec
				newpos = get_stochasrec(precipitons.pos[i])
			
			precipitons.pos[i] = newpos
			# print("D")





# GUI setup
gui = ti.GUI("Field Visualization", res=(nx, ny))
init_field()


print("YOLO")
# plt.ioff()
fig,ax = plt.subplots()
im = ax.imshow(hw.to_numpy(), cmap = "Blues", vmin = 0., vmax = 0.5)
fig.show()

it = 0
morpho = False
while True:
	it+=1
	if(it>10):
		morpho = True
	print(it)
	for _ in range(10000):
		clock += dt
		run_cpu(clock, morpho)
	thw = hw.to_numpy()
	print(np.sum(thw))
	im.set_data(thw)
	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.001)
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