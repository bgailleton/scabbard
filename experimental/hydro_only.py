import taichi as ti
import scabbard as scb
import numpy as np
import matplotlib.pyplot as plt

def f2img(tfield):
	out = tfield.to_numpy()
	return np.rot90((out - np.nanmin(out))/(np.nanmax(out) - np.nanmin(out)), -1)

res = 1

fnamedem = f"/home/bgailleton/Desktop/code/FastFlood2.0/FastFlood2_Boris/graphflood/paper_scripts/data/green_river_{res}.tif"
# fnamedem = f"/home/bgailleton/Desktop/code/Ron/DEM_Ardeche_US_clipped_without_Pont.tif"

dt = 1e-4
P = 2e-5
manning = 0.033
env = scb.env_from_DEM(fnamedem)
nx,ny = env.grid.nx, env.grid.ny
dx,dy = env.grid.dx, env.grid.dy

hmaxplot = 1.5

ti.init(arch=ti.gpu)  # Initialize Taichi to use the CPU


Z = ti.field(dtype=ti.f32, shape=(ny,nx))
Z.from_numpy(env.grid.Z2D.astype(np.float32))
hw = ti.field(dtype=ti.f32, shape=(ny,nx))

QwA = ti.field(dtype=ti.f32, shape=(ny,nx))
QwB = ti.field(dtype=ti.f32, shape=(ny,nx))
QwC = ti.field(dtype=ti.f32, shape=(ny,nx))


##

input_points = False
totalQw = 500
indices = np.arange(nx * ny).reshape(env.grid.rshp)
input_nodes = np.array([[11,59],[51,193],[48,180],[45,167],[41,156],[39,145],[35,134],[31,123],
                          [27,111],[24,98],[19,87],[18,76],[15,68]]).astype(np.int32)

row_input = input_nodes[:,0]
col_input = input_nodes[:,1]
input_nodes_r = ti.field(dtype=ti.i32, shape=(row_input.shape[0]))
input_nodes_r.from_numpy(row_input)
input_nodes_c = ti.field(dtype=ti.i32, shape=(col_input.shape[0]))
input_nodes_c.from_numpy(col_input)

tQw = np.full_like(row_input,totalQw/row_input.shape[0],dtype = np.float32)
input_Qw = ti.field(dtype=ti.f32, shape=(row_input.shape[0]))
input_Qw.from_numpy(tQw)





@ti.kernel
def init_field():
	for i, j in Z:
		hw[i,j] = 0



if(input_points):
	@ti.kernel
	def stepinit():
		for i,j in Z:
			QwB[i,j] = 0
		for i in input_Qw:
			QwA[input_nodes_r[i], input_nodes_c[i]] = input_Qw[i]
			QwB[input_nodes_r[i], input_nodes_c[i]] = input_Qw[i]
else:
	@ti.kernel
	def stepinit():
		for i,j in Z:
			QwA[i,j] += P * dx * dy
			QwB[i,j] = P * dx * dy

@ti.func
def Zw(i,j) -> ti.f32:
	return Z[i,j] + hw[i,j]

@ti.func
def Sw(i,j, ir,jr)->ti.f32:
	return (Zw(i,j) - Zw(ir,jr))/dx

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

		if(i == ny-1 or i ==0 or j==0 or j == nx-1 or Z[i,j] < 0.):
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
		if(gradSw == 0):
			continue
		QwC[i,j] = dx/manning * ti.math.pow(hw[i,j], 5./3) *sumSw/ti.math.sqrt(gradSw)
		# print(QwC[i,j])
		for k in range(4):
			ir,jr = neighbour(i,j,k)
			if(ir == -1):
				continue
			ti.atomic_add(QwB[ir,jr], Sws[k]/sumSw * QwA[i,j])



@ti.kernel
def compute_hw():
	for i,j in Z:
		QwA[i,j] = QwB[i,j]
		if(i == ny-1 or i ==0 or j==0 or j == nx-1 or Z[i,j] < 0.):
			continue
		
		hw[i,j] = ti.math.max(0.,hw[i,j] + (QwA[i,j] - QwC[i,j]) * dt/(dx*dy) ) 



# GUI setup

init_field()
stepinit()
compute_Qw()
compute_hw()


print("YOLO")

fig,ax = plt.subplots(figsize = (12,12))
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
im = ax.imshow(hw.to_numpy(), cmap = "Blues", vmin = 0., vmax = hmaxplot, extent = env.grid.extent())
plt.colorbar(im,label = 'Flow Depth (m)')
fig.show()
it = 0
while True:
	it += 1
	print(it)
	for iii in range(100):
		stepinit()
		compute_Qw()
		compute_hw()


	thw = hw.to_numpy()
	im.set_data(thw)
	# im.set_clim(np.nanmin(thw), np.nanmax(thw))

	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.001)

