import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scabbard as scb


nx,ny = 512,512
dx,dy = 4.,4.
dxy = m.sqrt(dx**2 + dy**2)
nodata = -2147483648
S0 = 5e-2
noise = 1e-3
ndt = 500000
halfwidth_input = round(10/8)
halfwidth_input = 10
totalQw = 50
N_iterations = 500000
dt = 1e-4

D = 8e-3


# surf = scb.generate_u_shaped_sloped_surface(nx, ny, dx, dy, slope=S0, Umag = 1) + np.random.rand(ny,nx)*noise
# env = scb.env_from_array(surf ,nx = nx, ny = ny, dx = dx, dy = dy, E="periodic", N="forcein", W="periodic", S='out')
env = scb.env_from_slope(noise_magnitude=noise, dx = dx, dy = dy, slope=S0, nx = nx,ny = ny)
# env.param.set_gf2Bbval(S0)
Z = np.copy(env.grid.Z2D)
Z[0,:] += 100
Z = Z.ravel().astype(np.float32)

node_i = np.arange(round(nx/2) - halfwidth_input,round(nx/2) + halfwidth_input , dtype = np.int32)
nodeQw = scb.gaussian_spread_on_1D(X = node_i, M = totalQw, x_c = node_i[node_i.shape[0] // 2], sigma = 5).astype(np.float32)
node_i += nx

QwA = np.zeros_like(Z, dtype = np.float32)
QwB = np.zeros_like(Z, dtype = np.float32)
hw = np.zeros_like(Z, dtype = np.float32)

BC = np.ones_like(env.grid.Z2D, dtype = np.uint8)
BC[-1,:] = 3
BC[:,[0,-1]] = 0
BC[0,:] = 0
BC = BC.ravel()



# Load the CUDA source code from an external file
with open('kernels.cu', 'r') as file:
    kernel_code = file.read()

# Your PyCUDA program continues as before...
# Compile the kernel code
mod = SourceModule(kernel_code)
add_Qwin_local = mod.get_function("add_Qwin_local")
compute_Qwin = mod.get_function("compute_Qwin")
swapQwin = mod.get_function("swapQwin")
compute_Qwout = mod.get_function("compute_Qwout")
increment_hw = mod.get_function("increment_hw")


# Initialize and copy the constant arrays
neighbourers = []
neighbourers.append(np.array([ -nx-1 , -nx , -nx+1 , -1 , 1 , nx-1 , nx , nx+1 ], dtype=np.int32))
neighbourers.append(np.array([nodata, nodata , nodata , nodata , 1 , nodata , nx , nx+1 ], dtype=np.int32))
neighbourers.append(np.array([ nodata , nodata , nodata , -1 , 1 , nx-1 , nx , nx+1 ], dtype=np.int32))
neighbourers.append(np.array([ nodata , nodata , nodata , -1 , nodata , nx-1 , nx , nodata ], dtype=np.int32))
neighbourers.append(np.array([ nodata , -nx , -nx+1 , nodata , 1 , nodata , nx , nx+1 ], dtype=np.int32))
neighbourers.append(np.array([ -nx-1 , -nx , nodata , -1 , nodata , nx-1 , nx , nodata ], dtype=np.int32))
neighbourers.append(np.array([ nodata , -nx , -nx+1 , nodata , 1 , nodata , nodata , nodata ], dtype=np.int32))
neighbourers.append(np.array([ -nx-1 , -nx , -nx+1 , -1 , 1 , nodata , nodata , nodata ], dtype=np.int32))
neighbourers.append(np.array([ -nx-1 , -nx , nodata , -1 , nodata , nodata , nodata , nodata ], dtype=np.int32))
neighbourers = np.array(neighbourers, dtype=np.int32)
neighbourers_gpu = mod.get_global("neighbourers")[0]
drv.memcpy_htod(neighbourers_gpu, neighbourers)


dXs = np.array([dxy, dy, dxy, dx, dx, dxy, dy, dxy], dtype=np.float32)
dXs_gpu = mod.get_global("dXs")[0]
drv.memcpy_htod(dXs_gpu, dXs)

dYs = np.array([dxy, dx, dxy, dy, dy, dxy, dx, dxy], dtype=np.float32)
dYs_gpu = mod.get_global("dYs")[0]
drv.memcpy_htod(dYs_gpu, dYs)

manning = np.float32(0.033)
manning_gpu = mod.get_global("manning")[0]
drv.memcpy_htod(manning_gpu, manning)

cellarea = np.float32(dx*dy)
cellarea_gpu = mod.get_global("cellarea")[0]
drv.memcpy_htod(cellarea_gpu, cellarea)

dt = np.float32(dt)
dt_gpu = mod.get_global("dt")[0]
drv.memcpy_htod(dt_gpu, dt)

nx = np.int32(nx)
nx_gpu = mod.get_global("nx")[0]
drv.memcpy_htod(nx_gpu, nx)

ny = np.int32(ny)
ny_gpu = mod.get_global("ny")[0]
drv.memcpy_htod(ny_gpu, ny)

nodata = np.int32(nodata)
nodata_gpu = mod.get_global("nodata")[0]
drv.memcpy_htod(nodata_gpu, nodata)


# Allocate memory on the GPU and copy the data to it
node_i_gpu = drv.mem_alloc(node_i.nbytes)
drv.memcpy_htod(node_i_gpu, node_i)


# Allocate memory on the GPU and copy the data to it
node_i_gpu = drv.mem_alloc(node_i.nbytes)
drv.memcpy_htod(node_i_gpu, node_i)

nodeQw_gpu = drv.mem_alloc(nodeQw.nbytes)
drv.memcpy_htod(nodeQw_gpu, nodeQw)

QwA_gpu = drv.mem_alloc(QwA.nbytes)
drv.memcpy_htod(QwA_gpu, QwA)

QwB_gpu = drv.mem_alloc(QwB.nbytes)
drv.memcpy_htod(QwB_gpu, QwB)

Z_gpu = drv.mem_alloc(Z.nbytes)
drv.memcpy_htod(Z_gpu, Z)

hw_gpu = drv.mem_alloc(hw.nbytes)
drv.memcpy_htod(hw_gpu, hw)

BC_gpu = drv.mem_alloc(BC.nbytes)
drv.memcpy_htod(BC_gpu, BC)



# Define block and grid sizes
block = (32, 32, 1)
grid = (int(np.ceil(nx / block[0])), int(np.ceil(ny / block[1])))


print('Running')

# Apply kernels sequentially for a given number of iterations
for _ in range(N_iterations):
    add_Qwin_local(node_i_gpu, nodeQw_gpu, QwA_gpu, QwB_gpu, np.int32(node_i.shape[0]), block = block, grid = grid)
    compute_Qwin(hw_gpu, Z_gpu, QwA_gpu, QwB_gpu, BC_gpu, block = block, grid = grid)
    swapQwin(QwA_gpu, QwB_gpu, block = block, grid = grid)
    compute_Qwout(hw_gpu, Z_gpu, QwB_gpu, BC_gpu, block = block, grid = grid)
    increment_hw(hw_gpu, Z_gpu,QwA_gpu, QwB_gpu, BC_gpu, block = block, grid = grid)

# Copy the result back to the CPU
drv.memcpy_dtoh(hw, hw_gpu)
print('done')


fig,ax = plt.subplots()
cb = ax.imshow(hw.reshape(ny,nx))
plt.colorbar(cb)
plt.show()


# # Create a 2D NumPy array
# width, height = 1048, 1048
# data = np.random.rand(height, width).astype(np.float32)

# # Apply the kernels and get the modified array back
# iterations = 5  # Number of times to apply the sequence of kernels
# output = apply_kernels(data, width, height, iterations)

# print(output)