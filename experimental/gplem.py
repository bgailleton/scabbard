import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scabbard as scb


def MPM_params(D, rho_water, rho_sediment, gravity, theta_c):
    R = rho_sediment/rho_water - 1
    tau_c = rho_water * gravity * R * D * theta_c
    E = 8/(rho_water**0.5 * (rho_sediment - rho_water) * gravity)
    return E, tau_c

nx,ny = 200,512
dx,dy = 4.,4.
dxy = m.sqrt(dx**2 + dy**2)
nodata = -2147483648
S0 = 5e-2
noise = 1e-2
ndt = 500000
halfwidth_input = 10
totalQw = 50

D8 = False

update_every = 1


# Morpho
rho_water = 1000
rho_sediment = 2650
gravity = 9.81
manning = 0.033
D = 8e-3
theta_c = 0.047
prop_sed = 0.
alphalat = 1e-3
soil_D = 5e-5

N_iterations = 1000000000
dt_hydro = 1e-2
dt_morpho = dt_hydro * 1
start_morpho = 100

Qsout_mult = 1.1
Urate = 1e-3
stochQw = 0.

#Plotting options
row_CS = 50


E_MPM, tau_c = MPM_params(D, rho_water, rho_sediment, gravity, theta_c)
print("E is", E_MPM, "and tau_c is", tau_c)




# surf = scb.generate_u_shaped_sloped_surface(nx, ny, dx, dy, slope=S0, Umag = 1) + np.random.rand(ny,nx)*noise
# env = scb.env_from_array(surf ,nx = nx, ny = ny, dx = dx, dy = dy, E="periodic", N="forcein", W="periodic", S='out')
env = scb.env_from_slope(noise_magnitude=noise, dx = dx, dy = dy, slope=S0, nx = nx,ny = ny)
# env.param.set_gf2Bbval(S0)
Z = np.copy(env.grid.Z2D)
Z[0,:] += 100
Z = Z.ravel().astype(np.float32)

node_i = np.arange(round(nx/2) - halfwidth_input,round(nx/2) + halfwidth_input , dtype = np.int32)
nodeQw = scb.gaussian_spread_on_1D(X = node_i, M = totalQw, x_c = node_i[node_i.shape[0] // 2], sigma = 2).astype(np.float32)
nodeQs = nodeQw * prop_sed
node_i += nx

env.init_connector()
env.init_GF2()
env.graphflood.init()
env.graphflood.set_Qw_input_points(node_i, nodeQw)
env.graphflood.set_dt(1e-3)
env.graphflood.run()


QwA = np.zeros_like(Z, dtype = np.float32)
QwB = np.zeros_like(Z, dtype = np.float32)
hw = np.zeros_like(Z, dtype = np.float32)
QwA += env.d('Qw').ravel()

# print(QwA.shape,QwB.shape)

QsA = np.zeros_like(Z, dtype = np.float32)
QsB = np.zeros_like(Z, dtype = np.float32)


BC = np.ones_like(env.grid.Z2D, dtype = np.uint8)
BC[-1,:] = 3
BC[:,[0,-1]] = 0
BC[0,:] = 0
BC = BC.ravel()



# Load the CUDA source code from an external file
newline = '\n'
with open('kernels.cu', 'r') as file:
    kernel_code = file.read().replace("MACROTOREPLACE_D4D8" , "8" if D8 else "4")


# Your PyCUDA program continues as before...
# Compile the kernel code
mod = SourceModule(kernel_code)
add_Qwin_local = mod.get_function("add_Qwin_local")
compute_Qwin = mod.get_function("compute_Qwin")
swapQwin = mod.get_function("swapQwin")
compute_Qwout = mod.get_function("compute_Qwout")
increment_hw = mod.get_function("increment_hw")
add_Qsin_local = mod.get_function('add_Qsin_local')
compute_MPM = mod.get_function('compute_MPM')
increment_hs = mod.get_function('increment_hs')
multiply_array = mod.get_function('multiply_array')
equal = mod.get_function('equal')
diffuse = mod.get_function('diffuse')
add_constant = mod.get_function('add_constant')
grid_to = mod.get_function('grid_to')



# Initialize and copy the constant arrays
neighbourers = []

if(D8):
    neighbourers.append(np.array([ -nx-1 , -nx , -nx+1 , -1 , 1 , nx-1 , nx , nx+1 ], dtype=np.int32))
    neighbourers.append(np.array([nodata, nodata , nodata , nodata , 1 , nodata , nx , nx+1 ], dtype=np.int32))
    neighbourers.append(np.array([ nodata , nodata , nodata , -1 , 1 , nx-1 , nx , nx+1 ], dtype=np.int32))
    neighbourers.append(np.array([ nodata , nodata , nodata , -1 , nodata , nx-1 , nx , nodata ], dtype=np.int32))
    neighbourers.append(np.array([ nodata , -nx , -nx+1 , nodata , 1 , nodata , nx , nx+1 ], dtype=np.int32))
    neighbourers.append(np.array([ -nx-1 , -nx , nodata , -1 , nodata , nx-1 , nx , nodata ], dtype=np.int32))
    neighbourers.append(np.array([ nodata , -nx , -nx+1 , nodata , 1 , nodata , nodata , nodata ], dtype=np.int32))
    neighbourers.append(np.array([ -nx-1 , -nx , -nx+1 , -1 , 1 , nodata , nodata , nodata ], dtype=np.int32))
    neighbourers.append(np.array([ -nx-1 , -nx , nodata , -1 , nodata , nodata , nodata , nodata ], dtype=np.int32))
else:
    neighbourers.append(np.array([  -nx , -1 , 1 , nx  ], dtype=np.int32))
    neighbourers.append(np.array([ nodata  , nodata , 1  , nx ], dtype=np.int32))
    neighbourers.append(np.array([   nodata  , -1 , 1 ,  nx  ], dtype=np.int32))
    neighbourers.append(np.array([  nodata ,  -1 , nodata ,  nx  ], dtype=np.int32))
    neighbourers.append(np.array([ -nx , nodata , 1 , nx ], dtype=np.int32))
    neighbourers.append(np.array([  -nx ,  -1 , nodata , nx  ], dtype=np.int32))
    neighbourers.append(np.array([   -nx ,  nodata , 1 ,  nodata  ], dtype=np.int32))
    neighbourers.append(np.array([  -nx ,  -1 , 1 ,  nodata  ], dtype=np.int32))
    neighbourers.append(np.array([  -nx ,  -1 , nodata , nodata  ], dtype=np.int32))

neighbourers = np.array(neighbourers, dtype=np.int32)
neighbourers_gpu = mod.get_global("neighbourers")[0]
drv.memcpy_htod(neighbourers_gpu, neighbourers)


dXs = np.array([dxy, dy, dxy, dx, dx, dxy, dy, dxy], dtype=np.float32) if D8 else np.array([ dy, dx, dx, dy], dtype=np.float32) 
dXs_gpu = mod.get_global("dXs")[0]
drv.memcpy_htod(dXs_gpu, dXs)

dYs = np.array([dxy, dx, dxy, dy, dy, dxy, dx, dxy], dtype=np.float32) if D8 else np.array([ dx, dy, dy, dx], dtype=np.float32) 
dYs_gpu = mod.get_global("dYs")[0]
drv.memcpy_htod(dYs_gpu, dYs)

manning = np.float32(manning)
manning_gpu = mod.get_global("manning")[0]
drv.memcpy_htod(manning_gpu, manning)

rho_water = np.float32(rho_water)
rho_water_gpu = mod.get_global("rho_water")[0]
drv.memcpy_htod(rho_water_gpu, rho_water)

gravity = np.float32(gravity)
gravity_gpu = mod.get_global("gravity")[0]
drv.memcpy_htod(gravity_gpu, gravity)

tau_c = np.float32(tau_c)
tau_c_gpu = mod.get_global("tau_c")[0]
drv.memcpy_htod(tau_c_gpu, tau_c)

E_MPM = np.float32(E_MPM)
E_MPM_gpu = mod.get_global("E_MPM")[0]
drv.memcpy_htod(E_MPM_gpu, E_MPM)

alphalat = np.float32(alphalat)
alphalat_gpu = mod.get_global("alphalat")[0]
drv.memcpy_htod(alphalat_gpu, alphalat)


cellarea = np.float32(dx*dy)
cellarea_gpu = mod.get_global("cellarea")[0]
drv.memcpy_htod(cellarea_gpu, cellarea)

dt_hydro = np.float32(dt_hydro)
dt_hydro_gpu = mod.get_global("dt_hydro")[0]
drv.memcpy_htod(dt_hydro_gpu, dt_hydro)

dt_morpho = np.float32(dt_morpho)
dt_morpho_gpu = mod.get_global("dt_morpho")[0]
drv.memcpy_htod(dt_morpho_gpu, dt_morpho)

Qsout_mult = np.float32(Qsout_mult)
Qsout_mult_gpu = mod.get_global("Qsout_mult")[0]
drv.memcpy_htod(Qsout_mult_gpu, Qsout_mult)


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

nodeQs_gpu = drv.mem_alloc(nodeQs.nbytes)
drv.memcpy_htod(nodeQs_gpu, nodeQs)

QwA_gpu = drv.mem_alloc(QwA.nbytes)
drv.memcpy_htod(QwA_gpu, QwA)

QwB_gpu = drv.mem_alloc(QwB.nbytes)
drv.memcpy_htod(QwB_gpu, QwB)

QsA_gpu = drv.mem_alloc(QsA.nbytes)
drv.memcpy_htod(QsA_gpu, QsA)

QsB_gpu = drv.mem_alloc(QsB.nbytes)
drv.memcpy_htod(QsB_gpu, QsB)

Z_gpu = drv.mem_alloc(Z.nbytes)
drv.memcpy_htod(Z_gpu, Z)

temp_gpu = drv.mem_alloc(np.zeros_like(Z, dtype = np.float32).nbytes)
drv.memcpy_htod(temp_gpu, np.zeros_like(Z, dtype = np.float32))

hw_gpu = drv.mem_alloc(hw.nbytes)
drv.memcpy_htod(hw_gpu, hw)

BC_gpu = drv.mem_alloc(BC.nbytes)
drv.memcpy_htod(BC_gpu, BC)



# Define block and grid sizes
# block = (32, 32, 1)
# grid = (int(np.ceil(nx / block[0])), int(np.ceil(ny / block[1])))
# print(block,grid)
# quit()

# grid block for the landscape
block_size_x = 32
block_size_y = 32
block = (int(block_size_x), int(block_size_y), 1)
grid_size_x = (nx + block_size_x - 1) // block_size_x
grid_size_y = (ny + block_size_y - 1) // block_size_y
grid = (int(grid_size_x), int(grid_size_y))

# grid block for the input
block_input = 256  # This is an arbitrary value; tune it based on your GPU architecture
grid_input = (node_i.shape[0] + block_input - 1) // block_input
block_input = (block_input,1,1)
grid_input = (grid_input,1)


# plt.ioff()
fig,axes = plt.subplots(1,3)
ax = axes[0]
ax1 = axes[1]
ax2 = axes[2]
imZ = ax.imshow(Z.reshape(ny,nx), cmap = "gist_earth")
im = ax.imshow(hw.reshape(ny,nx), cmap = "Blues", vmin = 0, vmax = .3)
plt.colorbar(im)
imQw = ax2.imshow(QwA.reshape(ny,nx), cmap = "Purples")
plt.colorbar(imQw)
pl = ax1.plot(env.grid.X, env.grid.Z2D[row_CS,:], color = 'k')
plhw = ax1.plot(env.grid.X, env.grid.Z2D[row_CS,:], color = 'b')
plt.tight_layout()
plt.show(block=False)

print('Running')

# Apply kernels sequentially for a given number of iterations
for step in range(N_iterations):
    grid_to(QwB_gpu,np.float32(0.), block = block, grid = grid)
    add_Qwin_local(node_i_gpu, nodeQw_gpu, QwA_gpu, QwB_gpu, np.int32(node_i.shape[0]), block = block_input, grid = grid_input)
    compute_Qwin(hw_gpu, Z_gpu, QwA_gpu, QwB_gpu, BC_gpu, block = block, grid = grid)
    swapQwin(QwA_gpu, QwB_gpu, block = block, grid = grid)
    compute_Qwout(hw_gpu, Z_gpu, QwB_gpu, BC_gpu, block = block, grid = grid)
    increment_hw(hw_gpu, Z_gpu,QwA_gpu, QwB_gpu, BC_gpu, block = block, grid = grid)
    # add_constant(Z_gpu, BC_gpu, np.float32(Urate), np.int32(nx), np.int32(ny), block = block, grid = grid)
    
    if(step > start_morpho and step % 10 == 0):
        grid_to(QsA_gpu,np.float32(0.), block = block, grid = grid)
        grid_to(QsB_gpu,np.float32(0.), block = block, grid = grid)
        add_Qsin_local(node_i_gpu, nodeQs_gpu, QsA_gpu, QsB_gpu, np.int32(node_i.shape[0]), block = block_input, grid = grid_input)
        compute_MPM(hw_gpu, Z_gpu, QsA_gpu, QsB_gpu, BC_gpu, block = block, grid = grid)
        increment_hs(hw_gpu, Z_gpu,QsA_gpu, QsB_gpu, BC_gpu, block = block, grid = grid)
       
    # if(step % 1000 == 0 and False):
    #     drv.memcpy_dtoh(hw, hw_gpu)
    #     drv.memcpy_dtoh(Z, Z_gpu)
    #     env.data.set_surface(hw + Z)
    #     env.data.set_hw(hw)
    #     env.graphflood.run()
    #     drv.memcpy_htod(QwA_gpu, env.d('Qw').ravel().astype(np.float32))
    #     drv.memcpy_htod(Z_gpu, env.d('bed_surface').ravel().astype(np.float32))
    #     drv.memcpy_htod(hw_gpu, env.d('hw').ravel().astype(np.float32))
    #     if(stochQw > 0):
    #         drv.memcpy_htod(nodeQw_gpu, (np.random.rand() * 2*stochQw - stochQw) * nodeQw + nodeQw )







    if(step % update_every == 0):
    # if(step % (1 if step >1000 else 1000) == 0):
        drv.memcpy_dtoh(hw, hw_gpu)
        drv.memcpy_dtoh(QwA, QwA_gpu)
        drv.memcpy_dtoh(Z, Z_gpu)
        # drv.memcpy_dtoh(hw, QwA_gpu)
        print('step:', step, end = "     \r")
        thw = np.copy(hw.reshape(ny,nx))
        imZ.set_data(Z.reshape(ny,nx))
        try:
            thw[thw<0.05] = np.nan
            im.set_data(thw)
        except:
            pass
        imQw.set_data(QwA.reshape(ny,nx))
        pl[0].set_ydata(Z.reshape(ny,nx)[row_CS,:])
        plhw[0].set_ydata(Z.reshape(ny,nx)[row_CS,:] + hw.reshape(ny,nx)[row_CS,:])

        ax1.set_ylim((Z).reshape(ny,nx)[row_CS,:].min(),(Z+hw).reshape(ny,nx)[row_CS,:].max())

        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.01)

    # input()


# Copy the result back to the CPU
print('done')





# # Create a 2D NumPy array
# width, height = 1048, 1048
# data = np.random.rand(height, width).astype(np.float32)

# # Apply the kernels and get the modified array back
# iterations = 5  # Number of times to apply the sequence of kernels
# output = apply_kernels(data, width, height, iterations)

# print(output)


# One way to describe the lateral shear stress according to https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=fd7f632b6f33aba7672e2b3962740fd3f08dc63d
