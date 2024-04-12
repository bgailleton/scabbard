import taichi as ti
import numpy as np



def f2img(tfield):
	out = tfield.to_numpy()
	return (out - np.nanmin(out))/(np.nanmax(out) - np.nanmin(out))

ti.init(arch=ti.vulkan)  # Initialize Taichi to use the CPU

# Define a 2D field (array) with float32 elements
N = 512
Z = ti.field(dtype=ti.f32, shape=(N, N))
A_a = ti.field(dtype=ti.f32, shape=(N, N))
A_b = ti.field(dtype=ti.f32, shape=(N, N))
Z_b = ti.field(dtype=ti.f32, shape=(N, N))

gradient = ti.field(dtype=ti.f32, shape=(N, N))
Kerr = 1e-4
Derr = 5e-1
uplift_rate = 1e-3
dt = 20

dx,dy = 50.,50.
nx,ny = N,N

# Initialize the field with some values (example)
@ti.kernel
def init_field():
	for i, j in Z:
		Z[i, j] = ti.random()
		A_a[i, j] = dx*dy
		A_b[i, j] = 0.

@ti.kernel
def switch_A():
	for i, j in Z:
		A_a[i,j] = A_b[i,j]
		A_b[i,j] = 0.

@ti.kernel
def compute_A():
	for i, j in Z:
		
		if(i == 0 or i == ny-1 or j ==0 or j == nx-1):
			continue

		A_a[i,j] += dx * dy
		A_b[i,j] += dx * dy

		mini = -1
		minj = -1
		miniZ = Z[i,j]
		while(mini == -1):
			miniZ = Z[i,j]
			
			# ti.loop_config(serialize=True)
			for ai in range(-1,2):
				for aj in range(-1,2):
					if(abs(ai) == abs(aj)):
						continue
					ni,nj = i + ai, j + aj
					if(Z[ni,nj] < miniZ):
						miniZ = Z[ni,nj]
						mini = ni
						minj = nj
			if(mini == -1):
				Z[i,j] += 1e-2

		ti.atomic_add(A_b[mini,minj], A_a[i,j])
		gradient[i,j] = (Z[i,j] - miniZ)/dx

@ti.kernel
def SPL():
	for i,j in Z:
		if(i == 0 or i == ny-1 or j ==0 or j == nx-1):
			continue
		Z[i,j] = Z[i,j] - A_a[i,j]**0.45 * gradient[i,j] * Kerr * dt
		# print(gradient[i,j])
@ti.kernel
def uplift():
	for i,j in Z:
		if(i == 0 or i == ny-1 or j ==0 or j == nx-1):
			continue
		Z[i,j] += uplift_rate * dt

@ti.kernel
def diffuse():
	for i,j in Z:
		if(i == 0 or i == ny-1 or j ==0 or j == nx-1):
			continue
		Z_b[i,j] = Z[i,j] + Derr * (Z[i-1,j] + Z[i+1,j] + Z[i,j-1] + Z[i,j+1] - 4 * Z[i,j])/(dx**2) * dt

@ti.kernel
def switch_Z():

	for i,j in Z:
		Z[i,j] = Z_b[i,j]



# # Calculate the gradient and find the steepest receiver direction
# @ti.kernel
# def calculate_gradient_and_receiver():
# 	for i, j in field:
# 		# Assuming periodic boundary conditions for simplicity
# 		left = field[(i - 1) % N, j] - field[i, j]
# 		right = field[(i + 1) % N, j] - field[i, j]
# 		up = field[i, (j + 1) % N] - field[i, j]
# 		down = field[i, (j - 1) % N] - field[i, j]

# 		# Calculate gradient magnitude (simplified)
# 		grad = ti.sqrt(left**2 + right**2 + up**2 + down**2)
# 		gradient[i, j] = grad

# 		# Determine the direction of the steepest descent
# 		# For simplicity, encoding directions as integers (0: left, 1: right, 2: up, 3: down)
# 		directions = [left, right, up, down]
# 		steepest_dir = ti.argmin(directions)
# 		steepest_receiver[i, j] = steepest_dir

# # Perform an atomic add operation on the steepest receiver
# @ti.kernel
# def atomic_add_on_receiver(receiver_value: ti.f32):
# 	for i, j in steepest_receiver:
# 		if steepest_receiver[i, j] == 0:
# 			ti.atomic_add(field[(i - 1) % N, j], receiver_value)
# 		elif steepest_receiver[i, j] == 1:
# 			ti.atomic_add(field[(i + 1) % N, j], receiver_value)
# 		elif steepest_receiver[i, j] == 2:
# 			ti.atomic_add(field[i, (j + 1) % N], receiver_value)
# 		elif steepest_receiver[i, j] == 3:
# 			ti.atomic_add(field[i, (j - 1) % N], receiver_value)

# # Initialize and run the kernels
# init_field()
# calculate_gradient_and_receiver()
# atomic_add_on_receiver(0.1)  # Example value to add
# GUI setup
gui = ti.GUI("Field Visualization", res=(N, N))
init_field()
for gug in range(500):
	compute_A()
	switch_A()

print("YOLO")
# Optionally, print or visualize your results
N = 0
while gui.running:
	# for _ in range(10):  # Perform updates before drawing
	# 	atomic_add_on_receiver(0.1)
	for gug in range(1000):
		compute_A()
		switch_A()
		uplift()
		SPL()
		diffuse()
		switch_Z()
	N += 1000
	uplift_rate += np.random.rand() * 1e-3 - 0.5e-3
	uplift_rate = max(0.5e-3,uplift_rate)

	print("N",N*dt)

	# calculate_gradient_and_receiver()  # Recalculate after updates

	# Display the field or any other property
	# `field.to_numpy()` converts the Taichi field to a NumPy array for visualization
	gui.set_image(f2img(Z))
	# For visualizing the gradient magnitude, you could use: gui.set_image(gradient.to_numpy())
	# For a custom visualization (like direction), you'll need additional processing

	gui.show()  # Update the GUI window

# import matplotlib.pyplot as plt

# plt.imshow(Z.to_numpy())
# # plt.imshow(A_a.to_numpy())
# plt.show()