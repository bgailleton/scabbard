import numpy as np
import matplotlib.pyplot as plt
import numba as nb
# Assuming 'scabbard' is a custom module you have for loading data
# If not, you can replace it with an appropriate data loading mechanism
import scabbard as scb
import taichi as ti



# Function to compute ray direction from camera through pixel
@ti.func
def compute_ray_direction(
	px:ti.template(), 
	py:ti.template(), 
	camera_position:ti.template(), 
	camera_direction:ti.template(), 
	camera_up:ti.template(), 
	camera_right:ti.template(), 
	focal_length:ti.f32
	):
	# Compute the point on the image plane in world coordinates
	# image_plane_point = ti.math.vec3(0.)
	image_plane_point = (camera_position +
						 camera_direction * focal_length +
						 camera_right * px +
						 camera_up * py)
	
	# Compute the ray direction from camera position to image plane point
	# ray_direction = ti.math.vec3(image_plane_point - camera_position[0], image_plane_point - camera_position[1], image_plane_point - camera_position[2])
	ray_direction = image_plane_point - camera_position
	norm = ti.math.length(ray_direction)
	ray_direction /= norm
	return ray_direction

# Function to interpolate ZZ at (x, y) using bilinear interpolation
@ti.func
def interpolate_ZZ(
	x:ti.template(), 
	y:ti.template(), 
	XX:ti.template(), 
	YY:ti.template(), 
	ZZ:ti.template()
	):
	
	nx,ny = XX.shape[1],XX.shape[0]
	# Ensure x and y are within the bounds
	z = -9999.
	if x < XX[0, 0] or x > XX[0, nx-1] or y < YY[0, 0] or y > YY[ny-1, 0]:
		z =  -9999.0
	else:
		# Find the spacing between grid points
		dx = XX[0,1] - XX[0,0]
		dy = YY[1,0] - YY[0,0]

		# Compute indices in grid
		i = (x - XX[0,0]) / dx
		j = (y - YY[0,0]) / dy

		i0 = int(ti.math.floor(i))
		j0 = int(ti.math.floor(j))

		# Ensure indices are within bounds
		i0 = max(0, min(i0, nx - 2))
		j0 = max(0, min(j0, ny - 2))

		i1 = i0 + 1
		j1 = j0 + 1

		# Compute fractional parts
		s = i - i0
		t = j - j0

		# Get Z values at the corners
		z00 = ZZ[j0, i0]
		z10 = ZZ[j0, i1]
		z01 = ZZ[j1, i0]
		z11 = ZZ[j1, i1]

		# Perform bilinear interpolation
		z0 = z00 * (1 - s) + z10 * s
		z1 = z01 * (1 - s) + z11 * s
		z = z0 * (1 - t) + z1 * t

	return z

# Function to compute normal at (x, y) using gradients
@ti.func
def compute_normal(
	x:ti.template(),
	y:ti.template(), 
	XX:ti.template(), 
	YY:ti.template(), 
	ZZ:ti.template()
	):
	# Compute gradients using central differences
	h = 1e-5  # Small step for numerical derivative

	# Compute partial derivatives
	dz_dx = (interpolate_ZZ(x + h, y, XX, YY, ZZ) - interpolate_ZZ(x - h, y, XX, YY, ZZ)) / (2 * h)
	dz_dy = (interpolate_ZZ(x, y + h, XX, YY, ZZ) - interpolate_ZZ(x, y - h, XX, YY, ZZ)) / (2 * h)

	# The normal vector is [-dz/dx, -dz/dy, 1]
	normal = ti.math.vec3(-dz_dx, -dz_dy, 1.0)
	norm = ti.math.length(normal)
	normal /= norm

	return normal

# Function to perform ray-surface intersection using ray marching
@ti.func
def ray_surface_intersection(
	ray_origin:ti.math.vec3, 
	ray_direction:ti.math.vec3,
	XX:ti.template(),
	YY:ti.template(),
	ZZ:ti.template(),

	):
	# Initialize t (ray parameter)
	t = 0.0   # Start from t = 0.0
	max_t = 10.0  # Maximum distance to march
	dt = 0.01      # Step size (smaller for higher accuracy)

	prev_dz = -9999.0
	intersection_point = ti.math.vec3(-9999.)
	normal = ti.math.vec3(-9999.)
	dz = 10.
	nx,ny = XX.shape[1],XX.shape[0]

	while t < max_t:
		# Compute current point along the ray
		point = ray_origin + t * ray_direction
		x, y, z = point

		# Get surface Z at (x, y) if within domain
		if x >= XX[0,0] and x <= XX[0,nx-1] and y >= YY[0,0] and y <= YY[ny-1,0]:
			surface_z = interpolate_ZZ(x, y,XX,YY,ZZ)
			if surface_z != -9999.0 and z != -9999.0:
				# Compute difference between ray's z and surface z
				dz = z - surface_z

				if prev_dz != -9999.0:
					if dz * prev_dz < 0:
						# The sign of dz changed, indicating a crossing
						# Perform linear interpolation to find intersection t
						t_intersect = t - dt * dz / (dz - prev_dz)
						# Compute intersection point
						intersection_point = ray_origin + t_intersect * ray_direction
						x_int, y_int, z_int = intersection_point
						# Compute normal at intersection point
						normal = compute_normal((x_int), (y_int),XX,YY,ZZ)
						break
					elif dz == 0.0:
						# Ray is exactly on the surface
						intersection_point = point
						normal = compute_normal(x, y,XX,YY,ZZ)
						break

				prev_dz = dz
			else:
				# Surface Z is invalid
				pass
		else:
			# If the ray is outside the domain and moving away, terminate early
			if prev_dz != -9999.0 and dz > 0:
				break  # Ray is above the surface and moving upwards

		t += dt

	# No intersection found
	return intersection_point, normal



@ti.kernel
def compute_image( 
	XX:ti.template(),
	YY:ti.template(),
	ZZ:ti.template(),
	image:ti.template(),
	PX:ti.template(),
	PY:ti.template(),
	camera_position:ti.template(),
	camera_direction:ti.template(),
	camera_up:ti.template(),
	camera_right:ti.template(),
	focal_length:ti.f32, 
	image_height:ti.i32, 
	image_width:ti.i32
	):
	# Loop over each pixel in the image
	for i,j in ti.ndrange((0,image_height),(0,image_width)):
		# Compute pixel coordinates in image plane
		px = PX[i, j]
		py = PY[i, j]

		# Compute ray direction
		ray_direction = compute_ray_direction(px, py, camera_position,camera_direction,camera_up,camera_right,focal_length)

		# Ray origin is the camera position
		ray_origin = ti.math.vec3(camera_position[0],camera_position[1],camera_position[2])

		# Perform ray-surface intersection
		intersection_point, normal = ray_surface_intersection(ray_origin, ray_direction, XX, YY, ZZ)

		if intersection_point[0] != -9999.0:
			# Simple shading using Lambertian reflection
			# Define light direction (e.g., from above)
			light_direction = ti.math.vec3(1.0, 1.0, 1.0)  # Light coming from (1,1,1)
			light_direction /= ti.math.length(light_direction)

			# Compute intensity
			intensity = ti.math.dot(normal, light_direction)
			intensity = max(0.0, min(intensity, 1.0))

			# Assign color based on intensity
			color = intensity * ti.math.vec3(1.0, 1.0, 1.0)  # White color scaled by intensity

			image[i, j, 0] = color[0]
			image[i, j, 1] = color[1]
			image[i, j, 2] = color[2]
		else:
			# Background color (e.g., sky blue)
			image[i, j, 0] = 0. # Light blue background
			image[i, j, 1] = 0. # Light blue background
			image[i, j, 2] = 0. # Light blue background


# ==========================================
# Step 6: Main Rendering Loop
# ==========================================


@ti.kernel
def rotate(
	tp:ti.template(),
	image:ti.template(),
	PX:ti.template()
	):

	for i,j in PX:
		tp[j,i,0] = image[i,j,0] 
		tp[j,i,1] = image[i,j,1] 
		tp[j,i,2] = image[i,j,2] 



def gray_RT(
	grid:scb.raster.RegularRasterGrid,
	exaggeration_factor = 0.5,  # Adjust this as needed
	camera_distance = 2.0,      # Distance from the center (adjust for zoom)
	camera_azimuth_deg = 270,   # Azimuth angle in degrees
	camera_elevation_deg = 45,  # Elevation angle from horizontal in degrees
	focal_length = 1.0,         # Adjust as needed
	image_width = 1200,         # Image width in pixels
	image_height = 900,         # Image height in pixels
	fov_deg = 60,               # Field of view in degrees (reduced to minimize distortion)


	):

	ti.init(ti.gpu)

	# ==========================================
	# Step 1: Data Preparation
	# ==========================================

	Z = grid.Z[1:-1,1:-1]

	# Normalize Z to range from 0 to 1, then apply exaggeration factor
	Z_min, Z_max = Z.min(), Z.max()
	Z_normalized = (Z - Z_min) / (Z_max - Z_min) * exaggeration_factor

	# Generate XX and YY coordinates ranging from -1 to 1
	N, M = Z.shape
	ny, nx = Z.shape
	x = np.linspace(-1, 1, M)
	y = np.linspace(-1, 1, N)
	_XX, _YY = np.meshgrid(x, y)

	# Set ZZ to the normalized Z values
	_ZZ = Z_normalized

	# ==========================================
	# Step 2: Camera Setup
	# ==========================================

	# Compute the center point (0, 0, median ZZ)
	center_Z = np.median(_ZZ)
	center_point = np.array([0.0, 0.0, center_Z])



	# Convert angles to radians
	theta = np.deg2rad(camera_azimuth_deg)      # Azimuth angle in radians
	phi = np.deg2rad(camera_elevation_deg)      # Elevation angle in radians

	# Compute camera position using spherical coordinates
	camera_x = center_point[0] + camera_distance * np.cos(phi) * np.cos(theta)
	camera_y = center_point[1] + camera_distance * np.cos(phi) * np.sin(theta)
	camera_z = center_point[2] + camera_distance * np.sin(phi)
	_camera_position = np.array([camera_x, camera_y, camera_z])

	# ==========================================
	# Step 3: Camera Coordinate System
	# ==========================================

	# Camera direction vector (from camera position to center point)
	_camera_direction = center_point - _camera_position
	_camera_direction /= np.linalg.norm(_camera_direction)  # Normalize

	# World up vector (Z-axis)
	world_up = np.array([0, 0, 1])

	# Camera right vector
	_camera_right = np.cross(_camera_direction, world_up)
	_camera_right /= np.linalg.norm(_camera_right)  # Normalize

	# Camera up vector
	_camera_up = np.cross(_camera_right, _camera_direction)
	_camera_up /= np.linalg.norm(_camera_up)

	# ==========================================
	# Step 4: Image Plane Setup
	# ==========================================

	

	# Set up the image plane dimensions based on field of view and aspect ratio
	aspect_ratio = image_width / image_height
	fov_rad = np.deg2rad(fov_deg)  # Convert to radians

	# Compute image plane dimensions
	image_plane_height = 2 * focal_length * np.tan(fov_rad / 2)
	image_plane_width = image_plane_height * aspect_ratio

	# Generate pixel coordinates in the image plane
	px = np.linspace(-0.5 * image_plane_width, 0.5 * image_plane_width, image_width)
	py = np.linspace(-0.5 * image_plane_height, 0.5 * image_plane_height, image_height)
	_PX, _PY = np.meshgrid(px, py)



	PX = ti.field(ti.f32, shape = _PX.shape)
	PX.from_numpy(_PX.astype(np.float32))
	PY = ti.field(ti.f32, shape = _PY.shape)
	PY.from_numpy(_PY.astype(np.float32))
	
	image = ti.field(ti.f32, shape = (image_height, image_width, 3) )
	image.fill(0.)
	
	out = ti.field(ti.f32, shape = (image_height, image_width, 3) )

	XX = ti.field(ti.f32, shape = _XX.shape)
	XX.from_numpy(_XX.astype(np.float32))
	YY = ti.field(ti.f32, shape = _YY.shape)
	YY.from_numpy(_YY.astype(np.float32))
	ZZ = ti.field(ti.f32, shape = _ZZ.shape)
	ZZ.from_numpy(_ZZ.astype(np.float32))

	camera_position = ti.Vector(_camera_position.astype(np.float32))
	camera_direction = ti.Vector(_camera_direction.astype(np.float32))
	camera_right = ti.Vector(_camera_right.astype(np.float32))
	camera_up = ti.Vector(_camera_up.astype(np.float32))

	compute_image(XX,YY,ZZ,image,PX,PY,camera_position,camera_direction,camera_up,camera_right,focal_length,image_height,image_width)
	
	# rotate(out, image, PX)


	return image.to_numpy()[::-1]











