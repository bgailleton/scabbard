import numpy as np
import matplotlib.pyplot as plt
import numba as nb
# Assuming 'scabbard' is a custom module you have for loading data
# If not, you can replace it with an appropriate data loading mechanism
import scabbard as scb
import taichi as ti
from taichi.ui import Window, Canvas, GUI

###################
#### CONSTANTS ####
###################

## World up vector (Z-axis), constant
WORLD_UP = np.array([0, 0, 1])
## Background color in RGB 0-1
BGCOL_r = 0.
BGCOL_g = 0.
BGCOL_b = 0.

## shadow color in RGB 0-1
SHADOW_BLEND = 0.

## Render modes
'''
RENDER_COL is the mode of render coloration:
0: grayscale linked to normals (classic 3D mesh)
1: Water related
'''
RENDER_COL = 0

## Precision
RAY_STEP = 0.01      # Step size (smaller for higher accuracy)

## thresholds for plotting water
MIN_WATER = 0.01
## Therhold weight for water: minimum water height starts at MINWACOL * water col
MINWACOL = 0.5


@ti.func
def generate_realistic_water_colors():
	# Randomly generate blue and green dominant colors
	r = ti.random(ti.f32) * 50./255               # Low red to keep the color cool
	g = ti.random(ti.f32) * (80./255 )+ 100./255   # Mid-range green for teal/aqua tones
	b = ti.random(ti.f32) * (50./255) + 155./255  # High blue for a watery effect
	
	return ti.math.vec3(r,g,b)

@ti.func
def generic_water_colors():
	# Randomly generate blue and green dominant colors
	r = 30./255   # Low red to keep the color cool
	g = 80./255   # Mid-range green for teal/aqua tones
	b = 230./255  # High blue for a watery effect
	
	return ti.math.vec3(r,g,b)

@ti.func
def mix_rgb(color1:ti.math.vec3, color2:ti.math.vec3, weight:ti.f32 = 0.5):
	# Mix the two colors by weighted averaging each channel
	return color1 * weight + color2 * (1 - weight)


@ti.kernel
def reinhard_tone_mapping(image: ti.template(), PX: ti.template(), gamma: ti.f32):
	
	inv_gamma = 1.0 / gamma
	
	for i, j in PX:
		# Fetch the original color
		color = ti.Vector([image[i, j, 0], image[i, j, 1], image[i, j, 2]])
		R = color[0]
		G = color[1]
		B = color[2]

		# Apply Reinhard tone mapping to each channel
		R_mapped = R / (1.0 + R)
		G_mapped = G / (1.0 + G)
		B_mapped = B / (1.0 + B)

		# Apply gamma correction
		R_mapped = ti.pow(R_mapped, inv_gamma)
		G_mapped = ti.pow(G_mapped, inv_gamma)
		B_mapped = ti.pow(B_mapped, inv_gamma)

		# Write the mapped color to the output image
		image[i, j, 0] = R_mapped
		image[i, j, 1] = G_mapped
		image[i, j, 2] = B_mapped

@ti.func
def aces_tone_mapping(color: ti.types.vector(3, ti.f32)):
	# Constants for ACES approximation
	a = 2.51
	b = 0.03
	c = 2.43
	d = 0.59
	e = 0.14
	color_mapped = (color * (a * color + b)) / (color * (c * color + d) + e)
	return ti.math.clamp(color_mapped, 0.0, 1.0)

@ti.kernel
def aces_tone_mapping_kernel(image: ti.template(), PX:ti.template(), gamma: ti.f32):
	inv_gamma = 1.0 / gamma
	for i, j in PX:
		color = ti.Vector([image[i, j, 0], image[i, j, 1], image[i, j, 2]])
		# Apply ACES tone mapping
		color_mapped = aces_tone_mapping(color)
		# Apply gamma correction
		color_mapped = ti.pow(color_mapped, inv_gamma)
		image[i, j, 0] = color_mapped[0]
		image[i, j, 1] = color_mapped[1]
		image[i, j, 2] = color_mapped[2]

@ti.kernel
def toon_shader(image: ti.template(), PX:ti.template(), N_bin: ti.u8):
	for i, j in PX:
		color = ti.Vector([image[i, j, 0], image[i, j, 1], image[i, j, 2]])
		color *= N_bin
		color[0] = ti.math.round(color[0])
		color[1] = ti.math.round(color[1])
		color[2] = ti.math.round(color[2])

		image[i, j, 0] = color[0]/N_bin
		image[i, j, 1] = color[1]/N_bin
		image[i, j, 2] = color[2]/N_bin



@ti.func
def xy_to_ij(x:ti.template(), y:ti.template(), XX:ti.template(), YY:ti.template(), dx:ti.template(), dy:ti.template(), nx:ti.template(), ny:ti.template()):
	# Compute indices in grid
	j = (x - XX[0,0]) / dx
	i = (y - YY[0,0]) / dy

	i0 = int(ti.math.floor(i))
	j0 = int(ti.math.floor(j))

	# Ensure indices are within bounds
	i0 = max(0, min(i0, ny - 2))
	j0 = max(0, min(j0, nx - 2))

	return i, j, i0, j0



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


@ti.func
def is_in_shadow(
	point: ti.math.vec3,
	light_direction: ti.math.vec3,
	XX: ti.template(),
	YY: ti.template(),
	ZZ: ti.template(),
) -> ti.u1:
	# Offset the point slightly to prevent self-intersection (shadow acne)
	bias = 1e-4
	shadow_origin = point + bias * light_direction
	# Cast a ray towards the light source
	t = 0.0
	max_t = 10.0  # Adjust based on scene size
	nx, ny = XX.shape[1], XX.shape[0]

	ret = 0
	while t < max_t:
		current_point = shadow_origin + t * light_direction
		x, y, z = current_point

		# Check if the point is within the scene bounds
		if x < XX[0, 0] or x > XX[0, nx - 1] or y < YY[0, 0] or y > YY[ny - 1, 0]:
			break  # Light is not obstructed within the scene

		# Get surface Z at (x, y)
		surface_z = interpolate_AA(x, y, XX, YY, ZZ)
		if surface_z != -9999.0 and z <= surface_z:
			ret = 1  # Point is in shadow
			break

		t += RAY_STEP

	return ret  # Point is not in shadow


# Function to interpolate ZZ at (x, y) using bilinear interpolation
@ti.func
def interpolate_AA(
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

		i,j,i0,j0 = xy_to_ij(x, y, XX, YY, dx, dy, nx, ny)

		i1 = i0 + 1
		j1 = j0 + 1

		# Compute fractional parts
		s = i - i0
		t = j - j0

		# Get Z values at the corners
		z00 = ZZ[i0, j0]
		z10 = ZZ[i1, j0]
		z01 = ZZ[i0, j1]
		z11 = ZZ[i1, j1]

		if(z00 == -9999.0 or z01 == -9999.0 or z10 == -9999.0 or z11 == -9999.0):
			z = -9999.0
		else:
			# Perform bilinear interpolation
			z0 = z00 * (1 - s) + z10 * s
			z1 = z01 * (1 - s) + z11 * s
			z  = z0  * (1 - t) + z1  * t

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
	dz_dx = (interpolate_AA(x + h, y, XX, YY, ZZ) - interpolate_AA(x - h, y, XX, YY, ZZ)) / (2 * h)
	dz_dy = (interpolate_AA(x, y + h, XX, YY, ZZ) - interpolate_AA(x, y - h, XX, YY, ZZ)) / (2 * h)

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
	prev_dz = -9999.0
	intersection_point = ti.math.vec3(-9999.)
	normal = ti.math.vec3(-9999.)
	dz = 10.
	nx,ny = XX.shape[1],XX.shape[0]
	cont = True
	xf = -9999.
	while t < max_t and cont:
		# Compute current point along the ray
		point = ray_origin + t * ray_direction
		x, y, z = point

		# Get surface Z at (x, y) if within domain
		if x >= XX[0,0] and x <= XX[0,nx-1] and y >= YY[0,0] and y <= YY[ny-1,0]:
			surface_z = interpolate_AA(x, y, XX, YY, ZZ)
			if surface_z != -9999.0 and z != -9999.0:
				# Compute difference between ray's z and surface z
				dz = z - surface_z

				if prev_dz != -9999.0:
					if dz * prev_dz < 0:
						# The sign of dz changed, indicating a crossing
						# Perform linear interpolation to find intersection t
						t_intersect = t - RAY_STEP * dz / (dz - prev_dz)
						# Compute intersection point
						intersection_point = ray_origin + t_intersect * ray_direction
						x_int, y_int, z_int = intersection_point
						# Compute normal at intersection point
						normal = compute_normal((x_int), (y_int), XX, YY, ZZ)
						cont = False
					elif dz == 0.0:
						# Ray is exactly on the surface
						intersection_point = point
						normal = compute_normal(x, y,XX,YY,ZZ)
						cont = False

				prev_dz = dz
			else:
				# Surface Z is invalid
				pass
		else:
			# If the ray is outside the domain and moving away, terminate early
			if prev_dz != -9999.0 and dz > 0:
				break  # Ray is above the surface and moving upwards

		t += RAY_STEP

	# No intersection found
	return intersection_point, normal

@ti.func
def compute_ray_ortho(
	px: ti.template(),
	py: ti.template(),
	camera_position: ti.template(),
	camera_direction: ti.template(),
	camera_up: ti.template(),
	camera_right: ti.template(),
):
	# Ray direction is constant in orthographic projection
	ray_direction = camera_direction

	# Ray origin varies across the image plane
	ray_origin = (camera_position +
				  camera_right * px +
				  camera_up * py)
	return ray_origin, ray_direction

@ti.func
def compute_ray_ortho(
	px: ti.template(),
	py: ti.template(),
	camera_position: ti.template(),
	camera_direction: ti.template(),
	camera_up: ti.template(),
	camera_right: ti.template(),
	dx:ti.f32,
	dy:ti.f32
):
	# Ray direction is constant in orthographic projection
	ray_direction = ti.math.vec3(camera_direction)

	# Ray origin varies across the image plane
	ray_origin = ti.math.vec3(camera_position +
				  camera_right * (px + dx) +
				  camera_up * (py + dy))
	return ray_origin, ray_direction

@ti.func
def compute_ray_persp(
	px: ti.template(),
	py: ti.template(),
	camera_position: ti.template(),
	camera_direction: ti.template(),
	camera_up: ti.template(),
	camera_right: ti.template(),
	focal_length:ti.f32,
	dx:ti.f32,
	dy:ti.f32
):
	ray_origin = ti.math.vec3(camera_position[0],camera_position[1],camera_position[2])
	# Compute ray direction
	ray_direction = compute_ray_direction(px+dx, py+dy, camera_position,camera_direction,camera_up,camera_right,focal_length)
	return ray_origin, ray_direction

	

@ti.kernel
def render_gpu( 
	XX:ti.template(),
	YY:ti.template(),
	ZZ:ti.template(),
	HH:ti.template(),
	image:ti.template(),
	PX:ti.template(),
	PY:ti.template(),
	camera_position:ti.template(),
	camera_direction:ti.template(),
	camera_up:ti.template(),
	camera_right:ti.template(),
	focal_length:ti.f32, 
	image_height:ti.i32, 
	image_plane_height:ti.f32, 
	image_width:ti.i32,
	image_plane_width:ti.f32,
	N_AA:ti.i32,
	ortho:ti.u1,

	):
	
	pixel_width = (image_plane_width / image_width)
	pixel_height = (image_plane_height / image_height)
	nx,ny = XX.shape[1],XX.shape[0]

	# Simple shading using Lambertian reflection
	# Define light direction (e.g., from above)
	light_direction = ti.math.vec3(1.0, 1.0, 0.75)  # Light coming from (1,1,1)
	light_direction /= ti.math.length(light_direction)

	# Loop over each pixel in the image
	for i,j in ti.ndrange((0,image_height),(0,image_width)):
		# Compute pixel coordinates in image plane
		px = PX[i, j]
		py = PY[i, j]
		
		color = ti.math.vec3(0.)
		nav = 0
		ray_origin = ti.math.vec3(0.) 
		ray_direction = ti.math.vec3(0.)

		save_intersection_point = ti.math.vec3(0.)
		has_intersection = False

		for kk in range(N_AA):

			# Generate random offsets within the pixel
			dx = (ti.random(ti.f32) - 0.5) * pixel_width
			dy = (ti.random(ti.f32) - 0.5) * pixel_height


			# Ray origin is the camera position
			if(ortho == False):
				ray_origin, ray_direction = compute_ray_persp(px,py,camera_position,camera_direction,camera_up,camera_right,focal_length,dx,dy)	
			else:
				ray_origin, ray_direction = compute_ray_ortho(px,py,camera_position,camera_direction,camera_up,camera_right,dx,dy)	

			# Perform ray-surface intersection
			intersection_point, normal = ray_surface_intersection(ray_origin, ray_direction, XX, YY, ZZ)
			
			if intersection_point[0] != -9999.0:
				has_intersection = True
				save_intersection_point = intersection_point

				# Compute intensity
				intensity = ti.math.dot(normal, light_direction)
				intensity = max(0.01, min(intensity, 1.0))

				tcolor = ti.math.vec3(0.)
				# Assign color based on intensity
				if RENDER_COL == 0:
					tcolor += intensity * ti.math.vec3(1.0, 1.0, 1.0)  # White color scaled by intensity

				elif RENDER_COL == 1:
					x,y = intersection_point[0], intersection_point[1]
					tti, ttj, i0, j0 = xy_to_ij(x, y, XX, YY, dx, dy, nx, ny)
					tcolor += intensity * ti.math.vec3(1.0, 1.0, 1.0)
					th = interpolate_AA(x,y, XX, YY, HH)
					
					if(th > MIN_WATER):
						weight = max(MINWACOL, min(1., th/2.))
						tcolor = mix_rgb(tcolor, generic_water_colors(), (1. - weight))

				
				color += tcolor
				nav += 1

		color /= ti.max(1,nav)

		if(has_intersection and  SHADOW_BLEND > 0):
			if(is_in_shadow(save_intersection_point, light_direction, XX, YY, ZZ)):
				color = mix_rgb(color, ti.math.vec3(0.), SHADOW_BLEND)

		if(color[0] > 0 or color[1] > 0 or color[2] > 0):
			image[i, j, 0] = color[0]
			image[i, j, 1] = color[1]
			image[i, j, 2] = color[2]
		else:
			image[i, j, 0] = BGCOL_r
			image[i, j, 1] = BGCOL_g
			image[i, j, 2] = BGCOL_b





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







class RT_data:

	def __init__(self, grid, exaggeration_factor = 0.5, hwater = None):


		# ==========================================
		# Step 1: Data Preparation
		# ==========================================
		self.grid = grid

		Z = self.grid.Z[1:-1,1:-1]

		self._exaggeration_factor = exaggeration_factor

		mask = Z!=-9999

		# Normalize Z to range from 0 to 1, then apply exaggeration factor
		Z_min, Z_max = Z[mask].min(), Z[mask].max()
		Z_normalized = np.copy(Z)
		Z_normalized[mask] = (Z[mask] - Z_min) / (Z_max - Z_min) * self._exaggeration_factor


		self.ny, self.nx = Z.shape
		self.x = np.linspace(-1, 1, self.nx)
		rat = self.ny/self.nx
		self.y = np.linspace(-1*rat, 1*rat, self.ny)
		_XX, _YY = np.meshgrid(self.x, self.y)

		# Set ZZ to the normalized Z values
		_ZZ = Z_normalized

		# Compute the center point (0, 0, median ZZ)
		self.center_Z = np.median(_ZZ[mask])
		self.center_point = np.array([0.0, 0.0, self.center_Z])

		self.XX = ti.field(ti.f32, shape = _XX.shape)
		self.XX.from_numpy(_XX.astype(np.float32))
		self.YY = ti.field(ti.f32, shape = _YY.shape)
		self.YY.from_numpy(_YY.astype(np.float32))
		self.ZZ = ti.field(ti.f32, shape = _ZZ.shape)

		if(hwater is not None):
			th = (np.copy(hwater[1:-1,1:-1])) / (Z_max - Z_min) * self._exaggeration_factor
			self.HH = ti.field(ti.f32, shape = _ZZ.shape)
			self.HH.from_numpy(hwater[1:-1,1:-1].astype(np.float32))
			self.ZZ.from_numpy(_ZZ.astype(np.float32) + th.astype(np.float32))
		else:
			# Dummy array. Needed.
			self.HH = ti.field(ti.f32, shape = (4,4))
			self.HH.fill(0.)
			self.ZZ.from_numpy(_ZZ.astype(np.float32))



class RT_camera:
	def __init__(self,
		rtdata,
		camera_distance = 2.0,      # Distance from the center (adjust for zoom)
		camera_azimuth_deg = 270,   # Azimuth angle in degrees
		camera_elevation_deg = 45,  # Elevation angle from horizontal in degrees):
	):
		'''
		Sets a first time the camera attributes
		'''
		self.rtdata = rtdata
		self.camera_distance = None
		self.camera_azimuth_deg = None
		self.camera_elevation_deg = None
		self.camera_position = ti.Vector(np.zeros(3, dtype = np.float32))
		self.camera_direction = ti.Vector(np.zeros(3, dtype = np.float32))
		self.camera_right = ti.Vector(np.zeros(3, dtype = np.float32))
		self.camera_up  = ti.Vector(np.zeros(3, dtype = np.float32))

		self.update(camera_distance,camera_azimuth_deg,camera_elevation_deg)


		

	def update(self,
		camera_distance,
		camera_azimuth_deg,
		camera_elevation_deg,
	):

		self.camera_distance = camera_distance
		self.camera_azimuth_deg = camera_azimuth_deg
		self.camera_elevation_deg = camera_elevation_deg

		# Convert angles to radians
		theta = np.deg2rad(self.camera_azimuth_deg)      # Azimuth angle in radians
		phi = np.deg2rad(self.camera_elevation_deg)      # Elevation angle in radians

		# Compute self.camera position using spherical coordinates
		camera_x = self.rtdata.center_point[0] + self.camera_distance * np.cos(phi) * np.cos(theta)
		camera_y = self.rtdata.center_point[1] + self.camera_distance * np.cos(phi) * np.sin(theta)
		camera_z = self.rtdata.center_point[2] + self.camera_distance * np.sin(phi)
		_camera_position = np.array([camera_x, camera_y, camera_z])

		# Camera direction vector (from camera position to center point)
		_camera_direction = self.rtdata.center_point - _camera_position
		_camera_direction /= np.linalg.norm(_camera_direction)  # Normalize

		# Camera right vector
		_camera_right = np.cross(_camera_direction, WORLD_UP)
		_camera_right /= np.linalg.norm(_camera_right)  # Normalize

		# Camera up vector
		_camera_up = np.cross(_camera_right, _camera_direction)
		_camera_up /= np.linalg.norm(_camera_up)

		_camera_position = np.array([camera_x, camera_y, camera_z])

		self.camera_position = ti.Vector(_camera_position.astype(np.float32))
		self.camera_direction = ti.Vector(_camera_direction.astype(np.float32))
		self.camera_right = ti.Vector(_camera_right.astype(np.float32))
		self.camera_up = ti.Vector(_camera_up.astype(np.float32))


class RT_image:

	def __init__(self,
		image_width = 1200, 
		image_height = 900,
		focal_length = 1.0,         # Adjust as needed
		fov_deg = 60,               # Field of view in degrees (reduced to minimize distortion)
		
		ortho = False,
		ortho_height = 5.
		):

		self.image_width = image_width
		self.image_height = image_height
		self.focal_length = focal_length
		self.fov_deg = fov_deg
		# Set up the image plane dimensions based on field of view and aspect ratio
		self.aspect_ratio = image_width / image_height
		self.fov_rad = np.deg2rad(fov_deg)  # Convert to radians


		self.ortho = ortho
		self.ortho_height = ortho_height
		
		# Compute image plane dimensions
		if(self.ortho):
			self.image_plane_height = self.ortho_height
			self.image_plane_width = self.image_plane_height * self.aspect_ratio
		else:
			self.image_plane_height = 2 * self.focal_length * np.tan(self.fov_rad / 2)
			self.image_plane_width = self.image_plane_height * self.aspect_ratio

		# Generate pixel coordinates in the image plane
		self.px = np.linspace(-0.5 * self.image_plane_width, 0.5 * self.image_plane_width, self.image_width)
		self.py = np.linspace(-0.5 * self.image_plane_height, 0.5 * self.image_plane_height, self.image_height)
		_PX, _PY = np.meshgrid(self.px, self.py)

		self.PX = ti.field(ti.f32, shape = _PX.shape)
		self.PX.from_numpy(_PX.astype(np.float32))
		self.PY = ti.field(ti.f32, shape = _PY.shape)
		self.PY.from_numpy(_PY.astype(np.float32))
		
		self.image = ti.field(ti.f32, shape = (image_height, image_width, 3) )
		self.image.fill(0.)

class RT_renderer:

	def __init__(self,
		N_AA = 5
	):
		self.N_AA = N_AA

def _render_RT(
	rtdata,
	camera,
	image,
	renderer,
	which = 'gray'
):
	'''
	Internal wrapper for the gray renderer operating from the RT objects
	'''
	render_gpu(
		rtdata.XX,rtdata.YY,rtdata.ZZ,rtdata.HH,
		image.image,image.PX,image.PY,
		camera.camera_position,camera.camera_direction,camera.camera_up,camera.camera_right,
		image.focal_length,image.image_height,image.image_plane_height,image.image_width,image.image_plane_width, 
		renderer.N_AA, image.ortho
	)



def std_gray_RT(
	grid:scb.raster.RegularRasterGrid,
	exaggeration_factor = 0.5,  # Adjust this as needed
	camera_distance = 2.0,      # Distance from the center (adjust for zoom)
	camera_azimuth_deg = 270,   # Azimuth angle in degrees
	camera_elevation_deg = 45,  # Elevation angle from horizontal in degrees
	focal_length = 1.0,         # Adjust as needed
	image_width = 1200,         # Image width in pixels
	image_height = 900,         # Image height in pixels
	fov_deg = 60,               # Field of view in degrees (reduced to minimize distortion)
	N_AA = 5,
	tone_mapping = False,
	toon = 0,
	ortho = False,
	ortho_height = 5.,
	hw = None

	):

	ti.init(ti.gpu)

	rtdata = RT_data(grid, exaggeration_factor = exaggeration_factor) if (hw is None) else RT_data(grid, exaggeration_factor = exaggeration_factor, hwater = hw)

	camera = RT_camera(rtdata, camera_distance = camera_distance, camera_azimuth_deg = camera_azimuth_deg, camera_elevation_deg = camera_elevation_deg)
	image = RT_image(image_width = image_width, image_height = image_height, focal_length = focal_length, fov_deg = fov_deg, ortho = ortho,ortho_height = ortho_height)
	renderer = RT_renderer(N_AA = N_AA)

	_render_RT(rtdata, camera, image, renderer)

	if(tone_mapping):
		# reinhard_tone_mapping(image.image, image.PX, 2.2)
		aces_tone_mapping_kernel(image.image, image.PX, 2.2)

	if(toon > 0):
		toon_shader(image.image, image.PX, toon)

	return image.image.to_numpy()[::-1]

def std_water_RT(
	grid:scb.raster.RegularRasterGrid,
	hw:np.ndarray,
	exaggeration_factor = 0.5,  # Adjust this as needed
	camera_distance = 2.0,      # Distance from the center (adjust for zoom)
	camera_azimuth_deg = 270,   # Azimuth angle in degrees
	camera_elevation_deg = 45,  # Elevation angle from horizontal in degrees
	focal_length = 1.0,         # Adjust as needed
	image_width = 1200,         # Image width in pixels
	image_height = 900,         # Image height in pixels
	fov_deg = 60,               # Field of view in degrees (reduced to minimize distortion)
	N_AA = 5,
	tone_mapping = False,
	toon = 0

	):

	ti.init(ti.gpu)

	rtdata = RT_data(grid, exaggeration_factor = exaggeration_factor, hwater = hw)
	camera = RT_camera(rtdata, camera_distance = camera_distance, camera_azimuth_deg = camera_azimuth_deg, camera_elevation_deg = camera_elevation_deg)
	image = RT_image(image_width = image_width, image_height = image_height, focal_length = focal_length, fov_deg = fov_deg )
	renderer = RT_renderer(N_AA = N_AA)

	_render_RT(rtdata, camera, image, renderer, which = 'water1')

	if(tone_mapping):
		# reinhard_tone_mapping(image.image, image.PX, 2.2)
		aces_tone_mapping_kernel(image.image, image.PX, 2.2)

	if(toon > 0):
		toon_shader(image.image, image.PX, toon)

	return image.image.to_numpy()[::-1]






def set_bg_constants(r,g,b):
	'''
	To be called before any RT operation
	Sets the constants for the background color in the RT visu
	'''
	global BGCOL_r, BGCOL_g, BGCOL_b
	BGCOL_r = r
	BGCOL_g = g
	BGCOL_b = b






def set_render_mode(which = 'gray'):
	global RENDER_COL
	if(which.lower() == 'gray'):
		RENDER_COL = 0
	elif(which.lower() == 'water'):
		RENDER_COL = 1






















#########################################################
#########################################################
#########################################################
################# LEGACY ################################
#########################################################
#########################################################
#########################################################





























def yolo(
	grid:scb.raster.RegularRasterGrid,
	exaggeration_factor = 0.5,  # Adjust this as needed
	camera_distance = 2.0,      # Distance from the center (adjust for zoom)
	camera_azimuth_deg = 270,   # Azimuth angle in degrees
	camera_elevation_deg = 45,  # Elevation angle from horizontal in degrees
	focal_length = 1.0,         # Adjust as needed
	image_width = 1200,         # Image width in pixels
	image_height = 900,         # Image height in pixels
	fov_deg = 60,               # Field of view in degrees (reduced to minimize distortion)
	N_AA = 10
):

	
	ti.init(ti.gpu)

	rtdata = RT_data(grid, exaggeration_factor = exaggeration_factor)
	camera = RT_camera(rtdata, camera_distance = camera_distance, camera_azimuth_deg = camera_azimuth_deg, camera_elevation_deg = camera_elevation_deg)
	image = RT_image(image_width = image_width, image_height = image_height, focal_length = focal_length, fov_deg = fov_deg )
	renderer = RT_renderer(N_AA = N_AA)

	window = ti.ui.Window("YOLO", (image_width, image_height))
	canvas = window.get_canvas()
	gui = window.get_gui()

	out = ti.field(ti.f32, shape = (image_height, image_width, 3) )

	while window.running:


		# Start GUI
		gui.begin("Settings", 0.05, 0.05, 0.3, 0.3)
		gui.text("Camera Controls")

		camera_azimuth_deg = gui.slider_float("Azimuth", camera_azimuth_deg, 0.0, 360.0)
		camera_elevation_deg = gui.slider_float("Elevation", camera_elevation_deg, 0.0, 90.0)
		camera_distance = gui.slider_float("Distance", camera_distance, 0.1, 10.0)
		
		gui.end()

		# Update camera parameters
		camera.update(camera_distance, camera_azimuth_deg, camera_elevation_deg)

		# Render the image
		_gray_RT(rtdata, camera, image, renderer)
		rotate(out, image.image, image.PX)
		# Draw the image onto the canvas
		canvas.set_image(out)

		# Show the frame
		window.show()




# LEGACY RT, to keep while the others are not ready
def legacy_gray_RT(
	grid:scb.raster.RegularRasterGrid,
	exaggeration_factor = 0.5,  # Adjust this as needed
	camera_distance = 2.0,      # Distance from the center (adjust for zoom)
	camera_azimuth_deg = 270,   # Azimuth angle in degrees
	camera_elevation_deg = 45,  # Elevation angle from horizontal in degrees
	focal_length = 1.0,         # Adjust as needed
	image_width = 1200,         # Image width in pixels
	image_height = 900,         # Image height in pixels
	fov_deg = 60,               # Field of view in degrees (reduced to minimize distortion)
	N_AA = 5,

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
	WORLD_UP = np.array([0, 0, 1])

	# Camera right vector
	_camera_right = np.cross(_camera_direction, WORLD_UP)
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


	render_gray(XX,YY,ZZ,image,PX,PY,camera_position,camera_direction,camera_up,camera_right,focal_length,image_height,image_plane_height,image_width,image_plane_width, N_AA)
	
	# rotate(out, image, PX)


	return image.to_numpy()[::-1]




@ti.kernel
def render_gray( 
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
	image_plane_height:ti.f32, 
	image_width:ti.i32,
	image_plane_width:ti.f32,
	N_AA:ti.i32,

	):
	
	pixel_width = (image_plane_width / image_width)
	pixel_height = (image_plane_height / image_height)
	nx,ny = XX.shape[1],XX.shape[0]

	# Loop over each pixel in the image
	for i,j in ti.ndrange((0,image_height),(0,image_width)):
		# Compute pixel coordinates in image plane
		px = PX[i, j]
		py = PY[i, j]
		

		# Ray origin is the camera position
		ray_origin = ti.math.vec3(camera_position[0],camera_position[1],camera_position[2])

		color = ti.math.vec3(0.)
		nav = 0

		for kk in range(N_AA):

			# Generate random offsets within the pixel
			dx = (ti.random(ti.f32) - 0.5) * pixel_width
			dy = (ti.random(ti.f32) - 0.5) * pixel_height

			# Compute ray direction
			ray_direction = compute_ray_direction(px+dx, py+dy, camera_position,camera_direction,camera_up,camera_right,focal_length)

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
				color += intensity * ti.math.vec3(1.0, 1.0, 1.0)  # White color scaled by intensity
				nav += 1

		color /= ti.max(1,nav)

		image[i, j, 0] = color[0]
		image[i, j, 1] = color[1]
		image[i, j, 2] = color[2]

@ti.kernel
def render_with_water( 
	XX:ti.template(),
	YY:ti.template(),
	ZZ:ti.template(),
	HH:ti.template(),
	image:ti.template(),
	PX:ti.template(),
	PY:ti.template(),
	camera_position:ti.template(),
	camera_direction:ti.template(),
	camera_up:ti.template(),
	camera_right:ti.template(),
	focal_length:ti.f32, 
	image_height:ti.i32, 
	image_plane_height:ti.f32, 
	image_width:ti.i32,
	image_plane_width:ti.f32,
	N_AA:ti.i32,

	):
	
	nx,ny = XX.shape[1], XX.shape[0]
	XX[0,1] - XX[0,0], YY[0,1] - YY[0,0]
	pixel_width = (image_plane_width / image_width)
	pixel_height = (image_plane_height / image_height)

	# Loop over each pixel in the image
	for i,j in ti.ndrange((0,image_height),(0,image_width)):
		# Compute pixel coordinates in image plane
		px = PX[i, j]
		py = PY[i, j]
		

		# Ray origin is the camera position
		ray_origin = ti.math.vec3(camera_position[0],camera_position[1],camera_position[2])

		color = ti.math.vec3(0.)
		nav = 0

		for kk in range(N_AA):

			# Generate random offsets within the pixel
			dx = (ti.random(ti.f32) - 0.5) * pixel_width
			dy = (ti.random(ti.f32) - 0.5) * pixel_height

			# Compute ray direction
			ray_direction = compute_ray_direction(px+dx, py+dy, camera_position,camera_direction,camera_up,camera_right,focal_length)

			# Perform ray-surface intersection
			intersection_point, normal = ray_surface_intersection(ray_origin, ray_direction, XX, YY, ZZ)
			
			if intersection_point[2] != -9999.0:

				# Simple shading using Lambertian reflection
				# Define light direction (e.g., from above)
				light_direction = ti.math.vec3(1.0, 1.0, 1.0)  # Light coming from (1,1,1)
				light_direction /= ti.math.length(light_direction)

				# Compute intensity
				intensity = ti.math.dot(normal, light_direction)
				intensity = max(0.0, min(intensity, 1.0))
				x,y = intersection_point[0], intersection_point[1]

				tcolor = ti.math.vec3(0.,0.,0.)
				if x >= XX[0,0] and x <= XX[0,nx-1] and y >= YY[0,0] and y <= YY[ny-1,0]:
					tti, ttj, i0, j0 = xy_to_ij(x, y, XX, YY, dx, dy, nx, ny)
					# print(i0,j0)

					tcolor = intensity * ti.math.vec3(1.0, 1.0, 1.0)

					weight = 1. if HH[i0,j0] > 0.05 else 0.

					tz = interpolate_AA(x,y, XX, YY, ZZ)
					th = interpolate_AA(x,y, XX, YY, HH)
					# if(HH[i0,j0]> 1.):
					# 	print(i0,j0,HH[i0,j0])
					weight = min(1., th/2.)

					tcolor = mix_rgb(tcolor, generate_realistic_water_colors(), (1. - weight))
					# tcolor = generic_water_colors() if th > 0.05 else tcolor
					# tcolor = ti.math.vec3(1.0, 1.0, 1.0) * interpolate_AA(x,y,XX,YY,ZZ)

					# Assign color based on intensity
					color += tcolor # White color scaled by intensity
					nav += 1

		color /= ti.max(1,nav)
		image[i, j, 0] = color[0]
		image[i, j, 1] = color[1]
		image[i, j, 2] = color[2]