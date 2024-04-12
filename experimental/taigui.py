import scabbard as scb
import dagger as dag
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import math as m
import matplotlib
import random
from scipy.ndimage import gaussian_filter
import time
import sys

import taichi as ti

import numpy as np

def calculate_orbit_position(center, distance, azimuth, elevation):
	"""
	Calculate the new camera position for orbiting around a center point.
	
	Arguments:
	- center: The orbit center point.
	- distance: The distance from the center to the camera.
	- azimuth: The azimuth angle in degrees around the Z-axis.
	- elevation: The elevation angle in degrees from the XY-plane.
	
	Returns:
	- The new camera position as a numpy array.
	"""
	# Convert degrees to radians
	azimuth_rad = np.radians(azimuth)
	elevation_rad = np.radians(elevation)
	
	# Ensure the elevation angle is within bounds to avoid flipping over the top
	elevation_rad = np.clip(elevation_rad, -np.pi/2 + 1e-5, np.pi/2 - 1e-5)
	
	# Calculate camera position
	x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
	y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
	z = distance * np.sin(elevation_rad)
	
	return center + np.array([x, y, z])

def update_camera(camera_position, center_point, azimuth_change, elevation_change, distance):
	"""
	Update the camera position by calculating its orbit around a center point.
	"""
	# Calculate current azimuth and elevation based on the camera's position
	offset_vector = camera_position - center_point
	current_distance = np.linalg.norm(offset_vector)
	
	current_azimuth = np.arctan2(offset_vector[1], offset_vector[0])
	current_elevation = np.arcsin(offset_vector[2] / current_distance)
	
	# Update azimuth and elevation
	new_azimuth = np.degrees(current_azimuth) + azimuth_change
	new_elevation = np.degrees(current_elevation) + elevation_change
	
	# Calculate the new camera position
	new_position = calculate_orbit_position(center_point, distance, new_azimuth, new_elevation)
	return new_position




def process_camera_event(window, campos, center, step, stepdist, distance):

	if window.is_pressed(ti.ui.UP) and window.is_pressed(ti.ui.SHIFT):
		distance += stepdist
		campos = update_camera(campos, center, 0, 0, distance)

	elif window.is_pressed(ti.ui.DOWN) and window.is_pressed(ti.ui.SHIFT):
		distance -= stepdist
		campos = update_camera(campos, center, 0, 0, distance)
	elif window.is_pressed(ti.ui.LEFT):
		campos = update_camera(campos, center, -step, 0, distance)

	elif window.is_pressed(ti.ui.RIGHT):
		campos = update_camera(campos, center, step, 0, distance)

	elif window.is_pressed(ti.ui.DOWN):
		campos = update_camera(campos, center, 0, step, distance)

	elif window.is_pressed(ti.ui.UP):
		campos = update_camera(campos, center, 0, -step, distance)



	# elif window.is_pressed(ti.ui.DOWN):
	# 	campos = update_camera(campos, center, 0, step, distance)

	# elif window.is_pressed(ti.ui.UP):
	# 	campos = update_camera(campos, center, 0, -step, distance)

	return window, campos, distance









env = scb.env_from_DEM("dem.tif")
env.init_connector()
env.init_GF2()
env.graphflood.set_uniform_P(50 * 1e-3/3600)
env.graphflood.init()
env.graphflood.compute_entry_points_from_P(0.5)
env.graphflood.set_dt(5e-3)

env.graphflood.prepare_tsg()
for i in range(10):
	env.graphflood.run()




import taichi as ti
ti.init(arch=ti.cuda) # Alternatively, ti.init(arch=ti.cpu)

nx, ny = env.grid.nx, env.grid.ny

z = ti.field( dtype = ti.f32, shape=(ny, nx))
tz = env.grid.Z2D.astype(np.float32)

z.from_numpy((tz - tz.min())/(tz.max() - tz.min()) * 0.4)

num_triangles = (nx - 1) * (ny - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, nx * ny)



@ti.kernel
def set_indices():
	for i, j in ti.ndrange(ny, nx):
		if i < ny - 1 and j < nx - 1:
			square_id = (i * (nx - 1)) + j
			# 1st triangle of the square
			indices[square_id * 6 + 0] = i * nx + j
			indices[square_id * 6 + 1] = (i + 1) * nx + j
			indices[square_id * 6 + 2] = i * nx + (j + 1)
			# 2nd triangle of the square
			indices[square_id * 6 + 3] = (i + 1) * nx + j + 1
			indices[square_id * 6 + 4] = i * nx + (j + 1)
			indices[square_id * 6 + 5] = (i + 1) * nx + j


@ti.kernel
def set_vertices():
	for i, j in ti.ndrange(ny, nx):
		vertices[i * nx + j].x = i/nx
		vertices[i * nx + j].y = j/ny
		vertices[i * nx + j].z = z[i, j]

set_indices()

window = ti.ui.Window("Cloth", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
campos = ti.ui.make_camera()

	# for i in range(30):
	# 	step()

camera = ti.ui.Camera()
camera.projection_mode(ti.ui.ProjectionMode.Orthogonal)
set_vertices()

import time

camposx = 0.5
camposy = -0.5
camposz = 0.5

campos = np.array([camposx,camposy,camposz])
center = np.array([0.5, 0.5, 0.5])
distance = np.linalg.norm(campos - center)
step = 5
stepdist = 5

while window.running:
	
	window, campos, distance = process_camera_event(window, campos, center, step, stepdist, distance)
	# if window.is_pressed(ti.ui.LEFT):
	# 	campos = update_camera(campos, center, -step, 0, distance)
	# if window.is_pressed(ti.ui.RIGHT):
	# 	campos = update_camera(campos, center, step, 0, distance)
	# if window.is_pressed(ti.ui.DOWN):
	# 	campos = update_camera(campos, center, 0, step, distance)
	# if window.is_pressed(ti.ui.UP):
	# 	campos = update_camera(campos, center, 0, -step, distance)

	camera.position(campos[0],campos[1],campos[2])  # x, y, z
	camera.lookat(center[0], center[1], center[2])
	camera.up(0, 0, 1)
	scene.set_camera(camera)    
	scene.point_light(pos=(0.5, 0.5, 2), color=(1, 1, 1))
	scene.mesh(vertices, indices=indices, 
			   color=(0.5, 0.5, 0.5), two_sided = True)
	canvas.scene(scene)

	window.show()