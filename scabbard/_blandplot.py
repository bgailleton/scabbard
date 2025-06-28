################################################################################
# Import Libraries
import bpy
import numpy as np
import math
from mathutils import Matrix
import importlib
import sys
import os
import argparse


# Create the parser for command-line arguments
parser = argparse.ArgumentParser(description='Process some arguments.')

# Add arguments for Blender options and rendering parameters
parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration for rendering.')
parser.add_argument('--perspective', action='store_true', help='Set camera to perspective mode (otherwise orthogonal).')
parser.add_argument('--fprefix', type=str, help='File prefix for input data (e.g., topography, displacement, color arrays).')
parser.add_argument('--save_prefix', type=str, help='File prefix for the output rendered image.')
parser.add_argument('--ortho_scale', type=float, default = 2., help='Orthogonal scale for the camera (when in orthogonal mode).')
parser.add_argument('--focal_length', type=float, default = 40., help='Focal length of the camera in mm (when in perspective mode).')
parser.add_argument('--f_stop', type=float, default = 500000., help='F-stop value for depth of field (lower for shallow DoF, higher for wide DoF).')
parser.add_argument('--dx', type=float, default = 30., help='Spatial resolution (grid cell size) in meters.')
parser.add_argument('--shiftx', type=float, default = 0., help='Camera shift in the X-direction to center the topography.')
parser.add_argument('--shifty', type=float, default = 0., help='Camera shift in the Y-direction to center the topography.')
parser.add_argument('--camera_tilt', type = float, default = 45.0, help = 'Camera tilt in degrees from horizontal.')
parser.add_argument('--camera_rotation', type = float, default = 0., help = 'Camera rotation in degrees clockwise from North.')
parser.add_argument('--sun_tilt', type = float, default = 25.0, help = 'Sun tilt in degrees from horizontal.')
parser.add_argument('--sun_rotation', type = float, default = 315.0, help = 'Sun rotation in degrees clockwise from North.')
parser.add_argument('--sun_intensity', type = float, default = 0.2, help = 'Intensity of the sun light.')
parser.add_argument('--exaggeration', type = float, default = 1.5, help = 'Vertical exaggeration factor for the topography.')
parser.add_argument('--recast_minZ', type = float, default = 0., help = 'Minimum Z-value for recasting (normalization).')
parser.add_argument('--recast_maxZ', type = float, default = 1.4, help = 'Maximum Z-value for recasting (normalization).')
parser.add_argument('--number_of_subdivisions', type = int, default = 1000 , help = 'Number of subdivisions for the mesh, higher values increase detail.')
parser.add_argument('--res_x', type = int, default = 1920 , help = 'X-resolution of the rendered image.')
parser.add_argument('--res_y', type = int, default = 1080 , help = 'Y-resolution of the rendered image.')
parser.add_argument('--samples', type = int, default = 50 , help = 'Number of rendering samples (higher for better quality, longer render time).')
parser.add_argument('--relief', type = float, default = 50. , help = 'Relief scaling factor.')







args, unknown = parser.parse_known_args()

print(args)
print('\n\n\n')
print(unknown)
print('\n\n\n')


################################################################################
# This sets the render engine to CYCLES.
# In order for TopoBlender to render your topography correctly, you must use
# this render engine. This engine is optimized for GPUs, so if your computer
# lacks a GPU, TopoBlender may be slow.
GPU_boolean = args.gpu
delete_all = True

# Define current scene
scn = bpy.context.scene
if(delete_all):
    # Select all objects in the scene
    bpy.ops.object.select_all(action='SELECT')

# Delete selected objects
bpy.ops.object.delete()

# Check the render engine, and change to CYCLES if not already set
if not scn.render.engine == 'CYCLES':
    scn.render.engine = 'CYCLES'

# If Cycles is the render engine, enable GPU rendering if user selected GPU option
if scn.render.engine == 'CYCLES':
    if GPU_boolean == 1:
        scn.cycles.device = 'GPU'


# Load input data from numpy files
data = np.load(args.fprefix + ".npy")
displacement_array = np.load(args.fprefix + "_displacement_array.npy")
color_array = np.load(args.fprefix + "_color_array.npy")

# Set output file path
output = args.save_prefix + ".png"

# Get dimensions and spatial resolution
ny,nx = data.shape
dx,dy = args.dx, args.dx

# Standard nodata value
nodata_value = -9999


########## Camera Parameters ##########
# Camera type (perspective or orthogonal)
camera_type = 'perspective' if args.perspective else 'orthogonal' 

# Orthogonal camera settings
ortho_scale = args.ortho_scale # When using orthogonal scale, increase to "zoom" out

# Perspective camera settings
focal_length = args.focal_length # mm when using perspective camera, increase to zoom in
f_stop = args.f_stop # Affects depth of field, lower for a shallow DoF, higher for wide DoF
shift_x = args.shiftx # You may need to shift the camera to center the topo in the frame
shift_y = args.shifty # You may need to shift the camera to center the topo in the frame

# Camera location and orientation
camera_tilt = args.camera_tilt # Degrees from horizontal
camera_rotation = args.camera_rotation # Camera location degrees CW from North
######################################


########## Sun Properties ##############
sun_tilt = args.sun_tilt # Degrees from horizontal
sun_rotation = args.sun_rotation # Degrees CW from North
sun_intensity = args.sun_intensity # Sun intensity
######################################


##### Landscape Representation #########
number_of_subdivisions = args.number_of_subdivisions # Number of subdivisions, more increases the detail
exaggeration = args.exaggeration # Vertical exaggeration factor
recast_minZ = args.recast_minZ # Minimum Z-value for recasting (normalization)
recast_maxZ = args.recast_maxZ # Maximum Z-value for recasting (normalization)
luminosity_scale = 0.7 # Scaling factor for luminosity
######################################

######### Render Settings ##############
res_x = args.res_x # X-resolution of the render
res_y = args.res_y # Y-resolution of the render
samples = args.samples # Number of samples that decides how "good" the render looks. More is better but takes longer
######################################

# Convert nodata value to NaN in the data array
data[data==nodata_value] = np.nan
# Calculate y_length for scaling the topography
y_length = (2.0+float(data.shape[1])) * dx


# Create Blender image data blocks from numpy arrays
displacement_image = bpy.data.images.new('displacement_data', data.shape[0]+2, data.shape[1]+2, alpha=False, float_buffer = True)
color_image = bpy.data.images.new('color_data', data.shape[0]+2, data.shape[1]+2, alpha=False, float_buffer = True)
# Fast way to set pixels (since 2.83)
displacement_image.pixels.foreach_set(displacement_array)
color_image.pixels.foreach_set(color_array)
# Pack the images into the .blend file so they are saved with it
displacement_image.pack()
color_image.pack()
print("C") 
# Create a plane mesh in Blender
plane_size = 1.0
topo_mesh = bpy.ops.mesh.primitive_plane_add(size=plane_size)
topo_obj = bpy.context.active_object

# Change the scale of the object to match the data aspect ratio
topo_obj.scale = ((2.0+data.shape[0])/(2.0+data.shape[1]),1,1)

# Add a new material to the object
topo_mat = bpy.data.materials.new("topo_mat")
topo_mat.cycles.displacement_method = "DISPLACEMENT" # Set displacement method for Cycles
topo_mat.use_nodes = True # Enable node-based material

# Calculate subdivisions for the mesh based on the desired number of subdivisions
order_of_magnitude = math.floor(math.log10(number_of_subdivisions))
first_digit = int(np.round(number_of_subdivisions / (10.0 ** order_of_magnitude)))

# Assign the material to the object
topo_obj.data.materials.append(topo_mat)
# Enter edit mode to subdivide the mesh
bpy.ops.object.mode_set(mode="EDIT")
for i in range(0,order_of_magnitude):
    bpy.ops.mesh.subdivide(number_cuts=10)
bpy.ops.mesh.subdivide(number_cuts=first_digit)
# Exit edit mode
bpy.ops.object.mode_set(mode="OBJECT")

# Add an image texture node for displacement
displacement_image_node = topo_mat.node_tree.nodes.new("ShaderNodeTexImage")
# Assign the displacement image to the node
displacement_image_node.image = displacement_image
# Change color space to linear for accurate displacement
displacement_image_node.image.colorspace_settings.name="Linear FilmLight E-Gamut"

# Add an image texture node for color
color_image_node = topo_mat.node_tree.nodes.new("ShaderNodeTexImage")
# Assign the color image to the node
color_image_node.image = color_image
# Change color space to linear for accurate color representation
color_image_node.image.colorspace_settings.name="Linear FilmLight E-Gamut"
    
# Add a displacement node
displacement_node = topo_mat.node_tree.nodes.new("ShaderNodeDisplacement")
# Set the scale of the displacement based on exaggeration, plane size, relief, and y_length
displacement_node.inputs.get("Scale").default_value = exaggeration * plane_size * args.relief / y_length
displacement_node.inputs.get("Midlevel").default_value = 0.0 # Set mid-level for displacement

# Add a world sky texture for lighting
topo_world = bpy.data.worlds.new('topo_world')
topo_world.use_nodes = True # Enable node-based world settings
topo_world_node = topo_world.node_tree.nodes.new("ShaderNodeTexSky")
# Set sun elevation and rotation based on arguments
topo_world_node.sun_elevation = np.radians(sun_tilt)
topo_world_node.sun_rotation = np.radians(sun_rotation-90.0)
# Set sun intensity
topo_world_node.sun_intensity = sun_intensity
# Connect sky texture output to world background input
topo_world.node_tree.links.new(topo_world_node.outputs['Color'], topo_world.node_tree.nodes['Background'].inputs[0])

# Add a camera
camera_distance = 2.0 # meters
cam = bpy.data.cameras.new('topo_cam')
cam_obj = bpy.data.objects.new('topo_cam',cam)
# Set camera rotation based on arguments
cam_obj.rotation_euler = (np.radians(90.0 - camera_tilt), np.radians(0), np.radians(270.0 - camera_rotation))
# Set camera distance
cam_obj.matrix_basis @= Matrix.Translation((0.0, 0.0, camera_distance))

# Configure camera type (orthogonal or perspective)
if camera_type == 'orthogonal':
    bpy.data.cameras['topo_cam'].type = 'ORTHO'
    bpy.data.cameras['topo_cam'].ortho_scale = ortho_scale
elif camera_type == 'perspective':
    bpy.data.cameras['topo_cam'].type = 'PERSP'
    bpy.data.cameras['topo_cam'].lens = focal_length
    bpy.data.cameras['topo_cam'].shift_x = shift_x
    bpy.data.cameras['topo_cam'].shift_y = shift_y
    bpy.data.cameras['topo_cam'].dof.use_dof = True # Enable depth of field
    bpy.data.cameras['topo_cam'].dof.aperture_fstop = f_stop
    bpy.data.cameras['topo_cam'].dof.focus_distance = camera_distance
    
# Connect material nodes
topo_mat.node_tree.links.new(displacement_image_node.outputs["Color"], \
                             displacement_node.inputs["Height"])
topo_mat.node_tree.links.new(displacement_node.outputs["Displacement"], \
                             topo_mat.node_tree.nodes["Material Output"].inputs["Displacement"])
topo_mat.node_tree.links.new(color_image_node.outputs["Color"], \
                             topo_mat.node_tree.nodes["Principled BSDF"].inputs[0])

# Link camera to scene and set world
bpy.context.scene.collection.objects.link(cam_obj)                         
bpy.context.scene.camera = cam_obj    
bpy.context.scene.world = topo_world

# Configure render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = samples
bpy.context.scene.render.resolution_x = int(res_x)
bpy.context.scene.render.resolution_y = int(res_y)
bpy.context.scene.render.filepath = output
bpy.ops.render.render(write_still=True)