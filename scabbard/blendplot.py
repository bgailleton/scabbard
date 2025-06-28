from importlib import resources 
import numpy as np
import numba as nb
import scabbard as scb

import subprocess as sub

# Path to the Blender script used for rendering
with resources.open_text('scabbard', '_blandplot.py') as config_file:
	blender_script = config_file.name



@nb.njit()
def _fill_arrays(data,displacement_array,color_array, intensity = 0.7):
    """
    Numba-optimized internal function to fill displacement and color arrays for Blender.

    This function prepares the topographic data and color information into 1D arrays
    suitable for Blender's image pixel format. It adds a 1-pixel border around the
    topography and handles NaN values.

    Args:
        data (numpy.ndarray): 2D NumPy array of topographic elevation data.
        displacement_array (numpy.ndarray): 1D NumPy array to store displacement values (modified in-place).
        color_array (numpy.ndarray): 1D NumPy array to store color values (modified in-place).
        intensity (float, optional): Luminosity intensity for the color array. Defaults to 0.7.

    Returns:
        None: `displacement_array` and `color_array` are modified in-place.

    Author: B.G.
    """
    k=0
    for j in range(0,data.shape[1]+2):
        for i in range(0,data.shape[0]+2):
            i_data = i - 1
            j_data = j - 1

            i_minus = max(i_data-1,0)
            i_plus = min(i_data+1,data.shape[0]-1)
            j_minus = max(j_data-1,0)
            j_plus = min(j_data+1,data.shape[1]-1)
            
            i_data = min(max(i_data,0),data.shape[0]-1)
            j_data = min(max(j_data,0),data.shape[1]-1)
            
            if i >= 1 and j >= 1 and i <= data.shape[0] and j <= data.shape[1]:
                if np.isnan(data[i_data,j_data]):
                    displacement_array[k:k+4] = np.nan
                    color_array[k:k+4] = 1.0
                else:
                    displacement_array[k:k+4] = data[i_data,j_data]
                    # Get the RGBA color for the specified value
    #                rgba_color = np.array(sm.to_rgba(data[i_data,j_data]))
    #                rgba_color *= luminosity_scale
                    
                    color_array[k:k+4] = data[i_data,j_data] * intensity
                    # color_array[k:k+3] *= luminosity_scale
    #                color_array[k:k+4] = rgba_color
            else:
                displacement_array[k:k+4] = np.nan
                color_array[k:k+4] = 1.0
            k+=4
            
@nb.njit()
def _fill_arrays_cc(data, colors,displacement_array,color_array, intensity = 0.7):
    """
    Numba-optimized internal function to fill displacement and color arrays for Blender with custom colors.

    This function is similar to `_fill_arrays` but uses a pre-defined `colors` array
    for the color information instead of deriving it from the data itself.

    Args:
        data (numpy.ndarray): 2D NumPy array of topographic elevation data.
        colors (numpy.ndarray): 2D NumPy array of custom color values.
        displacement_array (numpy.ndarray): 1D NumPy array to store displacement values (modified in-place).
        color_array (numpy.ndarray): 1D NumPy array to store color values (modified in-place).
        intensity (float, optional): Luminosity intensity for the color array. Defaults to 0.7.

    Returns:
        None: `displacement_array` and `color_array` are modified in-place.

    Author: B.G.
    """
    k=0
    for j in range(0,data.shape[1]+2):
        for i in range(0,data.shape[0]+2):
            i_data = i - 1
            j_data = j - 1

            i_minus = max(i_data-1,0)
            i_plus = min(i_data+1,data.shape[0]-1)
            j_minus = max(j_data-1,0)
            j_plus = min(j_data+1,data.shape[1]-1)
            
            i_data = min(max(i_data,0),data.shape[0]-1)
            j_data = min(max(j_data,0),data.shape[1]-1)
            
            if i >= 1 and j >= 1 and i <= data.shape[0] and j <= data.shape[1]:
                if np.isnan(data[i_data,j_data]):
                    displacement_array[k:k+4] = np.nan
                    color_array[k:k+4] = 1.0
                else:
                    displacement_array[k:k+4] = data[i_data,j_data]
                    # Get the RGBA color for the specified value
    #                rgba_color = np.array(sm.to_rgba(data[i_data,j_data]))
    #                rgba_color *= luminosity_scale
                    
                    color_array[k:k+4] = colors[i_data,j_data] 
                    # color_array[k:k+3] *= luminosity_scale
    #                color_array[k:k+4] = rgba_color
            else:
                displacement_array[k:k+4] = np.nan
                color_array[k:k+4] = 1.0
            k+=4
            



def plot_blender_from_array(_data, 
	fprefix = "__blend",
	save_prefix = "render",
	dx = 5.,
	gpu = True,
	perspective = True,
	ortho_scale = 2.,
	focal_length = 40,
	f_stop = 50000,
	shiftx = 0.,
	shifty = 0.,
	camera_tilt =  45.0,
	camera_rotation =  0.,
	sun_tilt =  25.0,
	sun_rotation =  315.0,
	sun_intensity =  0.2,
	exaggeration =  1.5,
	recast_minZ =  0.,
	recast_maxZ =  1.,
	number_of_subdivisions = 1000,
	res_x = 1920,
	res_y = 1080,
	samples = 50,
	intensity = 0.7,
	custom_colors = None

	):
	"""
	Generates a 3D render of a topographic array using Blender.

	This function takes a 2D NumPy array representing topography, prepares it for Blender,
	and then executes a Blender script to render a 3D visualization. It handles data
	normalization, color mapping, and various camera and lighting settings.

	Args:
		_data (numpy.ndarray): The 2D NumPy array of topographic elevation data.
		fprefix (str, optional): File prefix for intermediate NumPy files used by Blender.
							Defaults to "__blend".
		save_prefix (str, optional): File prefix for the output rendered image (PNG).
							Defaults to "render".
		dx (float, optional): Spatial resolution (grid cell size) in meters. Defaults to 5.0.
		gpu (bool, optional): If True, enables GPU acceleration for Blender rendering. Defaults to True.
		perspective (bool, optional): If True, sets the Blender camera to perspective mode;
								otherwise, sets it to orthogonal mode. Defaults to True.
		ortho_scale (float, optional): Orthogonal scale for the camera (when in orthogonal mode).
								Higher values "zoom" out. Defaults to 2.0.
		focal_length (float, optional): Focal length of the camera in mm (when in perspective mode).
								Higher values "zoom" in. Defaults to 40.
		f_stop (float, optional): F-stop value for depth of field. Lower values result in a
								shallower depth of field, higher values for a wider depth of field.
								Defaults to 50000.
		shiftx (float, optional): Camera shift in the X-direction to center the topography.
							Defaults to 0.0.
		shifty (float, optional): Camera shift in the Y-direction to center the topography.
							Defaults to 0.0.
		camera_tilt (float, optional): Camera tilt in degrees from horizontal. Defaults to 45.0.
		camera_rotation (float, optional): Camera rotation in degrees clockwise from North. Defaults to 0.0.
		sun_tilt (float, optional): Sun tilt in degrees from horizontal. Defaults to 25.0.
		sun_rotation (float, optional): Sun rotation in degrees clockwise from North. Defaults to 315.0.
		sun_intensity (float, optional): Intensity of the sun light. Defaults to 0.2.
		exaggeration (float, optional): Vertical exaggeration factor for the topography. Defaults to 1.5.
		recast_minZ (float, optional): Minimum Z-value for recasting (normalization). Defaults to 0.0.
		recast_maxZ (float, optional): Maximum Z-value for recasting (normalization). Defaults to 1.0.
		number_of_subdivisions (int, optional): Number of subdivisions for the mesh. More subdivisions
								increase detail but also rendering time. Defaults to 1000.
		res_x (int, optional): X-resolution of the rendered image. Defaults to 1920.
		res_y (int, optional): Y-resolution of the rendered image. Defaults to 1080.
		samples (int, optional): Number of rendering samples. More samples result in better quality
							but longer render times. Defaults to 50.
		intensity (float, optional): Luminosity intensity for the color array. Defaults to 0.7.
		custom_colors (numpy.ndarray, optional): Optional 2D NumPy array for custom color mapping.
								If provided, `intensity` is ignored for color. Defaults to None.

	Returns:
		None: The function executes Blender as a subprocess to generate the render.

	Author: B.G.
	"""
	
	data = np.copy(_data)

	# Create empty arrays to hold displacement and color data for Blender
	displacement_array = np.zeros(((data.shape[0]+2)*(data.shape[1]+2)*4), dtype=np.float32)
	color_array = np.zeros(((data.shape[0]+2)*(data.shape[1]+2)*4), dtype=np.float32)

	# Find min/max elevation and normalize data
	data_min = np.min(data[data!=-9999])
	data_max = np.max(data[data!=-9999])
	relief = data_max - data_min
	data[data!=-9999] = (data[data!=-9999] - data_min) / relief
	# Recast values based on specified min/max Z for visualization
	data[data!=-9999] = (data[data!=-9999] - recast_minZ) / (recast_maxZ - recast_minZ)

	# Fill the displacement and color arrays using Numba-optimized helper functions
	_fill_arrays(data,displacement_array,color_array, intensity) if custom_colors is None else _fill_arrays_cc(data, custom_colors, displacement_array, color_array, intensity)



	# Save intermediate NumPy arrays for Blender to load
	np.save(fprefix + ".npy", data.astype(np.float32))
	np.save(fprefix + "_displacement_array.npy", displacement_array)
	np.save(fprefix + "_color_array.npy", color_array)


	# Construct the Blender command with all arguments
	cmd = ""
	cmd += scb.config.query('blender') # Get Blender executable path from scabbard config
	cmd += "  --background --python " + blender_script + " " # Run Blender in background with the script
	cmd += "--gpu " if gpu else "" # Add GPU flag if enabled
	cmd += "--perspective " if perspective else "" # Add perspective flag if enabled
	cmd += f"--fprefix '{fprefix}' "
	cmd += f"--save_prefix '{save_prefix}' "
	cmd += f"--ortho_scale {ortho_scale} "
	cmd += f"--focal_length {focal_length} "
	cmd += f"--f_stop {f_stop} "
	cmd += f"--dx {dx} "
	cmd += f"--shiftx {shiftx} "
	cmd += f"--shifty {shifty} "
	cmd += f"--camera_tilt {camera_tilt} "
	cmd += f"--camera_rotation {camera_rotation} "
	cmd += f"--sun_tilt {sun_tilt} "
	cmd += f"--sun_rotation {sun_rotation} "
	cmd += f"--sun_intensity {sun_intensity} "
	cmd += f"--exaggeration {exaggeration} "
	cmd += f"--recast_minZ {recast_minZ} "
	cmd += f"--recast_maxZ {recast_maxZ} "
	cmd += f"--number_of_subdivisions {number_of_subdivisions} "
	cmd += f"--res_x {res_x} "
	cmd += f"--res_y {res_y} "
	cmd += f"--samples {samples} "
	cmd += f"--relief {relief} "



	# Execute the Blender command as a subprocess
	sub.run(cmd, shell = True, check = False)
