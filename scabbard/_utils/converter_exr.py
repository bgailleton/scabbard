import scabbard as scb
import numpy as np
import rasterio
import click
from scipy.ndimage import sobel
import matplotlib.pyplot as plt

def calculate_normal_map(elevation_data):
	'''
	Calculate the normal map (vector in 3D pointing to the )
	'''
	# Compute gradients using the Sobel operator
	gradient_x = sobel(elevation_data, axis=1)
	gradient_y = sobel(elevation_data, axis=0)

	# Calculate the negative gradients
	neg_gradient_x = -gradient_x
	neg_gradient_y = -gradient_y

	# Create an array for the normals
	normals = np.zeros((*elevation_data.shape, 3), dtype=np.float32)

	# Calculate the normal vectors
	length = np.sqrt(neg_gradient_x**2 + neg_gradient_y**2 + 1)
	normals[..., 0] = neg_gradient_x / length
	normals[..., 1] = neg_gradient_y / length
	normals[..., 2] = 1 / length

	# Map normals from [-1, 1] to [0, 1]
	normal_map = (normals + 1) / 2
	normal_map = normal_map.astype(np.float32)

	# plt.imshow(normal_map)
	# plt.show()

	return normal_map


def write_exr(array, nx:int, ny:int, filename:str, norm:bool = True):

	try:
		import OpenEXR
		import Imath
	except:
		raise ValueError("To write data to EXR files (image textures), you need the openEXR package: pip install openexr")



	# Convert the numpy array to an EXR image
	header = OpenEXR.Header(nx,ny)
	if('.exr' not in filename):
		filename+='.exr'
	exr = OpenEXR.OutputFile(filename, header)
	if (array.ndim == 2):
		if(norm):
			array-=array.min()
			array/=array.max()

		exr.writePixels({'R': array.tobytes(), 'G': array.tobytes(), 'B': array.tobytes()})
	elif (array.ndim == 3 and array.shape[2] == 3):
		exr.writePixels({'R': array[:,:,0].tobytes(), 'G': array[:,:,1].tobytes(), 'B': array[:,:,2].tobytes()})
	else:
		raise ValueError('exr can only save RBG data or single value data')

	exr.close()



@click.command()
@click.option("--normal", "-n", is_flag=True, show_default=True, default=False, help="Generate normal")
@click.argument('fname', type = str)
def cli_convert_to_EXR(fname, normal):
	'''
	Command line tool to convert a DEM to an EXR texture, possibly with a normal map
	
	Arguments:
		- fname: path+name of the dem to input
		- normal: if True, generate a normal map
	
	Return:
		- Nothing, writes 1 or 2 files

	Authors:
		- B.G. (last modifications: 04/2025) 
	'''

	prefix = '.'.join(fname.split('.')[:-1])

	dem = scb.io.load_raster(fname)

	write_exr(dem.Z, dem.geo.nx, dem.geo.ny, prefix+'.exr')
	if(normal):
		nmap = calculate_normal_map(dem.Z)
		write_exr(nmap, dem.geo.nx, dem.geo.ny, prefix+'_normal.exr')
