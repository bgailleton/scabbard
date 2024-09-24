
import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import scabbard as scb
import numpy as np
import math as m
import os

PATH2STEENBOK = os.path.join(os.path.dirname(__file__))

def concat_kernel_code(topology):
	# Step 1: gather the code
	# empty file
	kernel_code = ""
	# add the file in the right order (unfortunately required to get the macro first and all)
	files = [
	"includer.cu",
	"macros_holder.cu",
	"constants_holder.cu",
	"bc_helper.cu",
	"neighbourer.cu",
	"grid_tools.cu",
	"array_utils.cu",
	"graphflood_hydro.cu",
	# "graphflood_morpho_general.cu",
	# "graphflood_morpho_MPM.cu",
	# "graphflood_morpho_EROS.cu",
	# "gp_hydro.cu",
	# "gp_morpho.cu",
	]

	for file in files:
			with open(os.path.join(PATH2STEENBOK, file), 'r') as f:
					kernel_code += f.read() + '\n\n'

	# Could work, but files have interdependencies, more robust and stable to do manual imports
	# for file in os.listdir(PATH2STEENBOK):
	# 		if file.endswith(".cu"):
	# 				with open(os.path.join(PATH2STEENBOK, file), 'r') as f:
	# 						kernel_code += f.read() + '\n\n'

	kernel_code = kernel_code.replace("MACRO2SETNNEIGHBOURS", "8" if topology == "D8" else "4")


	return kernel_code

def debug_kernel(topology):
	kernel_code = concat_kernel_code(topology)

	with open("DEBUGKERNEL.cu", 'w') as f:
		f.write(kernel_code)

	#	Step 3: auto get all the functions
	mod = SourceModule(kernel_code)

def build_kernel(topology):

	
	kernel_code = concat_kernel_code(topology)
	
	# Step 2: define the Macros


	#	Step 3: auto get all the functions
	mod = SourceModule(kernel_code)
	functions = {}

	# Checking code line by line
	for line in kernel_code.splitlines():
		# split line by space
		tline = line.split(' ')

		# Checking if __global__ in the list
		globi = -1
		for index, item in enumerate(tline):
			if "__global__" in item:
					globi = index

		if globi == -1:
			continue

		tfunc = tline[globi+2].split('(')[0]
		print('Fetching', tfunc)
		try:
			functions[tfunc] = mod.get_function(tfunc)
		except:
			print("failed, no", tfunc, "found")
		print('OK')




	return mod, functions



	# add_Qwin_local = mod.get_function("add_Qwin_local")










