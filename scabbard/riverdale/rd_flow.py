import taichi as ti
from scabbard.riverdale.rd_grid import GRID
import scabbard.riverdale.rd_grid as gridfuncs
import scabbard.riverdale.rd_helper_surfw as hsw


"""
This module contains Taichi kernels for computing flow directions (D4) based on topography (Z) and water depth (hw).

Author: B.G.
"""

@ti.kernel
def compute_D4(Z:ti.template(), D4dir:ti.template(), BCs:ti.template()):
	"""
	Computes the D4 (four-direction) flow direction for each grid cell based on topography.

	This function is experimental and should not be used in production at this time.

	Args:
		Z (ti.template()): Taichi field representing the topography.
		D4dir (ti.template()): Taichi field to store the computed D4 flow directions.
		BCs (ti.template()): Taichi field representing boundary conditions.

	Author: B.G.
	"""

	# Iterate over each grid cell
	for i,j in Z:

		# Initialize D4dir to a default value (5 indicates no flow direction yet)
		D4dir[i,j] = ti.uint8(5)

		# Skip cells that cannot contribute to flow or are inactive
		if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Initialize steepest slope to 0
		SS = 0.
		checked = True # This variable seems unused, consider removing if not needed

	
		# Iterate over the four direct neighbors (D4)
		for k in range(4):
			# Get neighbor coordinates (ir, jr) based on standard (see rd_grid header)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# If ir is -1, it means there is no valid neighbor in this direction
			if(ir == -1):
				continue

			# Calculate local hydraulic slope
			tS = Z[i,j] - Z[ir,jr]
			tS /= GRID.dx

			# If slope is non-positive, the neighbor is a donor or at the same elevation, so skip
			if(tS <= 0):
				continue

			# Register the steepest slope and corresponding direction
			if(tS > SS):
				D4dir[i,j] = ti.uint8(k)
				SS = tS

@ti.kernel
def compute_D4_Zw(Z:ti.template(), hw:ti.template(), D4dir:ti.template(), BCs:ti.template()):
	"""
	Computes the D4 (four-direction) flow direction for each grid cell based on water surface elevation (Z + hw).

	This function is experimental and should not be used in production at this time.

	Args:
		Z (ti.template()): Taichi field representing the topography.
		hw (ti.template()): Taichi field representing the water depth.
		D4dir (ti.template()): Taichi field to store the computed D4 flow directions.
		BCs (ti.template()): Taichi field representing boundary conditions.

	Author: B.G.
	"""

	# Iterate over each grid cell
	for i,j in Z:

		# Initialize D4dir to a default value (5 indicates no flow direction yet)
		D4dir[i,j] = ti.uint8(5)

		# Skip cells that cannot contribute to flow or are inactive
		if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Initialize steepest slope to 0
		SS = 0.
		checked = True # This variable seems unused, consider removing if not needed

	
		# Iterate over the four direct neighbors (D4)
		for k in range(4):
			# Get neighbor coordinates (ir, jr) based on standard (see rd_grid header)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# If ir is -1, it means there is no valid neighbor in this direction
			if(ir == -1):
				continue

			# Calculate local hydraulic slope based on water surface elevation (Z + hw)
			tS = hsw.Zw(Z,hw,i,j) - hsw.Zw(Z,hw,ir,jr)
			tS /= GRID.dx

			# If slope is non-positive, the neighbor is a donor or at the same elevation, so skip
			if(tS <= 0):
				continue

			# Register the steepest slope and corresponding direction
			if(tS > SS):
				D4dir[i,j] = ti.uint8(k)
				SS = tS