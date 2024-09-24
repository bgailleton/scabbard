import taichi as ti
from scabbard.riverdale.rd_grid import GRID
import scabbard.riverdale.rd_grid as gridfuncs
import scabbard.riverdale.rd_helper_surfw as hsw



@ti.kernel
def compute_D4(Z:ti.template(), D4dir:ti.template(), BCs:ti.template()):
	'''
	Experimental tests on drainage area calculations
	Do not use at the moment
	B.G.
	'''

	# Traversing each nodes
	for i,j in Z:

		D4dir[i,j] = ti.uint8(5)

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SS = 0.
		checked = True

	
		# Traversing Neighbours
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			# Local hydraulic slope
			tS = Z[i,j] - Z[ir,jr]
			tS /= GRID.dx

			# If < 0, neighbour is a donor and I am not interested
			if(tS <= 0):
				continue

			# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
			if(tS > SS):
				D4dir[i,j] = ti.uint8(k)
				SS = tS


@ti.kernel
def compute_D4_Zw(Z:ti.template(), hw:ti.template(), D4dir:ti.template(), BCs:ti.template()):
	'''
	Experimental tests on drainage area calculations
	Do not use at the moment
	B.G.
	'''

	# Traversing each nodes
	for i,j in Z:

		D4dir[i,j] = ti.uint8(5)

		# If the node cannot give and can only receive, I pass this node
		if(gridfuncs.can_give(i,j,BCs) == False or gridfuncs.is_active(i,j,BCs) == False):
			continue

		# Keeping in mind the steepest slope in the x and y direction to calculate the norm of the vector
		SS = 0.
		checked = True

	
		# Traversing Neighbours
		for k in range(4):
			# getting neighbour k (see rd_grid header lines for erxplanation on the standard)
			ir,jr = gridfuncs.neighbours(i,j,k, BCs)

			# if not a neighbours, by convention is < 0 and I pass
			if(ir == -1):
				continue

			# Local hydraulic slope
			tS = hsw.Zw(Z,hw,i,j) - hsw.Zw(Z,hw,ir,jr)
			tS /= GRID.dx

			# If < 0, neighbour is a donor and I am not interested
			if(tS <= 0):
				continue

			# Registering the steepest clope in both directions (see rd_grid header lines for erxplanation on the standard)
			if(tS > SS):
				D4dir[i,j] = ti.uint8(k)
				SS = tS