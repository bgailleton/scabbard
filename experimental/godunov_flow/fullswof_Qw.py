'''
testing a cpu adaptation of FullSWOF-2D 1.10 with selected components (manning, hydrostat. reconst., )
'''

import numpy as np
import matplotlib.pyplot as plt
import scabbard as scb
import numba as nb
import math as m


# Helper for MPM
def calculate_MPM_from_D(D, l_transp, rho_water = 1000, gravity = 9.8, rho_sediment=2600, theta_c = 0.047):
	R = rho_sediment/rho_water - 1
	tau_c = (rho_sediment - rho_water) * gravity  * D * theta_c
	E_MPM = 8/(rho_water**0.5 * (rho_sediment - rho_water) * gravity)
	k_erosion = E_MPM/l_transp
	return k_erosion, tau_c

# Helper for MPM
def calculate_E_tau_c_from_D(D, rho_water = 1000, gravity = 9.8, rho_sediment=2600, theta_c = 0.047):
	R = rho_sediment/rho_water - 1
	tau_c = (rho_sediment - rho_water) * gravity  * D * theta_c
	E_MPM = 8/(rho_water**0.5 * (rho_sediment - rho_water) * gravity)
	k_erosion = E_MPM
	return E_MPM, tau_c


# Constants
HE_CA       = 1e-6
VE_CA       = 1e-6
GRAV        = 9.81
GRAV_DEM    = 4.905
CONST_CFL_X = 0.5
CONST_CFL_Y = 0.5
HE_CA       = 1.e-12
VE_CA       = 1.e-12
MAX_CFL_X   = 0.
MAX_CFL_Y   = 0.
MAX_ITER    = 1000000000
ZERO        = 0.
IE_CA       = 1.e-8
EPSILON     = 1.e-13
RHOW        = 1000.

@nb.njit()
def hydrorec(hxr, hxl, hxright, hxleft, delzx, ny, nx, dirx = True):
	'''
	Compute the Hydrostatic reconstruction at boundaries between two nodes
	+ Best reconstruction method to accomodate for variable topography and mitigate oscillations
	- Less accurate than MUSCL or ENO - but I do not think we need that level of precision for our cases
	Adapted from hydrostatic.cpp of FullSWOF2D-1.10.
	
	function located at the staggered grid representing the boundaries

	Arguments:
		- hxr:     flow depth at the right of the boundary
		- hxl:     flow depth at the left of the boundary
		- hxright: reconstructed flow depth at the right of the boundary
		- hxleft:  reconstructed flow depth at the left of the boundary
		- delzx:   topographic gradient at the boundary
		- ny,nx:   number of nodes in the y and x direction
		- dirx:    Are we in the x direction or the y direction
	
	'''
	# in the x direction
	if(dirx):
		# For all nodes
		for i in range(ny):
			for j in range(1,nx):
				# getting the left hw (which is the right of the previous node)
				hl = hxr[i,j-1]
				# getting the right hw (which is the left of the current node)
				hr = hxl[i,j]
				# topogradient is +/- gradient between this cell and the previous one (so delz[node-1] in the right x/y dir)
				dz = delzx[i,j-1]
				# The reconstructed left
				hl_rec = max(0.,hl-max(0.,dz));
				# The reconstructed right (not 100% sure about the theory lol)
				hr_rec = max(0.,hr-max(0.,-dz));
				hxright[i,j-1] = hl_rec
				hxleft[i,j] = hr_rec

	else:
		for i in range(1,ny):
			for j in range(nx):
				hl = hxr[i-1,j]
				hr = hxl[i,j]
				dz = delzx[i-1,j]
				hl_rec = max(0.,hl-max(0.,dz));
				hr_rec = max(0.,hr-max(0.,-dz));
				hxright[i-1,j] = hl_rec
				hxleft[i,j] = hr_rec


@nb.njit()
def _flux_calc_hll(h_L, u_L, v_L, h_R, u_R, v_R):
	'''
	Internal calculation of flux (HLL -> Harten-Lax-van Lee, approximate Riemann solver)
	'''
	f1 = 0.
	f2 = 0.
	f3 = 0.
	cfl = 0.
	if (h_L < HE_CA and h_R<HE_CA):
		f1 = 0.
		f2 = 0.
		f3 = 0.
		cfl = 0.

	else:
		# local_variables
		grav_h_L = GRAV*h_L
		grav_h_R = GRAV*h_R
		q_R = u_R*h_R
		q_L = u_L*h_L
		c1 = min(u_L-m.sqrt(grav_h_L),u_R-m.sqrt(grav_h_R))
		c2 = max(u_L+m.sqrt(grav_h_L),u_R+m.sqrt(grav_h_R))
		
		# cfl is the velocity to calculate the real cfl=max(abs(c1),abs(c2))*tx with tx=dt/dx
		if (abs(c1)<EPSILON and abs(c2)<EPSILON):             #dry state
			f1=0.
			f2=0.
			f3=0.
			cfl=0. # max(abs(c1),abs(c2))=0
		
		elif (c1>=EPSILON): # supercritical flow, from left to right : we have max(abs(c1),abs(c2))=c2>0
			f1=q_L
			f2=q_L*u_L+GRAV*h_L*h_L*0.5
			f3=q_L*v_L
			cfl=c2 # max(abs(c1),abs(c2))=c2>0
		
		elif (c2<=-EPSILON): # supercritical flow, from right to left : we have max(abs(c1),abs(c2))=-c1>0
			f1=q_R
			f2=q_R*u_R+GRAV*h_R*h_R*0.5
			f3=q_R*v_R
			cfl=abs(c1) # max(abs(c1),abs(c2))=abs(c1)

		else: # subcritical flow
			tmp = 1./(c2-c1)
			f1=(c2*q_L-c1*q_R)*tmp+c1*c2*(h_R-h_L)*tmp
			f2=(c2*(q_L*u_L+GRAV*h_L*h_L*0.5)-c1*(q_R*u_R+GRAV*h_R*h_R*0.5))*tmp+c1*c2*(q_R-q_L)*tmp
			f3=(c2*(q_L*v_L)-c1*(q_R*v_R))*tmp+c1*c2*(h_R*v_R-h_L*v_L)*tmp
			cfl=max(abs(c1),abs(c2))

	return f1, f2, f3, cfl

@nb.njit()
def flux_calc_hll(fx1, fx2, fx3, hxleft, hxright, uxl, uxr, vxl, vxr, ny, nx, dirx ):
	'''
	Wrap-up function for flux calculation with HLL (-> Harten-Lax-van Lee, approximate Riemann solver) in a single direction
	Remember that the values are the reconstructed one at the left and right of each boundary, so the "left" one in the referential of the cell is the "right" one for the boundary
	
	You'll note that in the Y direction, fluxes G and H (treating u and v) are inverted -> makes sense as v and u are perpendicular
	(Seems obvious but not if you're not used to this kind of equaitons/NS)
	
	Arguments:
		- fx1,fx2,fx3: the 3 components of the fluxes (See Delestre et al., 2013 - equation 7, fluxes U G H if I get it correctly)
		- hxr:         flow depth at the right of the boundary
		- hxl:         flow depth at the left of the boundary
		- hxright:     reconstructed flow depth at the right of the boundary
		- hxleft:      reconstructed flow depth at the left of the boundary
		- delzx:       topographic gradient at the boundary
		- ny,nx:       number of nodes in the y and x direction
		- dirx:        Are we in the x direction or the y direction

	'''
	cfl = -1
	if(dirx):
		for i in range(ny):
			for j in range(1,nx):
				fx1[i,j],fx2[i,j],fx3[i,j], cfl = _flux_calc_hll(hxright[i,j-1], uxr[i,j-1], vxr[i,j-1], hxleft[i,j], uxl[i,j], vxl[i,j])
	else:
		for i in range(1,ny):
			for j in range(nx):
				fx1[i,j],fx3[i,j],fx2[i,j], cfl = _flux_calc_hll(hxright[i-1,j], vxr[i-1,j], uxr[i-1,j], hxleft[i,j], vxl[i,j], uxl[i,j])

	return cfl


@nb.njit()
def main_comp(h, hs, fx1, fy1, Rain, tx, ty, dt, ny, nx):
	'''
	Continuity equation: updates h

	Arguments:
		- h:       flow depth
		- hs:      flow depth at t+1
		- fx1,fy1: flux U in the x and y direction
		- Rain:    precipitation rates
		- tx,ty:   transfer time factor (=dt/dx) 
		- dt:      time step
		- ny,nx:   dimension of the grid
	'''
	for i in range(ny-1):
		for j in range(nx-1):

			# Solution of the equation of mass conservation (First equation of Shallow-Water)
			hs[i,j] = h[i,j]-tx*(fx1[i,j+1]-fx1[i,j])-ty*(fy1[i+1,j]-fy1[i,j])+Rain[i,j]*dt;

			if(hs[i,j]<0):
				hs[i,j] = 0.

@nb.njit()
def mome_comp(qx, qy, h, u, v, fx2, fx3, fy2, fy3, hxl, hxleft, hxr, hxright, hyl, hyleft, hyr, hyright, delzcx, delzcy, GRAV_DEM):
	'''
	Momentum equation (updates q)

	Arguments:
		- fx2, fx3, fy2, fy3: the 2 components of the velocity fluxes for each directions (See Delestre et al., 2013 - equation 7, fluxes U G H if I get it correctly)
		- hxr:                flow depth at the right of the boundary
		- hxl:                flow depth at the left of the boundary
		- hxright:            reconstructed flow depth at the right of the boundary
		- hxleft:             reconstructed flow depth at the left of the boundary
		- delzx:              topographic gradient at the boundary
		- ny,nx:              number of nodes in the y and x direction
		- ...y:               equivalent fluxes in the y direction
		- delzcx/delzcy:      Ignore. These fluxes are not used in the scheme I use so far (2nd order schemes)

	'''

	for i in range(ny-1):
		for j in range(nx-1):
			# comments for fullswof
			# Solution of the equation of momentum (Second and third equation of Shallow-Water)
			# This expression for the flux (instead of the differences of the squares) avoids numerical errors
			# see http://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html section "Cancellation".

			qx[i,j] = (h[i,j]*u[i,j]-tx*(fx2[i,j+1]-fx2[i,j]+GRAV_DEM*((hxleft[i,j]-hxl[i,j])*(hxleft[i,j]+hxl[i,j])+(hxr[i,j]-hxright[i,j])*(hxr[i,j]+hxright[i,j])+(hxl[i,j]+hxr[i,j])*delzcx[i,j]))-ty*(fy2[i+1,j]-fy2[i,j]));
			qy[i,j] = (h[i,j]*v[i,j]-tx*(fx3[i,j+1]-fx3[i,j])-ty*(fy3[i+1,j]-fy3[i,j]+GRAV_DEM*((hyleft[i,j]-hyl[i,j])*(hyleft[i,j]+hyl[i,j])+(hyr[i,j]-hyright[i,j])*(hyr[i,j]+hyright[i,j])+(hyl[i,j]+hyr[i,j])*delzcy[i,j])));

@nb.njit()
def fric_calc(u, v, hs, qx, qy, dt, fric_coeff, GRAV):
	'''
	Using Manning's friction from fullSWOF
	'''
	for i in range(ny):
		for j in range(nx):
			if(hs[i,j] > 0):
				qx[i,j] = qx[i,j] / (1.+fric_coeff*fric_coeff*GRAV*m.sqrt(u[i,j]*u[i,j]+v[i,j]*v[i,j])*dt / m.pow(hs[i,j],4./3.) );
				qy[i,j] = qy[i,j] / (1.+fric_coeff*fric_coeff*GRAV*m.sqrt(u[i,j]*u[i,j]+v[i,j]*v[i,j])*dt / m.pow(hs[i,j],4./3.) );

@nb.njit()
def morpho_step(h, z, Sfx, Sfy, qsx, qsy, dt, GRAV, RHOW, Ke, TL, tau_c):

	# z_p1 = z.copy()
	qsin = np.zeros_like(Sfx)
	qsout = np.zeros_like(Sfx)

	for i in range(1,ny):
		for j in range(1,nx):

			taux = h[i,j] * Sfx[i,j] * RHOW * GRAV
			sign = 1 if taux > 0 else -1
			taux = abs(taux)
			qsx[i,j] = Ke * max(0., taux - tau_c)**(1.5)
			qsout[i,j] += qsx[i,j]
			if sign>0:
				qsin[i,j+1] += qsx[i,j]
			else:
				qsin[i,j-1] += qsx[i,j]
			qsx[i,j] *= sign


			if(np.isfinite(taux) == False):
				print(taux, Sfx[i,j], h[i,j])
			
			tauy = h[i,j] * Sfy[i,j] * RHOW * GRAV
			sign = 1 if tauy > 0 else -1
			tauy = abs(tauy)
			qsy[i,j] = Ke * max(0., tauy - tau_c)**(1.5)
			qsout[i,j] += qsy[i,j]
			if sign>0:
				qsin[i-1,j] += qsy[i,j]
			else:
				qsin[i+1,j] += qsy[i,j]
			qsy[i,j] *= sign
						
	for i in range(ny-1):
		for j in range(nx-1):
			dz = (qsin[i,j] - qsout[i,j])*dt/dx
			# if( abs(dz) > 0):
			# 	print("dz::",dz)
			z[i,j] -= dz
			# h[i,j] -= dz
			if(h[i,j] < 0.):
				h[i,j] = 0.


# @nb.njit()
# def morpho_step_v2(h, z, u, v, qsx, qsy, qx, qy, dt, GRAV, RHOW, Ke, TL, tau_c):


# 	for i in range(1,ny):
# 		for j in range(1,nx):

# 			tau = RHOW * h[i,j] * 
			
						
# 	for i in range(ny-1):
# 		for j in range(nx-1):
# 			dz = (qsx[i,j+1] - qsx[i,j] + qsy[i+1,j] - qsy[i,j])*dt/dx
# 			# if( abs(dz) > 0):
# 			# 	print("dz::",dz)
# 			z[i,j] -= dz
# 			# h[i,j] -= dz
# 			if(h[i,j] < 0.):
# 				h[i,j] = 0.



@nb.njit()
def Sf_manning(h, u, v, manning, Sfx, Sfy, HE_CA):
	# //Sf = Manning = v|v|/c^2*h^{4/3}
	# void Fr_Manning::calculSf(const TAB & h,const TAB &  u,const TAB &  v){
	  
	#   /**
	#    * @details
	#    * Explicit friction term: \f$ S_f=c^2\frac{U|U|}{h^{4/3}}\f$
	#    * where \f$c\f$ is the friction coefficient.
	#    * @param[in] h water height.
	#    * @param[in] u velocity in the first direction, first component of \f$ U \f$ in the above formula.
	#    * @param[in] v velocity in the second direction, second component of \f$ U \f$ in the above formula.
	#    * @par Modifies
	#    * Friction#Sfx explicit friction term in the first direction,\n
	#    * Friction#Sfy explicit friction term in the second direction.
	#    * @note This explicit friction term will be used for erosion.
	#    */
	
	Fricoeff = manning
	for i in range(2,ny):
		for j in range(2,nx):

			if (h[i,j] > HE_CA):
				# print('yolo')
				Sfx[i,j] = Fricoeff*Fricoeff*u[i,j]*m.sqrt(u[i,j]*u[i,j]+v[i,j]*v[i,j])/(m.pow(h[i,j],4./3.))
				Sfy[i,j] = Fricoeff*Fricoeff*v[i,j]*m.sqrt(u[i,j]*u[i,j]+v[i,j]*v[i,j])/(m.pow(h[i,j],4./3.))
				# print(Sfy[i,j], u[i,j], v[i,j])
			else:
				Sfx[i,j] = ZERO
				Sfy[i,j] = ZERO
			



ny,nx = 1024,128
dx = 2.
dy = dx
S0 = 1e-2
# Gridding
grid, tBCs = scb.raster.slope2D_S(nx = nx, ny = ny, dx = dx, z_base = 0, slope = S0)

# scb.io.save_raster(grid, 'input.asc', crs='EPSG:32601', dtype=np.float32, driver = 'AAIGrid')
# scb.io.save_raster(grid, 'input.tif', crs='EPSG:32601', dtype=np.float32, driver = 'GTiff')
scb.io.save_ascii_grid(grid, 'input.asc', dtype = np.float32)
# saving boundaries



Rain = np.zeros_like(grid.Z)

hin = 1.

# Unsused here
Qin  = 50 # m^3 p s
nQin = 10
midn = round(nx/2)
halfn = round(nQin/2)
loc  = np.arange(midn - halfn, midn + halfn)
# loc = np.arange(2, nx-3)

# grid.Z[:,loc] -= 2
# grid.Z[200:220,:] += 1

# morpho
D = 4e-3
TL = 100.
# Ke, tau_c = calculate_MPM_from_D(D, TL, rho_water = 1000, gravity = GRAV_DEM, rho_sediment=2600, theta_c = 0.047)
Ke, tau_c = calculate_E_tau_c_from_D(D, rho_water = 1000, gravity = 9.8, rho_sediment=2600, theta_c = 0.047)
fric_coeff = 0.033


# s
# In Fullswoth: 0 -> Nx/y + 1
z = grid.Z

# z[round(ny/3):round(2*ny/3),round(nx/3):round(2*nx/3)] += 50
# z[:,round(nx/3):round(2*nx/3)] -= 5

oz = z.copy()
# z[:,1] += 5
h  = np.zeros_like(z)
hs  = np.zeros_like(z)
u  = np.zeros_like(z) # en x
us  = np.zeros_like(z) # en x
v  = np.zeros_like(z) # en y
vs  = np.zeros_like(z) # en y

# Vectors
# in FS : nx/y so here the last col is useless
# They represent the links between 2 cells (in x-col and y-row)
## Discahrge per unit width
qx = np.zeros_like(z)
qy = np.zeros_like(z)
## Intermediary fluxes
fx1 = np.zeros_like(z)
fx2 = np.zeros_like(z)
fx3 = np.zeros_like(z)
fy1 = np.zeros_like(z)
fy2 = np.zeros_like(z)
fy3 = np.zeros_like(z)
## topo gradients
delzx = np.zeros_like(z)
delzy = np.zeros_like(z)
## reconstructed topo gradient for order 2
delzcx = np.zeros_like(z)
delzcy = np.zeros_like(z)

## half vector data: getting reconstructed data on each side of each links
## By definition, r is right and lest for x, and bottom/top for y 
hxr = np.zeros_like(z)
hxright = np.zeros_like(z)
uxr = np.zeros_like(z)
vxr = np.zeros_like(z)
hxl = np.zeros_like(z)
hxleft = np.zeros_like(z)
uxl = np.zeros_like(z)
vxl = np.zeros_like(z)
hyr = np.zeros_like(z)
hyright = np.zeros_like(z)
uyr = np.zeros_like(z)
vyr = np.zeros_like(z)
hyl = np.zeros_like(z)
hyleft = np.zeros_like(z)
uyl = np.zeros_like(z)
vyl = np.zeros_like(z)

# morpho
D = 4e-3
TL = 100.
# Ke, tau_c = calculate_MPM_from_D(D, TL, rho_water = 1000, gravity = GRAV_DEM, rho_sediment=2600, theta_c = 0.047)
Ke, tau_c = calculate_E_tau_c_from_D(D, rho_water = 1000, gravity = 9.8, rho_sediment=2600, theta_c = 0.047)
qsx = np.zeros_like(z)
qsy = np.zeros_like(z)
Sfx = np.zeros_like(z)
Sfy = np.zeros_like(z)

qsyin = Ke * (hin * S0 * GRAV * RHOW - tau_c)**(1.5)
print('qsin is nonchalently equal to', qsyin)



# Step 1: boundary conditions (To Do Later)
def boundary(z, h, u, v, loc, Qin, nQin, dx, qsyin, qsy):
	# tloc = loc[:1]
	# nQin = 1
	# tloc = loc
	# h[0,tloc] = Qin/(nQin * dx * dx)
	# h[-1,:] = 0.
	# u[0,tloc] = 0.
	# v[0,tloc] = Qin/(nQin * dx * h[0,tloc])
	# z[0, :] = z[0,0] 
	h[0,round(128/2)-5:round(128/2)+5] = hin
	qsy[0, round(128/2)-5:round(128/2)+5] = qsyin


# print(delzx)
# quit()

fig,axs = plt.subplots(1,2, figsize = (10,8))
ax = axs[0]
imh = ax.imshow(h, vmin = 0., vmax = 0.8, cmap = 'Blues')
plt.colorbar(imh, label = 'flow depth (m)')

ax1 = axs[1]
pzz = ax1.plot(z[:,round(nx/2)], color = 'k')
pzh = ax1.plot(z[:,round(nx/2)] + h[:,round(nx/2)], color = 'b')

ax1.set_ylabel('z (m)')
ax1.set_xlabel('x (m)')
ax1.set_ylim(0,8)

fig.show()

it = 0
time = 0
boundary(z, h, u, v, loc, Qin, nQin, dx, qsyin, qsy)
qx = u * h
qy = v * h

# # Saving to go on the calib'
# grid.Z = h
# scb.io.save_ascii_grid(grid, 'h.asc', dtype = np.float32)
# grid.Z = u
# scb.io.save_ascii_grid(grid, 'u.asc', dtype = np.float32)
# grid.Z = v
# scb.io.save_ascii_grid(grid, 'v.asc', dtype = np.float32)
# quit()


# #MAINCALCFLUX
cflfix= 0.5 # TBD
T = 0. # TBD
dt_max = 10 # TBD
dt = 0.01 # TBD
velocity_max_x = 5000. # TBD
velocity_max_y = 5000.	# TBDfullswof.


itfig = 0.
import time

st = time.time()

while(True):
# while(T < 600):
	it+=1
	# print("Starting", it)
	# print(T)

	# Step 2: precalculate delz (last col/row is 0)
	delzx[:,:-1] = z[:,1:] - z[:,:-1] # x
	delzy[:-1,:] = z[1:,:] - z[:-1,:] # y
	# print(delzy)

	# Deal with extreme values
	mask = h<=HE_CA
	h[mask] = 0.
	u[mask] = 0.
	v[mask] = 0.
	qx[mask] = 0.
	qy[mask] = 0.
	mask = np.abs(u)<=VE_CA
	u[mask] = 0.
	qx[mask] = 0.
	mask = np.abs(v)<=VE_CA
	v[mask] = 0.
	qy[mask] = 0.
	

	# Intermediate arrays
	# the left and right value of h for flux computation for each links
	## en xl
	# hxl[:,:-1] = h[:,:-1]
	# hxl[:,[-1]] = 0.
	# uxl[:,:-1] = u[:,:-1]
	# uxl[:,[-1]] = 0.
	# vxl[:,:-1] = v[:,:-1]
	# vxl[:,[-1]] = 0.

	# ## en xr
	# hxr[:,:-1] = h[:,1:]
	# hxr[:,-1] = 0.
	# uxr[:,:-1] = u[:,1:]
	# uxr[:,-1] = 0.
	# vxr[:,:-1] = v[:,1:]
	# vxr[:,-1] = 0.

	# #en yl (top)
	# hyl[:-1,:] = h[:-1,:]
	# hyl[-1,:] = 0.
	# uyl[:-1,:] = u[:-1,:]
	# uyl[-1,:] = 0.
	# vyl[:-1,:] = v[:-1,:]
	# vyl[-1,:] = 0.

	# ## en yr (bottom)
	# hyr[:-1,:] = h[1:,:]
	# hyr[-1,:] = 0.
	# uyr[:-1,:] = u[1:,:]
	# uyr[-1,:] = 0.
	# vyr[:-1,:] = v[1:,:]
	# vyr[-1,:] = 0.

	hxl = np.copy(h)
	uxl = np.copy(u)
	vxl = np.copy(v)
	hxr = np.copy(h)
	uxr = np.copy(u)
	vxr = np.copy(v)
	hyl = np.copy(h)
	uyl = np.copy(u)
	vyl = np.copy(v)
	hyr = np.copy(h)
	uyr = np.copy(u)
	vyr = np.copy(v)

	hydrorec(hxr, hxl, hxright, hxleft, delzx, ny, nx, True)
	cflx = flux_calc_hll(fx1, fx2, fx3, hxleft, hxright, uxl, uxr, vxl, vxr, ny, nx, True)
	hydrorec(hyr, hyl, hyright, hyleft, delzy, ny, nx, False)
	cfly = flux_calc_hll(fy1, fy2, fy3, hyleft, hyright, uyl, uyr, vyl, vyr, ny, nx, False)
	# print('yolo')
	# quit()
	# dtx = dt_max
	# dty = dt_max
	# dt_tmp = 0.

	# if(abs(cflx*dt/dx) < EPSILON):
	# 	dt_tmp = dt_max
	# else:
	# 	dt_tmp = cflfix*dx/cflx

	# dtx = min(min(dt,dt_tmp),dtx);

	# dtx = dt_max
	# dty = dt_max
	

	# dt_tmp = 0.

	# if(abs(cflx*dt/dx) < EPSILON):
	# 	dt_tmp = dt_max
	# else:
	# 	dt_tmp = cflfix*dx/cflx
		
	# dtx = min(min(dt,dt_tmp),dtx);
	# velocity_max_x = max(velocity_max_x, cflx);

	# dt_tmp = 0.

	# if(abs(cfly*dt/dx) < EPSILON):
	# 	dt_tmp = dt_max
	# else:
	# 	dt_tmp = cflfix*dx/cfly
		
	# dty = min(min(dt,dt_tmp),dty);
	# velocity_max_x = max(velocity_max_y, cfly);
	# dt=min(dtx,dty);

	tx=dt/dx;
	ty=dt/dy;

	T += dt

	# MAINCALCSCHEME
	# void Scheme:: maincalcscheme(TAB & he, TAB & ve1, TAB & ve2, TAB & qe1, TAB & qe2, TAB & hes, TAB & ves1, TAB & ves2, TAB & qes1, TAB & qes2,TAB & Vin,  curtime,  dt, int n){

	# Periodic boundaries management
	# add if periodic

	# /*-------------- Rainfall and infiltration --------------------------------------------------------------*/

	# Do stuff with rain here? seems that the code only update/simulate precipitation rates here

	# /*-------------- Main computation ------------------------------------------------------------------------*/
	# Updates h
	main_comp(h, hs, fx1, fy1, Rain, tx, ty, dt, ny, nx)


	# //Infiltration
	# Here lies the code dealing with infiltration (not interested for now)

	# Momentum calculation
	mome_comp(qx, qy, h, u, v, fx2, fx3, fy2, fy3, hxl, hxleft, hxr, hxright, hyl, hyleft, hyr, hyright, delzcx, delzcy, GRAV_DEM)

	# friction
	fric_calc(u, v, hs, qx, qy, dt, fric_coeff, GRAV)


	mask = hs>1e-6
	u[mask]  = qx[mask] / hs[mask]
	v[mask]  = qy[mask] / hs[mask]
	qx[mask] = u[mask]  * hs[mask]
	qy[mask] = v[mask]  * hs[mask]
	u[~mask]  = 0.
	v[~mask]  = 0.
	qx[~mask] = 0.
	qy[~mask] = 0.

	h = hs.copy()

	# if it%2 == 0:
	# 	Sf_manning(h, u, v, fric_coeff, Sfx, Sfy, HE_CA)
	# 	morpho_step(h, z, Sfx, Sfy, qsx, qsy, dt, GRAV, RHOW, Ke, TL, tau_c)
	# quit() if it == 20 else 0

	# u[mask] = 0
	# v[mask] = 0
	boundary(z, h, u, v, loc, Qin, nQin, dx, qsyin, qsy)
	# print("Done", it)


	if(it % 10000 == 0):
		ax.set_title('time ='+str(T)+'s')
		pzz[0].set_ydata(z[:,round(nx/2)])
		pzh[0].set_ydata(z[:,round(nx/2)] + h[:,round(nx/2)])
		imh.set_data(h)
		# imh.set_data(np.abs(Sfy))
		# imh.set_clim(np.abs(Sfy).min(), np.abs(Sfy).max())

		# imh.set_data(np.abs(oz - z))
		# imh.set_clim(np.abs(oz - z).min(), np.abs(oz - z).max())

		fig.canvas.draw_idle()
		fig.canvas.start_event_loop(0.01)
		print('took', time.time() - st)
		st = time.time()

		# itfig+=1
		# name = str(itfig)
		# while(len(name) <5):
		# 	name = '0' + name
		# plt.savefig('tempfig/'+ name+'.png', dpi = 300)
# np.save('final_h_calib.py', h)
# fig,ax = plt.subplots()
# cb = ax.imshow(h, cmap = 'jet', vmin =0, vmax = 1.)
# # cb = ax.imshow(fy1, cmap = 'jet', vmin =-1, vmax = 1.)
# plt.show()