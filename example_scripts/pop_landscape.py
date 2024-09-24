import scabbard as scb
import dagger as dag
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import math

noise = scb.generate_noise_RGrid(nx = 8, ny = 8)


graph, con = noise.export_graphcon()
pop = dag.popscape(graph,con)
pop.set_topo(noise.Z.ravel())
rshp = list(noise.rshp)

fig,ax = plt.subplots()

im = ax.imshow(pop.get_topo().reshape(rshp), cmap = 'gray', extent = noise.extent())
cb = plt.colorbar(im)

ax.set_title(f"rows: {rshp[0]} cols: {rshp[1]}")
# pop.StSt(5)

def run(event):
	pop.StSt(5)
	ttop = pop.get_topo()
	im.set_data(ttop.reshape(rshp))
	im.set_clim(ttop.min(),ttop.max())
	plt.draw()

def restrict(event):
	pop.restriction(1.)
	rshp[0]*=2
	rshp[1]*=2
	ax.set_title(f"rows: {rshp[0]} cols: {rshp[1]}")
	ttop = pop.get_topo()
	im.set_data(ttop.reshape(rshp))
	im.set_clim(ttop.min(),ttop.max())
	plt.draw()


def interp(event):
	pop.interpolation()
	rshp[0] = math.floor(rshp[0]/2)
	rshp[1] = math.floor(rshp[1]/2)
	ax.set_title(f"rows: {rshp[0]} cols: {rshp[1]}")
	im.set_data(pop.get_topo().reshape(rshp))
	plt.draw()

def smooth(event):
	pop.smooth(1.)
	ttop = pop.get_topo()
	im.set_data(ttop.reshape(rshp))
	im.set_clim(ttop.min(),ttop.max())
	plt.draw()

def swoosh(event):
	pop.simple_Kfz(0.5,0.2,0.5);
	# ttop = pop.get_chistar()
	ttop = pop.get_topo()
	im.set_data(ttop.reshape(rshp))
	im.set_clim(ttop.min(),ttop.max())
	plt.draw()


axbrun = fig.add_axes([0.87, 0.88, 0.1, 0.075])
b_run = Button(axbrun, "iter") 
b_run.on_clicked(run)

axbrestrict = fig.add_axes([0.87, 0.77, 0.1, 0.075])
b_restrict = Button(axbrestrict, "restrict") 
b_restrict.on_clicked(restrict)

axbinterp = fig.add_axes([0.87, 0.66, 0.1, 0.075])
b_interp = Button(axbinterp, "interp") 
b_interp.on_clicked(interp)

axbsmooth = fig.add_axes([0.87, 0.55, 0.1, 0.075])
b_smooth = Button(axbsmooth, "smooth") 
b_smooth.on_clicked(smooth)

axbswoosh = fig.add_axes([0.87, 0.45, 0.1, 0.075])
b_swoosh = Button(axbswoosh, "swoosh") 
b_swoosh.on_clicked(swoosh)

plt.ion()
plt.show()

print("YOLO")



input("press [enter] to finish")