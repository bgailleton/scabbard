# Plotting wizard
import click
import scabbard as scb
import matplotlib.pyplot as plt
import dagger as dag

# ANSI escape sequences for colors
RESET = "\x1b[0m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"

@click.command()
@click.argument('fname', type = str)
def simplemapwizard(fname):
	plt.ioff()
	dem = scb.raster2RGrid(fname)
	atlas = scb.Dplot.basemap(dem)
	atlas.fig.show()
	plt.pause(0.01)
	input("press Enter to continue")


@click.command()
@click.option('-c', '--courant', 'courant',  type = float, default = 1e-3)
@click.option('-d', '--dt', 'dt',  type = float, default = None)
@click.option('-P', '--precipitations', 'precipitations',  type = float, default = 30)
@click.option('-m', '--manning', 'manning',  type = float, default = 0.033)
@click.option('-S', '--SFD', 'SFD', type = bool, default=False)
@click.option('-U', '--update_step', 'nupdate', type = int, default=10)
@click.option('-exp', '--experimental', 'experimental', type = bool, default=False, is_flag = True)
@click.argument('fname', type = str, default = 1e-3)
def graphflood_basic(fname,courant,dt,precipitations,manning,SFD,nupdate, experimental):
	print("EXPERIMENTAL IS ", experimental)

	P = precipitations /1e3/3600
	if(dt is not None and courant == 1e-3):
		print(RED + "WARNING, dt set to constant value AND courant, is that normal?\n\n" + RESET)

	print("+=+=+=+=+=+=+=+=+=+=+=+=+")
	print("+=+=+=GRAPHFLOOD+=+=+=+=+")
	print("+=+=+=+=+=+=+=+=+=+=+=+=+\n\n")

	print("Precipitation rates = ", P, " m/s (",precipitations," mm/h)")


	mod = scb.ModelHelper()
	mod.init_dem_model(fname, sea_level = 0., P = 1e-5)

	# Number of nodes in the Y direction
	ny = mod.grid.ny
	nx = mod.grid.nx


	mod.courant = False if(dt is not None) else True
	mod.stationary = True 
	# mod.gf.flood.enable_Qwout_recording()

	# manning friction:
	mod.mannings = manning

	# Single flow solver?
	mod.SFD = SFD;


	mod.dt = dt if dt is not None else 1e-3
	mod.min_courant_dt = 1e-6
	mod.courant_number = courant

	ph = scb.PlotHelper(mod)
	ph.init_hw_plot(use_extent = False)

	update_fig = nupdate
	i = 0
	j = 0
	while True:
		i+=1
		mod.run() if(experimental == False) else mod.gf.flood.run_hydro_only()
		if(i % update_fig > 0):
			continue
		hw = mod.gf.flood.get_hw().reshape(mod.gf.grid.rshp)
		print("Running step", i)
		ph.update()




@click.command()
@click.argument('fname', type = str)
def _debug_1(fname):
	plt.ioff()
	dem = scb.raster2RGrid(fname)
	atlas = scb.Dplot.basemap(dem)
	atlas.fig.show()
	plt.pause(0.01)
	while(True):
		plt.pause(1)
		dem.add_random_noise(-10,10)
		atlas.update()
	input("press Enter to continue")
