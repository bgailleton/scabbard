# Plotting wizard
import click
import scabbard as scb
import matplotlib.pyplot as plt


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
