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
