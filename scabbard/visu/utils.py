import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def m_to_km(ax, ndec = 1):
	'''
	Takes a matplotlib axis and converts the units from m to km
	Cannot really check what the axis units are so effectively just divides them by 1000

	Arguments:
		- ax: the matplotlib axis
		- ndec (int): the number of decimals to round for

	Returns:
		- nothing, edits in place

	Authors:
		- B.G. (last modifications: 08/2024)

	'''

	# Nested function to format the tick labels
	def format_func(value, tick_number):
		return f'{round(value / 1000, ndec)}'

	# Apply the formatting function to x and y axis tick labels
	ax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(ax.get_xticks())])
	ax.set_yticklabels([format_func(tick, i) for i, tick in enumerate(ax.get_yticks())])

	# Eventually changes the axis labels
	if '(m)' in ax.get_xlabel():
		ax.set_xlabel(ax.get_xlabel().replace('(m)','(km)'))
	if '(m)' in ax.get_ylabel():
		ax.set_ylabel(ax.get_ylabel().replace('(m)','(km)'))		



def tickgrid(ax, marker = '+', size = 40, color = 'k', alpha = 0.7):
	'''
	Adds a tick grid to an existing axis
	tickgrid is a grid only plotting markers at the crossings of the grid lines
	Feat. Luca Malatesta.

	Arguments:
		- ax: the matplotlib axis
		- marker: the matplotlib marker type
		- size: the point size (argument s in scatter plots)
		- color: the color (any matplotlib accepted color)
		- alpha: the transparency (0-1)

	Returns:
		- nothing, edits in place

	Authors:
		- B.G. (last modifications: 08/2024)

	'''
	X = []
	Y = []

	for x in list(ax.get_xticks()):
		if(x < ax.get_xlim()[0] or x > ax.get_xlim()[1]):
			continue
		for y in list(ax.get_yticks()):
			if(y < ax.get_ylim()[0] or y > ax.get_ylim()[1]):
				continue
			X.append(x)
			Y.append(y)

	ax.scatter(X, Y, marker = marker, s = size, alpha = alpha, edgecolor = color, facecolor = color, lw = 1)


def convert_log_colorbar_labels_to_scientific(cb):
	'''
	Takes a matplotlib colorbar and converts its label to 10^value

	Arguments:
		- cb: the colorbar object
		- ndec (int): the number of decimals to round for

	Returns:
		- nothing, edits in place

	Authors:
		- B.G. (last modifications: 08/2024)

	'''

	# Define a formatter function to convert log10 values to scientific notation
	def scientific_ticks(x, pos):
		return f"$10^{{{int(x)}}}$"

	# Apply the formatter to the colorbar
	cb.ax.xaxis.set_major_formatter(FuncFormatter(scientific_ticks))
	cb.ax.yaxis.set_major_formatter(FuncFormatter(scientific_ticks))