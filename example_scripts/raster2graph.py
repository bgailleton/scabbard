import scabbard as scb
import dagger as dag

# Loads 
con, graph, dem = scb.io.raster2graphcon("example.tif")

# COmputing the graph witht he given elevation for SFD
graph.compute_graph(dem['array'].ravel(), True, True)

# Getting a bunch of metrics:
stack = graph.get_SFD_stack()
SFD_receivers = con.get_SFD_receivers()
SFD_dx2receivers = con.get_SFD_dx()

HS = dag.hillshade(con,dem['array'])


print("dem is a dictionary with a couple of interesting geometrical info about the dem (and the dem itself as a 2D array):")
for keys,val in dem.items():
	print(keys)
