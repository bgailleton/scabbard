'''
This example shows how to use my homemade, WIP-for-fun-and-to-make-pretty-figure ray-traycing engine plot a 3D topography

'''

import scabbard as scb
import matplotlib.pyplot as plt

grid = scb.io.load_raster('dem.tif')
# grid = scb.filters.gaussian_fourier(grid, magnitude = 100)

image = scb.visu.std_gray_RT(grid, exaggeration_factor = 0.3, N_AA = 10, tone_mapping = False, toon = 0)

# fig,ax = plt.subplots(dpi=150)

# # Display the image without axis and padding
# im = ax.imshow(image)
# ax.axis('off')  # Hide axis
# ax.set_position([0, 0, 1, 1])  # Remove padding and fill figure with image
# plt.show()

res = scb.graphflood.std_run(grid, N_dt = 20000, P = 1e-5)

# plt.imshow(res['hw'])
# plt.show()
# quit()

image = scb.visu.std_water_RT(grid, res['hw'], exaggeration_factor = 0.3, N_AA = 10, tone_mapping = False, toon = 0)

fig,ax = plt.subplots(dpi=150)

# Display the image without axis and padding
im = ax.imshow(image)
ax.axis('off')  # Hide axis
ax.set_position([0, 0, 1, 1])  # Remove padding and fill figure with image
plt.show()