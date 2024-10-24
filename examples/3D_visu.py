'''
This example shows how to use my homemade, WIP-for-fun-and-to-make-pretty-figure ray-traycing engine plot a 3D topography

'''

import scabbard as scb
import matplotlib.pyplot as plt

grid = scb.io.load_raster('dem.tif')
grid.Z[grid.Z<50] = 50

plt.imshow(grid.Z)
plt.show()

image = scb.visu.gray_RT(grid, exaggeration_factor = 0.2)
# print(image.shape)
# quit()
# Assuming `image_array` is your NumPy array with shape (height, width, 4) for RGBA
fig,ax = plt.subplots(figsize=(image.shape[1]/100, image.shape[0]/100), dpi=100)

# Display the image without axis and padding
ax.imshow(image)
ax.axis('off')  # Hide axis
ax.set_position([0, 0, 1, 1])  # Remove padding and fill figure with image
plt.show()