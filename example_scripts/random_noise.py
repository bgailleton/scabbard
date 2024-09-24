import scabbard as scb
import matplotlib.pyplot as plt

noise = scb.generate_noise_RGrid(nx = 512, ny = 512, noise_type="perlin", octaves = 14, n_gaussian_smoothing = 0)

plt.imshow( noise.Z2D, extent = noise.extent(), cmap = "magma")
plt.show()
