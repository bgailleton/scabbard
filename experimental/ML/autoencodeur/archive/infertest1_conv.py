# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import Autoencoder, generate_noisy_2d_array, ConvolutionalAutoencoder, ConvolutionalAutoencoder2, normalise

# Define the parameters
nx, ny = 128, 128  # Dimensions of the 2D array
input_size = nx * ny  # Size of the input layer (flattened 2D array)
hidden_size = 512  # Size of the hidden layer
latent_size = 32  # Size of the latent space (bottleneck)


# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nx,ny = 128,128

Ztest = generate_noisy_2d_array(nx, ny)
Ztest = (Ztest - Ztest.min()) / (Ztest.max() - Ztest.min())

# Instantiate the model again
model = ConvolutionalAutoencoder2()
model = model.to(device)

# Load the saved weights
model.load_state_dict(torch.load('autoencoder_bobo_conv.pth', weights_only = True))

# Set the model to evaluation mode if you're done training
model.eval()

temp = torch.FloatTensor(Ztest).view(1, nx,ny).to(device)
with torch.no_grad():
	Zcomp = model.encoder(temp)

	Z_decomp = model.decoder(Zcomp)

fig,axs = plt.subplots(1,3)
im0=axs[0].imshow(Ztest)
im1=axs[1].imshow(Z_decomp.view(ny,nx).cpu())
im2=axs[2].imshow(Z_decomp.view(ny,nx).cpu(), cmap = 'RdBu_r', vmin = -1e-1, vmax = 1e-1)

fig.show()

while True:

	Ztest = generate_noisy_2d_array(nx, ny)
	Ztest = (Ztest - Ztest.min()) / (Ztest.max() - Ztest.min())
	temp = torch.FloatTensor(Ztest).view(1, nx,ny).to(device)
	print(temp)
	with torch.no_grad():
		Zcomp = model.encoder(temp)

		Z_decomp = model.decoder(Zcomp)

	im0.set_data(Ztest)
	im1.set_data(Z_decomp.view(ny,nx).cpu().numpy())
	im2.set_data(Ztest - Z_decomp.view(ny,nx).cpu().numpy())

	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(5)