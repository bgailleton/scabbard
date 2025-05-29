# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import dagger as dag
import random

def generate_landscape():
	ny,nx = 512,512
	dy,dx = 200,200
	Urate = 5e-4
	Kr = 1e-6
	m = random.uniform(0.2,1.) 
	n = m/random.uniform(0.25,0.55) 
	rshp = (ny,nx)
	
	# Initialising an empty model in the variable ts
	ts = dag.trackscape()
	# Initialising the topography and its dimensions
	ts.init_perlin(nx, ny, dx, dy, "periodic_NS", 5, 8, 100, round(np.random.rand()*1000000), True)
	# ts.connector.set_default_boundaries("periodic_NS")
	topo = ts.get_topo().reshape(rshp)
	connector = dag.D8N(nx, ny, dx, dx, 0., 0.)
	connector.set_default_boundaries("periodic_NS")
	graph = dag.graph(connector)
	pop = dag.popscape(graph,connector)
	pop.set_topo(topo.ravel())
	pop.set_m(m)
	pop.set_n(n)
	pop.set_Kbase(Kr)
	[pop.interpolation() for p in range(5)]
	for i in range(5):
		pop.StSt(5)
		pop.restriction(1e-2)
	
	ret = pop.get_topo().reshape(ny,nx)
	del pop
	del ts

	return ret

# Function to generate a noisy 2D array
def generate_noisy_2d_array(nx, ny):
	"""
	Generate a noisy 2D array with sine/cosine waves.

	Args:
		nx (int): Number of points in the x-direction.
		ny (int): Number of points in the y-direction.

	Returns:
		np.ndarray: Noisy 2D array.
	"""
	x = np.linspace(0, 2 * np.pi, nx)
	y = np.linspace(0, 2 * np.pi, ny)
	X, Y = np.meshgrid(x, y)

	# Create sine and cosine waves with different wavelengths
	Z = np.sin(X*(np.random.rand()*5+0.75)) + np.cos(Y*(np.random.rand()*5+0.75))

	# Add some noise
	noise = np.random.normal(0, 0.1, (nx, ny))
	Z_noisy = Z + noise
	# plt.imshow(Z_noisy)
	# plt.show()
	return Z_noisy

def normalise(x):
	return (x - x.min())/(x.max() - x.min())

def normalise_m11(x):
	return 2*(x - x.min())/(x.max() - x.min()) - 1

def normalise_2pi(x):
	return 2*np.pi*(x - x.min())/(x.max() - x.min())

# Custom SIREN activation function
class SineActivation(nn.Module):
	def __init__(self):
		super(SineActivation, self).__init__()

	def forward(self, x):
		return torch.sin(x)

# Define the Autoencoder class
class Autoencoder(nn.Module):
	def __init__(self, input_size, hidden_size, latent_size):
		"""
		Initialize the Autoencoder.

		Args:
			input_size (int): Size of the input layer (flattened 2D array).
			hidden_size (int): Size of the hidden layer.
			latent_size (int): Size of the latent space (bottleneck).
		"""
		super(Autoencoder, self).__init__()

		# Encoder layers
		self.encoder = nn.Sequential(
			nn.Linear(input_size, hidden_size),  # First hidden layer
			SineActivation(),  # SineActivation activation function
			nn.Linear(hidden_size, latent_size),  # Latent space layer
			SineActivation()  # SineActivation activation function
		)

		# Decoder layers
		self.decoder = nn.Sequential(
			nn.Linear(latent_size, hidden_size),  # First hidden layer in decoder
			SineActivation(),  # ReLU activation function
			nn.Linear(hidden_size, input_size),  # Output layer
			nn.Sigmoid()  # Sigmoid activation function to ensure output is between 0 and 1
		)

	def forward(self, x):
		"""
		Forward pass of the autoencoder.

		Args:
			x (torch.Tensor): Input tensor.

		Returns:
			torch.Tensor: Reconstructed output tensor.
		"""
		# Encode the input
		encoded = self.encoder(x)

		# Decode the encoded input
		decoded = self.decoder(encoded)

		return decoded


# Define the Convolutional Autoencoder class
class ConvolutionalAutoencoder(nn.Module):
	def __init__(self):
		super(ConvolutionalAutoencoder, self).__init__()

		# Encoder
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 16, 5, stride=2, padding=2),  # (batch_size, 16, 64, 64)
			SineActivation(),
			nn.Conv2d(16, 32, 5, stride=2, padding=2),  # (batch_size, 32, 32, 32)
			SineActivation(),
			nn.Conv2d(32, 64, 5, stride=2, padding=2),  # (batch_size, 64, 16, 16)
			SineActivation(),
			nn.Conv2d(64, 128, 5, stride=2, padding=2),  # (batch_size, 128, 8, 8)
			SineActivation(),
			nn.Conv2d(128, 256, 5, stride=2, padding=2),  # (batch_size, 256, 4, 4)
			SineActivation(),
			nn.Conv2d(256, 512, 5, stride=2, padding=2),  # (batch_size, 512, 2, 2)
			SineActivation(),
			nn.Conv2d(512, 1024, 5, stride=2, padding=2)  # (batch_size, 1024, 1, 1)
		)

		# Decoder
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 512, 2, 2)
			SineActivation(),
			nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 256, 4, 4)
			SineActivation(),
			nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 128, 8, 8)
			SineActivation(),
			nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 64, 16, 16)
			SineActivation(),
			nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 32, 32, 32)
			SineActivation(),
			nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 16, 64, 64)
			SineActivation(),
			nn.ConvTranspose2d(16, 1, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 1, 128, 128)
			nn.Sigmoid()
		)

	def forward(self, x):
		latent = self.encoder(x)
		reconstructed = self.decoder(latent)
		return reconstructed

class ConvolutionalAutoencoder2(nn.Module):
	def __init__(self):
		super(ConvolutionalAutoencoder2, self).__init__()

		# Encoder
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 12, 5, stride=2, padding=2),  # (batch_size, 16, 256, 256)
			SineActivation(),
			nn.Conv2d(12, 24, 5, stride=2, padding=2),  # (batch_size, 32, 32, 32)
			SineActivation(),
			nn.Conv2d(24, 48, 5, stride=2, padding=2),  # (batch_size, 64, 16, 16)
			SineActivation(),
			nn.Conv2d(48, 96, 5, stride=2, padding=2),  # (batch_size, 128, 8, 8)
			SineActivation(),
			nn.Conv2d(96, 10*96, 5, stride=2, padding=2),  # (batch_size, 256, 4, 4)
			SineActivation(),
			# nn.Conv2d(256, 512, 5, stride=2, padding=2),  # (batch_size, 512, 2, 2)
			# SineActivation(),
			# nn.Conv2d(512, 1024, 5, stride=2, padding=2)  # (batch_size, 1024, 1, 1)
		)
		# for module in self.encoder:
		#   if isinstance(module, nn.Conv2d):
		#       # Custom initialization to [0, 2π]
		#       nn.init.uniform_(module.weight, 0, 2 * np.pi)
		#       # Initialize biases if needed
		#       if module.bias is not None:
		#           nn.init.constant_(module.bias, 0)

		# Decoder
		self.decoder = nn.Sequential(
			# nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 512, 2, 2)
			# SineActivation(),
			# nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 256, 4, 4)
			# SineActivation(),
			nn.ConvTranspose2d(10*96, 96, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 128, 8, 8)
			SineActivation(),
			nn.ConvTranspose2d(96, 48, 5, stride=2, padding=2, output_padding=1),  # (batch_size, 64, 16, 16)
			SineActivation(),
			nn.ConvTranspose2d(48, 24, 5, stride=2, padding=2, output_padding=1),  
			SineActivation(),
			nn.ConvTranspose2d(24, 12, 5, stride=2, padding=2, output_padding=1),  
			SineActivation(),
			nn.ConvTranspose2d(12, 1, 5, stride=2, padding=2, output_padding=1),  
			# nn.Sigmoid()
		)
		# for module in self.decoder:
		#   if isinstance(module, nn.Conv2d):
		#       # Custom initialization to [0, 2π]
		#       nn.init.uniform_(module.weight, 0, 2 * np.pi)
		#       # Initialize biases if needed
		#       if module.bias is not None:
		#           nn.init.constant_(module.bias, 0)

		

	def forward(self, x):
		latent = self.encoder(x)
		reconstructed = self.decoder(latent)
		return reconstructed

class CustomStepLR(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-6, last_epoch=-1, verbose=False):
		self.step_size = step_size
		self.gamma = gamma
		self.min_lr = min_lr
		super(CustomStepLR, self).__init__(optimizer, last_epoch, verbose)

	def get_lr(self):
		if not self._get_lr_called_within_step:
			warnings.warn("To get the last learning rate computed by the scheduler, "
						  "please use `get_last_lr()`.", UserWarning)

		return [max(lr * (self.gamma ** (self.last_epoch // self.step_size)), self.min_lr)
				for lr in self.base_lrs]
# NEXT STEPS
# TRAIN ONE BY ONE