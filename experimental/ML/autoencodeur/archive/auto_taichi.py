import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import taichi as ti


# Define a dot product function using Numba
@ti.kernel
def dot_1d(a:ti.template(), b:ti.template(), out:ti.template()):
	"""Compute the dot product of two matrices using Numba."""

	out[None] = 0.0
	for i in a:
		out[None] += a[i] * b[i]

@ti.kernel
def dot_2d1d(a:ti.template(), b:ti.template(), out:ti.template()):
	"""Compute the dot product of two matrices using Numba."""
	for i ,j in a:
		out[i] += a[i, j] * b[j]

@ti.kernel
def dot_2d2d(a:ti.template(), b:ti.template(), out:ti.template()):
	"""Compute the dot product of two matrices using Numba."""

	for i,j, k in ti.ndrange(a.shape[0], b.shape[1], a.shape[1]):
		out[i, j] += a[i, k] * b[k, j]

# Step 3: Choose Activation Functions
@ti.kernel
def relu(x:ti.template(), out:ti.template()):
	for i in x:
		out[i] = ti.math.max(0, x[i])
@ti.kernel
def sigmoid(x:ti.template(), out:ti.template()):
	for i in x:
		out[i] = 1 / (1 + ti.math.exp(-x[i]))

# Step 4: Define the Loss Function
@ti.kernel
def mse_loss(y_true:ti.template(), y_pred:ti.template()):
	quant = 0.
	N = 0
	for i in y_true:
		ti.atomic_add(quant,(y_true[i] - y_pred[i]) ** 2)
		ti.atomic_add(N, 1)
	return quant/N


# Step 5: Compile the Model
@ti.kernel
def forward(x:ti.template(), W1:ti.template(), b1:ti.template(), W2:ti.template(), b2:ti.template(), W3:ti.template(), b3:ti.template(), W4:ti.template(), b4:ti.template()):
	# Encoder
	hidden_layer = relu(dot_numba(x, W1) + b1)
	latent_layer = relu(dot_numba(hidden_layer, W2) + b2)

	# Decoder
	hidden_layer_decoder = relu(dot_numba(latent_layer, W3) + b3)
	output_layer = sigmoid(dot_numba(hidden_layer_decoder, W4) + b4)

	return output_layer, hidden_layer, latent_layer, hidden_layer_decoder

def backward(x:ti.template(), output:ti.template(), hidden_layer:ti.template(), latent_layer:ti.template(), hidden_layer_decoder:ti.template(), W1:ti.template(), b1:ti.template(), W2:ti.template(), b2:ti.template(), W3:ti.template(), b3:ti.template(), W4:ti.template(), b4:ti.template(), learning_rate=0.01):
	# Compute gradients
	d_output = 2 * (output - x) / x.size  # Derivative of MSE

	d_hidden_layer_decoder = dot_numba(d_output, W4.T) * (hidden_layer_decoder > 0)
	d_latent_layer = dot_numba(d_hidden_layer_decoder, W3.T) * (latent_layer > 0)
	d_hidden_layer = dot_numba(d_latent_layer, W2.T) * (hidden_layer > 0)

	# Update weights and biases
	W4 -= learning_rate * dot_numba(hidden_layer_decoder.T, d_output)
	b4 -= learning_rate * np.sum(d_output, axis=0)
	W3 -= learning_rate * dot_numba(latent_layer.T, d_hidden_layer_decoder)
	b3 -= learning_rate * np.sum(d_hidden_layer_decoder, axis=0)
	W2 -= learning_rate * dot_numba(hidden_layer.T, d_latent_layer)
	b2 -= learning_rate * np.sum(d_latent_layer, axis=0)
	W1 -= learning_rate * dot_numba(x.T, d_hidden_layer)
	b1 -= learning_rate * np.sum(d_hidden_layer, axis=0)

	return W1, b1, W2, b2, W3, b3, W4, b4


ti.init(ti.gpu)

nx, ny = 128, 128

# Create a 2D array with sine/cosine waves and noise
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
X, Y = np.meshgrid(x, y)

# Create sine and cosine waves with different wavelengths
Z = np.sin(X) + np.cos(Y)

# Add some noise
noise = np.random.normal(0, 0.1, (ny, nx))
Z_noisy = Z + noise

Z_gpu = ti.field(ti.f32, shape=(ny,nx))
Z_gpu.from_numpy(Z)

Znoisy_gpu = ti.field(ti.f32, shape=(ny,nx))
Znoisy_gpu.from_numpy(Z_noisy)

A = ti.field(ti.f32, shape=(ny,nx))
A.fill(0.)

print(A + Znoisy_gpu)
quit()
# Step 1: Understand the Basics of Autoencoders
# An autoencoder is a type of neural network used to learn efficient codings of input data.
# It consists of two main parts: an encoder and a decoder.

# Step 2: Define the Architecture
input_size = nx * ny  # Size of the input layer (flattened 2D array)
hidden_size = 64      # Size of the hidden layer
latent_size = 32      # Size of the latent space (bottleneck)

# Initialize weights and biases
np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size)  # Encoder weights
b1 = np.zeros(hidden_size)  # Encoder biases
W2 = np.random.randn(hidden_size, latent_size)  # Latent space weights
b2 = np.zeros(latent_size)  # Latent space biases
W3 = np.random.randn(latent_size, hidden_size)  # Decoder weights
b3 = np.zeros(hidden_size)  # Decoder biases
W4 = np.random.randn(hidden_size, input_size)  # Output layer weights
b4 = np.zeros(input_size)  # Output layer biases



# Step 6: Train the Model
def train(X, W1, b1, W2, b2, W3, b3, W4, b4, epochs=10000, learning_rate=0.01):
	for epoch in range(epochs):
		# Forward pass
		output, hidden_layer, latent_layer, hidden_layer_decoder = forward(X, W1, b1, W2, b2, W3, b3, W4, b4)

		# Compute loss
		loss = mse_loss(X, output)

		# Backward pass
		W1, b1, W2, b2, W3, b3, W4, b4 = backward(X, output, hidden_layer, latent_layer, hidden_layer_decoder, W1, b1, W2, b2, W3, b3, W4, b4, learning_rate)

		if epoch % 1000 == 0:
			print(f"Epoch {epoch}, Loss: {loss}")

	return W1, b1, W2, b2, W3, b3, W4, b4

# Flatten the input data
X_flattened = Z_noisy.reshape(1, -1)

# Train the autoencoder
W1, b1, W2, b2, W3, b3, W4, b4 = train(X_flattened, W1, b1, W2, b2, W3, b3, W4, b4)

# Step 7: Evaluate the Model
def evaluate(X, W1, b1, W2, b2, W3, b3, W4, b4):
	output, _, _, _ = forward(X, W1, b1, W2, b2, W3, b3, W4, b4)
	return output

# Evaluate the model
output = evaluate(X_flattened, W1, b1, W2, b2, W3, b3, W4, b4)
output_reshaped = output.reshape(nx, ny)

# Plot the reconstructed output
plt.imshow(output_reshaped, cmap='viridis')
plt.colorbar()
plt.title('Reconstructed 2D Array')
plt.show()

# Step 8: Use the Model
# Once trained, you can use the encoder part of the autoencoder to get the compressed representation of new data.