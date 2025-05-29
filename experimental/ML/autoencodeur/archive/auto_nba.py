import numpy as np
import matplotlib.pyplot as plt
from numba import jit

nx, ny = 128, 128

# Create a 2D array with sine/cosine waves and noise
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
X, Y = np.meshgrid(x, y)

# Create sine and cosine waves with different wavelengths
Z = np.sin(X) + np.cos(Y)

# Add some noise
noise = np.random.normal(0, 0.1, (nx, ny))
Z_noisy = Z + noise

# Plot the noisy 2D array
plt.imshow(Z_noisy, cmap='viridis')
plt.colorbar()
plt.title('Noisy 2D Array with Sine/Cosine Waves')
plt.show()

# Define a dot product function using Numba
@jit(nopython=True)
def dot_numba(a, b):
    """Compute the dot product of two matrices using Numba."""
    if a.ndim == 1 and b.ndim == 1:
        result = 0.0
        for i in range(a.shape[0]):
            result += a[i] * b[i]
        return result
    elif a.ndim == 2 and b.ndim == 1:
        result = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                result[i] += a[i, j] * b[j]
        return result
    elif a.ndim == 2 and b.ndim == 2:
        result = np.zeros((a.shape[0], b.shape[1]))
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                for k in range(a.shape[1]):
                    result[i, j] += a[i, k] * b[k, j]
        return result
    else:
        raise ValueError("Unsupported dimensions for dot product.")

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

# Step 3: Choose Activation Functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 4: Define the Loss Function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 5: Compile the Model
def forward(x, W1, b1, W2, b2, W3, b3, W4, b4):
    # Encoder
    hidden_layer = relu(dot_numba(x, W1) + b1)
    latent_layer = relu(dot_numba(hidden_layer, W2) + b2)

    # Decoder
    hidden_layer_decoder = relu(dot_numba(latent_layer, W3) + b3)
    output_layer = sigmoid(dot_numba(hidden_layer_decoder, W4) + b4)

    return output_layer, hidden_layer, latent_layer, hidden_layer_decoder

def backward(x, output, hidden_layer, latent_layer, hidden_layer_decoder, W1, b1, W2, b2, W3, b3, W4, b4, learning_rate=0.01):
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