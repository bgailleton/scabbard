import numpy as np
import matplotlib.pyplot as plt

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
def forward(x):
    # Encoder
    hidden_layer = relu(np.dot(x, W1) + b1)
    latent_layer = relu(np.dot(hidden_layer, W2) + b2)

    # Decoder
    hidden_layer_decoder = relu(np.dot(latent_layer, W3) + b3)
    output_layer = sigmoid(np.dot(hidden_layer_decoder, W4) + b4)

    return output_layer, hidden_layer, latent_layer, hidden_layer_decoder

def backward(x, W1, W2, W3, W4, b1, b2, b3, b4, output, hidden_layer, latent_layer, hidden_layer_decoder, learning_rate=0.01):
    # Compute gradients
    d_output = 2 * (output - x) / x.size  # Derivative of MSE

    d_hidden_layer_decoder = np.dot(d_output, W4.T) * (hidden_layer_decoder > 0)
    d_latent_layer = np.dot(d_hidden_layer_decoder, W3.T) * (latent_layer > 0)
    d_hidden_layer = np.dot(d_latent_layer, W2.T) * (hidden_layer > 0)

    # Update weights and biases
    W4 -= learning_rate * np.dot(hidden_layer_decoder.T, d_output)
    b4 -= learning_rate * np.sum(d_output, axis=0)
    W3 -= learning_rate * np.dot(latent_layer.T, d_hidden_layer_decoder)
    b3 -= learning_rate * np.sum(d_hidden_layer_decoder, axis=0)
    W2 -= learning_rate * np.dot(hidden_layer.T, d_latent_layer)
    b2 -= learning_rate * np.sum(d_latent_layer, axis=0)
    W1 -= learning_rate * np.dot(x.T, d_hidden_layer)
    b1 -= learning_rate * np.sum(d_hidden_layer, axis=0)

# Step 6: Train the Model
def train(X, W1, W2, W3, W4, b1, b2, b3, b4, epochs=10000, learning_rate=0.01):
    for epoch in range(epochs):
        # Forward pass
        output, hidden_layer, latent_layer, hidden_layer_decoder = forward(X)

        # Compute loss
        loss = mse_loss(X, output)

        # Backward pass
        backward(X, W1, W2, W3, W4, b1, b2, b3, b4, output, hidden_layer, latent_layer, hidden_layer_decoder, learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

# Flatten the input data
X_flattened = Z_noisy.reshape(1, -1)

# Train the autoencoder
train(X_flattened, W1, W2, W3, W4, b1, b2, b3, b4, epochs=50000, learning_rate=0.01)

# Step 7: Evaluate the Model
def evaluate(X):
    output, _, _, _ = forward(X)
    return output

# Evaluate the model
output = evaluate(X_flattened)
output_reshaped = output.reshape(nx, ny)

# Plot the reconstructed output
plt.imshow(output_reshaped, cmap='viridis')
plt.colorbar()
plt.title('Reconstructed 2D Array')
plt.show()

# Step 8: Use the Model
# Once trained, you can use the encoder part of the autoencoder to get the compressed representation of new data.