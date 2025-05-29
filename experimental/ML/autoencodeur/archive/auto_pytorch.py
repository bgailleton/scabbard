# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import Autoencoder, generate_noisy_2d_array


# Define the parameters
nx, ny = 128, 128  # Dimensions of the 2D array
input_size = nx * ny  # Size of the input layer (flattened 2D array)
hidden_size = 512  # Size of the hidden layer
latent_size = 32  # Size of the latent space (bottleneck)



# Generate multiple samples of noisy 2D arrays
num_samples = 1000
samples = [generate_noisy_2d_array(nx, ny) for _ in range(num_samples)]

# Normalize each sample to the range [0, 1]
samples_normalized = [(sample - sample.min()) / (sample.max() - sample.min()) for sample in samples]

# Plot one of the noisy 2D arrays
plt.imshow(samples_normalized[0], cmap='viridis')
plt.colorbar()
plt.title('Normalized Noisy 2D Array with Sine/Cosine Waves')
plt.show()

# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert the normalized noisy 2D arrays to PyTorch tensors, flatten them, and move them to the device
X_tensors = [torch.FloatTensor(sample).view(1, -1).to(device) for sample in samples_normalized]

# Initialize the autoencoder and move it to the device
autoencoder = Autoencoder(input_size, hidden_size, latent_size).to(device)

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define the optimizer (Adam)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-5)

# Train the autoencoder
num_epochs = 100000
for epoch in range(num_epochs):
    total_loss = 0
    # for X_tensor in X_tensors:
    #     # Forward pass
    #     output = autoencoder(X_tensor)

    #     # Compute the loss
    #     loss = criterion(output, X_tensor)

    #     # Backward pass and optimize
    #     optimizer.zero_grad()  # Clear the gradients
    #     loss.backward()  # Compute the gradients
    #     optimizer.step()  # Update the weights

    #     total_loss += loss.item()

    for i in range(1000):

        sample = generate_noisy_2d_array(nx, ny)
        sample = (sample - sample.min()) / (sample.max() - sample.min())

        X_tensor = torch.FloatTensor(sample).view(1, -1).to(device)

        # Forward pass
        output = autoencoder(X_tensor)

        # Compute the loss
        loss = criterion(output, X_tensor)

        # Backward pass and optimize
        optimizer.zero_grad()  # Clear the gradients
        loss.backward()  # Compute the gradients
        optimizer.step()  # Update the weights

        total_loss += loss.item()

    # Print the average loss every 1000 epochs
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {total_loss / num_samples:.8f}')

    if(epoch % 100 == 0):
        print('saving!')
        # Save the model's weights
        torch.save(autoencoder.state_dict(), 'autoencoder_bobo.pth')
# Evaluate the model on one of the samples
with torch.no_grad():  # Disable gradient computation for evaluation
    output = autoencoder(X_tensors[0])

# Reshape the output to the original 2D shape and convert it to a numpy array
output_reshaped = output.view(nx, ny).cpu().numpy()

# Plot the reconstructed output
plt.imshow(output_reshaped, cmap='viridis')
plt.colorbar()
plt.title('Reconstructed 2D Array')
plt.show()

# Function to get the compressed representation of new data
def get_compressed_representation(autoencoder, new_data):
    """
    Get the compressed representation of new data using the encoder part of the autoencoder.

    Args:
        autoencoder (Autoencoder): Trained autoencoder model.
        new_data (np.ndarray): New input data to be compressed.

    Returns:
        np.ndarray: Compressed representation of the new data.
    """
    # Normalize the new data to the range [0, 1]
    new_data_normalized = (new_data - new_data.min()) / (new_data.max() - new_data.min())

    # Convert the normalized new data to a PyTorch tensor, flatten it, and move it to the device
    new_data_tensor = torch.FloatTensor(new_data_normalized).view(1, -1).to(device)

    # Get the compressed representation using the encoder
    with torch.no_grad():
        compressed_representation = autoencoder.encoder(new_data_tensor)

    # Convert the compressed representation to a numpy array
    compressed_representation = compressed_representation.cpu().numpy()

    return compressed_representation

# Example usage of the get_compressed_representation function
new_data = np.random.rand(nx, ny)  # Example new data
compressed_representation = get_compressed_representation(autoencoder, new_data)
print(f"Compressed representation shape: {compressed_representation.shape}")

# Save the model's weights
torch.save(autoencoder.state_dict(), 'autoencoder_bobo.pth')