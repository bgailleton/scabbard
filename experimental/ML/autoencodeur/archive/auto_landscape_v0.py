# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import Autoencoder, generate_noisy_2d_array, ConvolutionalAutoencoder, normalise, ConvolutionalAutoencoder2, normalise_m11, CustomStepLR, generate_landscape

import pickle



# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
nx, ny = 512, 512
batch_size = 32
epochs = 400
megaepochs = 10000
learning_rate = 1e-3
load_weights = False


# Initialize the model, loss function, and optimizer
model = ConvolutionalAutoencoder2().to(device)
if(load_weights):
    model.load_state_dict(torch.load('autoldscapev0.pth', weights_only = True))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel'
# )





scheduler = CustomStepLR(optimizer, step_size=100, gamma=0.99, min_lr=4e-5, verbose=True)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8, verbose = True)

# Generate training data


losses_log = []
# Training loop
for megaepoch in range(megaepochs):

    # if(epoch == 300):
    #     learning_rate /= 10
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # elif epoch == 3000:
    #     learning_rate /= 10
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('megaepoch', megaepoch, 'so epoch', megaepoch*epochs)
    print('generating... ', end='', flush = True)
    cputdata = np.array([normalise(generate_landscape()) for _ in range(batch_size)])
    # cputdata = np.array([normalise(np.random.rand(512,512)) for _ in range(batch_size)])
    train_data = torch.FloatTensor(cputdata).view(batch_size, 1, nx, ny).to(device)
    print('OK!')

    
    for epoch in range(epochs):

        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_data)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            enc = model.encoder(train_data[0])
        scheduler.step() 
        current_lr = optimizer.param_groups[0]['lr']

        # Save
        losses_log.append(loss.item())
        with open('ldscv1.pkl', 'wb') as f:
            pickle.dump(losses_log, f)

        if((epoch+1) % 10 == 0):
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}, Current lr: {current_lr}, shape: {np.prod(train_data[0].shape)} -> {np.prod(enc.shape)}')
        if((epoch+1) % 100 == 0):
            print('saving!')
            # Save the model's weights
            torch.save(model.state_dict(), 'autoldscapev0.pth')
    del cputdata

# # Plot the original and reconstructed images
# with torch.no_grad():
#     reconstructed = model(train_data)

# fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# axs[0].imshow(train_data.view(nx, ny), cmap='gray')
# axs[0].set_title('Original Image')
# axs[1].imshow(reconstructed[0].view(nx, ny), cmap='gray')
# axs[1].set_title('Reconstructed Image')
# plt.show()

quit()
    
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