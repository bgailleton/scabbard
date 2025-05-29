import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from collections import defaultdict
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_data(N):
    """
    Generic function to generate N samples of 512x512 data
    Replace this with your actual data generation logic
    """
    # Placeholder: generating random data for demonstration
    return np.random.randn(N, 1, 512, 512).astype(np.float32)

class SimpleAutoencoder(nn.Module):
    """
    Simple 3-layer autoencoder - Start here!
    Architecture: 512x512 -> 128x128 -> 32x32 -> bottleneck -> 32x32 -> 128x128 -> 512x512
    """
    def __init__(self, bottleneck_size=1024):
        super(SimpleAutoencoder, self).__init__()
        
        # ENCODER - Simple version (3 conv layers)
        self.encoder = nn.Sequential(
            # Layer 1: 512x512x1 -> 256x256x32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),  # Stabilizes training
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 256x256x32 -> 128x128x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 128x128x64 -> 64x64x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # TO ADD MORE COMPLEXITY LATER:
            # nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 32x32x256
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> 16x16x512
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate the flattened size after convolutions
        # For simple version: 64x64x128 = 524288
        self.encoder_output_size = 64 * 64 * 128
        
        # BOTTLENECK - Dense layers for compression
        self.bottleneck_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder_output_size, bottleneck_size),
            nn.LeakyReLU(0.2, inplace=True),
            # OPTIONAL: Add another dense layer for more compression
            # nn.Linear(bottleneck_size, bottleneck_size // 2),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.bottleneck_decoder = nn.Sequential(
            # OPTIONAL: Mirror the additional dense layer if added above
            # nn.Linear(bottleneck_size // 2, bottleneck_size),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(bottleneck_size, self.encoder_output_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # DECODER - Mirror of encoder
        self.decoder = nn.Sequential(
            # Reshape to 64x64x128
            # Layer 3: 64x64x128 -> 128x128x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 128x128x64 -> 256x256x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 1: 256x256x32 -> 512x512x1
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # No activation on final layer - linear output for reconstruction
            
            # TO ADD MORE COMPLEXITY LATER (add these BEFORE the existing layers):
            # nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.bottleneck_size = bottleneck_size
        
    def encode(self, x):
        """Encode input to bottleneck representation"""
        x = self.encoder(x)
        x = self.bottleneck_encoder(x)
        return x
    
    def decode(self, x):
        """Decode from bottleneck to reconstruction"""
        x = self.bottleneck_decoder(x)
        x = x.view(-1, 128, 64, 64)  # Reshape for decoder
        # FOR MORE COMPLEX VERSION: x = x.view(-1, 512, 16, 16)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded

class TrainingMonitor:
    """Monitor training progress and handle learning rate scheduling"""
    def __init__(self, patience=10, min_delta=1e-6):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.patience = patience
        self.wait = 0
        self.min_delta = min_delta
        self.early_stop = False
        
    def update(self, train_loss, val_loss, lr):
        """Update monitoring metrics"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        
        # Early stopping logic
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                
    def plot_progress(self):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.set_yscale('log')
        
        ax2.plot(self.learning_rates)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.show()

def calculate_compression_ratio(model):
    """Calculate compression ratio achieved by the model"""
    original_size = 512 * 512  # Original data size
    compressed_size = model.bottleneck_size
    ratio = original_size / compressed_size
    print(f"Compression ratio: {ratio:.2f}x (from {original_size} to {compressed_size})")
    return ratio

def visualize_reconstructions(model, data_loader, num_samples=4):
    """Visualize original vs reconstructed samples"""
    model.eval()
    with torch.no_grad():
        data_iter = iter(data_loader)
        images, _ = next(data_iter)
        images = images[:num_samples].to(device)
        
        reconstructed, _ = model(images)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
            
        plt.tight_layout()
        plt.show()

def train_autoencoder():
    """Main training function with progressive data generation strategy"""
    
    # HYPERPARAMETERS - Start conservative, tune later
    INITIAL_BATCH_SIZE = 16  # Small batch for stability
    BOTTLENECK_SIZE = 1024   # Start large, reduce later: try 512, 256, 128
    INITIAL_LR = 1e-3
    SAMPLES_PER_GENERATION = 500  # Generate this many samples at a time
    EPOCHS_PER_GENERATION = 20    # Train for this many epochs before generating new data
    TOTAL_GENERATIONS = 10        # How many times to generate new data
    
    # Initialize model
    model = SimpleAutoencoder(bottleneck_size=BOTTLENECK_SIZE).to(device)
    
    # PROGRESSIVE COMPLEXITY STRATEGY:
    # 1. First train this simple model until convergence
    # 2. Then uncomment the additional layers in the model
    # 3. Load the trained weights and continue training with deeper architecture
    # 4. Gradually reduce BOTTLENECK_SIZE: 1024 -> 512 -> 256 -> 128
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    # ALTERNATIVE LOSS FUNCTIONS TO TRY:
    # criterion = nn.L1Loss()  # Less sensitive to outliers
    # criterion = nn.SmoothL1Loss()  # Combination of L1 and L2
    
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    
    # Learning rate scheduler - reduces LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # Training monitor
    monitor = TrainingMonitor(patience=15)
    
    print(f"Model architecture:")
    print(f"- Bottleneck size: {BOTTLENECK_SIZE}")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    calculate_compression_ratio(model)
    
    # PROGRESSIVE TRAINING STRATEGY
    for generation in range(TOTAL_GENERATIONS):
        print(f"\n=== Generation {generation + 1}/{TOTAL_GENERATIONS} ===")
        
        # Generate new training data
        print(f"Generating {SAMPLES_PER_GENERATION} new samples...")
        start_time = time.time()
        train_data = generate_data(SAMPLES_PER_GENERATION)
        val_data = generate_data(SAMPLES_PER_GENERATION // 5)  # Smaller validation set
        print(f"Data generation took {time.time() - start_time:.2f}s")
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_data))
        val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.FloatTensor(val_data))
        
        train_loader = DataLoader(train_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=False)
        
        # Train for specified epochs on this data
        for epoch in range(EPOCHS_PER_GENERATION):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                
                optimizer.zero_grad()
                reconstructed, encoded = model(data)
                loss = criterion(reconstructed, data)
                
                # ADDITIONAL LOSS TERMS TO CONSIDER:
                # sparsity_loss = torch.mean(torch.abs(encoded))  # Encourage sparse representations
                # loss += 0.001 * sparsity_loss  # Small weight for sparsity
                
                loss.backward()
                
                # GRADIENT CLIPPING - uncomment if training becomes unstable
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
                # Progress reporting
                if batch_idx % 10 == 0:
                    print(f'Gen {generation+1}, Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    reconstructed, _ = model(data)
                    val_loss += criterion(reconstructed, data).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update monitor
            monitor.update(train_loss, val_loss, current_lr)
            
            print(f'Gen {generation+1}, Epoch {epoch+1}: Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}')
            
            # Early stopping check
            if monitor.early_stop:
                print("Early stopping triggered!")
                break
        
        # Visualize progress every few generations
        if (generation + 1) % 3 == 0:
            print("Visualizing reconstructions...")
            visualize_reconstructions(model, val_loader)
        
        # ADAPTIVE STRATEGY - reduce bottleneck size if reconstruction is good
        if generation > 0 and monitor.val_losses[-1] < 0.001:  # Adjust threshold as needed
            print("Good reconstruction achieved! Consider reducing bottleneck size next run.")
    
    # Final evaluation and visualization
    print("\n=== Training Complete ===")
    monitor.plot_progress()
    visualize_reconstructions(model, val_loader, num_samples=8)
    
    # SAVE MODEL FOR LATER USE
    torch.save({
        'model_state_dict': model.state_dict(),
        'bottleneck_size': BOTTLENECK_SIZE,
        'train_losses': monitor.train_losses,
        'val_losses': monitor.val_losses,
    }, 'simple_autoencoder.pth')
    
    print("Model saved as 'simple_autoencoder.pth'")
    
    # NEXT STEPS STRATEGY:
    print("\n=== Next Steps for Progressive Complexity ===")
    print("1. If reconstruction quality is good, reduce bottleneck_size and retrain")
    print("2. If quality is poor, uncomment additional layers in the model")
    print("3. Consider adding skip connections (U-Net style) if details are lost")
    print("4. Experiment with different loss functions (L1, Perceptual loss)")
    print("5. Add regularization (dropout, weight decay) if overfitting")
    
    return model, monitor

# ADVANCED STRATEGIES TO IMPLEMENT LATER:

def progressive_training_strategy():
    """
    Advanced strategy for progressive complexity:
    1. Start with current simple model
    2. Once converged, add layers gradually
    3. Use transfer learning to initialize new layers
    """
    # STEP 1: Train simple model (done above)
    
    # STEP 2: Create deeper model and transfer weights
    # simple_model = torch.load('simple_autoencoder.pth')
    # deeper_model = DeeperAutoencoder()  # Would need to implement
    # # Transfer compatible weights from simple to deeper model
    
    # STEP 3: Continue training with new architecture
    pass

def hyperparameter_tuning_strategy():
    """
    Systematic hyperparameter tuning approach:
    """
    hyperparams_to_try = [
        {'bottleneck_size': 1024, 'lr': 1e-3, 'batch_size': 16},
        {'bottleneck_size': 512, 'lr': 1e-3, 'batch_size': 16},
        {'bottleneck_size': 256, 'lr': 5e-4, 'batch_size': 32},
        {'bottleneck_size': 128, 'lr': 1e-4, 'batch_size': 32},
    ]
    
    # Grid search implementation would go here
    pass

def advanced_loss_functions():
    """
    More sophisticated loss functions to try:
    """
    # Perceptual loss using pre-trained features
    # SSIM loss for better visual quality
    # Adversarial loss (GAN-like training)
    pass

if __name__ == "__main__":
    # Run the training
    model, monitor = train_autoencoder()
    
    # DEBUGGING TIPS:
    # 1. If loss doesn't decrease: reduce learning rate, check data normalization
    # 2. If loss decreases but reconstructions are blurry: try L1 loss, add perceptual loss
    # 3. If overfitting (train loss << val loss): add dropout, reduce model complexity
    # 4. If underfitting: increase model capacity, train longer, check data quality