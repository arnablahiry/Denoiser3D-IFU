"""
3D U-Net Training Pipeline for IFU Datacube Denoising

This script implements a comprehensive training pipeline for 3D U-Net neural networks
applied to Integral Field Unit (IFU) datacube denoising. The pipeline includes
dataset loading, train/validation/test splitting, model architecture configuration,
and training with validation monitoring.

Pipeline Overview:
-----------------
1. Load preprocessed synthetic training datasets with noise characteristics
2. Split datasets into training, validation, and test sets (80:10:10 ratio)
3. Configure 3D U-Net architecture with padding wrapper for variable input sizes
4. Train model using Adam optimizer with MSE loss and validation monitoring
5. Save best model weights based on validation performance

Dependencies:
------------
- Custom modules: u_net_model.py, toy_cube_dataset.py
- Standard libraries: torch, numpy, astropy, pickle

"""

# ========================================================================================
# LIBRARY IMPORTS AND ENVIRONMENT SETUP
# ========================================================================================

import pickle                                        # Python object serialization
from u_net_model import *                           # Custom 3D U-Net architecture definitions
from toy_cube_dataset import *         # Synthetic datacube dataset classes
import torch                                        # PyTorch deep learning framework
import os                                          # Operating system interface
from torch.utils.data import DataLoader, random_split  # Data loading and splitting utilities
import torch.nn as nn                              # Neural network modules and loss functions

# ========================================================================================
# DATASET CONFIGURATION AND LOADING
# ========================================================================================

print(f'(*) Loading dataset for convolved noises')

# Dataset parameters for synthetic IFU cube generation
n_spectral_slices = 40     # Number of velocity/frequency channels in synthetic cubes
final_grid_size = 96       # Spatial resolution (pixels) for training cubes
n_cubes = 20000           # Total number of synthetic cubes in the training dataset
batch_size = 16           # Mini-batch size for training (adjust based on GPU memory)

# Dataset file path configuration
# Alternative path template for parameterized datasets:
# fname = f'/home/alahiry/data/toy_cubes/datasets/final/final_dataset_{n_spectral_slices}_{final_grid_size}_{n_cubes}.pkl'
fname = '/home/alahiry/data/toy_cubes/datasets/FINAL/final_dataset_40_96_20000.pkl'

print(f'(*) Loading dataset from: {fname}')
print(f'(*) Dataset configuration: {n_spectral_slices} channels, {final_grid_size}x{final_grid_size} spatial, {n_cubes} cubes')

# Load the pickled dataset object containing synthetic IFU cubes
# Dataset includes clean cubes, noisy observations, and associated parameters
with open(fname, "rb") as file:
    dataset = pickle.load(file)


print(f'(*) Whole dataset ({n_cubes} cubes) loaded successfully')
print('(*) Dataset statistics (mean and std):', dataset.return_stats())

# ========================================================================================
# DATASET SPLITTING AND DATALOADER CREATION  
# ========================================================================================

print('(*) Splitting dataset into training, validation and test set (80:10:10)')

# Configure reproducible random splitting for consistent train/val/test sets
# Using fixed seed ensures reproducibility across runs
generator = torch.Generator().manual_seed(42)

# Define split proportions following standard machine learning practices
# 80% training: Large set for gradient-based optimization
# 10% validation: Monitor overfitting and hyperparameter tuning  
# 10% test: Final unbiased performance evaluation
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size  # Ensure complete dataset coverage

# Perform random dataset splitting with reproducible generator
train_dataset, valid_dataset, test_dataset = random_split(
    dataset, [train_size, valid_size, test_size], generator=generator)

print('(*) Constructing PyTorch DataLoaders for efficient batch processing')

# Create DataLoaders for training and validation
# Training loader: Shuffled for better gradient estimation and convergence
# Validation loader: Shuffled to avoid potential ordering biases
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


# Display dataset split information for verification
print(f"(*) Dataset split completed:")
print(f"    Train size: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
print(f"    Valid size: {len(valid_dataset)} ({len(valid_dataset)/len(dataset)*100:.1f}%)")
print(f"    Test size: {len(test_dataset)} ({len(test_dataset)/len(dataset)*100:.1f}%)")

# ========================================================================================
# COMPUTATIONAL DEVICE CONFIGURATION
# ========================================================================================

# Configure GPU usage for accelerated training
# Note: Using cuda:1 for multi-GPU systems; adjust based on available hardware
print('(*) Choosing GPU if available:')

device = torch.device('cuda:1')  # Specify particular GPU (change to 'cuda:0' or 'cpu' as needed)

print(f'(*) Currently using device: {device}')

# ========================================================================================
# OUTPUT DIRECTORY CONFIGURATION
# ========================================================================================

# Configure output directory for model weights and training logs
# Directory naming convention includes key hyperparameters for organization
# Alternative naming template (commented out):
# dir_wt = '/home/alahiry/deep_learning/denoise_comparison/x{}_n{}'.format(final_grid_size, n_cubes)

dir_wt = '/home/alahiry/deep_learning/denoise_comparison/FINAL_CRISTAL/x{}_n{}_16_filters_batch_{}'.format(
    final_grid_size, n_cubes, batch_size)

print(f'(*) Output directory for weights: {dir_wt}')

# Create output directory if it doesn't exist
if not os.path.exists(dir_wt):
    os.makedirs(dir_wt)
    print(f'(*) Created new directory: {dir_wt}')
else:
    print(f'(*) Using existing directory: {dir_wt}')

print('(*) Starting U-Net training and validation pipeline...\n')

# ========================================================================================
# TRAINING FUNCTION DEFINITION
# ========================================================================================

def fit(iterations, model, train_loader, val_loader, device):
    """
    Train 3D U-Net model for IFU spectroscopic data denoising.
    
    This function implements the complete training loop including forward/backward passes,
    validation monitoring, checkpointing, and loss logging. The training uses Mean Squared
    Error (MSE) loss with Adam optimization, which is well-suited for regression tasks
    like denoising where we want to minimize pixel-wise reconstruction errors.
    
    Parameters:
    -----------
    iterations : int
        Number of training epochs to perform
    model : torch.nn.Module
        3D U-Net model instance to train
    train_loader : torch.utils.data.DataLoader
        DataLoader providing training batches
    val_loader : torch.utils.data.DataLoader
        DataLoader providing validation batches
    device : torch.device
        Computing device (CPU or GPU) for training
    
    Returns:
    --------
    min_valid_loss : float
        Best validation loss achieved during training
        
    Training Strategy:
    -----------------
    - Adam optimizer with batch-size-scaled learning rate
    - MSE loss for pixel-wise denoising optimization
    - Validation-based checkpointing to prevent overfitting
    - Comprehensive loss logging for training monitoring
    
    Scientific Rationale:
    --------------------
    The MSE loss encourages the model to minimize the L2 distance between denoised
    and clean spectral cubes, which preserves both morphological and flux characteristics
    important for astronomical analysis. The validation monitoring prevents overfitting
    to training noise patterns that may not generalize to real observations.
    """
    
    # Initialize training state variables
    min_valid_loss = float("inf")   # Track best validation performance for checkpointing
    
    # ---- Optimizer Configuration ----
    # Scale learning rate by batch size following standard practice
    # Larger batches provide more stable gradients, allowing higher learning rates
    base_lr = 5e-5
    lr = base_lr * (batch_size / 64)  # Reference batch size of 64
    
    print(f'(*) Configured Adam optimizer with learning rate: {lr:.2e}')
    print(f'(*) Learning rate scaling: {base_lr:.2e} * ({batch_size}/64) = {lr:.2e}')
    
    # Adam optimizer: adaptive learning rate with momentum
    # Well-suited for noisy gradients in deep neural network training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # ---- Loss Function Configuration ----
    # Mean Squared Error for regression-based denoising
    # Penalizes large reconstruction errors while being differentiable
    criterion = nn.MSELoss()
    
    print(f'(*) Using MSE loss for denoising optimization')
    
    # Move model to specified device (CPU/GPU)
    model.to(device)
    print(f'(*) Model moved to device: {device}')
    
    print(f'(*) Starting training for {iterations} epochs...\n')
    
    # ---- Main Training Loop ----
    for iteration in range(iterations):
        
        # ================================================================
        # TRAINING PHASE
        # ================================================================
        
        total_samples = 0     # Track total samples for accurate loss averaging
        train_loss_sum = 0.0  # Accumulate batch losses
        
        model.train()  # Enable training mode (dropout, batchnorm, etc.)
        model.train()  # Enable training mode (dropout, batchnorm, etc.)
        
        # Process all training batches
        for batch_idx, (noisy_batch, clean_batch, cube_params, cube_vels) in enumerate(train_loader):
            
            # Reset gradients from previous iteration
            optimizer.zero_grad()
            
            # Move data tensors to computational device
            # noisy_batch: Input IFU cubes with realistic noise characteristics
            # clean_batch: Ground truth clean cubes for supervised learning
            noisy_batch = noisy_batch.to(device)  # Shape: [batch_size, 1, depth, height, width]
            clean_batch = clean_batch.to(device)  # Shape: [batch_size, 1, depth, height, width]
            
            # ---- Forward Pass ----
            # U-Net prediction: noisy cube → denoised cube
            denoised_batch = model(noisy_batch)
            
            # ---- Loss Calculation ----
            # MSE between predicted denoised and ground truth clean cubes
            # Penalizes pixel-wise reconstruction errors across all spectral channels
            loss = criterion(denoised_batch, clean_batch)
            
            # ---- Backward Pass and Optimization ----
            loss.backward()        # Compute gradients via backpropagation
            optimizer.step()       # Update model parameters using gradients
            
            # ---- Loss Accumulation for Epoch Averaging ----
            batch_size_current = noisy_batch.size(0)  # Handle potential incomplete final batch
            train_loss_sum += loss.item() * batch_size_current  # Weight by batch size
            total_samples += batch_size_current
        
        # Calculate epoch training loss (average over all samples)
        train_loss = train_loss_sum / total_samples
        
        # ================================================================
        # VALIDATION PHASE
        # ================================================================
        
        valid_loss_sum = 0.0   # Reset validation loss accumulator
        total_samples = 0      # Reset sample counter for validation
        
        model.eval()  # Enable evaluation mode (disable dropout, etc.)
        
        # Disable gradient computation for faster validation and memory efficiency
        with torch.no_grad():
            for noisy_batch, clean_batch, cube_params, cube_vels in val_loader:
                
                # Move validation data to device
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)
                
                # ---- Validation Forward Pass ----
                # Generate denoised predictions on validation set
                denoised_cube = model(noisy_batch)
                
                # ---- Validation Loss Calculation ----
                # Same MSE metric as training for consistent monitoring
                loss = criterion(denoised_cube, clean_batch)
                
                # Accumulate validation loss with proper weighting
                batch_size_current = noisy_batch.size(0)
                valid_loss_sum += loss.item() * batch_size_current
                total_samples += batch_size_current
        
        # Calculate epoch validation loss (average over all validation samples)
        valid_loss = valid_loss_sum / total_samples

        # Calculate epoch validation loss (average over all validation samples)
        valid_loss = valid_loss_sum / total_samples
        
        # ================================================================
        # MODEL CHECKPOINTING AND PROGRESS LOGGING
        # ================================================================
        
        # Save model weights when validation loss improves
        # This prevents overfitting by storing the best-generalizing model
        if valid_loss < min_valid_loss:
            # Update best validation loss tracker
            min_valid_loss = valid_loss
            
            # Save model state dict (weights and biases only, not architecture)
            fweights = os.path.join(dir_wt, "weights_best.pt")
            torch.save(model.state_dict(), fweights)
            
            # Log improvement with visual indicator
            print(f"Epoch [{iteration+1:3d}/{iterations}], "
                  f"Train Loss: {train_loss:.5e}, "
                  f"Valid Loss: {valid_loss:.5e} [SAVED]")
        else:
            # Log progress without saving
            print(f"Epoch [{iteration+1:3d}/{iterations}], "
                  f"Train Loss: {train_loss:.5e}, "
                  f"Valid Loss: {valid_loss:.5e}")
        
        # ================================================================
        # TRAINING HISTORY LOGGING
        # ================================================================
        
        # Log detailed training metrics to text file for analysis
        # Format: epoch_number train_loss validation_loss
        floss = os.path.join(dir_wt, "losses.txt")
        with open(floss, "a") as f:
            f.write(f"{iteration} {train_loss:.5e} {valid_loss:.5e}\n")
    
    print(f'\n(*) Training completed!')
    print(f'(*) Best validation loss achieved: {min_valid_loss:.5e}')
    print(f'(*) Best model weights saved to: {os.path.join(dir_wt, "weights_best.pt")}')
    
    return min_valid_loss


# ========================================================================================
# MODEL ARCHITECTURE SETUP AND TRAINING EXECUTION
# ========================================================================================

print('(*) Configuring 3D U-Net architecture...')

# ---- Base U-Net Architecture ----
# Create core 3D U-Net with single input channel and configurable filter count
# n_channels=1: Single intensity channel (not RGB or multi-band)
# filters=16: Number of filters in the first convolutional layer
#            Subsequent layers follow standard U-Net doubling pattern: 16→32→64→128...
base_model = UNet3D(n_channels=1, filters=16)

print(f'(*) Base U-Net created with {sum(p.numel() for p in base_model.parameters())} parameters')

# ---- Padding Wrapper for Variable Input Sizes ----
# Wrap base model to handle input cubes of different sizes
# target_shape=(48, 96, 96): [depth, height, width] for padded processing
# The wrapper automatically:
# 1. Pads smaller inputs to target_shape for consistent processing
# 2. Crops outputs back to original input dimensions
# 3. Enables training on cubes smaller than network's expected input size
model = UNet3DWithPadCrop(base_model, target_shape=(48, 96, 96))

print(f'(*) Created padded wrapper for target shape: (48, 96, 96)')
print(f'(*) Model ready for variable input sizes with automatic padding/cropping')

# ---- Training Configuration ----
n_training_epochs = 200  # Number of complete passes through training dataset

print(f'\n(*) Starting training with configuration:')
print(f'    - Training epochs: {n_training_epochs}')
print(f'    - Batch size: {batch_size}')
print(f'    - Dataset size: {len(train_dataset)} training samples')
print(f'    - Validation size: {len(valid_dataset)} samples')
print(f'    - Device: {device}')

# ---- Execute Training Pipeline ----
min_valid_loss = fit(
    iterations=n_training_epochs,
    model=model,
    train_loader=train_loader,
    val_loader=valid_loader,
    device=device
)

# ========================================================================================
# TRAINING COMPLETION AND SUMMARY
# ========================================================================================

print('\n' + '='*70)
print('TRAINING PIPELINE COMPLETED SUCCESSFULLY')
print('='*70)
print(f'(*) Neural network training finished!')
print(f'(*) Best validation loss: {min_valid_loss:.5e}')
print(f'(*) Model weights saved to: {dir_wt}/weights_best.pt')
print(f'(*) Training history logged to: {dir_wt}/losses.txt')
print(f'(*) Ready for evaluation on test set or real observations')
print('='*70)