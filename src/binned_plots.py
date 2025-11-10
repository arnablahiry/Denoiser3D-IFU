#!/usr/bin/env python
"""
IFU Denoising Algorithm Benchmarking and Performance Analysis

This script provides a comprehensive benchmarking framework for evaluating multiple 
denoising algorithms on synthetic IFU (Integral Field Unit) spectral cube datasets. 
The analysis focuses on performance across different source sizes and signal-to-noise 
ratios to understand algorithm behavior in various observational regimes.

Key Features
-----------
- Multi-algorithm comparison: U-Net, IST (Iterative Soft Thresholding), PCA, ICA
- Binned analysis by source size (resolved vs unresolved) and SNR levels
- Statistical performance metrics including flux conservation and RMSE
- Systematic sampling across parameter space for robust evaluation
- Comprehensive data management and result serialization

Scientific Context
-----------------
This benchmarking addresses critical questions in astronomical data processing:
1. How do denoising algorithms perform on sources of different sizes relative to beam?
2. What is the SNR dependence of each algorithm's effectiveness?
3. How well do algorithms preserve photometric accuracy (flux conservation)?
4. Which methods provide optimal noise reduction vs artifact introduction trade-offs?

Applications
-----------
- Algorithm selection for specific observational datasets
- Performance validation for new denoising methods
- Understanding systematic biases in different approaches
- Optimization of denoising parameters for astronomical workflows
"""

# ===================================================================
# CORE LIBRARY IMPORTS
# ===================================================================

# Numerical computation and data handling
import numpy as np
import pickle
import pandas as pd

# Machine learning and data processing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error

# Astronomical data analysis
from astrodendro import Dendrogram
import matplotlib.pyplot as plt

# System utilities
import os

# ===================================================================
# DENOISING ALGORITHM IMPORTS
# ===================================================================

# Deep learning approaches
from u_net_model import *

# Traditional signal processing methods
from pca_denoising import *
from ica_denoising import *

# Wavelet-based methods
from wavelet_denoising import *

# Dataset and utility functions
from toy_cube_dataset import *
from functions import *

# ===================================================================
# DATASET CONFIGURATION AND LOADING
# ===================================================================

# Define dataset parameters for consistent analysis
final_grid_size = 96      # Spatial resolution (pixels per side)
n_spectral_slices = 40    # Number of spectral channels per cube
n_cubes = 20000          # Total number of synthetic cubes in dataset

# Load the complete synthetic dataset
print("Loading the entire dataset")

# Construct filepath for the specific dataset configuration
fname = f'/Users/arnablahiry/repos/3D_IFU_Denoising/data/toy_cubes/datasets/final/final_dataset_{n_spectral_slices}_{final_grid_size}_{n_cubes}.pkl'

# Load pickled dataset containing synthetic IFU cubes with various SNR and source sizes
with open(fname, "rb") as file:
    dataset = pickle.load(file)

# Extract dataset normalization statistics for proper scaling
mean_dataset, std_dataset = dataset.return_stats()


print(f'(*) Whole data set ({len(dataset)} cubes) loaded')

# ===================================================================
# DATASET SPLITTING FOR EVALUATION
# ===================================================================

print('(*) Splitting dataset into training, validation and test set (80:10:10)')

# Set random seed for reproducible dataset splits
generator = torch.Generator().manual_seed(42)

# Define split proportions following standard ML practices
train_size = int(0.8 * len(dataset))  # 80% for training (not used in this script)
valid_size = int(0.1 * len(dataset))  # 10% for validation (not used in this script)  
test_size = len(dataset) - train_size - valid_size  # 10% for testing - ensure full coverage

# Perform random dataset split with reproducible seeding
# Note: Only test_dataset is used in this benchmarking script
_, _, test_dataset = random_split(dataset, [train_size, valid_size, test_size], generator=generator)

# ===================================================================
# DATA LOADER SETUP FOR BATCH PROCESSING
# ===================================================================

# Create data loader for efficient batch processing during inference
# Batch size of 32 balances memory usage with computational efficiency
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===================================================================
# U-NET MODEL CONFIGURATION AND LOADING
# ===================================================================

# Configure compute device (GPU preferred for deep learning inference)
device = torch.device('cuda:0')

# Initialize base U-Net architecture with specified parameters
# 1 input channel (single spectral cube), 16 filters for feature extraction
base_model = UNet3D(n_channels=1, filters=16)

# Wrap base model with padding/cropping layer for consistent input/output shapes
# Target shape (48, 96, 96) matches expected spectral cube dimensions
model = UNet3DWithPadCrop(base_model, target_shape=(48, 96, 96)).to(device)

# Define batch size for consistent processing
batch_size = 32

# Path to pre-trained model weights from CRISTAL dataset training
fweights_best = '/home/alahiry/deep_learning/denoise_comparison/FINAL_CRISTAL/x96_n20000_16_filters_batch_16/weights_best.pt'

# Load pre-trained weights if available
if os.path.exists(fweights_best):
    # Load state dict with proper device mapping and safety checks
    model.load_state_dict(torch.load(fweights_best, map_location=device, weights_only=True))
    print("\nModel and weights loaded!")
else:
    # Raise informative error if weights not found
    raise FileNotFoundError(f"Model weights not found at {fweights_best}. Please check the path or train the model first.")

# Set model to evaluation mode (disables dropout, batch norm updates)
model.eval()

generator = torch.Generator().manual_seed(42)

train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size  # Ensure full dataset coverage

# Randomly split dataset
_, _, test_dataset = random_split(dataset, [train_size, valid_size, test_size], generator=generator)


test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)





# Setup model
device = torch.device('cuda:0')
base_model = UNet3D(n_channels=1, filters=16)

model = UNet3DWithPadCrop(base_model, target_shape=(48, 96, 96)).to(device)

batch_size=32
fweights_best = '/home/alahiry/deep_learning/denoise_comparison/FINAL_CRISTAL/x96_n20000_16_filters_batch_16/weights_best.pt' #dir_wt + '/weights_best.pt'

if os.path.exists(fweights_best):
    model.load_state_dict(torch.load(fweights_best, map_location=device, weights_only=True))
    print("\nModel and weights loaded!")

else:
    raise FileNotFoundError(f"Model weights not found at {fweights_best}. Please check the path or train the model first.")

model.eval()

# ===================================================================
# U-NET INFERENCE AND PERFORMANCE ANALYSIS
# ===================================================================

# Initialize results storage for comprehensive performance tracking
results = []

print("Performing U-net denoising for all the cubes in the test set (batch-wise)")

# Storage for effective radius statistics
a = []

# Perform inference without gradient computation for efficiency
with torch.no_grad():
    # Process test data in batches for memory efficiency
    for noisy_batch, clean_batch, cube_params, cube_vels in test_loader:
        # Move data to GPU for accelerated inference
        noisy_batch = noisy_batch.to(device)
        clean_batch = clean_batch.to(device)
        
        # Apply U-Net denoising model to noisy input
        denoised_batch = model(noisy_batch)

        # Convert tensors to numpy arrays for analysis
        # Remove singleton channel dimension with squeeze(1)
        noisy_np = noisy_batch.cpu().numpy().squeeze(1)
        clean_np = clean_batch.cpu().numpy().squeeze(1)
        denoised_np = denoised_batch.cpu().numpy().squeeze(1)

        # Process each cube in the current batch individually
        for i in range(noisy_np.shape[0]):
            
            # Find location of peak emission in clean cube for reference
            max_voxel_index = np.argmax(clean_np[i])  # Get flattened index
            max_channel, max_y, max_x = np.unravel_index(max_voxel_index, clean_np[i].shape)  # Convert to 3D coordinates
            
            # Extract cube parameters from dataset metadata
            peak_snr = cube_params[i][0].item()  # Peak signal-to-noise ratio
            R_e = 5/cube_params[i][1].item()     # Effective radius in pixels (inverted scaling)
            
            # Store effective radius for statistical analysis
            a.append(R_e)

            # Denormalize cubes using dataset statistics for physical interpretation
            # Convert from normalized values back to original flux units
            noisy_cube = (noisy_np[i]*std_dataset)+mean_dataset
            clean_cube = (clean_np[i]*std_dataset)+mean_dataset  
            denoised_unet_cube = (denoised_np[i]*std_dataset)+mean_dataset

            # Create circular aperture mask for photometric analysis
            # Mask radius based on source effective radius and beam size (3.75 pixels)
            mask = create_circular_aperture_mask(noisy_cube, R_e, 3.75)

            # Calculate total flux within aperture for each cube version
            total_flux_noisy = np.sum(mask*noisy_cube)            # Noisy observation
            total_flux_clean = np.sum(mask*clean_cube)            # Ground truth  
            total_flux_unet_denoised = np.sum(mask*denoised_unet_cube)  # U-Net result

            # Compute comprehensive performance statistics for noisy input
            stats_noisy = compute_flux_residual_rmse_stats(noisy_cube, clean_cube, mask)
            
            # Compute comprehensive performance statistics for U-Net denoised result
            stats_denoised = compute_flux_residual_rmse_stats(denoised_unet_cube, clean_cube, mask)

            # Package all results for systematic analysis
            result = {
                "peak_snr": peak_snr,                    # Input SNR level
                "R_e": R_e,                              # Source effective radius
                "clean_cube": clean_cube,                # Ground truth cube
                "noisy_cube": noisy_cube,                # Noisy input cube
                "clean_mask": mask,                      # Aperture mask for analysis
                "unet_denoised_cube": denoised_unet_cube,  # U-Net denoised result
            }

            # Add to results collection for binned analysis
            results.append(result)

# ===================================================================
# DATAFRAME CONSTRUCTION AND BINNING ANALYSIS
# ===================================================================

# Convert results list to pandas DataFrame for systematic analysis
df = pd.DataFrame(results)

# Define instrumental beam size for resolution analysis
beam_size = 3.75  # pixels (FWHM of synthesized beam)

# ===================================================================
# SOURCE SIZE CLASSIFICATION RELATIVE TO BEAM
# ===================================================================

# Compute ratio of source diameter to beam size for resolution classification
# Factor of 2 converts effective radius to diameter for proper comparison
df["de_beam_ratio"] = 2 * df["R_e"] / beam_size

# Initialize beam size bin column for classification
df["beam_size_bin"] = None

# Classify sources as resolved or unresolved based on size relative to beam
# Criterion: source diameter ≤ beam size → unresolved, otherwise resolved
df.loc[df["de_beam_ratio"] <= 1, "beam_size_bin"] = "Unresolved"  # Beam-limited sources
df.loc[df["de_beam_ratio"] > 1, "beam_size_bin"] = "Resolved"     # Spatially resolved sources

# Convert to ordered categorical for consistent analysis and plotting
df["beam_size_bin"] = pd.Categorical(
    df["beam_size_bin"],
    categories=["Unresolved", "Resolved"],  # Logical ordering from small to large
    ordered=True
)

# Extract ordered categories for systematic iteration
sorted_beam_bins = df["beam_size_bin"].cat.categories

# ===================================================================
# SIGNAL-TO-NOISE RATIO BINNING
# ===================================================================

# Define SNR bin edges for systematic performance analysis
# Covers range from challenging low-SNR to high-quality observations
bin_edges = [2.5, 3, 3.5, 4, 5, 6, 8, 10]  # Customize based on data distribution

# Calculate bin centers for plotting and analysis
bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

# Create SNR bins using pandas cut function
# This assigns each cube to an SNR range for systematic comparison
df["peak_snr_bin"] = pd.cut(df["peak_snr"], bins=bin_edges)

# Initialize storage for sampled datasets from each bin
sampled_dfs = []

# Remove rows with missing bin assignments to ensure clean analysis
df_valid = df.dropna(subset=["beam_size_bin", "peak_snr_bin"])


# ===================================================================
# DENOISING ALGORITHM WRAPPER FUNCTIONS
# ===================================================================

def IST_wrapper(row, threshold=5):
    """
    Wrapper function for Iterative Soft Thresholding (IST) 2D-1D wavelet denoising.
    
    This function applies the IST algorithm using 2D-1D wavelet decomposition
    specifically designed for spectral cube denoising. The method performs
    iterative soft thresholding with reweighting and debiasing steps to
    optimize noise reduction while preserving source structure.
    
    Parameters
    ----------
    row : pandas.Series
        DataFrame row containing cube data with keys:
        - "noisy_cube": Input noisy spectral cube
        - "clean_cube": Ground truth for validation
        - "clean_mask": Emission region mask for analysis
    threshold : float, optional
        Threshold level for wavelet coefficient selection, by default 5
        Higher values → more aggressive denoising, lower values → more conservative
        
    Returns
    -------
    numpy.ndarray
        Denoised spectral cube with same dimensions as input
        
    Notes
    -----
    - Uses soft thresholding for smooth coefficient transitions
    - Applies 100 iterations each for reweighting and debiasing
    - Automatic scale selection for optimal multi-scale representation
    - Emission mask guides algorithm focus on scientifically relevant regions
    """
    
    # Initialize IST denoiser with soft thresholding for smooth transitions
    denoiser3d_soft = Denoiser2D1D(threshold_type='soft', verbose = True, plot = False)

    # Apply denoising with comprehensive parameter set
    results = denoiser3d_soft(row["noisy_cube"],          # Input noisy data
                              row["clean_cube"],          # Ground truth reference
                              threshold_level=threshold,  # Threshold for wavelet coefficients
                              method='iterative',         # Use iterative optimization
                              emission_mask=row["clean_mask"],  # Focus on emission regions
                              num_iter_reweight = 100,    # Reweighting iterations for optimization
                              num_iter_debias=100,        # Debiasing iterations to reduce artifacts
                              num_scales_2d=None,         # Auto-select 2D scales
                              num_scales_1d=None,         # Auto-select 1D scales  
                              noise_cube=None)            # No explicit noise model

    # Return denoised cube (first element of results tuple)
    return results[0]

def PCA_wrapper(row):
    """
    Wrapper function for Principal Component Analysis (PCA) based denoising.
    
    This function applies PCA denoising to remove noise while preserving
    the dominant spectral and spatial structures in the cube. The method
    decomposes the data into principal components and reconstructs using
    only the most significant components containing signal information.
    
    Parameters
    ----------
    row : pandas.Series
        DataFrame row containing cube data with keys:
        - "noisy_cube": Input noisy spectral cube for processing
        - "clean_mask": Emission region mask to focus analysis
        
    Returns
    -------
    numpy.ndarray
        PCA-denoised spectral cube with noise-dominated components removed
        
    Notes
    -----
    - Identifies signal-dominated vs noise-dominated principal components
    - Reconstruction uses only components containing significant signal
    - Mask guides component selection toward emission regions
    - Preserves dominant spectral features while removing uncorrelated noise
    """
    
    # Apply PCA denoising with emission mask guidance
    results = pca_denoising(row["noisy_cube"],    # Input noisy spectral cube
                     mask=row["clean_mask"],      # Emission region mask for component selection
                     plot=False,                  # Suppress diagnostic plots
                     verbose=False)               # Suppress detailed output
    
    # Return denoised cube (first element of results tuple)
    return results[0]


def ICA_wrapper(row):
    """
    Wrapper function for Independent Component Analysis (ICA) based denoising.
    
    This function applies ICA to separate independent signal and noise
    components in spectral cubes. The method assumes that signal and noise
    have different statistical independence properties, allowing separation
    and reconstruction using only signal-dominated components.
    
    Parameters
    ----------
    row : pandas.Series
        DataFrame row containing cube data with keys:
        - "noisy_cube": Input noisy spectral cube for decomposition
        - "clean_mask": Emission region mask for component evaluation
        
    Returns
    -------
    numpy.ndarray
        ICA-denoised spectral cube with independent noise components removed
        
    Notes
    -----
    - Separates statistically independent signal and noise components
    - Reconstructs data using only signal-dominated independent components  
    - Mask helps identify components containing genuine emission features
    - Effective for removing noise with different statistical properties than signal
    """
    
    # Apply ICA denoising with emission region guidance
    results = ica_denoising(row["noisy_cube"],      # Input noisy spectral cube
                            mask=row["clean_mask"], # Emission region mask for analysis
                            plot=False,             # Suppress diagnostic plots
                            verbose=False)          # Suppress detailed output
    
    # Return denoised cube (first element of results tuple)
    return results[0]

def compute_stats_wrapper(row, mode):
    """
    Compute comprehensive performance statistics for denoising algorithm evaluation.
    
    This function calculates key metrics for assessing denoising algorithm
    performance, focusing on flux conservation (photometric accuracy) and
    noise reduction effectiveness (RMSE improvement) within emission regions.
    
    Parameters
    ----------
    row : pandas.Series
        DataFrame row containing cube data with keys:
        - "clean_cube": Ground truth spectral cube
        - "noisy_cube": Original noisy input cube  
        - "clean_mask": Emission region mask for focused analysis
        - f"{mode}_denoised_cube": Algorithm-specific denoised result
    mode : str
        Algorithm identifier ('unet', 'IST', 'PCA', 'ICA') to specify
        which denoised cube to analyze
        
    Returns
    -------
    tuple of float
        (flux_conservation, rmse_normalized) where:
        - flux_conservation: Percentage flux error relative to ground truth
        - rmse_normalized: RMSE improvement factor relative to input noise
        
    Notes
    -----
    Performance Metrics:
    - Flux Conservation: Measures photometric accuracy preservation
      Values near 100% indicate good flux preservation
      Deviations suggest systematic biases in the algorithm
      
    - Normalized RMSE: Quantifies noise reduction effectiveness  
      Values < 1.0 indicate improvement over input noise levels
      Lower values represent better denoising performance
      Normalization enables fair comparison across SNR levels
    """
    
    # Calculate total flux in emission regions for each cube version
    total_flux_denoised = np.sum(row['clean_mask']*row[f"{mode}_denoised_cube"])  # Algorithm result
    total_flux_clean =  np.sum(row['clean_mask']*row['clean_cube'])               # Ground truth
    
    # Compute flux conservation as percentage relative to ground truth
    # Formula: (1 + relative_error) × 100% 
    flux_conservation = (1+(total_flux_denoised - total_flux_clean)/total_flux_clean)*100

    # Calculate normalized RMSE for denoising performance assessment
    # Numerator: RMSE between denoised and ground truth (residual error)
    # Denominator: RMSE between noisy and ground truth (original noise level)
    rmse_masked = np.sqrt(mean_squared_error((row['clean_mask']*row['clean_cube']).ravel(),( row['clean_mask']*row[f"{mode}_denoised_cube"]).ravel()))/np.sqrt(mean_squared_error((row['clean_mask']*row['clean_cube']).ravel(),(row['clean_mask']*row["noisy_cube"]).ravel())) if row[f"{mode}_denoised_cube"].size > 0 or row["noisy_cube"].size > 0 else 0.0,  #stats_denoised['rmse_masked'],

    # Return performance metrics as tuple
    return (flux_conservation, rmse_masked)


# ===================================================================
# SYSTEMATIC BENCHMARKING ACROSS PARAMETER SPACE
# ===================================================================

# Loop over beam size bins (Unresolved and Resolved sources)
for beam_bin in df_valid["beam_size_bin"].cat.categories:
    # Filter data for current beam size category
    df_beam = df_valid[df_valid["beam_size_bin"] == beam_bin]
    
    # Loop over peak SNR bins for comprehensive parameter space coverage
    for snr_bin in df_beam["peak_snr_bin"].cat.categories:
        # Filter data for current SNR range within beam size category
        df_bin = df_beam[df_beam["peak_snr_bin"] == snr_bin]
        
        # Skip empty bins to avoid processing errors
        if len(df_bin) == 0:
            continue
        
        # ===================================================================
        # STATISTICAL SAMPLING FOR ROBUST EVALUATION
        # ===================================================================
        
        # Define sample size for statistical significance
        sample_n = 50  # Balance between statistical power and computational efficiency
        
        # Determine sampling strategy based on available data
        # Use replacement if insufficient data, otherwise sample without replacement
        replace_flag = len(df_bin) < sample_n
        
        # Perform stratified sampling with reproducible random seed
        sampled = df_bin.sample(n=sample_n, replace=replace_flag, random_state=42)

        # ===================================================================
        # APPLY ALL DENOISING ALGORITHMS TO SAMPLED DATA
        # ===================================================================

        # Apply IST wavelet denoising with threshold parameter
        sampled["IST_denoised_cube"] = sampled.apply(lambda row: IST_wrapper(row, threshold = 5), axis=1)        
        
        # Apply PCA-based denoising
        sampled["PCA_denoised_cube"] = sampled.apply(lambda row: PCA_wrapper(row), axis=1)
        
        # Apply ICA-based denoising  
        sampled["ICA_denoised_cube"] = sampled.apply(lambda row: ICA_wrapper(row), axis=1)

        # ===================================================================
        # COMPUTE PERFORMANCE METRICS FOR ALL ALGORITHMS
        # ===================================================================
        
        # Define all algorithms to evaluate systematically
        modes = ['unet', 'IST', 'PCA', 'ICA']

        # Calculate performance statistics for each denoising method
        for mode in modes:
            # Apply statistics wrapper to compute flux conservation and normalized RMSE
            # Results stored as separate columns for systematic comparison
            sampled[[f"flux_conservation_{mode}", f"rmse_norm_noisy_{mode}"]] = sampled.apply(lambda row: compute_stats_wrapper(row, mode), axis=1).apply(pd.Series)

        # Add processed sample to collection for final analysis
        sampled_dfs.append(sampled)

# ===================================================================
# FINAL DATA CONSOLIDATION AND EXPORT
# ===================================================================

# Combine all sampled bins back into a single comprehensive DataFrame
# This creates a balanced dataset across parameter space for analysis
df_sampled = pd.concat(sampled_dfs).reset_index(drop=True)

# Export processed results for further analysis and visualization
# Pickle format preserves all data types including numpy arrays
with open('/home/alahiry/codes/denoise_comparison/codes/final_dataset_stats_new_data.pkl', 'wb') as f:
    pickle.dump(df_sampled, f)
    