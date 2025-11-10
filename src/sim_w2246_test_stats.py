"""
Unified Denoising Statistics Pipeline for IFU Data Analysis

This script combines the functionality of sim_stats.py and w2246_stats.py into a single
configurable pipeline that can process both synthetic simulation data and real observational
data (like W2246) using the same denoising algorithms and evaluation metrics.

Key Features:
- Simple configuration via variables at top of file
- Unified U-Net and IST wavelet denoising pipeline
- Consistent performance evaluation metrics across data types
- Flexible noise modeling for different data characteristics
- Standardized output format for comparative analysis

Usage:
    1. Edit the configuration section below
    2. Run: python unified_stats.py

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from functions import * 
import scipy
import pickle
from wavelet_denoising import *
from u_net_model import *
import os
from sklearn.metrics import mean_squared_error
from spectral_cube import SpectralCube
import importlib

# ========================================================================================
# CONFIGURATION SECTION - EDIT THESE PARAMETERS
# ========================================================================================

# Data source configuration
data_type = 'sim'  # 'sim' for simulation data, 'obs' for observational data

# File paths
if data_type == 'sim':
    input_path = '/home/alahiry/data/mock_ifu/processed_cubes/clean_cube.npy'
else:
    input_path = '/home/alahiry/data/obvs_data/W2246_C2_125.fits'

# Model configuration
model_weights = '/home/alahiry/deep_learning/denoise_comparison/final/x72_n20000_64_filters_batch_16/weights_best.pt'

# Processing parameters
n_samples_per_bin = 50 if data_type == 'sim' else 200
beam_width_px = 3.75
device = 'cpu'

# Observational data specific parameters (only used when data_type == 'obs')
obs_crop_params = (195, 395, 200, 400)  # xlo, xhi, ylo, yhi
obs_mean = 1.5502414918156043e-06
obs_std = 0.0001832498356025477

# Output configuration
output_suffix = ''  # Additional suffix for output files

# ========================================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# ========================================================================================

def load_simulation_data(input_path):
    """
    Load and preprocess simulation data from .npy file.
    
    Parameters:
    -----------
    input_path : str
        Path to the .npy file containing clean simulation cube
        
    Returns:
    --------
    clean_cube : np.ndarray
        Preprocessed clean cube ready for analysis
    """
    print(f'(*) Loading simulation data from: {input_path}')
    
    # Load clean cube from numpy file
    clean_cube = np.load(input_path)
    
    # Apply simulation-specific zoom factors and rotation
    zoom_factors = (48/200, 72/500, 72/500)
    clean_cube = np.rot90(scipy.ndimage.zoom(clean_cube, zoom=zoom_factors, order=0), k=2, axes=[1,2])
    
    print(f'(*) Simulation cube processed to shape: {clean_cube.shape}')
    return clean_cube

def load_observational_data(input_path, crop_params):
    """
    Load and preprocess observational data from .fits file.
    
    Parameters:
    -----------
    input_path : str
        Path to the .fits file containing observational cube
    crop_params : tuple
        Crop parameters: (xlo, xhi, ylo, yhi)
        
    Returns:
    --------
    clean_cube : np.ndarray
        Preprocessed observational cube ready for analysis
    """
    print(f'(*) Loading observational data from: {input_path}')
    
    # Parse crop parameters
    xlo, xhi, ylo, yhi = crop_params
    print(f'(*) Crop parameters: x=[{xlo}:{xhi}], y=[{ylo}:{yhi}]')
    
    # Load the spectral cube
    cube = SpectralCube.read(input_path)
    
    # Crop the cube spatially
    cropped_cube = cube.subcube(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi)
    
    # Convert to numpy array
    clean_cube = cropped_cube.unmasked_data[:].value
    
    # Apply observational-specific zoom factors and rotation
    zoom_factors = (36/125, 72/200, 72/200)
    clean_cube = np.rot90(scipy.ndimage.zoom(clean_cube, zoom=zoom_factors, order=3), k=2, axes=[1,2])
    
    print(f'(*) Observational cube processed to shape: {clean_cube.shape}')
    return clean_cube

# ========================================================================================
# NOISE GENERATION FUNCTIONS
# ========================================================================================

def add_noise_to_target_snr(spectral_cube, target_snr, signal_mask=None):
    """
    Add beam-correlated noise to reduce peak SNR of the input cube to target_snr.
    
    Parameters:
    -----------
    spectral_cube : np.ndarray
        Input spectral cube
    target_snr : float
        Desired final peak SNR
    signal_mask : np.ndarray or None
        True where signal is present, False for noise background
    
    Returns:
    --------
    noisy_cube : np.ndarray
        Cube with additional noise added
    """
    # Get peak signal
    peak_flux = np.max(spectral_cube)

    # Measure original RMS from noise-only region
    if signal_mask is not None:
        background = spectral_cube[~signal_mask]
    else:
        raise ValueError("You must provide a signal mask to estimate the current noise level.")

    sigma_orig = np.std(background)
    sigma_target = peak_flux / target_snr

    # Compute additional noise required
    if sigma_target < sigma_orig:
        raise ValueError("Target SNR is higher than current SNR. Cannot reduce noise this way.")
    
    sigma_add = np.sqrt(sigma_target**2 - sigma_orig**2)

    # Generate and convolve new noise
    white_noise = np.random.normal(0, 1.0, spectral_cube.shape)
    convolved_noise = convolve_beam(white_noise, 8.932458162307025*(72/200), 7.1870291233057*(72/200), -66.1529)

    # Normalize and scale
    current_rms = np.std(convolved_noise) 
    scaled_noise = convolved_noise * (sigma_add / current_rms)

    # Add to original cube
    noisy_cube = spectral_cube + scaled_noise

    return noisy_cube

def get_background_mask(cube_shape):
    """
    Create a mask where background (noise-only) region is True.
    
    Parameters:
    -----------
    cube_shape : tuple
        Shape of the cube (ns, nx, ny)
    
    Returns:
    --------
    mask : np.ndarray
        Mask with True where background
    """
    ns, nx, ny = cube_shape
    mask = np.zeros((ns, nx, ny), dtype=bool)
    mask[:, 0:22, 50:72] = True
    return mask

# ========================================================================================
# MODEL SETUP AND NORMALIZATION
# ========================================================================================

def setup_model(model_weights_path, device, data_type):
    """
    Setup the U-Net model with appropriate configuration.
    
    Parameters:
    -----------
    model_weights_path : str
        Path to trained model weights
    device : torch.device
        Computing device
    data_type : str
        Type of data ('sim' or 'obs')
        
    Returns:
    --------
    model : torch.nn.Module
        Loaded and configured model
    """
    print(f'(*) Setting up U-Net model for {data_type} data...')
    
    # Create base model
    base_model = UNet3D(n_channels=1, filters=16)
    
    # Configure target shape based on data type
    if data_type == 'sim':
        target_shape = (48, 80, 80)
    else:  # obs
        target_shape = (48, 80, 80)
    
    # Wrap with padding
    model = UNet3DWithPadCrop(base_model, target_shape=target_shape).to(device)
    
    # Load weights
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
        print(f"(*) Model weights loaded from: {model_weights_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")
    
    model.eval()
    return model

def get_normalization_params(data_type, mean=None, std=None):
    """
    Get normalization parameters based on data type.
    
    Parameters:
    -----------
    data_type : str
        Type of data ('sim' or 'obs')
    mean, std : float
        For obs data, normalization values
        
    Returns:
    --------
    tuple : (mean, std) normalization parameters
    """
    if data_type == 'sim':
        # For simulation data, use dynamic normalization
        return None, None
    else:
        # For observational data, use fixed normalization
        return mean, std

# ========================================================================================
# DENOISING AND EVALUATION FUNCTIONS
# ========================================================================================

def apply_unet_denoising(noisy_cube, model, device, data_type, mean=None, std=None):
    """
    Apply U-Net denoising with appropriate normalization.
    
    Parameters:
    -----------
    noisy_cube : np.ndarray
        Noisy input cube
    model : torch.nn.Module
        Trained U-Net model
    device : torch.device
        Computing device
    data_type : str
        Type of data ('sim' or 'obs')
    mean, std : float
        Normalization parameters (for obs data)
        
    Returns:
    --------
    denoised_unet : np.ndarray
        Denoised cube from U-Net
    """
    if data_type == 'sim':
        # Dynamic normalization for simulation data
        normalized_cube = (noisy_cube - noisy_cube.mean()) / noisy_cube.std()
        cube_mean, cube_std = noisy_cube.mean(), noisy_cube.std()
    else:
        # Fixed normalization for observational data
        normalized_cube = (noisy_cube - mean) / std
        cube_mean, cube_std = mean, std
    
    # Prepare tensor
    test = torch.tensor(
        np.expand_dims(np.expand_dims(normalized_cube, axis=0), axis=0),
        dtype=torch.float32
    ).to(device)

    # Apply denoising
    with torch.no_grad():
        denoised_tensor = model(test)
    
    # Denormalize
    denoised_unet = ((denoised_tensor * cube_std) + cube_mean)[0, 0].cpu().numpy()
    
    return denoised_unet

def calculate_metrics(denoised_cube, clean_cube, noisy_cube, mask, data_type):
    """
    Calculate performance metrics with data-type-specific formulations.
    
    Parameters:
    -----------
    denoised_cube : np.ndarray
        Denoised cube
    clean_cube : np.ndarray
        Clean reference cube
    noisy_cube : np.ndarray
        Noisy input cube
    mask : np.ndarray
        Evaluation mask
    data_type : str
        Type of data ('sim' or 'obs')
        
    Returns:
    --------
    tuple : (flux_sum_metric, rmse_metric)
    """
    if data_type == 'sim':
        # Simulation metrics: percentage relative to clean
        sum_metric = (1 + ((np.sum(denoised_cube[mask]) - np.sum(clean_cube[mask])) / 
                          np.sum(clean_cube[mask]))) * 100
        
        # Full-cube RMSE normalized by noisy RMSE
        rmse_noisy = np.sqrt(np.mean((noisy_cube - clean_cube) ** 2))
        rmse_metric = np.sqrt(np.mean((denoised_cube - clean_cube) ** 2)) / rmse_noisy
        
    else:  # obs
        # Observational metrics: direct percentage and masked RMSE
        sum_metric = 100 * np.sum(mask * denoised_cube) / np.sum(mask * clean_cube)
        
        # Masked RMSE using sklearn
        rmse_noisy = np.sqrt(mean_squared_error((mask * noisy_cube).ravel(), (mask * clean_cube).ravel()))
        rmse_metric = np.sqrt(mean_squared_error((mask * denoised_cube).ravel(), (mask * clean_cube).ravel())) / rmse_noisy
    
    return sum_metric, rmse_metric

# ========================================================================================
# MAIN PROCESSING PIPELINE
# ========================================================================================

def main():
    print("=" * 80)
    print("UNIFIED IFU DENOISING STATISTICS PIPELINE")
    print("=" * 80)

    # Setup device
    device_obj = torch.device(device)
    print(f'(*) Using device: {device_obj}')

    # Load data based on type
    if data_type == 'sim':
        clean_cube = load_simulation_data(input_path)
        mask_params = (3, beam_width_px)
    else:
        clean_cube = load_observational_data(input_path, obs_crop_params)
        mask_params = (5, beam_width_px)

    # Setup model and normalization
    model = setup_model(model_weights, device_obj, data_type)
    mean, std = get_normalization_params(data_type, obs_mean, obs_std)

    # Create evaluation mask
    mask_clean = create_circular_aperture_mask(clean_cube, mask_params[0], mask_params[1])

    # Initialize statistics storage
    denoised_stats_unet = {'sum_mean': [], 'sum_sem': [], 'rmse_mean': [], 'rmse_sem': []}
    denoised_stats_ist = {'sum_mean': [], 'sum_sem': [], 'rmse_mean': [], 'rmse_sem': []}

    # Configure SNR bins
    if data_type == 'sim':
        snr_bins = [(2.5, 3), (3, 3.5), (3.5, 4), (4, 4.5), (5, 6), (6, 8), (8, 10)]
    else:
        snr_bins = [(2.5, 3), (3, 3.5), (3.5, 4), (4, 5), (5, 6), (6, 8), (8, 10)]

    # Initialize IST denoiser
    denoiser3d_soft = Denoiser2D1D(threshold_type='soft', verbose=False, plot=False)

    print(f'(*) Processing {len(snr_bins)} SNR bins with {n_samples_per_bin} samples each...')

    # Main evaluation loop
    for bin_idx, (snr_min, snr_max) in enumerate(snr_bins):
        print(f'(*) Processing SNR bin {bin_idx+1}/{len(snr_bins)}: [{snr_min}, {snr_max}]')
        
        sampled_snrs = np.random.uniform(snr_min, snr_max, n_samples_per_bin)
        sum_vals_unet, rmses_unet = [], []
        sum_vals_ist, rmses_ist = [], []

        for sample_idx, snr in enumerate(sampled_snrs):
            if sample_idx % 10 == 0:
                print(f'    Sample {sample_idx+1}/{n_samples_per_bin}, SNR={snr:.2f}')
            
            # Generate noisy cube
            if data_type == 'sim':
                noisy_cube = apply_and_convolve_noise(clean_cube, snr, beam_width_px, beam_width_px, 0)
            else:
                noisy_cube = add_noise_to_target_snr(clean_cube, snr, signal_mask=get_background_mask(clean_cube.shape))
            
            # U-Net denoising
            denoised_unet = apply_unet_denoising(noisy_cube, model, device_obj, data_type, mean, std)
            
            # IST denoising
            denoised_ist, *_ = denoiser3d_soft(
                noisy_cube, clean_cube,
                threshold_level=5, method='iterative', emission_mask=mask_clean,
                num_iter_reweight=100, num_iter_debias=100,
                num_scales_2d=None, num_scales_1d=None, noise_cube=None
            )
            
            # Calculate metrics
            sum_unet, rmse_unet = calculate_metrics(denoised_unet, clean_cube, noisy_cube, mask_clean, data_type)
            sum_ist, rmse_ist = calculate_metrics(denoised_ist, clean_cube, noisy_cube, mask_clean, data_type)
            
            # Store results
            sum_vals_unet.append(sum_unet)
            rmses_unet.append(rmse_unet)
            sum_vals_ist.append(sum_ist)
            rmses_ist.append(rmse_ist)

        # Calculate bin statistics
        for stats, sums, rmses in zip(
            [denoised_stats_unet, denoised_stats_ist],
            [sum_vals_unet, sum_vals_ist],
            [rmses_unet, rmses_ist]
        ):
            stats['sum_mean'].append(np.mean(sums))
            stats['sum_sem'].append(np.std(sums) / np.sqrt(n_samples_per_bin))
            stats['rmse_mean'].append(np.mean(rmses))
            stats['rmse_sem'].append(np.std(rmses) / np.sqrt(n_samples_per_bin))
        
        print(f'    Bin completed - U-Net flux: {np.mean(sum_vals_unet):.2f}%, RMSE: {np.mean(rmses_unet):.3f}')

    # Save results with appropriate filenames
    suffix = f"_{data_type}{output_suffix}" if output_suffix else f"_{data_type}"
    if data_type == 'sim':
        suffix += "_1"
    else:
        suffix += f"_{n_samples_per_bin}"

    unet_filename = f'denoised_stats_unet{suffix}.pkl'
    ist_filename = f'denoised_stats_ist{suffix}.pkl'

    with open(unet_filename, 'wb') as f:
        pickle.dump(denoised_stats_unet, f)

    with open(ist_filename, 'wb') as f:
        pickle.dump(denoised_stats_ist, f)

    print(f'\n(*) Processing completed!')
    print(f'(*) Results saved to:')
    print(f'    U-Net: {unet_filename}')
    print(f'    IST: {ist_filename}')
    print("=" * 80)


if __name__ == "__main__":
    main()