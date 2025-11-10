"""
3D U-Net Architecture for IFU Spectral Cube Denoising
=====================================================

This module implements a 3D U-Net convolutional neural network specifically designed 
for denoising Integral Field Unit (IFU) spectral cubes. The architecture handles 
3D data with dimensions (velocity, y, x) to preserve both spatial and spectral 
information during the denoising process.

Key Features:
- 3D convolutional layers for processing spectral-spatial data cubes
- Encoder-decoder architecture with skip connections
- Reflective padding to handle boundary conditions
- Configurable activation functions and filter sizes
- Wrapper class for handling variable input sizes with padding/cropping

Classes:
- UNet3D: Core 3D U-Net implementation
- UNet3DWithPadCrop: Wrapper for handling variable input shapes

Scientific Context:
IFU observations produce 3D data cubes where each voxel contains spectral information.
The 3D U-Net preserves correlations between neighboring voxels in all three dimensions,
making it ideal for denoising while maintaining:
- Spatial structure of galaxies
- Spectral line profiles  
- Velocity coherence across channels

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    """
    3D U-Net Convolutional Neural Network for spectral cube denoising.
    
    This implementation follows the classic U-Net architecture adapted for 3D data,
    featuring an encoder-decoder structure with skip connections to preserve 
    fine-scale features during the reconstruction process.
    
    Architecture Overview:
    - Encoder: 5 levels of downsampling with increasing channel depth
    - Bottleneck: Deepest representation with maximum channel depth
    - Decoder: 4 levels of upsampling with skip connections from encoder
    - Output: Single channel representing the denoised cube
    
    Parameters
    ----------
    n_channels : int
        Number of input channels (typically 1 for single spectral cube).
    filters : int, default 8
        Base number of convolutional filters. Doubles at each encoder level.
    pad : int, default 3
        Kernel size for convolutional operations (pad x pad x pad).
    last_act : str, default 'identity'
        Final activation function:
        - 'identity': No activation (linear output)
        - 'sigmoid': Sigmoid activation (0-1 output range)
        - 'softplus': Softplus activation (always positive)
        
    Network Architecture:
    --------------------
    Encoder Path (Contracting):
    - Level 1: input fearures (1) → filters (input processing)
    - Level 2: filters → 2×filters (feature extraction) 
    - Level 3: 2×filters → 4×filters (pattern recognition)
    - Level 4: 4×filters → 8×filters (high-level features)
    - Level 5: 8×filters → 16×filters (bottleneck)
    
    Decoder Path (Expanding):
    - Level 6: 16×filters → 8×filters (+ skip connection from level 4)
    - Level 7: 8×filters → 4×filters (+ skip connection from level 3)
    - Level 8: 4×filters → 2×filters (+ skip connection from level 2)
    - Level 9: 2×filters → filters (+ skip connection from level 1)
    - Output: filters → 1 channel (final reconstruction)
    
    Key Design Choices:
    ------------------
    - Reflective padding: Handles boundary conditions naturally for astronomical data
    - LeakyReLU activation: Prevents dead neurons and handles negative values
    - Average pooling: Reduces spatial/spectral dimensions while preserving information
    - Skip connections: Preserve fine details lost during downsampling
    - No bias terms: Reduces overfitting for limited training data
    
    Input/Output:
    ------------
    - Input: (batch_size, n_channels, depth, height, width)
    - Output: (batch_size, 1, depth, height, width)
    
    Examples
    --------
    >>> # Standard denoising model
    >>> model = UNet3D(n_channels=1, filters=16, pad=3, last_act='identity')
    >>> 
    >>> # Model for flux-positive outputs
    >>> model = UNet3D(n_channels=1, filters=8, pad=5, last_act='softplus')
    >>> 
    >>> # Forward pass
    >>> noisy_cube = torch.randn(1, 1, 40, 125, 125)  # Example shape
    >>> clean_cube = model(noisy_cube)
    
    Notes
    -----
    - Designed specifically for astronomical spectral cubes
    - Preserves spectral-spatial correlations crucial for IFU data analysis
    - Reflective padding respects boundary conditions in finite observations
    - Skip connections ensure preservation of emission line profiles
    """

    def __init__(self, n_channels, filters=8, pad=3, last_act='identity'):
        super(UNet3D, self).__init__()

        # =====================================================================
        # ENCODER PATH - CONTRACTING LAYERS
        # =====================================================================
        
        # Level 1: Initial feature extraction (input channels → base filters)
        self.conv1 = nn.Conv3d(n_channels, filters, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')
        self.conv1b = nn.Conv3d(filters, filters, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')

        # Level 2: Increased feature depth (filters → 2×filters)
        self.conv2 = nn.Conv3d(filters, filters*2, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')
        self.conv2b = nn.Conv3d(filters*2, filters*2, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')

        # Level 3: Mid-level feature extraction (2×filters → 4×filters)
        self.conv3 = nn.Conv3d(filters*2, filters*4, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')
        self.conv3b = nn.Conv3d(filters*4, filters*4, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')

        # Level 4: High-level feature representation (4×filters → 8×filters)
        self.conv4 = nn.Conv3d(filters*4, filters*8, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')
        self.conv4b = nn.Conv3d(filters*8, filters*8, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')

        # Level 5: Bottleneck - deepest representation (8×filters → 16×filters)
        self.conv5 = nn.Conv3d(filters*8, filters*16, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')
        self.conv5b = nn.Conv3d(filters*16, filters*16, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')

        # Pooling and activation layers used throughout encoder
        self.pool = nn.AvgPool3d(2)    # 2×2×2 average pooling for downsampling
        self.act = nn.LeakyReLU()      # LeakyReLU to handle negative flux values

        # =====================================================================
        # DECODER PATH - TRANSPOSE CONVOLUTIONS FOR UPSAMPLING  
        # =====================================================================
        
        # Transpose convolutions for learned upsampling (2×2×2 kernels, stride 2)
        self.transp1 = nn.ConvTranspose3d(filters*16, filters*8, kernel_size=2, stride=2, bias=False)
        self.transp2 = nn.ConvTranspose3d(filters*8, filters*4, kernel_size=2, stride=2, bias=False)
        self.transp3 = nn.ConvTranspose3d(filters*4, filters*2, kernel_size=2, stride=2, bias=False)
        self.transp4 = nn.ConvTranspose3d(filters*2, filters*1, kernel_size=2, stride=2, bias=False)

        # =====================================================================
        # DECODER PATH - CONTRACTING LAYERS WITH SKIP CONNECTIONS
        # =====================================================================
        
        # Level 6: First decoder level (skip from level 4: 16×filters → 8×filters)
        self.conv6_1 = nn.Conv3d(filters*16, filters*8, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')
        self.conv6_1b = nn.Conv3d(filters*8, filters*8, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')

        # Level 7: Second decoder level (skip from level 3: 8×filters → 4×filters)
        self.conv7_1 = nn.Conv3d(filters*8, filters*4, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')
        self.conv7_1b = nn.Conv3d(filters*4, filters*4, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')

        # Level 8: Third decoder level (skip from level 2: 4×filters → 2×filters)
        self.conv8_1 = nn.Conv3d(filters*4, filters*2, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')
        self.conv8_1b = nn.Conv3d(filters*2, filters*2, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')

        # Level 9: Fourth decoder level (skip from level 1: 2×filters → filters)
        self.conv9_1 = nn.Conv3d(filters*2, filters, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')
        self.conv9_1b = nn.Conv3d(filters, filters, pad, bias=False, padding=int((pad-1)/2), padding_mode='reflect')

        # Final output layer: reduce to single channel (1×1×1 convolution)
        self.conv10_1 = nn.Conv3d(filters, 1, 1, bias=False)

        # Configurable final activation function
        if last_act == 'sigmoid':
            self.last_act = nn.Sigmoid()      # Output range: [0, 1]
        elif last_act == 'softplus':
            self.last_act = nn.Softplus()     # Output range: [0, ∞) - ensures positive flux
        else:
            self.last_act = nn.Identity()     # Linear output - no transformation

    def center_crop(self, enc_feat, target_shape):
        """
        Crop encoder features to match target dimensions for skip connections.
        
        Due to pooling and upsampling operations, encoder and decoder features
        may have slightly different sizes. This function crops the encoder 
        features to exactly match the target shape for concatenation.
        
        Parameters
        ----------
        enc_feat : torch.Tensor
            Encoder feature tensor of shape (batch, channels, d, h, w).
        target_shape : tuple
            Target (d, h, w) dimensions for cropping.
            
        Returns
        -------
        torch.Tensor
            Cropped encoder features matching target_shape.
            
        Notes
        -----
        - Crops symmetrically from center of each dimension
        - Essential for skip connections in U-Net architecture
        - Handles dimension mismatches from pooling/upsampling operations
        """
        _, _, d, h, w = enc_feat.shape
        td, th, tw = target_shape
        
        # Calculate symmetric crop indices for each dimension
        d1 = (d - td) // 2
        h1 = (h - th) // 2
        w1 = (w - tw) // 2
        
        return enc_feat[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]

    def forward(self, x):
        """
        Forward pass through the 3D U-Net.
        
        Implements the complete encoder-decoder architecture with skip connections
        for spectral cube denoising. The network progressively downsamples to 
        capture global context, then upsamples while incorporating fine details
        from skip connections.
        
        Parameters
        ----------
        x : torch.Tensor
            Input noisy spectral cube of shape (batch_size, channels, depth, height, width).
            
        Returns
        -------
        torch.Tensor
            Denoised spectral cube of shape (batch_size, 1, depth, height, width).
        """
        # ====================================================================
        # ENCODER PATH - FEATURE EXTRACTION AND DOWNSAMPLING
        # ====================================================================
        
        # Level 1: Initial feature extraction
        x1 = self.act(self.conv1(x))        # Extract initial features from input
        x1 = self.act(self.conv1b(x1))      # Refine initial features
        x12 = self.pool(x1)                 # Downsample for next level

        # Level 2: Increased feature depth
        x2 = self.act(self.conv2(x12))      # Extract mid-level features
        x2 = self.act(self.conv2b(x2))      # Refine mid-level features
        x22 = self.pool(x2)                 # Downsample for next level

        # Level 3: Higher-level feature extraction
        x3 = self.act(self.conv3(x22))      # Extract higher-level patterns
        x3 = self.act(self.conv3b(x3))      # Refine higher-level patterns
        x32 = self.pool(x3)                 # Downsample for next level

        # Level 4: High-level semantic features
        x4 = self.act(self.conv4(x32))      # Extract semantic features
        x4 = self.act(self.conv4b(x4))      # Refine semantic features
        x42 = self.pool(x4)                 # Downsample to bottleneck

        # Level 5: Bottleneck - deepest representation
        x5 = self.act(self.conv5(x42))      # Extract deepest features
        x5 = self.act(self.conv5b(x5))      # Refine bottleneck representation

        # ====================================================================
        # DECODER PATH - RECONSTRUCTION WITH SKIP CONNECTIONS
        # ====================================================================
        
        # Level 6: First upsampling + skip connection from level 4
        up6 = self.transp1(x5)                                     # Upsample from bottleneck
        x4_crop = self.center_crop(x4, up6.shape[2:])             # Crop encoder features to match
        x1_6 = self.act(self.conv6_1(torch.cat([up6, x4_crop], dim=1)))  # Concatenate and process
        x1_6 = self.act(self.conv6_1b(x1_6))                      # Refine combined features

        # Level 7: Second upsampling + skip connection from level 3
        up7 = self.transp2(x1_6)                                  # Continue upsampling
        x3_crop = self.center_crop(x3, up7.shape[2:])             # Crop encoder features
        x1_7 = self.act(self.conv7_1(torch.cat([up7, x3_crop], dim=1)))  # Combine and process
        x1_7 = self.act(self.conv7_1b(x1_7))                      # Refine features

        # Level 8: Third upsampling + skip connection from level 2
        up8 = self.transp3(x1_7)                                  # Continue upsampling
        x2_crop = self.center_crop(x2, up8.shape[2:])             # Crop encoder features
        x1_8 = self.act(self.conv8_1(torch.cat([up8, x2_crop], dim=1)))  # Combine and process
        x1_8 = self.act(self.conv8_1b(x1_8))                      # Refine features

        # Level 9: Final upsampling + skip connection from level 1
        up9 = self.transp4(x1_8)                                  # Final upsampling
        x1_crop = self.center_crop(x1, up9.shape[2:])             # Crop initial features
        x1_9 = self.act(self.conv9_1(torch.cat([up9, x1_crop], dim=1)))  # Combine and process
        x1_9 = self.act(self.conv9_1b(x1_9))                      # Final feature refinement

        # ====================================================================
        # OUTPUT LAYER - FINAL RECONSTRUCTION
        # ====================================================================
        
        # Convert features to single output channel
        x1_10 = self.conv10_1(x1_9)         # Reduce to single channel
        x1_last = self.last_act(x1_10)      # Apply final activation
        
        return x1_last





class UNet3DWithPadCrop(torch.nn.Module):
    """
    Wrapper for 3D U-Net that handles variable input sizes through padding and cropping.
    
    This class addresses a common challenge in astronomical data processing where
    spectral cubes may have different dimensions. The wrapper ensures the U-Net
    can process cubes of any size by:
    1. Padding inputs to a fixed target size for processing
    2. Running inference through the base U-Net
    3. Cropping outputs back to the original input size
    
    This approach allows a single trained model to handle diverse observation
    scenarios without retraining on each specific cube dimension.
    
    Parameters
    ----------
    unet_model : UNet3D
        Pre-trained or initialized UNet3D instance.
    target_shape : tuple of int
        Target (depth, height, width) dimensions for internal processing.
        Should be chosen based on typical training data dimensions.
        
    Attributes
    ----------
    unet : UNet3D
        The wrapped U-Net model.
    target_shape : tuple
        Fixed internal processing dimensions.
        
    Methods
    -------
    pad_to_shape(x, target_shape)
        Pad input to target dimensions using reflective padding.
    crop_to_shape(x, target_shape) 
        Crop output back to original dimensions.
    forward(x)
        Complete forward pass with padding/cropping.
        
    Examples
    --------
    >>> # Create base U-Net model
    >>> base_unet = UNet3D(n_channels=1, filters=16)
    >>> 
    >>> # Wrap for variable input sizes (target: 48×80×80)
    >>> model = UNet3DWithPadCrop(base_unet, target_shape=(48, 80, 80))
    >>> 
    >>> # Process cube of any size
    >>> small_cube = torch.randn(1, 1, 30, 60, 60)
    >>> large_cube = torch.randn(1, 1, 50, 100, 100)
    >>> 
    >>> denoised_small = model(small_cube)  # Output: (1, 1, 30, 60, 60)
    >>> denoised_large = model(large_cube)  # Output: (1, 1, 50, 100, 100)
    
    Notes
    -----
    - Reflective padding preserves boundary conditions naturally
    - Target shape should be larger than typical input dimensions
    - Adds computational overhead but provides flexibility
    - Particularly useful for diverse astronomical survey data
    - Maintains exact input/output size correspondence
    """

    def __init__(self, unet_model, target_shape):
        """
        Initialize the padded U-Net wrapper.
        
        Parameters
        ----------
        unet_model : UNet3D
            Your original UNet3D instance (pre-trained or fresh).
        target_shape : tuple of int
            Target (D, H, W) dimensions you want to pad inputs to, e.g. (48, 80, 80).
            Should be chosen based on your typical training data size.
        """
        super().__init__()
        self.unet = unet_model           # Store the wrapped U-Net model
        self.target_shape = target_shape # Fixed processing dimensions

    def pad_to_shape(self, x, target_shape):
        """
        Pad input tensor to target shape using reflective padding.
        
        Reflective padding is ideal for astronomical data as it preserves
        the natural boundary conditions and avoids introducing artifacts
        at cube edges that could affect subsequent analysis.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, d, h, w).
        target_shape : tuple
            Target (d, h, w) dimensions for padding.
            
        Returns
        -------
        torch.Tensor
            Padded tensor with shape (batch, channels, target_d, target_h, target_w).
            
        Notes
        -----
        - Reflective padding mirrors values across boundaries
        - Padding is applied symmetrically when possible
        - Handles odd padding amounts by adding extra to the end
        """
        _, _, d, h, w = x.shape
        td, th, tw = target_shape

        # Calculate padding amounts for each dimension
        pad_d = td - d
        pad_h = th - h
        pad_w = tw - w

        # PyTorch padding format: [left, right, top, bottom, front, back]
        # Distribute padding symmetrically, with extra padding at end if odd
        pad = [
            pad_w // 2, pad_w - pad_w // 2,    # Width padding (left, right)
            pad_h // 2, pad_h - pad_h // 2,    # Height padding (top, bottom)  
            pad_d // 2, pad_d - pad_d // 2     # Depth padding (front, back)
        ]
        return F.pad(x, pad, mode='reflect', value=0)

    def crop_to_shape(self, x, target_shape):
        """
        Crop tensor to target shape by removing symmetric border regions.
        
        This function removes the padding added during preprocessing to
        return outputs to their original input dimensions.
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor to crop of shape (batch, channels, d, h, w).
        target_shape : tuple
            Target (d, h, w) dimensions after cropping.
            
        Returns
        -------
        torch.Tensor
            Cropped tensor with shape (batch, channels, target_d, target_h, target_w).
            
        Notes
        -----
        - Crops symmetrically from center of each dimension
        - Inverse operation of pad_to_shape()
        - Preserves the most central/important regions of the output
        """
        _, _, d, h, w = x.shape
        td, th, tw = target_shape
        
        # Calculate symmetric crop indices
        d1 = (d - td) // 2
        h1 = (h - th) // 2
        w1 = (w - tw) // 2
        
        return x[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]

    def forward(self, x):
        """
        Forward pass with automatic padding and cropping.
        
        Implements the complete workflow:
        1. Store original input dimensions
        2. Pad input to target processing size
        3. Run U-Net inference on padded data
        4. Crop output back to original size
        
        Parameters
        ----------
        x : torch.Tensor
            Input spectral cube of arbitrary size (batch, channels, d, h, w).
            
        Returns
        -------
        torch.Tensor
            Denoised spectral cube with same dimensions as input.
            
        Notes
        -----
        - Completely transparent to the user - input and output sizes match
        - Handles any input size as long as target_shape is larger
        - Adds computational overhead but provides crucial flexibility
        - Preserves all original spatial and spectral sampling
        """
        original_shape = x.shape[2:]                    # Store original (D,H,W)
        x_padded = self.pad_to_shape(x, self.target_shape)  # Pad to processing size
        y = self.unet(x_padded)                        # Run U-Net on padded data
        y_cropped = self.crop_to_shape(y, original_shape)   # Crop back to original size
        return y_cropped
