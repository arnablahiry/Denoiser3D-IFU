
#!/usr/bin/env python
"""
3D IFU Spectral Cube Wavelet Denoising Module

This module provides advanced 3D wavelet denoising capabilities for Integral Field Unit (IFU) 
spectral cubes, utilizing a 2D-1D multi-scale wavelet decomposition approach for astronomical 
data analysis.

Scientific Background
--------------------
IFU spectroscopy produces 3D data cubes where each spatial pixel contains a full spectrum,
enabling detailed analysis of galaxy kinematics, chemistry, and morphology. However, these
observations often suffer from significant noise that can obscure faint emission features
or distort velocity measurements. Traditional 2D denoising methods applied slice-by-slice
ignore valuable spectral correlations that can be exploited for better noise removal.

This implementation extends traditional 2D starlet denoising by incorporating spectral
information through a hybrid 2D-1D wavelet transform that:
- Applies 2D starlet decomposition to preserve spatial morphology
- Uses 1D wavelet analysis along the spectral axis to capture line profiles
- Employs adaptive thresholding based on noise characteristics in each sub-band
- Supports both hard and soft thresholding with iterative refinement

Technical Approach
-----------------
The denoising framework assumes a simple additive noise model:
    Y = X + N
where Y is the observed noisy cube, X is the true signal, and N is the noise.

The 2D-1D wavelet transform decomposes the cube into multiple scales:
- 2D spatial scales capture structure at different angular sizes
- 1D spectral scales capture features at different velocity bins
- Each (i,j) sub-band represents specific spatial-spectral scale combinations

Thresholding strategies:
- Hard thresholding: Set coefficients below threshold to zero
- Soft thresholding: Shrink coefficients by threshold amount
- Iterative approaches: Refine estimates through multiple iterations
- Adaptive weighting: Account for varying noise levels across sub-bands
  and iteratively re-weights to account for soft thresholding bias

Applications
-----------
This module is particularly effective for:
- ALMA observations with low signal-to-noise ratios
- Recovering faint emission lines in high-redshift galaxies  
- Preprocessing data for kinematic analysis
- Enhancing detection of extended emission structures

Dependencies
-----------
- pysparse: Multi-resolution 2D-1D wavelet transforms : Sparse2D
- numpy: Numerical computations and array operations
- matplotlib: Diagnostic plotting and visualization
- pycs: Cosmostat library for wavelets and statistics : CosmoStat

Classes
-------
Wavelet2D1DTransform : Core wavelet decomposition and reconstruction
Denoiser2D1D : Main denoising interface with multiple algorithms

Functions
---------
mock_noise_value : Utility for generating synthetic noise realizations

"""


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pysparse
from pycs.misc.cosmostat_init import *
from pycs.misc.stats import *


class Wavelet2D1DTransform(object):
    """
    2D-1D Multi-Scale Wavelet Transform for 3D Spectral Cubes
    
    This class implements a hybrid wavelet decomposition that combines 2D spatial 
    starlet transforms with 1D spectral wavelet analysis. The approach is specifically 
    designed for IFU spectral cubes where spatial and spectral information should be 
    treated differently to preserve both morphological and kinematic features.
    
    The decomposition treats the 3D cube as a stack of 2D images (spatial planes) 
    along the spectral axis, applying:
    1. 2D starlet transform to capture spatial structure at multiple scales
    2. 1D wavelet transform along the spectral direction for each spatial pixel
    
    This preserves the multi-scale nature of astronomical sources while maintaining
    spectral line profiles and velocity structure.
    
    Attributes
    ----------
    NOISE_TAB : np.ndarray
        Pre-computed noise scaling factors for different wavelet sub-bands.
        Shape (5, 5) for up to 5 spatial and 5 spectral scales.
        Currently not used but available for future noise modeling.
        
    Methods
    -------
    decompose(cube, num_scales_2d, num_scales_1d)
        Perform forward 2D-1D wavelet transform
    reconstruct(coeffs)
        Perform inverse transform to recover spatial cube
    energy_per_scale(num_scales_2d, num_scales_1d)
        Get noise scaling for requested number of scales
        
    Notes
    -----
    - The transform preserves flux conservation
    - Coefficients are organized by scale with metadata for reconstruction
    - Transform type and normalization controlled by pysparse.MR2D1D parameters
    """
    
    # Pre-computed noise scaling factors for different wavelet sub-bands
    # TODO: Compute values more accurately by excluding borders of each sub-band cube
    NOISE_TAB = np.array([[0.9610, 0.9261, 0.9611, 0.9810, 0.9933],
                          [0.2368, 0.2282, 0.2369, 0.2417, 0.2449],
                          [0.1049, 0.1011, 0.1050, 0.1071, 0.1082],
                          [0.0527, 0.0507, 0.0528, 0.0539, 0.0543],
                          [0.0283, 0.0272, 0.0283, 0.0288, 0.0295]])

    def __init__(self, transform_type=2):
        """
        Initialize the 2D-1D wavelet transform with specified parameters.
        
        This is a wrapper for pysparse's MR2D1D transform, providing a simplified
        interface for 3D spectral cube analysis. The transform combines 2D spatial
        decomposition with 1D spectral analysis to preserve both morphological
        and kinematic information in astronomical data cubes.

        Parameters
        ----------
        transform_type : int, optional
            Type of wavelet transform to perform with `decompose`, by default 2.
            Available options from pysparse documentation:
            - 2: 2D starlet (undecimated) + 1D 7/9 filter bank (decimated) [default]
            
        Notes
        -----
        The transform object (self._mr2d1d) is created lazily when decompose() 
        is first called. This allows automatic parameter detection based on 
        input cube dimensions and avoids unnecessary memory allocation.
        
        The default choice (transform_type=2) provides:
        - Isotropic 2D starlets: Excellent for point sources and extended emission
        - 1D 7/9 filter bank: Good spectral localization for line profiles
        - Undecimated spatial: Preserves spatial resolution and avoids artifacts
        - Decimated spectral: Reduces computational cost while preserving features
        
        References
        ----------
        See pysparse documentation for detailed descriptions of available transforms
        and their mathematical properties.
        """
        self.transform_type = transform_type
        self._mr2d1d = None  # Initialized lazily in decompose method

    def decompose(self, cube, num_scales_2d, num_scales_1d):
        """
        Perform forward 2D-1D multi-scale wavelet decomposition of spectral cube.
        
        Decomposes the input 3D cube into wavelet coefficients using a hybrid approach:
        - 2D spatial decomposition preserves morphological structure at multiple scales
        - 1D spectral decomposition captures velocity/frequency features
        - Combined analysis exploits correlations between spatial and spectral dimensions
        
        The decomposition creates (num_scales_2d × num_scales_1d) sub-bands, each
        representing specific combinations of spatial and spectral scales. This allows
        for scale-dependent processing during denoising operations.

        Parameters
        ----------
        cube : np.ndarray, shape (nz, ny, nx)
            Input 3D spectral data cube. The spectral axis should be first (axis=0),
            followed by spatial dimensions. Typical astronomical convention where
            each cube[i,:,:] slice is a 2D image at a specific wavelength/velocity.
        num_scales_2d : int
            Number of scales for 2D spatial starlet decomposition.
            Must be >= 2 and <= int(log2(min(ny, nx))).
            Controls spatial resolution analysis - more scales capture finer details.
        num_scales_1d : int  
            Number of scales for 1D spectral wavelet decomposition.
            Must be >= 2 and <= int(log2(nz)).
            Controls spectral feature resolution - more scales capture narrower lines.

        Returns
        -------
        inds : list of list of tuples
            Nested index structure for accessing coefficients by scale.
            inds[i][j] = (start, end) gives indices for sub-band (i,j) in coeffs array.
            Use as: coeffs[start:end] to extract coefficients for scale pair (i,j).
        shapes : list of list of tuples  
            Nested shape information for each wavelet sub-band.
            shapes[i][j] = (nx, ny, nz) gives dimensions of sub-band (i,j).
        coeffs : np.ndarray, 1D
            Flattened array containing all wavelet coefficients plus metadata.
            Structure: [n_scales_2d, n_scales_1d, shape_info, coefficients...]
            Required for reconstruction via reconstruct() method.
            
        Notes
        -----
        Transform object is initialized on first call. Subsequent calls reuse
        the same transform for efficiency. Coefficients are organized in 
        increasing order of scale (fine to coarse).
        
        See Also
        --------
        reconstruct : Inverse transform to recover original cube
        _extract_metadata : Internal method for parsing coefficient structure
        """
        # Initialize the 2D-1D multi-resolution transform object
        # Configure with specified transform type and scale parameters
        self._mr2d1d = pysparse.MR2D1D(type_of_transform=self.transform_type,
                                       normalize=False,
                                       verbose=False,
                                       NbrScale2d=num_scales_2d,
                                       Nbr_Plan=num_scales_1d)
        
        # Perform the forward wavelet transform
        coeffs = self._mr2d1d.transform(cube)

        # Extract index mapping and shape information for coefficient organization
        # This metadata is essential for accessing specific sub-bands during processing
        inds, shapes = self._extract_metadata(coeffs)

        return inds, shapes, coeffs

    def reconstruct(self, coeffs):
        """
        Perform inverse 2D-1D wavelet transform to recover original cube.
        
        Reconstructs the 3D spectral cube from its multi-scale wavelet representation.
        This is the inverse operation of decompose(), combining information from all
        spatial and spectral scales to recover the original data structure.
        
        The reconstruction process:
        1. Parses the coefficient structure from decompose()
        2. Applies inverse 1D spectral wavelets 
        3. Applies inverse 2D spatial starlets
        4. Combines all scales to form the final cube

        Parameters
        ----------
        coeffs : np.ndarray, 1D
            Wavelet coefficients as returned by decompose().
            Must include metadata (scales, shapes) followed by coefficient values.
            Can be modified coefficients (e.g., after thresholding for denoising).

        Returns
        -------
        cube : np.ndarray, shape (nz, ny, nx)
            Reconstructed 3D spectral data cube.
            Shape matches the original input to decompose().
            Preserves flux conservation (sum of pixels) unless coefficients modified.
            
        Notes
        -----
        The transform object must be initialized (via decompose()) before calling
        this method. The reconstruction uses the same transform parameters as
        the forward decomposition.
        
        Modified coefficients (e.g., thresholded for denoising) will produce
        a reconstructed cube with altered characteristics while preserving
        the overall structure encoded in the retained coefficients.
        """
        # Ensure the transform object exists (decompose must be called first)
        # This assertion helps catch usage errors but is commented for performance
        # assert hasattr(self, '_mr2d1d'), "Need to call decompose first."

        # Perform the inverse transform using pysparse reconstruction
        reconstructed = self._mr2d1d.reconstruct(coeffs)
        
        return reconstructed

    def energy_per_scale(self, num_scales_2d, num_scales_1d):
        """
        Get pre-computed noise energy scaling factors for wavelet sub-bands.
        
        Returns the noise scaling factors that account for how noise propagates
        through the wavelet transform at different scales. These values can be
        used to normalize thresholds for denoising operations.

        Parameters
        ----------
        num_scales_2d : int
            Number of 2D spatial scales to retrieve factors for.
        num_scales_1d : int  
            Number of 1D spectral scales to retrieve factors for.

        Returns
        -------
        np.ndarray, shape (num_scales_2d, num_scales_1d)
            Noise scaling factors for each (2D scale, 1D scale) combination.
            Values represent the multiplicative factor for noise standard deviation
            in each wavelet sub-band relative to the input noise level.
            
        Notes
        -----
        Currently uses pre-computed values in NOISE_TAB. Future versions should
        compute these more accurately by excluding border effects in each sub-band.
        """
        return self.NOISE_TAB[:num_scales_2d, :num_scales_1d]

    @property
    def num_precomputed(self):
        """
        Get the maximum number of scales for which noise factors are pre-computed.
        
        Returns
        -------
        tuple of int
            (max_2d_scales, max_1d_scales) available in NOISE_TAB.
        """
        return self.NOISE_TAB.shape

    def _extract_metadata(self, coeffs):
        """
        Parse coefficient array structure to extract indexing and shape metadata.
        
        The coefficients array from pysparse contains metadata followed by the actual
        wavelet coefficients. This method parses the structure to create convenient
        indexing arrays for accessing specific sub-bands during processing.
        
        Coefficient Array Structure:
        - coeffs[0]: Number of 2D scales used 
        - coeffs[1]: Number of 1D scales used
        - coeffs[2:5]: Shape (nx,ny,nz) for sub-band (0,0)
        - coeffs[5:5+nx*ny*nz]: Coefficients for sub-band (0,0) 
        - coeffs[...]: Shape and coefficients for remaining sub-bands
        - Sub-bands ordered by: for scale2d in range(n_scales_2d): 
                                   for scale1d in range(n_scales_1d): ...

        Parameters
        ----------
        coeffs : np.ndarray, 1D
            Flattened coefficient array from decompose() containing metadata
            and coefficients for all wavelet sub-bands.

        Returns
        -------
        inds : list of list of tuples
            Index structure where inds[i][j] = (start, end) gives the slice
            indices for accessing coefficients of sub-band (2d_scale=i, 1d_scale=j).
            Use as: coeffs[start:end] to get coefficients for that sub-band.
        shapes : list of list of tuples  
            Shape structure where shapes[i][j] = (nx, ny, nz) gives the 3D
            dimensions of sub-band (2d_scale=i, 1d_scale=j). Essential for
            reshaping flattened coefficients back to 3D arrays.
            
        Notes
        -----
        The returned structures are essential for:
        - Accessing specific wavelet sub-bands for thresholding
        - Reshaping coefficients for 3D processing
        - Reconstruction operations

        """
        # Extract number of scales from the first two elements
        n_scales_2d = int(coeffs[0])
        n_scales_1d = int(coeffs[1])

        # Initialize nested lists for index ranges and shapes
        # Structure: inds[2d_scale][1d_scale] = (start_idx, end_idx)
        inds = [[() for _ in range(n_scales_1d)] for _ in range(n_scales_2d)]
        shapes = [[() for _ in range(n_scales_1d)] for _ in range(n_scales_2d)]

        # Parse through the coefficient array to extract metadata for each sub-band
        start = end = 2  # Skip the first two scale count elements

        # Traverse all scale combinations in order used by pysparse
        for ii in range(n_scales_2d):
            for jj in range(n_scales_1d):
                # Each sub-band starts with 3 shape values (nx, ny, nz)
                start = end + 3
                
                # Extract sub-band dimensions
                nx, ny, nz = map(int, coeffs[start-3 : start])
                shapes[ii][jj] = (nx, ny, nz)
                
                # Calculate coefficient count and ending index for this sub-band
                ncoeff = nx * ny * nz
                end = start + ncoeff
                
                # Store the index range for this sub-band
                inds[ii][jj] = (start, end)

        return inds, shapes


# =============================================================================
# 3D Wavelet Denoising Framework
# =============================================================================
# 
# The Denoiser2D1D class below is modeled after Aymeric's StarletDenoiser that 
# performs 2D denoising on each frequency slice of the input data. This 3D 
# extension leverages additional correlations in the spectral direction for 
# improved denoising performance.
# 
# Key Features:
# - Supports both simple (one-step) and iterative denoising algorithms
# - Implements hard and soft thresholding strategies
# - Accepts optional noise realizations for accurate noise modeling
# - Provides adaptive thresholding based on local noise characteristics
# - Includes convergence monitoring and diagnostic plotting
# 
# The denoise() method accepts an optional noise cube realization as input. 
# When provided, the noise is transformed and its standard deviation is computed 
# in each wavelet band to establish accurate noise levels. If not provided, 
# the noise in each band is estimated from the data itself using robust statistics.


class Denoiser2D1D(object):
    """
    Advanced 3D Spectral Cube Denoiser using 2D-1D Multi-Scale Wavelets
    
    This class implements sophisticated denoising algorithms for IFU spectral cubes
    using a hybrid 2D-1D wavelet decomposition. The approach combines spatial and
    spectral information to achieve superior noise reduction while preserving both
    morphological and kinematic features.
    
    The denoising framework assumes an additive noise model:
        Y = X + N
    where Y is the observed noisy cube, X is the true signal, and N is the noise.
    
    Available Denoising Methods:
    ---------------------------
    1. Iterative Hard: Multiple iterations with binary (hard) thresholding  
    2. Iterative Soft: Multiple iterations and re-weighting with adaptive (soft) thresholding
    
    The iterative methods use advanced techniques:
    - Adaptive re-weighting to reduce bias from soft thresholding
    - Residual signal extraction to recover previously missed features
    - Plateau-based convergence criteria for robust stopping
    
    Key Advantages:
    --------------
    - Preserves spatial morphology through 2D starlet decomposition
    - Maintains spectral line profiles via 1D wavelet analysis
    - Adapts thresholds to local noise characteristics
    - Recovers faint emission through iterative refinement
    - Supports both synthetic and observational noise modeling
    
    Typical Applications:
    --------------------
    - ALMA/JWST IFU observations with low SNR
    - High-redshift galaxy emission line recovery
    - Continuum-subtracted line cube cleaning
    - Kinematic analysis preprocessing
    - Extended emission detection enhancement
    
    Attributes
    ----------
    mr2d1d : Wavelet2D1DTransform
        The wavelet transform object for decomposition/reconstruction
    _threshold_type : str
        Thresholding strategy ('soft' or 'hard')  
    _verbose : bool
        Controls progress reporting and diagnostics
    _plot : bool
        Enables diagnostic plotting during processing
        
    Methods
    -------
    denoise(x, y, method='simple', **kwargs)
        Main denoising interface with multiple algorithm options
    __call__(*args, **kwargs)  
        Convenience alias for denoise() method
        
    Private Methods
    ---------------

    _denoise_iterative_hard(**kwargs)
        Multi-iteration hard thresholding with L0 regularization
    _denoise_iterative_soft(**kwargs) 
        Multi-iteration soft thresholding with adaptive reweighting
    _generate_hard_threshold_mask(coeffs, thresh, noise_level)
        Create binary masks for hard thresholding
    _residual_signal_extraction_l0/l1(...)
        Extract additional signal from residuals
    _estimate_noise(array)
        Robust noise estimation using median absolute deviation
    _compute_emission_rmse(model)
        Calculate reconstruction error in emission regions
    _prox_positivity_constraint(array)
        Apply positivity constraint for astrophysical sources
    """

    def __init__(self, threshold_type='soft', verbose=True, plot=False):
        """
        Initialize the 3D denoising framework with specified parameters.
        
        Sets up the wavelet transform object and configures algorithm parameters
        for subsequent denoising operations. The choice of threshold type determines
        which iterative algorithms are available.

        Parameters
        ----------
        threshold_type : str, optional
            Thresholding strategy for coefficient processing, by default 'soft'.
            Options:
            - 'soft': Shrink coefficients by threshold amount (preserves gradients)
            - 'hard': Set coefficients below threshold to zero (creates sparsity)
            Soft thresholding generally produces smoother results but may over-smooth.
            Hard thresholding preserves sharp features but can introduce artifacts.
        verbose : bool, optional  
            Enable detailed progress reporting and diagnostics, by default True.
            When True, prints iteration progress, convergence status, flux measurements,
            and algorithm-specific information during processing.
        plot : bool, optional
            Enable diagnostic plotting during denoising, by default False. 
            When True, displays:
            - Coefficient distributions before/after thresholding
            - Iteration-by-iteration reconstruction progress  
            - Residual analysis and signal extraction visualization
            - Final comparison plots (original/denoised/residual)
            Useful for algorithm development and result validation.
            
        Notes
        -----
        The wavelet transform object is created with default parameters and can
        handle cubes with dimensions up to the limits of available memory.
        Transform parameters are automatically configured based on input cube
        dimensions during the first call to denoise().
        
        For large cubes (>1GB), consider setting verbose=False to reduce I/O
        overhead during processing.
        """
        # Initialize the 2D-1D wavelet transform engine
        self.mr2d1d = Wavelet2D1DTransform()
        
        # Store algorithm configuration
        self._threshold_type = threshold_type
        self._verbose = verbose
        self._plot = plot

    def __call__(self, *args, **kwargs):
        """
        Convenience method to call denoise() directly on the object.
        
        Allows using the denoiser object as a function:
        denoiser(cube, signal, method='iterative')
        
        This is equivalent to denoiser.denoise(cube, signal, method='iterative')
        
        Returns
        -------
        Denoised cube or tuple of results depending on method used.
        """
        return self.denoise(*args, **kwargs)

    def denoise(self, x, y, method='iterative', threshold_level=3,
                threshold_increment_high_freq=2, num_scales_2d=None, 
                num_scales_1d=None, noise_cube=None, emission_mask=None, **kwargs_method):
        """
        Denoise a 3D spectral cube using advanced 2D-1D wavelet techniques.
        
        This is the main interface for all denoising algorithms. The method applies
        multi-scale wavelet decomposition followed by intelligent thresholding to
        remove noise while preserving both spatial morphology and spectral features.
        
        The algorithm automatically determines optimal scale numbers if not specified,
        adapts thresholds to noise characteristics, and can incorporate ground truth
        for algorithm validation and noise modeling.

        Parameters
        ----------
        x : np.ndarray, shape (nz, ny, nx)
            Input noisy data cube. The spectral/frequency axis must be first (axis=0),
            followed by spatial dimensions. Each x[i,:,:] is a 2D image at wavelength i.
        y : np.ndarray, shape (nz, ny, nx)
            Clean signal cube (ground truth) with same shape as x.
            Used for algorithm validation, noise modeling, and performance assessment.
            In real observations, this would be unknown - here used for synthetic testing.
        method : str, optional
            Denoising algorithm to use, by default 'simple'.
            Options:
            - 'simple': Single-iteration thresholding (fast, good for high SNR)
            - 'iterative': Multi-iteration adaptive algorithm (slower, better for low SNR)
              For iterative method, the specific algorithm depends on threshold_type:
              * 'hard' → Iterative hard thresholding with L0 regularization
              * 'soft' → Iterative soft thresholding with adaptive reweighting
        threshold_level : float, optional
            Base threshold level in noise standard deviations, by default 3.
            Typical range: 2-5 for 2-5σ detection significance.
            Lower values preserve more signal but retain more noise.
            Higher values create cleaner results but may remove faint features.
        threshold_increment_high_freq : float, optional
            Additional threshold increment for highest frequency scales, by default 2.
            These scales typically contain pure noise and benefit from higher thresholds.
            Final threshold = threshold_level + threshold_increment_high_freq for finest scales.
        num_scales_2d : int, optional
            Number of 2D spatial starlet scales, by default None (auto-determined).
            If None, uses maximum: int(log2(min(ny, nx))).
            Range: 2 to maximum allowed by image dimensions.
            More scales capture finer spatial details but increase computation.
        num_scales_1d : int, optional  
            Number of 1D spectral wavelet scales, by default None (auto-determined).
            If None, uses maximum: int(log2(nz)).
            Range: 2 to maximum allowed by spectral dimensions.
            More scales capture narrower spectral features but increase computation.
        noise_cube : np.ndarray, shape (nz, ny, nx), optional
            Independent noise realization with same shape as input, by default None.
            If provided, used to accurately estimate noise levels in each wavelet sub-band.
            If None, noise levels estimated from the data using robust statistics.
            Recommended for synthetic data where true noise is known.
        emission_mask : np.ndarray, shape (nz, ny, nx), optional
            Binary mask indicating emission regions (1=emission, 0=background), by default None.
            If None, creates mask of all ones (assumes entire cube contains signal).
            Used for targeted error calculation and emission-focused denoising.
            Particularly useful for line emission cubes with known source extent.
        **kwargs_method : dict
            Additional keyword arguments passed to specific denoising methods:
            
            For method='iterative' with threshold_type='hard':
            - num_iter : int, default 20
                Number of hard thresholding iterations
                
            For method='iterative' with threshold_type='soft':  
            - num_iter_reweight : int, default 20
                Number of reweighting iterations
            - num_iter_debias : int, default 20
                Number of debiasing iterations  
            - debias : bool, default True
                Whether to perform debiasing step

        Returns
        -------
        result : np.ndarray or tuple
            Denoised results. Return format depends on method:
            
            For method='simple':
                np.ndarray, shape (nz, ny, nx) : Denoised cube
                
            For method='iterative':
                tuple with multiple outputs depending on threshold_type:
                
                Hard thresholding returns:
                - best_model : Denoised cube at best iteration
                - deltas : Accumulated residual signals extracted
                - residual_stds : Standard deviation history per iteration  
                - best_iteration : Iteration number of best result
                - noise_levels : Noise estimates per wavelet sub-band
                
                Soft thresholding returns:
                - best_model : Final denoised cube
                - model_1_step : Result after first reweighting phase
                - model_no_reweight : Result without reweighting
                - deltas : Accumulated residual signals from debiasing
                - residual_stds_reweight : Residual history during reweighting
                - residual_stds_debias : Residual history during debiasing  
                - best_iteration : Iteration of best result
                - dists : Selected coefficient distributions for diagnostics
                - noise_levels : Noise level estimates per sub-band
                
        Raises
        ------
        ValueError
            If method is not 'simple' or 'iterative'
        AssertionError  
            If noise_cube provided but shape doesn't match input x
        NotImplementedError
            If requested number of scales exceeds pre-computed limits
            
        Notes
        -----
        Processing Time:
        - Simple method: ~10-30 seconds for 100³ cube
        - Iterative method: ~2-10 minutes depending on iterations and convergence
        
        Memory Requirements:
        - Peak usage ~4-5x input cube size during transform operations
        - Coefficient storage ~1.5x input cube size
        
        Algorithm Selection Guide:
        - High SNR (>10): Simple method often sufficient
        - Low SNR (<5): Iterative soft thresholding recommended
        - Preserving sharp features: Hard thresholding
        - Smooth reconstruction: Soft thresholding
        """
        
        # Validate and set default number of 2D decomposition scales
        num_scales_2d_max = int(np.log2(x.shape[1]))
        if num_scales_2d is None or num_scales_2d < 2 or num_scales_2d > num_scales_2d_max:
            # choose the maximum allowed number of scales
            num_scales_2d = num_scales_2d_max
            if self._verbose is True:
                print(f"Number of 2D wavelet scales set to {num_scales_2d} "
                      "(maximum value allowed by input image)")
        
        # Set the number of 1D decomposition scales
        num_scales_1d_max = int(np.log2(x.shape[0]))
        if num_scales_1d is None or num_scales_1d < 2 or num_scales_1d > num_scales_1d_max:
            # choose the maximum allowed number of scales
            num_scales_1d = num_scales_1d_max
            if self._verbose is True:
                print(f"Number of 1D wavelet scales set to {num_scales_1d} "
                      "(maximum value allowed by input image)")
                
        # Check that the pre-computed noise scaling exists for the requested scales
        # if (num_scales_2d - 1 > self.mr2d1d.num_precomputed[0] or 
        #     num_scales_1d - 1 > self.mr2d1d.num_precomputed[1]):
        #     raise NotImplementedError(f"Pre-computed noise in wavelet space has been implemented"
        #                               f" for up to {self.mr2d1d.NOISE_TAB.shape} scales "
        #                               f"[({num_scales_2d}, {num_scales_1d}) required)]")
            
        # Check that the noise realisation has the same shape as the input

        if noise_cube is not None:
            assert x.shape == noise_cube.shape, "Invalid noise estimate shape"

        # Initialise settings for the denoiser
        self._data = x
        self._signal = y
        self._num_bands = self._data.shape[0]
        self._num_pixels = self._data.shape[1] * self._data.shape[2]
        self._num_scales_2d = num_scales_2d
        self._num_scales_1d = num_scales_1d
        self._threshold_level = float(threshold_level)
        self._thresh_increm = float(threshold_increment_high_freq)
        self._noise = noise_cube
        if emission_mask is None:
            emission_mask = np.ones_like(y)
        self._mask = emission_mask

        # Select and run the denoiser
        if method == 'simple':
            if self._verbose:   print('\n--- [ PERFORMING SIMPLE (ONE-STEP) DENOISING ] ---\n')
            result = self._denoise_simple()
        elif method == 'iterative':
            if self._verbose:   print('\n--- [ PERFORMING ITERATIVE DENOISING ] ---\n')
            if self._threshold_type == 'hard':
                result = self._denoise_iterative_hard(**kwargs_method)
            if self._threshold_type == 'soft':
                result = self._denoise_iterative_soft(**kwargs_method)
        else:
            raise ValueError(f"Denoising method '{method}' is not supported")
        
        return result



    def _generate_hard_threshold_mask(self, coeffs, thresh, noise_level):
        """
        Generate binary mask for hard thresholding of wavelet coefficients.
        
        Creates a binary mask that identifies significant coefficients above the
        noise threshold. This mask is used for hard thresholding where coefficients
        are either kept (if significant) or set to zero (if below threshold).

        Parameters
        ----------
        coeffs : np.ndarray
            1D array of wavelet coefficients for a specific sub-band.
            These are the transformed values at a particular spatial/spectral scale.
        thresh : float
            Threshold level in units of noise standard deviations.
            Typical values: 3-5 for 3-5σ detection significance.
        noise_level : float
            Estimated noise standard deviation for this specific sub-band.
            Accounts for how noise propagates through the wavelet transform.

        Returns
        -------
        mask_coeff : np.ndarray
            Binary mask array with same shape as coeffs.
            Values: 1 for coefficients above threshold, 0 for those below.
            Used to multiply coefficients: coeffs_thresh = coeffs * mask_coeff
            
        Notes
        -----
        Hard thresholding creates sparsity by completely removing coefficients
        below the threshold. This preserves sharp features but can introduce
        artifacts. The threshold is computed as:
            threshold_value = thresh × noise_level
        
        Coefficients with |coeff| > threshold_value are retained (mask=1).
        Coefficients with |coeff| ≤ threshold_value are removed (mask=0).
        """
        # Calculate the absolute threshold value for this sub-band
        threshold = thresh * noise_level 
        
        # Initialize mask to ones (keep all coefficients by default)
        mask_coeff = np.ones_like(coeffs)
        
        # Set mask to zero for coefficients below threshold (hard thresholding)
        mask_coeff[np.abs(coeffs) <= threshold] = 0
        
        return mask_coeff





    def _residual_signal_extraction_l0(self, model, mask_coeff, iteration):
        """
        Extract additional signal from residuals using L0 regularization (hard thresholding).
        
        This method implements one iteration of the iterative hard thresholding algorithm
        for signal recovery. It analyzes the residual between the current model and the 
        noisy data, applies the previously computed significance mask, and extracts
        previously undetected signal components.
        
        The approach follows these steps:
        1. Compute residual = observed_data - current_model
        2. Transform residual into wavelet domain
        3. Apply previously computed significance mask to identify signal
        4. Reconstruct the masked residual to extract new signal  
        5. Apply positivity constraint and update the model

        Parameters
        ----------
        model : np.ndarray
            Current estimate of the denoised cube from previous iteration.
            Shape (nz, ny, nx) matching the input data dimensions.
        mask_coeff : np.ndarray  
            Binary coefficient mask from first iteration identifying significant features.
            Applied to residual coefficients to extract previously missed signal.
        iteration : int
            Current iteration number for progress reporting and diagnostic plotting.

        Returns
        -------
        model : np.ndarray
            Updated model incorporating newly extracted signal.
            Same shape as input model with additional recovered features.
        delta : np.ndarray
            Extracted signal component added to the model in this iteration.
            Shows what new information was recovered from the residuals.
            
        Notes
        -----
        This method is designed for L0 (hard thresholding) regularization where the
        sparsity pattern is fixed after the first iteration. The mask computed initially
        defines which coefficients are considered significant throughout all iterations.
        
        The positivity constraint ensures astrophysical realism by preventing 
        negative flux values, which are typically unphysical for emission sources.
        
        Optional plotting shows:
        - Current model vs true signal comparison
        - Residual analysis and signal extraction visualization  
        - Updated model after incorporating new signal
        
        This approach is particularly effective for recovering faint emission that
        was initially below the detection threshold but becomes apparent as the
        noise level decreases through iterative processing.
        """

        # Find the peak signal location for plotting reference
        max_voxel_index = np.argmax(self._signal)  # Get flattened index
        iz, max_y, max_x = np.unravel_index(max_voxel_index, self._signal.shape)  # Convert to 3D index

        # Calculate the residual between observed data and current model
        residual = self._data - model
                
        # Optional diagnostic plotting to visualize the iteration progress
        if self._plot:
            # Create comprehensive visualization of current state
            fig, axs = plt.subplots(2, 3, figsize=(16, 13), constrained_layout=True)

            # Top row: Current model, signal residual, and true signal
            im1 = axs[0,0].imshow(model[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,0].set_title('Previously Denoised (Iteration #{})'.format(iteration))

            im2 = axs[0,1].imshow((self._signal - model)[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,1].set_title('SIGNAL Residual')

            im3 = axs[0,2].imshow(self._signal[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,2].set_title('SIGNAL')

            axs[0,0].axis('off')
            axs[0,1].axis('off')
            axs[0,2].axis('off')

            # Add colorbars with proper labeling
            cbar1 = fig.colorbar(im1, ax=axs[0, 0], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar2 = fig.colorbar(im2, ax=axs[0, 1], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar3 = fig.colorbar(im3, ax=axs[0, 2], orientation='horizontal', fraction=0.05, pad=0.02)

            cbar1.ax.tick_params()
            cbar2.ax.tick_params()
            cbar3.ax.tick_params()

            cbar1.set_label('Flux')
            cbar2.set_label('Flux')
            cbar3.set_label('Flux')

        # Progress reporting for first iteration
        if iteration==1:
            if self._verbose: print('(*) Decomposing residual into wavelet scales')

        # Transform the residual into wavelet domain for analysis
        inds, shapes, w_residual = self.mr2d1d.decompose(residual,
                                                            self._num_scales_2d,
                                                            self._num_scales_1d)

        # Apply the significance mask computed in the first iteration
        # This identifies coefficients that represent signal rather than noise
        if iteration==1:
            if self._verbose: print('(*) Applying previously calculated mask on the residual coefficients, and\n considering the unmased coefficients as previously unnoticed signal coefficients')
        
        # Apply hard thresholding using the fixed sparsity pattern
        w_residual *= mask_coeff

        # Reconstruct the extracted signal from masked residual coefficients        
        if iteration==1:
            if self._verbose: print('(*) Reconstructing the new signal coefficients into the real space')
        # Inverse transform to recover spatial structure of extracted signal
        delta = self.mr2d1d.reconstruct(w_residual)

        # Apply positivity constraint for astrophysical realism
        if iteration==1:
            if self._verbose: print('(*) Applying the positivity constraint')
        
        # Ensure non-negative flux values (typical for emission sources)
        delta = np.maximum(0, delta)

        # Update the model with newly detected signal
        if iteration==1:
            if self._verbose: print('(*) Updating the model with the newly detected signal')
        
        # Add the extracted signal to the current model estimate
        model = model + delta
        # Ensure the updated model also satisfies positivity constraints
        model = np.maximum(0, model)
        
        # Complete the diagnostic plotting if enabled
        if self._plot:
            # Bottom row: Residual analysis and model update visualization
            im4 = axs[1,0].imshow(residual[iz], cmap = 'RdBu_r')
            axs[1,0].set_title('Residual')

            im5 = axs[1,1].imshow(delta[iz], cmap = 'RdBu_r', vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,1].set_title('Residual Information')

            im6 = axs[1,2].imshow(model[iz], cmap = 'RdBu_r', vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,2].set_title('Updated Model (Iteration #{})'.format(iteration+1))

            # Add colorbars for bottom row
            cbar4 = fig.colorbar(im4, ax=axs[1, 0], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar5 = fig.colorbar(im5, ax=axs[1, 1], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar6 = fig.colorbar(im6, ax=axs[1, 2], orientation='horizontal', fraction=0.05, pad=0.02)

            cbar4.ax.tick_params()
            cbar5.ax.tick_params()
            cbar6.ax.tick_params()

            cbar4.set_label('Flux')
            cbar5.set_label('Flux')
            cbar6.set_label('Flux')

            # Finalize plot appearance
            axs[1,0].axis('off')
            axs[1,1].axis('off')
            axs[1,2].axis('off')

            plt.subplots_adjust(hspace=1)  # Increase vertical gap between rows
            plt.show()

        return model, delta
        

    def _denoise_iterative_hard(self, num_iter=20):
        """
        Perform iterative hard thresholding with L0 regularization and fixed sparsity.
        
        This method implements an iterative hard thresholding algorithm that maintains
        a fixed sparsity pattern (binary mask) determined from the first iteration. 
        The algorithm alternates between:
        1. Applying the fixed threshold mask to preserve significant coefficients
        2. Extracting additional signal from residuals using the same mask
        3. Monitoring convergence through plateau detection
        
        The L0 regularization promotes sparsity while the iterative approach allows
        recovery of faint signals that become apparent as noise is reduced.

        Parameters
        ----------
        num_iter : int, optional
            Maximum number of iterations to perform, by default 20.
            Algorithm may converge earlier based on plateau condition.
            Typical range: 10-50 iterations depending on noise level and complexity.

        Returns
        -------
        best_model : np.ndarray
            Denoised data cube at the iteration with lowest residual standard deviation.
            Shape matches input cube dimensions (nz, ny, nx).
        deltas : np.ndarray
            Accumulated extracted signals from all iterations after the first.
            Shows total signal recovered through iterative residual analysis.
        residual_stds : list of float
            Standard deviation of residuals at each iteration.
            Used for convergence monitoring and algorithm diagnostics.
        best_iteration : int
            Iteration number where the best model was achieved.
            Indicates when optimal denoising occurred.
        noise_levels : list of float
            Estimated noise levels for each wavelet sub-band.
            Used for threshold scaling and algorithm validation.
            
        Notes
        -----
        Algorithm Details:
        - First iteration: Compute wavelet transform, estimate noise, create binary mask
        - Subsequent iterations: Apply fixed mask to residuals, extract new signal
        - Convergence: Plateau detection with multiple tolerance levels (p=4 to 0)
        - Constraints: Positivity enforced at each step for physical realism
        
        The hard thresholding approach:
        - Sets coefficients to zero if below threshold (creates sparsity)
        - Preserves coefficients unchanged if above threshold
        - Maintains sharp features but may introduce artifacts
        - Suitable for sources with compact, well-defined structure
        
        Convergence Strategy:
        - Tries plateau conditions from strict (p=4) to relaxed (p=0)
        - Monitors relative change in residual standard deviation
        - Returns best model if convergence not achieved
        - Epsilon tolerance: 1e-3 for stability detection
        
        See Also
        --------
        _denoise_iterative_soft : Soft thresholding alternative with adaptive weights
        _residual_signal_extraction_l0 : Core residual processing for hard thresholding
        _generate_hard_threshold_mask : Binary mask creation for sparsity
        """

        if self._verbose:
            print('L0 regularisation : HARD thresholding')
            print('{} denoising iterations'.format(num_iter))

        # Initialize the model with the input noisy data
        model = self._data.copy()

        # Initialize tracking variables for algorithm performance
        flux_history = []
        residual_stds = []
        previous_residual_std = 1e-33  # Small value to avoid division by zero

        # Find peak signal location for diagnostic plotting reference
        max_voxel_index = np.argmax(self._signal)  # Get flattened index
        iz, max_y, max_x = np.unravel_index(max_voxel_index, self._signal.shape)  # Convert to 3D index

        # Track accumulated signal extracted from residual analysis
        deltas = np.zeros_like(self._data)

        # Convergence parameters
        p_init = 4  # Start with strict plateau condition
        epsilon = 1e-3  # Relative tolerance for plateau detection
        
        converged = False

        

        # Multi-level convergence strategy: try different plateau conditions
        for p in range(p_init, -1, -1):  # Try p from 4 down to 0
            if self._verbose:
                print(f'\n[*] Trying with plateau condition: {p} consecutive stable residuals needed for convergence')

            # Reset convergence tracking for this plateau level
            plateau_counter = 0
            previous_residual_std = 1e-33  # Reset each time p changes
            best_model = None
            best_iteration = 0
            min_residual_std = np.inf

            # Initialize diagnostic tracking
            dists = []
            noise_levels = [] 
            
            # Main iterative denoising loop
            for iteration in range(num_iter):

                if self._verbose: 
                    print('\n\n--- [ DE-NOISING ITERATION #{} ] ---\n'.format(iteration + 1))

                if iteration == 0:
                    # FIRST ITERATION: Establish the sparsity pattern through hard thresholding
                    
                    # Transform input data to wavelet domain for coefficient analysis
                    if self._verbose: 
                        print('(*) Decomposing the noisy data into wavelet coefficients')
                    inds, shapes, w_data = self.mr2d1d.decompose(self._data,
                                                                self._num_scales_2d,
                                                                self._num_scales_1d)

                    # Initialize binary mask arrays for coefficient selection
                    mask_coeff = np.zeros_like(w_data)
                    mask_coeff_final = np.zeros_like(w_data)

                    if self._verbose: 
                        print('(*) Applying hard thresholding in the wavelet space based on the threshold scale (lambda = {} in noise units)'.format(self._threshold_level))

                    # Process each wavelet sub-band to create significance mask
                    for scale2d in range(self._num_scales_2d):
                        for scale1d in range(self._num_scales_1d):
                            
                            start, end = inds[scale2d][scale1d]

                            # Skip the coarse scale (contains low-frequency approximation)
                            if scale2d == self._num_scales_2d - 1 and scale1d == self._num_scales_1d - 1:
                                continue

                            # Extract coefficients for this specific scale combination
                            c_data = w_data[start:end]

                            # Estimate noise level for this sub-band using robust statistics
                            noise_level = self._estimate_noise(c_data)

                            # Apply hard thresholding to create binary significance mask
                            thresh = self._threshold_level
                            mask_coeff[start:end] = self._generate_hard_threshold_mask(c_data, thresh, noise_level)

                    if self._verbose: 
                        print('(*) Calculating a mask in wavelet space...')
                        print('(*) Applying the mask to perform denoising for the first iteration')
                    
                    # Apply the binary mask to retain only significant coefficients
                    w_data = w_data * mask_coeff

                    # Reconstruct the denoised cube from filtered coefficients
                    if self._verbose: 
                        print('(*) Reconstructing the denoised data from wavelet to real space')
                    model = self.mr2d1d.reconstruct(w_data)
                    
                    # Apply positivity constraint for astrophysical realism
                    if self._verbose: 
                        print('(*) Applying positivity constraint')
                    model = np.maximum(0, model)
                    model_print = model.copy()  # Save for potential diagnostics

                else:
                    # SUBSEQUENT ITERATIONS: Extract additional signal from residuals
                    
                    # Use the fixed mask from first iteration to analyze residuals
                    model, delta = self._residual_signal_extraction_l0(model, mask_coeff, iteration)

                    if iteration == 1 and self._verbose:
                        print('Goal: Find previously unnoticed signal in residuals')
                        print('(*) Calculating residual for this iteration')

                    # Accumulate all extracted signal components
                    deltas += delta

                # Compute performance metrics for this iteration
                aperture_flux = np.sum(model)  # Total flux in current model
                residual_std = np.std(model)   # Standard deviation as quality metric

                if self._verbose:
                    print(f"(*) Aperture Flux: {aperture_flux:.3e}, Residual noise std: {residual_std:.3e}")

                # Track performance history for diagnostics
                flux_history.append(aperture_flux)
                residual_stds.append(residual_std)

                # Update best model tracking (lowest residual std indicates best quality)
                if residual_std < min_residual_std:
                    min_residual_std = residual_std
                    best_model = model.copy()
                    best_iteration = iteration + 1

                # Convergence detection via plateau condition
                # Check if residual std has stabilized (relative change < epsilon)
                if abs(residual_std - previous_residual_std) / previous_residual_std <= epsilon:
                    plateau_counter += 1
                else:
                    plateau_counter = 0  # Reset counter if not stable

                # Check if we've reached the required plateau length
                if plateau_counter >= p:
                    if self._verbose:
                        print(f'\nflux: {aperture_flux}')
                        print(f'noise: {residual_std}')
                        print(f'Convergence achieved at iteration #{iteration + 1} with p = {p}')
                    converged = True
                    break
                else:
                    previous_residual_std = residual_std

                # Progress reporting for subsequent iterations
                if iteration == 1 and self._verbose:
                    print(f'(*) Repeating these steps for subsequent {num_iter - 2} iterations')

            # Exit plateau loop if convergence achieved with current p value
            if converged:
                break

        # Handle case where convergence was not achieved with any plateau condition
        if not converged:
            if self._verbose:
                print('[Warning] Convergence not achieved for any p value from 4 to 0')
                print(f'Using best model at iteration #{best_iteration} with residual std = {min_residual_std:.3e}')

        return best_model, deltas, residual_stds, best_iteration, noise_levels






    

#--------
#  SOFT
#--------

       

    def _residual_signal_extraction_l1(self, model, mask_coeff, all_weights, iteration, noise_levels):
        """
        Perform residual signal extraction using L1 soft-thresholding with adaptive weights.

        This method implements one iteration of weighted soft-thresholding for the debiasing 
        step in iterative soft-thresholding algorithms. Unlike hard thresholding that uses 
        binary masks, this approach applies adaptive weights that account for the bias 
        introduced by soft-thresholding in previous iterations.

        The weighted soft-thresholding approach:
        1. Decomposes residuals into wavelet domain
        2. Applies location-dependent weights to counter soft-thresholding bias
        3. Uses different strategies for previously detected vs. undetected coefficients
        4. Reconstructs and applies positivity constraints

        Parameters
        ----------
        model : np.ndarray
            Current estimate of the denoised cube from previous iteration.
            Shape (nz, ny, nx) matching input data dimensions.
        mask_coeff : np.ndarray
            Boolean mask indicating coefficients previously identified as significant.
            Used to apply different thresholding strategies to different regions.
        all_weights : np.ndarray
            Adaptive weights for soft-thresholding computed from previous iterations.
            These weights compensate for bias introduced by soft-thresholding.
        iteration : int
            Current iteration number for progress reporting and conditional processing.
        noise_levels : list of float
            Pre-computed noise level estimates for each wavelet sub-band.
            Avoids recomputation and ensures consistency across iterations.

        Returns
        -------
        model : np.ndarray
            Updated model after incorporating newly extracted residual signal.
            Same shape as input with additional recovered features.
        delta : np.ndarray
            Extracted signal component added to model in this iteration.
            Shows incremental improvement from residual analysis.

        Notes
        -----

        Processing Steps:
        1. Calculate residual between current model and observed data
        2. Transform residual to wavelet domain
        3. Apply scale-dependent median centering
        4. Perform weighted soft-thresholding by sub-band
        5. Reconstruct spatial signal from processed coefficients
        6. Apply positivity constraint and update model

        See Also
        --------
        _residual_signal_extraction_l0 : Hard thresholding alternative for L0 regularization
        _denoise_iterative_soft : Main soft-thresholding algorithm using this method
        """
        
        # Find peak signal location for diagnostic plotting reference
        max_voxel_index = np.argmax(self._signal)  # Get flattened index
        iz, _, _ = np.unravel_index(max_voxel_index, self._signal.shape)  # Convert to 3D index

        # Calculate residual between observed data and current model estimate
        residual = self._data - model
        thresh = self._threshold_level
        # Optional diagnostic plotting for algorithm visualization
        if self._plot:
            # Create comprehensive 2x3 subplot layout for iteration tracking
            fig, axs = plt.subplots(2, 3, figsize=(16, 13), constrained_layout=True)

            # Top row: Model progression and signal comparison
            im1 = axs[0,0].imshow(model[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,0].set_title('Previously Denoised (Iteration #{})'.format(iteration))

            im2 = axs[0,1].imshow((self._signal - model)[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,1].set_title('SIGNAL Residual')

            im3 = axs[0,2].imshow(self._signal[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,2].set_title('SIGNAL')

            # Remove axis ticks for cleaner appearance
            axs[0,0].axis('off')
            axs[0,1].axis('off')
            axs[0,2].axis('off')

            # Add colorbars with consistent formatting
            cbar1 = fig.colorbar(im1, ax=axs[0, 0], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar2 = fig.colorbar(im2, ax=axs[0, 1], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar3 = fig.colorbar(im3, ax=axs[0, 2], orientation='horizontal', fraction=0.05, pad=0.02)

            # Configure colorbar appearance
            cbar1.ax.tick_params()
            cbar2.ax.tick_params()
            cbar3.ax.tick_params()

            cbar1.set_label('Flux')
            cbar2.set_label('Flux')
            cbar3.set_label('Flux')

        # Progress reporting for debiasing iterations
        if iteration==1:
            if self._verbose: print('(*) Decomposing residual into wavelet scales')

        # Transform residual into wavelet domain for coefficient-wise processing
        inds, shapes, w_residual = self.mr2d1d.decompose(residual,
                                                            self._num_scales_2d,
                                                            self._num_scales_1d)

        # Progress indication for weighted debiasing process
        if iteration==0:
            if self._verbose: print('(*) Performing Weighted de-biasing with previously calculated weights')
        
        # Initialize sub-band counter for noise level indexing
        i=0 
        # Process each wavelet sub-band with adaptive weighted soft-thresholding
        for scale2d in range(self._num_scales_2d):
            for scale1d in range(self._num_scales_1d):

                # Get coefficient indices for current scale combination
                start, end = inds[scale2d][scale1d]

                # Skip coarse scale (contains low-frequency approximation)
                if scale2d==self._num_scales_2d-1 and scale1d == self._num_scales_1d - 1:
                    w_residual[start:end] = np.zeros_like(w_residual[start:end])
                    continue  # Do not threshold the coarse scale

                # Extract and center coefficients for this sub-band
                # Median subtraction removes potential bias in residual coefficients
                c_data = w_residual[start:end] - np.median(w_residual[start:end])

                # Use pre-computed noise level for this sub-band (more stable than recomputation)
                noise_level = noise_levels[i]

                # Extract mask and weights for adaptive processing
                mask = mask_coeff[start:end].astype(bool)  # Convert to boolean for indexing
                weights = all_weights[start:end]

                # Apply differential soft-thresholding based on previous detection status
                
                # For previously detected coefficients (mask=True): use adaptive weights
                # Weights compensate for bias introduced by soft-thresholding in earlier iterations
                w_residual[start:end][mask] = np.sign(c_data[mask]) * np.maximum(
                    np.abs(c_data[mask]) - weights[mask] * thresh * noise_level, 0.0
                )

                # For previously undetected coefficients (mask=False): standard soft-thresholding
                # Use uniform threshold without bias correction
                w_residual[start:end][~mask] = np.sign(c_data[~mask]) * np.maximum(
                    np.abs(c_data[~mask]) - thresh * noise_level, 0.0
                )

                # Increment sub-band counter for noise level indexing
                i+=1





        # Reconstruct spatial signal from processed wavelet coefficients
        if iteration==0:
            if self._verbose: print('(*) Reconstructing the new signal coefficients into the real space')
        
        # Inverse wavelet transform to recover spatial structure
        # Note: commented operations were for experimental bias correction
        # w_residual = w_residual - np.mean(w_residual)  # Global mean removal
        # w_residual -= w_residual.mean()                # Alternative mean removal
        delta = self.mr2d1d.reconstruct(w_residual)

        # Post-processing and model update
        if iteration==0:
            if self._verbose: print('(*) Updating the model with the newly detected signal')
            
        # Apply positivity constraint to extracted signal (astrophysical realism)
        delta = np.maximum(0, delta)
        
        # Optional flux conservation (currently disabled)
        # These lines would normalize delta to match residual flux
        residual_flux = np.sum(residual)
        delta_flux = np.sum(delta)
        # if delta_flux != 0:
        #     delta *= residual_flux / delta_flux

        # Remove mean to prevent systematic flux bias accumulation
        delta -= delta.mean()

        # Update model with extracted signal
        model = model + delta
        
        # Note: Model positivity constraint applied at calling level
        if iteration==0:
            if self._verbose: print('(*) Applying the positivity constraint')
        # model = np.maximum(0, model)  # Commented - handled externally
        # Complete diagnostic plotting if enabled
        if self._plot:
            # Bottom row: Residual analysis and updated model visualization
            im4 = axs[1,0].imshow(residual[iz], cmap = 'RdBu_r')
            axs[1,0].set_title('Residual')

            im5 = axs[1,1].imshow(delta[iz], cmap = 'RdBu_r', vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,1].set_title('Residual Information')

            im6 = axs[1,2].imshow(model[iz], cmap = 'RdBu_r', vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,2].set_title('Updated Model (Iteration #{})'.format(iteration+1))

            # Add colorbars for bottom row
            cbar4 = fig.colorbar(im4, ax=axs[1, 0], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar5 = fig.colorbar(im5, ax=axs[1, 1], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar6 = fig.colorbar(im6, ax=axs[1, 2], orientation='horizontal', fraction=0.05, pad=0.02)

            # Configure colorbar formatting
            cbar4.ax.tick_params()
            cbar5.ax.tick_params()
            cbar6.ax.tick_params()

            cbar4.set_label('Flux')
            cbar5.set_label('Flux')
            cbar6.set_label('Flux')

            # Clean up plot appearance
            axs[1,0].axis('off')
            axs[1,1].axis('off')
            axs[1,2].axis('off')

            # Adjust layout and display
            plt.subplots_adjust(hspace=1)  # Increase vertical gap between rows
            plt.show()

        return model, delta
    

    def _denoise_iterative_soft(self, num_iter_reweight=20, num_iter_debias=20, debias = True):

        """
        Perform iterative soft-thresholding denoising on the 3D data cube.

        This method applies multiple re-weighting iterations followed by an optional
        debiasing step to recover the underlying signal from noisy data. It uses a 
        2D-1D multiscale wavelet decomposition for denoising, adaptive thresholding, 
        and plateau-based convergence criteria.

        Parameters
        ----------
        num_iter_reweight : int, optional
            Number of iterations for the re-weighting denoising step (default is 20).
        num_iter_debias : int, optional
            Number of iterations for the debiasing step to extract residual signal (default is 20).
        debias : bool, optional
            Whether to perform the debiasing step (default is True).

        Returns
        -------
        best_model : np.ndarray
            Final denoised model after all iterations.
        model_1_step : np.ndarray
            Model after the first re-weighted denoising iteration.
        model_no_reweight : np.ndarray
            Model obtained without re-weighting in the first iteration.
        deltas : np.ndarray
            Accumulated residual signals extracted during debiasing.
        residual_stds_reweight : list of float
            Standard deviations of residuals during the re-weighting step.
        residual_stds_debias : list of float
            Standard deviations of residuals during the debiasing step.
        best_iteration : int
            Iteration number where the best model (lowest residual std) was achieved.
        dists : list of np.ndarray
            Selected wavelet sub-band distributions for diagnostic purposes.
        noise_levels : list of float
            Estimated noise levels for each wavelet sub-band.

        Soft-Thresholding with Adaptive Weights:
        - For masked regions (previously detected): weighted soft-thresholding
        - For unmasked regions: standard soft-thresholding 
        - Weights inversely related to coefficient magnitude (debias larger coefficients more)
        - Coarse scale (approximation) excluded from thresholding

        The L1 formulation promotes sparsity while preserving gradients:
        - Shrinks coefficients toward zero by threshold amount
        - Maintains sign information (unlike hard thresholding)
        - Reduces over-smoothing through adaptive weighting
        - Better preserves extended emission compared to hard thresholding

        Notes
        -------
        - Positivity is enforced on the denoised models at each step.
        - Convergence is determined via a plateau condition on the residual standard deviation.
        - Optional plotting shows intermediate and final results for diagnostics.
        """

        if self._verbose:
            
            print('----[ Denosiing with ITERATIVE SOFT THRESHOLDING ]----')



        # Initialize the model
        self.mean, self.std = np.mean(self._data), np.std(self._data)
        model = self._data.copy()
        thresh = self._threshold_level
        
        
        p_init = 1

        max_voxel_index = np.argmax(self._signal)  # Get flattened index
        iz, max_y, max_x = np.unravel_index(max_voxel_index, self._signal.shape)  # Convert to 3D index

        converged = False
        for p in range(p_init, -1, -1):  # Try p from 5 down to 0
            if self._verbose:
                print(f'\n[*] Trying with plateau condition: {p} consecutive stable residuals needed for convergence')

            plateau_counter = 0
            previous_residual_std = 1e-33  # Reset each time p changes
            min_residual_std = np.inf
            dists = []
            noise_levels = []
            epsilon = 1e-3


            flux_history = []
            residual_stds_reweight = []

            inds, shapes, w_data_data = self.mr2d1d.decompose(self._data,
                                            self._num_scales_2d,
                                            self._num_scales_1d)
            
            for scale2d in range(self._num_scales_2d):
                    for scale1d in range(self._num_scales_1d):
                        start, end = inds[scale2d][scale1d]
                        if (scale2d==5) and (scale1d==0):
                            dists.append(w_data_data[start:end])



            
            for iteration in range(num_iter_reweight):


            
                if self._verbose:   print('\n\n--- [ DE-NOISING ITERATION #{} ] ---\n'.format(iteration+1))



                if iteration==0:
                    if self._verbose: print('(*) Decomposing noisy data into wavelet coefficients')
                inds, shapes, w_data_weights = self.mr2d1d.decompose(model,
                                            self._num_scales_2d,
                                            self._num_scales_1d)
                            
        
                
                #model_update = np.zeros_like(model)
                #model_update = data3d.copy()

                mu = 0.5
                gradient = -2*(self._data - model)

                model_update = model - mu * gradient


                inds, shapes, w_data = self.mr2d1d.decompose(model_update,
                                                            self._num_scales_2d,
                                                            self._num_scales_1d)
            
                if iteration == 0:
                    if self._verbose: print('(*) The gradient in the first iteration is 0')


                if self._verbose:
                    if iteration==1:
                        print('(*) Updating model with gradient with respect to data')
                        print('(*) Calculating weights for each iteration (except #1) to account for the soft thresholding bias')
                # Loop through all sub-bands (excluding last/coarse scale) to find the max usable threshold based on noise levels in each

                w_data_copy = w_data.copy()

                mask_coeff = np.zeros_like(w_data, dtype=bool)

                all_weights = np.ones_like(w_data)

                noise_levels = []

                for scale2d in range(self._num_scales_2d):
                    for scale1d in range(self._num_scales_1d):

                        start, end = inds[scale2d][scale1d]

                        # Skip the coarse scale (approximation band) in the 1D spectral transform
                        #if scale2d==self._num_scales_2d-1 and scale1d == self._num_scales_1d - 1: 
                            #continue  # Do not threshold the coarse scale


                        #w_data[start:end]  -= np.median(w_data[start:end])
                        c_data = w_data[start:end] 
                        #c_data -= c_data.median()
                        
                        noise_level = self._estimate_noise(c_data)
                        noise_levels.append(noise_level)

                        noise_level_weight = self._estimate_noise(w_data_weights[start:end])

                    
                        # Compute the mask
                        mask = np.abs(c_data) > thresh * noise_level




                        # Compute weights only where mask is True
                        if iteration == 0:
                            weights = np.ones_like(c_data)
                        else:
                            weights = thresh*noise_level_weight/(np.abs(w_data_weights[start:end]) + 1e-33)

                        all_weights[start:end] = weights
                        mask_coeff[start:end] = mask
                        # Apply soft-thresholding with adaptive weights where mask is True
                        w_data[start:end][mask] = np.sign(c_data[mask]) * np.maximum(
                            np.abs(c_data)[mask] - weights[mask] * thresh * noise_level, 0.0
                        )

                        # Apply uniform soft-thresholding where mask is False
                        w_data[start:end][~mask] = np.sign(c_data[~mask]) * np.maximum(
                            np.abs(c_data)[~mask] - thresh * noise_level, 0.0
                        )

                    

                        #w_data[start:end] -= w_data[start:end].mean()
                        #c_data[~mask] = 0.0
                        #w_data[start:end][~mask] = 0.0
                        



                        if (scale2d==5) and (scale1d==0):
                            dists.append(w_data[start:end])
                            noise_levels.append(noise_level)

                        if self._plot:
                            if (scale2d==5) and (scale1d==0):

                                denosied_dist = w_data[start:end]
                                threshold_noise = thresh * noise_level

                                bins = np.linspace(w_data_copy[start:end].min(), w_data_copy[start:end].max(),100)

                                plt.figure(figsize = (11,7))
                                plt.hist(w_data[start:end], bins = bins, color = 'xkcd:blue', alpha = 0.5, label = 'Denoised')
                                #plt.hist(w_data_weights[start:end], bins = bins, color = 'xkcd:blue', alpha = 0.5, label = 'Denoised')
                                plt.hist(w_data_copy[start:end], bins = bins, histtype='step', color = 'black', alpha = 1, label = 'Original')
                                plt.axvline(thresh * noise_level, color = 'black', linestyle = 'dashed', label = '{:.1f}'.format(self._threshold_level)+r'$\sigma$' + ' Threshold')
                                plt.axvline(-thresh * noise_level, color = 'black', linestyle = 'dashed')
                                #plt.title('Iteration {}\n2D: {}, 1D: {}'.format(iteration+1, scale2d+1, scale1d+1))
                                plt.yscale('log')
                                plt.ylim(0,5e5)
                                #plt.xlim(-0.005, 0.005)
                                plt.ylabel('$N_{C_{ij}}$')
                                plt.xlabel('$C_{ij}$')
                                plt.legend()

                                plt.grid(True)
                                plt.show()


                # Reconstruct the image from the updated coefficients


                if iteration==0:
                        if self._verbose: print('(*) Reconstructing the new signal coefficients into the real space')
        
                model_denoised = self.mr2d1d.reconstruct(np.ascontiguousarray(w_data, dtype=np.float32))
                model_print = self.mr2d1d.reconstruct(np.ascontiguousarray(w_data_copy, dtype=np.float32))


                if iteration==0:
                        if self._verbose: print('(*) Applying the positivity constraint')
                model_denoised = np.maximum(0, model_denoised) #positivity constraint


                
                if self._plot:
                    plt.figure(figsize = (15,12))
                    plt.subplot(221)
                    plt.imshow(model_print[iz], cmap = 'RdBu_r',  vmin=self._data[iz].min(), vmax=self._data[iz].max())
                    plt.colorbar()
                    plt.axis('off')
                    plt.title('Input')

                    plt.subplot(222)
                    plt.imshow(model_denoised[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
                    plt.colorbar()
                    plt.axis('off')
                    plt.title('Denoised Iteration #{}'.format(iteration+1))

                    plt.figure(figsize = (15,12))
                    plt.subplot(223)
                    plt.imshow(self._signal[iz], cmap = 'RdBu_r',  vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
                    plt.colorbar()
                    plt.axis('off')
                    plt.title('Signal')

                    plt.subplot(224)
                    plt.imshow((self._signal - model_denoised)[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
                    plt.colorbar()
                    plt.axis('off')
                    plt.title('SIGNAL Residual')

                    
                    plt.show()


            
                

                if iteration==0:
                    if self._verbose: print('(*) Repeating these steps for subsequent iterations')

                model = model_denoised# - model_denosied.mean()

                aperture_flux = np.sum(model)
                residual_std = np.std(self._data - model)

                residual_stds_reweight.append(residual_std)

                if self._verbose:
                    print(f"(*) Aperture Flux: {aperture_flux:.3e}, Clean Flux: {np.sum(self._signal):.3e}, Residual STD: {residual_std:.3e}")


                # Track best model so far
                if residual_std < min_residual_std:
                    min_residual_std = residual_std
                    best_model = model.copy()
                    best_iteration = iteration + 1

                # Plateau condition check
                if abs(residual_std - previous_residual_std) / previous_residual_std <= epsilon:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if plateau_counter >= p:
                    if self._verbose:
                        print(f'\nflux: {aperture_flux}')
                        print(f'noise: {residual_std}')
                        print(f'Re-weight Convergence achieved at iteration #{iteration + 1} with p = {p}')
                    converged = True
                    break
                else:
                    previous_residual_std = residual_std

                if iteration == 1 and self._verbose:
                    print(f'(*) Repeating these steps for subsequent {num_iter_reweight - 2} iterations')

                if iteration==0:
                    model_no_reweight = model_denoised

            if converged:
                break

        if not converged:
            if self._verbose:
                print('[Warning] Re-weight convergence not achieved for any p value from 5 to 0')
                print(f'Using best model at iteration #{best_iteration} with residual std = {min_residual_std:.3e}')

        


        if self._verbose:   '\n----[ DE-BIASING ]----\n'

        if self._verbose:   print('(*) Iteratively extracting remaining signal from residual')


        p_init_debias=1
        epsilon_debias = 5e-4
        model_1_step = best_model.copy()

        for p in range(p_init_debias, -1, -1):  # Try p from 5 down to 0
            if self._verbose:
                print(f'\n[*] Trying with plateau condition: {p} consecutive stable residuals needed for convergence')

            plateau_counter = 0
            previous_residual_std = 1e-33  # Reset each time p changes
            deltas = np.zeros_like(model)
            min_residual_std = np.inf
            residual_stds_debias=[]

            model = model_1_step.copy()

            for iteration in range(num_iter_debias):

                model, delta = self._residual_signal_extraction_l1(model, mask_coeff, all_weights, iteration, noise_levels)
                model = np.maximum(0, model)  # Apply positivity constraint
                deltas+=delta


                aperture_flux = np.sum(model)
                residual_std = np.std(self._data - model)

                if self._verbose:
                    print(f"(*) Aperture Flux: {aperture_flux}, Residual STD: {residual_std:.3e}")


                
                flux_history.append(aperture_flux)
                residual_stds_debias.append(residual_std)

                # Track best model so far
                if residual_std < min_residual_std:
                    min_residual_std = residual_std
                    best_model = model.copy()
                    best_iteration = iteration + 1


                # Plateau condition check
                if abs(residual_std - previous_residual_std) / previous_residual_std <= epsilon_debias:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if plateau_counter >= p:
                    if self._verbose:
                        print(f'\nflux: {aperture_flux}')
                        print(f'noise: {residual_std}')
                        print(f'Convergence achieved at iteration #{iteration + 1} with p = {p}')
                    converged = True
                    break
                else:
                    previous_residual_std = residual_std

                if iteration == 1 and self._verbose:
                    print(f'(*) Repeating these steps until convergence')

            if converged:
                break

        if not converged:
            if self._verbose:
                print('[Warning] Convergence not achieved for any p value from 5 to 0')
                print(f'Using best model at iteration #{best_iteration} with residual std = {min_residual_std:.3e}')

        if self._plot:
            plt.figure(figsize=(28, 11))
            plt.subplot(121)
            plt.imshow(self._data[iz], cmap='RdBu_r')
            plt.title('Noisy Data')
            plt.colorbar()

            plt.subplot(122)
            plt.imshow(self._signal[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
            plt.title('Clean Signal')
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(28, 9))
            plt.subplot(131)
            plt.imshow(model_1_step[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
            plt.title('One-Step Denoising')
            plt.colorbar()
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(np.maximum(deltas, 0)[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
            plt.title('Residual Signal')
            plt.colorbar()
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(best_model[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
            plt.title('Final Denoised')
            plt.colorbar()
            plt.axis('off')
            plt.show()

                
        return best_model, model_1_step, model_no_reweight, deltas, residual_stds_reweight, residual_stds_debias, best_iteration, dists, noise_levels #, denosied_dist, threshold_noise






    def _compute_emission_rmse(self, model):
        """
        Compute Root Mean Square Error in emission regions only.
        
        Calculates reconstruction error specifically in regions identified as containing
        emission, providing a focused metric for algorithm performance assessment.
        This is more meaningful than full-cube RMSE when the signal is spatially localized.

        Parameters
        ----------
        model : np.ndarray
            Reconstructed/denoised data cube with same shape as original signal.

        Returns
        -------
        float
            Root mean square error between masked signal and model.
            Lower values indicate better reconstruction in emission regions.
            
        Notes
        -----
        Uses the emission mask stored in self._mask to focus error calculation
        on scientifically relevant regions. Ignores background/noise-only regions
        that may dominate error metrics but are less important for analysis.
        """
        # Apply emission mask to both signal and model, then compute RMSE
        masked_diff = self._mask * self._signal - self._mask * model
        rmse = np.sqrt(np.mean(masked_diff ** 2))
        return rmse

    @staticmethod
    def _prox_positivity_constraint(array):
        """
        Apply positivity constraint (proximal operator) to enforce physical realism.
        
        Implements the proximal operator for the non-negativity constraint, which is
        essential for astrophysical applications where negative flux values are typically
        unphysical. This constraint is commonly used in iterative algorithms to ensure
        the solution remains in the feasible set of non-negative values.
        
        The proximal operator of the indicator function for the positive orthant is
        simply the projection onto the non-negative values, implemented as max(0, x).

        Parameters
        ----------
        array : np.ndarray
            Input array that may contain negative values to be constrained.
            Can be any dimensional array (1D, 2D, 3D, etc.).

        Returns
        -------
        np.ndarray
            Array with same shape as input but with all negative entries set to zero.
            Positive values are preserved unchanged.
            
        Notes
        -----
        This is a key component in constrained optimization for astronomical imaging:
        - Preserves flux conservation for positive sources
        - Eliminates unphysical negative flux artifacts
        - Maintains algorithmic convergence properties
        - Has zero computational cost (element-wise maximum operation)
        
        In the context of denoising, this constraint:
        - Prevents noise from creating negative flux regions
        - Maintains the physical interpretation of flux measurements
        - Helps preserve source morphology by avoiding flux cancellation
        
        Examples
        --------
        >>> data = np.array([-1, 0, 2, -3, 5])
        >>> constrained = _prox_positivity_constraint(data)
        >>> print(constrained)  # [0, 0, 2, 0, 5]
        """
        return np.maximum(0, array)


    def _estimate_noise(self, array):
        """
        Estimate noise standard deviation using robust Median Absolute Deviation (MAD).
        
        This method provides a robust estimate of the noise level in wavelet coefficients
        that is less sensitive to outliers (signal) than the standard deviation. The MAD
        is particularly effective when the data contains a mixture of noise and signal,
        as it focuses on the central distribution of values.
        
        The conversion factor 1.48 transforms the MAD into an estimate of the standard
        deviation under the assumption of Gaussian noise distribution.

        Parameters
        ----------
        array : np.ndarray
            Array of values (typically wavelet coefficients) for noise estimation.
            Should contain a mixture of noise and signal, where noise dominates.
            Common inputs: fine-scale wavelet coefficients, background regions.

        Returns
        -------
        float
            Estimated noise standard deviation.
            Represents the characteristic noise level in the input array.
            
        Notes
        -----
        The MAD estimator is defined as:
            MAD = median(|X - median(X)|)
            
        For Gaussian noise, the relationship between MAD and standard deviation is:
            σ ≈ 1.48 × MAD
            
        This method is robust against outliers (up to ~50% contamination) making it
        ideal for astronomical data where signal and noise coexist. It works well
        even when signal occupies a significant fraction of the data.
        
        Alternative noise estimation methods:
        - Standard deviation: Fast but sensitive to signal contamination
        - Iterative sigma clipping: More accurate but computationally expensive
        - Scale estimation from background regions: Requires prior source detection
        
        References
        ----------
        Rousseeuw, P. J. & Croux, C. (1993). "Alternatives to the Median Absolute Deviation."
        Journal of the American Statistical Association, 88, 1273-1283.
        """
        # Calculate median of the array (robust location estimator)
        median_val = np.median(array)
        
        # Compute absolute deviations from the median
        abs_dev = np.abs(array - median_val)
        
        # Calculate the median of absolute deviations (MAD)
        mad = np.median(abs_dev)
        
        # Convert MAD to standard deviation estimate for Gaussian noise
        # Factor 1.48 ≈ 1/Φ^(-1)(3/4) where Φ^(-1) is inverse normal CDF
        return 1.48 * mad


def mock_noise_value(mock_cube, peak_snr):
    """
    Calculate noise level for synthetic data cube based on desired peak SNR.
    
    This utility function determines the noise standard deviation needed to achieve
    a specified signal-to-noise ratio at the peak of a synthetic data cube. Used
    for creating realistic noise realizations in synthetic IFU observations.

    Parameters
    ----------
    mock_cube : np.ndarray
        Clean synthetic data cube without noise.
        The peak value will be used as the reference signal level.
    peak_snr : float
        Desired signal-to-noise ratio at the peak of the cube.
        Typical values: 5-50 for astronomical observations.
        Higher values create cleaner data, lower values increase noise challenge.

    Returns
    -------
    float
        Noise standard deviation to achieve the desired peak SNR.
        Used as σ in noise_cube = σ × randn(cube.shape) for additive Gaussian noise.
        
    Notes
    -----
    The relationship is: noise_sigma = peak_signal / desired_SNR
    where peak_signal = max(mock_cube).
    
    This assumes the peak represents the strongest emission feature and that
    noise will be uniformly distributed with standard deviation noise_sigma.
    
    For realistic astronomical noise modeling, consider:
    - Poisson noise from photon statistics (signal-dependent)
    - Read noise from detector electronics (constant level)
    - Background noise from sky emission (spatially variable)
    - Calibration uncertainties (systematic effects)
    
    Examples
    --------
    >>> noise_std = mock_noise_value(clean_cube, peak_snr=10)
    >>> noise_cube = noise_std * np.random.randn(*clean_cube.shape)
    >>> noisy_cube = clean_cube + noise_cube
    """
    # Calculate the maximum signal value in the cube
    peak_signal = np.max(mock_cube)
    
    # Determine noise level to achieve desired SNR at peak
    mock_cube_noise = peak_signal / peak_snr
    
    # Progress reporting for noise level verification
    if True:  # Could be controlled by verbose parameter
        print(f'Max SNR: {peak_snr}')
        print(f'Mock noise level: {mock_cube_noise:.6e}')

    return mock_cube_noise