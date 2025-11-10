"""
Synthetic IFU Spectral Cube Dataset Generator
============================================

This module generates realistic synthetic spectral cubes for Integral Field Unit (IFU)
observations, simulating galaxy observations with proper astronomical physics.

The pipeline creates:
1. 3D galaxy models with Sérsic spatial profiles and exponential vertical distributions
2. Rotation curves based on analytical models 
3. 3D velocity fields with realistic kinematics
4. Spectral cubes with velocity binning and cosmological effects
5. Observational effects like beam convolution and noise

Scientific Applications:
- Training data for denoising algorithms
- Testing analysis pipelines on known ground truth
- Understanding observational biases in IFU surveys
- Validating moment map and kinematic analysis methods

Key Features:
- Realistic galaxy morphologies (Sérsic profiles)
- Analytical rotation curves with physical velocity dispersions
- Proper coordinate transformations and rotations
- Variable spatial resolution (resolved vs unresolved)
- Multiple galaxy systems with Hubble flow
- Beam convolution and noise simulation

Classes:
- ResolvedSpectralCubeDataset: Main class for generating high-resolution cubes
- FinalSpectralCubeDataset: PyTorch dataset wrapper with observational effects
"""

import numpy as np
import os
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
import random
from astropy.convolution import Gaussian2DKernel, convolve_fft
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functions import *
from astropy.cosmology import FlatLambdaCDM
import matplotlib.patches as patches
from astropy import units as u
from scipy.ndimage import gaussian_filter1d

# Standard flat ΛCDM cosmology for distance calculations
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


class ResolvedSpectralCubeDataset():
    """
    Generate synthetic 3D spectral cubes simulating IFU observations of galaxies.
    
    This class creates realistic synthetic observations by:
    1. Modeling 3D galaxy structure with Sérsic profiles
    2. Computing rotation curves and velocity fields  
    3. Applying coordinate transformations and rotations
    4. Creating spectral cubes with velocity binning
    5. Accounting for cosmological effects and observational parameters
    
    The synthetic data includes multiple resolution scenarios (resolved, unresolved, or mixed)
    to study how spatial resolution affects galaxy property measurements.
    
    Parameters
    ----------
    n_gals : int, optional
        Number of galaxies per cube. If None, randomly samples 1-3 galaxies.
    n_cubes : int, default 1
        Number of spectral cubes to generate.
    resolution : str, default 'all'
        Resolution scenario:
        - 'all': Mixed resolution (r = 0.25-4, where r = Re/beam_radius)
        - 'unresolved': Low resolution (r = 0.25 - 1, galaxies smaller than beam)  
        - 'resolved': High resolution (r = 1-4, galaxies larger than beam)
        - 'visualise': Special mode with fixed r values [0.3, 1.2, 2, 3, 5]
    offset_gals : float, default 5
        Maximum spatial offset between galaxies in pixel units.
    beam_size : float, default 5
        Telescope beam size in pixels (FWHM).
    init_grid_size : int, default 101
        Initial grid size for individual galaxy generation.
    final_grid_size : int, default 125  
        Final grid size for combined spectral cube.
    n_spectral_slices : int, default 40
        Number of velocity channels (will be multiplied by 5 internally).
    fname : str, optional
        Custom directory path for saving output cubes.
    verbose : bool, default True
        Enable detailed progress reporting.
    plot : bool, default False
        Generate diagnostic plots during processing.
    seed : int, optional
        Random seed for reproducible results.
        
    Attributes
    ----------
    spectral_cubes : list
        Generated 3D spectral cubes [n_velocity, n_y, n_x].
    system_params : list  
        Physical and observational parameters for each cube.
    resolution : str
        Resolution mode setting.
    beam_size_px : float
        Beam size in pixels.
        
    Methods
    -------
    milky_way_rot_curve_analytical(R, v_0, R_e, n)
        Calculate rotation velocity at radius R using an empirical analytical model.
    redshift_from_kpc_scale(s, smin, smax, zmin, zmax)  
        Empirical relation converting spatial scale to redshift.
    sersic_intensity_3d(x, y, z, Ie, Re, n, hz)
        Compute 3D Sérsic profile with exponential vertical component.
    rotated_system(params_gal_rot)
        Generate single galaxy with rotation and velocity field.
    make_spectral_cube(rotated_disks, rotated_vel_z_cubes, pix_spatial_scale)
        Combine multiple galaxies into velocity-binned spectral cube.
    
    Examples
    --------
    >>> # Generate resolved galaxy cubes
    >>> dataset = ResolvedSpectralCubeDataset(
    ...     n_gals=2, n_cubes=5, resolution='resolved', 
    ...     beam_size=4, verbose=True, seed=42)
    >>> 
    >>> # Access first cube and parameters
    >>> cube, params = dataset[0]
    >>> print(f"Cube shape: {cube.shape}")
    >>> print(f"Pixel scale: {params['pix_spatial_scale']:.3f} kpc/pixel")
    
    Notes
    -----
    - Galaxies follow Sérsic surface brightness profiles in the disk plane
    - Vertical structure follows exponential scale heights  
    - Rotation curves use analytical approximations to realistic galaxy models
    - Velocity dispersions include both rotational and random motions
    - Output cubes are in Jy/pixel units with proper angular scale conversion
    """
    def __init__(self, n_gals=None, n_cubes=1, resolution='all', offset_gals=5, beam_size = 5, init_grid_size=101, final_grid_size=125, n_spectral_slices=40, fname=None, verbose=True, plot=False, seed=None):

        # Initialize random seeds for reproducible results
        #self.central_Re_kpc = 5 #kpc

        # Store configuration parameters
        self.resolution = resolution
        self.plot = plot
        self.fname = fname
        self.seed = seed
        if self.seed is not None:
            # Set all random number generators for consistency
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        # Galaxy separation parameter (affects interaction dynamics)
        self.offset_gals = offset_gals

        # Determine number of galaxies per cube
        if not n_gals:
            # Randomly sample 1-3 galaxies per cube for variety
            self.n_gals = np.random.randint(1, 3, n_cubes)
        else:
            # Fixed number of galaxies across all cubes
            self.n_gals = [n_gals for _ in range(n_cubes)]

        # Grid and observational parameters
        self.n_cubes = n_cubes
        self.init_grid_size = init_grid_size        # Size for individual galaxy generation
        self.final_grid_size = final_grid_size      # Size for combined output cube
        self.n_spectral_slices = 5*n_spectral_slices + 1  # 5x oversampling + 1 for binning
        self.beam_size_px = beam_size               # Telescope beam FWHM in pixels
        self.verbose = verbose
        self._verbose = verbose
        # Initialize storage arrays for galaxy and system parameters
        self.spectral_cubes = []           # Final 3D spectral cubes
        self.system_params = []            # Observational and physical parameters
        self.all_gal_vz_sigmas = []       # Velocity dispersion along line of sight
        self.all_gal_x_angles = []        # Rotation angles about X-axis (inclination)
        self.all_gal_y_angles = []        # Rotation angles about Y-axis (position angle)
        self.all_Re = []                  # Effective radii in pixels
        self.all_hz = []                  # Vertical scale heights in pixels
        self.all_Ie = []                  # Effective surface brightness
        self.all_n = []                   # Sérsic indices
        self.all_pix_spatial_scales = []  # Physical scale per pixel in kpc
        self.all_gal_v_0 = []            # Characteristic rotation velocity


        # =======================================================================
        # RESOLUTION PARAMETER SETUP
        # =======================================================================
        # Define the ratio r = Re/beam_radius to control spatial resolution
        # r < 1: Unresolved (galaxy smaller than beam)
        # r > 1: Resolved (galaxy larger than beam)
        
        if self.resolution != 'visualise':
            if self.resolution == 'all':
                # Mixed resolution: wide range from unresolved to well-resolved
                r_min = 0.25   # Heavily beam-dominated
                r_max = 4      # Well-resolved structure
            elif self.resolution == 'unresolved':
                # Unresolved scenario: galaxy smaller than beam
                r_min = 0.25
                r_max = 1
            elif self.resolution == 'resolved':
                # Resolved scenario: galaxy much larger than beam
                r_min = 1
                r_max = 4
                
            # Log-uniform sampling to ensure good coverage across orders of magnitude
            log_r_min = np.log10(r_min)
            log_r_max = np.log10(r_max)
            log_r = np.random.uniform(log_r_min, log_r_max, size=n_cubes)
            r = 10 ** log_r

        else:
            # Special visualization mode with fixed resolution values
            r=np.asarray([0.3,1.2,2,3,5])

        # Convert resolution ratio to effective radius in pixels
        # Re = r * (beam_size / 2) gives effective radius
        Re_central = r * self.beam_size_px /2


        # =======================================================================
        # GALAXY PARAMETER GENERATION  
        # =======================================================================
        # Generate physical parameters for each galaxy system
        
        # Fixed effective radius in physical units (could be varied)
        central_Re_kpc = np.random.uniform(5, 5, n_cubes)  # Central Re in kpc

        # Generate parameters for each cube
        for i in range(n_cubes):

            # Calculate pixel scale: kpc per pixel
            pix_spatial_scale = central_Re_kpc[i] / Re_central[i]  # Scale in pixels relative to Re

            # Primary galaxy parameters
            Re = [Re_central[i]]                               # Effective radius in pixels
            hz = [np.random.uniform(0.5, 1) / pix_spatial_scale]  # Scale height (thinner in high-res)
            
            # Adjust surface brightness based on resolution
            # Unresolved galaxies get higher flux to compensate for smaller size
            if r[i]>1:
                # Resolved galaxies: standard surface brightness
                Ie = [np.random.uniform(0.2, 1) * 1e-11] 
            else:
                # Unresolved galaxies: enhanced surface brightness
                Ie = [np.random.uniform(0.2, 1) * (1e-11)*5] 
                
            # Orientation angles (disk inclination and position angle)
            gal_x_angles = [np.random.uniform(0, 85)]   # Inclination: 0°=face-on, 90°=edge-on
            gal_y_angles = [np.random.uniform(0, 85)]   # Position angle in sky plane
            
            n_gal = self.n_gals[i]

            # Generate satellite galaxies if multi-galaxy system
            if n_gal > 1:
                # Satellites are smaller and fainter than the primary
                Re += list(np.random.uniform(Re[0]/3, Re[0]/2, n_gal - 1))
                hz += list(np.random.uniform(hz[0]/3, hz[0]/2, n_gal - 1))
                Ie += list(np.random.uniform(Ie[0]/3, Ie[0]/2, n_gal - 1))
                # Random orientations for satellites
                gal_x_angles += list(np.random.uniform(-180, 180, n_gal - 1))
                gal_y_angles += list(np.random.uniform(-180, 180, n_gal - 1))

            # Store parameters for this cube's galaxies
            self.all_pix_spatial_scales.append(np.full(n_gal, pix_spatial_scale))
            self.all_gal_vz_sigmas.append(np.random.uniform(30, 50, n_gal))      # Velocity dispersion 30-50 km/s
            self.all_gal_x_angles.append(np.asarray(gal_x_angles))
            self.all_gal_y_angles.append(np.asarray(gal_y_angles))
            self.all_Re.append(np.asarray(Re))
            self.all_hz.append(np.asarray(hz))
            self.all_Ie.append(np.asarray(Ie))
            self.all_n.append(np.random.uniform(0.5, 1.5, n_gal))               # Sérsic index: 0.5-1.5
            self.all_gal_v_0.append(np.random.uniform(200, 200, n_gal))         # Rotation velocity: fixed at 200 km/s

        # Initialize cube generation process
        self.fname = fname
        self._generate_cubes()


    # ==========================================================================
    # STATIC METHODS FOR GALAXY PHYSICS
    # ==========================================================================

    @staticmethod
    def milky_way_rot_curve_analytical(R,v_0, R_e,n):
        """
        Calculate rotation velocity using an analytical galaxy rotation curve model.
        
        Based on empirical fits to observed galaxy rotation curves, this function
        computes the circular velocity at a given galactocentric radius.
        
        Parameters
        ----------
        R : float or array_like
            Galactocentric radius in kpc where rotation velocity is calculated.
        v_0 : float
            Characteristic rotation velocity in km/s (typically 200-300 km/s).
        R_e : float
            Effective radius in kpc (scale length of the galaxy).
        n : float
            Sérsic index affecting the shape of the rotation curve.
            
        Returns
        -------
        vel : float or array_like
            Circular rotation velocity in km/s at radius R.
            
        Notes
        -----
        - Uses analytical approximation: v(R) = v_0 * 1.022 * (R/R_0)^0.0803
        - R_0 is computed from effective radius and Sérsic index
        - The form approximates realistic galaxy rotation curves
        - Valid for disk-dominated galaxies at moderate radii
        
        References
        ----------
        Based on empirical relations from galaxy kinematic studies.
        See https://www.aanda.org/articles/aa/pdf/2017/05/aa30540-17.pdf
        """
        # Sérsic parameter calculation using series expansion
        bn_func = lambda k: 2 * k - 1/3 + 4 / (405 * k) + 46 / (25515 * k**2) + 131 / (1148175 * k**3) - 2194697 / (30690717750 * k**4)
        bn = bn_func(n)

        # Scale radius derived from effective radius and Sérsic index
        R_0 = 2*(R_e/((bn)**n))

        # Analytical rotation curve with empirically-motivated parameters
        vel = v_0 * 1.022 * np.power((R/R_0),0.0803)
        return vel
        
        #ref: https://www.aanda.org/articles/aa/pdf/2017/05/aa30540-17.pdf


    @staticmethod
    def redshift_from_kpc_scale(s, smin=1.0, smax=4.0, zmin=3.5, zmax=6.5):
        """
        Empirical relation to assign redshift based on spatial scale.
        
        Converts spatial resolution (kpc/pixel) to approximate redshift using
        a linear relationship. This provides realistic redshifts for synthetic
        galaxies based on their apparent size.
        
        Parameters
        ----------
        s : float
            Spatial scale in kpc/pixel.
        smin, smax : float, default 1.0, 4.0
            Range of spatial scales in kpc/pixel.
        zmin, zmax : float, default 3.5, 6.5
            Corresponding redshift range.
            
        Returns
        -------
        z : float
            Estimated redshift based on spatial scale.
            
        Notes
        -----
        - Higher spatial scales (worse resolution) correspond to higher redshifts
        - This is an approximation for simulation purposes
        - Real observations have more complex scale-redshift relationships
        """
        return zmin + (s - smin) * (zmax - zmin) / (smax - smin)


    @staticmethod
    def sersic_intensity_3d(x, y, z, Ie, Re, n, hz):
        """
        Compute 3D Sérsic surface brightness profile for a galaxy disk.
        
        Combines a Sérsic profile in the disk plane (x-y) with exponential
        fall-off in the vertical direction (z-axis). This represents the
        3D light distribution of a typical disk galaxy.
        
        Parameters
        ----------
        x, y, z : array_like
            3D spatial coordinate grids in physical units (kpc).
        Ie : float
            Surface brightness at the effective radius in arbitrary units.
        Re : float  
            Effective radius in the same units as x, y coordinates.
        n : float
            Sérsic index controlling profile shape:
            - n = 1: Exponential disk (typical for disk galaxies)
            - n = 4: de Vaucouleurs profile (elliptical galaxies)
            - 0.5 < n < 1.5: Range used for this simulation
        hz : float
            Exponential scale height in z-direction (kpc).
            
        Returns
        -------
        I : array_like
            3D surface brightness distribution matching input coordinate shape.
            
        Notes
        -----
        - Uses circular symmetry (axis ratio q = 1) in disk plane
        - Sérsic parameter bn calculated using series expansion approximation
        - Vertical profile: I_z(z) = exp(-|z|/hz)
        - Radial profile: I_r(r) = Ie * exp(-bn * ((r/Re)^(1/n) - 1))
        - Total profile: I(x,y,z) = I_r(r) * I_z(z)
        
        References
        ----------
        Sérsic profile: Sérsic, J. L. 1963, Boletín de la Asociación Argentina 
        de Astronomía, 6, 41
        """
        # Assume circular disk (could be generalized to elliptical)
        q = 1 

        # Calculate Sérsic parameter bn using series expansion
        bn_func = lambda k: 2 * k - 1/3 + 4 / (405 * k) + 46 / (25515 * k**2) + 131 / (1148175 * k**3) - 2194697 / (30690717750 * k**4)
        bn = bn_func(n)
        
        # Compute elliptical radius in disk plane
        r_elliptical = np.sqrt(x**2 + (y / q)**2)
        
        # Sérsic profile in the disk plane
        profile_xy = np.exp(-bn * ((r_elliptical / Re)**(1/n) - 1))
        
        # Exponential profile in vertical direction
        profile_z = np.exp(-np.abs(z) / hz)
        
        # Combined 3D profile
        I = Ie * profile_xy * profile_z

        return I




    def rotated_system(self, params_gal_rot):
        """
        Generate a single galaxy with 3D structure, kinematics, and coordinate rotations.
        
        This method creates the core 3D galaxy model by:
        1. Computing 3D Sérsic surface brightness distribution
        2. Calculating rotation curve and velocity field
        3. Applying coordinate transformations for realistic viewing angles
        4. Generating diagnostic plots if requested
        
        Parameters
        ----------
        params_gal_rot : dict
            Galaxy parameters containing:
            - 'pix_spatial_scale': Physical scale in kpc/pixel
            - 'Re': Effective radius in pixels
            - 'hz': Vertical scale height in pixels
            - 'Ie': Effective surface brightness
            - 'n': Sérsic index
            - 'gal_x_angle': Rotation angle about X-axis (inclination)
            - 'gal_y_angle': Rotation angle about Y-axis (position angle)
            - 'gal_vz_sigma': Velocity dispersion in km/s
            - 'redshift': Cosmological redshift
            - 'v_0': Characteristic rotation velocity in km/s
            
        Returns
        -------
        rotated_disk_xy : ndarray, shape (grid_size, grid_size, grid_size)
            3D surface brightness distribution after rotations.
        rotated_vel_z_cube_xy : ndarray, shape (grid_size, grid_size, grid_size)
            3D line-of-sight velocity field after rotations.
            
        Notes
        -----
        - Creates initial galaxy on grid with size self.init_grid_size
        - Applies cosmological surface brightness dimming: I ∝ (1+z)^-3
        - Velocity field includes rotation curve + random velocity dispersion
        - Coordinate rotations simulate realistic viewing angles
        - Line-of-sight velocities include projection effects
        """

        # Extract galaxy parameters from input dictionary
        pix_spatial_scale = params_gal_rot['pix_spatial_scale']
        Re_kpc = params_gal_rot['Re']*pix_spatial_scale      # Convert to physical units
        hz_kpc = params_gal_rot['hz']*pix_spatial_scale      # Convert to physical units
        Ie = params_gal_rot['Ie']
        n = params_gal_rot['n']
        angle_x = params_gal_rot['gal_x_angle']              # Inclination angle
        angle_y = params_gal_rot['gal_y_angle']              # Position angle
        sigma_vz = params_gal_rot['gal_vz_sigma']            # Velocity dispersion
        redshift = params_gal_rot['redshift']
        v_0 = params_gal_rot['v_0']                          # Rotation velocity scale


        #--------------------------------------------------------------------------------------------------------------------------#
        #                                          § GENERATING THE 3D SPATIAL CUBE §                                              # 
        #--------------------------------------------------------------------------------------------------------------------------#

        grid_size = self.init_grid_size
        centre = np.array([(grid_size - 1) / 2] * 3)    # Center of the 3D grid

        # Create 3D coordinate grid centered at origin
        if self._verbose:
            print('Calculating the flux density values at each spatial location')
        x = np.arange(grid_size) - (grid_size - 1) / 2  # Pixel coordinates
        y = np.arange(grid_size) - (grid_size - 1) / 2
        z = np.arange(grid_size) - (grid_size - 1) / 2
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Convert pixel coordinates to physical coordinates (kpc)
        X_kpc = X * pix_spatial_scale
        Y_kpc = Y * pix_spatial_scale
        Z_kpc = Z* pix_spatial_scale

        # Compute 3D galaxy surface brightness profile
        # Apply cosmological surface brightness dimming: I ∝ (1+z)^-3
        disk = self.sersic_intensity_3d(X_kpc, Y_kpc, Z_kpc, Ie, Re_kpc, n, hz_kpc)/ (1 + redshift)**3

        #--------------------------------------------------------------------------------------------------------------------------#
        #                                  § Calculating the velocity magnitudes and vectors §                                     # 
        #--------------------------------------------------------------------------------------------------------------------------#


        if self._verbose:
            print('Calculating and assigning velocity vectors...')
        vel_x_cube = np.zeros((grid_size, grid_size, grid_size))
        vel_y_cube = np.zeros((grid_size, grid_size, grid_size))
        vel_z_cube = np.zeros((grid_size, grid_size, grid_size))

        vel_mag_cube = np.zeros((grid_size, grid_size, grid_size))


        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):

                    coords = np.asarray([i,j,k])

                    pos_vect = coords[:2] - centre[:2]

                    tangent_vect = np.cross(pos_vect, [0,0,1])

                    r = np.linalg.norm(pos_vect)*pix_spatial_scale

                    velocity_mag_value = self.milky_way_rot_curve_analytical(r,v_0, Re_kpc, n)

                    if r !=0:   
                        tangent_unit_vect = tangent_vect/np.linalg.norm(tangent_vect)
                    else:
                        tangent_unit_vect = np.array([0,0,0])

                    vel_x_cube[i,j,k], vel_y_cube[i,j,k], vel_z_cube[i,j,k] = (velocity_mag_value * tangent_unit_vect[0]), (velocity_mag_value * tangent_unit_vect[1]), np.random.normal(0, sigma_vz)

                    vel_mag_cube[i,j,k] = velocity_mag_value



        #--------------------------------------------------------------------------------------------------------------------------#
        #                                                       § Rotations §                                                      # 
        #--------------------------------------------------------------------------------------------------------------------------#



        axes = [(0,2), (1,2)]


        rotation_angles = np.asarray([angle_x, angle_y, 0])



        #------------------------------------------- § Rotating/transforming the system § ---------------------------------------- # 

        if self._verbose:
            print('Rotating {:.2f} degrees about X axis and {:.2f} degrees about Y axis:'.format(rotation_angles[0], rotation_angles[1]))
            print('1. Rotating/transforming the whole system...')

        rotated_disk_x = rotate(disk, rotation_angles[0], axes=axes[0], reshape=False,)
        rotated_disk_xy = rotate(rotated_disk_x, rotation_angles[1], axes=axes[1], reshape=False)

        transformed_vel_z_cube_x = rotate(vel_z_cube, rotation_angles[0], axes=axes[0], reshape=False)
        transformed_vel_z_cube_xy = rotate(transformed_vel_z_cube_x, rotation_angles[1], axes=axes[1], reshape=False)

        transformed_vel_y_cube_x = rotate(vel_y_cube, rotation_angles[0], axes=axes[0], reshape=False)
        transformed_vel_y_cube_xy = rotate(transformed_vel_y_cube_x, rotation_angles[1], axes=axes[1], reshape=False)

        transformed_vel_x_cube_x = rotate(vel_x_cube, rotation_angles[0], axes=axes[0], reshape=False)
        transformed_vel_x_cube_xy = rotate(transformed_vel_x_cube_x, rotation_angles[1], axes=axes[1], reshape=False)

        
        #------------------------------------------ § Rotating the velocity vectors § ---------------------------------------- # 

        if self._verbose:
            print('2: Rotating the individual velocity vectors...')

        rotated_vel_z_cube_xy = np.zeros((grid_size,grid_size,grid_size))

        rotation = R.from_euler('yxz', rotation_angles, degrees=True)
        rotation_matrix = rotation.as_matrix()

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):

                    vel_vector = np.asarray([transformed_vel_x_cube_xy[i,j,k], transformed_vel_y_cube_xy[i,j,k], transformed_vel_z_cube_xy[i,j,k]])
                    rotated_vel_vector_xy = rotation_matrix @ vel_vector
                    rotated_vel_z_cube_xy[i,j,k] = rotated_vel_vector_xy[2]

        if self.plot:
            plt.figure(figsize=(27, 5))
            center = grid_size // 2
            radius = self.beam_size_px/ 2
            # Subplot 1: Rotated Disk XY
            ax1 = plt.subplot(1, 4, 1)
            im1 = ax1.imshow(np.sum(rotated_disk_xy, axis=2), origin='lower', cmap='RdBu_r')
            ax1.axhline(center, color='cyan', linestyle='--', linewidth=1)
            ax1.axvline(center, color='cyan', linestyle='--', linewidth=1)
            circle1 = patches.Circle((center, center), radius, edgecolor='yellow', facecolor='none', linewidth=1.5)
            ax1.add_patch(circle1)
            plt.colorbar(im1, ax=ax1, label='Flux Density')
            ax1.set_title('Rotated Disk XY Projection')
            ax1.set_xlabel('X-axis')
            ax1.set_ylabel('Y-axis')

            # Subplot 2: Rotated Disk XZ
            ax2 = plt.subplot(1, 4, 2)
            im2 = ax2.imshow(np.sum(rotated_disk_xy, axis=1), origin='lower', cmap='RdBu_r')
            circle2 = patches.Circle((center, center), radius, edgecolor='yellow', facecolor='none', linewidth=1.5)
            ax2.add_patch(circle2)
            plt.colorbar(im2, ax=ax2, label='Flux Density')
            ax2.set_title('Rotated Disk XZ Projection')
            ax2.set_xlabel('X-axis')
            ax2.set_ylabel('Z-axis')

            # Subplot 3: Rotated Disk YZ
            ax3 = plt.subplot(1, 4, 3)
            im3 = ax3.imshow(np.sum(rotated_disk_xy, axis=0), origin='lower', cmap='RdBu_r')
            circle3 = patches.Circle((center, center), radius, edgecolor='yellow', facecolor='none', linewidth=1.5)
            ax3.add_patch(circle3)
            plt.colorbar(im3, ax=ax3, label='Flux Density')
            ax3.set_title('Rotated Disk YZ Projection')
            ax3.set_xlabel('Y-axis')
            ax3.set_ylabel('Z-axis')

            # Subplot 4: Rotated Velocity XY
            ax4 = plt.subplot(1, 4, 4)
            im4 = ax4.imshow(np.sum(rotated_vel_z_cube_xy, axis=2), origin='lower', cmap='viridis')
            circle4 = patches.Circle((center, center), radius, edgecolor='red', facecolor='none', linewidth=1.5)
            ax4.add_patch(circle4)
            plt.colorbar(im4, ax=ax4, label='Velocity (km/s)')
            ax4.set_title('Rotated Velocity XY Projection')
            ax4.set_xlabel('X-axis')
            ax4.set_ylabel('Y-axis')

            plt.tight_layout()
            plt.show()

        return rotated_disk_xy, rotated_vel_z_cube_xy


    def make_spectral_cube(self, rotated_disks, rotated_vel_z_cubes, pix_spatial_scale):

        init_grid_size = self.init_grid_size
        final_grid_size = self.final_grid_size
        n_spectral_slices = self.n_spectral_slices
        n_galaxies = len(rotated_disks)
        assert n_galaxies == len(rotated_vel_z_cubes), "Mismatch between disks and velocity cubes"

        redshift = self.redshift_from_kpc_scale(pix_spatial_scale)

        center_final_cube = np.array([(final_grid_size + 1) / 2] * 3)
        offset_range_1 = 0
        offset_range_2 = self.offset_gals #/pix_spatial_scale

        galaxy_centers = []

        half_size = init_grid_size // 2
        min_pos = half_size
        max_pos = final_grid_size - half_size

        # First galaxy near the center
        x = int(np.clip(center_final_cube[0] + np.random.randint(-offset_range_1, offset_range_1 + 1), min_pos, max_pos - 1))
        y = int(np.clip(center_final_cube[1] + np.random.randint(-offset_range_1, offset_range_1 + 1), min_pos, max_pos - 1))
        z = int(np.clip(center_final_cube[2] + np.random.randint(-offset_range_1, offset_range_1 + 1), min_pos, max_pos - 1))
        galaxy_centers.append(np.array([x, y, z]))

        # Additional galaxies nearby but offset

        for i in range(1, n_galaxies):
            x = int(np.clip(galaxy_centers[0][0] + np.random.randint(-offset_range_2, offset_range_2 + 1), min_pos, max_pos - 1))
            y = int(np.clip(galaxy_centers[0][1] + np.random.randint(-offset_range_2, offset_range_2 + 1), min_pos, max_pos - 1))
            z = int(np.clip(galaxy_centers[0][2] + np.random.randint(-offset_range_2, offset_range_2 + 1), min_pos, max_pos - 1))
            galaxy_centers.append(np.array([x, y, z]))

        if self._verbose:
            for idx, center in enumerate(galaxy_centers):
                print(f"Centre of galaxy {idx + 1}: {center}")

        # Apply Hubble flow relative to the first galaxy
        reference_z = galaxy_centers[0][2]
        H_z = cosmo.H(redshift).value  # km/s/Mpc

        for i in range(1, n_galaxies):
            delta_z_kpc = (galaxy_centers[i][2] - reference_z)*pix_spatial_scale
            delta_z_mpc = delta_z_kpc * 1e-3  # Convert kpc to Mpc
            relative_velocity = H_z * delta_z_mpc

            if self._verbose:
                direction = "farther" if delta_z_kpc > 0 else "closer"
                print(f"Galaxy {i+1} is {direction} than galaxy 1 by {delta_z_kpc:.2f} kpc")
                print(f"→ Adjusting velocity cube by {relative_velocity:.2f} km/s")

            # Add velocity offset to simulate redshift/blueshift
            rotated_vel_z_cubes[i] = (rotated_vel_z_cubes[i]+relative_velocity)
        



        # Creating lower and upper limits for the velocity observation bins
        # Create velocity bin edges across all galaxies

        all_velocities = np.array([vel_cube for vel_cube in rotated_vel_z_cubes])

        #vel_limit = self.milky_way_rot_curve_analytical(4*5)*(1+redshift)  # Maximum velocity at 4 Re
        min_vel = -600 #np.min([np.min(v) for v in all_velocities])
        max_vel = 600 #np.max([np.max(v) for v in all_velocities])

        limit = np.max([abs(min_vel), abs(max_vel)])  # Use the maximum absolute value for limits

        limits = np.linspace(-limit, limit, n_spectral_slices)

        if self._verbose:
            print('Overlaying all galaxy observations in a bigger spatial grid')
            print('Calculating the projected flux density of every voxel within the limits in each velocity slice')

        spectral_cube_I_px = []
        average_vels = np.zeros((n_spectral_slices - 1))


        for i in range(n_spectral_slices - 1):
            combined_cube = np.zeros((final_grid_size, final_grid_size, final_grid_size))
            for _, (disk, vel_cube, center) in enumerate(zip(rotated_disks, rotated_vel_z_cubes, galaxy_centers)):

                # Determine the voxels within current velocity bin
                if i < n_spectral_slices - 2:
                    condition = (vel_cube >= limits[i]) & (vel_cube < limits[i+1])
                else:
                    condition = (vel_cube >= limits[i]) & (vel_cube <= limits[i+1])  # include last edge
                selected_cube = np.zeros_like(disk)
                selected_cube[np.where(condition)] = disk[np.where(condition)]


                # Insert selected cube into the larger grid at the galaxy's center position
                xg, yg, zg = center
                half_size = init_grid_size // 2
                if init_grid_size % 2 == 0:
                    xs, xe = xg - half_size, xg + half_size
                    ys, ye = yg - half_size, yg + half_size
                    zs, ze = zg - half_size, zg + half_size
                else:
                    xs, xe = xg - half_size, xg + half_size + 1
                    ys, ye = yg - half_size, yg + half_size + 1
                    zs, ze = zg - half_size, zg + half_size + 1


                combined_cube[xs:xe, ys:ye, zs:ze] += selected_cube

           
            # Projecting along the LoS (Z-axis)
            spectral_slice = np.sum(combined_cube, axis=2)
            spectral_cube_I_px.append(spectral_slice)  # Transpose if needed

            # Store average velocity of this slice
            average_vel = np.mean([limits[i], limits[i+1]])
            average_vels[i] = average_vel

        spectral_cube_I_px = np.array(spectral_cube_I_px)

        # Angular diameter distance
        DA = cosmo.angular_diameter_distance(redshift).to(u.kpc)  # [kpc]

        # Pixel angular scale
        theta_pix_rad = (pix_spatial_scale * u.kpc / DA).decompose().value  # [radians]
        theta_pix_arcsec = np.degrees(theta_pix_rad) * 3600  # Convert to arcseconds

        solid_ang_sr = theta_pix_rad**2  # Solid angle in steradians

        spectral_cube_Jy_px = (spectral_cube_I_px * 1e23 * solid_ang_sr)

        spectral_cube_Jy_px = spectral_cube_Jy_px.reshape(spectral_cube_Jy_px.shape[0]//5, 5, spectral_cube_Jy_px.shape[1], spectral_cube_Jy_px.shape[2]).mean(axis=1)  

        # You can update the params_gals dictionary as needed
        params_gen = {
            'galaxy_centers': galaxy_centers,
            'average_vels': average_vels,
            'beam_size_px': self.beam_size_px,
            'n_gals': n_galaxies,
            'pix_spatial_scale': pix_spatial_scale,
        }


        if self.plot:
            plt.figure(figsize=(10, 5))
            ax = plt.subplot(111)
            im=ax.imshow(np.sum(spectral_cube_Jy_px, axis=0), cmap='RdBu_r', origin='lower')
            ax.set_title('Projected Flux Density of All Slices')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            center = self.final_grid_size // 2
            circle = patches.Circle((center, center), self.beam_size_px/2, edgecolor='yellow', facecolor='none', linewidth=1)
            ax.add_patch(circle)
            #ax.add_patch(circle1)
            plt.colorbar(im, ax=ax, label='Flux Density')
            plt.show()

        return spectral_cube_Jy_px, params_gen



    def _generate_cubes(self):

        print(f'\n[ § Creating {self.n_cubes} highly resolved cubes of dimensions {self.n_spectral_slices/5-1} (spectral) x {self.final_grid_size} x {self.final_grid_size} (spatial) § ]\n')

        for i in range(self.n_cubes):

            if self._verbose:
                    print(f'\n\n\u00a7--------------------- Creating cube # {i + 1} ---------------------\u00a7', end='\r')

            rotated_disks = []
            rotated_vel_z_cubes = []

            for j in range(self.n_gals[i]):

                params_gal_rot = {
                    'Re': self.all_Re[i][j],
                    'hz': self.all_hz[i][j],
                    'Ie': self.all_Ie[i][j],
                    'n': self.all_n[i][j],
                    'gal_x_angle': self.all_gal_x_angles[i][j],
                    'gal_y_angle': self.all_gal_y_angles[i][j],
                    'gal_vz_sigma': self.all_gal_vz_sigmas[i][j],
                    'gal_vz_sigma': self.all_gal_vz_sigmas[i][j],
                    'pix_spatial_scale': self.all_pix_spatial_scales[i][j],
                    'redshift': self.redshift_from_kpc_scale(self.all_pix_spatial_scales[i][j]),
                    'v_0': self.all_gal_v_0[i][j]
                }

                if self.verbose:
                    print(f'\nCreating disk #{j+1}...')


                rotated_disk, rotated_vel_z_cube = self.rotated_system(params_gal_rot)

                if self.verbose:
                    print(f'Disk #{j+1} generated!')

                rotated_disks.append(rotated_disk)
                rotated_vel_z_cubes.append(rotated_vel_z_cube)


            if self.verbose:
                print('\nCreating spectral cube...')
       
            spectral_cube_final, params = self.make_spectral_cube(rotated_disks, rotated_vel_z_cubes, self.all_pix_spatial_scales[i][0])


            self.system_params.append(params)

            if self.verbose:
                print('\nSpectral cube created!')


            #Setting possible negative values to 0
            spectral_cube_final = np.maximum(spectral_cube_final, 0)

            self.spectral_cubes.append(spectral_cube_final)

            if self.fname is None:
                fname_save = '/Users/arnablahiry/Work/data/toy_cubes/datasets/raw_data/{}_{}_{}'.format(self.n_spectral_slices-1, self.final_grid_size, self.n_cubes)
                if not os.path.exists(fname_save):
                    os.makedirs(fname_save)
            else:
                fname_save = self.fname
                if not os.path.exists(fname_save):
                    os.makedirs(fname_save)
                    
            np.save(fname_save+'/cube_{}.npy'.format(i+1),spectral_cube_final)

            if self._verbose:
                print('saved as ' + fname_save + '/cube_{}.npy'.format(i+1))

        self.spectral_cubes = np.asarray(self.spectral_cubes)




    def __len__(self):
        return self.n_cubes

    def __getitem__(self, idx):

        return self.spectral_cubes[idx], self.system_params[idx]



class FinalSpectralCubeDataset(Dataset):
    """
    PyTorch dataset wrapper for synthetic spectral cubes with observational effects.
    
    This class takes the high-resolution cubes from ResolvedSpectralCubeDataset and
    applies realistic observational effects to create training data for denoising
    algorithms and analysis pipelines.
    
    Processing Steps:
    1. Load resolved spectral cubes from saved dataset
    2. Apply beam convolution to simulate telescope resolution
    3. Add realistic noise based on peak signal-to-noise ratio
    4. Normalize data for machine learning applications
    5. Provide PyTorch dataset interface for training
    
    Parameters
    ----------
    n_spectral_slices : int
        Number of velocity channels in the spectral cubes.
    final_grid_size : int
        Spatial grid size (assumes square grids).
    fname : str, optional
        Custom path to saved resolved dataset. If None, uses default path.
    verbose : bool, default True
        Enable detailed progress reporting.
    transform : callable, optional
        Optional data transformations (not currently used).
    seed : int, optional
        Random seed for reproducible noise generation.
    peak_snrs : list, optional
        Specific peak SNR values for each cube. If None, randomly samples 2.5-10.
    cube_norm_params : tuple, optional
        Pre-computed (mean, std) for normalization. If None, computes from data.
        
    Attributes
    ----------
    processed_data : list
        Precomputed cubes with observational effects applied.
    mean, std : float
        Dataset normalization parameters.
    resolved_dataset : ResolvedSpectralCubeDataset
        Source high-resolution dataset.
        
    Methods
    -------
    return_stats()
        Return dataset normalization statistics.
    sanitize_params(params)
        Clean parameter dictionaries for consistent data types.
    __getitem__(idx)
        Return (noisy_cube, clean_cube, metadata, velocities) for training.
        
    Examples
    --------
    >>> # Create dataset with specific SNR values
    >>> dataset = FinalSpectralCubeDataset(
    ...     n_spectral_slices=40, final_grid_size=125,
    ...     peak_snrs=[5.0, 7.5, 10.0], seed=42)
    >>> 
    >>> # Get training sample
    >>> noisy, clean, meta, vels = dataset[0]
    >>> print(f"SNR: {meta[0]:.1f}, Scale: {meta[1]:.3f} kpc/pix")
    
    Notes
    -----
    - Applies beam convolution using convolve_beam() function
    - Noise follows realistic IFU noise characteristics
    - Normalization uses global statistics across all cubes
    - Returns standardized PyTorch tensors ready for training
    - Metadata includes SNR, pixel scale, and number of galaxies
    """
    def __init__(self, n_spectral_slices, final_grid_size, fname = None,verbose = True, transform=None, seed=None, peak_snrs=None, cube_norm_params=None):

        self.cube_norm_params = cube_norm_params
        self.peak_snrs = peak_snrs
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        
        
        if not fname:
            self.folder_path = f'/home/alahiry/data/toy_cubes/datasets/resolved_dataset_{n_spectral_slices}_{final_grid_size}_{final_grid_size}.pkl'
        else:
            self.folder_path = fname+f'resolved_dataset_{n_spectral_slices}_{final_grid_size}.pkl'

        # Load the saved resolved dataset
        with open(self.folder_path, 'rb') as file:
            self.resolved_dataset = pickle.load(file)


        print(f'\n[ § Constructing final (convolved and noisy) dataset of {self.resolved_dataset.__len__()} cubes of dimensions {n_spectral_slices} (spectral) x {final_grid_size} x {final_grid_size} (spatial) § ]\n')


        self.transform = transform
        #self.files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.npy')])
        self.verbose = verbose
        

        if self.verbose:    print(f'(*) Loading resolved cubes from {self.folder_path}')
        if self.verbose:    print('(*) Uniformly sampling beam width and peak SNR (noise level) for each cube')


        
        # Precompute everything
        self.processed_data = self._precompute()
        
        # Compute dataset-wide mean and standard deviation

        


    def _precompute(self):
        """Precompute convolved and noisy versions of the cubes"""
        all_voxels = []

        if self.verbose:
            print('(*) Convolving each cube with chosen beam width and overlaying additive white Gaussian noise')

        processed_data = []

        for i in range(self.resolved_dataset.__len__()):

            cube = self.resolved_dataset[i][0]
            gal_system_params = self.resolved_dataset[i][1]
            self.beam_size_px = gal_system_params['beam_size_px']
            
            

            if self.peak_snrs:
                peak_snr = self.peak_snrs[i]
            
            else:
                peak_snr = random.uniform(2.5, 10)
            
            # Apply transformations
            #cube_lsf = gaussian_filter1d(cube, sigma=0.2, axis=0)
            cube_clean = convolve_beam(cube, self.beam_size_px)
            cube_noisy = apply_and_convolve_noise(cube_clean, peak_snr, self.beam_size_px)
            
            
            all_voxels.append(cube_noisy.flatten())  # Save flattened noisy cube voxels

            # Save for now — we'll standardize later once mean/std is known
            processed_data.append({
                'cube_clean': cube_clean,
                'cube_noisy': cube_noisy,
                'params': {'peak_snr': peak_snr,}
                           | gal_system_params})

        # Compute global stats across all noisy cubes

        if not self.cube_norm_params:
            all_voxels_flat = np.concatenate(all_voxels)
            self.mean = np.mean(all_voxels_flat)
            self.std = np.std(all_voxels_flat)

            if self.verbose:
                print(f"(*) Computed stats AFTER convolution + noise: mean={self.mean:.5g}, std={self.std:.5g}")

        else:
            self.mean, self.std = self.cube_norm_params
            if self.verbose:
                print(f"(*) Normalised with separate input mean = {self.mean} and std = {self.std}")

        # Now standardize cubes and wrap up
        final_data = []
        for item in processed_data:
            cube_noisy_std = (item['cube_noisy'] - self.mean) / self.std
            cube_clean_std = (item['cube_clean'] - self.mean) / self.std

            final_data.append({
                'cube_noisy': torch.tensor(np.expand_dims(cube_noisy_std, axis=0), dtype=torch.float32),
                'cube_clean': torch.tensor(np.expand_dims(cube_clean_std, axis=0), dtype=torch.float32),
                'params': item['params']
            })

        if self.verbose:
            print('(*) Standardized cubes using global mean and std')

        return final_data



    def return_stats(self):
        return self.mean, self.std
    


    def __len__(self):
        return len(self.resolved_dataset)
        

    @staticmethod
    def sanitize_params(params):
        sanitized = {}
        for k, v in params.items():
            v = np.array(v)  # ensure it's a numpy array
            if v.size == 1:
                sanitized[k] = float(v.item())  # convert scalar or singleton to float
            else:
                sanitized[k] = v.astype(np.float32)  # keep as array, cast to consistent dtype
        return sanitized


    def __getitem__(self, idx):
        """Retrieve precomputed data and normalize"""

        data = self.processed_data[idx]


        return data['cube_noisy'], data['cube_clean'], torch.tensor([data['params']['peak_snr'], data['params']['pix_spatial_scale'], data['params']['n_gals']],dtype = torch.float32), torch.tensor(data['params']['average_vels'],dtype = torch.float32)