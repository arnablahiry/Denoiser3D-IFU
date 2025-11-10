import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

def ica(spectral_cube, n_ica_components):
    spectral_cube = spectral_cube.T
    noisy_centered_data_matrix = spectral_cube.reshape(
        (spectral_cube.shape[0] * spectral_cube.shape[1]), spectral_cube.shape[2]
    )

    ica = FastICA(n_ica_components, whiten='unit-variance', max_iter=2000, tol=5e-1)
    ica_components = ica.fit_transform(noisy_centered_data_matrix)
    reconstructed_data_matrix = ica.inverse_transform(ica_components)
    denoised_data_cube = reconstructed_data_matrix.reshape(
        spectral_cube.shape[0], spectral_cube.shape[1], spectral_cube.shape[2]
    ).T

    return denoised_data_cube



def ica_denoising(spectral_cube, mask = None, verbose = True, plot = True):

    if mask is None:
        raise ValueError("Emission mask must be provided to compute RMSE.")
    
    final_ica_components = spectral_cube.shape[0]
    init_ica_components = 2


    flux_history = []
    noise_history = []

    epsilon_flux = 1e-2  # Flux tolerance
    p = 3
    best_model = None
    previous_flux = 1e-33
    min_noise = float('inf')
    
    for i in range(final_ica_components-1):

        denoised_data_cube = ica(spectral_cube, init_ica_components)


        flux = np.sum(mask*denoised_data_cube)
        noise = np.std((1-mask)*denoised_data_cube)
        

        flux_history.append(flux)
        noise_history.append(noise)

        
        if verbose: print('ICs {}: flux: {:.4e}, noise: {:.4e}'.format(init_ica_components, flux, noise))

        if abs(flux - previous_flux)/previous_flux <= epsilon_flux:
            plateau_counter += 1
        else:
            plateau_counter = 0
        
        if noise < min_noise and plateau_counter >= p:
            min_noise = noise
            best_model = denoised_data_cube
            n_ic = init_ica_components
            if verbose:
                print('\nflux: ',flux)
                print('noise: {}\n'.format(noise))

        #if i == 0:
            #min_noise = noise
        # Update previous flux for next iteration
        previous_flux = flux

        init_ica_components += 1

    if 'n_ic' not in locals():
        n_ic = np.argmin(noise_history) + 2  # +2 because init_ica_components starts at 2
        if verbose:
            print(f'\nPlateau not reached. Using {n_ic} ICs (min noise fallback).')

        best_model = ica(spectral_cube, n_ic)


    if plot:

        print('3. Statistical Plots')
        x = np.linspace(2,final_ica_components, final_ica_components-1)
        plt.plot(x, flux_history, color = 'xkcd:blue', label = 'Mean Flux')
        plt.xlabel('number of ICA components used for reconstruction')
        plt.ylabel('Flux')
        plt.yscale('log')
        plt.axvline(x=n_ic, color='xkcd:red', linestyle='--', label=f'Ideal number of ICs = {n_ic}')
        plt.legend(frameon=True)
        plt.show()

    

    return best_model, flux_history, noise_history