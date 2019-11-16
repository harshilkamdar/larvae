import numpy as np

def add_noise_to_mock_spectra(spectra, snr=100, return_original=True):
    """
    add gaussian noise to spectra
    """
    noise = 1/SNR #assuming the flux is continuum-normalized??
    noisy_spectra = np.zeros_like(spectra)
    