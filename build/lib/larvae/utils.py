import numpy as np

def add_noise_to_mock_spectra(spectra, snr=100):
    """
    add gaussian noise to spectra
    
    Parameters
    ----------

    spectra : (normalized flux  x  pixels) : numpy.ndarray
    snr : desired signal-to-noise ratio  : int or numpy.array
    """
    noise = 1/snr #assuming the flux is continuum-normalized??
    noise *= np.random.randn(spectra.shape[0], spectra.shape[1])
    noisy_spectra = spectra + noise

    return noisy_spectra, spectra


