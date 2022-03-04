import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from scipy import interpolate
from astropy.io import fits
from astropy import constants as c
from PyAstronomy import pyasl
from spectres import spectres
from num2tex import num2tex


def resample(wav, wav_band, R, flux):
    
    """
    Resample the data in wavelength space so they all cover the same range with the same number of points
    """
    
    wav_min, wav_max = wav_band
    
    # Get central wavelenth 
    wav_central = (wav_min + wav_max) / 2
    
    # Get the spacing between wavelengths for a given resolution
    wav_delta = wav_central / R
    
    # Resampled wavelength + flux arrays
    wav_resampled = np.arange(wav_min, wav_max, wav_delta)
    flux_resampled = spectres(wav_resampled, wav, flux)
    
    return wav_resampled, flux_resampled


def get_R(wavs):
    """
    Get the spectral resolution for a given wavelength array
    """
    diffs = np.diff(wavs)  # Calculates the spacing between each wavelength
    diffs = np.append(diffs, diffs[-1])  # Keeps len(diffs) == len(wavs)
    return wavs / diffs  # R = lambda / Delta lambda