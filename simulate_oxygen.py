from typing import Optional
from ssl import RAND_pseudo_bytes
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from scipy import interpolate
from astropy.io import fits
import astropy.constants as const
import astropy.units as u
from PyAstronomy import pyasl
from spectres import spectres
from num2tex import num2tex
import dataprep as dp
import attr
import multiprocessing as mp
import pdb


@attr.s(auto_attribs=True)
class OxygenSimulationResult:

    significance: np.array

    n_transits_white: int
    """Number of transits needed for a detection with just white noise"""

    n_transits_white_red: int
    """Number of transits needed for a detection with white noise and red noise"""


@attr.s(auto_attribs=True)
class OxygenSimulator:
    """Simulate an observed exoplanet model spectra with O2

    Given models for stellar, telluric, and  O2 spectra, this function will simulate an observed model spectra
    using the techniques outlined in Lopez-Morales 2019, and output the number of transits needed to detect O2.

    Example:
        >>> simulator = OxygenSimulator(...)
    """
    
    star: np.ndarray
    """Normalized stellar spectra in the form of a numpy array """

    wav_star: np.ndarray
    """wavelength array corresponding to star spectra"""

    planet: np.ndarray
    """Normalized Molecular O2 spectra in the form of a numpy array (HITRAN)"""
    
    wav_planet: np.ndarray
    """wavelength array corresponding to planet spectra"""

    telluric: Optional[np.ndarray] = None
    """Normalized Earth telluric spectra in the form of a numpy array. Default is None"""

    wav_tell: Optional[np.ndarray] = None
    """wavelength array corresponding to telluric spectra"""

    v_sys: float = 20 
    """The relative velocity of the system with respect to Earth. Default is 20 km/s"""

    R_p: float = const.R_earth.cgs.value
    """Radius of the planet in centimeters. Default is 1 Earth radii"""

    R_a: float = (80 * u.km).to(u.cm).value
    """Radius of the planet's atmosphere. Default is 80 km/s = 8000000cm, the estimated size of the Earth's atmosphere"""

    R_s: float = 0.26 * const.R_sun.cgs.value
    """Radius of the star in centimeters. Default is an M1V star with radii 0.26 x R_Sun."""

    R: float = 3e5
    """Resolution of observations. Default is 300,000"""

    wav_band: np.ndarray = np.array([759*1e-7, 772*1e-7])
    """Array of wavelength space to be sampled in centimeters. Default is 759 to 772 nanometer: np.array([759*1e-7, 772*1e-7])"""

    N_photons: Optional[float] = 1e8
    """Number of photons observed in each transit. Default is 1e6"""

    sigma_w: float = 1/np.sqrt(1e8)
    """White noise in the data. Default is 1/sqrt(Nphotons)"""
    
    sigma_r: Optional[float] = 1.20 * 1/np.sqrt(1e8)
    """Red noise in the data. Default is 1.2 x sigma_w. Default can be set to None to disable red noise calculation"""

    N_transits: np.ndarray = np.linspace(1, 100, 100)
    """Array containing transit numbers to be tested"""

    def simulate(self) -> OxygenSimulationResult:
        """Run the simulation
        
        The output model spectra will include a white noise only sample, and a white noise + red noise sample.

        Returns:
            str: Number of transits needed for white noise only model
            str: Number of transits needed for white noise + red noise model 
            np.ndarray(N_transits, significance):  Two arrays, one for the number of transits, and the other for the corresponding significance levels
        """


        # First perform without telluric
        if self.telluric == None:

            # Resample data so they all have the same number of points
            resampled_star_wav, resampled_star = dp.resample(
                    wav = self.wav_star,
                    wav_band = self.wav_band,
                    R = self.R,
                    flux = self.star,
                )
        
            resampled_planet_wav, resampled_planet = dp.resample(
                    wav = self.wav_planet,
                    wav_band = self.wav_band,
                    R = self.R,
                    flux = self.planet,
                )

            # Normalize data as a sanity check (Should already be normalized)
            resampled_planet_norm = resampled_planet / np.nanmax(resampled_planet)
            resampled_star_norm = resampled_star / np.max(resampled_star)

            # Compute Doppler shift using v_sys
            shifted_star, _ = pyasl.dopplerShift(
                                                        resampled_star_wav, 
                                                        resampled_star_norm, 
                                                        self.v_sys, 
                                                        edgeHandling="firstlast",
                                                        )


            shifted_planet, _ = pyasl.dopplerShift(
                                                        resampled_planet_wav, 
                                                        resampled_planet_norm, 
                                                        self.v_sys, 
                                                        edgeHandling="firstlast",
                                                        )
           
            # Now combine star and planet to create the model spectra
            in_transit_flux, out_transit_flux = self._create_model_spectra(shifted_planet=shifted_planet, shifted_star=shifted_star)

            # Now loop through the N_transits and bootstrap errors
            number_of_cpus_detected = mp.cpu_count()
            
            with mp.Pool(number_of_cpus_detected) as pool:  #999 Create a threadpool with a thread for each cpu
                signif = pool.map(self._bootstrap, ((in_transit_flux, out_transit_flux, resampled_planet_norm, resampled_planet_wav, N) for N in self.N_transits))
            
            # signif = []
            
            # for N in self.N_transits:
            #     signif.append(self._bootstrap(in_transit_flux, out_transit_flux, resampled_planet_norm, resampled_planet_wav, N))

            # Significance conversion table 
            sigma_convert = pd.read_csv('sigma_convert.csv')

            # Sigma interpolation function
            x =  sigma_convert['Significance'].tolist()
            y = sigma_convert['Sigma'].tolist()
            f = interpolate.interp1d(x, y, fill_value='extrapolate')

            # Convert significance to to sigma
            sigma = f(np.array(signif) * 100)

            # Only return values less than 3.5 sigma
            sigma = sigma[sigma < 3.5]

            if self.sigma_r is not None:
                # Find the first place where sigma is > 3
               
                n_transits_white_red = np.where(sigma == sigma[sigma > 3].min())[0][0]
                n_transits_white = None
            else:
                n_transits_white = np.where(sigma == sigma[sigma > 3].min())[0][0]
                n_transits_white_red = None

            return OxygenSimulationResult(significance=sigma, n_transits_white = n_transits_white, n_transits_white_red= n_transits_white_red)
            

    def _create_model_spectra(self, shifted_planet: np.ndarray, shifted_star: np.ndarray) -> np.ndarray:
        """Create model spectra by combining Doppler shifted star spectra and planet spectra                                            

        Args:
            shifted_planet (np.ndarray): Doppler shifted planet spectra
            shifted_star (np.ndarray): Doppler shifted star spectra
        """

        # The amount of stellar flux received in transit is the star flux minus the flux blocked by the planet
        ratio_starlight_blocked = ((np.pi * (self.R_p + self.R_a)**2) / (np.pi * (self.R_s)**2)) 
        out_transit_flux = shifted_star - (shifted_star * ratio_starlight_blocked )  

        # Ratio of areas between the stellar disk and the atmospheric ring of the planet                
        eps = (self.R_a**2 + 2*self.R_a*self.R_p) / (self.R_s **2)  

        # Now with epsilon, we can include how much of the starlight is masked by the planet's atmosphere.
        in_transit_flux = out_transit_flux  + eps*out_transit_flux*shifted_planet

        return in_transit_flux, out_transit_flux


    # def _bootstrap(self, in_transit_flux, out_transit_flux, resampled_planet, resampled_planet_wav, N) -> np.ndarray:
    def _bootstrap(self, args):
        """
        Bootstrap analysis to calculate the model spectra including white and red noise

        Args (tuple): Should contain 4 different variables:
            in_transit_flux (np.ndarray): Model spectra of planet and star to add noise to
            out_transit_flux (np.ndarray): Model spectra of just star to add noise to
            resampled_planet (np.ndarray): Array containing the planet array after resampling
            resampled_planet_wav (np.ndarray): Array containing the planet wavelenght array after resampling
            N (int): Number of transits to be tested

        Returns:
            np.ndarray: significance levels for input numbers of transits
        """

        in_transit_flux = args[0]
        out_transit_flux = args[1]
        resampled_planet = args[2]
        resampled_planet_wav = args[3]
        N = args[4]

        
        M = 1000 # Number of datasets to make for each transit
        delta_v = 0.5 # Change in velocity (the increment when dataling doppler shift)

        CCF_vals = []

        # Add noise

        N_photons_total = N*self.N_photons
        self.sigma_w = 1/np.sqrt(N_photons_total)
        self.sigma_r = 1.20*self.sigma_w  
        noise = np.sqrt(((self.sigma_w)**2)+ (self.sigma_r)**2)

        # Generate M bootstrapped datasets and calculate CCF
        for i in range(M):

            # In transit
            in_transit_noise = np.random.uniform(low = -1, high = 1, size = np.size(in_transit_flux)) * noise
            in_transit_flux_noise = in_transit_flux+ in_transit_noise

            # Out of transit
            out_transit_noise = np.random.uniform(low = -1, high = 1, size = np.size(out_transit_flux)) * noise
            out_transit_flux_noise = out_transit_flux + out_transit_noise 

            # P Obs with noise
            P_obs_noise = in_transit_flux_noise - out_transit_flux_noise

        
            # Perform CCF
            rv_sub_noise, cc_sub_noise = pyasl.crosscorrRV(
                    resampled_planet_wav, P_obs_noise, # spectrum 1
                    resampled_planet_wav, resampled_planet, # spectrum 2
                    -30, # lower bound velocity shift
                    30,  # upper bound velocity shift
                    2, # velocity increment
                    skipedge = 1000
                )



            CCF_val = rv_sub_noise[np.argmax(cc_sub_noise)]
            CCF_vals.append(CCF_val)

        CCF_vals_array = np.array(CCF_vals)
        m = len(CCF_vals_array[(CCF_vals_array <= (self.v_sys + delta_v)) & ((self.v_sys - delta_v) <= CCF_vals_array)])
        signif = m/M

        return signif



    def plot_ntransits(N_transits, significance):
        """
        This function make a plot of the significance level vs th enumber of transits needed to make a detection

        Args:
            N_transits (np.ndarray): The number of transits needed (x axis)
            signigicance (np.ndarray): Significance of the detection (y axis) (must be same length as x axis)
        """



    def plot_spectra():
        """
        This function makes a plot of the model spectra once star+planet+telluric are appropriately combined
        """


