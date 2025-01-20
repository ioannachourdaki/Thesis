import mne
import io
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
from scipy.signal import convolve
from scipy.fft import fft, fftfreq


freq_bands = {"Delta": (0.5, 3),
              "Theta": (4, 7),
              "Alpha": (8, 12),
              "Beta": (13, 30),
              "Gamma": (30, 50)}


def find_peak_frequency_in_band(signals, fs, f_low, f_high, choose_fc):
    """
    Finds the mean frequency of the top 10% (90th percentile) of the FFT amplitude values
    within a specified frequency band.

    Parameters:
        signals: The EEG raw numpy data.
        fs (float): Sampling frequency of the EEG signal.
        f_low (float): Lower bound of the frequency band.
        f_high (float): Upper bound of the frequency band.

    Returns:
        float: Mean frequency of the top 10% amplitude values in the FFT within the given band.
    """

    # Compute FFT and frequencies
    freq = fftfreq(signals.shape[1], d=1/fs)
    fft_vals = np.abs(fft(signals, axis=1))

    # Select only the frequencies within the desired band
    band_mask = (freq >= f_low) & (freq <= f_high)
    band_freqs = freq[band_mask]
    band_fft_vals = fft_vals[:, band_mask]

    if choose_fc == 'mean':
      # Calculate the 90th percentile threshold for the FFT amplitudes in this band
      threshold = np.percentile(band_fft_vals, 90, axis=1)
      # Compute the mean of the 90th percentile top frequencies
      return np.mean(np.mean([band_freqs[band_fft_vals[i] >= threshold[i]] 
                              for i in range(signals.shape[0])], axis=1))

    # Or return the mean peak frequency across all channels
    elif choose_fc == 'max':
      return np.mean(band_freqs[np.argmax(band_fft_vals, axis=1)])
    
    # Else return the peak frequency per channel
    elif choose_fc == 'channel_max':
      return band_freqs[np.argmax(band_fft_vals, axis=1)]

    else:
      raise ValueError(f"Invalid fc calculation: {choose_fc}")


def gabor_filter(t,fc,a):
    gabor = np.exp(-a**2 * t**2) * np.cos(2 * np.pi * fc * t)
    return gabor / np.sum(np.abs(gabor))


def band_filtering(raw, f_low, f_high, filterType, choose_fc, window=1):

    if filterType == 'gabor':

      # Time vector for the Gabor filter
      fs = raw.info['sfreq']
      t = np.arange(-window, window, 1/fs)

      signals = raw.get_data()

      # Central frequency
      fc = find_peak_frequency_in_band(signals, fs, f_low, f_high, choose_fc)
      bandwidth = f_high - f_low

      if choose_fc == 'channel_max':
        filtered_signals = [np.convolve(signal, 
                                        gabor_filter(t, channel_fc, bandwidth), 
                                        mode='same')
                            for signal, channel_fc in zip(signals, fc)]

      else:
        filtered_signals = [np.convolve(signal, 
                                        gabor_filter(t, fc, bandwidth), 
                                        mode='same') 
                            for signal in signals]

      return mne.io.RawArray(np.array(filtered_signals), raw.info)


    elif filterType == 'fir':
     return raw.copy().filter(
                              l_freq=f_low,
                              h_freq=f_high,
                              l_trans_bandwidth=0.5,
                              h_trans_bandwidth=0.5,
                              method='fir',
                              fir_window='hamming'
                              )


    elif filterType == 'cheby2':
      return raw.copy().filter(
                              l_freq=f_low,
                              h_freq=f_high,
                              l_trans_bandwidth=0.5,
                              h_trans_bandwidth=0.5,
                              method='iir',
                              iir_params={'order': 10, 'ftype': 'cheby2', 'rs': 30}
                              )


    elif filterType == 'butter':
      return raw.copy().filter(
                              l_freq=f_low,
                              h_freq=f_high,
                              l_trans_bandwidth=0.5,
                              h_trans_bandwidth=0.5,
                              method='iir',
                              iir_params={'order': 10, 'ftype': 'butter'}
                              )


    # elif filterType == 'filterbanks':

    else:
      raise ValueError(f"Invalid filter type: {filterType}")


def apply_band_filtering(raw, filterType, choose_fc):
    return {
            band: band_filtering(raw, fmin, fmax, filterType, choose_fc)
            for band, (fmin, fmax) in freq_bands.items()
          }