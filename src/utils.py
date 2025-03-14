import mne
import io
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
from scipy.signal import convolve
from scipy.fft import fft, fftfreq
from scipy.signal import fftconvolve
from mne.filter import filter_data


freq_bands = {"Delta": (0.5, 3),
              "Theta": (4, 7),
              "Alpha": (8, 12),
              "Beta": (13, 30),
              "Gamma": (30, 50),
              "All": (0, 60)}


def z_score(data):
    return (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)
    

def find_peak_frequency_in_band(signals, sfreq, l_freq, h_freq, choose_fc):
    """
    Finds the mean frequency of the top 10% (90th percentile) of the FFT amplitude values
    within a specified frequency band.

    Parameters:
        signals: The EEG raw numpy data.
        sfreq (float): Sampling frequency of the EEG signal.
        l_freq (float): Lower bound of the frequency band.
        h_freq (float): Upper bound of the frequency band.

    Returns:
        float: Mean frequency of the top 10% amplitude values in the FFT within the given band.
    """

    # Compute FFT and frequencies
    freq = fftfreq(signals.shape[1], d=1/sfreq)
    fft_vals = np.abs(fft(signals, axis=1))

    # Select only the frequencies within the desired band
    band_mask = (freq >= l_freq) & (freq <= h_freq)
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


def gaborfilt(signal, centerFreq, alpha, samplingFreq):
    beta = alpha / samplingFreq
    centerOmega = 2 * np.pi * centerFreq / samplingFreq

    N = int(np.ceil((3 / beta) + 1))

    n = np.arange(-N, N + 1)
    g = np.exp(-(beta * n) ** 2)
    g /= np.sqrt(sum(g**2))
    filterResponse = g * np.cos(n * centerOmega)
    return fftconvolve(signal, filterResponse, mode='same')


def band_filtering(signals, sfreq, l_freq, h_freq, filterType, choose_fc, filterNo, window=1):
    if filterType == 'fir':
     return filter_data(signals,
                        sfreq=sfreq,
                        l_freq=l_freq,
                        h_freq=h_freq,
                        l_trans_bandwidth=0.5,
                        h_trans_bandwidth=0.5,
                        method='fir',
                        fir_window='hamming',
                        verbose=False
                        )


    elif filterType == 'cheby2':
      return filter_data(signals,
                         sfreq=sfreq,
                         l_freq=l_freq,
                         h_freq=h_freq,
                         l_trans_bandwidth=0.5,
                         h_trans_bandwidth=0.5,
                         method='iir',
                         iir_params={'order': 10, 'ftype': 'cheby2', 'rs': 30},
                         verbose=False
                         )


    elif filterType == 'butter':
      return filter_data(signals,
                         sfreq=sfreq,
                         l_freq=l_freq,
                         h_freq=h_freq,
                         l_trans_bandwidth=0.5,
                         h_trans_bandwidth=0.5,
                         method='iir',
                         iir_params={'order': 10, 'ftype': 'butter'},
                         verbose=False
                         )


    elif filterType == 'gabor':
      fc = find_peak_frequency_in_band(signals, sfreq, l_freq, h_freq, 'max')
      return np.array([gaborfilt(signal, fc, alpha, sfreq) for signal in signals])


    elif filterType == 'filterbanks':
      alpha = h_freq - l_freq
      step_freq = alpha / (filterNo - 1)
      range_freq = np.arange(l_freq, h_freq, step_freq)

      return np.array([[gaborfilt(signal, fc, alpha, sfreq)
                          for fc in range_freq]
                        for signal in signals])

    else:
      raise ValueError(f"Invalid filter type: {filterType}")


def apply_band_filtering(raw, filterType, choose_fc, filterNo):
    return np.array([band_filtering(raw.get_data(), raw.info['sfreq'], l_freq, h_freq, filterType, choose_fc, filterNo)
            for band, (l_freq, h_freq) in freq_bands.items()])
