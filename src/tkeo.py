from src.utils import freq_bands
import numpy as np
import mne
from scipy.signal import medfilt, convolve


# Disable warnings
np.seterr(invalid='ignore', divide='ignore')


def tkeo(x, BinFil=False):
  x_prev = np.roll(x, 1)
  x_next = np.roll(x, -1)
  EoSig = x**2 - x_next * x_prev
  EoSig[0] = EoSig[1]
  EoSig[-1] = EoSig[-2]
  e = max(np.max(np.abs(EoSig)) * 1e-5, np.finfo(float).eps) 
  if BinFil:
    EoSig = BinomialFilter(EoSig)
  return EoSig, e


def get_energy(psi):
  return np.sqrt(psi)


def BinomialFilter(signal):  
  # Initialize binomial kernel
  BinFil = np.array([0.25, 0.50, 0.25])
  BinFil = np.convolve([0.25, 0.50, 0.25], BinFil)
  return convolve(signal, BinFil, mode='same')


def cleanup_handler(EoSig, temp, A, F, isDesa2=False):
  # If A,F are complex --> temp < -1 or temp > 1
  negIdx = np.where((temp < -1) | (temp > 1))[0]

  F[negIdx] = np.abs(np.arccos(temp[negIdx].astype(np.complex128)))
  F[negIdx] = np.where(isDesa2, 0.5 * F[negIdx], F[negIdx])

  F[EoSig <= 0] = 0
  iPoints = np.where(EoSig == 0)[0]

  if iPoints.size > 0:
      m = 3
      F_interp = medfilt(F, kernel_size=m)
      F[iPoints] = F_interp[iPoints]
      A[iPoints] = 0

  A[negIdx] = 0
  A[EoSig <= 0] = 0
  A[np.isinf(A)] = 0

  return A,F


def DESA_1a(x, BinFil):
  # Calculate TKEO
  EoSig, e = tkeo(x, BinFil)
  EoSig[EoSig < e] = 0

  # diff = x(n) - x(n-1)
  diff = x - np.roll(x,1) 
  diff[0] = diff[1]
  # Demodulation
  EoD3,_ = tkeo(diff)
  if BinFil:
    EoD3 = BinomialFilter(EoD3)
  EoD3[EoD3 < e] = 0

  # Calculate inst. envelope and frequency
  temp = 1 - (EoD3 / (2*EoSig + e))
  F = np.arccos(temp)
  A = np.sqrt(EoSig / (1 - temp**2))
  # Handle divideByZero Error, NaN/Complex/Negative values etc
  A,F = cleanup_handler(EoSig, temp, A, F)
  return A,F


def DESA_1(x,BinFil):
  # Calculate TKEO
  EoSig, e = tkeo(x, BinFil)
  EoSig[EoSig < e] = 0

  # diff = x(n) - x(n-1)
  diff = x - np.roll(x,1) 
  diff[0] = diff[1]
  # Demodulation
  EoD1,_ = tkeo(diff)
  # EoD1 = EoD1[x(n)] + EoD1[x(n+1)]
  EoD1 = EoD1 + np.roll(EoD1,-1)
  EoD1[-1] = EoD1[-2]
  if BinFil:
    EoD1 = BinomialFilter(EoD1)
  EoD1[EoD1 < e] = 0

  # Calculate inst. envelope and frequency
  temp = 1 - (EoD1 / (4*EoSig + e))
  F = np.arccos(temp)
  A = np.sqrt(EoSig / (1 - temp**2))
  # Handle divideByZero Error, NaN/Complex/Negative values etc
  A,F = cleanup_handler(EoSig, temp, A, F)
  return A,F


def DESA_2(x,BinFil):
  # Calculate TKEO
  EoSig, e = tkeo(x, BinFil)
  EoSig[EoSig < e] = 0

  # diff = x(n+1) - x(n-1)
  diff = np.roll(x,-1) - np.roll(x,1)
  diff[0] = diff[1]
  diff[-1] = diff[-2]
  EoD2,_ = tkeo(diff)
  if BinFil:
    EoD2 = BinomialFilter(EoD2)
  EoD2[EoD2 < e] = 0

  # Calculate inst. envelope and frequency
  temp = 1 - (EoD2 / (2*EoSig + e))
  F = 0.5 * np.arccos(temp)
  A = 2 * (EoSig / np.sqrt(EoD2))
  # Handle divideByZero Error, NaN/Complex/Negative values etc
  A,F = cleanup_handler(EoSig, temp, A, F, True)
  return A,F


def get_tkeo_band(signals, DESA, BinFil):
  # If filterbanks are used -> signals.shape = (channels x filterbanks x frames)
  if len(signals.shape) == 3:
    filterbanks = np.array([[tkeo(filterbank)[0] for filterbank in signal] for signal in signals])
    mean_filterbanks = np.mean(filterbanks, axis=-1)
    iFilter = np.argmax(mean_filterbanks, axis=-1)
    signals = np.take_along_axis(signals, iFilter[:, np.newaxis, np.newaxis], axis=1).squeeze(1)

  tkeo_channels = [tkeo(signal, BinFil)[0] for signal in signals]
  energy_channels = [get_energy(tkeo_channel) for tkeo_channel in tkeo_channels]

  if DESA == 'simple':
    return np.array(tkeo_channels), np.array(energy_channels)

  if DESA == 'desa1a':
    AMFMs = [DESA_1a(signal, BinFil) for signal in signals]

  elif DESA == 'desa1':
    AMFMs = [DESA_1(signal, BinFil) for signal in signals]

  if DESA == 'desa2':
    AMFMs = [DESA_2(signal, BinFil) for signal in signals]

  return np.array(tkeo_channels), np.array(energy_channels), np.array(AMFMs)[:,0], np.array(AMFMs)[:,1]


def apply_tkeo_to_eeg(bandSignals, DESA, BinFil):
  return np.array([get_tkeo_band(bandSignal, DESA, BinFil) for bandSignal in bandSignals])
