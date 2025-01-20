from src.utils import freq_bands
import numpy as np
import mne

def tkeo(x):
  x_prev = np.roll(x, 1)
  x_prev[0] = x[0]
  x_next = np.roll(x, -1)
  x_next[-1] = x[-1]
  return abs(x**2 - x_next * x_prev)


def get_energy(psi):
  return np.sqrt(psi)


def set_for_zero_psi(psi,envelope,inst_freq):
  idx = np.where(psi==0)[0]

  for i in idx:
    envelope[i] = 0
    if i == 0:
      inst_freq[i] = 0
    else:
      inst_freq[i] = inst_freq[i-1]

  return envelope, inst_freq


def set_nan(envelope, inst_freq):
  for i in np.where(np.isnan(envelope) | np.isnan(inst_freq))[0]:
    envelope[i] = 0
    inst_freq[i] = 0
  return envelope, inst_freq



def DESA_1a(x):
  x_prev = np.roll(x, 1)
  x_prev[0] = x[0]

  psi_diff = tkeo(x-x_prev)
  psi = tkeo(x)

  envelope = np.sqrt(psi / (1 - (1 - (psi_diff / (2*psi)))**2))
  inst_freq = np.arccos(1 - (psi_diff / (2*psi)))

  envelope, inst_freq = set_for_zero_psi(psi, envelope, inst_freq)
  envelope, inst_freq = set_nan(envelope, inst_freq)

  return envelope, inst_freq


def DESA_1(x):
  x_prev = np.roll(x, 1)
  x_prev[0] = x[0]
  y = x - x_prev
  y_next = np.roll(y, -1)
  y_next[-1] = y[-1]

  psi_x = tkeo(x)
  psi_y = tkeo(y)
  psi_ynext = tkeo(y_next)

  envelope = np.sqrt(psi_x / (1 - (1 - ((psi_y + psi_ynext) / (4*psi_x)))**2))
  inst_freq = np.arccos(1 - ((psi_y + psi_ynext) / (4*psi_x)))

  envelope, inst_freq = set_for_zero_psi(psi_x, envelope, inst_freq)
  envelope, inst_freq = set_nan(envelope, inst_freq)

  return envelope, inst_freq


def DESA_2(x):
  x_prev = np.roll(x, 1)
  x_prev[0] = x[0]
  x_next = np.roll(x, -1)
  x_next[-1] = x[-1]

  psi_diff = tkeo(x_next-x_prev)
  psi = tkeo(x)

  envelope = 2*psi / np.sqrt(psi_diff)
  inst_freq = 0.5 * np.arccos(1 - (psi_diff / (2*psi)))

  envelope, inst_freq = set_for_zero_psi(psi,envelope,inst_freq)
  envelope, inst_freq = set_nan(envelope, inst_freq)

  return envelope, inst_freq


def get_tkeo_band(raw_band, algorithm):
  tkeo_channels = []
  energy_channels = []
  envelopes = []
  inst_freqs = []

  signals = raw_band.get_data()

  for signal in signals:

    # Get TKEO and energy of TKEO
    tkeo_channels.append(tkeo(signal))
    energy_channels.append(get_energy(tkeo_channels[-1]))

    if algorithm == 'simple':
      continue
    elif algorithm == 'desa1a':
      envelope, inst_freq = DESA_1a(signal)
      envelopes.append(envelope)
      inst_freqs.append(inst_freq)
    elif algorithm == 'desa1':
      envelope, inst_freq = DESA_1(signal)
      envelopes.append(envelope)
      inst_freqs.append(inst_freq)
    elif algorithm == 'desa2':
      envelope, inst_freq = DESA_2(signal)
      envelopes.append(envelope)
      inst_freqs.append(inst_freq)
    else:
      raise ValueError(f"Unknown algorithm: {algorithm}")

  return np.array(tkeo_channels), np.array(energy_channels), np.array(envelopes), np.array(inst_freqs)


def apply_tkeo_to_eeg(raw_bands, algorithm='simple'):
  tkeo_dict = {}
  energy_dict = {}
  envelope_dict = {}
  freq_dict = {}

  for band in freq_bands.keys():
    band_array = get_tkeo_band(raw_bands[band], algorithm)

    tkeo_dict[band] = mne.io.RawArray(band_array[0], raw_bands[band].info)
    energy_dict[band] = mne.io.RawArray(band_array[1], raw_bands[band].info)

    if algorithm != 'simple':
      envelope_dict[band] = mne.io.RawArray(band_array[2], raw_bands[band].info)
      freq_dict[band] = mne.io.RawArray(band_array[3], raw_bands[band].info)

  tkeo_contents = {
      'signal': tkeo_dict,
      'energy': energy_dict,
      'envelope': envelope_dict,
      'freq': freq_dict
                   }

  return tkeo_contents
