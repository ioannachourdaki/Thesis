from src.tkeo import apply_tkeo_to_eeg
from src.utils import apply_band_filtering, freq_bands
import numpy as np

# Relative Energy: RE_freqband = E_freqband / E_total
def relative_energy(signal, band, **kwargs):
  mode = kwargs.get('mode', 'linear')

  band_signal = signal[band].get_data()
  all_signal = (signal['Delta'].get_data() + signal['Theta'].get_data() +
                signal['Alpha'].get_data() + signal['Beta'].get_data() +
                signal['Gamma'].get_data())

  if mode == 'linear':
    return band_signal / all_signal
  elif mode == 'log':
    return np.log10(band_signal / all_signal)
  else:
    raise ValueError(f"Invalid mode: {mode}")


# mean - Instantaneous Amplitude/Frequency Modulation
def MIA(signal, **kwargs):
  window = int(kwargs.get('window', 200))
  overlap = kwargs.get('overlap', 0.5)

  # step = window - (window * overlap)
  step = int(window * (1 - overlap))
  return np.nanmean(np.lib.stride_tricks.sliding_window_view(signal,
                                                             window_shape=window)[::step], 
                                                             axis=1)


def MIF(signalAmpl, signalFreq, **kwargs):
  window = int(kwargs.get('window', 200))
  overlap = kwargs.get('overlap', 0.5)

  # step = window - (window * overlap)
  step = int(window * (1 - overlap))

  instantAmpl = np.lib.stride_tricks.sliding_window_view(signalAmpl,
                                                  window_shape=window)[::step]
  instantFreq = np.lib.stride_tricks.sliding_window_view(signalFreq,
                                                  window_shape=window)[::step]
  
  weightedFreq = np.nansum(instantFreq * (instantAmpl ** 2), axis=1)
  sqAmpl = np.nansum(instantAmpl ** 2, axis=1)
  MIFweighted = weightedFreq / sqAmpl
  # Exclude divisionByZero
  zeroAmplIdx = (sqAmpl == 0)
  MIFweighted[zeroAmplIdx] = 0
  return MIFweighted


# varience - Inst. Frequency Modulation
def VIF(signal, **kwargs):
  window = int(kwargs.get('window', 200))
  overlap = kwargs.get('overlap', 0.5)

  # step = window - (window * overlap)
  step = int(window * (1 - overlap))
  return np.nanvar(np.lib.stride_tricks.sliding_window_view(signal,
                                                             window_shape=window)[::step], 
                                                             axis=1)


# Higuchi Fractal Dimension
from numpy.linalg import lstsq

def hfd(signal, **kwargs):
  kmax = kwargs.get('kmax', 8)
  window = int(kwargs.get('window', 200)) * 15 # Set a 15sec window
  overlap = kwargs.get('overlap', 0.5)

  def hfd_core(X, kmax):
    """
    Taken from PyEEG library.
    """

    if kmax < 2:
      raise ValueError(f"kmax value should be greater than 2.")

    L = []
    x = []
    N = len(X)

    for k in range(1, kmax+1):
        Lk = []
        for m in range(k):
            Lmk = 0
            segment_length = np.floor((N - m) / k)

            if segment_length > 1:
                for i in range(1, int(segment_length)):
                    Lmk += abs(X[m + i * k] - X[m + (i - 1) * k])
                Lmk = (Lmk * (N - 1) / (segment_length * k)) / k
                Lk.append(Lmk)

        if len(Lk) > 0 and np.mean(Lk) > 0:
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(1.0 / k), 1])

    if len(L) < 2:
        raise ValueError("Not enough valid segments to calculate HFD.")

    (p, _, _, _) = lstsq(np.array(x), np.array(L), rcond=None)
    return p[0]

  # step = window - (window * overlap)
  step = int(window * (1 - overlap))
  segments = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window)[::step]
  return np.array([hfd_core(segment, kmax) for segment in segments])


def apply_feature(method, raw, **kwargs):

  if method == relative_energy:
    return np.array([method(raw, band, **kwargs)
                     for band in freq_bands.keys()])

  if method == MIF:
    return np.array([np.array([method(signalAmpl, signalFreq, **kwargs) 
                              for (signalAmpl, signalFreq) 
                              in zip(raw['envelope'][band].get_data(), raw['freq'][band].get_data())])
                     for band in freq_bands.keys()])

  return np.array([np.array([method(signal, **kwargs) 
                             for signal in raw[band].get_data()])
                  for band in freq_bands.keys()])


def feature_extractor(tkeo_raw, features, **kwargs):
  feat_list = []
  
  if 'rel_energy' in features:
    feat_list.append(apply_feature(relative_energy, tkeo_raw['signal'], **kwargs))

  if 'mean_iam' in features:
    feat_list.append(apply_feature(MIA, tkeo_raw['envelope'], **kwargs))

  if 'mean_ifm' in features:
    feat_list.append(apply_feature(MIF, tkeo_raw, **kwargs))

  if 'var_ifm' in features:
    feat_list.append(apply_feature(VIF, tkeo_raw['freq'], **kwargs))
  
  if 'hfd' in features:
    feat_list.append(apply_feature(hfd, tkeo_raw['signal'], **kwargs))
  
  return {
          band: feat_array
          for band, feat_array in zip(freq_bands.keys(), 
                                      np.concatenate(feat_list, axis=2))
        }


def feature_extraction(dataset, features, DESA, filterType='gabor', 
                       choose_fc='mean', **kwargs,):
  feature_matrix = []

  for signal in dataset.data:
    raw_bands = apply_band_filtering(signal['raw'], filterType, choose_fc)
    tkeo_raw = apply_tkeo_to_eeg(raw_bands, DESA)
    feature_matrix.append(feature_extractor(tkeo_raw, features, **kwargs))

  return np.array(feature_matrix)
