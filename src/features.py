from src.tkeo import apply_tkeo_to_eeg
from src.utils import apply_band_filtering, freq_bands
import numpy as np
from tqdm import tqdm

# Relative Energy: RE_freqband = E_freqband / E_total
def relative_energy(signalBands, **kwargs):
  mode = kwargs.get('mode', 'linear')

  if mode == 'linear':
    ratio = signalBands / np.sum(signalBands, axis=0)
  elif mode == 'log':
    ratio = np.log10(signalBands / np.sum(signalBands, axis=0))
  else:
    raise ValueError(f"Invalid mode: {mode}")

  # Handle timeframes where all bands are 0 -> sum(bands) = 0
  zeroIdx = np.argwhere(np.isnan(ratio))
  for idx in zeroIdx:
    ratio[tuple(idx)] = 0

  return ratio


# mean - Instantaneous Amplitude/Frequency Modulation
def MIA(signal, **kwargs):
  window = int(kwargs.get('window', 200))
  overlap = kwargs.get('overlap', 0.5)

  # step = window - (window * overlap)
  step = int(window * (1 - overlap))
  windows = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window, axis=-1)
  windows = windows[:, :, ::step, :]
  return np.nanmean(windows, axis=-1)


def MIF(signalAmpl, signalFreq, **kwargs):
  window = int(kwargs.get('window', 200))
  overlap = kwargs.get('overlap', 0.5)

  # step = window - (window * overlap)
  step = int(window * (1 - overlap))

  windows = np.lib.stride_tricks.sliding_window_view(signalAmpl, window_shape=window, axis=-1)
  instantAmpl = windows[:,:, ::step, :]
  windows = np.lib.stride_tricks.sliding_window_view(signalFreq, window_shape=window, axis=-1)
  instantFreq = windows[:,:, ::step, :]
  
  weightedFreq = np.nansum(instantFreq * (instantAmpl ** 2), axis=-1)
  sqAmpl = np.nansum(instantAmpl ** 2, axis=-1)
  MIFweighted = weightedFreq / sqAmpl
  # Exclude divisionByZero
  zeroAmplIdx = np.argwhere(sqAmpl == 0)
  for idx in zeroAmplIdx:
    MIFweighted[tuple(idx)] = 0
  return MIFweighted


# varience - Inst. Frequency Modulation
def VIF(signal, **kwargs):
  window = int(kwargs.get('window', 200))
  overlap = kwargs.get('overlap', 0.5)

  # step = window - (window * overlap)
  step = int(window * (1 - overlap))
  windows = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window, axis=-1)
  windows = windows[:,:, ::step, :]
  return np.nanvar(windows, axis=-1)


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


from scipy.signal import welch

def psd(signal, **kwargs):
  fs = int(kwargs.get('fs', 128))
  return welch(signal, fs=fs, nperseg=fs)[1][:-1]


def apply_feature(method, signalBands, **kwargs):
  return np.array([np.array([method(signalChannel, **kwargs) for signalChannel in signalBand])
                  for signalBand in signalBands])


def feature_extractor(tkeo_raw, features, **kwargs):
  feat_list = []
  
  if 'rel_energy' in features:
    feat_list.append(relative_energy(tkeo_raw[:,0], **kwargs))

  if 'mean_iam' in features:
    feat_list.append(MIA(tkeo_raw[:,2], **kwargs))

  if 'mean_ifm' in features:
    feat_list.append(MIF(tkeo_raw[:,2], tkeo_raw[:,3], **kwargs))

  if 'var_ifm' in features:
    feat_list.append(VIF(tkeo_raw[:,3], **kwargs))
  
  if 'hfd' in features:
    feat_list.append(apply_feature(hfd, tkeo_raw[:,0], **kwargs))

  return np.concatenate(feat_list, axis=2)


def baseline_extractor(raw, features, **kwargs):
  feat_list = []

  if 'psd' in features:
    feat_list.append(apply_feature(psd, raw, **kwargs))

  return np.concatenate(feat_list, axis=2)


def feature_extraction(dataset, features, DESA, filterType='gabor', 
                       choose_fc='mean', BinFil=True, filterNo=12, **kwargs):
  feature_matrix = []
  baselines = []

  for signal in tqdm(dataset.data, desc="Feature Matrix Extraction"):
    raw_bands = apply_band_filtering(signal['raw'], filterType, choose_fc, filterNo)
    # tkeo_raw = apply_tkeo_to_eeg(raw_bands, DESA, BinFil)
    # feature_matrix.append({'feat': feature_extractor(tkeo_raw, features, **kwargs),
    #                        'label': signal['label'],
    #                        'subject': signal['subject']})

    baselines.append({'feat': baseline_extractor(raw_bands, features, **kwargs),
                      'label': signal['label'],
                      'subject': signal['subject']})

  return np.array(feature_matrix), np.array(baselines)
