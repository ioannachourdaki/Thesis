import scipy.io
import mne
import numpy as np
import os
from tqdm import tqdm


class SEEDLoader:
    def __init__(self, data_dir, label_file, sfreq=200, selChannels=None, window_length=20, overlap=0.5, trim_length=1):
        """
        Initialize the SEEDDataset.

        Parameters:
        - data_dir: str, directory containing the .mat files.
        - label_file: str, path to the label.mat file.
        - sfreq: float, sampling frequency of the EEG data.
        - channels: list of str, names of EEG channels.
        """
        self.data_dir = data_dir
        self.labels = scipy.io.loadmat(label_file)['label'][0]
        self.sfreq = sfreq
        self.window_length = window_length
        self.overlap = overlap
        self.trim_length = trim_length
        self.channels = [
            'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6',
            'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'Cb1',
            'O1', 'Oz', 'O2', 'Cb2'
        ]
        self.selChannels = selChannels or self.channels
        self.n_channels = len(self.selChannels)
        self.data = self.load_dataset()

    def get_segments(self, data):
        window = int(self.window_length * self.sfreq)
        step = int(window * (1 - self.overlap))
        result = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=1)[:, ::step]
        return np.transpose(result, axes=(1,0,2))


    def load_file(self, file_path):
        """
        Load all EEG signals and their labels from a single .mat file.

        Parameters:
        - file_path: str, path to the .mat file.

        Returns:
        - data: list of dicts, each containing 'raw' and 'label'.
        """
        info = mne.create_info(self.selChannels, sfreq=self.sfreq, ch_types=['eeg'] * self.n_channels, verbose=False)
        mat_data = scipy.io.loadmat(os.path.join(self.data_dir, file_path))
        trim_size = self.trim_length * self.sfreq

        # Select channels
        if self.n_channels != 62:
            idxChannels = [self.channels.index(channel) for channel in self.selChannels]

        eegs = []

        # 15 EEGs of the subject on this day
        for i, label in zip(range(15), self.labels):

            # Divide each sample in smaller segments
            key = [k for k in mat_data.keys() if k.endswith(f"_eeg{i+1}")][0]

            if self.n_channels != 62:
                segments = self.get_segments((mat_data[key][idxChannels] * 1e-6)[:, trim_size:-trim_size])
            else:
                segments = self.get_segments((mat_data[key] * 1e-6)[:, trim_size:-trim_size])

            for segment in segments:
                eegs.append({'raw': mne.io.RawArray(segment, info, verbose=False), 
                             'label': label,
                             'subject': int(file_path[0])
                            })

        return eegs

    def load_dataset(self):
        all_eegs = []
        # mat_files = ['10_20131130.mat'] #, '10_20131204.mat', '10_20131211.mat']
        mat_files = [f for f in os.listdir(self.data_dir) 
                     if f.endswith('.mat') and not f.startswith('label')]
        
        for mat_file in tqdm(mat_files, desc="Loading SEED Data"):
          all_eegs.extend(self.load_file(mat_file))
        return all_eegs
